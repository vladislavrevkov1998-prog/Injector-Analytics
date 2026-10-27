"""Positive, flow-normalized air kernels on the existing fuel cell grid."""
from dataclasses import dataclass
from collections import OrderedDict
from threading import Lock
import json
import numpy as np
from .air_sources import validate_air_model, resolve_sources
from .air_balance import solve_balance, BalanceResult
from .air_jets import trace_jets, section_crossing

_PATH_CACHE = OrderedDict()
_PATH_CACHE_LOCK = Lock()


def clear_air_cache():
    with _PATH_CACHE_LOCK:
        _PATH_CACHE.clear()


def cached_paths(sources, flows, cfg, model):
    # Fuel pressures, displayed z lists, colors and metadata do not alter the
    # air trajectory. Effective resolved coordinates do. The cache is bounded
    # and never returns a mutable source/array to an external editor.
    records = [{key: source[key] for key in ('id', 'z_mm', 'radius_mm', 'theta_deg',
                'direction', 'penetration_factor', 'sigma_r_mm', 'sigma_theta_deg', 'kernel', 'wall')} for source in sources]
    key = json.dumps(dict(sources=records, flows=flows, geometry=getattr(cfg, 'geometry', {}),
                          rin=cfg.r_in_mm, rout=cfg.r_out_mm,
                          parameters={key: model[key] for key in ('step_mm', 'max_path_mm', 'interaction_strength', 'interaction_length_mm')}),
                     sort_keys=True, ensure_ascii=False, allow_nan=False)
    with _PATH_CACHE_LOCK:
        if key in _PATH_CACHE:
            value = _PATH_CACHE[key]
            _PATH_CACHE.move_to_end(key)
            # Other source metadata, mixing laws and kernel widths may have
            # changed without changing the path. Keep the resolved records.
            return type(value)(sources, value.path_mm, value.positions, value.directions,
                               value.stop_steps, value.stop_reasons, value.diagnostics, value.ambiguity_path_mm)
    value = trace_jets(sources, flows, cfg, model)
    for array in (value.path_mm, value.positions, value.directions, value.stop_steps, value.ambiguity_path_mm):
        if array is None:
            continue
        array.setflags(write=False)
    with _PATH_CACHE_LOCK:
        _PATH_CACHE[key] = value
        while len(_PATH_CACHE) > 4:
            _PATH_CACHE.popitem(last=False)
    return value


def entrainment_fraction(source, z_mm, path_mm):
    if source['entrainment_stages']:
        return float(sum(portion['mixed_fraction'] for portion in portion_ledger(source, z_mm)))
    points = source['entrainment_points']
    if points:
        return float(np.interp(z_mm, [p[0] for p in points], [p[1] for p in points]))
    initial = source['mixed_fraction']
    length = source['mixing_length_mm']
    return float(initial if length is None else initial+(1-initial)*(-np.expm1(-max(0., path_mm)/length)))


def portion_ledger(source, z_mm):
    output = []
    for stage in source['entrainment_stages']:
        start, end = stage['z_start_mm'], stage['z_end_mm']
        progress = float(z_mm >= start) if end == start else float(np.clip((z_mm-start)/(end-start), 0., 1.))
        output.append(dict(portion_id=stage['id'], source_id=source['id'], z_mm=float(z_mm),
                           fraction=stage['fraction'], mixed_fraction=stage['fraction']*progress))
    return output


def normalized_kernel(plane, source, center, path_mm=0., *, uniform=False, mixed=False):
    """Discrete integrals via positive area quadrature, then exact normalization.

    Annular bands have unequal areas; uniform is area-weighted, not equal per
    cell. The periodic Gaussian angular kernel uses physical global theta.
    Local kernels use 3×3 cell quadrature to reduce coarse-grid centre bias.
    Widths are explicit empirical parameters. No kernel invents mass.
    """
    area = np.asarray(plane.cell_area_m2, dtype=float)
    if area.ndim != 2 or not np.isfinite(area).all() or np.any(area <= 0):
        raise ValueError('Воздух: нужны положительные конечные площади ячеек')
    if uniform or source['kernel'] == 'uniform':
        return area / area.sum()
    edges = np.asarray(plane.radial_edges_mm)
    angles = np.asarray(plane.theta_centers_deg)
    radial_center, theta_center = center[1:]
    if source['kernel'] == 'wall':
        wall = source['wall']
        if wall == 'free':
            wall = 'inner' if radial_center <= (edges[0]+edges[-1])/2 else 'outer'
        radial_center = edges[0] if wall == 'inner' else edges[-1]
    spread = source['spread_mm_per_mm']*path_mm
    sigma_r = source['sigma_r_mm'] + spread
    sigma_theta = source['sigma_theta_deg'] + np.rad2deg(spread/max(radial_center, 1e-9))
    if mixed:
        sigma_r *= np.sqrt(2.)
        sigma_theta *= np.sqrt(2.)
    # Separable log-space quadrature prevents underflow for a narrow source
    # between coarse cells. Exactly zero width is a normalized point/band
    # measure; an internal boundary is shared equally by adjacent cells.
    offsets = np.array([-.3872983346207417, 0., .3872983346207417])
    quad_weights = np.array([5/18, 4/9, 5/18])
    half_width = 180./len(angles)
    if sigma_r == 0:
        radial = ((edges[:-1] <= radial_center+1e-9) & (edges[1:] >= radial_center-1e-9)).astype(float)
    else:
        radius = ((edges[:-1]+edges[1:])/2)[:, None] + np.diff(edges)[:, None]*offsets
        log_r = -.5*((radius.astype(np.longdouble)-radial_center)/sigma_r)**2
        radial = np.asarray(np.sum(np.exp(log_r-log_r.max())*quad_weights*radius, axis=1)*np.diff(edges), dtype=float)
    if sigma_theta == 0:
        angular = (np.abs((angles-theta_center+180.) % 360.-180.) <= half_width+1e-9).astype(float)
    else:
        theta = angles[:, None]+offsets*(2*half_width)
        delta = ((theta-theta_center+180.) % 360.)-180.
        log_t = -.5*(delta.astype(np.longdouble)/sigma_theta)**2
        angular = np.asarray(np.sum(np.exp(log_t-log_t.max())*quad_weights, axis=1), dtype=float)
    weight = radial[:, None]*angular[None, :]
    total = float(weight.sum())
    if not np.isfinite(total) or total <= 0:
        raise ValueError(source['id'] + ': пространственное ядро не нормируется')
    return weight/total


@dataclass
class AirFieldResult:
    z_mm: float
    air_total_kg_s: np.ndarray
    air_core_kg_s: np.ndarray
    air_mixed_kg_s: np.ndarray
    reverse_air_kg_s: np.ndarray
    unsupported_mask: np.ndarray
    balance: BalanceResult
    source_states: list
    diagnostics: tuple
    provenance: dict


def compute_air_field(plane, cfg, nozzles, *, paths=None):
    raw = getattr(cfg, 'air_model', None)
    model = validate_air_model(raw)
    if not model['enabled']:
        return None
    balance = solve_balance(model)
    sources = resolve_sources(model, cfg, nozzles)
    if paths is None:
        paths = cached_paths(sources, balance.flows, cfg, model)
    shape = np.asarray(plane.cell_mass_kg_s).shape
    if len(shape) != 2 or np.asarray(plane.cell_area_m2).shape != shape or len(plane.radial_edges_mm) != shape[0]+1 or len(plane.theta_centers_deg) != shape[1]:
        raise ValueError('Воздух: формы площадей/топливных ячеек/границ/углов сетки должны совпадать')
    core = np.zeros(shape)
    mixed = np.zeros(shape)
    reverse = np.zeros(shape)
    unsupported = np.zeros(shape, dtype=bool)
    states = []
    diagnostics = list(balance.diagnostics)+list(paths.diagnostics)
    for index, source in enumerate(paths.sources):
        flow = balance.flows.get(source['id'], 0.)
        crossing = section_crossing(paths, index, float(plane.z_mm), cfg)
        state = dict(id=source['id'], kind=source['kind'], assigned_kg_s=flow,
                     crossed_kg_s=0., reverse_kg_s=0., not_crossed_kg_s=0., outside_kg_s=0.,
                     E=0., center=None, direction=None, status='not_crossed',
                     provenance=source['provenance'], physical_key=source['physical_key'],
                     stop_reason=paths.stop_reasons[index], portions=portion_ledger(source, float(plane.z_mm)))
        if crossing is not None:
            state.update(center=crossing['center'], direction=crossing['direction'], path_mm=crossing['path_mm'])
            e = entrainment_fraction(source, float(plane.z_mm), crossing['path_mm'])
            state['E'] = e
            if paths.ambiguity_path_mm is not None and crossing['path_mm'] >= paths.ambiguity_path_mm[index] and flow:
                state['direction_ambiguity'] = True
                unsupported[:] = True
            w_core = normalized_kernel(plane, source, crossing['center'], crossing['path_mm'], uniform=model['distribution']=='uniform')
            w_mixed = normalized_kernel(plane, source, crossing['center'], crossing['path_mm'], uniform=model['distribution']=='uniform', mixed=True)
            if crossing['sign'] < 0:
                state.update(reverse_kg_s=flow, status='reverse')
                reverse += flow*((1-e)*w_core+e*w_mixed)
                if flow:
                    unsupported |= ((w_core+w_mixed) > 0)
                    diagnostics.append(source['id'] + ': обратный осевой расход исключён из положительной матрицы; α здесь не определён.')
            elif crossing['multiple_crossings']:
                state.update(not_crossed_kg_s=flow, status='multiple_crossings')
                if flow:
                    unsupported |= ((w_core+w_mixed) > 0)
                    diagnostics.append(source['id'] + ': многократное пересечение сечения; локальный α не определён.')
            else:
                state.update(crossed_kg_s=flow, status='forward')
                core += flow*(1-e)*w_core
                mixed += flow*e*w_mixed
        elif paths.stop_reasons[index] == 'wall_exit':
            state.update(outside_kg_s=flow, status='wall_exit')
            if flow:
                diagnostics.append(source['id'] + ': центр струи покинул тракт до пересечения; расход вынесен в отдельный outside ledger.')
        else:
            state['not_crossed_kg_s'] = flow
            if paths.stop_reasons[index] == 'max_path' and flow:
                diagnostics.append(source['id'] + ': сечение не пересечено в пределах max_path_mm; конечная длина пути не доказывает физическое отсутствие воздуха.')
        for portion in state['portions']:
            portion['assigned_kg_s'] = flow*portion['fraction']
            portion['forward_kg_s'] = state['crossed_kg_s']*portion['fraction']
            portion['mixed_kg_s'] = state['crossed_kg_s']*portion['mixed_fraction']
            portion['core_kg_s'] = portion['forward_kg_s']-portion['mixed_kg_s']
        states.append(state)
    total = core+mixed
    # Every registered assigned inlet flow has exactly one section state.
    ledger = sum(sum(s[key] for key in ('crossed_kg_s', 'reverse_kg_s', 'not_crossed_kg_s', 'outside_kg_s')) for s in states)
    if not np.isclose(ledger, balance.assigned_kg_s, rtol=1e-11, atol=1e-14):
        raise ValueError('Не замкнут входной/секционный баланс воздуха')
    if not np.isclose(total.sum(), sum(s['crossed_kg_s'] for s in states), rtol=1e-11, atol=1e-14):
        raise ValueError('Нарушена нормировка пространственного воздуха')
    provenance = dict(model='empirical conditional air rays; no velocity, physical momentum, reaction or temperature',
                      units='kg/s per cell; lengths mm; global theta degrees +X/CCW',
                      distribution=model['distribution'], normalization='nonnegative cell integrals, sum W = 1',
                      unknown_residual_kg_s=balance.unassigned_kg_s, source_inputs=model['sources'],
                      internal_step_mm=model['step_mm'], max_path_mm=model['max_path_mm'],
                      interaction_strength=model['interaction_strength'],
                      radial_edges_mm=np.asarray(plane.radial_edges_mm).tolist(),
                      theta_centers_deg=np.asarray(plane.theta_centers_deg).tolist())
    return AirFieldResult(float(plane.z_mm), total, core, mixed, reverse, unsupported,
                          balance, states, tuple(dict.fromkeys(diagnostics)), provenance)
