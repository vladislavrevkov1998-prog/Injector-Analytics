"""Air/fuel flow ratios; never a reacting-mixture or temperature solver."""
from dataclasses import dataclass
import numpy as np
from .air_sources import validate_air_model
from .air_distribution import compute_air_field


def flow_ratio(air, fuel, l0):
    """Retain ordinary division accuracy and recover under/overflow in L0*Gf."""
    air,fuel=np.broadcast_arrays(np.asarray(air,dtype=float),np.asarray(fuel,dtype=float))
    result=np.full(fuel.shape,np.nan)
    positive=fuel>0
    with np.errstate(over='ignore',under='ignore',divide='ignore',invalid='ignore'):
        denominator=l0*fuel
        np.divide(air,denominator,out=result,where=positive&(denominator>0)&np.isfinite(denominator))
        result[positive&(air==0)]=0.
        recover=positive&(air>0)&(~np.isfinite(result)|~np.isfinite(denominator)|(denominator==0))
        result[recover]=np.exp(np.log(air[recover])-np.log(fuel[recover])-np.log(l0))
    result[~np.isfinite(result)]=np.nan
    return result


@dataclass
class MixtureResult:
    z_mm: float
    fuel_kg_s: np.ndarray
    air_total_kg_s: np.ndarray
    air_core_kg_s: np.ndarray
    air_mixed_kg_s: np.ndarray
    alpha_total: np.ndarray
    alpha_mixed: np.ndarray
    alpha_section_total: float
    alpha_section_mixed: float
    undefined_area_fraction: float
    l0: object
    balance: object
    source_states: list
    diagnostics: tuple
    provenance: dict
    is_composite: bool = False
    reverse_air_kg_s: object = None


def compute_mixture(plane, cfg, nozzles, *, air_result=None):
    model = validate_air_model(getattr(cfg, 'air_model', None))
    if not model['enabled']:
        return None
    air = compute_air_field(plane, cfg, nozzles) if air_result is None else air_result
    if not np.isfinite(float(air.z_mm)) or not np.isclose(float(air.z_mm), float(plane.z_mm), rtol=0., atol=1e-9):
        raise ValueError('Для α воздух и топливо должны относиться к одному физическому сечению z')
    for key in ('radial_edges_mm', 'theta_centers_deg'):
        if key in air.provenance:
            left, right = np.asarray(air.provenance[key]), np.asarray(getattr(plane, key))
            if left.shape != right.shape or not np.allclose(left, right, rtol=0., atol=1e-9):
                raise ValueError('Для α воздух и топливо должны иметь одну физическую сетку: ' + key)
    suspended = getattr(plane, 'suspended_cell_mass_kg_s', None)
    fuel = np.asarray(plane.cell_mass_kg_s if suspended is None else suspended, dtype=float).copy()
    if fuel.shape != air.air_total_kg_s.shape or np.any(fuel < 0) or not np.isfinite(fuel).all():
        raise ValueError('Для α нужны согласованные конечные неотрицательные топливные расходы')
    total = np.full(fuel.shape, np.nan)
    mixed = total.copy()
    diagnostics = list(air.diagnostics)
    l0 = model['l0']
    valid = (fuel > 0) & ~air.unsupported_mask
    if l0 is None or l0 <= 0:
        valid[:] = False
        diagnostics.append('L0 неизвестен или равен нулю: баланс воздуха доступен, α не определён.')
    else:
        total[valid]=flow_ratio(air.air_total_kg_s,fuel,l0)[valid]
        mixed[valid]=flow_ratio(air.air_mixed_kg_s,fuel,l0)[valid]
        overflow=valid&(~np.isfinite(total)|~np.isfinite(mixed))
        if overflow.any():
            valid[overflow]=False;total[overflow]=np.nan;mixed[overflow]=np.nan
            diagnostics.append('α за пределами представимого числового диапазона: ячейки маскированы, бесконечные значения не выдаются.')
    area = np.asarray(plane.cell_area_m2)
    undefined_area = float(area[~valid].sum()/area.sum())
    composite = bool(getattr(plane, 'is_composite', False))
    integral_total = integral_mixed = float('nan')
    if l0 is not None and l0 > 0 and fuel.sum() > 0 and not air.unsupported_mask.any() and not composite:
        integral_total = float(flow_ratio(air.air_total_kg_s.sum(),fuel.sum(),l0))
        integral_mixed = float(flow_ratio(air.air_mixed_kg_s.sum(),fuel.sum(),l0))
    if composite:
        diagnostics.append('Пространственно-временная сборка: α ячеек относится к их времени; единый мгновенный α/баланс сечения не определён.')
    if suspended is not None:
        diagnostics.append('В знаменателе α только взвесь; стеночные плёнки исключены и остаются в топливной диагностике.')
    provenance = dict(air.provenance, alpha='ratio of associated effective axial mass flows; no burning/evaporation model',
                      fuel_model='confined suspended only' if suspended is not None else 'legacy independent fuel cells',
                      l0=l0, is_composite=composite,
                      film_inner_kg_s=float(np.sum(getattr(plane, 'film_inner_cell_mass_kg_s', None))) if getattr(plane, 'film_inner_cell_mass_kg_s', None) is not None else 0.,
                      film_outer_kg_s=float(np.sum(getattr(plane, 'film_outer_cell_mass_kg_s', None))) if getattr(plane, 'film_outer_cell_mass_kg_s', None) is not None else 0.)
    return MixtureResult(float(plane.z_mm), fuel, air.air_total_kg_s, air.air_core_kg_s,
                         air.air_mixed_kg_s, total, mixed, integral_total, integral_mixed,
                         undefined_area, l0, air.balance, air.source_states,
                         tuple(dict.fromkeys(diagnostics)), provenance, composite, air.reverse_air_kg_s)
