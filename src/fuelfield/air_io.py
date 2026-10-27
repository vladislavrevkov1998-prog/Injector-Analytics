"""Air and mixture reports. Cell values are flows [kg/s], never cell masses.

The reports keep the original fuel arrays untouched, export surface films
separately, and leave undefined alpha cells blank with an explicit mask.
"""
from dataclasses import asdict, is_dataclass
import csv
import json
import re
from pathlib import Path

import numpy as np

from .analytics_io import write_workbook
from .coordinates import from_global


def _plain(value):
    if is_dataclass(value):
        value = asdict(value)
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, np.ndarray):
        return _plain(value.tolist())
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _json(value):
    return json.dumps(_plain(value), ensure_ascii=False, allow_nan=False, sort_keys=True)


def _finite(value):
    return float(value) if value is not None and np.isfinite(value) else None


def air_tables(results, cfg, planes, *, times_s=None, pressures=None):
    """Build export tables from already computed mixture results and fuel planes.

    ``times_s`` identifies real snapshots. Composite columns carry their own
    scan time/step and intentionally have no single section alpha or balance.
    """
    results, planes = list(results), list(planes)
    if not results or len(results) != len(planes):
        raise ValueError('Нужен результат воздуха для каждого топливного сечения.')
    if pressures is not None and len(pressures) != len(planes):
        raise ValueError('Число наборов давления должно совпадать с числом сечений.')
    if times_s is not None and len(times_s) != len(planes):
        raise ValueError('Число времён должно совпадать с числом сечений.')
    cells = [['z_mm', 'scope', 'step', 'time_s', 'pressures_kgf_cm2_json', 'radial_level', 'theta_node',
              'radius_center_mm', 'theta_physical_deg', 'theta_measurement_deg', 'cell_area_m2',
              'fuel_eligible_kg_s', 'fuel_total_including_film_kg_s', 'film_inner_kg_s', 'film_outer_kg_s',
              'air_total_kg_s', 'air_core_kg_s', 'air_mixed_kg_s', 'air_reverse_kg_s', 'air_total_flux_kg_m2_s',
              'alpha_total', 'alpha_mixed', 'alpha_total_defined', 'alpha_mixed_defined']]
    sections = [['z_mm', 'scope', 'time_s', 'fuel_eligible_kg_s', 'air_total_kg_s',
                 'air_core_kg_s', 'air_mixed_kg_s', 'air_reverse_kg_s', 'alpha_section_total', 'alpha_section_mixed',
                 'undefined_area_fraction', 'air_balance_json', 'diagnostics_json']]
    for k, (result, plane) in enumerate(zip(results, planes)):
        if not np.isclose(result.z_mm, plane.z_mm, rtol=0., atol=1e-7):
            raise ValueError('Координата z результата воздуха не совпадает с топливным сечением.')
        shape = plane.cell_mass_kg_s.shape
        for key in ('fuel_kg_s', 'air_total_kg_s', 'air_core_kg_s', 'air_mixed_kg_s', 'alpha_total', 'alpha_mixed'):
            if np.shape(getattr(result, key)) != shape:
                raise ValueError('Матрицы топлива и воздуха должны иметь одну сетку: ' + key)
        composite = bool(getattr(plane, 'is_composite', False))
        scope = 'composite_different_times_no_global_balance' if composite else 'instantaneous'
        time = None if times_s is None else times_s[k]
        zeros = np.zeros(shape)
        inner = getattr(plane, 'film_inner_cell_mass_kg_s', None)
        outer = getattr(plane, 'film_outer_cell_mass_kg_s', None)
        inner = zeros if inner is None else inner
        outer = zeros if outer is None else outer
        reverse = getattr(result, 'reverse_air_kg_s', None)
        reverse = zeros if reverse is None else np.asarray(reverse)
        if reverse.shape != shape: raise ValueError('Обратный воздух должен иметь ту же сетку.')
        measured = from_global(plane.theta_centers_deg, cfg)
        for i, radius in enumerate(plane.radial_centers_mm):
            for j, theta in enumerate(plane.theta_centers_deg):
                scan_times = getattr(plane, 'scan_times_s', None)
                scan_steps = getattr(plane, 'scan_steps', None)
                stamp = scan_times[j] if composite and scan_times is not None else time
                step = int(scan_steps[j]) if composite and scan_steps is not None else None
                scan_pressures = getattr(plane, 'scan_pressures', None)
                pressure = scan_pressures[j] if composite and scan_pressures is not None else (None if pressures is None else pressures[k])
                area = float(plane.cell_area_m2[i, j])
                at = _finite(result.alpha_total[i, j]); am = _finite(result.alpha_mixed[i, j])
                cells.append([float(plane.z_mm), scope, step, _finite(stamp), None if pressure is None else _json(pressure), i + 1, j + 1,
                              float(radius), float(theta), float(measured[j]), area,
                              float(result.fuel_kg_s[i, j]), float(plane.cell_mass_kg_s[i, j]),
                              float(inner[i, j]), float(outer[i, j]),
                              float(result.air_total_kg_s[i, j]), float(result.air_core_kg_s[i, j]),
                              float(result.air_mixed_kg_s[i, j]), float(reverse[i, j]), float(result.air_total_kg_s[i, j]) / area,
                              at, am, at is not None, am is not None])
        sections.append([float(plane.z_mm), scope, _finite(time), float(np.sum(result.fuel_kg_s)),
                         float(np.sum(result.air_total_kg_s)), float(np.sum(result.air_core_kg_s)),
                         float(np.sum(result.air_mixed_kg_s)), float(np.sum(reverse)),
                         None if composite else _finite(result.alpha_section_total),
                         None if composite else _finite(result.alpha_section_mixed),
                         _finite(result.undefined_area_fraction), _json(result.balance), _json(result.diagnostics)])
    states = [['z_mm', 'source_id', 'assigned_kg_s', 'crossed_kg_s', 'reverse_kg_s',
               'not_crossed_kg_s', 'outside_kg_s', 'entrainment_fraction', 'status', 'state_json']]
    for result, plane in zip(results, planes):
        for state in getattr(result, 'source_states', []):
            states.append([float(plane.z_mm), state.get('id'),
                           *[_finite(state.get(key)) for key in ('assigned_kg_s', 'crossed_kg_s', 'reverse_kg_s',
                                                               'not_crossed_kg_s', 'outside_kg_s', 'E')],
                           state.get('status'), _json(state)])
    sources = [['source_id', 'source_kind', 'nozzle_id', 'object_id', 'source_input_json']]
    for source in getattr(cfg, 'air_model', {}).get('sources', []):
        sources.append([source.get('id'), source.get('kind'), source.get('nozzle_id'),
                        source.get('object_id'), _json(source)])
    metadata_entries = [
        ['format', 'injector-air-mixture-v1'],
        ['units', 'cell flows kg/s; flux kg/(m2 s); geometry mm and degrees; L0 kg air/kg fuel; alpha dimensionless'],
        ['alpha_scope', 'Conditional flow ratios; no temperature, chemistry or validated local reacting composition.'],
        ['undefined_alpha', 'Blank cell plus defined=false; no fuel or unavailable L0/unsupported local flow.'],
        ['normalization', 'Positive per-source spatial kernels normalized by mass flow, not equal cell areas.'],
        ['composite', 'Different columns can represent different times. No single instantaneous section alpha/balance.'],
        ['fuel_model', _json(getattr(cfg, 'fuel_transport', {'mode': 'legacy'}))],
        ['air_model', _json(getattr(cfg, 'air_model', {}))],
        ['provenance', _json([getattr(r, 'provenance', {}) for r in results])],
        ['metadata_chunking', 'Append key__part_0002 etc. to the first value for long JSON entries; chunks <=30000 characters.'],
    ]
    metadata = [['key', 'value']]
    for key, value in metadata_entries:
        chunks = [value[offset:offset + 30000] for offset in range(0, len(value), 30000)] or ['']
        metadata.extend([[key if part == 0 else key + '__part_' + str(part + 1).zfill(4), chunk]
                         for part, chunk in enumerate(chunks)])
    tables = [('Air_cells', cells), ('Air_sections', sections), ('Air_sources', sources), ('Air_source_states', states)]
    long_text = [['reference', 'part', 'text']]
    for name, rows in tables:
        for r, row in enumerate(rows):
            for c, value in enumerate(row):
                if isinstance(value, str) and len(value) > 30000:
                    reference = f'{name}!R{r + 1}C{c + 1}'
                    row[c] = '@Air_long_text:' + reference
                    for part, offset in enumerate(range(0, len(value), 30000), 1):
                        long_text.append([reference, part, value[offset:offset + 30000]])
    if len(long_text) > 1:
        tables.append(('Air_long_text', long_text))
        metadata.append(['long_text', '@Air_long_text references are reconstructed by concatenating ordered parts.'])
    tables.append(('Air_metadata', metadata))
    return tables


def write_air_csv(path, results, cfg, planes, *, times_s=None, pressures=None):
    """Write cells to path, named companion tables and a strict JSON manifest."""
    path = Path(path)
    tables = air_tables(results, cfg, planes, times_s=times_s, pressures=pressures)
    for index, (name, rows) in enumerate(tables):
        destination = path if index == 0 else path.with_name(path.stem + '_' + name + '.csv')
        with destination.open('w', encoding='utf-8-sig', newline='') as stream:
            csv.writer(stream, delimiter=';').writerows(rows)
    metadata = {}
    for key, value in tables[-1][1][1:]:
        match = re.fullmatch(r'(.*)__part_[0-9]{4}', key)
        if match: metadata[match[1]] += value
        else: metadata[key] = value
    path.with_name(path.stem + '_air.json').write_text(_json(metadata) + '\n', encoding='utf-8')
    return path


def write_air_xlsx(path, results, cfg, planes, *, times_s=None, pressures=None):
    """Undefined values are empty worksheet cells, never NaN/Inf XML values."""
    write_workbook(path, air_tables(results, cfg, planes, times_s=times_s, pressures=pressures))
    return Path(path)
