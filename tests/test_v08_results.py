"""Report contracts and real Qt views preserve phase and time semantics."""
from copy import deepcopy
from dataclasses import dataclass
import csv
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch
from zipfile import ZipFile
from xml.etree import ElementTree as ET

import numpy as np

from fuelfield.air_io import air_tables, write_air_csv, write_air_xlsx
from fuelfield.core import Settings, PlaneResult
from fuelfield.data_io import _xlsx_sheets


@dataclass
class BalanceFixture:
    total_kg_s: float = 20.
    assigned_kg_s: float = 19.
    unassigned_kg_s: float = 1.


def case(composite=False, scale=1.):
    cfg = Settings(n_radial=2, n_angular=8, z_mm=[10., 20.])
    cfg.air_model = {'schema_version': 1, 'enabled': True, 'total_kg_s': 10.,
                     'sources': [{'id': 's1', 'kind': 'other', 'flow_kg_s': 10.,
                                  'provenance': {'source_kind': 'synthetic'}, 'future': {'keep': True}}]}
    fuel = np.full((2, 8), scale); fuel[0, 0] = 0.
    area = np.repeat(np.array([[1.], [3.]]) * 1e-3, 8, axis=1)
    film_i = np.full((2, 8), .1); film_o = np.full((2, 8), .2)
    total = fuel + film_i + film_o
    plane = PlaneResult(10., total, area, total / area, np.array([225., 275.]),
                        np.array([200., 250., 300.]), np.arange(8) * 45.,
                        float(total.sum()), float(total.sum()), 0., 0.,
                        is_composite=composite,
                        suspended_cell_mass_kg_s=fuel, film_inner_cell_mass_kg_s=film_i,
                        film_outer_cell_mass_kg_s=film_o)
    if composite:
        plane.scan_times_s = np.arange(8) * .5; plane.scan_steps = np.arange(8)
    air = np.arange(1., 17.).reshape(2, 8) / 10.
    alpha = np.divide(air, 2. * fuel, out=np.full((2, 8), np.nan), where=fuel > 0)
    balance = BalanceFixture()
    # Real BalanceResult is a dataclass. Use a plain dict for serialized fixture.
    result = SimpleNamespace(z_mm=10., fuel_kg_s=fuel, air_total_kg_s=air,
                             air_core_kg_s=.25 * air, air_mixed_kg_s=.75 * air,
                             alpha_total=alpha, alpha_mixed=.75 * alpha,
                             alpha_section_total=float(air.sum() / (2. * fuel.sum())),
                             alpha_section_mixed=float(.75 * air.sum() / (2. * fuel.sum())),
                             undefined_area_fraction=float(area[0, 0] / area.sum()),
                             diagnostics=('synthetic example',), provenance={'source_kind': 'synthetic'},
                             source_states=[dict(id='s1', assigned_kg_s=19., crossed_kg_s=float(air.sum()),
                                                 reverse_kg_s=0., not_crossed_kg_s=0., outside_kg_s=0., E=.75,
                                                 status='supported')],
                             balance={'total_kg_s': 20., 'assigned_kg_s': 19., 'unassigned_kg_s': 1.})
    return cfg, plane, result, balance


class TestAirReports(unittest.TestCase):
    def test_csv_keeps_nan_mask_phases_units_provenance_and_cell_sum(self):
        cfg, plane, result, _ = case()
        before = deepcopy(cfg.air_model); fuel_before = plane.cell_mass_kg_s.copy()
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'air.csv'; write_air_csv(path, [result], cfg, [plane], times_s=[12.5])
            with path.open(encoding='utf-8-sig') as stream:
                rows = list(csv.DictReader(stream, delimiter=';'))
            self.assertEqual(rows[0]['alpha_total'], '')
            self.assertEqual(rows[0]['alpha_total_defined'], 'False')
            self.assertEqual(rows[1]['alpha_total_defined'], 'True')
            self.assertEqual(rows[0]['time_s'], '12.5')
            self.assertAlmostEqual(sum(float(r['air_total_kg_s']) for r in rows), result.air_total_kg_s.sum())
            self.assertAlmostEqual(sum(float(r['fuel_eligible_kg_s']) for r in rows), result.fuel_kg_s.sum())
            for row in rows:
                self.assertAlmostEqual(float(row['fuel_total_including_film_kg_s']),
                                       sum(float(row[k]) for k in ('fuel_eligible_kg_s', 'film_inner_kg_s', 'film_outer_kg_s')))
                self.assertAlmostEqual(float(row['air_total_flux_kg_m2_s']) * float(row['cell_area_m2']), float(row['air_total_kg_s']))
            metadata = json.loads(path.with_name('air_air.json').read_text())
            self.assertIn('kg/s', metadata['units'])
            self.assertEqual(json.loads(metadata['air_model']), cfg.air_model)
            self.assertTrue(path.with_name('air_Air_source_states.csv').exists())
        np.testing.assert_array_equal(plane.cell_mass_kg_s, fuel_before); self.assertEqual(cfg.air_model, before)

    def test_xlsx_masks_and_composite_global_ratios_are_blank(self):
        cfg, plane, result, _ = case(composite=True)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'air.xlsx'; write_air_xlsx(path, [result], cfg, [plane])
            tables = dict(_xlsx_sheets(path)); cells = tables['Air_cells']; sections = tables['Air_sections']
            header = sections[0]; row = sections[1]
            self.assertIn('different_times', row[header.index('scope')])
            self.assertIsNone(row[header.index('alpha_section_total')])
            self.assertIsNone(row[header.index('alpha_section_mixed')])
            h = cells[0]
            self.assertEqual(float(cells[2][h.index('time_s')]), .5)
            self.assertEqual(float(cells[2][h.index('step')]), 1.)
            self.assertIsNone(cells[1][h.index('alpha_total')])
            with ZipFile(path) as archive:
                for name in archive.namelist():
                    if name.startswith('xl/worksheets/'):
                        content = archive.read(name).lower()
                        self.assertNotIn(b'<v>nan</v>', content); self.assertNotIn(b'<v>inf</v>', content)
            states = tables['Air_source_states']; self.assertEqual(states[1][1], 's1')

    def test_large_source_registry_metadata_is_chunked_for_excel(self):
        cfg, plane, result, _ = case()
        cfg.air_model['sources'] = [dict(id='s'+str(i), kind='swirler', nozzle_id=i+1,
                                        provenance={'note': 'учебный источник '*30}) for i in range(140)]
        result.provenance['long_note'] = 'длинная запись '*4000
        cfg.air_model['sources'][0]['provenance']['long_source_note'] = 'источник '*5000
        metadata = dict(air_tables([result], cfg, [plane]))['Air_metadata']
        parts = [row for row in metadata[1:] if row[0] == 'air_model' or row[0].startswith('air_model__part_')]
        self.assertGreater(len(parts), 1); self.assertTrue(all(len(row[1]) <= 30000 for row in parts))
        self.assertEqual(json.loads(''.join(row[1] for row in parts)), cfg.air_model)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'air.csv'; write_air_csv(path, [result], cfg, [plane])
            manifest = json.loads(path.with_name('air_air.json').read_text())
            self.assertEqual(json.loads(manifest['air_model']), cfg.air_model)
            xlsx = Path(folder) / 'air.xlsx'; write_air_xlsx(xlsx, [result], cfg, [plane])
            with ZipFile(xlsx) as archive:
                for name in archive.namelist():
                    if name.startswith('xl/worksheets/'):
                        tree = ET.fromstring(archive.read(name))
                        for item in tree.iter('{http://schemas.openxmlformats.org/spreadsheetml/2006/main}t'):
                            self.assertLessEqual(len(item.text or ''), 32767)

    def test_actual_air_kernel_exports_exclude_films_and_preserve_flow(self):
        from fuelfield.mixture_field import compute_mixture
        cfg, plane, _, _ = case()
        cfg.air_model = dict(schema_version=1, enabled=True, total_kg_s=10., l0=2.,
                             distribution='uniform', max_path_mm=30., step_mm=1.,
                             sources=[dict(id='air', kind='other', method='fixed', flow_kg_s=10.,
                                           z_mm=0., radius_mm=250., kernel='uniform', mixed_fraction=.4)])
        result = compute_mixture(plane, cfg, [])
        self.assertAlmostEqual(result.air_total_kg_s.sum(), 10., places=13)
        self.assertAlmostEqual(result.air_mixed_kg_s.sum(), 4., places=13)
        self.assertAlmostEqual(result.alpha_section_total, 10. / (2. * result.fuel_kg_s.sum()))
        self.assertTrue(np.isnan(result.alpha_total[0, 0]))
        tables = dict(air_tables([result], cfg, [plane])); rows = tables['Air_cells']; header = rows[0]
        exported = sum(row[header.index('fuel_eligible_kg_s')] for row in rows[1:])
        self.assertEqual(exported, plane.suspended_cell_mass_kg_s.sum())
        self.assertLess(exported, plane.cell_mass_kg_s.sum())
        states = tables['Air_source_states']; self.assertEqual(states[1][1], 'air')
        self.assertEqual(states[1][3], 10.)

    def test_reverse_air_is_exported_separately_from_positive_air(self):
        cfg, plane, result, _ = case(); result.reverse_air_kg_s = np.full((2, 8), .5)
        tables = dict(air_tables([result], cfg, [plane])); cells = tables['Air_cells']; header = cells[0]
        self.assertEqual(sum(row[header.index('air_reverse_kg_s')] for row in cells[1:]), 8.)
        self.assertAlmostEqual(sum(row[header.index('air_total_kg_s')] for row in cells[1:]), result.air_total_kg_s.sum())
        sections = tables['Air_sections']; self.assertEqual(sections[1][sections[0].index('air_reverse_kg_s')], 8.)

    def test_different_z_cannot_be_paired_even_on_the_same_grid(self):
        cfg, plane, result, _ = case(); result.z_mm = 20.
        with self.assertRaisesRegex(ValueError, 'Координата z'):
            air_tables([result], cfg, [plane])

    def test_invalid_result_grid_is_rejected(self):
        cfg, plane, result, _ = case(); result.air_total_kg_s = np.zeros((1, 1))
        with self.assertRaisesRegex(ValueError, 'одну сетку'):
            air_tables([result], cfg, [plane])


class TestAirResultView(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from PyQt5.QtWidgets import QApplication
        cls.app = QApplication.instance() or QApplication([])

    def test_real_qt_selector_and_quasi_snapshot_do_not_recalculate_fuel(self):
        from fuelfield.air_results_gui import AirResultsDialog
        cfg, p0, r0, balance = case(composite=True)
        p1 = deepcopy(p0); p1.z_mm = 20.
        calls = []
        class Quasi:
            settings = SimpleNamespace(n_steps=3, times=np.array([0., 1., 2.]))
            def snapshot(self, index, step):
                calls.append((index, step)); plane = deepcopy((p0, p1)[index])
                plane.is_composite = False; plane.scan_times_s = None; plane.scan_steps = None
                return plane
        def mixture(plane, *_, **kwargs):
            result = deepcopy(r0); result.z_mm = plane.z_mm; result.balance = balance; return result
        with patch('fuelfield.air_results_gui.compute_mixture', side_effect=mixture), patch('fuelfield.air_results_gui.compute_air_field', return_value=object()), patch('fuelfield.core.calculate_plane', side_effect=AssertionError('fuel rerun')):
            dialog = AirResultsDialog(cfg, [], [p0, p1], Quasi()); dialog.show(); self.app.processEvents()
            self.assertIsNotNone(dialog.current_result); self.assertEqual(calls, [(0, 0)])
            self.assertEqual(dialog.source_table.item(0, 0).text(), 's1')
            dialog.time_slider.setValue(2); dialog.refresh(); self.app.processEvents()
            self.assertIn((0, 2), calls); self.assertIn('t=2', dialog.time_label.text())
            results, planes, times = dialog.export_bundle()
            self.assertEqual(times, [2., 2.]); self.assertEqual(len(results), 2); self.assertIn((1, 2), calls)
            dialog.scope.setCurrentIndex(1); self.app.processEvents()
            self.assertIn('единый мгновенный баланс', dialog.summary.text()); self.assertFalse(dialog.time_slider.isEnabled())
            self.assertEqual(len(dialog.figure.axes), 2)
            dialog.metric.setCurrentIndex(dialog.metric.findData('alpha_total')); self.app.processEvents()
            self.assertIn('α', dialog.figure.axes[0].get_title())
            self.assertEqual(cfg.air_model['sources'][0]['future'], {'keep': True})
            dialog.close(); self.app.processEvents()
