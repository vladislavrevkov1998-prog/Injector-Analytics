"""Independent v0.6 project migration and cross-feature preservation checks.

The existing user projects stay untouched. Synthetic geometry is deliberately
marked as such; no synthetic measurements replace their source records.
"""
from copy import deepcopy
from dataclasses import replace
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

from fuelfield.chamber import Chamber, apply_groups, legacy_geometry, sync_groups
from fuelfield.core import (calculate_plane, evaluate_continuous_field,
                            project_from_json, project_to_json)
from fuelfield.quasi import PressureChannel, QuasiSettings, calculate_quasi
from fuelfield.regression import RegressionSnapshot, RegressionTarget
from fuelfield.analytics_io import write_epures_xlsx


LEGACY_PROJECTS = (
    'examples/demo_project_140.json',
    'examples/demo_project_v02.json',
    'examples/demo_project_dityakin.json',
    'examples/demo_project_user_field.json',
    'examples/demo_project_v04_synthetic.json',
    'examples/fuel_field_project_quasi.json',
    'fuel_field_project_clock.json',
    'fuel_field_project_quasi.json',
)


def user_project():
    return project_from_json(json.loads(
        (ROOT / 'examples/demo_project_user_field.json').read_text()))


def numeric_case(custom_radii=False):
    """Keep all 140 identities and measurements; only reduce quadrature cost."""
    cfg, nozzles = user_project()
    cfg.z_mm = [0., 30., 60.]
    cfg.n_angular = 16
    cfg.n_sample_radial = 8
    cfg.n_sample_angular = 72
    cfg.preview_resolution = 200
    cfg.placement_mode = 'manual'
    if custom_radii:
        cfg.radial_centers_mm = (cfg.r_in_mm +
            np.array([.04, .19, .37, .54, .76, .95]) *
            (cfg.r_out_mm-cfg.r_in_mm)).tolist()
    return cfg, nozzles


def decorated_geometry(cfg, nozzles, variable=False):
    raw = legacy_geometry(cfg, nozzles)
    raw['enabled'] = True
    raw['custom_geometry'] = {'list': [None, False, 1.25, {'keep': 'geometry'}]}
    raw['types'] = {'user_lance': {'units': {'unknown': 'test-unit'},
                                 'arbitrary': {'keep': [1, 2, 3]}}}
    raw['primitives'][0]['custom_primitive'] = {'keep': [1, None, 'inner']}
    if variable:
        raw['primitives'][1].update(kind='arc', start=[0., 300.],
                                    end=[60., 300.], center=[30., 300.],
                                    clockwise=True, radius=30.)
    raw['primitives'][1]['custom_arc'] = {'keep': {'radius_note': 'analytical'}}
    raw['sections'][1]['custom_section'] = {'keep': [30., 'manual coordinate']}
    raw['objects'][0]['custom_nozzle_object'] = {'keep': 'stable linked object'}
    raw['objects'].append(dict(
        id='custom-insert', type='user_lance', wall='outer',
        primitive_id=raw['primitives'][1]['id'], z_mm=30., radius_mm=300.,
        theta_deg=12.5, radial_offset_mm=-3.,
        parameters={'flow_kg_s': None, 'penetration_mm': 3.,
                    'inclination_deg': None, 'unknown_parameter': [1, None]},
        units={'penetration_mm': 'mm', 'unknown_parameter': 'custom-unit'},
        provenance={'source_kind': 'synthetic', 'custom': {'keep': True}},
        custom_object={'keep': [1, {'nested': 'insert'}]}))
    raw['groups'] = [dict(id='irregular-nozzles', members=['N1', 'N2', 'N3'],
                          mode='steps', steps_deg=[110., 120., 130.],
                          direction=-1, phase_deg=17., row_offset_deg=.25,
                          custom_group={'keep': ['steps', {'value': 360}]})]
    return raw


def decorate_project():
    cfg, nozzles = numeric_case()
    cfg.geometry = decorated_geometry(cfg, nozzles, variable=True)
    raw = project_to_json(cfg, nozzles)
    raw['custom_root'] = {'keep': [1, None, {'source': 'operator'}]}
    raw['settings']['custom_settings'] = {'keep': [False, 'notes']}
    raw['nozzles'][0]['custom_nozzle'] = {'keep': [1.25, {'raw': True}]}
    return raw


class TestV06Migration(unittest.TestCase):
    def assert_plane_equal(self, actual, expected, exact=True):
        compare = np.testing.assert_array_equal if exact else np.testing.assert_allclose
        for key in ('cell_mass_kg_s', 'cell_area_m2', 'flux_kg_m2_s',
                    'radial_centers_mm', 'radial_edges_mm', 'theta_centers_deg'):
            compare(getattr(actual, key), getattr(expected, key), err_msg=key)
        for key in ('injected_kg_s', 'inside_kg_s', 'inner_kg_s', 'outer_kg_s',
                    'wall_bound_kg_s'):
            if expected.is_composite and np.isnan(getattr(expected, key)):
                self.assertTrue(actual.is_composite)
                self.assertTrue(np.isnan(getattr(actual, key)), key)
                continue
            self.assertAlmostEqual(getattr(actual, key), getattr(expected, key),
                                   delta=1e-13, msg=key)

    def test_all_legacy_projects_keep_140_records_and_unknown_fields(self):
        for filename in LEGACY_PROJECTS:
            with self.subTest(project=filename):
                raw = json.loads((ROOT / filename).read_text())
                raw['custom_root'] = {'keep': [1, {'label': filename}]}
                raw['settings']['custom_settings'] = {'keep': [None, True]}
                raw['nozzles'][0]['custom_nozzle'] = {'keep': ['sectors', 36]}
                before = deepcopy(raw)
                cfg, nozzles = project_from_json(raw)
                self.assertEqual(raw, before, 'Migration must not modify caller JSON')
                canonical = project_to_json(cfg, nozzles)
                self.assertEqual(canonical['format'], 'fuel-field-pilot-v1')
                self.assertEqual(canonical['schema_version'], 2)
                self.assertEqual(canonical['version'], '0.6.0')
                self.assertEqual([n.id for n in nozzles],
                                 [n['id'] for n in raw['nozzles']])
                self.assertEqual({n.id for n in nozzles}, set(range(1, 141)))
                for source, target in zip(raw['nozzles'], canonical['nozzles']):
                    for key, value in source.items():
                        self.assertEqual(target[key], value, f'{filename}: {key}')
                self.assertEqual(canonical['custom_root'], raw['custom_root'])
                self.assertEqual(canonical['settings']['custom_settings'],
                                 raw['settings']['custom_settings'])
                restored = project_to_json(*project_from_json(
                    json.loads(json.dumps(canonical))))
                self.assertEqual(restored, canonical)

    def test_unknown_geometry_at_every_level_roundtrips_and_snapshot_is_exact(self):
        raw = decorate_project()
        cfg, nozzles = project_from_json(raw)
        canonical = project_to_json(cfg, nozzles)
        self.assertEqual(canonical, raw)
        snapshot = RegressionSnapshot.capture(cfg, nozzles)
        self.assertEqual(project_to_json(*snapshot.restore()), raw)
        self.assertEqual(project_to_json(*project_from_json(
            json.loads(json.dumps(raw, ensure_ascii=False)))), raw)

    def test_loaded_input_and_export_have_no_mutable_aliases(self):
        raw = decorate_project()
        reference = deepcopy(raw)
        cfg, nozzles = project_from_json(raw)
        raw['custom_root']['keep'].append('mutated input')
        raw['settings']['geometry']['objects'][0]['custom_nozzle_object']['keep'] = 'bad'
        raw['nozzles'][0]['custom_nozzle']['keep'][1]['raw'] = False
        self.assertEqual(project_to_json(cfg, nozzles), reference)
        exported = project_to_json(cfg, nozzles)
        exported['custom_root']['keep'].append('mutated output')
        exported['settings']['custom_settings']['keep'].append('mutated output')
        exported['settings']['geometry']['primitives'][0]['custom_primitive']['keep'].append('bad')
        exported['nozzles'][0]['custom_nozzle']['keep'].append('bad')
        self.assertEqual(project_to_json(cfg, nozzles), reference,
                         'Export must be an isolated JSON value')

    def test_future_project_and_geometry_schemas_are_rejected(self):
        raw = decorate_project()
        for schema in (3, 9, True, '2', 2.5, None):
            with self.subTest(schema=schema):
                broken = deepcopy(raw)
                broken['schema_version'] = schema
                with self.assertRaises(ValueError):
                    project_from_json(broken)
        broken = deepcopy(raw)
        broken['settings']['geometry']['schema_version'] = 2
        with self.assertRaises(ValueError):
            project_from_json(broken)
        for schema in (1, 2):
            accepted = deepcopy(raw)
            accepted['schema_version'] = schema
            self.assertEqual(project_to_json(*project_from_json(accepted))['schema_version'], 2)

    def test_other_json_kinds_do_not_open_as_projects(self):
        for filename in ('examples/demo_regression_v04_synthetic.json',
                         'examples/demo_user_field_140_two_collectors.quasi.json',
                         'examples/pressure_example_pulse.quasi.json'):
            with self.subTest(filename=filename):
                with self.assertRaises(ValueError):
                    project_from_json(json.loads((ROOT / filename).read_text()))

    def test_empty_disabled_and_constant_geometry_are_numerically_identical(self):
        for custom in (False, True):
            with self.subTest(custom_radii=custom):
                cfg, nozzles = numeric_case(custom)
                original = project_to_json(cfg, nozzles)
                baseline = [calculate_plane(nozzles, cfg, z) for z in cfg.z_mm]
                disabled = replace(cfg, geometry=legacy_geometry(cfg, nozzles))
                constant = deepcopy(disabled)
                constant.geometry['enabled'] = True
                for i, z in enumerate(cfg.z_mm):
                    self.assert_plane_equal(calculate_plane(nozzles, disabled, z), baseline[i])
                    self.assert_plane_equal(calculate_plane(nozzles, constant, z), baseline[i])
                a = evaluate_continuous_field(nozzles, cfg, 30., resolution=96)
                b = evaluate_continuous_field(nozzles, constant, 30., resolution=96)
                np.testing.assert_array_equal(a.flux_kg_m2_s, b.flux_kg_m2_s)
                np.testing.assert_array_equal(a.masked_flux_kg_m2_s.mask,
                                              b.masked_flux_kg_m2_s.mask)
                self.assertEqual(project_to_json(cfg, nozzles), original)

    def test_all_legacy_project_constant_profiles_preserve_numerical_results(self):
        for filename in LEGACY_PROJECTS:
            with self.subTest(project=filename):
                cfg, nozzles = project_from_json(json.loads((ROOT / filename).read_text()))
                # Integration-node counts are independent of the saved geometry.
                cfg.n_sample_radial = 8
                cfg.n_sample_angular = 72
                cfg.n_angular = 16
                cfg.geometry = {}
                canonical = project_to_json(cfg, nozzles)
                constant = deepcopy(cfg)
                constant.geometry = legacy_geometry(cfg, nozzles)
                constant.geometry['enabled'] = True
                for z in (cfg.z_mm[0], cfg.z_mm[-1]):
                    self.assert_plane_equal(calculate_plane(nozzles, constant, z),
                                            calculate_plane(nozzles, cfg, z))
                self.assertEqual(project_to_json(cfg, nozzles), canonical)

    def test_constant_geometry_quasi_keeps_custom_radii_and_all_manifolds(self):
        cfg, nozzles = numeric_case(custom_radii=True)
        cfg.n_angular = 8
        cfg.pressure_mode = 'calibrated'
        cfg.flow_pressure_kgf_cm2 = 10.
        cfg.sector_pressure_kgf_cm2 = 10.
        cfg.target_pressure_kgf_cm2 = 10.
        cfg.allow_extrapolation = False
        cfg.atomizer_types = [dict(id='synthetic-curve', name='Synthetic preservation test',
                                  kind='experimental', points=[[5., 10.], [10., 30.], [30., 100.]])]
        for n in nozzles:
            n.atomizer_type_id = 'synthetic-curve'
            n.flow_pressure_kgf_cm2 = 10.
            n.sector_pressure_kgf_cm2 = 10.
            n.manifold_id = 'A' if n.id <= 70 else 'B'
        mode = QuasiSettings(8, .25, 0., 0., 1, 'A', (
            PressureChannel('A', tuple(np.linspace(10., 30., 8))),
            PressureChannel('B', tuple(np.linspace(30., 10., 8)))))
        cfg.quasistationary = True
        cfg.quasi_settings = mode.to_dict()
        before = project_to_json(cfg, nozzles)
        baseline = calculate_quasi(nozzles, cfg, preview=False)
        constant = deepcopy(cfg)
        constant.geometry = legacy_geometry(cfg, nozzles)
        constant.geometry['enabled'] = True
        result = calculate_quasi(nozzles, constant, preview=False)
        np.testing.assert_array_equal(result.flows, baseline.flows)
        for i, plane in enumerate(result.composites):
            self.assert_plane_equal(plane, baseline.composites[i])
            np.testing.assert_array_equal(plane.scan_steps, baseline.composites[i].scan_steps)
            np.testing.assert_array_equal(plane.scan_times_s, baseline.composites[i].scan_times_s)
            for step in (0, 3, 7):
                self.assert_plane_equal(result.snapshot(i, step), baseline.snapshot(i, step))
        self.assertEqual(project_to_json(cfg, nozzles), before)
        self.assertEqual([n.id for n in result.nozzles], list(range(1, 141)))
        self.assertEqual([n.enabled for n in result.nozzles], [n.enabled for n in nozzles])
        self.assertEqual([n.manifold_id for n in result.nozzles], [n.manifold_id for n in nozzles])
        self.assertEqual([n.sector_ml for n in result.nozzles], [n.sector_ml for n in nozzles])

    def test_linked_nozzle_object_uses_effective_nozzle_not_stale_coordinates(self):
        cfg, nozzles = numeric_case()
        raw = decorated_geometry(cfg, nozzles)
        model = Chamber(raw)
        obj = model.objects[0]
        obj.update(z_mm=999., radius_mm=1., theta_deg=99.)
        nozzles[0].z_tip_mm = 2.75
        nozzles[0].geometry_overrides = list(set(nozzles[0].geometry_overrides) | {'z_tip_mm'})
        nozzles[0].radius_mm = 271.125
        nozzles[0].theta_deg = 357.125
        self.assertEqual(model.object_position(obj, nozzles, cfg),
                         (2.75, 271.125, 357.125))

    def test_bound_custom_objects_and_irregular_groups_preserve_metadata(self):
        cfg, nozzles = numeric_case()
        raw = decorated_geometry(cfg, nozzles, variable=True)
        before = deepcopy(raw)
        model = Chamber(raw)
        custom = next(obj for obj in model.objects if obj['id'] == 'custom-insert')
        self.assertEqual(model.object_position(custom, nozzles, cfg), (30., 327., 12.5))
        data, positioned = apply_groups(raw, nozzles)
        self.assertEqual(raw, before)
        self.assertEqual(data, before)
        np.testing.assert_array_equal([positioned[i].theta_deg for i in range(3)],
                                      [17.25, 267.25, 147.25])
        for a, b in zip(nozzles, positioned):
            va, vb = deepcopy(vars(a)), deepcopy(vars(b))
            va.pop('theta_deg'); vb.pop('theta_deg')
            self.assertEqual(va, vb)
        positioned[1].theta_deg = 250.5
        synced = sync_groups(data, positioned)
        self.assertEqual(synced['groups'][0]['mode'], 'individual')
        self.assertEqual(synced['groups'][0]['custom_group'], before['groups'][0]['custom_group'])
        self.assertEqual(synced['objects'], before['objects'])
        _, second = apply_groups(synced, positioned)
        np.testing.assert_array_equal([n.theta_deg for n in second],
                                      [n.theta_deg for n in positioned])

    def test_variable_profile_fuel_target_validates_per_observation_section(self):
        cfg, nozzles = numeric_case()
        cfg.geometry = decorated_geometry(cfg, nozzles, variable=True)
        for quantity in ('flux', 'cell_mass'):
            with self.subTest(quantity=quantity):
                target = RegressionTarget.from_project(cfg, nozzles, quantity,
                                                       source_kind='synthetic')
                mapping = target.validate(cfg)
                self.assertEqual(len(mapping), len(cfg.z_mm)*cfg.n_radial*cfg.n_angular)
                self.assertEqual({entry[0] for entry in mapping}, set(cfg.z_mm))
                self.assertNotEqual(cfg.radial_centers_and_edges(0.)[0].tolist(),
                                    cfg.radial_centers_and_edges(30.)[0].tolist())

    def test_geometry_metadata_xlsx_is_chunked_and_lossless(self):
        import openpyxl
        cfg, nozzles = numeric_case()
        cfg.geometry = decorated_geometry(cfg, nozzles, variable=True)
        cfg.geometry['custom_long_note'] = 'synthetic geometry metadata Ω ' * 2000
        planes = [calculate_plane(nozzles, cfg, z) for z in cfg.z_mm]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'geometry.xlsx'
            write_epures_xlsx(path, planes, cfg)
            workbook = openpyxl.load_workbook(path, data_only=True)
            try:
                self.assertIn('Metadata', workbook.sheetnames)
                for sheet in workbook:
                    for row in sheet.values:
                        for value in row:
                            if isinstance(value, str):
                                self.assertLessEqual(len(value), 32767,
                                                     f'Excel text overflow in {sheet.title}')
                rows = list(workbook['Metadata'].values)
                self.assertEqual(tuple(rows[0][:3]), ('record', 'part', 'json'))
                pieces = sorted((int(row[1]), row[2]) for row in rows[1:]
                                if row[0] == 'settings')
                restored = json.loads(''.join(piece for _, piece in pieces))
                self.assertEqual(restored, project_to_json(cfg, nozzles)['settings'])
            finally:
                workbook.close()


class TestV06MigrationQt(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from PyQt5.QtWidgets import QApplication
        cls.app = QApplication.instance() or QApplication([])

    def test_geometry_unknowns_survive_project_edit_undo_redo_and_snapshot(self):
        from fuelfield.gui import MainWindow
        window = MainWindow()
        try:
            raw = decorate_project()
            window._restoring = True
            window.cfg, window.nozzles = project_from_json(raw)
            window._fill_controls()
            window._fill_nozzles()
            window._reset_history()
            modified = deepcopy(raw)
            modified['settings']['geometry']['primitives'][1]['center'][1] += 1.
            modified['settings']['geometry']['primitives'][1]['start'][1] += 1.
            modified['settings']['geometry']['primitives'][1]['end'][1] += 1.
            window.apply_project_state(modified, 'Synthetic geometry migration test')
            self.assertEqual(project_to_json(window.cfg, window.nozzles), modified)
            window.undo_stack.undo()
            self.assertEqual(project_to_json(window.cfg, window.nozzles), raw)
            window.undo_stack.redo()
            self.assertEqual(project_to_json(window.cfg, window.nozzles), modified)
            self.assertEqual(project_to_json(*RegressionSnapshot.capture(
                window.cfg, window.nozzles).restore()), modified)
        finally:
            window.close()
            window.deleteLater()
            self.app.processEvents()


if __name__ == '__main__':
    unittest.main()
