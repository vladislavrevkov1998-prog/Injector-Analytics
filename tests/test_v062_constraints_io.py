"""Independent persistence and legacy checks for explicit Sketcher constraints."""
from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from fuelfield.chamber import Chamber, Primitive, legacy_geometry
from fuelfield.core import calculate_plane, project_from_json, project_to_json
from fuelfield.regression import RegressionSnapshot


def endpoint(identity, role, **extra):
    return dict(primitive_id=identity, endpoint=role, **extra)


def fixture(explicit=True):
    cfg, nozzles = project_from_json(json.loads(
        (ROOT / 'examples/demo_project_user_field.json').read_text()))
    cfg.z_mm = [0., 30., 60.]
    cfg.n_sample_radial = 8
    cfg.n_sample_angular = 72
    cfg.n_angular = 16
    cfg.placement_mode = 'manual'
    cfg.geometry = legacy_geometry(cfg, nozzles)
    cfg.geometry['enabled'] = True
    cfg.geometry['primitives'].extend([
        dict(id='old-user-arc-8b63', wall='reference', kind='arc',
             start=[5., 260.], end=[15., 260.], center=[10., 265.],
             clockwise=False, unit='mm', future_primitive={'keep': [None, 'arc']}),
        dict(id='old-user-line-b129', wall='reference', kind='line',
             start=[15., 260.], end=[25., 270.], unit='mm',
             future_line={'keep': {'unit': 'operator-defined'}}),
    ])
    cfg.geometry['future_geometry'] = {'keep': [1, False, {'note': 'synthetic constraints'}]}
    if explicit:
        cfg.geometry['schema_version'] = 2
        cfg.geometry['primitives'][-1]['fixed'] = True
        cfg.geometry['constraints'] = [dict(
            id='C1', kind='tangent',
            a=endpoint('old-user-arc-8b63', 'end', future_endpoint={'keep': [1, 2]}),
            b=endpoint('old-user-line-b129', 'start'),
            future_constraint={'keep': [None, {'source': 'operator'}]}), dict(
            id='C2', kind='coincident',
            a=endpoint('old-user-arc-8b63', 'start'),
            b=endpoint('old-user-line-b129', 'end'), enabled=False,
            future_disabled={'keep': 'draft relation'})]
    return cfg, nozzles


def api():
    from fuelfield import chamber_constraints
    return chamber_constraints


class TestConstraintPersistence(unittest.TestCase):
    def test_legacy_load_keeps_schema_ids_and_does_not_create_hidden_fields(self):
        cfg, nozzles = fixture(explicit=False)
        raw = project_to_json(cfg, nozzles)
        original = deepcopy(raw)
        loaded, restored = project_from_json(raw)
        self.assertEqual(project_to_json(loaded, restored), original)
        self.assertEqual(raw, original)
        self.assertEqual(loaded.geometry['schema_version'], 1)
        self.assertNotIn('constraints', loaded.geometry)
        self.assertTrue(all('fixed' not in p for p in loaded.geometry['primitives']))
        self.assertEqual([p['id'] for p in loaded.geometry['primitives']],
                         [p['id'] for p in cfg.geometry['primitives']])

    def test_schema_two_roundtrip_keeps_nested_unknowns_and_input_isolation(self):
        cfg, nozzles = fixture()
        raw = project_to_json(cfg, nozzles)
        raw['future_project'] = {'keep': [1, None]}
        raw['settings']['future_settings'] = {'keep': 'source'}
        raw['nozzles'][0]['future_nozzle'] = {'keep': [36, 140]}
        before = deepcopy(raw)
        loaded, restored = project_from_json(raw)
        self.assertEqual(project_to_json(loaded, restored), before)
        self.assertEqual(project_to_json(*RegressionSnapshot.capture(loaded, restored).restore()), before)
        raw['settings']['geometry']['constraints'][0]['a']['future_endpoint']['keep'].append('changed')
        self.assertEqual(project_to_json(loaded, restored), before)
        output = project_to_json(loaded, restored)
        output['settings']['geometry']['constraints'][0]['future_constraint']['keep'].append('changed')
        self.assertEqual(project_to_json(loaded, restored), before)
        self.assertEqual(project_to_json(*project_from_json(
            json.loads(json.dumps(before, ensure_ascii=False)))), before)

    def test_future_schema_fixed_types_and_constraint_references_are_diagnosed(self):
        raw = project_to_json(*fixture())
        changes = (
            ('future geometry schema', lambda g: g.__setitem__('schema_version', 3)),
            ('boolean geometry schema', lambda g: g.__setitem__('schema_version', True)),
            ('invalid fixed value', lambda g: g['primitives'][-1].__setitem__('fixed', 'false')),
            ('numeric fixed value', lambda g: g['primitives'][-1].__setitem__('fixed', 1)),
            ('invalid enabled value', lambda g: g['constraints'][0].__setitem__('enabled', 'true')),
            ('constraint must be object', lambda g: g['constraints'].__setitem__(0, None)),
            ('endpoint must be object', lambda g: g['constraints'][0].__setitem__('a', [])),
            ('missing primitive', lambda g: g['constraints'][0]['a'].__setitem__('primitive_id', 'missing')),
            ('not an endpoint', lambda g: g['constraints'][0]['a'].__setitem__('endpoint', 'center')),
            ('unknown active kind', lambda g: g['constraints'][0].__setitem__('kind', 'unsupported')),
            ('duplicate constraint ID', lambda g: g['constraints'].append(deepcopy(g['constraints'][0]))),
            ('deleted primitive', lambda g: g['primitives'].pop()),
        )
        for title, modify in changes:
            with self.subTest(case=title):
                data = deepcopy(raw)
                modify(data['settings']['geometry'])
                with self.assertRaises(ValueError):
                    project_from_json(data)

    def test_materialize_first_edit_preserves_all_record_metadata_and_physical_values(self):
        cfg, nozzles = fixture(explicit=False)
        original = deepcopy(cfg.geometry)
        edited = deepcopy(original)
        result = api().materialize_constraints(edited)
        migrated = edited if result is None else result
        self.assertEqual(cfg.geometry, original)
        self.assertEqual(migrated['schema_version'], 2)
        self.assertTrue(migrated['constraints'])
        for before, after in zip(original['primitives'], migrated['primitives']):
            for key, value in before.items():
                self.assertEqual(after[key], value)
        self.assertEqual(migrated['future_geometry'], original['future_geometry'])
        self.assertEqual(migrated['objects'], original['objects'])
        self.assertEqual(migrated['sections'], original['sections'])
        self.assertEqual(migrated['groups'], original['groups'])

    def test_explicit_empty_constraint_list_does_not_restore_removed_implicit_join(self):
        cfg, _ = fixture()
        cfg.geometry['constraints'] = []
        original = deepcopy(cfg.geometry)
        result = api().materialize_constraints(cfg.geometry)
        result = cfg.geometry if result is None else result
        self.assertEqual(result['constraints'], [])
        self.assertEqual(result, original)

    def test_remove_is_pure_and_adding_constraint_keeps_existing_unknown_metadata(self):
        cfg, _ = fixture()
        original = deepcopy(cfg.geometry)
        result = api().remove_constraints(cfg.geometry, ['C2'])
        self.assertEqual(cfg.geometry, original)
        self.assertEqual([c['id'] for c in result['constraints']], ['C1'])
        self.assertEqual(result['constraints'][0], original['constraints'][0])
        result['constraints'][0]['future_constraint']['keep'].append('changed')
        self.assertEqual(cfg.geometry, original)
        added = api().add_constraint(
            original, 'coincident', endpoint('old-user-arc-8b63', 'end'),
            endpoint('old-user-line-b129', 'start'))
        self.assertEqual(cfg.geometry, original)
        self.assertEqual(added['constraints'][:2], original['constraints'])
        ids = [entry['id'] for entry in added['constraints']]
        self.assertEqual(len(ids), len(set(ids)))

    def test_constraints_and_fixed_do_not_change_fuel_cells_or_140_nozzle_data(self):
        cfg, nozzles = fixture()
        baseline = deepcopy(cfg)
        baseline.geometry = {}
        before = project_to_json(cfg, nozzles)
        for z in cfg.z_mm:
            actual = calculate_plane(nozzles, cfg, z)
            expected = calculate_plane(nozzles, baseline, z)
            for key in ('cell_mass_kg_s', 'cell_area_m2', 'flux_kg_m2_s'):
                np.testing.assert_array_equal(getattr(actual, key), getattr(expected, key))
        restored_cfg, restored_nozzles = project_from_json(before)
        self.assertEqual(project_to_json(restored_cfg, restored_nozzles)['nozzles'], before['nozzles'])
        self.assertEqual(project_to_json(cfg, nozzles), before)

    def test_radius_message_is_stale_explicit_radius_not_coordinate_geometry(self):
        cfg, _ = fixture(explicit=False)
        arc = deepcopy(cfg.geometry['primitives'][-2])
        arc['radius'] = 7.
        with self.assertRaisesRegex(ValueError, 'радиус не согласован'):
            Primitive(arc)
        arc.pop('radius')
        self.assertAlmostEqual(Primitive(arc).radius, np.sqrt(50.))


class TestConstraintUndoIO(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from PyQt5.QtWidgets import QApplication
        cls.app = QApplication.instance() or QApplication([])

    def test_constraints_fixed_unknowns_survive_undo_native_save_open(self):
        from fuelfield.gui import MainWindow
        window = MainWindow()
        try:
            window._restoring = True
            window.cfg, window.nozzles = fixture()
            window._fill_controls()
            window._fill_nozzles()
            window._reset_history()
            before = project_to_json(window.cfg, window.nozzles)
            after = deepcopy(before)
            after['settings']['geometry']['constraints'] = api().remove_constraints(
                after['settings']['geometry'], ['C2'])['constraints']
            after['settings']['geometry']['primitives'][-2]['fixed'] = True
            window.apply_geometry_state(after, 'Synthetic relation removal and fixation')
            self.assertEqual(project_to_json(window.cfg, window.nozzles), after)
            window.undo_stack.undo()
            self.assertEqual(project_to_json(window.cfg, window.nozzles), before)
            window.undo_stack.redo()
            self.assertEqual(project_to_json(window.cfg, window.nozzles), after)
            with tempfile.TemporaryDirectory() as folder:
                path = Path(folder) / 'constraints_project.json'
                window.project_file = str(path)
                window.save_project()
                self.assertEqual(json.loads(path.read_text(encoding='utf-8')), after)
                with patch('fuelfield.gui.QFileDialog.getOpenFileName',
                           return_value=(str(path), 'JSON')):
                    window.open_project()
                self.assertEqual(project_to_json(window.cfg, window.nozzles), after)
                self.assertEqual(window.undo_stack.count(), 0)
        finally:
            window.close()
            window.deleteLater()
            self.app.processEvents()


if __name__ == '__main__':
    unittest.main()
