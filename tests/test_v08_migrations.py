"""Air is opt-in and cannot rewrite legacy fuel data or unknown JSON fields."""
from copy import deepcopy
import json
from pathlib import Path
import unittest

import numpy as np

from fuelfield.core import (Settings, calculate_plane, default_nozzles,
                            project_from_json, project_to_json)
from fuelfield.regression import RegressionSnapshot
from fuelfield.transport_settings import synthetic_settings


ROOT = Path(__file__).resolve().parents[1]


def project(schema=2):
    cfg = Settings(z_mm=[0., 60.], n_sample_radial=8, n_sample_angular=72,
                   n_angular=16, preview_resolution=200)
    nozzles = default_nozzles(cfg)
    for nozzle in nozzles:
        nozzle.sector_ml = [float(1 + (nozzle.id + sector) % 17)
                            for sector in range(36)]
        nozzle.sector_source = 'synthetic migration fixture'
        nozzle.manifold_id = 'A' if nozzle.id % 2 else 'B'
        nozzle.enabled = nozzle.id == 1
    nozzles[0].flow_kg_h = 36.
    if schema == 3:
        cfg.fuel_transport = synthetic_settings().to_dict()
    raw = project_to_json(cfg, nozzles)
    raw['schema_version'] = schema
    return raw


class TestAirPersistence(unittest.TestCase):
    def test_schema_1_2_3_without_air_never_enable_air_or_change_fuel_values(self):
        for schema in (1, 2, 3):
            with self.subTest(schema=schema):
                raw = project(schema)
                raw['settings'].pop('air_model')
                raw['future_project'] = {'keep': [None, False, 140]}
                raw['settings']['future_setting'] = {'sectors': [36]}
                raw['nozzles'][0]['future_nozzle'] = {'keep': ['original']}
                original = deepcopy(raw)
                cfg, nozzles = project_from_json(raw)
                self.assertEqual(raw, original)
                self.assertIs(cfg.air_model['enabled'], False)
                saved = project_to_json(cfg, nozzles)
                self.assertEqual(saved['nozzles'], original['nozzles'])
                for name, value in original['settings'].items():
                    self.assertEqual(saved['settings'][name], value, name)
                self.assertEqual(saved['future_project'], original['future_project'])
                self.assertEqual(saved['version'], '0.7.0' if schema == 3 else '0.6.0')
                self.assertEqual(saved['schema_version'], 3 if schema == 3 else 2)

    def test_disabled_air_roundtrip_preserves_nested_unknowns_without_aliases(self):
        raw = project()
        raw['settings']['air_model'] = {
            'schema_version': 1, 'enabled': False,
            'future_air': {'keep': [None, False, {'provenance': 'operator'}]}}
        original = deepcopy(raw)
        cfg, nozzles = project_from_json(raw)
        self.assertEqual(project_to_json(cfg, nozzles), original)
        raw['settings']['air_model']['future_air']['keep'].append('input edit')
        self.assertEqual(project_to_json(cfg, nozzles), original)
        output = project_to_json(cfg, nozzles)
        output['settings']['air_model']['future_air']['keep'].append('output edit')
        self.assertEqual(project_to_json(cfg, nozzles), original)
        restored = RegressionSnapshot.capture(cfg, nozzles).restore()
        self.assertEqual(project_to_json(*restored), original)
        self.assertEqual(project_to_json(*project_from_json(
            json.loads(json.dumps(original, allow_nan=False)))), original)

    def test_direct_settings_constructor_does_not_share_mutable_air_input(self):
        raw = {'schema_version': 1, 'enabled': False,
               'future': {'notes': ['independent']}}
        cfg = Settings(air_model=raw)
        raw['future']['notes'].append('changed')
        self.assertEqual(cfg.air_model['future']['notes'], ['independent'])
        first, second = Settings(), Settings()
        first.air_model['enabled'] = True
        self.assertIs(second.air_model['enabled'], False)

    def test_enabled_air_uses_own_version_without_changing_fuel_schema(self):
        for schema in (2, 3):
            with self.subTest(schema=schema):
                raw = project(schema)
                fuel = deepcopy(raw['nozzles'])
                raw['settings']['air_model'] = {
                    'schema_version': 1, 'enabled': True,
                    'total_kg_s': 0., 'sources': [],
                    'provenance': {'source_kind': 'synthetic migration test'},
                    'future_air': {'keep': [None, 140, 36]}}
                cfg, nozzles = project_from_json(raw)
                saved = project_to_json(cfg, nozzles)
                self.assertEqual(saved['version'], '0.8.4')
                self.assertEqual(saved['schema_version'], schema)
                self.assertEqual(saved['settings']['air_model'], raw['settings']['air_model'])
                self.assertEqual(saved['nozzles'], fuel)
                self.assertEqual(project_to_json(*project_from_json(saved)), saved)
                cfg.air_model['enabled'] = False
                disabled = project_to_json(cfg, nozzles)
                self.assertEqual(disabled['version'], '0.7.0' if schema == 3 else '0.6.0')
                self.assertEqual(disabled['settings']['air_model']['future_air'],
                                 {'keep': [None, 140, 36]})

    def test_invalid_air_records_are_diagnosed_without_mutating_input(self):
        for air in ([], {'schema_version': True, 'enabled': False},
                    {'schema_version': 2, 'enabled': False},
                    {'schema_version': 1, 'enabled': 'yes'},
                    {'schema_version': 1, 'enabled': True, 'total_kg_s': -1., 'sources': []}):
            with self.subTest(air=air):
                raw = project()
                raw['settings']['air_model'] = deepcopy(air)
                before = deepcopy(raw)
                with self.assertRaises(ValueError):
                    project_from_json(raw)
                self.assertEqual(raw, before)

    def test_archived_user_project_load_does_not_activate_air_or_rewrite_sources(self):
        path = ROOT / 'examples/demo_project_user_field.json'
        content = path.read_bytes()
        original = json.loads(content)
        cfg, nozzles = project_from_json(original)
        self.assertFalse(cfg.air_model['enabled'])
        saved = project_to_json(cfg, nozzles)
        for old, new in zip(original['nozzles'], saved['nozzles']):
            for name in ('id', 'enabled', 'sector_ml', 'flow_kg_h',
                         'theta_deg', 'radius_mm'):
                self.assertEqual(new[name], old[name], (old['id'], name))
        self.assertEqual(path.read_bytes(), content)

    def test_disabled_air_metadata_cannot_change_fuel_matrix(self):
        cfg, nozzles = project_from_json(project())
        before = calculate_plane(nozzles, cfg, 60.)
        cfg.air_model = {'schema_version': 1, 'enabled': False,
                         'future_metadata': {'source': 'nonphysical annotation'}}
        after = calculate_plane(nozzles, cfg, 60.)
        np.testing.assert_array_equal(before.cell_mass_kg_s, after.cell_mass_kg_s)
        np.testing.assert_array_equal(before.flux_kg_m2_s, after.flux_kg_m2_s)
        self.assertEqual(before.injected_kg_s, after.injected_kg_s)


class TestAirDraftPersistenceUI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from PyQt5.QtWidgets import QApplication
        cls.app = QApplication.instance() or QApplication([])

    def test_disabled_stale_links_and_overbudget_draft_remain_saveable(self):
        from fuelfield.air_gui import AirBalanceDialog
        raw = dict(schema_version=1, enabled=False, total_kg_s=1.,
                   sources=[dict(id='stale', kind='hole', method='fixed',
                                 flow_kg_s=2., object_id='removed-object')])
        cfg = Settings(air_model=raw)
        dialog = AirBalanceDialog(cfg, default_nozzles(cfg))
        try:
            received = []
            dialog.applied.connect(received.append)
            self.assertTrue(dialog.apply_button.isEnabled())
            dialog.apply()
            self.assertEqual(len(received), 1)
            self.assertFalse(received[0]['enabled'])
            self.assertEqual(received[0]['sources'][0]['object_id'], 'removed-object')
            self.assertEqual(received[0]['sources'][0]['flow_kg_s'], 2.)
            self.assertEqual(cfg.air_model, raw)
        finally:
            dialog.close()
            dialog.deleteLater()
            self.app.processEvents()

    def test_noop_editor_apply_preserves_values_below_display_precision(self):
        from fuelfield.air_gui import AirBalanceDialog
        raw = dict(schema_version=1, enabled=True, total_kg_s=1e-13, l0=1e-13,
                   interaction_strength=1e-7, area_allocation_fraction=1e-7,
                   sources=[])
        cfg = Settings(air_model=raw)
        dialog = AirBalanceDialog(cfg, default_nozzles(cfg))
        try:
            received = []
            dialog.applied.connect(received.append)
            dialog.apply()
            self.assertEqual(len(received), 1)
            for key in ('total_kg_s', 'l0', 'interaction_strength', 'area_allocation_fraction'):
                self.assertEqual(received[0][key], raw[key], key)
            self.assertEqual(cfg.air_model, raw)
        finally:
            dialog.close()
            dialog.deleteLater()
            self.app.processEvents()


if __name__ == '__main__':
    unittest.main()
