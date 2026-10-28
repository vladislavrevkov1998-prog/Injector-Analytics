"""Template air inputs inherit nozzle kind and retain explicit manual data."""
from copy import deepcopy
from types import SimpleNamespace
import unittest
from PyQt5.QtWidgets import QApplication
from fuelfield.core import Settings, default_nozzles, project_to_json, project_from_json
from fuelfield.air_sources import resolve_sources, validate_air_model
from fuelfield.air_templates import effective_air_model, inherited_swirlers
from fuelfield.v02 import validate_catalog


def fixture():
    cfg = Settings(air_model=dict(enabled=True, total_kg_s=10., sources=[]))
    cfg.nozzle_presets['uncooled']['air'] = dict(method='fixed', flow_kg_s=.02,
        sigma_r_mm=3., direction=[1., .2, .1], future={'keep': [1, 2]})
    cfg.nozzle_presets['cooled']['air'] = dict(method='fixed', flow_kg_s=.04,
        sigma_r_mm=8., direction=[1., 0., 0.])
    ns = default_nozzles(cfg)
    raw = dict(enabled=True, total_kg_s=10., sources=[dict(id='S1',kind='swirler',
        nozzle_id=1, method='unknown', template_mode='inherit', air_overrides=[])])
    return cfg, ns, raw


class TestAirTemplates(unittest.TestCase):
    def test_effective_parameters_follow_installed_nozzle_kind_without_mutating_input(self):
        cfg, ns, raw = fixture(); before = deepcopy(raw)
        first = resolve_sources(raw, cfg, ns)[0]
        self.assertEqual(first['flow_kg_s'], .02); self.assertEqual(first['sigma_r_mm'], 3.)
        ns[0].nozzle_kind = 'cooled'
        second = resolve_sources(raw, cfg, ns)[0]
        self.assertEqual(second['flow_kg_s'], .04); self.assertEqual(second['sigma_r_mm'], 8.)
        self.assertEqual(raw, before)

    def test_individual_override_survives_template_and_type_changes(self):
        cfg, ns, raw = fixture()
        raw['sources'][0].update(flow_kg_s=.123, air_overrides=['flow_kg_s'])
        cfg.nozzle_presets['uncooled']['air']['sigma_r_mm'] = 15.
        result = effective_air_model(raw, cfg, ns)['sources'][0]
        self.assertEqual(result['flow_kg_s'], .123); self.assertEqual(result['sigma_r_mm'], 15.)
        ns[0].nozzle_kind = 'cooled'
        result = effective_air_model(raw, cfg, ns)['sources'][0]
        self.assertEqual(result['flow_kg_s'], .123); self.assertEqual(result['sigma_r_mm'], 8.)

    def test_inherited_flow_reaches_the_actual_spatial_matrix_and_budget(self):
        from fuelfield.core import calculate_plane
        from fuelfield.air_distribution import compute_air_field
        cfg, ns, raw = fixture();cfg.air_model=raw;ns[0].flow_kg_h=3.6
        result=compute_air_field(calculate_plane(ns,cfg,20.),cfg,ns)
        self.assertAlmostEqual(result.air_total_kg_s.sum(),.02,places=12)
        self.assertAlmostEqual(result.balance.flows['S1'],.02,places=12)

    def test_legacy_manual_source_is_never_replaced_by_template(self):
        cfg, ns, raw = fixture()
        raw['sources'][0].pop('template_mode'); raw['sources'][0].update(method='fixed',flow_kg_s=.77)
        result = resolve_sources(raw, cfg, ns)[0]
        self.assertEqual(result['flow_kg_s'], .77)

    def test_unknown_flow_is_not_copied_from_fuel_and_disabled_nozzle_stays_disabled(self):
        cfg = Settings(); ns = default_nozzles(cfg); ns[0].flow_kg_h = 999.; ns[1].enabled = False
        added = inherited_swirlers(cfg.air_model, cfg, ns, force=True)
        effective = effective_air_model(added, cfg, ns)
        self.assertFalse(effective['enabled']); self.assertIsNone(effective['total_kg_s'])
        self.assertEqual(len(effective['sources']), 140)
        self.assertTrue(all(s['method']=='unknown' and s['flow_kg_s'] is None for s in effective['sources']))
        self.assertFalse(effective['sources'][1]['enabled'])

    def test_missing_required_inherited_value_is_rejected_after_resolution(self):
        cfg, ns, raw = fixture()
        raw['sources'][0].update(method='fixed', flow_kg_s=None, air_overrides=['method','flow_kg_s'])
        validate_air_model(raw)  # unresolved draft is a valid persistence record
        with self.assertRaisesRegex(ValueError, 'не задан'):
            effective_air_model(raw, cfg, ns)

    def test_templates_and_nested_future_metadata_roundtrip_without_changing_fuel_schema(self):
        cfg, ns, raw = fixture(); cfg.air_model = raw
        cfg.nozzle_presets['uncooled']['future'] = {'nested': [42]}
        validate_catalog(cfg.atomizer_types, cfg.nozzle_presets)
        encoded = project_to_json(cfg, ns); loaded, _ = project_from_json(encoded)
        self.assertEqual(loaded.nozzle_presets, cfg.nozzle_presets)
        self.assertEqual(loaded.air_model, cfg.air_model)
        self.assertEqual(encoded['schema_version'], 2)


class TestAirPresetEditor(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.app=QApplication.instance() or QApplication([])

    def test_fuel_noop_apply_retains_air_future_metadata_and_exact_geometry(self):
        from fuelfield.editors import PresetEditor
        from fuelfield.air_table import AirSourceTable
        cfg, _, _ = fixture(); cfg.nozzle_presets['uncooled']['future']={'keep':True}
        cfg.nozzle_presets['uncooled']['h0_mm']=4.123456789012345
        editor=PresetEditor(SimpleNamespace(cfg=cfg),AirSourceTable);editor.load()
        try:
            values=[];editor.changed.connect(values.append);editor.apply()
            self.assertEqual(values[0],cfg.nozzle_presets)
            self.assertEqual(editor.medium_tabs.tabText(0),'Топливо')
            self.assertEqual(editor.medium_tabs.tabText(1),'Воздух')
        finally:editor.deleteLater();self.app.processEvents()

    def test_air_template_apply_does_not_enable_air_or_change_geometry(self):
        from fuelfield.editors import PresetEditor
        from fuelfield.air_table import AirSourceTable
        cfg=Settings();original=deepcopy(cfg.nozzle_presets)
        editor=PresetEditor(SimpleNamespace(cfg=cfg),AirSourceTable);editor.load()
        try:
            received=[];editor.changed.connect(received.append);editor.apply_air()
            self.assertEqual(received[0]['uncooled']['air']['method'],'unknown')
            self.assertIsNone(received[0]['uncooled']['air']['flow_kg_s'])
            for key,value in original['uncooled'].items():self.assertEqual(received[0]['uncooled'][key],value)
            self.assertFalse(cfg.air_model['enabled'])
        finally:editor.deleteLater();self.app.processEvents()


if __name__=='__main__':unittest.main()
