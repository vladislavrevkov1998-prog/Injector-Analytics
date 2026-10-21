"""Explicit physical inputs, migration guards and atomic Qt transport edits."""
from copy import deepcopy
import json
import os
from pathlib import Path
import sys
import time
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
os.environ.setdefault('QT_QPA_PLATFORM','offscreen')

from fuelfield.core import project_from_json,project_to_json
from fuelfield.transport_settings import TransportSettings,synthetic_settings
from test_quasi import case


class TestTransportSettings(unittest.TestCase):
    def test_unknown_values_are_never_filled_with_guesses(self):
        settings=TransportSettings.from_dict({}).validate()
        self.assertFalse(settings.enabled)
        for name in ('carrier_inlet_velocity_m_s','injection_speed_m_s',
                     'relaxation_time_s','film_speed_m_s','cone_half_width_deg'):
            self.assertIsNone(getattr(settings,name))
            raw=synthetic_settings().to_dict();raw[name]=None
            with self.subTest(parameter=name),self.assertRaisesRegex(ValueError,'задайте параметр'):
                TransportSettings.from_dict(raw).validate()
        raw=synthetic_settings().to_dict();raw['turbulence_time_s']=None
        self.assertEqual(TransportSettings.from_dict(raw).validate().turbulence_rms_m_s,0)
        raw['turbulence_rms_m_s']=.1
        with self.assertRaisesRegex(ValueError,'время корреляции'):
            TransportSettings.from_dict(raw).validate()

    def test_decimal_comma_exponent_and_physical_boundaries(self):
        raw=synthetic_settings().to_dict()
        raw.update(relaxation_time_s='5e-3',cone_half_width_deg='0',
                   injection_speed_m_s='40,25',max_step_mm='1e-4',wall_tolerance_mm='1e-6')
        settings=TransportSettings.from_dict(raw).validate()
        self.assertEqual(settings.injection_speed_m_s,40.25)
        self.assertEqual(settings.relaxation_time_s,.005)
        for key,value in [('injection_speed_m_s',0),('relaxation_time_s',-.01),
                          ('cone_half_width_deg',90),('cone_half_width_deg',-.1),
                          ('turbulence_rms_m_s',-.1),('film_speed_m_s',True),
                          ('carrier_inlet_velocity_m_s','nan'),
                          ('injection_speed_m_s','Infinity'),('max_step_mm',1e-5),
                          ('wall_tolerance_mm',1e-7)]:
            with self.subTest(parameter=key,value=value),self.assertRaises(ValueError):
                TransportSettings.from_dict({**raw,key:value}).validate()

    def test_integer_numerics_and_unsupported_modes_are_rejected(self):
        raw=synthetic_settings().to_dict()
        for key,value in [('schema_version',True),('schema_version',1.),
                          ('schema_version',2),('mode',[]),('mode','RANS'),
                          ('wall_model','discard'),('carrier_radial_cells',3),
                          ('carrier_radial_cells',16.),('emission_radial_samples',True),
                          ('emission_angular_samples',145),('seed',-1),('seed',2**32)]:
            with self.subTest(parameter=key,value=value),self.assertRaises(ValueError):
                TransportSettings.from_dict({**raw,key:value}).validate()
        self.assertEqual(TransportSettings.from_dict({**raw,'seed':2**32-1}).validate().seed,2**32-1)

    def test_transport_unknown_fields_survive_without_mutable_aliases(self):
        raw=synthetic_settings().to_dict();raw['custom']={'keep':[1,None]}
        source=deepcopy(raw);settings=TransportSettings.from_dict(raw).validate()
        raw['custom']['keep'].append('external mutation')
        self.assertEqual(settings.to_dict(),source)
        output=settings.to_dict();output['custom']['keep'].append('output mutation')
        self.assertEqual(settings.to_dict(),source)
        self.assertEqual(settings.to_dict()['provenance']['source_kind'],'synthetic')

    def test_project_schema3_roundtrip_and_legacy_schema2_preserve_unknowns(self):
        cfg,nozzles,_=case();cfg.quasistationary=False
        cfg._project_extra={'operator':{'notes':['original']}}
        cfg._extra={'unknown_setting':{'keep':[None,False]}}
        nozzles[0]._extra={'unknown_nozzle':{'keep':[36]}}
        legacy=project_to_json(cfg,nozzles);legacy['settings'].pop('fuel_transport')
        before=deepcopy(legacy);old_cfg,old_nozzles=project_from_json(legacy)
        self.assertEqual(legacy,before)
        self.assertFalse(TransportSettings.from_dict(old_cfg.fuel_transport).enabled)
        self.assertEqual(project_to_json(old_cfg,old_nozzles)['schema_version'],2)
        cfg.fuel_transport=synthetic_settings().to_dict();cfg.fuel_transport['custom']={'keep':['transport']}
        current=json.loads(json.dumps(project_to_json(cfg,nozzles)))
        self.assertEqual(current['schema_version'],3);self.assertEqual(current['version'],'0.7.0')
        restored_cfg,restored_nozzles=project_from_json(json.loads(json.dumps(current)))
        self.assertEqual(project_to_json(restored_cfg,restored_nozzles),current)
        restored_cfg.fuel_transport['mode']='legacy'
        downgraded=project_to_json(restored_cfg,restored_nozzles)
        self.assertEqual(downgraded['schema_version'],2)
        self.assertEqual(downgraded['settings']['fuel_transport']['custom'],{'keep':['transport']})
        restored_cfg.fuel_transport['mode']='confined_inertial'
        self.assertEqual(project_to_json(restored_cfg,restored_nozzles),current)

    def test_schema_does_not_silently_enable_or_downgrade_the_new_model(self):
        cfg,nozzles,_=case();cfg.quasistationary=False
        legacy=project_to_json(cfg,nozzles)
        with self.assertRaisesRegex(ValueError,'явно заданную'):
            project_from_json({**legacy,'schema_version':3})
        cfg.fuel_transport=synthetic_settings().to_dict();active=project_to_json(cfg,nozzles)
        for schema in (1,2):
            with self.subTest(schema=schema),self.assertRaisesRegex(ValueError,'требует схему проекта 3'):
                project_from_json({**active,'schema_version':schema})
        with self.assertRaisesRegex(ValueError,'версия схемы'):
            project_from_json({**active,'schema_version':4})
        active['settings']['fuel_transport']['relaxation_time_s']=None
        with self.assertRaisesRegex(ValueError,'задайте параметр'):
            project_from_json(active)


class TestTransportSettingsUI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from PyQt5.QtWidgets import QApplication
        cls.app=QApplication.instance() or QApplication([])

    def setUp(self):
        from fuelfield.gui import MainWindow
        self.qt_errors=[];self.old_hook=sys.excepthook
        sys.excepthook=lambda kind,value,tb:self.qt_errors.append((kind,value))
        self.w=MainWindow();cfg,nozzles,_=case();cfg.quasistationary=False
        cfg.fuel_transport['custom']={'keep':['metadata']}
        self.w._restoring=True;self.w.cfg,self.w.nozzles=cfg,nozzles
        self.w._fill_controls();self.w._fill_nozzles();self.w._reset_history()
        self.w.preview_timer.stop();self.w.selection_timer.stop();self.w.show()
        self.app.processEvents();self.assertEqual(self.qt_errors,[])

    def tearDown(self):
        from PyQt5.QtTest import QTest
        try:
            for owner in (self.w,self.w.comparison):
                deadline=time.monotonic()+30
                while owner.thread is not None and time.monotonic()<deadline:QTest.qWait(5)
                self.assertIsNone(owner.thread)
            self.w.close();self.w.deleteLater();self.app.processEvents()
            self.assertEqual(self.qt_errors,[],'Unhandled Qt callback exception')
        finally:sys.excepthook=self.old_hook

    def test_menu_and_tab02_settings_open_the_same_dialog(self):
        from PyQt5.QtCore import Qt
        from PyQt5.QtTest import QTest
        self.w.transport_settings_action.trigger();self.app.processEvents()
        dialog=self.w.transport_dialog;self.assertTrue(dialog.isVisible())
        self.assertFalse(dialog.groups[0].isEnabled())
        self.w.open_settings();QTest.mouseClick(self.w.transport_settings_button,Qt.LeftButton)
        self.assertIs(self.w.transport_dialog,dialog)
        self.assertEqual(self.w.undo_stack.count(),0)

    def test_missing_parameters_block_apply_and_close_leaves_project_unchanged(self):
        self.w.open_transport_settings();dialog=self.w.transport_dialog
        before=project_to_json(self.w.cfg,self.w.nozzles)
        dialog.mode.setCurrentIndex(dialog.mode.findData('confined_inertial'))
        self.assertTrue(dialog.groups[0].isEnabled())
        self.assertEqual(dialog.fields['injection_speed_m_s'].text(),'')
        self.assertFalse(dialog.apply());self.assertIn('задайте параметр',dialog.status.text())
        self.assertEqual(project_to_json(self.w.cfg,self.w.nozzles),before)
        self.assertEqual(self.w.undo_stack.count(),0)
        dialog.use_example();self.assertEqual(project_to_json(self.w.cfg,self.w.nozzles),before)
        dialog.close();self.assertEqual(project_to_json(self.w.cfg,self.w.nozzles),before)

    def test_synthetic_draft_apply_is_one_undo_and_preserves_unrelated_edits(self):
        self.w.open_transport_settings();dialog=self.w.transport_dialog;dialog.use_example()
        # A different project setting changes while the model editor is open.
        self.w.duration_spin.setValue(181.25);self.w.duration_spin.editingFinished.emit()
        before=project_to_json(self.w.cfg,self.w.nozzles);count=self.w.undo_stack.count()
        self.assertIn('синтетические',dialog.provenance.text())
        self.assertTrue(dialog.apply());after=project_to_json(self.w.cfg,self.w.nozzles)
        self.assertEqual(self.w.undo_stack.count(),count+1)
        self.assertEqual(after['schema_version'],3);self.assertEqual(self.w.cfg.duration_s,181.25)
        self.assertEqual(self.w.cfg.fuel_transport['custom'],{'keep':['metadata']})
        self.assertEqual(after['nozzles'],before['nozzles'])
        self.w.undo_stack.undo();self.assertEqual(project_to_json(self.w.cfg,self.w.nozzles),before)
        self.w.undo_stack.redo();self.assertEqual(project_to_json(self.w.cfg,self.w.nozzles),after)
        self.assertTrue(dialog.apply());self.assertEqual(self.w.undo_stack.count(),count+1)
        dialog.mode.setCurrentIndex(dialog.mode.findData('legacy'));self.assertTrue(dialog.apply())
        self.assertEqual(project_to_json(self.w.cfg,self.w.nozzles)['schema_version'],2)
        self.assertEqual(self.w.cfg.fuel_transport['relaxation_time_s'],.005)
        self.w.undo_stack.undo();self.assertEqual(project_to_json(self.w.cfg,self.w.nozzles),after)

    def test_free_numeric_entry_preserves_precision_and_supports_full_uint32_seed(self):
        self.w.open_transport_settings();dialog=self.w.transport_dialog;dialog.use_example()
        dialog.fields['injection_speed_m_s'].setText('40,125')
        dialog.fields['relaxation_time_s'].setText('7.5e-4')
        dialog.fields['seed'].setText(str(2**32-1))
        self.assertTrue(dialog.apply())
        self.assertEqual(self.w.cfg.fuel_transport['seed'],2**32-1)
        self.assertEqual(self.w.cfg.fuel_transport['injection_speed_m_s'],40.125)
        self.assertEqual(self.w.cfg.fuel_transport['relaxation_time_s'],.00075)
        precise=20.123456789012344
        self.w.cfg.fuel_transport['carrier_inlet_velocity_m_s']=precise
        self.w._fill_controls();self.w._reset_history();self.w.open_transport_settings()
        self.assertEqual(self.w.transport_dialog.collect()['carrier_inlet_velocity_m_s'],precise)
        self.assertTrue(self.w.transport_dialog.apply());self.assertEqual(self.w.undo_stack.count(),0)

    def test_project_reload_closes_stale_transport_draft(self):
        self.w.open_transport_settings();dialog=self.w.transport_dialog;dialog.use_example()
        self.w._reset_history()
        self.assertFalse(dialog.isVisible())
        self.assertFalse(TransportSettings.from_dict(self.w.cfg.fuel_transport).enabled)


if __name__=='__main__':unittest.main()
