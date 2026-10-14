"""v0.2 integration, legacy golden fields, import and real Qt workflows."""
from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
import time
import unittest
import zipfile
from unittest.mock import patch
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from fuelfield.core import (Settings,default_nozzles,calculate_plane,geometry_at,
    source_flow_kg_s,project_to_json,project_from_json,evaluate_continuous_field)
from fuelfield.v02 import (flow_status,follow_row,effective_nozzle,remove_atomizer,
    GEOMETRY_KEYS,validate_nozzles)
from fuelfield.data_io import import_characteristic_file,write_characteristic_csv
from fuelfield.characteristics import ExperimentalCharacteristic,ExtrapolationWarning

CATALOG=[dict(id='test',name='Измеренная кривая',kind='experimental',points=[[2.5,2.],[30.,24.]])]


def fixture():
    cfg=Settings(n_sample_radial=8,n_sample_angular=72,preview_resolution=200,
                 atomizer_types=deepcopy(CATALOG),pressure_mode='calibrated')
    ns=default_nozzles(cfg)
    for n in ns:n.atomizer_type_id='test'
    return cfg,ns


class TestV02Physics(unittest.TestCase):
    def test_legacy_golden_fields_all_four_sections(self):
        root=Path(__file__).resolve().parents[1]
        cfg,ns=project_from_json(json.loads((root/'examples/demo_project_140.json').read_text()))
        self.assertEqual(cfg.pressure_mode,'legacy')
        with np.load(root/'tests/fixtures/legacy_v012_demo.npz') as golden:
            for i,z in enumerate(cfg.z_mm):
                p=calculate_plane(ns,cfg,z)
                np.testing.assert_allclose(p.cell_mass_kg_s,golden[f'mass_{i}'],rtol=1e-12,atol=1e-15)
                np.testing.assert_allclose([p.injected_kg_s,p.inside_kg_s,p.inner_kg_s,p.outer_kg_s],
                                          golden[f'balance_{i}'],rtol=1e-12,atol=1e-15)

    def test_sector_pressure_conversion_capture_and_independent_manual_measurement(self):
        cfg,ns=fixture();n=ns[0];n.flow_mode='sectors';n.flow_kg_h=1234;n.sector_ml=[10.]*36
        self.assertAlmostEqual(source_flow_kg_s(n,cfg),.0192)
        self.assertTrue(flow_status(n,cfg).estimated)
        self.assertIn('Полнота сбора',flow_status(n,cfg).notices[0])
        n.capture_efficiency=.5
        self.assertAlmostEqual(source_flow_kg_s(n,cfg),.0384)
        self.assertEqual(n.flow_kg_h,1234)
        n.sector_ml[0]=20
        self.assertAlmostEqual(source_flow_kg_s(n,cfg),370e-6*800/180/.5*12)
        self.assertEqual(n.flow_kg_h,1234)

    def test_manual_pressure_conversion_does_not_overwrite_measurement(self):
        cfg,ns=fixture();n=ns[0];n.flow_kg_h=120;cfg.target_pressure_kgf_cm2=2.5
        self.assertAlmostEqual(source_flow_kg_s(n,cfg)*3600,10)
        self.assertEqual(n.flow_kg_h,120)
        self.assertEqual(n.flow_pressure_kgf_cm2,30)
        self.assertTrue(flow_status(n,cfg).estimated)
        cfg.target_pressure_kgf_cm2=30
        self.assertEqual(source_flow_kg_s(n,cfg)*3600,120)
        self.assertFalse(flow_status(n,cfg).estimated)

    def test_curve_source_and_zero_reference_rejected(self):
        cfg,ns=fixture();n=ns[0];n.flow_mode='curve';n.flow_kg_h=999
        self.assertEqual(source_flow_kg_s(n,cfg)*3600,24)
        self.assertEqual(n.flow_kg_h,999)
        n.flow_mode='sectors';cfg.atomizer_types[0]['points'][0][1]=0
        with self.assertRaises(ValueError):source_flow_kg_s(n,cfg)

    def test_missing_curve_and_unconfirmed_extrapolation_are_not_silent(self):
        cfg,ns=fixture();n=ns[0];n.flow_kg_h=10;cfg.target_pressure_kgf_cm2=60
        with self.assertRaises(ValueError):source_flow_kg_s(n,cfg)
        cfg.allow_extrapolation=True
        with self.assertWarns(ExtrapolationWarning):status=flow_status(n,cfg)
        self.assertIn('Экстраполяция характеристики',status.notices)
        n.atomizer_type_id=''
        with self.assertRaises(ValueError):source_flow_kg_s(n,cfg)

    def test_mass_balance_new_modes_four_sections(self):
        cfg,ns=fixture()
        ns[0].flow_mode='sectors';ns[0].sector_ml=[10.]*36;ns[0].capture_efficiency=.75
        ns[70].flow_mode='curve';ns[1].flow_kg_h=10
        expected=sum(source_flow_kg_s(n,cfg) for n in ns if n.enabled)
        for z in cfg.z_mm:
            plane=calculate_plane(ns,cfg,z)
            self.assertAlmostEqual(plane.injected_kg_s,expected)
            self.assertLess(abs(plane.rel_balance_error),1e-12)

    def test_preset_inheritance_override_and_no_sector_mutation(self):
        cfg,ns=fixture();n=ns[0];before=deepcopy(n.sector_ml)
        n.nozzle_kind='cooled';n.geometry_overrides=[]
        cfg.nozzle_presets['cooled']['r_nozzle_mm']=20
        cfg.nozzle_presets['cooled']['h0_mm']=6
        cfg.nozzle_presets['cooled']['alpha_deg']=60
        eff=effective_nozzle(n,cfg)
        self.assertEqual((eff.rc0_mm,eff.h0_mm,eff.alpha_deg),(17,6,60))
        n.alpha_deg=50;n.geometry_overrides=['alpha_deg']
        cfg.nozzle_presets['cooled']['alpha_deg']=70
        self.assertEqual(effective_nozzle(n,cfg).alpha_deg,50)
        n.geometry_overrides=[]
        self.assertEqual(effective_nozzle(n,cfg).alpha_deg,70)
        self.assertEqual(n.sector_ml,before)

    def test_radius_migration_roundtrip_preserves_geometry_and_links(self):
        cfg,ns=fixture();ns[0].nozzle_kind='cooled';ns[0].geometry_overrides=[]
        data=project_to_json(cfg,ns)
        self.assertEqual(data['nozzles'][0]['r_nozzle_mm'],ns[0].rc0_mm+ns[0].h0_mm/2)
        restored,new_ns=project_from_json(json.loads(json.dumps(data)))
        self.assertEqual(project_to_json(restored,new_ns),data)
        for z in cfg.z_mm:self.assertEqual(geometry_at(ns[0],z,cfg),geometry_at(new_ns[0],z,restored))
        broken=deepcopy(data);broken['nozzles'][0]['r_nozzle_mm']+=1
        with self.assertRaises(ValueError):project_from_json(broken)
        no_legacy=deepcopy(data);del no_legacy['nozzles'][0]['rc0_mm']
        self.assertEqual(project_from_json(no_legacy)[1][0].rc0_mm,ns[0].rc0_mm)

    def test_type_deletion_dangling_reference_and_invalid_eta_rejected(self):
        cfg,ns=fixture()
        with self.assertRaises(ValueError):remove_atomizer(cfg.atomizer_types,ns,'test')
        for n in ns:n.atomizer_type_id=''
        self.assertEqual(remove_atomizer(cfg.atomizer_types,ns,'test'),[])
        ns[0].atomizer_type_id='missing'
        with self.assertRaises(ValueError):validate_nozzles(ns,cfg)
        ns[0].atomizer_type_id=''
        for eta in (0,-1,1.1,float('nan')):
            ns[0].capture_efficiency=eta
            with self.assertRaises(ValueError):validate_nozzles(ns,cfg)

    def test_invalid_runtime_settings_and_coordinates_are_rejected(self):
        cfg,ns=fixture()
        for key,value in (('allow_extrapolation','false'),('duration_s',float('nan')),
                          ('density_kg_m3',float('inf')),('z_mm',[0,20,float('nan'),60])):
            invalid=deepcopy(cfg);setattr(invalid,key,value)
            with self.subTest(key=key),self.assertRaises(ValueError):invalid.validate()
        ns[0].radius_mm=float('nan')
        with self.assertRaises(ValueError):validate_nozzles(ns,cfg)

    def test_auto_follow_keeps_both_clockwise_rows_and_disabled_positions(self):
        cfg,ns=fixture();before=deepcopy(ns);ns[10].enabled=False
        follow_row(ns,5,ns[4].theta_deg+12)
        for old,new in zip(before,ns):
            self.assertAlmostEqual(new.theta_deg,(old.theta_deg+12)%360,places=12)
            self.assertEqual(new.radius_mm,old.radius_mm)
        self.assertFalse(ns[10].enabled)

    def test_project_position_ids_cannot_be_moved_to_the_other_row(self):
        cfg,ns=fixture();ns[0].row='inner'
        with self.assertRaises(ValueError):project_from_json(project_to_json(cfg,ns))


class TestCharacteristicImports(unittest.TestCase):
    def test_csv_decimal_commas_units_and_roundtrip(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'curve.csv';p.write_text('Δp;Q\n30;1,5\n2,5;0,125\n',encoding='utf-8-sig')
            curve,sheet=import_characteristic_file(p,'l/min',800)
            self.assertEqual(sheet,'CSV')
            self.assertEqual(curve.points,((2.5,6.),(30.,72.)))
            write_characteristic_csv(p,curve)
            self.assertEqual(import_characteristic_file(p)[0],curve)

    def test_import_rejects_invalid_rows_duplicates_and_unknown_header(self):
        for raw in ('Δp;Q\n2.5;2\n30;bad\n','Δp;Q\n2.5;2\n2.5;3\n',
                    'bad;bad\n2.5;2\n','Δp;Q\n2.5;NaN\n'):
            with self.subTest(raw=raw),tempfile.TemporaryDirectory() as d:
                p=Path(d)/'curve.csv';p.write_text(raw)
                with self.assertRaises(ValueError):import_characteristic_file(p)

    def test_import_rejects_pressure_and_flow_unit_mismatches(self):
        for raw,unit in (('pressure_Pa;flow_kg_h\n100000;2','kg/h'),
                         ('pressure_kgf_cm2;flow_kg_h\n2.5;2','l/min')):
            with self.subTest(unit=unit),tempfile.TemporaryDirectory() as d:
                p=Path(d)/'units.csv';p.write_text(raw)
                with self.assertRaises(ValueError):import_characteristic_file(p,unit)

    def test_xlsx_multiple_sheets_explicit_selection(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'curve.xlsx'
            with zipfile.ZipFile(p,'w') as z:
                z.writestr('xl/workbook.xml','<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"><sheets><sheet name="A" sheetId="1" r:id="r1"/><sheet name="B" sheetId="2" r:id="r2"/></sheets></workbook>')
                z.writestr('xl/_rels/workbook.xml.rels','<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="r1" Target="worksheets/sheet1.xml"/><Relationship Id="r2" Target="worksheets/sheet2.xml"/></Relationships>')
                for i,q in ((1,2),(2,3)):
                    z.writestr(f'xl/worksheets/sheet{i}.xml',f'<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"><sheetData><row r="1"><c r="A1"><v>2.5</v></c><c r="B1"><v>{q}</v></c></row></sheetData></worksheet>')
            with self.assertRaises(ValueError):import_characteristic_file(p)
            curve,name=import_characteristic_file(p,sheet='B')
            self.assertEqual((curve.points,name),(((2.5,3.),),'B'))

    def test_broken_xlsx_is_a_readable_import_error(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'broken.xlsx';p.write_text('not a zip archive')
            with self.assertRaisesRegex(ValueError,'XLSX'):import_characteristic_file(p)


try:
    from PyQt5.QtCore import Qt
    from PyQt5.QtWidgets import QApplication,QTableWidgetItem,QMessageBox,QTableWidgetSelectionRange,QFileDialog
    HAS_QT=True
except ImportError:HAS_QT=False
if HAS_QT:
    from fuelfield.gui import MainWindow,INTCOL


@unittest.skipUnless(HAS_QT,'PyQt5 required')
class TestV02QtWorkflows(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.app=QApplication.instance() or QApplication([])
    def setUp(self):
        self.w=MainWindow();self.w.preview_timer.stop()
    def tearDown(self):
        self.w.preview_timer.stop();self.w.close();self.w.deleteLater();self.app.processEvents()
    def configure(self):
        self.w._restoring=True;self.w.cfg,self.w.nozzles=fixture()
        self.w._fill_controls();self.w._fill_nozzles();self.w._reset_history()
    def select(self,row,key):
        table=self.w.nozzle_table;table.clearSelection();table.setCurrentCell(row,INTCOL[key])
        table.setRangeSelected(QTableWidgetSelectionRange(row,INTCOL[key],row,INTCOL[key]),True)

    def test_catalog_create_assign_edit_undo_save_reopen_and_protect_deletion(self):
        editor=self.w.catalog_editor;editor.new_type();editor.name.setText('Проверка')
        editor.table.setRowCount(2)
        for row,(p,q) in enumerate(((2.5,2),(30,24))):
            editor.table.setItem(row,0,QTableWidgetItem(str(p)));editor.table.setItem(row,1,QTableWidgetItem(str(q)))
        editor.apply()
        self.assertEqual(len(self.w.cfg.atomizer_types),1)
        ident=self.w.cfg.atomizer_types[0]['id']
        self.select(0,'flow_kg_h');self.w.assign_type_selector.setCurrentIndex(1);self.w.assign_atomizer()
        self.assertEqual(self.w.nozzles[0].atomizer_type_id,ident)
        self.w.undo_stack.undo();self.assertEqual(self.w.nozzles[0].atomizer_type_id,'')
        self.w.undo_stack.redo();self.assertEqual(self.w.nozzles[0].atomizer_type_id,ident)
        with patch.object(QMessageBox,'warning') as warning:editor.delete_type()
        self.assertEqual(warning.call_count,1);self.assertEqual(len(self.w.cfg.atomizer_types),1)
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'project.json';self.w.project_file=str(path);self.w.save_project()
            self.assertEqual(json.loads(path.read_text())['settings']['atomizer_types'][0]['id'],ident)
            with patch.object(QFileDialog,'getOpenFileName',return_value=(str(path),'')):self.w.open_project()
            self.assertEqual(self.w.nozzles[0].atomizer_type_id,ident)
            self.assertEqual(self.w.undo_stack.count(),0)

    def test_catalog_invalid_edit_and_import_leave_project_unchanged(self):
        self.configure();editor=self.w.catalog_editor;before=deepcopy(self.w.cfg.atomizer_types)
        editor.table.item(1,0).setText('2.5')
        with patch.object(QMessageBox,'warning') as warning:editor.apply()
        self.assertEqual(warning.call_count,1)
        self.assertEqual(self.w.cfg.atomizer_types,before)
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'bad.csv';p.write_text('Δp;Q\n2.5;2\n30;bad')
            pending=editor.table.item(1,0).text()
            with patch.object(QFileDialog,'getOpenFileName',return_value=(str(p),'')),patch.object(QMessageBox,'warning'):
                editor.import_file()
            self.assertEqual(editor.table.item(1,0).text(),pending)

    def test_curve_point_deletion_and_unit_toggles_keep_canonical_points(self):
        self.configure();editor=self.w.catalog_editor;original=deepcopy(self.w.cfg.atomizer_types)
        for i in range(10):editor.unit.setCurrentIndex(i%2)
        self.assertEqual([list(p) for p in editor.read_curve().points],original[0]['points'])
        editor.apply();self.assertEqual(self.w.cfg.atomizer_types,original)
        editor.table.setCurrentCell(0,0);editor.delete_points()
        self.assertEqual(editor.table.rowCount(),1)
        editor.apply();self.assertEqual(len(self.w.cfg.atomizer_types[0]['points']),1)
        self.w.undo_stack.undo();self.assertEqual(self.w.cfg.atomizer_types,original)

    def test_density_change_refreshes_volumetric_curve_without_changing_mass_points(self):
        self.configure();editor=self.w.catalog_editor;editor.unit.setCurrentIndex(1)
        before=deepcopy(self.w.cfg.atomizer_types)
        self.w.density_spin.setValue(1000);self.w._record_edit()
        self.assertEqual(self.w.cfg.atomizer_types,before)
        self.assertEqual(editor.read_curve().points,((2.5,2.),(30.,24.)))
        self.assertAlmostEqual(float(editor.table.item(1,1).text()),.4)

    def test_flow_unit_switches_keep_results_and_edit_converts_mass(self):
        self.configure();n=self.w.nozzles[0];n.flow_kg_h=48
        self.w._fill_nozzles();self.w._reset_history()
        pairs=[(calculate_plane(self.w.nozzles,self.w.cfg,z),evaluate_continuous_field(self.w.nozzles,self.w.cfg,z)) for z in self.w.cfg.z_mm]
        self.w._done(pairs);results=self.w.results
        self.w._toggle_flow_unit(INTCOL['flow_kg_h'])
        self.assertEqual(self.w.cfg.display_flow_unit,'kg/h')
        self.w.flow_unit_menu.actions()[1].trigger()
        self.assertEqual(self.w.cfg.display_flow_unit,'l/min')
        self.assertEqual(self.w.nozzle_table.item(0,INTCOL['flow_kg_h']).text(),'1')
        self.assertIs(self.w.results,results)
        self.w.nozzle_table.item(0,INTCOL['flow_kg_h']).setText('2')
        self.assertEqual(self.w.nozzles[0].flow_kg_h,96)
        self.w.undo_stack.undo();self.assertEqual(self.w.nozzles[0].flow_kg_h,48)
        self.w.undo_stack.undo();self.assertEqual(self.w.cfg.display_flow_unit,'kg/h')

    def test_geometry_preset_reset_override_and_change_are_undoable(self):
        self.configure();self.select(0,'alpha_deg');self.w.assign_kind_selector.setCurrentIndex(1)
        self.w.assign_nozzle_kind();self.w.reset_overrides()
        self.assertEqual(self.w.nozzles[0].geometry_overrides,[])
        self.w.preset_editor.selector.setCurrentIndex(1)
        row=GEOMETRY_KEYS.index('alpha_deg');self.w.preset_editor.table.item(row,1).setText('60')
        self.w.preset_editor.apply()
        self.assertEqual(effective_nozzle(self.w.nozzles[0],self.w.cfg).alpha_deg,60)
        self.assertEqual(self.w.nozzles[0].geometry_overrides,[])
        self.w.nozzle_table.item(0,INTCOL['manual_write']).setCheckState(Qt.Checked)
        self.w.nozzle_table.item(0,INTCOL['alpha_deg']).setText('55')
        self.assertIn('alpha_deg',self.w.nozzles[0].geometry_overrides)
        self.assertEqual(effective_nozzle(self.w.nozzles[0],self.w.cfg).alpha_deg,55)
        self.w.undo_stack.undo();self.assertEqual(effective_nozzle(self.w.nozzles[0],self.w.cfg).alpha_deg,60)
        self.w.undo_stack.undo();self.assertEqual(self.w.nozzles[0].geometry_overrides,[])
        self.w.undo_stack.undo();self.assertEqual(effective_nozzle(self.w.nozzles[0],self.w.cfg).alpha_deg,45)

    def test_radius_and_thickness_ui_keep_outer_radius_fixed(self):
        self.configure();self.w.nozzle_table.item(0,INTCOL['r_nozzle_mm']).setText('20')
        self.w.nozzle_table.item(0,INTCOL['h0_mm']).setText('6')
        n=self.w.nozzles[0]
        self.assertEqual((n.r_nozzle_mm,n.rc0_mm,n.h0_mm),(20,17,6))
        self.w.undo_stack.undo();self.assertEqual((self.w.nozzles[0].r_nozzle_mm,self.w.nozzles[0].h0_mm),(20,4))

    def test_auto_follow_edit_and_undo_move_both_rows(self):
        self.configure();self.w.mode_widgets['placement_mode'].setCurrentIndex(1)
        self.assertFalse(self.w.nozzle_table.item(0,INTCOL['row']).flags() & Qt.ItemIsEditable)
        before=deepcopy(self.w.nozzles)
        self.w.nozzle_table.item(4,INTCOL['theta_deg']).setText('40')
        delta=40-before[4].theta_deg
        for old,new in zip(before,self.w.nozzles):
            self.assertAlmostEqual(new.theta_deg,(old.theta_deg+delta)%360)
        self.w.undo_stack.undo()
        self.assertEqual([n.theta_deg for n in self.w.nozzles],[n.theta_deg for n in before])

    def test_reordered_json_uses_ids_for_sector_edit_and_group_copy(self):
        from PyQt5.QtWidgets import QInputDialog
        self.configure();self.w._restoring=True
        self.w.nozzles[0],self.w.nozzles[1]=self.w.nozzles[1],self.w.nozzles[0]
        self.w._fill_nozzles();self.w._reset_history();self.w._show_sectors(1)
        self.w.sector_table.item(0,1).setText('10')
        self.assertEqual(self.w.nozzles[1].id,1)
        self.assertEqual(self.w.nozzles[1].sector_ml[0],10)
        self.assertEqual(self.w.nozzles[0].sector_ml[0],1)
        self.select(0,'flow_kg_h')
        with patch.object(QInputDialog,'getInt',return_value=(1,True)):self.w.copy_sectors()
        self.assertEqual(self.w.nozzles[0].id,2)
        self.assertEqual(self.w.nozzles[0].sector_ml[0],10)
        self.w.sum_sectors_to_flow();self.assertEqual(self.w.nozzles[0].flow_mode,'sectors')

    def test_flow_status_and_sector_changes_preserve_manual_measurement(self):
        self.configure();tab=self.w.nozzle_table
        tab.item(0,INTCOL['flow_kg_h']).setText('100')
        tab.item(0,INTCOL['flow_mode']).setText('sectors')
        self.w.sector_table.item(0,1).setText('10')
        self.assertEqual(self.w.nozzles[0].flow_kg_h,100)
        self.assertIn('η=1',self.w.flow_notice.text())
        self.assertIn('Недостающий объём',self.w.capture_label.text())
        self.assertIn('оценка',self.w.nozzle_status.text())

    def test_real_worker_runs_three_modes_and_presets_after_json_reopen(self):
        from PyQt5.QtTest import QTest
        self.configure();tab=self.w.nozzle_table
        tab.item(0,INTCOL['flow_mode']).setText('sectors')
        self.w.nozzles[0].capture_efficiency=.8  # Legacy explicit correction remains readable.
        sectors=self.w.sector_table;sectors.setCurrentCell(0,1)
        sectors.setRangeSelected(QTableWidgetSelectionRange(0,1,35,1),True);sectors.paste_text('10')
        tab.item(1,INTCOL['flow_kg_h']).setText('5')
        tab.item(70,INTCOL['flow_mode']).setText('curve')
        self.select(0,'nozzle_kind');self.w.assign_kind_selector.setCurrentIndex(1)
        self.w.assign_nozzle_kind();self.w.reset_overrides()
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'v02.json';self.w.project_file=str(path);self.w.save_project()
            with patch.object(QFileDialog,'getOpenFileName',return_value=(str(path),'')):self.w.open_project()
        expected=sum(source_flow_kg_s(n,self.w.cfg) for n in self.w.nozzles)
        errors=[]
        with patch.object(self.w,'_error',side_effect=errors.append):
            self.w.run_calc();deadline=time.monotonic()+10
            while self.w.thread is not None and time.monotonic()<deadline:
                self.app.processEvents();QTest.qWait(10)
        self.assertIsNone(self.w.thread);self.assertEqual(errors,[])
        self.assertEqual(len(self.w.results),4)
        for plane in self.w.results:
            self.assertAlmostEqual(plane.injected_kg_s,expected)
            self.assertLess(abs(plane.rel_balance_error),1e-12)
        self.assertEqual(self.w.nozzles[0].sector_ml,[10.]*36)
        self.assertEqual(self.w.nozzles[0].flow_kg_h,0)


if __name__=='__main__':unittest.main()
