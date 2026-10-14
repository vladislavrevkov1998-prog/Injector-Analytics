"""Review regressions exercised with the user's actual 140-nozzle project."""
from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
import time
import unittest
from unittest.mock import patch
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from fuelfield.core import (project_from_json,project_to_json,geometry_at,
    evaluate_continuous_field,calculate_plane,source_flow_kg_s)
from fuelfield.v02 import capture_report,effective_nozzle,follow_row,curve_for
from fuelfield.gui import MainWindow,INTCOL,FIELDS
from fuelfield.editors import HYDRAULIC_LABELS,HYDRAULIC_RESULTS
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication,QTableWidgetSelectionRange,QFileDialog,QMessageBox,QPushButton
from PyQt5.QtTest import QTest

PROJECT=Path(__file__).resolve().parents[1]/'examples/demo_project_user_field.json'

def fixture():return project_from_json(json.loads(PROJECT.read_text()))


class TestUserPhysics(unittest.TestCase):
    def test_requested_anchor_positions_both_rows_and_reordered_ids(self):
        cfg,ns=fixture();ns[12].enabled=False;before=deepcopy(ns);ns.reverse()
        follow_row(ns,38,0);by_id={n.id:n for n in ns}
        expected={38:0,39:354.857142857143,40:349.714285714286,37:5.142857142857,
                  36:10.285714285714,107:2.571428571429,108:357.428571428571}
        for ident,angle in expected.items():self.assertAlmostEqual(by_id[ident].theta_deg,angle,places=9)
        for n in before:
            self.assertEqual(by_id[n.id].radius_mm,n.radius_mm)
            self.assertEqual(by_id[n.id].sector_ml,n.sector_ml)
            self.assertEqual(by_id[n.id].enabled,n.enabled)
        follow_row(ns,108,0);self.assertAlmostEqual(by_id[38].theta_deg,360/140)
        self.assertAlmostEqual(by_id[107].theta_deg,360/70)

    def test_capture_from_measured_flow_and_curve_not_constant_sqrt(self):
        cfg,ns=fixture();n=ns[0];report=capture_report(n,cfg)
        self.assertAlmostEqual(report.expected_ml,1224.9635695307518,places=6)
        self.assertAlmostEqual(report.deviation_percent,20.65070144311759,places=7)
        self.assertAlmostEqual(report.missing_ml,252.9635695307518,places=6)
        before=n.flow_kg_h;n.sector_ml[0]+=10
        after=capture_report(n,cfg)
        self.assertAlmostEqual(after.missing_ml,report.missing_ml-10)
        self.assertAlmostEqual(after.deviation_percent,report.deviation_percent-1000/report.expected_ml)
        self.assertEqual(n.flow_kg_h,before)
        n.flow_mode='sectors';n.capture_efficiency=None
        self.assertNotAlmostEqual(source_flow_kg_s(n,cfg)*3600,n.flow_kg_h)
        n.sector_ml=[report.expected_ml/36]*36
        full=capture_report(n,cfg);self.assertAlmostEqual(full.deviation_percent,0,places=10)
        self.assertAlmostEqual(full.missing_ml,0,places=9)
        n.sector_ml[0]+=10
        self.assertLess(capture_report(n,cfg).deviation_percent,0)

    def test_local_uniform_field_is_smooth_and_rotationally_symmetric(self):
        cfg,ns=fixture();n=ns[0];eff=effective_nozzle(n,cfg);rc,h=geometry_at(n,0,cfg)
        x0=n.radius_mm*np.cos(np.deg2rad(n.theta_deg));y0=n.radius_mm*np.sin(np.deg2rad(n.theta_deg))
        span=12.;field=evaluate_continuous_field([n],cfg,0,resolution=400,
            bounds_xy=(x0-span,x0+span,y0-span,y0+span),mask_track=False)
        self.assertLess(field.x_mm[1]-field.x_mm[0],.061)
        angles=np.linspace(0,2*np.pi,361)
        ix=np.rint((x0+rc*np.cos(angles)-field.x_mm[0])/(field.x_mm[1]-field.x_mm[0])).astype(int)
        iy=np.rint((y0+rc*np.sin(angles)-field.y_mm[0])/(field.y_mm[1]-field.y_mm[0])).astype(int)
        values=field.flux_kg_m2_s[iy,ix]
        self.assertLess(np.ptp(values)/np.mean(values),.002)
        step=(field.x_mm[1]-field.x_mm[0])*1e-3
        self.assertAlmostEqual(field.flux_kg_m2_s.sum()*step**2/source_flow_kg_s(n,cfg),1,delta=.0001)

    def test_user_project_all_four_balances_and_common_bench_pressures(self):
        cfg,ns=fixture();original=[n.flow_kg_h for n in ns]
        for z in cfg.z_mm:self.assertLess(abs(calculate_plane(ns,cfg,z).rel_balance_error),1e-12)
        cfg.flow_pressure_kgf_cm2=40;cfg.sector_pressure_kgf_cm2=2
        for n in ns[:3]:
            curve=curve_for(n,cfg)
            expected=n.flow_kg_h*curve.evaluate(30).flow_kg_h/curve.evaluate(40).flow_kg_h
            self.assertAlmostEqual(source_flow_kg_s(n,cfg)*3600,expected)
            capture_report(n,cfg)
        self.assertEqual([n.flow_kg_h for n in ns],original)


class TestUserQt(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.app=QApplication.instance() or QApplication([])
    def setUp(self):
        self.w=MainWindow();self.w.preview_timer.stop();self.w.selection_timer.stop()
        self.w._restoring=True;self.w.cfg,self.w.nozzles=fixture()
        self.w._fill_controls();self.w._fill_nozzles();self.w._reset_history()
        self.w.preview_timer.stop();self.w.selection_timer.stop()
    def tearDown(self):
        self.w.preview_timer.stop();self.w.selection_timer.stop();self.w.close();self.w.deleteLater();self.app.processEvents()

    def test_columns_order_removed_bench_fields_and_header_menu_units(self):
        keys=[key for key,label in FIELDS]
        self.assertEqual(keys[:5],['id','enabled','nozzle_kind','atomizer_type_id','row'])
        self.assertEqual(keys[keys.index('flow_mode')+1],'effective_flow')
        for key in ('flow_pressure_kgf_cm2','sector_pressure_kgf_cm2','capture_efficiency','flow_status'):
            self.assertNotIn(key,keys)
        t=self.w.nozzle_table;self.assertTrue(t.horizontalHeaderItem(INTCOL['flow_kg_h']).text().startswith('Gт'))
        data=deepcopy(self.w.nozzles)
        plane=calculate_plane(self.w.nozzles,self.w.cfg,0)
        field=evaluate_continuous_field(self.w.nozzles,self.w.cfg,0)
        self.w._done([(plane,field)]);before=self.w.results
        self.w._toggle_flow_unit(INTCOL['flow_kg_h'])
        self.assertEqual(self.w.cfg.display_flow_unit,'l/min')
        self.w.flow_unit_menu.actions()[0].trigger()
        self.assertEqual(self.w.cfg.display_flow_unit,'kg/h');self.assertIs(self.w.results,before)
        self.assertEqual(self.w.nozzles,data)
        self.assertAlmostEqual(float(t.item(0,INTCOL['flow_kg_h']).text()),65.52)
        self.w.undo_stack.undo();self.assertEqual(self.w.cfg.display_flow_unit,'l/min')

    def test_manual_write_locks_freezes_and_reverts_to_current_preset(self):
        t=self.w.nozzle_table;manual=t.item(0,INTCOL['manual_write'])
        self.assertEqual(manual.checkState(),Qt.Unchecked)
        self.assertFalse(t.item(0,INTCOL['alpha_deg']).flags()&Qt.ItemIsEditable)
        manual.setCheckState(Qt.Checked)
        self.assertTrue(t.item(0,INTCOL['alpha_deg']).flags()&Qt.ItemIsEditable)
        t.item(0,INTCOL['alpha_deg']).setText('65')
        self.assertEqual(effective_nozzle(self.w.nozzles[0],self.w.cfg).alpha_deg,65)
        t.item(0,INTCOL['manual_write']).setCheckState(Qt.Unchecked)
        self.assertEqual(effective_nozzle(self.w.nozzles[0],self.w.cfg).alpha_deg,60)
        presets=deepcopy(self.w.cfg.nozzle_presets);presets['cooled']['alpha_deg']=70
        self.w._presets_changed(presets)
        t.item(0,INTCOL['manual_write']).setCheckState(Qt.Checked)
        self.assertEqual(effective_nozzle(self.w.nozzles[0],self.w.cfg).alpha_deg,70)
        t.item(0,INTCOL['manual_write']).setCheckState(Qt.Unchecked)
        self.w.undo_stack.undo();self.assertEqual(t.item(0,INTCOL['manual_write']).checkState(),Qt.Checked)

    def test_direct_cell_types_use_editable_lists_and_presets(self):
        t=self.w.nozzle_table;index=t.model().index(0,INTCOL['nozzle_kind'])
        delegate=t.itemDelegateForColumn(INTCOL['nozzle_kind']);editor=delegate.createEditor(t,None,index)
        delegate.setEditorData(editor,index);editor.setCurrentIndex(0);delegate.setModelData(editor,t.model(),index)
        self.assertEqual(self.w.nozzles[0].nozzle_kind,'uncooled')
        self.assertEqual(effective_nozzle(self.w.nozzles[0],self.w.cfg).alpha_deg,80)
        t.item(0,INTCOL['atomizer_type_id']).setText('atomizer-1')
        self.assertEqual(self.w.nozzles[0].atomizer_type_id,'atomizer-1')
        editor.deleteLater()
        buttons=[b.text() for b in self.w.findChildren(QPushButton)]
        self.assertFalse(any('Назначить' in text for text in buttons))

    def test_settings_open_once_and_common_pressures_roundtrip(self):
        count=self.w.tabs.count();self.w.open_settings();self.w.open_settings()
        self.assertEqual(self.w.tabs.count(),count)
        self.assertTrue(self.w.settings_page.isVisible())
        self.w.mode_widgets['flow_pressure_kgf_cm2'].setValue(40)
        self.w.mode_widgets['sector_pressure_kgf_cm2'].setValue(2)
        self.w._record_edit();self.assertEqual(self.w.cfg.flow_pressure_kgf_cm2,40)
        self.assertEqual({n.flow_pressure_kgf_cm2 for n in self.w.nozzles},{40})
        self.assertEqual({n.sector_pressure_kgf_cm2 for n in self.w.nozzles},{2})
        data=project_to_json(self.w.cfg,self.w.nozzles);cfg,ns=project_from_json(json.loads(json.dumps(data)))
        self.assertEqual(project_to_json(cfg,ns),data)
        self.w.undo_stack.undo();self.assertIsNone(self.w.cfg.flow_pressure_kgf_cm2)

    def test_capture_and_status_update_after_manual_sector_volume(self):
        before=capture_report(self.w.nozzles[0],self.w.cfg);measured=self.w.nozzles[0].flow_kg_h
        self.w.sector_table.item(0,1).setText('37')
        after=capture_report(self.w.nozzles[0],self.w.cfg)
        self.assertAlmostEqual(after.missing_ml,before.missing_ml-10)
        self.assertIn(f'{after.missing_ml:+.3f}',self.w.capture_label.text())
        self.assertEqual(self.w.nozzles[0].flow_kg_h,measured)
        self.assertIn('Δp=30',self.w.nozzle_status.text())
        self.w.undo_stack.undo();self.assertEqual(self.w.nozzles[0].sector_ml[0],27)

    def test_auto_follow_anchor_38_undo_and_disabled_stays_in_place(self):
        self.w.mode_widgets['placement_mode'].setCurrentIndex(1)
        old=[n.theta_deg for n in self.w.nozzles]
        self.w.nozzle_table.item(37,INTCOL['theta_deg']).setText('0')
        self.assertAlmostEqual(self.w.nozzles[106].theta_deg,360/140)
        self.assertAlmostEqual(self.w.nozzles[107].theta_deg,360-360/140)
        positioned=[n.theta_deg for n in self.w.nozzles]
        self.w.nozzle_table.item(12,INTCOL['enabled']).setCheckState(Qt.Unchecked)
        self.assertEqual([n.theta_deg for n in self.w.nozzles],positioned)
        self.w.undo_stack.undo();self.w.undo_stack.undo()
        self.assertEqual([n.theta_deg for n in self.w.nozzles],old)

    def test_selection_burst_defers_preview_and_same_source_reuses_it(self):
        from fuelfield import gui
        with patch.object(gui,'evaluate_continuous_field',wraps=evaluate_continuous_field) as evaluate:
            start=time.monotonic()
            for row in range(30):
                self.w.nozzle_table.setCurrentCell(row,INTCOL['flow_kg_h'])
                self.w.nozzle_table.setRangeSelected(QTableWidgetSelectionRange(0,5,row,12),True)
                self.app.processEvents()
            elapsed=time.monotonic()-start
            self.assertEqual(evaluate.call_count,0)
            QTest.qWait(350);self.app.processEvents()
            self.assertEqual(evaluate.call_count,1)
            self.w._draw_single_preview();self.assertEqual(evaluate.call_count,1)
            self.assertLess(elapsed,1.5)
            bounds=evaluate.call_args.kwargs['bounds_xy'];self.assertLess(bounds[1]-bounds[0],40)

    def test_dityakin_results_are_below_inputs_and_readonly(self):
        e=self.w.catalog_editor;start=len(HYDRAULIC_LABELS)
        self.assertEqual(e.theory_table.rowCount(),start+len(HYDRAULIC_RESULTS))
        for row,key in enumerate(HYDRAULIC_RESULTS,start=start):
            self.assertFalse(e.theory_table.item(row,1).flags()&Qt.ItemIsEditable)
            self.assertNotEqual(e.theory_table.item(row,1).text(),'—')
        row=start+list(HYDRAULIC_RESULTS).index('mass_kg_h')
        before=float(e.theory_table.item(row,1).text());e.diagnostic_pressure.setValue(5)
        after=float(e.theory_table.item(row,1).text());self.assertNotEqual(before,after)
        self.assertAlmostEqual(after,e.curve.evaluate(5).flow_kg_h,places=6)

    def test_user_project_save_reopen_real_worker_and_balance(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'user.json';self.w.project_file=str(path);self.w.save_project()
            with patch.object(QFileDialog,'getOpenFileName',return_value=(str(path),'')):self.w.open_project()
        errors=[]
        with patch.object(self.w,'_error',side_effect=errors.append):
            self.w.run_calc();deadline=time.monotonic()+20
            while self.w.thread is not None and time.monotonic()<deadline:
                self.app.processEvents();QTest.qWait(10)
        self.assertIsNone(self.w.thread);self.assertEqual(errors,[]);self.assertEqual(len(self.w.results),4)
        for plane in self.w.results:self.assertLess(abs(plane.rel_balance_error),1e-12)
        self.assertEqual(self.w.nozzles[0].flow_kg_h,65.52)
        self.assertEqual(self.w.nozzles[0].sector_ml,[27.]*36)

if __name__=='__main__':unittest.main()
