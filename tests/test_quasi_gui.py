"""Real Qt mode dispatch, persistence, pressure mapping and fast field updates."""
import sys,json,tempfile,time,unittest
from pathlib import Path
from copy import deepcopy
from unittest.mock import patch
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from PyQt5.QtWidgets import QApplication,QTextBrowser
from PyQt5.QtCore import Qt
from PyQt5.QtTest import QTest
from fuelfield.gui import MainWindow,INTCOL
from fuelfield.core import project_to_json,project_from_json
from fuelfield.quasi_io import save_settings,read_pressure_tables
from fuelfield.quasi_gui import PressureImportDialog
from fuelfield.analytics_io import write_workbook
from test_quasi import case,ROOT

class TestQuasiGUI(unittest.TestCase):
 @classmethod
 def setUpClass(cls):cls.app=QApplication.instance() or QApplication([])
 def setUp(self):
  self.w=MainWindow();cfg,ns,self.mode=case();self.w._restoring=True;self.w.cfg,self.w.nozzles=cfg,ns;self.w._fill_controls();self.w._fill_nozzles();self.w._reset_history();self.w.preview_timer.stop();self.w.selection_timer.stop();self.w.show();self.errors=[];self.w._error=self.errors.append
 def wait(self,owner):
  deadline=time.monotonic()+60
  while owner.thread is not None and time.monotonic()<deadline:QTest.qWait(10)
  self.assertIsNone(owner.thread)
 def tearDown(self):
  self.wait(self.w);self.wait(self.w.comparison);self.w.close();self.w.deleteLater();self.app.processEvents()
 def calculate(self):self.w.run_calc();self.wait(self.w);self.assertFalse(self.errors,self.errors);self.assertIsNotNone(self.w.quasi_result)
 def test_checkbox_opens_settings_toggle_preserves_config_and_four_tabs(self):
  self.assertEqual(self.w.tabs.count(),4);before=deepcopy(self.w.cfg.quasi_settings);self.w.open_settings();self.w.quasi_check.setChecked(False);self.assertFalse(self.w.cfg.quasistationary);QTest.mouseClick(self.w.quasi_check,Qt.LeftButton);self.app.processEvents();self.assertTrue(self.w.cfg.quasistationary);self.assertTrue(self.w.quasi_dialog.isVisible());self.assertEqual(self.w.cfg.quasi_settings,before)
  self.w.quasi_check.setChecked(False);self.assertEqual(self.w.cfg.quasi_settings,before);self.w.undo_stack.undo();self.assertTrue(self.w.cfg.quasistationary)
 def test_collector_column_preserves_case_and_undo(self):
  self.w.nozzle_table.item(0,INTCOL['manifold_id']).setText('B');self.assertEqual(self.w.nozzles[0].manifold_id,'B');self.w.undo_stack.undo();self.assertEqual(self.w.nozzles[0].manifold_id,'A')
 def test_portable_mode_bindings_apply_atomically_and_cancel_draft(self):
  with tempfile.TemporaryDirectory() as folder:
   path=save_settings(Path(folder)/'mode',self.mode,{'1':'B'});self.w.open_quasi_settings();d=self.w.quasi_dialog;d.load_path(path);self.assertEqual(self.w.nozzles[0].manifold_id,'A');self.assertTrue(d.bindings_box.isChecked());count=self.w.undo_stack.count();d.apply();self.assertEqual(self.w.nozzles[0].manifold_id,'B');self.assertEqual(self.w.undo_stack.count(),count+1);self.w.undo_stack.undo();self.assertEqual(self.w.nozzles[0].manifold_id,'A');self.w.undo_stack.redo();self.assertEqual(self.w.nozzles[0].manifold_id,'B')
   before=deepcopy(self.w.cfg.quasi_settings);d.table.item(0,3).setText('17');d.close();self.assertEqual(self.w.cfg.quasi_settings,before)
 def test_settings_instruction_and_xlsx_mapping_controls(self):
  self.w.open_quasi_settings();d=self.w.quasi_dialog;d.show_guide();self.assertTrue(d.guide_dialog.isVisible());text=d.guide_dialog.findChild(QTextBrowser).toPlainText();self.assertIn('Температуры',text);self.assertIn('Ползунок',text);d.guide_dialog.close()
  with tempfile.TemporaryDirectory() as folder:
   path=Path(folder)/'p.xlsx';write_workbook(path,[('pressure',[['t_s','A','B']]+[[i*.5,10+i,20-i] for i in range(8)])]);dialog=PressureImportDialog(d.read_mode(),read_pressure_tables(path),d);imported=dialog.result_settings();self.assertEqual(imported.dt_s,.5);self.assertEqual(imported.channels[0].pressure[7],17);dialog.close()
 def test_f5_writes_composite_main_results_and_epures_from_those_cells(self):
  self.calculate();self.assertTrue(all(p.is_composite for p in self.w.results));self.assertEqual(len(self.w.results),2);self.assertIn('Накопленная',self.w.balance_label.text());self.w.plots.setCurrentIndex(3);self.w.plot_count.setValue(1);self.app.processEvents();line=self.w.figcurves.axes[0].lines[0];np.testing.assert_allclose(line.get_ydata()[:-1],self.w.results[0].flux_kg_m2_s[0]);self.assertTrue(np.isnan(self.w.results[0].rel_balance_error))
 def test_slider_updates_field_without_rebuilding_matrix_or_full_canvas(self):
  self.calculate();self.w.plots.setCurrentIndex(1);self.app.processEvents();canvas=self.w.smoothcanvas;canvas.draw();QTest.qWait(40);anim=canvas.quasi_animator;before=anim.draw_count;identity=self.w.results;mass=self.w.results[0].cell_mass_kg_s.copy();cached=self.w.quasi_result.operators[0].preview_weights;self.w.time_slider.setValue(7);QTest.qWait(35);self.app.processEvents()
  self.assertEqual(canvas._quasi_step,7);self.assertIs(self.w.results,identity);np.testing.assert_array_equal(self.w.results[0].cell_mass_kg_s,mass);self.assertIs(self.w.quasi_result.operators[0].preview_weights,cached);self.assertEqual(anim.draw_count,before);self.assertGreater(anim.update_count,0)
  np.testing.assert_allclose(np.ma.asarray(anim.image.get_array()).filled(0),self.w.quasi_result.smooth(0,7).masked_flux_kg_m2_s.filled(0));self.assertIn('Шаг 7/7',self.w.time_label.text())
 def test_blit_pixels_match_full_render_with_marker_overlay_and_clip(self):
  self.w.cfg.color_scale_mode='manual';self.w.cfg.color_out_of_range='clip';self.w.cfg.color_vmin=.01;self.w.cfg.color_vmax=10.;self.w._fill_controls();self.w._reset_history();self.calculate();self.w.plots.setCurrentIndex(1);self.app.processEvents();canvas=self.w.smoothcanvas;canvas.draw();QTest.qWait(30)
  self.w.time_slider.setValue(7);QTest.qWait(30);self.app.processEvents();anim=canvas.quasi_animator;actual=np.asarray(canvas.buffer_rgba()).copy();canvas.mpl_disconnect(anim.cid);anim.image.set_animated(False);canvas.draw();reference=np.asarray(canvas.buffer_rgba()).copy();diff=np.abs(actual.astype(int)-reference.astype(int));self.assertLess(float(diff.mean()),.15);self.assertLess(float(np.percentile(diff,99)),3)
 def test_continuous_slider_movement_displays_intermediate_frames(self):
  self.calculate();self.w.plots.setCurrentIndex(1);self.app.processEvents();canvas=self.w.smoothcanvas;canvas.draw();QTest.qWait(30);before=canvas.quasi_animator.update_count
  for i in range(32):self.w.time_slider.setValue(1+i%7);QTest.qWait(2)
  self.assertGreater(canvas.quasi_animator.update_count,before,'Continuous movement must display frames before release')
  self.w.time_slider.setValue(7);QTest.qWait(40);self.assertEqual(canvas._quasi_step,7)
 def test_mode_project_roundtrip_then_stationary_dispatch_and_undo(self):
  self.calculate();self.w.quasi_check.setChecked(False);self.w.run_calc();self.wait(self.w);self.assertIsNone(self.w.quasi_result);self.assertTrue(all(not p.is_composite for p in self.w.results));self.assertFalse(self.w.time_controls.isVisible());self.w.undo_stack.undo();self.assertTrue(self.w.cfg.quasistationary);self.assertEqual(self.w.cfg.quasi_settings,self.mode.to_dict())
  cfg,ns=project_from_json(json.loads(json.dumps(project_to_json(self.w.cfg,self.w.nozzles))));self.assertTrue(cfg.quasistationary);self.assertEqual(ns[70].manifold_id,'B')
 def test_invalid_configuration_does_not_overwrite_existing_inputs(self):
  self.w.open_quasi_settings();d=self.w.quasi_dialog;before=deepcopy(self.w.cfg.quasi_settings);d.dt.setText('0');d.apply();self.assertEqual(self.w.cfg.quasi_settings,before);self.assertIn('Шаг времени',d.status.text());d.dt.setText('.25');d.table.item(0,3).setText('unfinished');rows=d.table.rowCount();d.add_channel();self.assertEqual(d.table.rowCount(),rows);self.assertEqual(d.table.item(0,3).text(),'unfinished')
 def test_quasi_comparison_project_honors_mode(self):
  self.calculate()
  with tempfile.TemporaryDirectory() as folder:
   path=Path(folder)/'compare.json';path.write_text(json.dumps(project_to_json(self.w.cfg,self.w.nozzles)));self.w.split_button.setChecked(True);self.w.comparison.load_path(path);self.wait(self.w.comparison);self.assertTrue(self.w.comparison.planes[0].is_composite);np.testing.assert_allclose(self.w.comparison.planes[0].cell_mass_kg_s,self.w.results[0].cell_mass_kg_s);self.w.comparison.view.setCurrentIndex(2);self.app.processEvents();self.assertIn('шаг 0',self.w.comparison.figure.axes[0].get_title())

if __name__=='__main__':unittest.main()
