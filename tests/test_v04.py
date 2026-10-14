"""v0.4 geometry, arbitrary sections, graph controls and staged fuel fitting."""
import json,sys,tempfile,time,unittest
from dataclasses import replace
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from fuelfield.core import (Settings,default_nozzles,calculate_plane,evaluate_continuous_field,geometry_at,angular_profile,
 sample_kernel,polar_epure_coordinates,project_from_json,project_to_json)
from fuelfield.v02 import effective_nozzle
from fuelfield.analysis import cell_values
from fuelfield.analytics_io import write_epures_xlsx
from fuelfield.data_io import _xlsx_sheets
from fuelfield.regression import RegressionSnapshot,RegressionTarget,ParameterSpec,fit,active_parameters,_metrics
from fuelfield.regression_io import load_target,write_target,write_fit_xlsx
from fuelfield.gui import MainWindow,INTCOL
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import Qt,QPoint
from PyQt5.QtTest import QTest
from matplotlib.backend_bases import MouseEvent,MouseButton
ROOT=Path(__file__).resolve().parents[1]
def user():return project_from_json(json.loads((ROOT/'examples/demo_project_user_field.json').read_text()))
def synthetic():
 cfg=Settings(z_mm=[10.,25.,45.],geometry_model='constant',n_sample_radial=24,n_sample_angular=360)
 ns=default_nozzles(cfg)
 for n in ns:
  n.enabled=n.id in (1,2,71);n.geometry_overrides=[];n.flow_kg_h=10.;n.sector_ml=[1.+k%9*.3 for k in range(36)]
 return cfg,ns

class TestFuelRegression(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  cls.cfg,cls.ns=synthetic();cls.truth=deepcopy(cls.cfg);cls.truth.nozzle_presets['uncooled']['clock_deg']=12.
  cls.target=RegressionTarget.from_project(cls.truth,cls.ns,source_kind='synthetic');cls.snapshot=RegressionSnapshot.capture(cls.cfg,cls.ns)
  cls.spec=ParameterSpec('clock_deg','uncooled',0,-20,25)
  cls.result=fit(cls.snapshot,cls.target,[cls.spec],budget=100,seed=3)
 def test_recovers_known_parameter_and_holdout(self):
  r=self.result;self.assertEqual(r.status,'converged');self.assertAlmostEqual(r.fitted[0],12,delta=.7)
  self.assertLess(r.metrics['train']['RMSE'],r.metrics['initial_train']['RMSE']*.15)
  self.assertGreater(r.metrics['holdout']['R2'],.999);self.assertLessEqual(r.evaluations,100)
  self.assertEqual(r.diagnostics['rank'],1)
 def test_alpha_and_linear_coefficient_recovery(self):
  for key,value,initial,low,high,model,tolerance in [('alpha_deg',52.3,45,35,65,'constant',.1),('k_linear',.14,.08,.02,.22,'linear',.002)]:
   cfg,ns=synthetic();cfg.geometry_model=model;truth=deepcopy(cfg);truth.nozzle_presets['uncooled'][key]=value
   target=RegressionTarget.from_project(truth,ns,source_kind='synthetic')
   result=fit(RegressionSnapshot.capture(cfg,ns),target,[ParameterSpec(key,'uncooled',initial,low,high)],budget=100)
   self.assertAlmostEqual(result.fitted[0],value,delta=tolerance);self.assertGreater(result.metrics['holdout']['R2'],.9999)
 def test_normalized_target_and_relative_error(self):
  target=RegressionTarget.from_project(self.truth,self.ns,'normalized_track',source_kind='synthetic')
  result=fit(self.snapshot,target,[self.spec],budget=100,criterion='relative',relative_floor=.01)
  self.assertLess(result.cost,result.initial_cost);self.assertEqual(result.target.unit,'1')
 def test_invalid_bounds_and_direction(self):
  with self.assertRaises(ValueError):fit(self.snapshot,self.target,[replace(self.spec,minimum=np.nan)])
  with self.assertRaises(ValueError):replace(self.target,metadata={'source_kind':'synthetic','theta_direction':'cw'}).validate(self.cfg)
 def test_snapshot_unknown_fields(self):
  cfg,ns=user();data=project_to_json(cfg,ns);data['custom']={'tag':'keep'};data['settings']['lab']='A';data['nozzles'][0]['operator_note']='keep'
  snapshot=RegressionSnapshot.capture(*project_from_json(data));self.assertEqual(project_to_json(*snapshot.restore()),data)
 def test_reproducible_and_no_mutation(self):
  result=fit(self.snapshot,self.target,[self.spec],budget=100,seed=3)
  self.assertEqual(result.fitted,self.result.fitted);self.assertEqual(result.cost,self.result.cost)
  self.assertEqual(self.snapshot.sha256,RegressionSnapshot.capture(self.cfg,self.ns).sha256)
 def test_candidate_preserves_experiments_and_balance(self):
  c,n=self.result.candidate();self.assertEqual([v.sector_ml for v in n],[v.sector_ml for v in self.ns]);self.assertEqual([v.flow_kg_h for v in n],[v.flow_kg_h for v in self.ns])
  self.assertEqual(c.atomizer_types,self.cfg.atomizer_types)
  for z in c.z_mm:self.assertLess(abs(calculate_plane(n,c,z).rel_balance_error),1e-13)
 def test_inactive_and_temperature_rejected(self):
  with self.assertRaises(ValueError):replace(self.target,quantity='temperature',unit='C').validate(self.cfg)
  with self.assertRaises(ValueError):replace(self.target,metadata={'source_kind':'temperature'}).validate(self.cfg)
  with self.assertRaises(ValueError):fit(self.snapshot,self.target,[ParameterSpec('k_quadratic','uncooled',0,-1,1)])
  z0=RegressionTarget.from_project(replace(self.cfg,z_mm=[0]),self.ns,source_kind='synthetic')
  self.assertNotIn('diffusion_deg_sqrt_mm',active_parameters(replace(self.cfg,angular_model='diffusion'),self.ns,z0,'uncooled'))
  full=RegressionTarget.from_project(replace(self.cfg,z_mm=[0,10]),self.ns,source_kind='synthetic')
  self.assertNotIn('alpha_deg',active_parameters(self.cfg,self.ns,full.subset(z=0),'uncooled'))
 def test_physical_mapping_and_area(self):
  with self.assertRaises(ValueError):replace(self.target,theta_deg=(.7,)+self.target.theta_deg[1:]).validate(self.cfg)
  with self.assertRaises(ValueError):replace(self.target,radius_mm=(111,)+self.target.radius_mm[1:]).validate(self.cfg)
  mass=RegressionTarget.from_project(self.cfg,self.ns,'cell_mass',source_kind='synthetic')
  with self.assertRaises(ValueError):replace(mass,area_m2=(1,)*len(mass.y)).validate(self.cfg)
 def test_constant_target_r2_na(self):
  self.assertIsNone(_metrics(np.ones(5),np.ones(5),np.ones(5,dtype=bool))['R2'])
 def test_cancel_and_budget(self):
  count=[0]
  def progress(n,cost):count[0]=n
  result=fit(self.snapshot,self.target,[self.spec],budget=100,progress=progress,cancel=lambda:count[0]>=3)
  self.assertEqual(result.status,'cancelled');self.assertEqual(result.snapshot.sha256,RegressionSnapshot.capture(*result.candidate()).sha256)
  small=fit(self.snapshot,self.target,[self.spec],budget=20);self.assertLessEqual(small.evaluations,20);self.assertEqual(small.status,'budget')
 def test_csv_json_xlsx_reports(self):
  with tempfile.TemporaryDirectory() as d:
   for suffix in ('.json','.csv'):
    path=Path(d)/('target'+suffix);write_target(path,self.target);target=load_target(path)
    np.testing.assert_array_equal(target.y,self.target.y);target.validate(self.cfg)
   path=Path(d)/'report.xlsx';write_fit_xlsx(path,self.result);sheets=list(_xlsx_sheets(path));self.assertEqual([s[0] for s in sheets],['Parameters','Metrics','Residuals','Metadata','Validation'])
   target=load_target(path);np.testing.assert_array_equal(target.y,self.target.y)
   import openpyxl
   workbook=openpyxl.load_workbook(path,data_only=True);self.assertEqual(len(workbook.worksheets),5);workbook.close()

class TestV04Qt(unittest.TestCase):
 @classmethod
 def setUpClass(cls):cls.app=QApplication.instance() or QApplication([])
 def setUp(self):
  self.w=MainWindow();self.w._restoring=True;self.w.cfg,self.w.nozzles=user();self.w._fill_controls();self.w._fill_nozzles();self.w._reset_history();self.w.preview_timer.stop();self.w.selection_timer.stop()
 def tearDown(self):
  deadline=time.monotonic()+30
  while self.w.thread is not None and time.monotonic()<deadline:QTest.qWait(20)
  dialog=getattr(self.w,'regression_dialog',None)
  while dialog is not None and dialog.thread is not None and time.monotonic()<deadline:QTest.qWait(20)
  self.w.close();self.w.deleteLater();self.app.processEvents()
 def calculate(self):
  self.w.run_calc();deadline=time.monotonic()+30
  while self.w.thread is not None and time.monotonic()<deadline:QTest.qWait(20)
  self.assertIsNone(self.w.thread)
 def test_ctrl_middle_resets_zoom_real_qt(self):
  self.w.cfg.z_mm=[30.];self.w._fill_controls();self.calculate();self.w.show();self.w.plots.setCurrentIndex(3);self.app.processEvents();canvas=self.w.curvecanvas;canvas.draw();ax=canvas.figure.axes[0]
  original=ax.get_xlim();ax.set_xlim(50,100);canvas.draw();px,py=ax.transData.transform((75,np.mean(ax.get_ylim())));point=QPoint(round(px),round(canvas.height()-py))
  QTest.mouseClick(canvas,Qt.MiddleButton,Qt.ControlModifier,pos=point);self.app.processEvents();self.assertEqual(ax.get_xlim(),original)
 def test_arbitrary_graph_selection_and_equal_axes(self):
  self.w.cfg.z_mm=[30.];self.w._fill_controls();self.calculate();self.w.show();self.w.plots.setCurrentIndex(3);self.w.plot_count.setValue(1)
  selection=self.w.graph_selection;selection._choose(float(self.w.results[0].radial_centers_mm[-1]),True);selection._choose(float(self.w.results[0].radial_centers_mm[0]),False)
  self.assertEqual(len(self.w.figcurves.axes),1);self.assertIn('Уровень 6',self.w.figcurves.axes[0].get_title())
  self.w.split_button.setChecked(True);c=self.w.comparison;c._loading_cfg=self.w.cfg;c._project_done(self.w.results);c.plot_count.setValue(1)
  c.graph_selection._choose(float(self.w.results[0].radial_centers_mm[3]),True);c.graph_selection._choose(float(self.w.results[0].radial_centers_mm[0]),False)
  self.w._fit_plot_scales();self.app.processEvents();a=self.w.figcurves.axes[0];b=c.figure.axes[0]
  self.assertEqual(a.get_xlim(),b.get_xlim());self.assertEqual(a.get_ylim(),b.get_ylim());self.assertAlmostEqual(a.bbox.width,b.bbox.width,delta=1);self.assertAlmostEqual(a.bbox.height,b.bbox.height,delta=1)
 def test_seven_sections_worker_and_safe_cancel(self):
  self.w.cfg.z_mm=list(range(7));self.w._fill_controls();self.calculate();self.assertEqual(len(self.w.results),7);self.assertEqual(self.w.progress.maximum(),7)
  results=self.w.results;self.w.run_calc();self.w.thread.requestInterruption();deadline=time.monotonic()+30
  while self.w.thread is not None and time.monotonic()<deadline:QTest.qWait(20)
  self.assertIs(self.w.results,results);self.assertFalse(self.w.cancel_calc_btn.isEnabled())
 def test_tip_ui_and_sections_editor(self):
  self.assertEqual(self.w.nozzle_table.item(0,INTCOL['z_tip_mm']).text(),'0')
  self.w.open_sections_editor();self.assertEqual(self.w.sections_table.rowCount(),4);self.w.sections_dialog.close()
 def test_real_regression_worker_and_stale_guard(self):
  cfg,ns=synthetic();self.w._restoring=True;self.w.cfg=cfg;self.w.nozzles=ns;self.w._fill_controls();self.w._fill_nozzles();self.w._reset_history()
  self.w.open_regression();dialog=self.w.regression_dialog;dialog._target_done(TestFuelRegression.target)
  for row in range(dialog.parameters.rowCount()):
   if dialog.parameters.item(row,0).data(Qt.UserRole)==('uncooled','clock_deg'):
    dialog.parameters.item(row,0).setCheckState(Qt.Checked);dialog.parameters.item(row,5).setText('-20');dialog.parameters.item(row,6).setText('25')
  before=RegressionSnapshot.capture(self.w.cfg,self.w.nozzles).sha256;dialog.run();deadline=time.monotonic()+30
  while dialog.thread is not None and time.monotonic()<deadline:QTest.qWait(20)
  self.assertIsNone(dialog.thread);self.assertIsNotNone(dialog.result);self.assertEqual(RegressionSnapshot.capture(self.w.cfg,self.w.nozzles).sha256,before)
  self.w.geom_widgets['r_out_mm'].setValue(305);self.w._record_edit();dialog.apply();self.assertIn('изменён',dialog.status.text());self.assertEqual(self.w.cfg.r_out_mm,305)
 def test_real_loading_worker_rejects_temperature(self):
  self.w.open_regression();dialog=self.w.regression_dialog
  with tempfile.TemporaryDirectory() as d:
   path=Path(d)/'temperature.csv';path.write_text('quantity,unit,source_kind,z_mm,radius_mm,theta_deg,value\ntemperature,C,temperature,0,275,0,300')
   dialog.load_path(path);deadline=time.monotonic()+10
   while dialog.thread is not None and time.monotonic()<deadline:QTest.qWait(20)
   self.assertIsNone(dialog.target);self.assertIn('температур',dialog.status.text())
 def test_regression_apply_undo_and_complete_rollback(self):
  cfg,ns=synthetic();self.w._restoring=True;self.w.cfg=cfg;self.w.nozzles=ns;self.w._fill_controls();self.w._fill_nozzles();self.w._reset_history()
  self.w.open_regression();dialog=self.w.regression_dialog;dialog._target_done(TestFuelRegression.target);dialog._fit_done(TestFuelRegression.result)
  before=RegressionSnapshot.capture(self.w.cfg,self.w.nozzles).payload;dialog.apply();self.assertNotEqual(RegressionSnapshot.capture(self.w.cfg,self.w.nozzles).payload,before)
  self.w.undo_stack.undo();self.assertEqual(RegressionSnapshot.capture(self.w.cfg,self.w.nozzles).payload,before)
  self.w.undo_stack.redo();dialog.rollback();self.assertEqual(RegressionSnapshot.capture(self.w.cfg,self.w.nozzles).payload,before)

if __name__=='__main__':unittest.main()
