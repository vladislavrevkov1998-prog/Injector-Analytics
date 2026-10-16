"""v0.4.1 numeric precision, independent maps, linked graphs and reference lines."""
import json,sys,time,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from PyQt5.QtWidgets import QApplication,QStyleOptionViewItem,QDoubleSpinBox
from PyQt5.QtCore import Qt,QPoint
from PyQt5.QtTest import QTest
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.backend_bases import MouseEvent,MouseButton
from fuelfield.gui import MainWindow,INTCOL
from fuelfield.core import project_from_json,calculate_plane
from fuelfield.regression import RegressionSnapshot,RegressionTarget,predict_snapshot
from fuelfield.graph_links import Curve,comparison_statistics
from fuelfield.plot_interaction import attach_interaction
from fuelfield.axes_settings import open_axes_settings
ROOT=Path(__file__).resolve().parents[1]

class TestCurves(unittest.TestCase):
 @classmethod
 def setUpClass(cls):cls.app=QApplication.instance() or QApplication([])
 def curve(self,x,y):
  fig=Figure();canvas=FigureCanvasQTAgg(fig);ax=fig.add_subplot();ax.plot(x,y);c=attach_interaction(canvas);return Curve(ax,c)
 def test_statistics_known_equation(self):
  a=self.curve([0,1,2,3],[1,2,3,4]);b=self.curve([0,1,2,3],[5,8,11,14]);s=comparison_statistics(a,b)
  self.assertAlmostEqual(s['slope'],3);self.assertAlmostEqual(s['offset'],2);self.assertAlmostEqual(s['r'],1);self.assertAlmostEqual(s['r2'],1)
 def test_gaps_no_extrapolation_and_invalid_x(self):
  a=self.curve([0,1,2,3],[1,np.nan,3,4]);self.assertTrue(np.isnan(a.at(.5)));self.assertTrue(np.isnan(a.at(-1)));self.assertEqual(a.at(2.5),3.5)
  self.assertFalse(self.curve([0,1,1],[1,2,3]).valid);self.assertFalse(self.curve([0,np.nan,2],[1,2,3]).valid)
  with self.assertRaises(ValueError):comparison_statistics(self.curve([0,1,2],[2,2,2]),self.curve([0,1,2],[1,2,3]))
 def test_constant_second_r2_undefined(self):
  s=comparison_statistics(self.curve([0,1,2],[1,2,3]),self.curve([0,1,2],[2,2,2]));self.assertIsNone(s['r']);self.assertIsNone(s['r2'])

class TestV041Qt(unittest.TestCase):
 @classmethod
 def setUpClass(cls):cls.app=QApplication.instance() or QApplication([])
 def setUp(self):
  self.w=MainWindow();self.w._restoring=True;self.w.cfg,self.w.nozzles=project_from_json(json.loads((ROOT/'examples/demo_project_user_field.json').read_text()))
  self.w.cfg.z_mm=[30.];self.w._fill_controls();self.w._fill_nozzles();self.w._reset_history();self.w.preview_timer.stop();self.w.selection_timer.stop();self.w.show()
 def tearDown(self):
  deadline=time.monotonic()+60
  while (self.w.thread is not None or self.w.comparison.thread is not None) and time.monotonic()<deadline:QTest.qWait(20)
  self.w.close();self.w.deleteLater();self.app.processEvents()
 def charts(self):
  self.w.run_calc();deadline=time.monotonic()+30
  while self.w.thread is not None and time.monotonic()<deadline:QTest.qWait(20)
  self.assertIsNone(self.w.thread);self.w.plots.setCurrentIndex(3);self.w.plot_count.setValue(1);self.w.split_button.setChecked(True)
  c=self.w.comparison;c._loading_cfg=self.w.cfg;c._project_done(self.w.results);c.plot_count.setValue(1);self.app.processEvents()
  self.w.curvecanvas.draw();c.canvas.draw();self.w.graph_links.refresh();self.app.processEvents()
  self.assertEqual(self.w.result_panels.count(),2)
  self.assertEqual(self.w.graph_links.overlay.geometry(),self.w.result_panels.geometry())
  return self.w.curvecanvas,c.canvas
 def event(self,canvas,ax,name,x,y,button=MouseButton.LEFT,double=False):
  return MouseEvent(name,canvas,*ax.transData.transform((x,y)),button=button,dblclick=double)
 def test_numeric_editor_step_preserves_original(self):
  self.assertEqual(self.w.scale_min.singleStep(),.0001);self.assertEqual(self.w.scale_max.singleStep(),.0001)
  table=self.w.nozzle_table;index=table.model().index(0,INTCOL['theta_deg']);table.item(0,INTCOL['theta_deg']).setText('12.123456789')
  delegate=table.itemDelegate();editor=delegate.createEditor(table,QStyleOptionViewItem(),index);self.assertIsInstance(editor,QDoubleSpinBox);delegate.setEditorData(editor,index)
  self.assertEqual(editor.singleStep(),.001);delegate.setModelData(editor,table.model(),index);self.assertEqual(index.data(),'12.123456789')
  editor.stepUp();delegate.setModelData(editor,table.model(),index);self.assertAlmostEqual(float(index.data()),12.124456789,places=10)
 def test_oblique_line_drag_limit_and_nearest_ignores_guides(self):
  canvas,_=self.charts();c=canvas.interaction;ax=canvas.figure.axes[0];x=180.;y=np.mean(ax.get_ylim());original=RegressionSnapshot.capture(self.w.cfg,self.w.nozzles).sha256
  for i in range(11):
   c.press(self.event(canvas,ax,'button_press_event',x,y));c.motion(self.event(canvas,ax,'motion_notify_event',240,y+np.ptp(ax.get_ylim())*.2));c.release(self.event(canvas,ax,'button_release_event',240,y))
  self.assertEqual(len(c.oblique),10);self.assertIn('y =',c.oblique[0].artists[1].get_text())
  c.press(self.event(canvas,ax,'button_press_event',180,y,double=True));self.assertEqual(len(c.points),1)
  c.clear_points();self.assertEqual(len(c.oblique),10);c.clear_oblique();self.assertFalse(c.oblique)
  self.assertEqual(RegressionSnapshot.capture(self.w.cfg,self.w.nozzles).sha256,original)
 def test_axis_limits_ticks_and_guard(self):
  a,b=self.charts();dialog=open_axes_settings(a,self.w);ax=a.figure.axes[0]
  for k,v in dict(xmin=10,xmax=20,ymin=0,ymax=.01,xmajor=2,xminor=.5,ymajor=.002,yminor=.001).items():dialog.widgets[k].setValue(v)
  dialog.apply_settings();self.assertEqual(ax.get_xlim(),(10,20));self.assertEqual(ax.get_ylim(),(0,.01));self.assertEqual(ax.xaxis.get_minor_locator()._edge.step,.5)
  dialog.widgets['xminor'].setValue(.0001);dialog.widgets['xmin'].setValue(5);dialog.apply_settings();self.assertEqual(ax.get_xlim(),(10,20));dialog.close()
 def test_link_real_circle_and_target_click_sync_hover_clear(self):
  a,b=self.charts();manager=self.w.graph_links;ax=a.figure.axes[0];bx=b.figure.axes[0];c=a.interaction;d=b.interaction
  source=manager.anchor(ax,c,True);point=manager.overlay.mapToGlobal(QPoint(round(source.x()),round(source.y())));point=a.mapFromGlobal(point)
  QTest.mouseClick(a,Qt.LeftButton,pos=point);self.assertIsNotNone(manager.pending)
  center=bx.transAxes.transform((.5,.5));QTest.mouseClick(b,Qt.LeftButton,pos=QPoint(round(center[0]),round(b.height()-center[1])));self.assertEqual(len(manager.links),1)
  ax.set_xlim(50,200);ax.set_ylim(0,.1);self.assertEqual(bx.get_xlim(),(50,200));self.assertEqual(bx.get_ylim(),(0,.1))
  x=100.;y=manager.links[0].a.at(x);c.press(self.event(a,ax,'button_press_event',x,y,double=True));self.assertEqual(len(c.points),1);self.assertEqual(len(d.points),1);self.assertAlmostEqual(c.points[0].x,d.points[0].x)
  text,color=manager.hover(ax,100);self.assertIn('ΔY',text);self.assertIn('link Y',text)
  c.clear_points();self.assertFalse(d.points);c.reset_zoom();self.assertEqual(ax.get_xlim(),bx.get_xlim())
  self.app.processEvents();a.draw();b.draw();self.app.processEvents()
  middle=(manager.anchor(ax,c,True)+manager.anchor(bx,d,False))/2
  parent=self.w.result_panels;point=QPoint(round(middle.x()),round(middle.y()));QTest.mouseMove(parent,point);self.app.processEvents();self.assertIs(manager.hovered,manager.links[0])
  with patch('fuelfield.graph_links.QMenu.exec_',lambda menu,pos:menu.actions()[0]):QTest.mouseClick(parent,Qt.LeftButton,pos=point)
  dialog=manager.analysis_dialog;self.assertGreater(dialog.statistics['n'],3);dialog.close()
 def test_no_field_calculation_during_linked_hover(self):
  a,b=self.charts();manager=self.w.graph_links;ax=a.figure.axes[0];bx=b.figure.axes[0];manager.connect(ax,a.interaction,bx,b.interaction)
  with patch('fuelfield.gui.calculate_plane',side_effect=AssertionError('physics called')),patch.object(a,'draw',side_effect=AssertionError('full draw')):
   for x in range(80,120):a.interaction.motion(self.event(a,ax,'motion_notify_event',x,np.mean(ax.get_ylim())))
 def test_json_split_matrix_and_summed_field_worker(self):
  c=self.w.comparison;c.load_path(ROOT/'examples/demo_project_user_field.json');deadline=time.monotonic()+60
  while c.thread is not None and time.monotonic()<deadline:QTest.qWait(20)
  self.assertIsNone(c.thread);self.assertEqual(len(c.planes),4);self.assertEqual(len(c.fields),4);self.assertIsNot(c.cfg,self.w.cfg)
  c.view.setCurrentIndex(1);self.app.processEvents();self.assertFalse(self.w.fit_scale_button.isEnabled());self.assertEqual(c.figure.axes[0].name,'polar');self.assertTrue(c.figure.axes[0].collections)
  c.view.setCurrentIndex(2);self.app.processEvents();self.assertEqual(len(c.figure.axes[0].images),1)
  c.view.setCurrentIndex(0);self.assertTrue(c.interaction.axes);self.assertTrue(c.figure.axes[0].lines)
 def test_xlsx_link_units_and_signed_difference(self):
  a,b=self.charts();p=self.w.results[0];x=p.theta_centers_deg;values=p.flux_kg_m2_s[0]
  from fuelfield.analytics_io import write_workbook
  with tempfile.TemporaryDirectory() as folder:
   path=Path(folder)/'other_R.xlsx';rows=[['Угол [°]','q [кг/(м²·с)]']]+[[float(t),float(v+1)] for t,v in zip(x,values)]
   write_workbook(path,[('other radius',rows)]);self.w.comparison.load_path(path);self.app.processEvents();b.draw();self.w.graph_links.refresh()
  manager=self.w.graph_links;ax=a.figure.axes[0];bx=b.figure.axes[0];manager.connect(ax,a.interaction,bx,b.interaction)
  text,color=manager.hover(ax,90);self.assertEqual(color,'#c33333');self.assertIn('-1',text)
  self.assertEqual(manager.hover(bx,90)[1],'#18833b')
  bx._compare_y='°C';self.assertIn('разные единицы',manager.hover(ax,90)[0]);bx._compare_x='mm'
  with self.assertRaises(ValueError):manager.connect(ax,a.interaction,bx,b.interaction)
 def test_regression_physical_epures_original_and_start_separate(self):
  from test_v04 import synthetic
  from fuelfield.regression import fit,ParameterSpec
  cfg,ns=synthetic();self.w._restoring=True;self.w.cfg=cfg;self.w.nozzles=ns;self.w._fill_controls();self.w._fill_nozzles();self.w._reset_history()
  snapshot=RegressionSnapshot.capture(cfg,ns);target=RegressionTarget.from_project(cfg,ns,source_kind='synthetic')
  result=fit(snapshot,target,[ParameterSpec('clock_deg','uncooled',5,-20,20)],budget=40)
  self.w.open_regression();dialog=self.w.regression_dialog;dialog._target_done(target);original=predict_snapshot(snapshot,target);dialog._fit_done((result,original))
  ax=dialog.figure.axes[0];lines=[l for l in ax.lines if not l.get_animated()];labels=[l.get_label() for l in lines]
  self.assertIn('Исходный проект',labels);self.assertIn('Старт подбора',labels);self.assertEqual(ax.get_xlabel(),'θ, °')
  self.assertTrue(all(np.asarray(l.get_xdata()).max()<=360 for l in lines));self.assertGreater(dialog.preview_pair.count(),1)
  dialog.preview_pair.setCurrentIndex(dialog.preview_pair.count()-1);self.assertIn('z=45',dialog.figure.axes[0].get_title())
  dialog.show_guide();self.assertTrue(dialog.guide_dialog.isVisible());dialog.guide_dialog.close();dialog.close()
 def test_invalidation_detaches_deleted_canvases_and_axes_dialog(self):
  from PyQt5.QtCore import QCoreApplication,QEvent
  a,b=self.charts();m=self.w.graph_links;m.connect(a.figure.axes[0],a.interaction,b.figure.axes[0],b.interaction)
  dialog=open_axes_settings(a,self.w);self.w._invalidate_results();QCoreApplication.sendPostedEvents(None,QEvent.DeferredDelete);self.app.processEvents()
  m.refresh();m.overlay.repaint();self.assertFalse(m.links);self.assertFalse(m.sources());self.assertEqual(self.w.result_panels.count(),2)
  self.assertIs(self.w.curvecanvas,self.w.plot_stacks[3].widget(0));dialog.apply_settings();dialog.close()
 def test_original_snapshot_prediction_matches_real_epures(self):
  target=RegressionTarget.from_project(self.w.cfg,self.w.nozzles,source_kind='synthetic');snapshot=RegressionSnapshot.capture(self.w.cfg,self.w.nozzles)
  pred=predict_snapshot(snapshot,target);np.testing.assert_allclose(pred,target.y,rtol=0,atol=0);self.assertEqual(RegressionSnapshot.capture(self.w.cfg,self.w.nozzles).sha256,snapshot.sha256)

if __name__=='__main__':unittest.main()
