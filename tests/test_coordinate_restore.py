"""Restored editing and measurement-frame features, including pressure scans."""
import json,sys,tempfile,time,unittest
from pathlib import Path
from copy import deepcopy
from dataclasses import replace
from unittest.mock import patch
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from PyQt5.QtCore import Qt,QLocale,QPoint
from PyQt5.QtWidgets import QApplication,QStyleOptionViewItem
from PyQt5.QtTest import QTest
from fuelfield.core import calculate_plane,evaluate_continuous_field,project_to_json,project_from_json
from fuelfield.coordinates import to_global,from_global,grid_angles,plot_x
from fuelfield.quasi import calculate_quasi
from fuelfield.numeric_input import DecimalSpinBox
from fuelfield.gui import MainWindow,INTCOL,FIELDS
from test_quasi import case


class TestCoordinateMath(unittest.TestCase):
 def test_frame_permutation_preserves_physical_field_and_mass(self):
  cfg,ns,mode=case(constant=True);baseline=calculate_plane(ns,cfg,30);frame=replace(cfg,theta_zero_deg=90,theta_direction=-1);before=deepcopy(ns);rotated=calculate_plane(ns,frame,30)
  indices=[2,1,0,7,6,5,4,3];np.testing.assert_allclose(rotated.cell_mass_kg_s,baseline.cell_mass_kg_s[:,indices],rtol=1e-13,atol=1e-15)
  np.testing.assert_array_equal(evaluate_continuous_field(ns,cfg,30).flux_kg_m2_s,evaluate_continuous_field(ns,frame,30).flux_kg_m2_s);self.assertEqual(ns,before);self.assertLess(abs(rotated.rel_balance_error),1e-12)
  self.assertEqual(float(to_global(10,frame)),80);self.assertEqual(float(from_global(80,frame)),10)
 def test_quasi_columns_follow_measurement_frame_at_each_timestamp(self):
  cfg,ns,mode=case();cfg.theta_zero_deg=90;cfg.theta_direction=-1;out=calculate_quasi(ns,cfg,preview=False)
  np.testing.assert_allclose(mode.global_angles(cfg),[90,45,0,315,270,225,180,135])
  for i,z in enumerate(cfg.z_mm):
   for step,physical_column in enumerate([2,1,0,7,6,5,4,3]):
    expected=np.zeros((cfg.n_radial,cfg.n_angular))
    for channel in mode.channels:
     canonical=replace(cfg,theta_zero_deg=0,theta_direction=1,target_pressure_kgf_cm2=channel.pressure[step]);expected+=calculate_plane([n for n in ns if n.manifold_id==channel.id],canonical,z).cell_mass_kg_s
    np.testing.assert_allclose(out.composites[i].cell_mass_kg_s[:,step],expected[:,physical_column],rtol=1e-13,atol=1e-15)
  self.assertLess(out.maximum_balance_error,1e-12)
 def test_140_cells_and_legacy_defaults_and_json_roundtrip(self):
  cfg,ns,mode=case(140);cfg.theta_zero_deg=73.25;cfg.theta_direction=-1;cfg.display_x_axis='cells';np.testing.assert_allclose(plot_x(grid_angles(cfg),cfg,'cells'),np.arange(140),atol=1e-12)
  self.assertAlmostEqual(float(to_global(1*360/140,cfg)),(73.25-360/140)%360)
  raw=project_to_json(cfg,ns);restored,other=project_from_json(json.loads(json.dumps(raw)));self.assertEqual(restored.theta_direction,-1);self.assertEqual(restored.theta_zero_deg,73.25);self.assertEqual(restored.display_x_axis,'cells');self.assertEqual([n.theta_deg for n in other],[n.theta_deg for n in ns])
  for key in ('theta_zero_deg','theta_direction','display_x_axis'):raw['settings'].pop(key)
  legacy,_=project_from_json(raw);self.assertEqual((legacy.theta_zero_deg,legacy.theta_direction,legacy.display_x_axis),(0,1,'angle'))
 def test_invalid_frame_is_rejected(self):
  cfg,ns,mode=case()
  for changes in (dict(theta_zero_deg=float('nan')),dict(theta_direction=0),dict(theta_direction=True),dict(display_x_axis='invalid')):
   with self.assertRaises(ValueError):replace(cfg,**changes).validate()


class TestCoordinateUI(unittest.TestCase):
 @classmethod
 def setUpClass(cls):cls.app=QApplication.instance() or QApplication([])
 def setUp(self):
  self.w=MainWindow();self.w._restoring=True;self.w.cfg,self.w.nozzles,self.mode=case(140);self.w.cfg.quasistationary=False;self.w._fill_controls();self.w._fill_nozzles();self.w._reset_history();self.w.show()
 def tearDown(self):
  for owner in (self.w,self.w.comparison):
   deadline=time.monotonic()+60
   while owner.thread is not None and time.monotonic()<deadline:QTest.qWait(5)
  self.w.close();self.w.deleteLater();self.app.processEvents()
 def calculate(self):
  cfg=self.w.cfg;p=calculate_plane(self.w.nozzles,cfg,30);field=evaluate_continuous_field(self.w.nozzles,cfg,30);self.w._done([(p,field)]);self.w.tabs.setCurrentIndex(3);self.w.plots.setCurrentIndex(3);self.w.plot_count.setValue(1);self.app.processEvents();return p
 def test_decimal_typing_locales_padding_and_significant_digits(self):
  spin=DecimalSpinBox();spin.setDecimals(12);spin.setRange(-1e8,1e8);spin.show()
  for locale,typed in ((QLocale.c(),'.12345'),(QLocale('ru_RU'),',12345')):
   spin.setLocale(locale);spin.setValue(0);edit=spin.lineEdit();edit.setCursorPosition(len(edit.text()));QTest.keyClicks(edit,typed);spin.interpretText();self.assertAlmostEqual(spin.value(),.12345);self.assertNotIn('0000',spin.text())
  spin.setLocale(QLocale.c());spin.setDecimals(4);edit=spin.lineEdit();edit.setText('1.2300');edit.setCursorPosition(3);QTest.keyClicks(edit,'4');spin.interpretText();self.assertAlmostEqual(spin.value(),1.243)
  spin.setValue(1.2345);edit.setCursorPosition(3);QTest.keyClicks(edit,'9');spin.interpretText();self.assertAlmostEqual(spin.value(),1.2345);spin.close();spin.deleteLater()
 def test_table_editor_preserves_original_precision_and_accepts_fraction(self):
  table=self.w.nozzle_table;column=INTCOL['flow_kg_h'];item=table.item(0,column);delegate=table.itemDelegateForColumn(column) or table.itemDelegate();index=table.model().index(0,column);option=QStyleOptionViewItem();editor=delegate.createEditor(table,option,index);delegate.setEditorData(editor,index);original=item.text();delegate.setModelData(editor,table.model(),index);self.assertEqual(item.text(),original)
  editor.setValue(0);edit=editor.lineEdit();edit.setCursorPosition(len(edit.text()));QTest.keyClicks(edit,'.125');editor.interpretText();delegate.setModelData(editor,table.model(),index);self.assertAlmostEqual(float(table.item(0,column).text()),.125);editor.deleteLater()
 def test_cells_change_x_only_and_split_uses_same_coordinates_and_link(self):
  p=self.calculate();before=p.cell_mass_kg_s.copy();self.w.x_axis.setCurrentIndex(self.w.x_axis.findData('cells'));self.app.processEvents();ax=self.w.figcurves.axes[0];self.assertEqual(ax.get_xlim(),(0,140));np.testing.assert_allclose(ax.lines[0].get_xdata(),np.arange(141),atol=1e-12);np.testing.assert_array_equal(p.cell_mass_kg_s,before);self.assertIs(self.w.results[0],p)
  c=self.w.comparison;self.w.split_button.setChecked(True);c._loading_cfg=deepcopy(self.w.cfg);c._project_done([p]);c.plot_count.setValue(1);self.app.processEvents();self.w.curvecanvas.draw();c.canvas.draw();self.w.graph_links.refresh();ax=self.w.figcurves.axes[0];other=c.figure.axes[0];self.assertEqual(other._compare_x,'cell_index');self.w.graph_links.connect(ax,self.w.curvecanvas.interaction,other,c.interaction);self.w.graph_links.pin(ax,20);self.assertAlmostEqual(c.interaction.points[-1].x,20)
  self.w.undo_stack.undo();self.assertEqual(self.w.cfg.display_x_axis,'angle');self.assertIs(self.w.results[0],p)
 def test_frame_controls_keep_global_positions_and_edit_local_angles(self):
  self.w.cfg.placement_mode='manual';self.w._fill_controls();self.w._reset_history();physical=[n.theta_deg for n in self.w.nozzles]
  self.w.geom_widgets['theta_zero_deg'].setValue(90);self.w.geom_widgets['theta_zero_deg'].editingFinished.emit();self.w.geom_widgets['theta_direction'].setCurrentIndex(self.w.geom_widgets['theta_direction'].findData(-1));self.assertEqual([n.theta_deg for n in self.w.nozzles],physical)
  expected=(90-physical[0])%360;self.assertAlmostEqual(float(self.w.nozzle_table.item(0,INTCOL['theta_deg']).text()),expected,places=8);self.w.nozzle_table.item(0,INTCOL['theta_deg']).setText('0');self.assertEqual(self.w.nozzles[0].theta_deg,90);self.w.undo_stack.undo();self.assertEqual(self.w.nozzles[0].theta_deg,physical[0])
 def test_external_angle_and_cell_tables_and_nonangular_x(self):
  self.w.cfg.theta_zero_deg=90;self.w.cfg.theta_direction=-1;self.w.cfg.display_x_axis='cells';self.w._fill_controls();self.w._reset_history();c=self.w.comparison
  x,order,label,unit=c._table_x(np.array([0.,45.,90.,135.]),'angle_deg');np.testing.assert_allclose(x,[0,17.5,35,122.5]);self.assertEqual(order.tolist(),[2,1,0,3]);self.assertEqual(unit,'cell_index')
  c.x_reference.setCurrentIndex(c.x_reference.findData('local'));x,order,label,unit=c._table_x(np.array([0.,35.,70.]),'cell_index');np.testing.assert_allclose(x,[0,35,70]);x,order,label,unit=c._table_x(np.array([0.,1.,2.]),'t_s');np.testing.assert_array_equal(x,[0,1,2]);self.assertIsNone(unit)
 def test_collector_precedes_atomizer_and_real_header_menu_still_opens(self):
  keys=[key for key,_ in FIELDS];self.assertEqual(keys.index('manifold_id')+1,keys.index('atomizer_type_id'));self.w.tabs.setCurrentIndex(1);self.app.processEvents();table=self.w.nozzle_table;header=table.horizontalHeader();table.scrollToItem(table.item(0,INTCOL['flow_kg_h']));self.app.processEvents()
  for key in ('flow_kg_h','effective_flow'):
   col=INTCOL[key];pos=QPoint(header.sectionViewportPosition(col)+header.sectionSize(col)//2,header.height()//2);QTest.mouseClick(header.viewport(),Qt.LeftButton,pos=pos);self.assertTrue(self.w.flow_unit_menu.isVisible());self.w.flow_unit_menu.close()

if __name__=='__main__':unittest.main()
