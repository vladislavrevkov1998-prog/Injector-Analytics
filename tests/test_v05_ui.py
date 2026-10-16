"""Fine decimal typing, reference frame, cell axes and v0.5 session workflows."""
import sys,json,tempfile,time,unittest
from pathlib import Path
from dataclasses import replace
from unittest.mock import patch
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from PyQt5.QtWidgets import QApplication,QStyleOptionViewItem
from PyQt5.QtCore import Qt,QPoint
from PyQt5.QtTest import QTest
from PyQt5.QtCore import QLocale
from fuelfield.gui import MainWindow,INTCOL
from fuelfield.numeric_input import DecimalSpinBox
from fuelfield.core import calculate_plane,evaluate_continuous_field,project_to_json,project_from_json
from fuelfield.measurement import MeasurementSession,MeasurementStep,PressureSeries
from fuelfield.measurement_io import read_tables,import_steps,import_pressures,write_report,save_session,load_session
from fuelfield.analytics_io import write_workbook
from fuelfield.data_io import _xlsx_sheets
from test_v05_math import case
ROOT=Path(__file__).resolve().parents[1]

class TestV05IO(unittest.TestCase):
 def test_mapping_csv_xlsx_missing_temperature_and_raw_provenance(self):
  with tempfile.TemporaryDirectory() as folder:
   p=Path(folder)/'protocol.csv';p.write_text('when;until;a;b;c;d;e;f\n0;1000;100;110;120;130;;150\n',encoding='utf-8')
   table=read_tables(p)[0];mapping={'t_start_s':'when','t_end_s':'until',**{f'T_R{i}_C':h for i,h in enumerate('abcdef',1)}}
   steps=import_steps(table,mapping,time_scale_s=.001);self.assertEqual(steps[0].t_end_s,1);self.assertIsNone(steps[0].temperatures_C[4]);self.assertFalse(steps[0].valid[4]);self.assertEqual(table['source']['raw_rows'][1][1],'1000')
   p=Path(folder)/'pressure.xlsx';rows=[['time','P','group','units','kind'],[0,980665,'A','Pa','delta_p'],[1000,10,'A','kgf/cm2','delta_p']];write_workbook(p,[('pressure',rows)]);tables=read_tables(p)
   series=import_pressures(tables[0],dict(t_s='time',pressure='P',manifold_id='group',unit='units',pressure_kind='kind'),time_scale_s=.001)
   self.assertEqual(series[0].t_s,(0,1));np.testing.assert_allclose(series[0].pressure,(10,10));self.assertEqual(series[0].metadata['source_units'],['Pa','kgf/cm2'])
 def test_timestamp_is_explicit_instantaneous_or_user_duration(self):
  table=dict(headers=['time',*[f'T_R{i}_C' for i in range(1,7)]],rows=[[2,*range(6)]])
  instant=import_steps(table,{'t_start_s':'time'})[0];self.assertEqual(instant.t_start_s,instant.t_end_s);self.assertEqual(instant.metadata['sampling_method'],'instantaneous')
  self.assertEqual(import_steps(table,{'t_start_s':'time'},duration_s=3)[0].t_end_s,5)
 def test_export_and_json_project_roundtrip(self):
  from fuelfield.transient import calculate_session
  cfg,ns,s=case();result=calculate_session(ns,cfg,s)
  with tempfile.TemporaryDirectory() as folder:
   path=Path(folder)/'report.xlsx';write_report(path,result);sheets=dict(_xlsx_sheets(path));self.assertEqual(len(sheets['Measurements']),13);self.assertEqual(len(sheets['Flows'][0]),144);self.assertIn('Instant_balances',sheets)
   files=write_report(Path(folder)/'report.csv',result);self.assertGreater(len(files),5);manifest=json.loads((Path(folder)/'report_manifest.json').read_text());self.assertIsNone(manifest['composite_global_balance'])
   path=Path(folder)/'session.json';save_session(path,s);self.assertEqual(load_session(path).to_dict(),s.to_dict());data=project_to_json(cfg,ns);data['measurement_session']=s.to_dict();path.write_text(json.dumps(data));self.assertEqual(load_session(path).to_dict(),s.to_dict())
 def test_json_rejects_ambiguous_pressure(self):
  cfg,ns,s=case();raw=s.to_dict();del raw['pressure_series'][0]['unit']
  with self.assertRaises(ValueError):MeasurementSession.from_dict(raw)

class TestV05UI(unittest.TestCase):
 @classmethod
 def setUpClass(cls):cls.app=QApplication.instance() or QApplication([])
 def setUp(self):
  self.w=MainWindow();cfg,ns,self.session=case();self.w._restoring=True;self.w.cfg=cfg;self.w.nozzles=ns;self.w._fill_controls();self.w._fill_nozzles();self.w._reset_history();self.w.preview_timer.stop();self.w.selection_timer.stop();self.w.show()
 def tearDown(self):
  for thread_owner in (self.w,self.w.measurement_panel,self.w.comparison):
   deadline=time.monotonic()+60
   while thread_owner.thread is not None and time.monotonic()<deadline:QTest.qWait(10)
  self.w.close();self.w.deleteLater();self.app.processEvents()
 def wait(self,panel):
  deadline=time.monotonic()+30
  while panel.thread is not None and time.monotonic()<deadline:QTest.qWait(10)
  self.assertIsNone(panel.thread)
 def calculate(self):
  cfg=self.w.cfg;plane=calculate_plane(self.w.nozzles,cfg,20);field=evaluate_continuous_field(self.w.nozzles,cfg,20,resolution=200);self.w._done([(plane,field)]);return plane
 def test_decimal_typing_without_deleting_zero_padding_and_comma(self):
  spin=DecimalSpinBox();spin.setLocale(QLocale.c());spin.setDecimals(12);spin.setRange(-1e8,1e8);spin.setValue(0);spin.show();edit=spin.lineEdit();edit.setCursorPosition(len(edit.text()));QTest.keyClicks(edit,'.12345');spin.interpretText();self.assertAlmostEqual(spin.value(),.12345);self.assertEqual(spin.text(),'0.12345')
  spin.setValue(0);edit.setCursorPosition(len(edit.text()));QTest.keyClicks(spin,',125');spin.interpretText();self.assertAlmostEqual(spin.value(),.125)
  spin.setDecimals(4);edit.setText('1.2300');edit.setCursorPosition(3);QTest.keyClicks(edit,'4');spin.interpretText();self.assertAlmostEqual(spin.value(),1.243)
  spin.close()
 def test_cell_axis_keeps_flux_and_split_link_coordinates(self):
  p=self.calculate();before=p.cell_mass_kg_s.copy();self.w.plots.setCurrentIndex(3);self.w.plot_count.setValue(1);self.w.x_axis.setCurrentIndex(self.w.x_axis.findData('cells'));self.app.processEvents();ax=self.w.figcurves.axes[0]
  self.assertEqual(ax.get_xlim(),(0,140));self.assertEqual(ax._compare_x,'cell_index');self.assertIn('ячейка',ax.get_xlabel());self.assertAlmostEqual(ax.lines[0].get_xdata()[1],1);np.testing.assert_array_equal(p.cell_mass_kg_s,before);self.assertIs(self.w.results[0],p)
  self.w.split_button.setChecked(True);c=self.w.comparison;c._loading_cfg=self.w.cfg;c._project_done([p]);c.plot_count.setValue(1);self.app.processEvents();self.w.curvecanvas.draw();c.canvas.draw();self.w.graph_links.refresh();link=self.w.graph_links.connect(ax,self.w.curvecanvas.interaction,c.figure.axes[0],c.interaction);self.w.graph_links.pin(ax,20);self.assertAlmostEqual(c.interaction.points[-1].x,20)
 def test_frame_preserves_physical_nozzles_and_undo(self):
  before=[n.theta_deg for n in self.w.nozzles];self.w.geom_widgets['theta_zero_deg'].setValue(90);self.w.geom_widgets['theta_direction'].setCurrentIndex(self.w.geom_widgets['theta_direction'].findData(-1));self.w._record_edit();self.assertEqual([n.theta_deg for n in self.w.nozzles],before)
  self.assertEqual(float(self.w.nozzle_table.item(0,INTCOL['theta_deg']).text()),90);self.w.undo_stack.undo();self.assertEqual(self.w.cfg.theta_direction,1);self.assertEqual(self.w.cfg.theta_zero_deg,0)
 def test_session_worker_cache_snapshot_and_cancel(self):
  panel=self.w.measurement_panel;panel.set_session(self.session);self.w.tabs.setCurrentWidget(panel);before=json.dumps(project_to_json(self.w.cfg,self.w.nozzles),sort_keys=True);panel.run();self.wait(panel);self.assertIsNotNone(panel.result);result=panel.result;operator=panel.operator;self.assertEqual(before,json.dumps(project_to_json(self.w.cfg,self.w.nozzles),sort_keys=True));self.assertEqual(len(panel.figure.axes),12)
  panel.run();self.wait(panel);self.assertIs(panel.operator,operator);result=panel.result
  panel.run();panel.cancel();self.wait(panel);self.assertIs(panel.result,result)
  for kind in range(5):panel.result_kind.setCurrentIndex(kind);self.app.processEvents()
  self.assertIn('невязка',panel.balance.text());self.assertIsNone(panel.result.manifest()['temperature_model'])
 def test_attach_undo_preserves_static_fields_and_json(self):
  p=self.calculate();panel=self.w.measurement_panel;panel.set_session(self.session);panel.attach();self.assertIn('measurement_session',self.w.cfg._project_extra);self.assertIs(self.w.results[0],p)
  self.w.undo_stack.undo();self.assertNotIn('measurement_session',self.w.cfg._project_extra);self.assertIs(self.w.results[0],p);self.assertEqual(panel.steps.rowCount(),0);self.w.undo_stack.redo();self.assertIn('measurement_session',self.w.cfg._project_extra);self.assertEqual(panel.steps.rowCount(),len(self.session.steps))
 def test_add_row_preserves_incomplete_input_and_worker_locks_buttons(self):
  panel=self.w.measurement_panel;panel.set_session(self.session);panel.steps.item(0,2).setText('unfinished');count=panel.steps.rowCount();panel.add_step();self.assertEqual(panel.steps.rowCount(),count+1);self.assertEqual(panel.steps.item(0,2).text(),'unfinished')
  panel.set_session(self.session);panel.run();self.assertTrue(all(not b.isEnabled() for b in panel.edit_buttons));self.wait(panel);self.assertTrue(all(b.isEnabled() for b in panel.edit_buttons))
 def test_imported_precision_survives_unrelated_edits(self):
  value=0.12345678901234567;step=replace(self.session.steps[0],t_start_s=value,theta_deg=value,temperatures_C=(123.12345678901234,)*6)
  source=replace(self.session,steps=(step,),z_mm=20+value,radii_mm=tuple(v+value for v in self.session.radii_mm),transport_delay_s=value,nozzle_delays_s={'1':value})
  panel=self.w.measurement_panel;panel.set_session(source);panel.steps.item(0,17).setText('edited note');actual=panel.read_session()
  for key in ('z_mm','radii_mm','transport_delay_s','nozzle_delays_s'):self.assertEqual(getattr(actual,key),getattr(source,key))
  self.assertEqual(actual.steps[0].t_start_s,value);self.assertEqual(actual.steps[0].theta_deg,value);self.assertEqual(actual.steps[0].temperatures_C,step.temperatures_C)
 def test_result_layout_reserves_plot_space_at_1280(self):
  panel=self.w.measurement_panel;self.w.tabs.setCurrentWidget(panel);self.w.resize(1280,800);panel.pages.setCurrentWidget(panel.result_page);self.app.processEvents()
  self.assertFalse(panel.passport_toggle.isChecked());self.assertGreater(panel.canvas.height(),350);self.assertGreater(panel.canvas.width(),500)
  panel.passport_toggle.setChecked(True);self.app.processEvents();self.assertTrue(panel.settings_widget.isVisible());panel.pages.setCurrentIndex(0);self.app.processEvents();self.assertTrue(panel.passport_toggle.isChecked())
 def test_table_mapping_import_and_raw_rows_preserved(self):
  panel=self.w.measurement_panel
  with tempfile.TemporaryDirectory() as folder:
   path=Path(folder)/'p.csv';path.write_text('clock;value\n0;10\n2;20\n')
   panel.import_table(read_tables(path),'pressure',dict(t_s='clock',pressure='value'),dict(unit='kgf/cm2',pressure_kind='delta_p',manifold_id='A'));self.assertEqual(len(panel.pressure_model.rows),2);self.assertIn('raw_imports',panel.metadata);self.assertEqual(panel.metadata['raw_imports'][0]['source']['raw_rows'][1],['0','10'])
 def test_invalid_data_keeps_previous_result(self):
  panel=self.w.measurement_panel;panel.set_session(self.session);panel.run();self.wait(panel);previous=panel.result;panel.pressure_model.rows[1][1]=0;panel.run();self.assertIs(panel.result,previous);self.assertIn('возрастать',panel.status.text())

if __name__=='__main__':unittest.main()
