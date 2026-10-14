"""v0.3 regressions using the user's project and independent XLSX reader."""
import json
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
import sys,tempfile,time,unittest
from unittest.mock import patch
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from fuelfield.core import project_from_json,project_to_json,calculate_plane,evaluate_continuous_field
from fuelfield.analysis import cell_values,METRICS,matching_index
from fuelfield.analytics_io import write_epures_xlsx,read_chart_tables
from fuelfield.data_io import _xlsx_sheets
from fuelfield.gui import MainWindow,INTCOL
from PyQt5.QtCore import Qt,QPoint
from PyQt5.QtTest import QTest
from PyQt5.QtWidgets import QApplication,QDialog

PROJECT=Path(__file__).resolve().parents[1]/'examples/demo_project_user_field.json'
def fixture():return project_from_json(json.loads(PROJECT.read_text()))

class TestQuantities(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg,ns=fixture();cls.planes=[calculate_plane(ns,cls.cfg,z) for z in cls.cfg.z_mm]
    def test_all_conversions_and_no_mutation(self):
        p=self.planes[2];before=p.cell_mass_kg_s.copy();rho=780;t=37.25
        for metric,expected in [('flux',p.cell_mass_kg_s/p.cell_area_m2),('kg_h',p.cell_mass_kg_s*3600),('l_min',p.cell_mass_kg_s*60000/rho),('volume',p.cell_mass_kg_s*t*1e6/rho)]:
            np.testing.assert_allclose(cell_values(p,metric,rho,t),expected)
        np.testing.assert_array_equal(before,p.cell_mass_kg_s)
        self.assertAlmostEqual(cell_values(p,'kg_h').sum(),p.inside_kg_s*3600)
    def test_indices_have_area_weighted_mean_one(self):
        p=self.planes[2]
        for metric,axis in [('ring',1),('track',None)]:
            values=cell_values(p,metric)
            np.testing.assert_allclose((values*p.cell_area_m2).sum(axis=axis)/p.cell_area_m2.sum(axis=axis),1)
    def test_zero_mean_is_undefined_not_zero(self):
        p=replace(self.planes[0],cell_mass_kg_s=np.zeros_like(self.planes[0].cell_mass_kg_s),flux_kg_m2_s=np.zeros_like(self.planes[0].flux_kg_m2_s))
        for metric in ('ring','track'):self.assertTrue(cell_values(p,metric).mask.all())
        self.assertFalse(cell_values(p,'volume').mask.any())
    def test_invalid_parameters(self):
        for metric,rho,t in [('bad',800,180),('flux',0,180),('volume',800,0),('flux',np.nan,180),('volume',800,np.inf)]:
            with self.assertRaises(ValueError):cell_values(self.planes[0],metric,rho,t)
    def test_matching_physical_values(self):
        self.assertEqual(matching_index([50,0,15,30],15),2)
        self.assertIsNone(matching_index([0,15],30))
        self.assertIsNone(matching_index([15,15],15))
    def test_xlsx_four_sheets_all_levels_values_coordinates_metadata(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'epures.xlsx';write_epures_xlsx(path,self.planes,self.cfg,'volume',45)
            sheets=list(_xlsx_sheets(path));self.assertEqual(len(sheets),4)
            for p,(_,rows) in zip(self.planes,sheets):
                self.assertEqual(len(rows),self.cfg.n_angular+1)
                self.assertEqual(len(rows[0]),1+3*self.cfg.n_radial+7)
                values=cell_values(p,'volume',self.cfg.density_kg_m3,45)
                for j,row in enumerate(rows[1:]):
                    self.assertAlmostEqual(float(row[0]),p.theta_centers_deg[j])
                    for k,r in enumerate(p.radial_centers_mm):
                        self.assertAlmostEqual(float(row[1+3*k]),values[k,j])
                        self.assertAlmostEqual(float(row[2+3*k]),r*np.cos(np.deg2rad(float(row[0]))))
                    self.assertEqual(row[-6],'мл');self.assertEqual(row[-5],self.cfg.pressure_mode)
                    self.assertEqual(json.loads(row[-1]),project_to_json(self.cfg,[])['settings'])
            self.assertEqual(len(read_chart_tables(path)),4)
            try:import openpyxl
            except ImportError:pass
            else:
                wb=openpyxl.load_workbook(path,data_only=True)
                self.assertEqual(len(wb.worksheets),4)
                self.assertAlmostEqual(wb.worksheets[2]['B2'].value,cell_values(self.planes[2],'volume',780,45)[0,0]);wb.close()
    def test_xlsx_nondefault_radial_grid(self):
        cfg,ns=fixture();cfg.n_radial=3;cfg.radial_centers_mm=[280,340,410]
        planes=[calculate_plane(ns,cfg,z) for z in cfg.z_mm]
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'custom.xlsx';write_epures_xlsx(path,planes,cfg)
            for _,rows in _xlsx_sheets(path):self.assertEqual(len(rows[0]),17)
    def test_external_csv_decimal_comma_and_invalid_data(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'data.csv';p.write_text('angle_deg;fuel [kg/h];temperature [C]\n0;1,5;300\n90;2,5;350\n',encoding='utf-8-sig')
            tables=read_chart_tables(p);np.testing.assert_array_equal(tables[0][2][1],[1.5,2.5])
            for text in ['a,b\n0,NaN\n1,2','a,b\n0,1\n1,','a,b\n0,inf\n1,2']:
                p.write_text(text)
                with self.assertRaises(ValueError):read_chart_tables(p)

class TestV03Qt(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.app=QApplication.instance() or QApplication([])
    def setUp(self):
        self.w=MainWindow();self.w._restoring=True;self.w.cfg,self.w.nozzles=fixture();self.w._fill_controls();self.w._fill_nozzles();self.w._reset_history()
        self.w.preview_timer.stop();self.w.selection_timer.stop()
    def tearDown(self):
        deadline=time.monotonic()+20
        while self.w.comparison.thread is not None and time.monotonic()<deadline:QTest.qWait(20)
        self.w.close();self.w.deleteLater();self.app.processEvents()
    def _results(self):
        p=calculate_plane(self.w.nozzles,self.w.cfg,0);f=evaluate_continuous_field(self.w.nozzles,self.w.cfg,0)
        self.w._done([(p,f)]);return p
    def test_settings_is_window_and_reopens_without_tabs(self):
        count=self.w.tabs.count();self.w.tabs.setCurrentIndex(1)
        self.w.open_settings();self.assertIsInstance(self.w.settings_page,QDialog)
        self.assertTrue(self.w.settings_page.isVisible());self.assertEqual(self.w.tabs.currentIndex(),1)
        self.w.mode_widgets['target_pressure_kgf_cm2'].setValue(25)
        self.w.settings_page.close();self.w.open_settings()
        self.assertEqual(self.w.tabs.count(),count);self.assertEqual(self.w.mode_widgets['target_pressure_kgf_cm2'].value(),25)
    def test_actual_header_mouse_click_opens_unit_menu(self):
        self.w.show();self.w.tabs.setCurrentIndex(1);self.app.processEvents()
        tab=self.w.nozzle_table;header=tab.horizontalHeader();col=INTCOL['flow_kg_h']
        tab.scrollToItem(tab.item(0,col));self.app.processEvents()
        position=QPoint(header.sectionViewportPosition(col)+header.sectionSize(col)//2,header.height()//2)
        QTest.mouseClick(header.viewport(),Qt.LeftButton,pos=position)
        self.assertTrue(self.w.flow_unit_menu.isVisible());self.assertEqual(len(self.w.flow_unit_menu.actions()),2)
        self.w.flow_unit_menu.close()
        col=INTCOL['effective_flow'];position=QPoint(header.sectionViewportPosition(col)+header.sectionSize(col)//2,header.height()//2)
        QTest.mouseClick(header.viewport(),Qt.LeftButton,pos=position)
        self.assertTrue(self.w.flow_unit_menu.isVisible());self.w.flow_unit_menu.close()
    def test_readonly_lavender_and_localized_names_preserve_ids(self):
        t=self.w.nozzle_table
        for key in ('id','row','effective_flow','alpha_deg'):
            self.assertEqual(t.item(0,INTCOL[key]).background().color().name(),'#eee6f6')
        self.assertNotEqual(t.item(0,INTCOL['flow_kg_h']).background().color().name(),'#eee6f6')
        self.assertEqual(t.item(0,INTCOL['row']).text(),'наружный');self.assertEqual(t.item(70,INTCOL['row']).text(),'внутренний')
        self.assertEqual(t.horizontalHeaderItem(INTCOL['atomizer_type_id']).text(),'Имя распылителя')
        old=deepcopy(self.w.nozzles);self.assertEqual(t.item(0,INTCOL['atomizer_type_id']).text(),'1,3')
        ns=self.w._read_nozzles();self.assertEqual([n.atomizer_type_id for n in ns],[n.atomizer_type_id for n in old])
        t.item(0,INTCOL['manual_write']).setCheckState(Qt.Checked)
        self.assertTrue(t.item(0,INTCOL['alpha_deg']).flags()&Qt.ItemIsEditable)
        self.assertNotEqual(t.item(0,INTCOL['alpha_deg']).background().color().name(),'#eee6f6')
    def test_catalog_rename_and_duplicate_names_preserve_ids(self):
        ids=[n.atomizer_type_id for n in self.w.nozzles]
        catalog=deepcopy(self.w.cfg.atomizer_types)
        for item in catalog:item['name']='Общее имя'
        self.w._catalog_changed(catalog)
        self.assertEqual([n.atomizer_type_id for n in self.w.nozzles],ids)
        self.assertEqual(self.w.nozzle_table.item(0,INTCOL['atomizer_type_id']).text(),'Общее имя')
        self.assertEqual([n.atomizer_type_id for n in self.w._read_nozzles()],ids)

    def test_name_assignment_keeps_reference_canonical(self):
        self.w.nozzle_table.item(0,INTCOL['atomizer_type_id']).setText('0,9')
        self.assertEqual(self.w.nozzles[0].atomizer_type_id,'atomizer-1')
        self.assertEqual(self.w.nozzle_table.item(0,INTCOL['atomizer_type_id']).text(),'0,9')
    def test_representations_keep_physics_and_cache_valid(self):
        p=self._results();before=p.cell_mass_kg_s.copy();results=self.w.results
        for key in METRICS:
            self.w.metric.setCurrentIndex(self.w.metric.findData(key));self.w.plots.setCurrentIndex(3)
            self.assertEqual(self.w.figcurves.axes[0].get_ylabel(),{'flux':'кг/(м²·с)','kg_h':'кг/ч','l_min':'л/мин','volume':'мл','ring':'1','track':'1'}[key])
            self.assertIs(self.w.results,results);np.testing.assert_array_equal(p.cell_mass_kg_s,before)
        self.w.metric.setCurrentIndex(self.w.metric.findData('volume'));self.w.volume_interval.setValue(50)
        np.testing.assert_allclose(self.w.figcurves.axes[0].lines[0].get_ydata()[:-1],cell_values(p,'volume',780,50)[0,np.argsort(p.theta_centers_deg)])
    def test_real_background_json_comparison_and_mismatch(self):
        p=self._results();c=self.w.comparison;c.load_path(PROJECT)
        deadline=time.monotonic()+20
        while c.thread is not None and time.monotonic()<deadline:QTest.qWait(20)
        self.assertIsNone(c.thread);self.assertEqual(len(c.planes),4);self.assertEqual(c.plane.currentIndex(),0)
        self.w.compare_radius.setCurrentIndex(3);self.assertEqual(c.radius.currentIndex(),3)
        c.planes.reverse();c._populate([str(v.z_mm) for v in c.planes]);c.sync_from_primary()
        self.assertEqual(c.plane.currentIndex(),3)  # physical z, not list index
        c.cfg=replace(c.cfg,density_kg_m3=1000.)
        self.w.metric.setCurrentIndex(self.w.metric.findData('l_min'))
        np.testing.assert_allclose(c.figure.axes[0].lines[0].get_ydata(),cell_values(c.planes[3],'l_min',1000.)[3,np.argsort(p.theta_centers_deg)])
        c.planes[3]=replace(c.planes[3],theta_centers_deg=p.theta_centers_deg+.5);c.sync_from_primary()
        self.assertIn('сетки различаются',c.notice.text())
        self.w.results=[replace(p,z_mm=999)];c.sync_from_primary();self.assertIn('Несовместимые z',c.notice.text())
        self.w.results=[p];self.w.compare_radius.blockSignals(True);self.w.compare_radius.clear();self.w.compare_radius.addItem('100',100.);self.w.compare_radius.blockSignals(False)
        c.sync_from_primary();self.assertIn('Радиальные уровни различаются',c.notice.text())
    def test_table_two_y_axes_and_chart_types(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'measurements.csv';path.write_text('angle;fuel kg/h;temperature C\n0;1;300\n90;2;350')
            c=self.w.comparison;c.load_path(path);c.y2.setCurrentIndex(c.y2.findData(2))
            self.assertEqual(len(c.figure.axes),2);self.assertEqual(c.figure.axes[1].get_ylabel(),'temperature C')
            for kind in range(3):
                c.kind.setCurrentIndex(kind)
                self.assertEqual(len(c.figure.axes),2)
                ax=c.figure.axes[0]
                if kind==0:self.assertEqual(len(ax.lines),1)
                elif kind==1:self.assertEqual(len(ax.collections),1)
                else:self.assertEqual(len(ax.patches),2)
    def test_bad_table_does_not_discard_previous_data(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'valid.csv';p.write_text('a,b\n0,1\n1,2');c=self.w.comparison;c.load_path(p);before=c.tables
            p.write_text('a,b\nNaN,x')
            with self.assertRaises(ValueError):c.load_path(p)
            self.assertIs(c.tables,before)

if __name__=='__main__':unittest.main()
