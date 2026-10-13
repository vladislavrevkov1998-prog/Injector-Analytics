"""Original-source coefficients, independent numerics and full Qt integration."""
from copy import deepcopy
from dataclasses import replace
import json
import math
from pathlib import Path
import sys
import tempfile
import time
import unittest
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from fuelfield.dityakin import (DityakinParameters,DityakinCharacteristic,
    chamber_friction,filling_coefficient,chamber_loss,KGF_CM2_TO_PA)
from fuelfield.characteristics import PressureOutOfRange
from fuelfield.core import (Settings,default_nozzles,source_flow_kg_s,calculate_plane,
    project_to_json,project_from_json)
from fuelfield.v02 import flow_status,validate_catalog


def catalog():
    return [dict(id='hydraulic',name='Демо — Дитякин',kind='theoretical',
                 method='dityakin_short_v1',parameters=DityakinParameters().to_dict(),
                 plot_pressure_kgf_cm2=[2.5,30.])]


class TestDityakinNumerics(unittest.TestCase):
    def test_source_friction_both_regimes_and_graph_consistency(self):
        # Figure 28: lg(100*lambda) near 1.14 at lg(Re)=3,
        # .65 at lg(Re)=4 and .29 at lg(Re)=5 (curve 1).
        for re,reading in ((1000,1.14),(10000,.65),(100000,.29)):
            self.assertAlmostEqual(math.log10(100*chamber_friction(re)),reading,delta=.025)
        self.assertAlmostEqual(chamber_friction(2300),24.6/2300**.75)
        self.assertAlmostEqual(chamber_friction(2301),1.22/2301**.36)

    def test_phi_matches_ideal_closed_form_and_maximum(self):
        for phi in (.05,.2,.5,.9):
            a=math.sqrt(2)*(1-phi)/phi**1.5
            self.assertAlmostEqual(filling_coefficient(a),phi,places=14)
            mu=1/math.sqrt(a*a/(1-phi)+1/phi**2)
            self.assertAlmostEqual(mu,math.sqrt(phi**3/(2-phi)),places=14)
            for nearby in (phi*.99,phi*1.01):
                self.assertGreater(mu,1/math.sqrt(a*a/(1-nearby)+1/nearby**2))

    def test_independent_rk4_momentum_work_and_maximum_flow_reference(self):
        data=json.loads((Path(__file__).parent/'fixtures/dityakin_reference.json').read_text())
        curve=DityakinCharacteristic(DityakinParameters.from_dict(data['parameters']),data['density_kg_m3'])
        for expected in data['states']:
            state=curve.state_for_mass(expected['mass_kg_s'])
            for key,value in expected.items():
                with self.subTest(mass=expected['mass_kg_s'],key=key):
                    self.assertAlmostEqual(getattr(state,key)/value,1.,delta=2e-9)
            inverse=curve.hydraulics(expected['pressure_pa']/KGF_CM2_TO_PA)
            self.assertAlmostEqual(inverse.mass_kg_s/expected['mass_kg_s'],1.,delta=2e-9)

    def test_chamber_loss_zero_friction_and_small_friction_limit(self):
        self.assertEqual(chamber_loss(6,0,4),0)
        lam=1e-9
        self.assertAlmostEqual(chamber_loss(6,lam,4)/(lam*6**3),1-1/4,delta=1e-7)

    def test_geometry_and_method_inputs_reject_invalid_or_unsupported_values(self):
        for key,value in (('inlet_count',3.5),('inlet_count',7),('cone_angle_deg',60),
            ('chamber_height_mm',2),('inlet_length_mm',.1),('outlet_length_mm',2),
            ('swirl_arm_mm',10),('kinematic_viscosity_mm2_s',0),
            ('outlet_diameter_mm',float('nan')),('inlet_diameter_mm',float('inf'))):
            with self.subTest(key=key,value=value),self.assertRaises(ValueError):
                DityakinParameters(**{key:value})
        params=DityakinParameters().to_dict();del params['inlet_count']
        with self.assertRaises(ValueError):DityakinParameters.from_dict(params)
        cfg=Settings();entry=catalog();entry[0]['method']='invented'
        with self.assertRaises(ValueError):validate_catalog(entry,cfg.nozzle_presets)
        entry=catalog();entry[0]['plot_pressure_kgf_cm2']=[30,2.5]
        with self.assertRaises(ValueError):validate_catalog(entry,cfg.nozzle_presets)

    def test_pressure_bounds_and_critical_section_cannot_be_overridden(self):
        curve=DityakinCharacteristic(DityakinParameters(),800)
        for pressure in (0,-1,float('nan'),.00001,1e8):
            with self.subTest(pressure=pressure),self.assertRaises(ValueError):
                curve.evaluate(pressure,allow_extrapolation=True)
        short=DityakinCharacteristic(replace(curve.parameters,outlet_length_mm=.25),800)
        with self.assertRaisesRegex(PressureOutOfRange,'4 толщин'):
            short.evaluate(2.5)
        six=DityakinCharacteristic(DityakinParameters(inlet_count=6,inlet_diameter_mm=.3,
            swirl_arm_mm=1.25,chamber_height_mm=.3,inlet_length_mm=.6),800)
        # Within the one-inlet graph but outside the total-inlet friction graph.
        mass=800*math.pi*(.0003/2)**2*6*(45000*1.7e-6/.0003)
        with self.assertRaisesRegex(PressureOutOfRange,'рис.28'):six.state_for_mass(mass)

    def test_pressure_ratio_varies_from_constant_mu_and_geometry_viscosity_matter(self):
        curve=DityakinCharacteristic(DityakinParameters(),800)
        ratio=curve.evaluate(30).flow_kg_h/curve.evaluate(2.5).flow_kg_h
        self.assertGreater(abs(ratio-math.sqrt(12)),.1)
        thick=DityakinCharacteristic(replace(curve.parameters,kinematic_viscosity_mm2_s=3.4),800)
        wide=DityakinCharacteristic(replace(curve.parameters,outlet_diameter_mm=1.2),800)
        self.assertNotAlmostEqual(curve.evaluate(30).flow_kg_h,thick.evaluate(30).flow_kg_h)
        self.assertNotAlmostEqual(curve.evaluate(30).flow_kg_h,wide.evaluate(30).flow_kg_h)

    def test_density_and_unit_scaling_without_changing_viscosity(self):
        p=DityakinParameters();c=DityakinCharacteristic(p,800)
        d=DityakinCharacteristic(p,1000)
        # Keep kinematic viscosity AND volume velocity fixed: pressure scales rho,
        # mass scales rho. This detects an accidental dynamic-viscosity treatment.
        s=c.hydraulics(10);t=d.hydraulics(12.5)
        self.assertAlmostEqual(t.mass_kg_s/s.mass_kg_s,1.25,places=9)
        self.assertAlmostEqual(t.mu,s.mu,places=10)
        self.assertEqual(c.evaluate(10).flow_kg_h,s.mass_kg_s*3600)

    def test_new_project_json_three_flow_modes_balance_and_preserve_measurements(self):
        cfg=Settings(atomizer_types=catalog(),pressure_mode='calibrated',n_sample_radial=8,n_sample_angular=72)
        ns=default_nozzles(cfg)
        for n in ns:n.enabled=False;n.atomizer_type_id='hydraulic'
        curve=DityakinCharacteristic(DityakinParameters(),800)
        ns[0].enabled=True;ns[0].flow_mode='curve';ns[0].flow_kg_h=123
        ns[1].enabled=True;ns[1].flow_mode='sectors';ns[1].capture_efficiency=.8
        ns[1].sector_ml=[10.]*36;ns[1].flow_kg_h=456
        ns[70].enabled=True;ns[70].flow_mode='manual';ns[70].flow_kg_h=12
        ns[70].flow_pressure_kgf_cm2=2.5
        expected_sector=360e-6*800/180/.8*curve.evaluate(30).flow_kg_h/curve.evaluate(2.5).flow_kg_h
        self.assertAlmostEqual(source_flow_kg_s(ns[1],cfg),expected_sector)
        self.assertIn('Дитякин',flow_status(ns[0],cfg).source)
        data=project_to_json(cfg,ns);new_cfg,new_ns=project_from_json(json.loads(json.dumps(data)))
        self.assertEqual(project_to_json(new_cfg,new_ns),data)
        for z in cfg.z_mm:
            plane=calculate_plane(new_ns,new_cfg,z)
            self.assertLess(abs(plane.rel_balance_error),1e-12)
        self.assertEqual([n.flow_kg_h for n in new_ns[:2]],[123,456])
        self.assertEqual(new_ns[1].sector_ml,[10.]*36)


from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication,QMessageBox,QFileDialog,QTableWidgetSelectionRange
from PyQt5.QtTest import QTest
from fuelfield.gui import MainWindow,INTCOL
from fuelfield.editors import HYDRAULIC_LABELS


class TestDityakinQt(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.app=QApplication.instance() or QApplication([])
    def setUp(self):self.w=MainWindow();self.w.preview_timer.stop()
    def tearDown(self):
        self.w.preview_timer.stop();self.w.close();self.w.deleteLater();self.app.processEvents()
    def create_type(self):
        e=self.w.catalog_editor;e.new_type();e.name.setText('Дитякин — проверка')
        e.kind.setCurrentIndex(e.kind.findData('theoretical'))
        with patch.object(QMessageBox,'warning') as warning:e.apply()
        self.assertEqual(warning.call_count,0)
        return e

    def test_editor_theoretical_crud_graph_precision_undo_and_protection(self):
        e=self.create_type();before=deepcopy(self.w.cfg.atomizer_types)
        self.assertEqual(before[0]['kind'],'theoretical');self.assertNotIn('points',before[0])
        self.assertIn('Aₑ',e.status.text());self.assertEqual(len(e.figure.axes[0].lines),1)
        self.assertFalse(e.point_actions[0].isEnabled())
        for i in range(6):e.unit.setCurrentIndex(i%2)
        e.apply();self.assertEqual(self.w.cfg.atomizer_types,before)
        row=list(HYDRAULIC_LABELS).index('kinematic_viscosity_mm2_s')
        e.theory_table.item(row,1).setText('3.4');e.apply()
        self.assertEqual(self.w.cfg.atomizer_types[0]['parameters']['kinematic_viscosity_mm2_s'],3.4)
        self.w.undo_stack.undo();self.assertEqual(self.w.cfg.atomizer_types,before)
        self.w.undo_stack.redo();self.assertEqual(self.w.cfg.atomizer_types[0]['parameters']['kinematic_viscosity_mm2_s'],3.4)
        self.w.nozzles[0].atomizer_type_id=before[0]['id']
        with patch.object(QMessageBox,'warning') as warning:e.delete_type()
        self.assertEqual(warning.call_count,1)

    def test_invalid_geometry_apply_is_atomic_and_shows_domain_error(self):
        e=self.create_type();before=deepcopy(self.w.cfg.atomizer_types)
        row=list(HYDRAULIC_LABELS).index('inlet_count');e.theory_table.item(row,1).setText('2.5')
        with patch.object(QMessageBox,'warning') as warning:e.apply()
        self.assertEqual(warning.call_count,1);self.assertEqual(self.w.cfg.atomizer_types,before)
        e.theory_table.item(row,1).setText('3');e.plot_max.setValue(100000)
        self.assertIsNone(e.curve);self.assertIn('вне диапазона',e.status.text())

    def test_hidden_measured_points_survive_theoretical_units_and_density(self):
        e=self.w.catalog_editor;e.new_type();e.name.setText('Измерения')
        e.fill_points([(2.5,2.),(30.,24.)]);e.kind.setCurrentIndex(1)
        e.unit.setCurrentIndex(1);self.w.cfg.density_kg_m3=1000;e.refresh_density()
        e.kind.setCurrentIndex(0)
        self.assertEqual(e.read_curve().points,((2.5,2.),(30.,24.)))
        self.assertAlmostEqual(float(e.table.item(1,1).text()),.4)

    def test_assign_save_reopen_real_worker_and_density_recompute(self):
        e=self.create_type();ident=self.w.cfg.atomizer_types[0]['id']
        table=self.w.nozzle_table;table.setCurrentCell(0,INTCOL['flow_mode'])
        table.setRangeSelected(QTableWidgetSelectionRange(0,INTCOL['flow_mode'],0,INTCOL['flow_mode']),True)
        self.w.assign_type_selector.setCurrentIndex(1);self.w.assign_atomizer()
        table.item(0,INTCOL['flow_mode']).setText('curve')
        self.assertEqual(self.w.nozzles[0].atomizer_type_id,ident)
        old=source_flow_kg_s(self.w.nozzles[0],self.w.cfg)
        self.w.density_spin.setValue(1000);self.w._record_edit()
        self.assertNotAlmostEqual(source_flow_kg_s(self.w.nozzles[0],self.w.cfg),old)
        self.assertIn('ρ=1000',e.status.text())
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'theory.json';self.w.project_file=str(path);self.w.save_project()
            with patch.object(QFileDialog,'getOpenFileName',return_value=(str(path),'')):self.w.open_project()
        self.assertEqual(self.w.catalog_editor.kind.currentData(),'theoretical')
        errors=[]
        with patch.object(self.w,'_error',side_effect=errors.append):
            self.w.run_calc();deadline=time.monotonic()+15
            while self.w.thread is not None and time.monotonic()<deadline:
                self.app.processEvents();QTest.qWait(10)
        self.assertIsNone(self.w.thread);self.assertEqual(errors,[]);self.assertEqual(len(self.w.results),4)
        for plane in self.w.results:self.assertLess(abs(plane.rel_balance_error),1e-12)


if __name__=='__main__':unittest.main()
