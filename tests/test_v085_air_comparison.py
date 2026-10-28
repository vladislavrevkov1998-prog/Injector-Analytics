"""The comparison column keeps its own air inputs, masks, time and axes."""
from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace
import unittest

import numpy as np
from PyQt5.QtCore import QCoreApplication,QEvent
from PyQt5.QtWidgets import QApplication,QComboBox,QDoubleSpinBox,QVBoxLayout,QWidget

from fuelfield.comparison import ComparisonPanel,ProjectWorker
from fuelfield.coordinates import grid_angles
from fuelfield.core import Settings,PlaneResult,default_nozzles
from fuelfield.quasi import CalculationOutput,PressureChannel,QuasiSettings
from fuelfield.unified_results import AIR_METRICS,build_air_results


def fixture(composite=False):
    cfg=Settings(n_radial=2,n_angular=8,z_mm=[20.],n_sample_radial=8,n_sample_angular=72)
    cfg.air_model=dict(enabled=True,total_kg_s=.2,l0=2.,distribution='uniform',
        max_path_mm=40.,sources=[dict(id='SYNTHETIC_AIR',kind='hole',method='fixed',
        flow_kg_s=.2,z_mm=0.,radius_mm=250.,theta_deg=0.,direction=[1.,0.,0.],mixed_fraction=.25)])
    nozzles=default_nozzles(cfg)
    centers,edges=cfg.radial_centers_and_edges(20.)
    area=np.repeat((np.pi*np.diff(edges**2)/cfg.n_angular*1e-6)[:,None],cfg.n_angular,axis=1)
    fuel=np.full(area.shape,.01)
    plane=PlaneResult(20.,fuel,area,fuel/area,centers,edges,grid_angles(cfg),
                      float(fuel.sum()),float(fuel.sum()),0.,0.,is_composite=composite)
    return cfg,nozzles,plane


class Owner(QWidget):
    def __init__(self):
        super().__init__()
        self.cfg=Settings();self.results=[]
        self.metric=QComboBox();self.metric.addItem('Fuel flux','flux')
        for key,(label,_) in AIR_METRICS.items():self.metric.addItem(label,key)
        self.volume_interval=QDoubleSpinBox();self.volume_interval.setRange(0.,1000.);self.volume_interval.setValue(180.)
        self.x_axis=QComboBox();self.x_axis.addItem('Degrees','angle')
        self.time_slider=SimpleNamespace(value=lambda:1)
        self.plane_select=QComboBox();self.compare_radius=QComboBox()
        # Deliberately inaccessible: comparison must use its own result set.
        self.air_results=SimpleNamespace(state=lambda *a,**k:(_ for _ in ()).throw(AssertionError('primary air used')))
        self.panel=ComparisonPanel(self)
        layout=QVBoxLayout(self);layout.addWidget(self.panel)
        self.metric.currentIndexChanged.connect(self.panel.draw)


class TestAirComparison(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.app=QApplication.instance() or QApplication([])

    def setUp(self):
        self.owner=Owner();self.panel=self.owner.panel

    def tearDown(self):
        self.owner.close();self.owner.deleteLater()
        QCoreApplication.sendPostedEvents(None,QEvent.DeferredDelete);self.app.processEvents()

    def load(self,cfg,nozzles,planes,air_results=None,quasi_result=None):
        output=CalculationOutput(planes,quasi_result)
        output.air_results=air_results
        self.panel._loading_cfg=cfg;self.panel._loading_ns=nozzles
        self.panel._project_done(output)

    def select(self,key):self.owner.metric.setCurrentIndex(self.owner.metric.findData(key))

    def test_worker_computes_own_air_before_emitting_the_completed_project(self):
        cfg,nozzles,_=fixture()
        for n in nozzles:n.enabled=False
        n=nozzles[0];n.enabled=True;n.flow_kg_h=360.;n.alpha_deg=0.;n.k_linear=0.;n.h0_mm=1.;n.rc0_mm=5.
        outputs=[];errors=[]
        worker=ProjectWorker(cfg,nozzles)
        worker.completed.connect(outputs.append);worker.failed.connect(errors.append)
        worker.run()
        self.assertEqual(errors,[]);self.assertEqual(len(outputs),1)
        output=outputs[0]
        self.assertEqual(len(output),1);self.assertIsNotNone(output.air_results)
        plane,mixture=output.air_results.state(0)
        self.assertIs(plane,output[0])
        self.assertAlmostEqual(mixture.air_total_kg_s.sum(),.2)
        self.assertAlmostEqual(mixture.air_mixed_kg_s.sum(),.05)
        self.assertAlmostEqual(mixture.alpha_section_total,.2/(2.*plane.cell_mass_kg_s.sum()))

    def test_alpha_epures_use_secondary_mass_and_l0_instead_of_primary_data(self):
        cfg,nozzles,plane=fixture();fuel_before=plane.cell_mass_kg_s.copy()
        result=build_air_results(cfg,nozzles,[plane])
        self.load(cfg,nozzles,[plane],result);self.select('alpha_total')
        expected=.2*plane.cell_area_m2/plane.cell_area_m2.sum()/(2.*plane.cell_mass_kg_s)
        ax=self.panel.figure.axes[0]
        np.testing.assert_allclose(ax.lines[0].get_ydata(),expected[0])
        self.assertEqual(ax._compare_y,'1')
        self.assertIn('второго проекта',self.panel.notice.text())
        np.testing.assert_array_equal(plane.cell_mass_kg_s,fuel_before)

    def test_quasi_worker_emits_air_with_composite_and_real_snapshot_semantics(self):
        cfg,nozzles,_=fixture();cfg.quasistationary=True
        cfg.quasi_settings=QuasiSettings(n_steps=8,
            channels=(PressureChannel('A',pressure=(30.,)*8),)).to_dict()
        for n in nozzles:n.enabled=False
        n=nozzles[0];n.enabled=True;n.flow_kg_h=360.;n.alpha_deg=0.;n.k_linear=0.;n.h0_mm=1.;n.rc0_mm=5.
        outputs=[];errors=[];worker=ProjectWorker(cfg,nozzles)
        worker.completed.connect(outputs.append);worker.failed.connect(errors.append);worker.run()
        self.assertEqual(errors,[]);self.assertEqual(len(outputs),1)
        output=outputs[0]
        self.assertIsNotNone(output.quasi_result)
        self.assertIs(output.air_results.quasi_result,output.quasi_result)
        composite,mixture=output.air_results.state(0)
        self.assertTrue(composite.is_composite);self.assertTrue(np.isnan(mixture.alpha_section_total))
        snapshot,instant=output.air_results.state(0,step=0)
        self.assertFalse(snapshot.is_composite)
        self.assertAlmostEqual(instant.air_total_kg_s.sum(),.2)
        self.assertAlmostEqual(instant.alpha_section_total,.2/(2.*snapshot.cell_mass_kg_s.sum()))

    def test_invalid_air_budget_does_not_emit_a_partial_fuel_only_project(self):
        cfg,nozzles,_=fixture()
        for n in nozzles:n.enabled=False
        n=nozzles[0];n.enabled=True;n.flow_kg_h=360.;n.alpha_deg=0.;n.k_linear=0.;n.h0_mm=1.;n.rc0_mm=5.
        cfg.air_model['sources'][0]['flow_kg_s']=.3
        outputs=[];errors=[];worker=ProjectWorker(cfg,nozzles)
        worker.completed.connect(outputs.append);worker.failed.connect(errors.append);worker.run()
        self.assertEqual(outputs,[]);self.assertEqual(len(errors),1)

    def test_air_missing_in_loaded_project_never_falls_back_to_fuel(self):
        cfg,nozzles,plane=fixture();cfg.air_model['enabled']=False
        self.load(cfg,nozzles,[plane]);self.select('alpha_total')
        for view in (0,1,2,1):
            with self.subTest(view=view):
                self.panel.view.setCurrentIndex(view);self.panel.draw()
                self.assertIn('Во втором проекте нет рассчитанных',self.panel.notice.text())
                self.assertEqual(len(self.panel.figure.axes[0].lines),0)
                self.assertEqual(len(self.panel.figure.axes[0].collections),0)

    def test_quasi_polar_snapshot_uses_own_step_zero_and_real_comb_angles(self):
        cfg,nozzles,composite=fixture(composite=True)
        snapshot=replace(composite,cell_mass_kg_s=2.*composite.cell_mass_kg_s,
                         flux_kg_m2_s=2.*composite.flux_kg_m2_s,is_composite=False)
        calls=[]
        def snapshot_at(index,step):calls.append((index,step));return snapshot
        quasi=SimpleNamespace(settings=SimpleNamespace(n_steps=2,times=[9.,11.],
            global_angles=lambda cfg:np.array([0.,90.])),snapshot=snapshot_at)
        result=build_air_results(cfg,nozzles,[composite],quasi)
        self.load(cfg,nozzles,[composite],result,quasi)
        self.select('alpha_total');self.panel.view.setCurrentIndex(2)
        ax=self.panel.figure.axes[0]
        expected=.2*snapshot.cell_area_m2/snapshot.cell_area_m2.sum()/(2.*snapshot.cell_mass_kg_s)
        np.testing.assert_allclose(ax.collections[0].get_array(),expected)
        self.assertEqual(calls,[(0,0)])
        self.assertEqual(ax.name,'polar');self.assertIn('шаг 0, t=9 с',ax.get_title())
        rings=[line for line in ax.lines if line.get_gid()=='nozzle-row-ring']
        self.assertEqual({float(line.get_ydata()[0]) for line in rings},{230.,270.})
        comb=next(item for item in ax.collections if item.get_gid()=='comb-pass-positions')
        np.testing.assert_allclose(np.asarray(comb.get_offsets())[:,0],[0.,np.pi/2])

    def test_replacing_only_air_results_cannot_reuse_the_previous_alpha_chart(self):
        cfg,nozzles,plane=fixture()
        result=build_air_results(cfg,nozzles,[plane]);self.load(cfg,nozzles,[plane],result)
        self.select('alpha_total');old_canvas=self.panel.canvas
        old_values=np.array(self.panel.figure.axes[0].lines[0].get_ydata(),copy=True)
        changed=deepcopy(cfg);changed.air_model['total_kg_s']=.4;changed.air_model['sources'][0]['flow_kg_s']=.4
        self.panel.air_results=build_air_results(changed,nozzles,[plane]);self.panel.draw()
        self.assertIsNot(self.panel.canvas,old_canvas)
        np.testing.assert_allclose(self.panel.figure.axes[0].lines[0].get_ydata(),2.*old_values)


if __name__=='__main__':unittest.main()
