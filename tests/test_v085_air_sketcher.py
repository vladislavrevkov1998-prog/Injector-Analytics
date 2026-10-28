"""Air-mode edit isolation and actual spatial projections, with native Qt."""
from copy import deepcopy
import sys
import unittest

import numpy as np
from PyQt5.QtCore import Qt, QPointF
from PyQt5.QtGui import QMouseEvent
from PyQt5.QtTest import QTest
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QUndoStack

from fuelfield.core import Settings, default_nozzles, project_to_json, project_from_json
from fuelfield.chamber import legacy_geometry
from fuelfield.chamber_gui import GeometryEditor
from fuelfield.gui import PasteTable
from fuelfield.air_sketcher import project_points, projected_envelopes
from fuelfield.air_jets import cartesian_to_local


def source(identity, theta, direction):
    return dict(id=identity, kind='hole', method='fixed', enabled=True,
                flow_kg_s=1., z_mm=0., radius_mm=250., theta_deg=theta,
                direction=direction, sigma_r_mm=5., sigma_theta_deg=10.,
                spread_mm_per_mm=.02, mixed_fraction=.2,
                provenance={'source_kind':'synthetic acceptance test'})


class Owner(QWidget):
    def __init__(self):
        super().__init__()
        self.cfg=Settings(n_radial=2,n_angular=8,z_mm=[0.,50.,100.])
        self.nozzles=default_nozzles(self.cfg)
        self.cfg.geometry=legacy_geometry(self.cfg,self.nozzles)
        self.cfg.geometry['enabled']=True
        self.cfg.air_model=dict(enabled=True,total_kg_s=2.,l0=15.,step_mm=2.,
            max_path_mm=120.,interaction_strength=2.,interaction_length_mm=30.,
            sources=[source('A',358.,[1.,0.,.4]),source('B',2.,[1.,0.,-.4])])
        self.undo_stack=QUndoStack(self)
        self.air_applies=[]
        self.editor=GeometryEditor(self,PasteTable)
        layout=QVBoxLayout(self);layout.addWidget(self.editor)
        self.editor.load(self.cfg,self.nozzles)
        self.resize(1280,800)

    def apply_geometry_state(self,state,label,record_history=False):
        self.cfg,self.nozzles=project_from_json(state)
        self.editor.load(self.cfg,self.nozzles)

    def apply_air_model(self,raw):
        self.air_applies.append(deepcopy(raw))
        self.cfg.air_model=deepcopy(raw)
        self.editor.refresh_air_model()

    def open_air_balance(self):
        pass


class TestAirSketcher(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app=QApplication.instance() or QApplication([])

    def setUp(self):
        self.qt_errors=[];self.previous_hook=sys.excepthook
        sys.excepthook=lambda kind,value,tb:self.qt_errors.append((kind,value))
        self.owner=Owner();self.owner.show();self.app.processEvents()
        self.e=self.owner.editor
        self.e.fit();self.app.processEvents()

    def tearDown(self):
        try:
            self.owner.close();self.owner.deleteLater();self.app.processEvents()
            self.assertEqual(self.qt_errors,[])
        finally:sys.excepthook=self.previous_hook

    def enter_air(self):
        self.e.mode_selector.setCurrentIndex(self.e.mode_selector.findData('air'))
        self.app.processEvents()
        self.assertTrue(self.e.is_air_mode)
        self.assertEqual(self.e.sketch.property('ffSketchMode'),'air')

    def drag(self, handle, target):
        view=self.e.sketch
        start=view.mapFromScene(handle.pos());end=view.mapFromScene(QPointF(*target))
        QTest.mousePress(view.viewport(),Qt.LeftButton,pos=start)
        event=QMouseEvent(QMouseEvent.MouseMove,QPointF(end),QPointF(view.viewport().mapToGlobal(end)),Qt.NoButton,Qt.LeftButton,Qt.NoModifier)
        QApplication.sendEvent(view.viewport(),event)
        QTest.mouseRelease(view.viewport(),Qt.LeftButton,pos=end)
        self.app.processEvents()

    def test_air_mode_freezes_native_drag_double_click_tables_and_geometry_apis(self):
        before=project_to_json(self.owner.cfg,self.owner.nozzles);raw=deepcopy(self.e.raw)
        self.enter_air()
        handle=next(h for h in self.e.handles if not h.object_handle and h.role=='start')
        self.drag(handle,(handle.pos().x()+12.,handle.pos().y()+7.))
        self.e.tool.setCurrentIndex(1)
        QTest.mouseDClick(self.e.sketch.viewport(),Qt.LeftButton,pos=self.e.sketch.mapFromScene(QPointF(20.,-250.)))
        self.e.add_primitive('point',([20.,250.],[20.,250.]))
        self.e.add_section();self.e.delete(self.e.primitive_table)
        self.e.base.setValue(4.);self.e.base.editingFinished.emit()
        self.e.primitive_table.item(0,3).setText('20')
        self.e.section_table.setFocus();QTest.keyClick(self.e.section_table,Qt.Key_Delete)
        self.app.processEvents()
        self.assertEqual(self.e.raw,raw)
        self.assertEqual(project_to_json(self.owner.cfg,self.owner.nozzles),before)
        self.assertEqual(self.owner.undo_stack.count(),0)
        self.assertEqual(self.e.primitive_table.editTriggers(),self.e.primitive_table.NoEditTriggers)
        self.assertFalse(self.e.primitive_table.item(0,3).flags() & Qt.ItemIsEditable)
        self.e.mode_selector.setCurrentIndex(0);self.app.processEvents()
        self.assertEqual(self.e.sketch.property('ffSketchMode'),'geometry')
        self.assertTrue(self.e.primitive_table.item(0,3).flags() & Qt.ItemIsEditable)
        self.e.add_primitive('point',([20.,250.],[20.,250.]))
        self.assertGreater(len(self.e.raw['primitives']),len(raw['primitives']))

    def test_live_air_width_updates_scene_without_geometry_or_fuel_edits(self):
        before=project_to_json(self.owner.cfg,self.owner.nozzles)
        self.enter_air();panel=self.e.air_panel
        original=[item.path().boundingRect() for item in panel.items if item.data(0)=='A' and item.data(1)=='air-envelope']
        panel.controls['sigma_r_mm'].setText('18');panel.update_draft()
        changed=[item.path().boundingRect() for item in panel.items if item.data(0)=='A' and item.data(1)=='air-envelope']
        self.assertNotEqual(original,changed)
        self.assertEqual(project_to_json(self.owner.cfg,self.owner.nozzles),before)
        panel.apply_button.click();self.app.processEvents()
        self.assertEqual(len(self.owner.air_applies),1)
        after=project_to_json(self.owner.cfg,self.owner.nozzles)
        self.assertEqual(after['nozzles'],before['nozzles'])
        self.assertEqual(after['settings']['geometry'],before['settings']['geometry'])
        self.assertEqual(after['settings']['air_model']['sources'][0]['sigma_r_mm'],18.)

    def test_resultant_vectors_come_from_tracer_in_both_projections(self):
        self.enter_air();panel=self.e.air_panel
        self.assertIsNotNone(panel.paths)
        index=next(i for i,s in enumerate(panel.paths.sources) if s['id']=='A')
        actual=panel.paths.directions[:,index]
        self.assertGreater(np.linalg.norm(actual[len(actual)//2]-actual[0]),1e-5)
        self.assertIn('Результирующее',panel.vector_label.text())
        meridian=[item.path() for item in panel.items if item.data(0)=='A' and item.data(1)=='air-trajectory'][0]
        self.e.angular.setChecked(True);self.app.processEvents()
        angular=[item.path() for item in panel.items if item.data(0)=='A' and item.data(1)=='air-trajectory'][0]
        self.assertNotEqual(meridian.boundingRect(),angular.boundingRect())
        positions=panel.paths.positions[:,index]
        self.assertAlmostEqual(angular.elementAt(0).x,positions[0,1])
        self.assertAlmostEqual(angular.elementAt(0).y,-positions[0,2])
        self.assertTrue(any(item.data(1)=='air-vector' for item in panel.items))

    def test_unknown_flows_only_show_explicit_input_preview(self):
        for s in self.owner.cfg.air_model['sources']:s.update(method='unknown',flow_kg_s=None)
        self.enter_air();panel=self.e.air_panel
        self.assertFalse(any(item.data(1)=='air-vector' for item in panel.items))
        self.assertTrue(any(item.data(1)=='air-input-preview' for item in panel.items))
        self.assertIn('Только направление ввода',panel.vector_label.text())
        self.assertIn('Расчётного расхода',panel.vector_label.text())

    def test_global_uniform_distribution_shows_domain_not_a_gaussian_tube(self):
        self.owner.cfg.air_model['distribution']='uniform'
        self.enter_air();panel=self.e.air_panel
        bounds=[item.path().boundingRect() for item in panel.items if item.data(0)=='A' and item.data(1)=='air-envelope']
        self.assertEqual(sorted(rect.y() for rect in bounds),[-300.,-200.])
        self.assertFalse(panel.controls['sigma_r_mm'].isEnabled())
        self.assertIn('границы тракта',panel.position.text())

    def test_inherited_parameters_create_only_the_edited_override(self):
        self.owner.cfg.nozzle_presets['uncooled']['air']=dict(method='fixed',flow_kg_s=1.,sigma_r_mm=7.)
        self.owner.cfg.air_model['sources']=[dict(id=f'S{i}',kind='swirler',nozzle_id=i,
            template_mode='inherit',air_overrides=[],method='unknown',flow_kg_s=None) for i in (1,2)]
        self.enter_air();panel=self.e.air_panel
        self.assertEqual(panel.controls['flow_kg_s'].text(),'1')
        self.assertEqual(panel.controls['sigma_r_mm'].text(),'7')
        panel.controls['sigma_r_mm'].setText('19');panel.update_draft()
        raw=panel.draft['sources'][0]
        self.assertEqual(raw['air_overrides'],['sigma_r_mm'])
        self.assertIsNone(raw['flow_kg_s'])
        self.assertEqual(raw['sigma_r_mm'],19.)
        panel.apply_button.click()
        self.assertEqual(len(self.owner.air_applies),1)
        self.assertEqual(self.owner.cfg.nozzle_presets['uncooled']['air']['sigma_r_mm'],7.)

    def test_invalid_input_can_be_repaired_without_losing_the_source(self):
        self.enter_air();panel=self.e.air_panel
        panel.controls['sigma_r_mm'].setText('NaN');panel.update_draft()
        self.assertFalse(panel.apply_button.isEnabled())
        panel.controls['sigma_r_mm'].setText('5');panel.update_draft()
        self.assertTrue(panel.apply_button.isEnabled())
        self.assertEqual(panel.selected_raw()['sigma_r_mm'],5.)

    def test_adding_a_nozzle_source_inherits_its_air_template(self):
        self.owner.cfg.air_model['total_kg_s']=3.
        self.owner.cfg.nozzle_presets['uncooled']['air']=dict(method='fixed',flow_kg_s=.4)
        before=deepcopy(self.owner.nozzles)
        self.enter_air();panel=self.e.air_panel
        self.e.select('N3',True);panel.add_button.click();self.app.processEvents()
        self.assertEqual(panel.selected_raw()['nozzle_id'],3)
        self.assertEqual(panel.selected_raw()['template_mode'],'inherit')
        self.assertEqual(panel.controls['flow_kg_s'].text(),'0.4')
        self.assertEqual(self.owner.nozzles,before)

    def test_axial_vector_is_explicit_when_angular_projection_vanishes(self):
        self.owner.cfg.air_model['interaction_strength']=0.
        for s in self.owner.cfg.air_model['sources']:s['direction']=[1.,0.,0.]
        self.enter_air();self.e.angular.setChecked(True);self.app.processEvents()
        panel=self.e.air_panel
        labels=[item for item in panel.items if item.data(1)=='air-vector' and hasattr(item,'text')]
        self.assertTrue(labels)
        self.assertIn('⊙',labels[0].text())
        self.assertIn('[dz, dr, dθ]',panel.vector_label.text())
        for s in self.owner.cfg.air_model['sources']:s.update(z_mm=80.,direction=[-1.,0.,0.])
        self.e.refresh_air_model();self.app.processEvents()
        labels=[item for item in panel.items if item.data(1)=='air-vector' and hasattr(item,'text')]
        self.assertTrue(labels)
        self.assertIn('⊗',labels[0].text())
        self.assertIn('⊗ = −dz',labels[0].toolTip())

    def test_local_geometry_undo_is_frozen_until_mode_returns(self):
        self.e.add_primitive('point',([20.,250.],[20.,250.]))
        current=project_to_json(self.owner.cfg,self.owner.nozzles)
        index=self.owner.undo_stack.index()
        self.enter_air();self.e.undo_geometry()
        self.assertEqual(self.owner.undo_stack.index(),index)
        self.assertEqual(project_to_json(self.owner.cfg,self.owner.nozzles),current)
        self.e.mode_selector.setCurrentIndex(0);self.e.undo_geometry()
        self.assertLess(self.owner.undo_stack.index(),index)


class TestAirProjectionMath(unittest.TestCase):
    def test_projection_uses_cartesian_angular_and_radial_meridional_geometry(self):
        points=np.array([[3.,3.,4.],[4.,0.,6.]])
        np.testing.assert_array_equal(project_points(points),[[3.,-5.],[4.,-6.]])
        np.testing.assert_array_equal(project_points(points,True),[[3.,-4.],[0.,-6.]])

    def test_kernel_width_spread_and_mixed_factor_change_actual_envelopes(self):
        s=source('A',0.,[1.,0.,0.]);points=np.array([[0.,250.,0.],[10.,250.,0.]])
        core=projected_envelopes(points,[0.,10.],s)
        self.assertAlmostEqual(core[0][1,1],-244.8)
        mixed=projected_envelopes(points,[0.,10.],s,mixed=True)
        self.assertAlmostEqual(mixed[0][1,1],-(250.-5.2*np.sqrt(2.)))
        a=projected_envelopes(points,[0.,10.],s,True)
        s['sigma_r_mm']=25.
        b=projected_envelopes(points,[0.,10.],s,True)
        self.assertFalse(np.array_equal(a[1],b[1]))


if __name__=='__main__':unittest.main()
