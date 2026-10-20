"""Native endpoint linking, fixed geometry and the reported arc/outlet workflow."""
from copy import deepcopy
from unittest.mock import patch
import unittest
import numpy as np
from PyQt5.QtCore import Qt,QPointF
from PyQt5.QtGui import QMouseEvent
from PyQt5.QtTest import QTest
from PyQt5.QtWidgets import QApplication,QDialog
from fuelfield.chamber import Chamber,Primitive
from fuelfield.core import project_to_json
from fuelfield.chamber_constraints_gui import ConstraintDialog
import test_v06_gui as previous


def ref(identity,role):return dict(primitive_id=identity,endpoint=role)


class TestV062Sketcher(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        previous.TestV06GeometryUI.setUpClass();cls.app=previous.TestV06GeometryUI.app
    setUp=previous.TestV06GeometryUI.setUp
    tearDown=previous.TestV06GeometryUI.tearDown
    record=previous.TestV06GeometryUI.record
    row=previous.TestV06GeometryUI.row
    edit_number=previous.TestV06GeometryUI.edit_number
    handle=previous.TestV06GeometryUI.handle

    def click_point(self,identity,role):
        handle=next(h for h in self.e.handles if h.identity==identity and h.role==role)
        QTest.mouseClick(self.e.sketch.viewport(),Qt.LeftButton,pos=self.e.sketch.mapFromScene(handle.pos()));self.app.processEvents()

    def test_two_click_coincidence_and_visible_deletion_really_unlink(self):
        before=project_to_json(self.w.cfg,self.w.nozzles)
        self.e.start_constraint('coincident');self.click_point('reference-line','start');self.click_point('reference-arc','start')
        np.testing.assert_allclose(self.record('reference-line')['start'],self.record('reference-arc')['start'])
        self.assertEqual(self.e.constraint_table.rowCount(),1);self.assertIsNone(self.e.constraint_mode)
        self.assertEqual(self.w.undo_stack.count(),1);linked=project_to_json(self.w.cfg,self.w.nozzles)
        self.e.pages.setCurrentIndex(4);self.e.constraint_table.selectRow(0);self.e.delete_constraints()
        self.assertEqual(self.e.raw['constraints'],[]);arc=deepcopy(self.record('reference-arc'))
        self.e.pages.setCurrentIndex(0);self.edit_number(self.e.primitive_table,'reference-line',3,'6.125')
        self.assertEqual(self.record('reference-arc'),arc);self.assertEqual(self.record('reference-line')['start'][0],6.125)
        self.w.undo_stack.undo();self.w.undo_stack.undo();self.assertEqual(project_to_json(self.w.cfg,self.w.nozzles),linked)
        self.w.undo_stack.undo();self.assertEqual(project_to_json(self.w.cfg,self.w.nozzles),before)

    def add_tangent_line(self):
        self.record('reference-arc')['radius']=Primitive(self.record('reference-arc')).radius
        line=dict(id='T1',wall='reference',kind='line',start=[15.,260.],end=[25.,265.],unit='mm',provenance={'keep':'operator'})
        self.e.raw['primitives'].append(line);self.assertTrue(self.e.commit('Synthetic tangent line'));self.w.undo_stack.clear();return deepcopy(line)

    def test_endpoint_edit_after_explicit_radius_has_no_stale_derived_radius(self):
        self.edit_number(self.e.primitive_table,'reference-arc',9,'12')
        self.edit_number(self.e.primitive_table,'reference-arc',3,'6.125')
        self.assertFalse(self.e._draft_invalid);arc=Primitive(self.record('reference-arc'));self.assertNotIn('radius',self.record('reference-arc'))
        self.assertAlmostEqual(np.linalg.norm(arc.start-arc.center),np.linalg.norm(arc.end-arc.center),places=10);Chamber(self.w.cfg.geometry)

    def test_native_endpoint_drag_after_radius_entry_stays_applied(self):
        self.edit_number(self.e.primitive_table,'reference-arc',9,'12')
        previous.TestV06GeometryUI.drag(self,'reference-arc',[6.,259.])
        self.assertFalse(self.e._draft_invalid);self.assertNotIn('radius',self.record('reference-arc'));Chamber(self.w.cfg.geometry)
        actual=Primitive(self.record('reference-arc'));handle=self.handle('reference-arc')
        np.testing.assert_allclose([handle.pos().x(),-handle.pos().y()],actual.start,rtol=0,atol=1e-10)
        self.assertEqual(self.w.undo_stack.count(),2)

    def test_tangent_dialog_adjusts_arc_and_fixes_line_without_toggle(self):
        line=self.add_tangent_line();old_arc=deepcopy(self.record('reference-arc'));dialog=ConstraintDialog(self.e.raw,'tangent',self.e)
        dialog.a.setCurrentIndex(dialog.a.findData('reference-arc'));dialog.b.setCurrentIndex(dialog.b.findData('T1'))
        with patch('fuelfield.chamber_gui.ConstraintDialog',return_value=dialog),patch.object(dialog,'exec_',return_value=QDialog.Accepted):self.e.constraint_dialog('tangent')
        actual=self.record('T1');self.assertTrue(actual['fixed'])
        self.assertEqual({k:v for k,v in actual.items() if k!='fixed'},line)
        arc=Primitive(self.record('reference-arc'));np.testing.assert_allclose(arc.center,[10.,270.]);np.testing.assert_allclose(arc.start,old_arc['start'])
        self.assertFalse(self.e._draft_invalid);Chamber(self.w.cfg.geometry);self.assertEqual(self.w.undo_stack.count(),1)

    def test_removing_tangency_reenables_derived_cells_and_radius_edit(self):
        self.add_tangent_line();self.assertTrue(self.e.create_constraint('tangent',ref('reference-arc','end'),ref('T1','start')))
        row=self.row(self.e.primitive_table,'reference-arc');self.assertFalse(self.e.primitive_table.item(row,9).flags()&Qt.ItemIsEditable)
        relation=next(i for i,c in enumerate(self.e.raw['constraints']) if c['kind']=='tangent')
        self.e.pages.setCurrentIndex(4);self.e.constraint_table.selectRow(relation);self.e.delete_constraints()
        self.assertTrue(self.e.primitive_table.item(row,9).flags()&Qt.ItemIsEditable)
        self.e.pages.setCurrentIndex(0);self.edit_number(self.e.primitive_table,'reference-arc',9,'12')
        self.assertAlmostEqual(Primitive(self.record('reference-arc')).radius,12.);self.assertTrue(self.record('T1')['fixed']);self.assertFalse(self.e._draft_invalid)

    def test_fixed_checkbox_blocks_drag_and_table_but_allows_display_units(self):
        row=self.row(self.e.primitive_table,'reference-line');self.e.primitive_table.item(row,12).setCheckState(Qt.Checked)
        self.assertEqual(self.e.primitive_table.horizontalHeader().visualIndex(12),1)
        self.assertTrue(self.record('reference-line')['fixed']);before=project_to_json(self.w.cfg,self.w.nozzles);count=self.w.undo_stack.count()
        self.assertFalse(self.e.primitive_table.item(row,3).flags()&Qt.ItemIsEditable)
        handle=next(h for h in self.e.handles if h.identity=='reference-line' and h.role=='start');view=self.e.sketch;start=view.mapFromScene(handle.pos());target=start+view.mapFromScene(QPointF(3,-2))-view.mapFromScene(QPointF(0,0))
        QTest.mousePress(view.viewport(),Qt.LeftButton,pos=start);event=QMouseEvent(QMouseEvent.MouseMove,QPointF(target),QPointF(view.viewport().mapToGlobal(target)),Qt.NoButton,Qt.LeftButton,Qt.NoModifier);QApplication.sendEvent(view.viewport(),event);QTest.mouseRelease(view.viewport(),Qt.LeftButton,pos=target)
        self.assertEqual(project_to_json(self.w.cfg,self.w.nozzles),before);self.assertEqual(self.w.undo_stack.count(),count)
        self.e.primitive_table.item(row,3).setText('6');self.assertEqual(project_to_json(self.w.cfg,self.w.nozzles),before)
        original=Primitive(self.record('reference-line'));self.e.primitive_table.item(row,11).setText('cm');current=Primitive(self.record('reference-line'))
        np.testing.assert_allclose(current.start,original.start);np.testing.assert_allclose(current.end,original.end);self.assertTrue(self.record('reference-line')['fixed'])

    def test_reported_outer_arc_is_repaired_once_by_radius_and_committed(self):
        raw=self.e.raw;raw['outlet_z_mm']=60.;raw['sections']=[dict(id='S1',z_mm=0.,is_outlet=False),dict(id='S2',z_mm=60.,is_outlet=True)]
        raw['primitives']=[dict(id='ВН1',wall='inner',kind='line',start=[0.,210.],end=[60.,180.],unit='mm'),dict(id='НР1',wall='outer',kind='line',start=[15.,300.],end=[60.,320.],unit='mm')]
        from fuelfield.chamber_gui import GeometryEditor
        # Exact endpoints/centre, but a stale, contradictory radius as on the screenshot.
        arc=dict(id='ОП1',wall='outer',kind='arc',start=[15.,300.],end=[0.,280.],unit='mm',**GeometryEditor.arc_data([15.,300.],[0.,280.]))
        arc['radius']=30.;raw['primitives'].append(arc);self.e.fill_tables();self.e.draw_scene();self.assertFalse(self.e.commit('Broken radius'))
        self.e.primitive_table.setCurrentCell(2,9);self.e.repair_arc();self.app.processEvents()
        self.assertFalse(self.e._draft_invalid);self.assertEqual(self.w.cfg.geometry['enabled'],True)
        model=Chamber(self.w.cfg.geometry);self.assertAlmostEqual(Primitive(self.record('ОП1')).radius,30.)
        np.testing.assert_allclose(model.bounds(0.),(210.,280.),rtol=0,atol=1e-10);self.assertTrue(all(path.pen().style()==Qt.SolidLine for path in self.e.paths.values()))

    def test_new_farthest_section_updates_outlet_and_preserves_existing_curves(self):
        original=deepcopy(self.e.raw['primitives']);self.e.add_section();identity=self.e.raw['sections'][-1]['id']
        self.e.pages.setCurrentIndex(3);self.edit_number(self.e.section_table,identity,1,'60')
        self.assertEqual(self.e.outlet.value(),60.);self.assertEqual(self.w.cfg.z_mm,[0.,30.,60.]);self.assertEqual(self.e.raw['primitives'][:len(original)],original)
        self.assertEqual(len(self.e.raw['primitives']),len(original)+2);self.assertTrue(self.e.raw['sections'][-1]['is_outlet']);Chamber(self.w.cfg.geometry)
        self.e.outlet.setValue(75.);self.e.outlet.editingFinished.emit();self.assertEqual(self.w.cfg.z_mm[-1],75.);self.assertEqual(self.w.cfg.geometry['outlet_z_mm'],75.)

    def test_internal_section_does_not_move_existing_walls(self):
        original=deepcopy(self.e.raw['primitives']);self.e.add_section();identity=self.e.raw['sections'][-1]['id'];self.e.pages.setCurrentIndex(3)
        self.edit_number(self.e.section_table,identity,1,'17.375');self.assertEqual(self.e.raw['primitives'],original);self.assertEqual(self.e.outlet.value(),30.)

    def test_split_rebinds_old_endpoint_and_removes_links_on_delete(self):
        self.assertTrue(self.e.create_constraint('coincident',ref('reference-line','end'),ref('reference-arc','start')))
        self.e.primitive_table.setCurrentCell(self.row(self.e.primitive_table,'reference-line'),0);self.e.split_primitive();Chamber(self.w.cfg.geometry)
        old_end_link=next(c for c in self.e.raw['constraints'] if c['b']==ref('reference-arc','start'))
        self.assertNotEqual(old_end_link['a']['primitive_id'],'reference-line');second=old_end_link['a']['primitive_id']
        self.e.primitive_table.selectRow(self.row(self.e.primitive_table,second));self.e.delete(self.e.primitive_table)
        self.assertTrue(all(c[k]['primitive_id']!=second for c in self.e.raw['constraints'] for k in ('a','b')));Chamber(self.w.cfg.geometry)
