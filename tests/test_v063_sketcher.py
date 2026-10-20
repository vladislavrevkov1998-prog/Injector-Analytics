"""Native object tangency and draft-only section intersections."""
from copy import deepcopy
import unittest
import numpy as np
from PyQt5.QtCore import Qt,QPointF
from PyQt5.QtTest import QTest
from PyQt5.QtGui import QTextDocument
from fuelfield.chamber import Chamber,Primitive
from fuelfield.chamber_editing import preview
from fuelfield.chamber_constraints import materialize_constraints,add_constraint,validate_constraints
from fuelfield.chamber_preview import draft_bounds
from fuelfield.help_text import HELP_IMAGES
import test_v06_gui as previous


class TestV063Sketcher(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        previous.TestV06GeometryUI.setUpClass();cls.app=previous.TestV06GeometryUI.app
    setUp=previous.TestV06GeometryUI.setUp
    tearDown=previous.TestV06GeometryUI.tearDown
    record=previous.TestV06GeometryUI.record
    row=previous.TestV06GeometryUI.row

    def click_body(self,identity):
        points,_=preview(self.record(identity),.001);p=points[len(points)//2]
        if len(points)==2:p=points.mean(axis=0)
        QTest.mouseClick(self.e.sketch.viewport(),Qt.LeftButton,pos=self.e.sketch.mapFromScene(QPointF(p[0],-p[1])));self.app.processEvents()

    def test_click_curve_bodies_adds_tangent_even_with_existing_coincidence(self):
        materialize_constraints(self.e.raw)
        self.e.raw['primitives'].append(dict(id='T1',wall='reference',kind='line',start=[15.,260.],end=[25.,265.],unit='mm'))
        self.e.raw=add_constraint(self.e.raw,'coincident',dict(primitive_id='reference-arc',endpoint='end'),dict(primitive_id='T1',endpoint='start'))
        self.assertTrue(self.e.commit('synthetic link'));self.e.draw_scene();self.e.fit();self.app.processEvents();self.w.undo_stack.clear()
        self.e.start_constraint('tangent');self.click_body('reference-arc');self.assertEqual(self.e._constraint_first,'reference-arc');self.click_body('T1')
        self.assertIsNone(self.e.constraint_mode);self.assertFalse(self.e._draft_invalid)
        self.assertEqual([c['kind'] for c in self.e.raw['constraints']],['coincident','tangent']);validate_constraints(self.e.raw)
        self.assertTrue(self.record('T1')['fixed']);self.assertEqual(self.w.undo_stack.count(),1)

    def test_arc_arc_body_clicks_preserve_reference_and_leave_it_editable(self):
        a=dict(id='A1',kind='arc',wall='reference',start=[30.,260.],end=[40.,250.],center=[30.,250.],clockwise=True,unit='mm')
        b=dict(id='A2',kind='arc',wall='reference',start=[40.,250.],end=[50.,240.],center=[50.,250.],clockwise=False,unit='mm')
        self.e.raw['primitives'].extend([a,b]);self.assertTrue(self.e.commit('synthetic arcs'));self.e.draw_scene();self.e.fit();self.app.processEvents();before=deepcopy(self.record('A2'))
        self.e.start_constraint('tangent');self.click_body('A1');self.click_body('A2')
        self.assertFalse(self.e._draft_invalid);self.assertIsNone(self.e.constraint_mode);self.assertEqual(self.record('A2'),before);validate_constraints(self.e.raw)
        row=self.row(self.e.primitive_table,'A2');self.assertTrue(self.e.primitive_table.item(row,9).flags()&Qt.ItemIsEditable)
        row=self.row(self.e.primitive_table,'A1');self.assertFalse(self.e.primitive_table.item(row,9).flags()&Qt.ItemIsEditable)

    def test_object_dialog_can_disambiguate_contact_by_explicit_endpoints(self):
        from fuelfield.chamber_constraints_gui import ConstraintDialog
        self.e.raw['primitives'].append(dict(id='T1',kind='line',wall='reference',start=[5.,250.],end=[15.,250.],unit='mm'))
        dialog=ConstraintDialog(self.e.raw,'tangent',self.e)
        dialog.a.setCurrentIndex(dialog.a.findData('reference-arc'));dialog.b.setCurrentIndex(dialog.b.findData('T1'))
        with self.assertRaisesRegex(ValueError,'неоднозначен'):dialog.references()
        dialog.manual.setChecked(True);dialog.a_endpoint.setCurrentIndex(1)
        self.assertEqual(dialog.references(),(dict(primitive_id='reference-arc',endpoint='end'),dict(primitive_id='T1',endpoint='start')))
        self.assertTrue(dialog.a_endpoint.isEnabled());dialog.close()

    def test_invalid_profile_sections_use_drawn_wall_intersections(self):
        raw=deepcopy(self.e.raw);raw['primitives']=[dict(id='i',kind='line',wall='inner',start=[0.,220.],end=[30.,230.],unit='mm'),dict(id='o',kind='line',wall='outer',start=[0.,280.],end=[30.,270.],unit='mm')]
        # Coverage error at outlet60 must not replace valid local radii by cfg.
        raw['outlet_z_mm']=60;self.e.raw=raw;self.e.draw_scene()
        for section in raw['sections']:
            bounds=draft_bounds(raw,section['z_mm']);self.assertIsNotNone(bounds)
            line=self.e.section_lines[section['id']].line();np.testing.assert_allclose([-line.y1(),-line.y2()],bounds)
        self.assertIsNone(draft_bounds(raw,45))

    def test_s_shaped_profile_is_accepted_with_stale_exit_and_keeps_all_curves(self):
        import sys
        from pathlib import Path
        sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
        from create_v063_example import profile
        geometry=profile();geometry['outlet_z_mm']=60.
        geometry['sections'][3]['is_outlet']=True;geometry['sections'][-1]['is_outlet']=False
        for key in ('primitives','sections','outlet_z_mm'):self.e.raw[key]=geometry[key]
        self.e.raw['constraints']=[];before=deepcopy(self.e.raw['primitives'])
        self.assertTrue(self.e.commit('S-shaped profile with old exit'))
        self.assertEqual(self.e.raw['outlet_z_mm'],160.);self.assertEqual(self.e.raw['primitives'],before)
        self.assertEqual(self.e.outlet.value(),160.);self.assertEqual(self.w.cfg.z_mm[-1],160.)
        model=Chamber(self.e.raw)
        for section in geometry['sections']:
            line=self.e.section_lines[section['id']].line()
            np.testing.assert_allclose([-line.y1(),-line.y2()],model.bounds(section['z_mm']))

    def test_reverse_major_arc_is_explicitly_repaired_without_moving_control_points(self):
        p=self.record('reference-arc');before=deepcopy(p);p['clockwise']=not p['clockwise'];self.assertFalse(self.e.commit('wrong direction'))
        self.e.primitive_table.setCurrentCell(self.row(self.e.primitive_table,'reference-arc'),0);self.e.repair_arc_direction()
        self.assertFalse(self.e._draft_invalid);self.assertEqual(self.record('reference-arc'),before);Primitive(self.record('reference-arc'))

    def test_help_has_renderable_embedded_arrow_diagrams(self):
        from PyQt5.QtWidgets import QTextBrowser
        from PyQt5.QtCore import QUrl
        self.e.show_guide();browser=self.e.guide_dialog.findChild(QTextBrowser)
        for relative in HELP_IMAGES:
            resource=browser.document().resource(QTextDocument.ImageResource,QUrl(relative))
            self.assertIsNotNone(resource);self.assertFalse(resource.isNull());self.assertGreater(resource.width(),600)
        self.assertIn('не действующий оператор',browser.toPlainText().lower());self.e.guide_dialog.close()
