"""Native constrained dragging through a bad arc and creation of a true biarc."""
from copy import deepcopy
import unittest
import numpy as np
from PyQt5.QtCore import Qt,QPointF
from PyQt5.QtGui import QMouseEvent
from PyQt5.QtTest import QTest
from PyQt5.QtWidgets import QApplication
from fuelfield.chamber import Primitive,UNITS
from fuelfield.chamber_constraints import add_constraint,materialize_constraints,validate_constraints
from fuelfield.core import project_to_json
from test_v064_constraints import pair
import test_v06_gui as previous


class TestV064ConstrainedGestures(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        previous.TestV06GeometryUI.setUpClass();cls.app=previous.TestV06GeometryUI.app
    setUp=previous.TestV06GeometryUI.setUp
    tearDown=previous.TestV06GeometryUI.tearDown
    record=previous.TestV06GeometryUI.record

    def check_handles(self):
        for h in self.e.handles:
            if h.object_handle:continue
            raw=self.record(h.identity);expected=np.array(raw[h.role])*UNITS[raw.get('unit','mm')]
            np.testing.assert_allclose([h.pos().x(),-h.pos().y()],expected,atol=1e-9,err_msg=h.identity+'.'+h.role)
        validate_constraints(self.e.raw)

    def test_coincidence_cannot_tear_when_drag_snaps_arc_end_to_its_opposite_end(self):
        materialize_constraints(self.e.raw)
        self.e.raw=add_constraint(self.e.raw,'coincident',dict(primitive_id='reference-line',endpoint='end'),dict(primitive_id='reference-arc',endpoint='start'))
        # Same-wall endpoint supplies an exact snap to the forbidden zero chord.
        self.e.raw['primitives'].append(dict(id='snap-target',kind='line',wall='reference',start=[15.,260.],end=[25.,262.],unit='mm'))
        self.assertTrue(self.e.commit('constrained arc and snap target'));self.w.undo_stack.clear();self.e.selected=None
        self.e.fit();self.app.processEvents();view=self.e.sketch
        start=view.mapFromScene(QPointF(5.,-260.));end=view.mapFromScene(QPointF(15.,-260.))
        QTest.mousePress(view.viewport(),Qt.LeftButton,pos=start)
        for fraction in (.25,.55,1.,.5):
            coherent=deepcopy(self.e.raw['primitives'])
            point=start+(end-start)*fraction
            event=QMouseEvent(QMouseEvent.MouseMove,QPointF(point),QPointF(view.viewport().mapToGlobal(point)),Qt.NoButton,Qt.LeftButton,Qt.NoModifier)
            QApplication.sendEvent(view.viewport(),event);self.check_handles()
            if fraction==1.:self.assertEqual(self.e.raw['primitives'],coherent)
            a=self.record('reference-line')['end'];b=self.record('reference-arc')['start']
            np.testing.assert_allclose(a,b,atol=1e-10)
        QTest.mouseRelease(view.viewport(),Qt.LeftButton,pos=point);self.app.processEvents();self.check_handles()
        self.assertFalse(self.e._draft_invalid);self.assertEqual(self.w.undo_stack.count(),1)

    def click_body(self,identity):
        points=Primitive(self.record(identity)).discretize(.001);point=points[len(points)//2]
        QTest.mouseClick(self.e.sketch.viewport(),Qt.LeftButton,pos=self.e.sketch.mapFromScene(QPointF(point[0],-point[1])));self.app.processEvents()

    def test_clicking_two_non_tangent_arcs_between_fixed_lines_refits_pair_and_undoes_once(self):
        materialize_constraints(self.e.raw);raw=pair();self.e.raw['primitives'].extend(raw['primitives']);self.e.raw['constraints'].extend(raw['constraints'])
        self.assertTrue(self.e.commit('two line-constrained arcs'));self.e.fit();self.app.processEvents();self.w.undo_stack.clear();before=project_to_json(self.w.cfg,self.w.nozzles)
        first=deepcopy(self.record('A1'));second=deepcopy(self.record('A2'))
        self.e.start_constraint('tangent');self.click_body('A1');self.click_body('A2')
        self.assertIsNone(self.e.constraint_mode);self.assertFalse(self.e._draft_invalid);self.check_handles()
        self.assertNotEqual(self.record('A1')['center'],first['center']);self.assertNotEqual(self.record('A2')['center'],second['center'])
        self.assertEqual(self.w.undo_stack.count(),1);after=project_to_json(self.w.cfg,self.w.nozzles)
        self.w.undo_stack.undo();self.assertEqual(project_to_json(self.w.cfg,self.w.nozzles),before);self.check_handles()
        self.w.undo_stack.redo();self.assertEqual(project_to_json(self.w.cfg,self.w.nozzles),after);self.check_handles()
