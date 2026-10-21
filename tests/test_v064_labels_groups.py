"""Endpoint numbering and native entry of a multi-value angle list."""
import unittest
from PyQt5.QtCore import Qt
from PyQt5.QtTest import QTest
from PyQt5.QtWidgets import QLineEdit
import test_v06_gui as previous


class TestLabelsAndGroups(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        previous.TestV06GeometryUI.setUpClass();cls.app=previous.TestV06GeometryUI.app
    setUp=previous.TestV06GeometryUI.setUp
    tearDown=previous.TestV06GeometryUI.tearDown
    row=previous.TestV06GeometryUI.row

    def test_labels_follow_each_curve_endpoint_without_claiming_mouse_events(self):
        for handle in self.e.handles:
            if handle.object_handle or handle.identity=='reference-point':self.assertIsNone(handle.endpoint_label);continue
            if handle.role=='center':self.assertIsNone(handle.endpoint_label);continue
            self.assertEqual(handle.endpoint_label.text(),'1' if handle.role=='start' else '2')
            self.assertEqual(handle.endpoint_label.parentItem(),handle)
            self.assertEqual(handle.endpoint_label.acceptedMouseButtons(),Qt.NoButton)
            self.assertEqual(handle.endpoint_label.brush().color().name(),'#737b85')
        handle=next(h for h in self.e.handles if h.identity=='reference-arc' and h.role=='start')
        before=handle.endpoint_label.scenePos();handle.setPos(handle.pos().x()+2,handle.pos().y()+3)
        self.assertAlmostEqual(handle.endpoint_label.scenePos().x()-before.x(),2.)
        self.assertAlmostEqual(handle.endpoint_label.scenePos().y()-before.y(),3.)

    def enter(self,table,row,column,text):
        table.setCurrentCell(row,column);table.editItem(table.item(row,column));self.app.processEvents()
        editor=next(w for w in table.findChildren(QLineEdit) if w.isVisible());editor.selectAll()
        QTest.keyClicks(editor,text);self.assertEqual(editor.text(),text)
        QTest.keyClick(editor,Qt.Key_Return);self.app.processEvents()

    def test_single_numeric_angle_remains_text_and_accepts_semicolon(self):
        self.e.raw['groups']=[dict(id='G1',members=['N1'],mode='individual',phase_deg=0.,direction=1,row_offset_deg=0.,angles_deg=[0.])]
        self.assertTrue(self.e.commit('Synthetic single member group',groups=True));self.e.pages.setCurrentIndex(2);self.app.processEvents()
        table=self.e.group_table;row=self.row(table,'G1');self.assertEqual(table.item(row,6).text(),'0')
        self.enter(table,row,6,'0;37.5')
        self.assertEqual(table.item(row,6).text(),'0;37.5');self.assertTrue(self.e._draft_invalid)
        self.enter(table,row,1,'N1;N2')
        self.assertFalse(self.e._draft_invalid);self.assertEqual(self.e.raw['groups'][0]['angles_deg'],[0.,37.5])
        nozzles={n.id:n for n in self.w.nozzles};self.assertAlmostEqual(nozzles[2].theta_deg,37.5)
