"""Native air table copy/paste, atomic validation and inheritance overrides."""
from copy import deepcopy
import unittest
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QKeySequence
from PyQt5.QtTest import QTest
from PyQt5.QtWidgets import QApplication,QTableWidgetSelectionRange,QComboBox,QStyleOptionViewItem
from fuelfield.core import Settings,default_nozzles
from fuelfield.air_gui import AirBalanceDialog,COLUMNS


def col(key):return next(i for i,c in enumerate(COLUMNS) if c[0]==key)


class TestAirTable(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.app=QApplication.instance() or QApplication([])

    def setUp(self):
        self.cfg=Settings(air_model=dict(enabled=True,total_kg_s=10.,sources=[
            dict(id='A',kind='hole',method='fraction',share=.1),
            dict(id='B',kind='hole',method='fraction',share=.1),
            dict(id='C',kind='hole',method='fixed',flow_kg_s=2.)]))
        self.dialog=AirBalanceDialog(self.cfg,default_nozzles(self.cfg));self.dialog.show();self.app.processEvents()

    def tearDown(self):self.dialog.close();self.dialog.deleteLater();self.app.processEvents()

    def select(self,r0,c0,r1,c1):
        table=self.dialog.table;table.clearSelection();table.setCurrentCell(r0,c0)
        table.setRangeSelected(QTableWidgetSelectionRange(r0,c0,r1,c1),True)

    def test_scalar_paste_fills_multiple_cells_and_tsv_copy_uses_native_keys(self):
        t=self.dialog.table;self.select(0,col('share'),1,col('share'))
        QApplication.clipboard().setText('.2');t.setFocus();QTest.keyClick(t,Qt.Key_V,Qt.ControlModifier)
        self.assertEqual([s['share'] for s in self.dialog.raw['sources'][:2]],[.2,.2])
        QTest.keyClick(t,Qt.Key_C,Qt.ControlModifier)
        self.assertEqual(QApplication.clipboard().text(),'.2\n.2')

    def test_invalid_multirow_paste_rolls_back_every_target(self):
        t=self.dialog.table;self.select(0,col('share'),0,col('share'));before=deepcopy(self.dialog.raw)
        self.assertFalse(t.paste_text('.2\n-1'))
        self.assertEqual(t.item(0,col('share')).text(),'0.1')
        self.assertEqual(t.item(1,col('share')).text(),'0.1');self.assertEqual(self.dialog.raw,before)
        self.assertIn('целиком',self.dialog.status.text())

    def test_fixed_flow_and_linked_coordinates_cannot_be_overwritten_by_paste(self):
        t=self.dialog.table;self.select(2,col('flow_kg_s'),2,col('flow_kg_s'))
        self.assertFalse(t.paste_text('7'));self.assertEqual(self.dialog.raw['sources'][2]['flow_kg_s'],2.)
        self.dialog.raw['sources'][0].update(kind='swirler',nozzle_id=1)
        self.dialog.fill();self.dialog.refresh();self.select(0,col('radius_mm'),0,col('radius_mm'))
        self.assertFalse(t.paste_text('240'))
        self.assertEqual(float(t.item(0,col('radius_mm')).text()),270.)

    def test_all_choice_columns_open_real_comboboxes(self):
        t=self.dialog.table
        for name in ('kind','method','wall','kernel','confidence'):
            index=t.model().index(0,col(name));delegate=t.itemDelegateForColumn(col(name))
            editor=delegate.createEditor(t,QStyleOptionViewItem(),index)
            self.assertIsInstance(editor,QComboBox);self.assertGreater(editor.count(),1)
            editor.deleteLater()

    def test_reset_template_preserves_protected_manual_flow(self):
        source=self.dialog.raw['sources'][2];source.update(kind='swirler',nozzle_id=3)
        self.dialog.fill();self.dialog.refresh();self.dialog.table.selectRow(2)
        self.dialog.reset_template()
        source=self.dialog.raw['sources'][2]
        self.assertEqual(source['flow_kg_s'],2.);self.assertEqual(source['method'],'fixed')
        self.assertIn('flow_kg_s',source['air_overrides']);self.assertIn('method',source['air_overrides'])

    def test_inherited_display_edit_creates_only_one_override_and_reset_clears_it(self):
        self.dialog.close();self.dialog.deleteLater();self.app.processEvents()
        cfg=Settings(air_model=dict(enabled=False,sources=[]))
        cfg.nozzle_presets['uncooled']['air']=dict(method='fixed',flow_kg_s=.01,sigma_r_mm=7.)
        self.dialog=AirBalanceDialog(cfg,default_nozzles(cfg));self.dialog.show();self.app.processEvents()
        t=self.dialog.table;self.assertEqual(t.rowCount(),140)
        self.assertEqual(float(t.item(0,col('sigma_r_mm')).text()),7.)
        self.select(0,col('sigma_r_mm'),0,col('sigma_r_mm'));self.assertTrue(t.paste_text('9'))
        source=self.dialog.raw['sources'][0]
        self.assertEqual(source['air_overrides'],['sigma_r_mm']);self.assertEqual(source['sigma_r_mm'],9.)
        self.dialog.reset_template()
        self.assertEqual(self.dialog.raw['sources'][0]['air_overrides'],[])
        self.assertEqual(float(t.item(0,col('sigma_r_mm')).text()),7.)


if __name__=='__main__':unittest.main()
