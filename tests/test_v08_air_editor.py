"""Transactional air edits preserve completed fuel fields and fixed inputs."""
from copy import deepcopy
import unittest
import numpy as np
from PyQt5.QtCore import Qt
from PyQt5.QtTest import QTest
from PyQt5.QtWidgets import QApplication
from fuelfield.core import Settings, default_nozzles, calculate_plane, project_to_json
from fuelfield.air_gui import AirBalanceDialog, COLUMNS
from fuelfield.gui import MainWindow


def model():
    return dict(schema_version=1, enabled=True, total_kg_s=10., l0=15.,distribution='uniform',
        sources=[dict(id='known',kind='hole',method='fixed',flow_kg_s=2.,radius_mm=250.),
                 dict(id='variable',kind='hole',method='fraction',share=.3,radius_mm=250.)],
        provenance={'origin':'synthetic UI test'})


class TestAirEditor(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.app=QApplication.instance() or QApplication([])

    def setUp(self):
        self.cfg=Settings(z_mm=[20.],n_sample_radial=8,n_sample_angular=72,air_model=model())
        self.ns=default_nozzles(self.cfg);self.dialog=AirBalanceDialog(self.cfg,self.ns)
        self.dialog.show();self.app.processEvents()

    def tearDown(self):
        self.dialog.close();self.dialog.deleteLater();self.app.processEvents()

    def test_slider_and_number_are_bidirectional_and_do_not_change_fixed_flow(self):
        snapshot=deepcopy(self.cfg.air_model)
        slider,number=self.dialog.groups['hole']
        slider.setFocus();slider.setValue(6000);self.app.processEvents()
        self.assertAlmostEqual(number.value(),60.)
        self.assertAlmostEqual(self.dialog.raw['sources'][0]['flow_kg_s'],2.)
        self.assertAlmostEqual(self.dialog.raw['sources'][1]['share'],.4)
        number.setValue(70.);self.app.processEvents()
        self.assertEqual(slider.value(),7000)
        self.assertAlmostEqual(self.dialog.balance.unassigned_kg_s,3.)
        # Real Qt keyboard input moves the actual QSlider, not a decoration.
        slider.setFocus();QTest.keyClick(slider,Qt.Key_Left);self.app.processEvents()
        self.assertLess(number.value(),70.)
        self.assertEqual(self.cfg.air_model,snapshot)

    def test_fixed_edit_protection_and_duplicate_sources_block_apply(self):
        col=next(i for i,c in enumerate(COLUMNS) if c[0]=='flow_kg_s')
        self.assertFalse(self.dialog.table.item(0,col).flags() & Qt.ItemIsEditable)
        self.dialog.unlock_fixed.setChecked(True)
        self.assertTrue(self.dialog.table.item(0,col).flags() & Qt.ItemIsEditable)
        self.dialog.table.item(1,0).setText('known');self.dialog.refresh()
        self.assertFalse(self.dialog.apply_button.isEnabled())
        self.assertIn('ID',self.dialog.status.text())

    def test_unknown_swirlers_are_not_copied_from_fuel_flows(self):
        self.ns[0].flow_kg_h=1234.
        self.dialog.add_swirlers()
        sources=[s for s in self.dialog.raw['sources'] if s['kind']=='swirler']
        self.assertEqual(len(sources),140)
        self.assertEqual({s['nozzle_id'] for s in sources},set(range(1,141)))
        self.assertTrue(all(s['method']=='unknown' and s['flow_kg_s'] is None for s in sources))

    def test_apply_is_atomic_and_cancel_does_not_touch_input(self):
        received=[];self.dialog.applied.connect(received.append)
        self.dialog.total.setValue(12.);self.dialog.apply()
        self.assertEqual(len(received),1)
        self.assertEqual(received[0]['total_kg_s'],12.)
        self.assertEqual(self.cfg.air_model['total_kg_s'],10.)
        self.dialog.reject();self.assertEqual(self.cfg.air_model['total_kg_s'],10.)

    def test_main_air_apply_and_undo_preserve_fuel_results(self):
        win=MainWindow()
        try:
            win.cfg=self.cfg;win.nozzles=self.ns
            self.ns[0].flow_kg_h=36.
            plane=calculate_plane(self.ns,self.cfg,20.)
            win._restoring=True;win._fill_controls();win._fill_nozzles();win._restoring=False
            win._edit_state=project_to_json(win.cfg,win.nozzles)
            win.results=[plane];before=plane.cell_mass_kg_s.copy()
            # Isolate postprocessing from drawing: a fuel field is already complete.
            win._plot_result=lambda *a:None;win.comparison.draw=lambda *a:None
            raw=deepcopy(self.cfg.air_model);raw['total_kg_s']=11.
            win.apply_air_model(raw)
            self.assertEqual(win.cfg.air_model['total_kg_s'],11.)
            self.assertIs(win.results[0],plane)
            win.undo_stack.undo()
            self.assertEqual(win.cfg.air_model['total_kg_s'],10.)
            self.assertIs(win.results[0],plane);np.testing.assert_array_equal(plane.cell_mass_kg_s,before)
        finally:win.close();win.deleteLater();self.app.processEvents()
