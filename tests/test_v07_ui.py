"""Native v0.7 parameter application, result navigation and field playback.

Tests assert correctness and reuse rather than fragile CPU timing thresholds.
All numerical inputs are labelled synthetic.  The real user example remains
unchanged; its legacy calculation is still the default until explicit opt-in.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import time
import unittest
from unittest.mock import patch

import numpy as np
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage
from PyQt5.QtTest import QTest
from PyQt5.QtWidgets import QApplication

from fuelfield.core import (calculate_plane, evaluate_continuous_field,
                            project_from_json, project_to_json)
from fuelfield.gui import MainWindow
from fuelfield.quasi import calculate_quasi, CalculationOutput


ROOT = Path(__file__).resolve().parents[1]
SYNTHETIC = ROOT/'examples/demo_project_v07_confined_synthetic.json'
ORIGINAL = ROOT/'examples/demo_project_user_field.json'


def pixels(canvas):
    image = canvas.grab().toImage().convertToFormat(QImage.Format_RGBA8888)
    buffer = image.bits()
    buffer.setsize(image.byteCount())
    return np.frombuffer(buffer, dtype=np.uint8).reshape(
        image.height(), image.bytesPerLine()//4, 4)[:, :image.width()].copy()


class TestV07NativeUI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])
        cls.app.setStyle('Fusion')
        cls.raw = json.loads(SYNTHETIC.read_text())
        cls.original_hash = hashlib.sha256(ORIGINAL.read_bytes()).hexdigest()
        cls.synthetic_hash = hashlib.sha256(SYNTHETIC.read_bytes()).hexdigest()
        cfg, ns = project_from_json(deepcopy(cls.raw))
        cls.planes = [calculate_plane(ns, cfg, z) for z in cfg.z_mm]
        cls.fields = [evaluate_continuous_field(ns, cfg, z) for z in cfg.z_mm]

    def setUp(self):
        self.w = MainWindow()
        self.w.resize(1480, 930)
        self.w.show()
        with patch('fuelfield.gui.QFileDialog.getOpenFileName', return_value=(str(SYNTHETIC), '')):
            self.w.open_project()
        self.w._done(list(zip(self.planes, self.fields)))
        self.w.tabs.setCurrentIndex(3)
        self.w.plane_select.setCurrentIndex(len(self.planes)-1)
        self.settle()

    def tearDown(self):
        for owner in (self.w, self.w.comparison):
            if owner.thread is not None:
                owner.thread.requestInterruption()
                self.wait_until(lambda: owner.thread is None)
        self.w.close()
        self.w.deleteLater()
        self.app.processEvents()
        self.assertEqual(hashlib.sha256(ORIGINAL.read_bytes()).hexdigest(), self.original_hash)
        self.assertEqual(hashlib.sha256(SYNTHETIC.read_bytes()).hexdigest(), self.synthetic_hash)

    def settle(self):
        for _ in range(3):
            self.app.processEvents()
        QTest.qWait(2)
        self.app.processEvents()

    def wait_until(self, condition, timeout=120.):
        deadline = time.monotonic()+timeout
        while not condition() and time.monotonic() < deadline:
            QTest.qWait(5)
        self.assertTrue(condition(), 'Timed out waiting for native Qt state')

    def click_view(self, view):
        bar = self.w.plots.tabBar()
        QTest.mouseClick(bar, Qt.LeftButton, pos=bar.tabRect(view).center())
        self.settle()
        self.assertEqual(self.w.plots.currentIndex(), view)
        return self.w.plot_stacks[view].currentWidget()

    def test_explicit_synthetic_settings_apply_then_native_f5_worker(self):
        # Start with the actual unchanged user file, whose default is legacy.
        with patch('fuelfield.gui.QFileDialog.getOpenFileName', return_value=(str(ORIGINAL), '')):
            self.w.open_project()
        baseline = project_to_json(self.w.cfg, self.w.nozzles)
        self.assertEqual(self.w.cfg.fuel_transport.get('mode', 'legacy'), 'legacy')
        self.w.open_transport_settings()
        dialog = self.w.transport_dialog
        self.settle()
        dialog.mode.setCurrentIndex(dialog.mode.findData('confined_inertial'))
        QTest.mouseClick(dialog.apply_button, Qt.LeftButton)
        self.assertIn('не применены', dialog.status.text())
        self.assertEqual(project_to_json(self.w.cfg, self.w.nozzles), baseline)
        QTest.mouseClick(dialog.example_button, Qt.LeftButton)
        self.assertEqual(project_to_json(self.w.cfg, self.w.nozzles), baseline)
        self.assertIn('синтетические', dialog.provenance.text().lower())
        # Lower quadrature only for this explicitly synthetic UI path check.
        dialog.fields['emission_radial_samples'].setValue(1)
        dialog.fields['emission_angular_samples'].setValue(36)
        dialog.fields['max_step_mm'].setText('2')
        QTest.mouseClick(dialog.apply_button, Qt.LeftButton)
        self.assertIn('применены', dialog.status.text().lower())
        self.assertEqual(self.w.cfg.fuel_transport['mode'], 'confined_inertial')
        self.assertEqual(self.w.cfg.fuel_transport['provenance']['source_kind'], 'synthetic')
        self.assertEqual(project_to_json(self.w.cfg, self.w.nozzles)['nozzles'], baseline['nozzles'])
        dialog.close()
        beats, errors = [], []
        timer = QTimer()
        timer.setInterval(5)
        timer.timeout.connect(lambda: beats.append(time.monotonic()))
        timer.start()
        with patch.object(self.w, '_error', side_effect=lambda value: errors.append(str(value))):
            QApplication.setActiveWindow(self.w)
            self.w.activateWindow()
            self.w.setFocus()
            QTest.qWait(10)
            QTest.keyClick(self.w, Qt.Key_F5)
            self.assertIsNotNone(self.w.thread)
            self.wait_until(lambda: self.w.thread is None)
        timer.stop()
        self.assertEqual(errors, [])
        self.assertGreater(len(beats), 1)
        self.assertTrue(self.w.calc_btn.isEnabled())
        self.assertFalse(self.w.cancel_calc_btn.isEnabled())
        self.assertEqual(len(self.w.results), len(self.w.cfg.z_mm))
        for plane in self.w.results:
            self.assertTrue(plane.transport_notice)
            self.assertLess(abs(plane.rel_balance_error), 1e-12)
        self.assertIn('плёнка', self.w.balance_label.text().lower())

    def test_warm_new_model_views_reuse_real_canvases_without_input_changes(self):
        before = project_to_json(self.w.cfg, self.w.nozzles)
        matrices = [plane.cell_mass_kg_s.copy() for plane in self.w.results]
        canvases = {view: self.click_view(view) for view in range(4)}
        with patch.object(self.w, '_draw_map', wraps=self.w._draw_map) as matrix, \
             patch.object(self.w, '_draw_smooth', wraps=self.w._draw_smooth) as field, \
             patch.object(self.w, '_draw_roses', wraps=self.w._draw_roses) as roses, \
             patch.object(self.w, '_draw_curves', wraps=self.w._draw_curves) as curves:
            for other in (0, 1, 2):
                self.w.tabs.setCurrentIndex(other)
                self.settle()
                self.w.tabs.setCurrentIndex(3)
                for view in range(4):
                    self.assertIs(self.click_view(view), canvases[view])
            for renderer in (matrix, field, roses, curves):
                renderer.assert_not_called()
        for plane, expected in zip(self.w.results, matrices):
            np.testing.assert_array_equal(plane.cell_mass_kg_s, expected)
        self.assertEqual(project_to_json(self.w.cfg, self.w.nozzles), before)

    def test_total_flux_and_phase_accounting_are_visible_without_legacy_loss_claim(self):
        plane = self.w.results[-1]
        self.assertGreater(plane.film_inner_cell_mass_kg_s.sum()+plane.film_outer_cell_mass_kg_s.sum(), 0.)
        label = self.w.balance_label.text().lower()
        self.assertIn('плёнка', label)
        self.assertIn('полный поток', label)
        self.assertNotIn('неразрешён', label)
        self.click_view(0)
        artist = self.w.figmap.axes[0].collections[0]
        np.testing.assert_allclose(np.asarray(artist.get_array()).reshape(plane.flux_kg_m2_s.shape),
                                   plane.flux_kg_m2_s, rtol=0, atol=0)
        self.click_view(1)
        self.assertTrue(self.w.figsmooth.axes)
        self.assertGreater(np.asarray(self.w.smooth_fields[-1].masked_flux_kg_m2_s).max(), 0.)

    def test_native_split_json_worker_exposes_matrix_and_field(self):
        QTest.mouseClick(self.w.split_button, Qt.LeftButton)
        self.settle()
        panel = self.w.comparison
        self.assertTrue(panel.isVisible())
        with patch('fuelfield.comparison.QFileDialog.getOpenFileName',
                   return_value=(str(SYNTHETIC), 'JSON')):
            QTest.mouseClick(panel.load_button, Qt.LeftButton)
        self.assertIsNotNone(panel.thread)
        self.wait_until(lambda: panel.thread is None)
        self.assertEqual(len(panel.planes), len(self.planes))
        self.assertEqual(len(panel.fields), len(self.fields))
        self.assertTrue(panel.view.isEnabled())
        panel.view.setCurrentIndex(1)
        self.settle()
        matrix_canvas = panel.canvas
        self.assertTrue(panel.figure.axes[0].collections)
        panel.view.setCurrentIndex(2)
        self.settle()
        field_canvas = panel.canvas
        self.assertTrue(panel.figure.axes[0].images)
        with patch.object(panel, '_draw_field', wraps=panel._draw_field) as draw_field:
            panel.view.setCurrentIndex(1)
            self.settle()
            self.assertIs(panel.canvas, matrix_canvas)
            panel.view.setCurrentIndex(2)
            self.settle()
            self.assertIs(panel.canvas, field_canvas)
            draw_field.assert_not_called()

    def test_quasi_slider_changes_native_pixels_and_preserves_composite_matrix(self):
        cfg = deepcopy(self.w.cfg)
        cfg.quasistationary = True
        result = calculate_quasi(self.w.nozzles, cfg)
        self.w.cfg = cfg
        self.w._fill_controls()
        self.w._done(CalculationOutput([(plane, result.smooth(index, 0))
            for index, plane in enumerate(result.composites)], result))
        self.w.plane_select.setCurrentIndex(len(result.composites)-1)
        canvas = self.click_view(1)
        canvas.draw()
        before = pixels(canvas)
        matrices = [plane.cell_mass_kg_s.copy() for plane in self.w.results]
        artists = tuple(canvas.figure.axes[0].images)
        old_draws = canvas.quasi_animator.draw_count
        # Native key navigation changes the real QSlider and its QTimer path.
        self.w.time_slider.setFocus()
        for _ in range(35):
            QTest.keyClick(self.w.time_slider, Qt.Key_Right)
        self.assertEqual(self.w.time_slider.value(), 35)
        self.wait_until(lambda: canvas._quasi_step == 35)
        self.settle()
        after = pixels(canvas)
        changed = np.any(np.abs(before[:, :, :3].astype(int)-after[:, :, :3].astype(int)) > 2, axis=2)
        self.assertGreater(int(changed.sum()), 100)
        self.assertEqual(tuple(canvas.figure.axes[0].images), artists)
        self.assertEqual(canvas.quasi_animator.draw_count, old_draws)
        self.assertIn('плёнка', self.w.balance_label.text().lower())
        self.assertIn('разное время', self.w.balance_label.text().lower())
        for plane, expected in zip(self.w.results, matrices):
            np.testing.assert_array_equal(plane.cell_mass_kg_s, expected)


if __name__ == '__main__':
    unittest.main()
