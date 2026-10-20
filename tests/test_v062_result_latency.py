"""Native result-navigation regressions; warm paths must reuse real artists.

The unchanged user example supplies all 140 nozzles and its full quadrature.
Wall-clock performance is recorded separately; these tests assert the causes
of lag so they remain meaningful on machines with different CPU speeds.
"""
from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

import numpy as np
from PyQt5.QtCore import Qt,QCoreApplication,QEvent
from PyQt5.QtTest import QTest
from PyQt5.QtWidgets import QApplication
from matplotlib.backend_bases import MouseEvent, MouseButton
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from fuelfield.core import project_from_json, project_to_json, calculate_plane, evaluate_continuous_field
from fuelfield.analysis import cell_values
from fuelfield.gui import MainWindow
from fuelfield.quasi import CalculationOutput
from fuelfield.axes_settings import open_axes_settings

ROOT = Path(__file__).resolve().parents[1]


class TestV062ResultNavigation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])
        cls.raw = json.loads((ROOT / 'examples/demo_project_user_field.json').read_text())
        cfg, ns = project_from_json(cls.raw)
        cls.planes = [calculate_plane(ns, cfg, z) for z in cfg.z_mm]
        cls.fields = [evaluate_continuous_field(ns, cfg, z) for z in cfg.z_mm]

    def setUp(self):
        self.w = MainWindow()
        self.w._restoring = True
        self.w.cfg, self.w.nozzles = project_from_json(deepcopy(self.raw))
        self.w.sector_id = 1
        self.w._fill_controls()
        self.w._fill_nozzles()
        self.w._reset_history()
        self.w.show()
        self.w._done(list(zip(self.planes, self.fields)))
        self.w.tabs.setCurrentIndex(3)
        self.w.plane_select.setCurrentIndex(2)
        self.settle()

    def tearDown(self):
        for owner in (self.w, self.w.comparison):
            limit = time.monotonic() + 30
            while owner.thread is not None and time.monotonic() < limit:
                QTest.qWait(5)
        self.w.close()
        self.w.deleteLater()
        self.app.processEvents()

    def settle(self):
        for _ in range(3):
            self.app.processEvents()
        QTest.qWait(2)
        self.app.processEvents()

    def click_view(self, index):
        bar = self.w.plots.tabBar()
        QTest.mouseClick(bar, Qt.LeftButton, pos=bar.tabRect(index).center())
        self.settle()
        return self.w.plot_stacks[index].currentWidget()

    def comparison(self, visible=True):
        c = self.w.comparison
        c._loading_cfg = deepcopy(self.w.cfg)
        c._loading_ns = deepcopy(self.w.nozzles)
        c._project_done(CalculationOutput(self.planes, fields=self.fields))
        if visible:
            self.w.split_button.setChecked(True)
        self.settle()
        return c

    def test_warm_views_and_returning_from_other_tabs_reuse_figures(self):
        before = project_to_json(self.w.cfg, self.w.nozzles)
        canvases = {view: self.click_view(view) for view in range(4)}
        with patch.object(self.w, '_draw_map', wraps=self.w._draw_map) as matrix, \
             patch.object(self.w, '_draw_smooth', wraps=self.w._draw_smooth) as field, \
             patch.object(self.w, '_draw_roses', wraps=self.w._draw_roses) as roses, \
             patch.object(self.w, '_draw_curves', wraps=self.w._draw_curves) as curves:
            for other in (0, 1, 2):
                self.w.tabs.setCurrentIndex(other)
                QTest.qWait(150)
                self.w.tabs.setCurrentIndex(3)
                for view in range(4):
                    self.assertIs(self.click_view(view), canvases[view])
            for render in (matrix, field, roses, curves):
                render.assert_not_called()
        self.assertEqual(project_to_json(self.w.cfg, self.w.nozzles), before)

    def test_epure_radius_count_and_columns_do_not_evict_other_views(self):
        canvases = {view: self.click_view(view) for view in range(3)}
        self.click_view(3)
        with patch.object(self.w, '_draw_map', wraps=self.w._draw_map) as matrix, \
             patch.object(self.w, '_draw_smooth', wraps=self.w._draw_smooth) as field, \
             patch.object(self.w, '_draw_roses', wraps=self.w._draw_roses) as roses:
            self.w.compare_radius.setCurrentIndex(2)
            self.w.plot_count.setValue(3)
            self.w.plot_columns.setValue(1)
            self.settle()
            for view in range(3):
                self.assertIs(self.click_view(view), canvases[view])
            for render in (matrix, field, roses):
                render.assert_not_called()

    def test_metric_changes_do_not_rebuild_true_flux_field_or_polar_profiles(self):
        field = self.click_view(1)
        roses = self.click_view(2)
        self.w.metric.setCurrentIndex(self.w.metric.findData('kg_h'))
        self.settle()
        self.assertIs(self.w.rosescanvas, roses)
        self.assertIs(self.click_view(1), field)
        self.click_view(0)
        values = np.asarray(self.w.figmap.axes[0].collections[0].get_array())
        np.testing.assert_allclose(values.reshape(self.planes[2].cell_mass_kg_s.shape),
                                   self.planes[2].cell_mass_kg_s * 3600)

    def test_hidden_comparison_defers_its_navigation_render_until_opened(self):
        c = self.comparison(False)
        with patch.object(c.canvas, 'draw', wraps=c.canvas.draw) as draw, \
             patch.object(c.figure, 'clear', wraps=c.figure.clear) as clear:
            self.w.plane_select.setCurrentIndex(3)
            self.settle()
            draw.assert_not_called()
            clear.assert_not_called()
        self.assertEqual(c.plane.currentIndex(), 3)
        self.w.split_button.setChecked(True)
        self.settle()
        np.testing.assert_allclose(c.figure.axes[0].lines[0].get_ydata(),
                                   cell_values(self.planes[3], 'flux', self.w.cfg.density_kg_m3)[0])

    def test_warm_split_z_preserves_cached_axes_points_zoom_and_correct_data(self):
        self.click_view(3)
        c = self.comparison()
        canvas = c.canvas
        ax = c.figure.axes[0]
        canvas.draw()
        ax.set_xlim(20, 160)
        x = 90.
        y = float(np.interp(x, ax.lines[0].get_xdata(), ax.lines[0].get_ydata()))
        pixel = ax.transData.transform((x, y))
        c.interaction.press(MouseEvent('button_press_event', canvas, *pixel,
                                       button=MouseButton.LEFT, dblclick=True))
        self.assertEqual(len(c.interaction.points), 1)
        self.w.plane_select.setCurrentIndex(3)
        self.settle()
        self.w.plane_select.setCurrentIndex(2)
        self.settle()
        self.assertIs(c.canvas, canvas)
        self.assertIs(c.figure.axes[0], ax)
        self.assertEqual(ax.get_xlim(), (20, 160))
        self.assertEqual(len(c.interaction.points), 1)
        np.testing.assert_allclose(ax.lines[0].get_ydata(),
                                   cell_values(self.planes[2], 'flux', self.w.cfg.density_kg_m3)[0])

    def test_hidden_cached_canvases_do_not_render_when_split_resizes_views(self):
        for view in range(4):
            self.click_view(view)
        hidden_draws = []
        original = FigureCanvasQTAgg.draw

        def observed(canvas, *args, **kwargs):
            if canvas in self.w._render_cache.values() and not canvas.isVisible():
                hidden_draws.append(canvas)
            return original(canvas, *args, **kwargs)

        with patch.object(FigureCanvasQTAgg, 'draw', observed):
            self.w.split_button.setChecked(True)
            self.settle()
            for view in range(4):
                self.click_view(view)
        self.assertEqual(hidden_draws, [])

    def test_render_cache_limits_are_bounded_and_keep_current_view_valid(self):
        self.w.RENDER_CACHE_LIMIT = 5
        self.click_view(3)
        self.w.plot_count.setValue(1)
        for index in range(6):
            self.w.compare_radius.setCurrentIndex(index)
            self.settle()
        self.assertLessEqual(len(self.w._render_cache), 5)
        self.assertTrue(self.w.curvecanvas.isVisible())
        self.assertEqual(len(self.w.figcurves.axes), 1)

    def test_lru_eviction_detaches_links_and_old_axes_dialog_safely(self):
        self.click_view(3)
        self.w.plot_count.setValue(1)
        c = self.comparison()
        manager = self.w.graph_links
        source, target = self.w.curvecanvas, c.canvas
        manager.connect(source.figure.axes[0], source.interaction,
                        target.figure.axes[0], target.interaction)
        dialog = open_axes_settings(source, self.w)
        self.w.RENDER_CACHE_LIMIT = 1
        c.CHART_CACHE_LIMIT = 1
        self.w.plane_select.setCurrentIndex(3)
        self.settle()
        QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)
        self.settle()
        manager.refresh()
        manager.overlay.repaint()
        self.assertEqual(manager.links, [])
        dialog.apply_settings()
        dialog.close()
        self.assertEqual(len(self.w._render_cache), 1)
        self.assertEqual(len(c._chart_cache), 1)

    def test_png_export_renders_hidden_cached_views_without_mutating_inputs(self):
        from PIL import Image
        before = project_to_json(self.w.cfg, self.w.nozzles)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'field.png'
            with patch('fuelfield.gui.QFileDialog.getSaveFileName', return_value=(str(path), 'PNG')), \
                 patch.object(self.w, '_error', side_effect=AssertionError):
                self.w.export_png()
            files = sorted(Path(folder).glob('*.png'))
            self.assertEqual(len(files), 4)
            for exported in files:
                with Image.open(exported) as image:
                    self.assertGreater(image.width, 100)
                    self.assertGreater(image.height, 100)
                    self.assertGreater(np.asarray(image).std(), 0)
        self.assertEqual(project_to_json(self.w.cfg, self.w.nozzles), before)


if __name__ == '__main__':
    unittest.main()
