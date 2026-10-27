"""Native Hide/destruction cancels drafts without repainting a dying canvas."""
from copy import deepcopy
import gc
import unittest
from unittest.mock import patch

import numpy as np
from PyQt5 import sip
from PyQt5.QtCore import QEvent, QPoint, QPointF, Qt
from PyQt5.QtGui import QMouseEvent
from PyQt5.QtTest import QTest
from PyQt5.QtWidgets import QApplication
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure

from fuelfield.core import Settings, calculate_plane, default_nozzles, project_to_json
from fuelfield.gui import MainWindow
from fuelfield.plot_interaction import attach_interaction
from fuelfield.themes import apply_theme, current_theme


def qt_point(canvas, axes, x, y):
    px, py = axes.transData.transform((x, y))
    ratio = canvas.device_pixel_ratio
    return QPoint(round(px / ratio), round((canvas.figure.bbox.height - py) / ratio))


def move_held(canvas, point):
    event = QMouseEvent(QEvent.MouseMove, QPointF(point),
                        QPointF(canvas.mapToGlobal(point)), Qt.NoButton,
                        Qt.LeftButton, Qt.NoModifier)
    QApplication.sendEvent(canvas, event)


class TestCanvasLifecycle(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.before_theme = current_theme()
        apply_theme(self.app, 'light', persist=False)

    def tearDown(self):
        apply_theme(self.app, self.before_theme, persist=False)
        self.app.processEvents()

    def canvas_with_active_draft(self):
        figure = Figure(figsize=(6, 4))
        canvas = FigureCanvasQTAgg(figure)
        axes = figure.add_subplot()
        axes.plot([0., 10.], [0., 20.])
        axes.set_xlim(0., 10.); axes.set_ylim(0., 20.)
        controller = attach_interaction(canvas)
        canvas.show(); self.app.processEvents(); canvas.draw()
        start = qt_point(canvas, axes, 3., 6.)
        end = qt_point(canvas, axes, 7., 16.)
        QTest.mouseDClick(canvas, Qt.LeftButton, pos=start)
        QTest.mouseRelease(canvas, Qt.LeftButton, pos=start)
        QTest.mousePress(canvas, Qt.LeftButton, pos=start)
        move_held(canvas, end)
        QTest.mouseRelease(canvas, Qt.LeftButton, pos=end)
        self.app.processEvents()
        self.assertEqual(len(controller.points), 1)
        self.assertEqual(len(controller.oblique), 1)
        history = controller.history()
        before = controller.snapshot()['signature']
        history_index = history.index
        QTest.mousePress(canvas, Qt.LeftButton, pos=start)
        for fraction in np.linspace(.1, 1., 20):
            point = start + QPoint(round((end.x() - start.x()) * fraction),
                                   round((end.y() - start.y()) * fraction))
            move_held(canvas, point)
        self.app.processEvents()
        self.assertEqual(len(controller.oblique), 2)
        self.assertIsNotNone(controller.slope_drag)
        self.assertEqual(history.index, history_index)
        return canvas, controller, history, history_index, before

    def assert_suspended(self, controller, history, history_index, before):
        self.assertIsNone(controller.slope_drag)
        self.assertIsNone(controller.drag)
        self.assertIsNone(controller.rectangle)
        self.assertIsNone(controller.background)
        self.assertIsNone(controller.base_background)
        self.assertFalse(controller.pressed)
        self.assertEqual(controller.snapshot()['signature'], before)
        self.assertEqual(history.index, history_index)

    def test_hide_cancels_only_preview_without_any_render_operation(self):
        canvas, controller, history, index, before = self.canvas_with_active_draft()
        methods = ('draw', 'draw_idle', 'blit', 'copy_from_bbox', 'restore_region')
        try:
            from contextlib import ExitStack
            with ExitStack() as stack:
                mocks = {name: stack.enter_context(patch.object(canvas, name)) for name in methods}
                canvas.hide(); self.app.processEvents()
                self.assert_suspended(controller, history, index, before)
                self.assertEqual({name: mock.call_count for name, mock in mocks.items()},
                                 {name: 0 for name in methods})
        finally:
            canvas.deleteLater(); QApplication.sendPostedEvents(None, QEvent.DeferredDelete)

    def test_delete_later_with_active_drag_does_not_paint_deleted_widget(self):
        canvas, controller, history, index, before = self.canvas_with_active_draft()
        from contextlib import ExitStack
        methods = ('draw', 'draw_idle', 'blit', 'copy_from_bbox', 'restore_region')
        with ExitStack() as stack:
            mocks = {name: stack.enter_context(patch.object(canvas, name)) for name in methods}
            canvas.deleteLater()
            QApplication.sendPostedEvents(None, QEvent.DeferredDelete)
            self.app.processEvents()
            self.assertTrue(sip.isdeleted(canvas))
            self.assert_suspended(controller, history, index, before)
            self.assertEqual({name: mock.call_count for name, mock in mocks.items()},
                             {name: 0 for name in methods})

    def test_repeated_render_close_and_deferred_theme_callbacks_are_safe(self):
        cfg = Settings(z_mm=[20.], n_radial=2, n_angular=8,
                       n_sample_radial=8, n_sample_angular=72)
        nozzles = default_nozzles(cfg)
        for nozzle in nozzles: nozzle.enabled = nozzle.id == 1
        nozzles[0].flow_kg_h = 3.6
        plane = calculate_plane(nozzles, cfg, 20.)
        expected = plane.cell_mass_kg_s.copy()
        destroyed = []
        for cycle in range(10):
            window = MainWindow()
            window._restoring = True
            window.cfg = deepcopy(cfg); window.nozzles = deepcopy(nozzles)
            window._fill_controls(); window._fill_nozzles()
            window._edit_state = project_to_json(window.cfg, window.nozzles)
            window._restoring = False
            window.tabs.setCurrentIndex(3)
            window._done([(plane, None)])
            window.plots.setCurrentIndex(3); window.plot_count.setValue(1)
            window.show(); self.app.processEvents(); window.curvecanvas.draw()
            canvas = window.curvecanvas
            canvas.destroyed.connect(lambda obj=None, n=cycle: destroyed.append(n))
            # Queue the appearance callback, then destroy its target first.
            QApplication.sendEvent(canvas, QEvent(QEvent.Show))
            window.close(); window.deleteLater()
            QApplication.sendPostedEvents(None, QEvent.DeferredDelete)
            self.app.processEvents()
            self.assertTrue(sip.isdeleted(canvas))
            np.testing.assert_array_equal(plane.cell_mass_kg_s, expected)
            del window, canvas
            gc.collect()
        self.assertEqual(sorted(destroyed), list(range(10)))
