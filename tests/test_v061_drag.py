"""User-first-click regressions for v0.6.1 Sketcher interaction.

No drag test calls GeometryEditor.select or selects a table row beforehand.
Native viewport events include their screen position, as Qt requires.
"""
from copy import deepcopy
from pathlib import Path
import sys
import unittest

import numpy as np
from PyQt5.QtCore import QPointF, Qt
from PyQt5.QtGui import QMouseEvent, QFont
from PyQt5.QtTest import QTest
from PyQt5.QtWidgets import QApplication

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from fuelfield.chamber import Chamber, Primitive
from fuelfield.core import project_to_json
from fuelfield.quasi import calculate_quasi, CalculationOutput
import test_v06_gui as previous


class TestV061FirstClickDrag(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        previous.TestV06GeometryUI.setUpClass()
        cls.app = previous.TestV06GeometryUI.app

    setUp = previous.TestV06GeometryUI.setUp
    tearDown = previous.TestV06GeometryUI.tearDown
    record = previous.TestV06GeometryUI.record
    row = previous.TestV06GeometryUI.row
    handle = previous.TestV06GeometryUI.handle
    edit_number = previous.TestV06GeometryUI.edit_number

    def gesture(self, start_scene, target_scene, inspect_frame=None):
        self.assertIsNone(self.e.selected, 'A drag must start with an ordinary first click')
        view = self.e.sketch
        start = view.mapFromScene(QPointF(*start_scene))
        target = view.mapFromScene(QPointF(*target_scene))
        # Native mouse coordinates are integer viewport pixels. Compute the
        # exact scene displacement those pixels represent instead of comparing
        # the drag with a requested subpixel world point using a loose bound.
        scene_delta = view.mapToScene(target) - view.mapToScene(start)
        expected_delta = np.array([scene_delta.x(), -scene_delta.y()])
        QTest.mousePress(view.viewport(), Qt.LeftButton, pos=start)
        try:
            for fraction in (.35, .7, 1.):
                position = start + (target - start) * fraction
                event = QMouseEvent(QMouseEvent.MouseMove, QPointF(position),
                                    QPointF(view.viewport().mapToGlobal(position)),
                                    Qt.NoButton, Qt.LeftButton, Qt.NoModifier)
                QApplication.sendEvent(view.viewport(), event)
                if inspect_frame is not None:
                    inspect_frame()
        finally:
            QTest.mouseRelease(view.viewport(), Qt.LeftButton, pos=target)
        self.app.processEvents()
        self.assertEqual(self.qt_errors, [])
        return expected_delta

    def assert_handles_match_raw(self, identity):
        raw = self.record(identity)
        factor = {'mm': 1., 'cm': 10., 'm': 1000.}[raw.get('unit', 'mm')]
        for handle in self.e.handles:
            if handle.identity != identity or handle.role not in raw:
                continue
            expected = np.array(raw[handle.role]) * factor
            np.testing.assert_allclose([handle.pos().x(), -handle.pos().y()], expected,
                                       atol=1e-8, err_msg=identity + ' ' + handle.role)

    def test_first_click_line_endpoint_does_not_drag_opposite_endpoint(self):
        original = deepcopy(self.record('reference-line'))
        before = project_to_json(self.w.cfg, self.w.nozzles)
        start = original['start']

        def inspect():
            self.assertEqual(self.record('reference-line')['end'], original['end'])
            self.assert_handles_match_raw('reference-line')
            self.assertEqual(project_to_json(self.w.cfg, self.w.nozzles), before)
            self.assertEqual(self.w.undo_stack.count(), 0)

        expected_delta = self.gesture((start[0], -start[1]), (7., -247.), inspect)
        actual = self.record('reference-line')
        np.testing.assert_allclose(actual['start'], np.array(start) + expected_delta,
                                   rtol=0, atol=1e-9)
        self.assertEqual(actual['end'], original['end'])
        self.assertEqual(self.w.undo_stack.count(), 1)
        self.w.undo_stack.undo()
        self.assertEqual(project_to_json(self.w.cfg, self.w.nozzles), before)

    def test_first_click_arc_endpoint_tracks_cursor_with_coherent_handles(self):
        original = deepcopy(self.record('reference-arc'))
        before = project_to_json(self.w.cfg, self.w.nozzles)

        def inspect():
            self.assertEqual(self.record('reference-arc')['end'], original['end'])
            self.assert_handles_match_raw('reference-arc')
            self.assertEqual(self.w.undo_stack.count(), 0)

        expected_delta = self.gesture((5., -260.), (8., -259.), inspect)
        actual = self.record('reference-arc')
        np.testing.assert_allclose(actual['start'], np.array(original['start']) + expected_delta,
                                   rtol=0, atol=1e-9)
        self.assertEqual(actual['end'], original['end'])
        Primitive(actual)
        self.assertIn('reference-arc', self.e.paths)
        self.assertEqual(self.w.undo_stack.count(), 1)
        self.w.undo_stack.undo()
        self.assertEqual(project_to_json(self.w.cfg, self.w.nozzles), before)
        self.w.undo_stack.redo()
        self.assert_handles_match_raw('reference-arc')

    def test_real_drag_shared_arc_vertex_in_mm_and_cm_is_exact_and_atomic(self):
        # A real inner-wall chain: the common vertex is represented once in
        # millimetres and once in centimetres. Other reference handles are
        # omitted so the first click is unambiguous between physical objects.
        outer = deepcopy(self.record('outer-1'))
        self.e.raw['primitives'] = [
            dict(id='arc-mm', wall='inner', kind='arc', unit='mm',
                 start=[0., 260.], end=[15., 260.], center=[7.5, 280.],
                 clockwise=False),
            dict(id='arc-cm', wall='inner', kind='arc', unit='cm',
                 start=[1.5, 26.], end=[3., 26.], center=[2.25, 28.],
                 clockwise=False),
            outer,
        ]
        self.assertTrue(self.e.commit('Синтетические соседние дуги mm/cm'))
        self.w.undo_stack.clear()
        self.e.fit()
        self.app.processEvents()
        before = project_to_json(self.w.cfg, self.w.nozzles)

        def assert_chain():
            first, second = (Primitive(self.record(identity))
                             for identity in ('arc-mm', 'arc-cm'))
            np.testing.assert_allclose(first.end, second.start, atol=1e-10)
            np.testing.assert_allclose(first.start, [0., 260.], atol=1e-10)
            np.testing.assert_allclose(second.end, [30., 260.], atol=1e-10)
            for primitive in (first, second):
                self.assertAlmostEqual(np.linalg.norm(primitive.start - primitive.center),
                                       np.linalg.norm(primitive.end - primitive.center), places=10)
                self.assert_handles_match_raw(primitive.id)
            return first, second

        def inspect():
            assert_chain()
            self.assertEqual(project_to_json(self.w.cfg, self.w.nozzles), before)
            self.assertEqual(self.w.undo_stack.count(), 0)

        expected_delta = self.gesture((15., -260.), (16., -262.), inspect)
        first, second = assert_chain()
        np.testing.assert_allclose(first.end, np.array([15., 260.]) + expected_delta,
                                   rtol=0, atol=1e-9)
        self.assertEqual(self.record('arc-mm')['unit'], 'mm')
        self.assertEqual(self.record('arc-cm')['unit'], 'cm')
        Chamber(self.w.cfg.geometry)
        self.assertEqual(self.w.undo_stack.count(), 1)
        after = project_to_json(self.w.cfg, self.w.nozzles)
        self.w.undo_stack.undo()
        self.assertEqual(project_to_json(self.w.cfg, self.w.nozzles), before)
        self.w.undo_stack.redo()
        self.assertEqual(project_to_json(self.w.cfg, self.w.nozzles), after)
        self.assertEqual(self.w.undo_stack.count(), 1)
        assert_chain()

    def test_first_click_arc_center_translates_whole_circle_freely(self):
        original = deepcopy(self.record('reference-arc'))
        radius = Primitive(original).radius
        expected_delta = self.gesture((10., -265.), (13., -267.),
                                      lambda: self.assert_handles_match_raw('reference-arc'))
        actual = self.record('reference-arc')
        delta = np.array(actual['center']) - original['center']
        np.testing.assert_allclose(delta, expected_delta, rtol=0, atol=1e-9)
        for role in ('start', 'end'):
            np.testing.assert_allclose(np.array(actual[role]) - original[role], delta,
                                       atol=1e-10)
        self.assertAlmostEqual(Primitive(actual).radius, radius, places=10)
        self.assertEqual(actual['clockwise'], original['clockwise'])
        self.assertEqual(self.w.undo_stack.count(), 1)

    def test_arc_body_drag_translates_without_changing_radius_or_other_primitives(self):
        original = deepcopy(self.record('reference-arc'))
        primitive = Primitive(original)
        points = primitive.discretize(.01)
        midpoint = points[len(points) // 2]
        other = deepcopy(self.record('reference-line'))
        expected_delta = self.gesture((midpoint[0], -midpoint[1]),
                                      (midpoint[0] + 3., -midpoint[1] - 2.),
                                      lambda: self.assert_handles_match_raw('reference-arc'))
        actual = self.record('reference-arc')
        delta = np.array(actual['center']) - original['center']
        np.testing.assert_allclose(delta, expected_delta, rtol=0, atol=1e-9)
        for role in ('start', 'end'):
            np.testing.assert_allclose(np.array(actual[role]) - original[role], delta,
                                       atol=1e-10)
        self.assertAlmostEqual(Primitive(actual).radius, primitive.radius, places=10)
        self.assertEqual(self.record('reference-line'), other)
        self.assertEqual(self.w.undo_stack.count(), 1)

    def test_radius_edit_rebuilds_circle_without_disappearing(self):
        original = deepcopy(self.record('reference-arc'))
        self.edit_number(self.e.primitive_table, 'reference-arc', 9, '10')
        actual = self.record('reference-arc')
        self.assertAlmostEqual(Primitive(actual).radius, 10., places=9)
        self.assertEqual(actual['start'], original['start'])
        self.assertEqual(actual['end'], original['end'])
        self.assertIn('reference-arc', self.e.paths)
        self.assertGreater(self.e.paths['reference-arc'].path().elementCount(), 2)
        self.assert_handles_match_raw('reference-arc')
        self.assertEqual(self.w.undo_stack.count(), 1)

    def test_impossible_radius_keeps_visible_draft_and_valid_calculation_state(self):
        before = project_to_json(self.w.cfg, self.w.nozzles)
        self.edit_number(self.e.primitive_table, 'reference-arc', 9, '3')
        self.assertEqual(project_to_json(self.w.cfg, self.w.nozzles), before)
        self.assertEqual(self.w.undo_stack.count(), 1)  # One draft action, no calculation change.
        self.assertIn('reference-arc', self.e.paths)
        self.assertGreater(self.e.paths['reference-arc'].path().elementCount(), 1)
        self.assertTrue(self.e.status.text())
        self.assertGreaterEqual(len([h for h in self.e.handles
                                     if h.identity == 'reference-arc']), 3)

    def test_invalid_drag_stays_visible_at_release_without_resetting_pointer_target(self):
        before = project_to_json(self.w.cfg, self.w.nozzles)
        original = deepcopy(self.record('inner-1'))
        target = [original['end'][0], self.w.cfg.r_out_mm + 4.]
        # Deliberately cross the outer wall. Unlike a zero-length arc chord,
        # invalidity here is robust to subpixel mouse-coordinate rounding.
        expected_delta = self.gesture((original['end'][0], -original['end'][1]),
                                      (target[0], -target[1]))
        actual = self.record('inner-1')
        np.testing.assert_allclose(actual['end'], np.array(original['end']) + expected_delta,
                                   rtol=0, atol=1e-9)
        self.assertNotEqual(actual['end'], original['end'])
        self.assertIn('inner-1', self.e.paths)
        self.assertEqual(project_to_json(self.w.cfg, self.w.nozzles), before)
        self.assertEqual(self.w.undo_stack.count(), 1)  # Release records the incomplete draft.
        self.assertTrue(self.e.status.text())

    def test_results_split_canvas_stays_inside_window_with_larger_ui_font(self):
        self.w.cfg.quasistationary = True
        result = calculate_quasi(self.w.nozzles, self.w.cfg)
        self.w._done(CalculationOutput(
            [(p, result.smooth(i, 0)) for i, p in enumerate(result.composites)], result))
        self.w.tabs.setCurrentIndex(3)
        self.w.plots.setCurrentIndex(1)
        font = QFont(self.w.font())
        font.setPointSize(13)
        self.w.setFont(font)
        self.w.split_button.setChecked(True)
        self.w.resize(1280, 720)
        self.app.processEvents()
        self.assertLessEqual(self.w.height(), 720)
        for canvas, panel in ((self.w.smoothcanvas, self.w.plots),
                              (self.w.comparison.canvas, self.w.comparison)):
            self.assertTrue(canvas.isVisible())
            bottom = canvas.mapTo(panel, canvas.rect().bottomRight())
            self.assertLessEqual(bottom.y(), panel.height() - 1,
                                 'The canvas protrudes below its result panel')
            bottom = canvas.mapTo(self.w.tabs, canvas.rect().bottomRight())
            self.assertLessEqual(bottom.y(), self.w.tabs.height() - 1,
                                 'The canvas protrudes below the application window')

    def test_results_first_entry_and_new_cached_canvases_do_not_grow_window(self):
        font = QFont(self.w.font())
        font.setPointSize(13)
        self.w.setFont(font)
        self.w.resize(1280, 720)
        self.app.processEvents()
        initial_height = self.w.height()
        self.assertLessEqual(initial_height, 720)
        self.w.cfg.quasistationary = True
        result = calculate_quasi(self.w.nozzles, self.w.cfg)
        self.w._done(CalculationOutput(
            [(p, result.smooth(i, 0)) for i, p in enumerate(result.composites)], result))
        self.app.processEvents()
        self.assertEqual(self.w.height(), initial_height)
        self.w.tabs.setCurrentIndex(3)
        self.app.processEvents()
        self.assertEqual(self.w.height(), initial_height)
        for split in (False, True):
            self.w.split_button.setChecked(split)
            self.app.processEvents()
            frame_bounds = self.w.result_panels.geometry()
            for index in (1, 2, 3, 0, 1):
                # The first visits allocate new canvases; subsequent visits
                # show cached widgets. Neither may feed Figure.sizeHint back
                # into growing the native application window.
                self.w.plots.setCurrentIndex(index)
                for _ in range(3):
                    self.app.processEvents()
                    self.assertEqual(self.w.height(), initial_height)
                    self.assertEqual(self.w.result_panels.geometry(), frame_bounds)


if __name__ == '__main__':
    unittest.main()
