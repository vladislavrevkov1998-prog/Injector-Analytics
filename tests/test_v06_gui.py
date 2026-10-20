"""Independent Qt acceptance tests for the peer v0.6 geometry editors.

The scene is exercised with real mouse/key events; drafts must never leak into
the calculation state. Small numerical quadrature keeps the UI checks cheap.
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
from PyQt5.QtCore import Qt, QPoint, QPointF
from PyQt5.QtGui import QMouseEvent, QWheelEvent
from PyQt5.QtTest import QTest
from PyQt5.QtWidgets import (QApplication, QDoubleSpinBox, QGraphicsLineItem,
                             QGraphicsSimpleTextItem)

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from fuelfield.chamber import Chamber, Primitive, legacy_geometry
from fuelfield.core import calculate_plane, project_from_json, project_to_json
from fuelfield.gui import MainWindow
from test_quasi import case


class TestV06GeometryUI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.qt_errors = []
        self.old_hook = sys.excepthook
        sys.excepthook = lambda kind, value, tb: self.qt_errors.append((kind, value))
        self.w = MainWindow()
        cfg, nozzles, _ = case(140)
        cfg.quasistationary = False
        cfg.placement_mode = 'manual'
        cfg.geometry = legacy_geometry(cfg, nozzles)
        cfg.geometry['enabled'] = True
        cfg.geometry['primitives'].extend([
            dict(id='reference-point', wall='reference', kind='point',
                 start=[12., 250.], unit='mm'),
            dict(id='reference-line', wall='reference', kind='line',
                 start=[4., 245.], end=[15., 245.], unit='mm'),
            dict(id='reference-arc', wall='reference', kind='arc',
                 start=[5., 260.], end=[15., 260.], center=[10., 265.],
                 clockwise=False, unit='mm'),
        ])
        cfg.geometry['objects'].append(dict(
            id='hole-test', type='hole', wall='free', z_mm=18., radius_mm=280.,
            theta_deg=32., parameters={'flow_kg_s': None},
            units={'length': 'mm'}, provenance={'source': 'synthetic UI test'}))
        self.w._restoring = True
        self.w.cfg, self.w.nozzles = cfg, nozzles
        self.w._fill_controls()
        self.w._fill_nozzles()
        self.w._reset_history()
        self.w.show()
        self.w.tabs.setCurrentIndex(0)
        self.w.geometry_pages.setCurrentIndex(1)
        self.app.processEvents()
        self.e = self.w.geometry_editor
        self.e.fit()
        self.app.processEvents()
        self.assertEqual(self.qt_errors, [])

    def tearDown(self):
        try:
            for owner in (self.w, self.w.comparison):
                deadline = time.monotonic() + 30
                while owner.thread is not None and time.monotonic() < deadline:
                    QTest.qWait(5)
            self.w.close()
            self.w.deleteLater()
            self.app.processEvents()
            self.assertEqual(self.qt_errors, [], 'Unhandled Qt callback exception')
        finally:
            sys.excepthook = self.old_hook

    def record(self, identity):
        return next(p for p in self.e.raw['primitives'] if p['id'] == identity)

    def row(self, table, identity):
        return next(i for i in range(table.rowCount())
                    if table.item(i, 0).text() == identity)

    def edit_number(self, table, identity, column, text):
        """Use the actual delegate, keyboard entry and edit commit."""
        row = self.row(table, identity)
        table.setCurrentCell(row, column)
        table.scrollToItem(table.item(row, column))
        table.editItem(table.item(row, column))
        self.app.processEvents()
        editors = [child for child in table.findChildren(QDoubleSpinBox)
                   if child.isVisible()]
        self.assertEqual(len(editors), 1)
        editor = editors[0]
        editor.lineEdit().selectAll()
        QTest.keyClicks(editor.lineEdit(), text)
        QTest.keyClick(editor.lineEdit(), Qt.Key_Return)
        self.app.processEvents()
        self.assertEqual(self.qt_errors, [])

    def handle(self, identity, role='start'):
        return next(h for h in self.e.handles
                    if h.identity == identity and h.role == role)

    def drag(self, identity, target, role='start'):
        h = self.handle(identity, role)
        view = self.e.sketch
        start = view.mapFromScene(h.pos())
        end = view.mapFromScene(QPointF(target[0], -target[1]))
        QTest.mousePress(view.viewport(), Qt.LeftButton, pos=start)
        # Explicit button state matters for QGraphicsView under offscreen Qt.
        for fraction in (.3, .65, 1.):
            position = start + (end - start) * fraction
            move = QMouseEvent(QMouseEvent.MouseMove, QPointF(position),
                               QPointF(view.viewport().mapToGlobal(position)),
                               Qt.NoButton, Qt.LeftButton, Qt.NoModifier)
            QApplication.sendEvent(view.viewport(), move)
        QTest.mouseRelease(view.viewport(), Qt.LeftButton, pos=end)
        self.app.processEvents()
        self.assertEqual(self.qt_errors, [])

    def test_peer_splitter_and_editable_types_visible_at_1280(self):
        self.w.resize(1280, 800)
        self.app.processEvents()
        self.assertLessEqual(self.w.width(), 1280)
        self.assertEqual(self.e.splitter.count(), 2)
        self.assertTrue(self.e.sketch.isVisible())
        self.assertTrue(self.e.primitive_table.isVisible())
        self.assertGreaterEqual(self.e.sketch.width(), 220)
        self.assertGreaterEqual(self.e.primitive_table.width(), 300)
        self.assertLessEqual(self.e.error.mapTo(self.e, self.e.error.rect().topRight()).x(),
                             self.e.width())
        initial = self.e.splitter.sizes()
        self.e.splitter.setSizes([initial[0] + 60, initial[1] - 60])
        self.assertNotEqual(initial, self.e.splitter.sizes())
        self.assertEqual(self.e.object_table.rowCount(), 141)
        for identity, kind in [('reference-point', 'point'),
                               ('reference-line', 'line'),
                               ('reference-arc', 'arc')]:
            self.assertEqual(self.record(identity)['kind'], kind)
            self.assertIn(identity, self.e.paths)

    def test_table_point_edit_updates_sketch_and_one_undo_redo(self):
        before = project_to_json(self.w.cfg, self.w.nozzles)
        self.edit_number(self.e.primitive_table, 'reference-point', 4, '251.125')
        self.assertEqual(self.w.undo_stack.count(), 1)
        self.assertEqual(self.record('reference-point')['start'], [12., 251.125])
        self.assertAlmostEqual(self.handle('reference-point').pos().y(), -251.125)
        self.w.undo_stack.undo()
        self.assertEqual(project_to_json(self.w.cfg, self.w.nozzles), before)
        self.assertAlmostEqual(self.handle('reference-point').pos().y(), -250.)
        self.w.undo_stack.redo()
        self.assertEqual(self.w.undo_stack.count(), 1)
        self.assertAlmostEqual(self.handle('reference-point').pos().y(), -251.125)

    def test_coordinate_edit_retains_scene_rows_and_avoids_hidden_editor_reloads(self):
        scene = self.e.sketch.scene()
        path = self.e.paths['reference-point']
        handle = self.handle('reference-point')
        untouched_path = self.e.paths['reference-line']
        object_item = self.e.object_table.item(0, 4)
        nozzle_item = self.w.nozzle_table.item(0, 0)
        transform = self.e.sketch.transform()
        before = project_to_json(self.w.cfg, self.w.nozzles)
        with patch.object(self.w.catalog_editor, 'load', wraps=self.w.catalog_editor.load) as catalog_load, \
             patch.object(self.w.preset_editor, 'load', wraps=self.w.preset_editor.load) as preset_load, \
             patch.object(self.w, '_read_nozzles', wraps=self.w._read_nozzles) as read_nozzles, \
             patch.object(self.w, '_fill_nozzles', wraps=self.w._fill_nozzles) as fill_nozzles:
            self.edit_number(self.e.primitive_table, 'reference-point', 4, '251.125')
            catalog_load.assert_not_called()
            preset_load.assert_not_called()
            read_nozzles.assert_not_called()
            fill_nozzles.assert_not_called()
        self.assertIs(self.e.sketch.scene(), scene)
        self.assertIs(self.e.paths['reference-point'], path)
        self.assertIs(self.e.paths['reference-line'], untouched_path)
        self.assertIs(self.handle('reference-point'), handle)
        self.assertIs(self.e.object_table.item(0, 4), object_item)
        self.assertIs(self.w.nozzle_table.item(0, 0), nozzle_item)
        self.assertEqual(self.e.sketch.transform(), transform)
        self.assertEqual(self.e.selected, 'reference-point')
        self.assertTrue(handle.isSelected())
        self.assertEqual(self.w.undo_stack.count(), 1)
        after = project_to_json(self.w.cfg, self.w.nozzles)
        self.w.undo_stack.undo()
        self.assertEqual(project_to_json(self.w.cfg, self.w.nozzles), before)
        self.w.undo_stack.redo()
        self.assertEqual(project_to_json(self.w.cfg, self.w.nozzles), after)

    def test_incremental_wall_edit_updates_bound_hole_row_and_existing_handle(self):
        hole = next(o for o in self.e.raw['objects'] if o['id'] == 'hole-test')
        hole.update(wall='inner', primitive_id='inner-1', radial_offset_mm=2.)
        self.assertTrue(self.e.commit('Привязать тестовое отверстие к стенке'))
        self.e.pages.setCurrentIndex(0)
        self.w.undo_stack.clear()
        row = self.row(self.e.object_table, 'hole-test')
        handle = self.handle('hole-test', 'position')
        untouched = self.e.object_table.item(self.row(self.e.object_table, 'N1'), 4)
        base_radius = self.w.cfg.r_in_mm
        self.edit_number(self.e.primitive_table, 'inner-1', 6, repr(base_radius + 4.))
        expected = base_radius + 4. * 18. / 30. + 2.
        self.assertAlmostEqual(float(self.e.object_table.item(row, 5).text()), expected)
        self.assertFalse(self.e.object_table.item(row, 5).flags() & Qt.ItemIsEditable)
        self.assertIs(self.handle('hole-test', 'position'), handle)
        self.assertAlmostEqual(handle.pos().y(), -expected)
        self.assertIs(self.e.object_table.item(self.row(self.e.object_table, 'N1'), 4), untouched)
        self.assertEqual(self.w.undo_stack.count(), 1)

    def test_incremental_wall_edit_updates_section_markers_to_actual_bounds(self):
        self.edit_number(self.e.primitive_table, 'inner-1', 6,
                         repr(self.w.cfg.r_in_mm + 4.))
        self.edit_number(self.e.primitive_table, 'outer-1', 6,
                         repr(self.w.cfg.r_out_mm + 4.))
        model = Chamber(self.w.cfg.geometry)
        markers = [item.line() for item in self.e.sketch.scene().items()
                   if isinstance(item, QGraphicsLineItem)]
        self.assertEqual(len(markers), len(model.sections))
        for section in model.sections:
            marker = next(line for line in markers
                          if abs(line.x1() - section['z_mm']) < 1e-8)
            rin, rout = model.bounds(section['z_mm'])
            self.assertAlmostEqual(marker.y1(), -rin)
            self.assertAlmostEqual(marker.y2(), -rout)
            label_text = section['id'] + (' · выход' if section.get('is_outlet') else '')
            label = next(item for item in self.e.sketch.scene().items()
                         if isinstance(item, QGraphicsSimpleTextItem)
                         and item.text() == label_text)
            self.assertAlmostEqual(label.pos().x(), section['z_mm'])
            self.assertAlmostEqual(label.pos().y(), -rout)

    def test_reopening_same_project_discards_invalid_geometry_draft(self):
        with tempfile.TemporaryDirectory() as folder:
            path = str(Path(folder) / 'geometry_project.json')
            self.w.project_file = path
            self.w.save_project()
            expected = deepcopy(self.w.cfg.geometry)
            row = self.row(self.e.primitive_table, 'reference-arc')
            self.e.primitive_table.item(row, 9).setText('1')
            self.assertIn('не применён', self.e.status.text())
            self.assertNotEqual(self.e.raw, expected)
            with patch('fuelfield.gui.QFileDialog.getOpenFileName', return_value=(path, 'JSON')):
                self.w.open_project()
            self.assertEqual(self.w.cfg.geometry, expected)
            self.assertEqual(self.e.raw, expected)
            self.assertIn('reference-arc', self.e.paths)
            self.assertEqual(self.w.undo_stack.count(), 0)
    def test_sketch_drag_updates_table_and_commits_once(self):
        before = project_to_json(self.w.cfg, self.w.nozzles)
        self.drag('reference-point', [18., 251.])
        actual = self.record('reference-point')['start']
        # Native mouse coordinates are integer pixels; scene precision varies
        # with the viewport size and the fitted geometry, not the data model.
        pixel = 1. / self.e.sketch.transform().m11()
        self.assertAlmostEqual(actual[0], 18., delta=pixel)
        self.assertAlmostEqual(actual[1], 251., delta=pixel)
        row = self.row(self.e.primitive_table, 'reference-point')
        self.assertAlmostEqual(float(self.e.primitive_table.item(row, 3).text()), actual[0])
        self.assertAlmostEqual(float(self.e.primitive_table.item(row, 4).text()), actual[1])
        self.assertEqual(self.w.undo_stack.count(), 1)
        self.w.undo_stack.undo()
        self.assertEqual(project_to_json(self.w.cfg, self.w.nozzles), before)
        self.w.undo_stack.redo()
        np.testing.assert_allclose(self.record('reference-point')['start'], actual)

    def test_table_and_scene_selection_are_linked_both_ways(self):
        table = self.e.primitive_table
        row = self.row(table, 'reference-line')
        table.setCurrentCell(row, 0)
        self.assertEqual(self.e.selected, 'reference-line')
        self.assertEqual({h.identity for h in self.e.handles if h.isSelected()},
                         {'reference-line'})
        h = self.handle('reference-point')
        QTest.mouseClick(self.e.sketch.viewport(), Qt.LeftButton,
                         pos=self.e.sketch.mapFromScene(h.pos()))
        self.app.processEvents()
        self.assertEqual(self.e.selected, 'reference-point')
        self.assertEqual(table.item(table.currentRow(), 0).text(), 'reference-point')
        self.assertEqual({h.identity for h in self.e.handles if h.isSelected()},
                         {'reference-point'})

    def test_arc_center_drag_remains_exact_and_table_tracks_it(self):
        original = deepcopy(self.record('reference-arc'))
        self.drag('reference-arc', [10., 268.], role='center')
        record = self.record('reference-arc')
        self.assertNotEqual(record['center'], original['center'])
        self.assertAlmostEqual(record['center'][0], 10., delta=.3)
        self.assertAlmostEqual(record['center'][1], 268., delta=.3)
        arc = Primitive(record)
        self.assertAlmostEqual(np.linalg.norm(arc.start - arc.center),
                               np.linalg.norm(arc.end - arc.center), places=10)
        delta=np.array(record['center'])-original['center']
        np.testing.assert_allclose(np.array(record['start'])-original['start'],delta,atol=1e-10)
        np.testing.assert_allclose(np.array(record['end'])-original['end'],delta,atol=1e-10)
        row = self.row(self.e.primitive_table, 'reference-arc')
        self.assertAlmostEqual(float(self.e.primitive_table.item(row, 8).text()),
                               record['center'][1], places=7)
        self.assertEqual(self.w.undo_stack.count(), 1)

    def test_arc_unit_edit_preserves_physical_scene_and_analytic_shape(self):
        original = Primitive(self.record('reference-arc'))
        row = self.row(self.e.primitive_table, 'reference-arc')
        self.e.primitive_table.item(row, 11).setText('cm')
        self.app.processEvents()
        changed = Primitive(self.record('reference-arc'))
        np.testing.assert_allclose(changed.start, original.start)
        np.testing.assert_allclose(changed.end, original.end)
        np.testing.assert_allclose(changed.center, original.center)
        self.assertEqual(changed.unit, 'cm')
        self.assertAlmostEqual(self.handle('reference-arc').pos().x(), original.start[0])

    def test_arc_table_center_edit_updates_analytic_sketch_and_undo(self):
        self.edit_number(self.e.primitive_table, 'reference-arc', 8, '268.125')
        arc = Primitive(self.record('reference-arc'))
        self.assertEqual(arc.center.tolist(), [10., 268.125])
        self.assertAlmostEqual(self.handle('reference-arc', 'center').pos().y(), -268.125)
        self.assertGreater(self.e.paths['reference-arc'].path().elementCount(), 2)
        self.w.undo_stack.undo()
        self.assertEqual(Primitive(self.record('reference-arc')).center.tolist(), [10., 265.])

    def test_sketch_double_click_creates_point_line_and_exact_arc(self):
        view = self.e.sketch
        before = len(self.e.raw['primitives'])
        positions = ((1, [(22., 240.)]),
                     (2, [(3., 233.), (11., 233.)]),
                     (3, [(5., 238.), (15., 238.)]))
        for tool, points in positions:
            self.e.tool.setCurrentIndex(tool)
            for z, radius in points:
                pos = view.mapFromScene(QPointF(z, -radius))
                QTest.mouseDClick(view.viewport(), Qt.LeftButton, pos=pos)
                self.app.processEvents()
        added = self.e.raw['primitives'][before:]
        self.assertEqual([p['kind'] for p in added], ['point', 'line', 'arc'])
        self.assertEqual(self.w.undo_stack.count(), 3)
        for raw in added:
            self.assertIn(raw['id'], self.e.paths)
            self.assertEqual(self.e.primitive_table.item(self.row(self.e.primitive_table, raw['id']), 0).text(), raw['id'])
            Primitive(raw)

    def test_invalid_arc_draft_keeps_valid_project_and_can_be_repaired(self):
        before = project_to_json(self.w.cfg, self.w.nozzles)
        row = self.row(self.e.primitive_table, 'reference-arc')
        self.e.primitive_table.item(row, 9).setText('1')
        self.assertEqual(project_to_json(self.w.cfg, self.w.nozzles), before)
        self.assertEqual(self.w.undo_stack.count(), 1)  # Draft edits are undoable too.
        self.assertIn('не применён', self.e.status.text())
        radius = float(np.linalg.norm(np.array([5., 260.]) - [10., 265.]))
        self.e.primitive_table.item(row, 9).setText(repr(radius))
        self.assertNotIn('Черновик', self.e.status.text())
        self.assertEqual(self.qt_errors, [])

    def test_wall_endpoint_joint_update_is_atomic(self):
        self.e.pages.setCurrentIndex(0)
        row = self.row(self.e.primitive_table, 'inner-1')
        self.e.primitive_table.setCurrentCell(row, 0)
        self.e.split_primitive()
        records = [p for p in self.e.raw['primitives'] if p['wall'] == 'inner']
        self.assertEqual(len(records), 2)
        first, second = records
        self.assertEqual(first['end'], second['start'])
        self.w.undo_stack.clear()
        self.edit_number(self.e.primitive_table, first['id'], 6, '201.25')
        first = self.record(first['id'])
        second = self.record(second['id'])
        self.assertEqual(first['end'], second['start'])
        self.assertEqual(first['end'][1], 201.25)
        Chamber(self.w.cfg.geometry)
        self.assertEqual(self.w.undo_stack.count(), 1)

    def test_object_table_and_angular_drag_change_nozzle_without_data_loss(self):
        original = deepcopy(self.w.nozzles[0])
        self.e.pages.setCurrentIndex(1)
        self.edit_number(self.e.object_table, 'N1', 6, '73.125')
        self.assertAlmostEqual(self.w.nozzles[0].theta_deg, 73.125)
        self.assertEqual(self.w.nozzles[0].id, original.id)
        self.assertEqual(self.w.nozzles[0].sector_ml, original.sector_ml)
        self.assertEqual(self.w.nozzles[0].manifold_id, original.manifold_id)
        self.assertEqual(self.w.nozzles[0].flow_kg_h, original.flow_kg_h)
        self.e.angular.setChecked(True)
        self.e.fit()
        self.app.processEvents()
        self.w.undo_stack.clear()
        self.drag('N1', [0., self.w.nozzles[0].radius_mm], role='position')
        self.assertAlmostEqual(self.w.nozzles[0].theta_deg, 90., delta=.4)
        self.assertEqual(self.w.undo_stack.count(), 1)
        self.assertEqual(len(self.w.nozzles), 140)

    def test_objects_groups_and_uneven_steps_preserve_existing_nozzle_records(self):
        self.e.pages.setCurrentIndex(1)
        for identity in ('N1', 'N2', 'N3'):
            row = self.row(self.e.object_table, identity)
            self.e.object_table.item(row, 0).setSelected(True)
        before = deepcopy(self.w.nozzles)
        self.e.add_group()
        self.assertEqual(len(self.e.raw['groups']), 1)
        group = self.e.raw['groups'][0]
        self.assertEqual(group['members'], ['N1', 'N2', 'N3'])
        group.update(mode='steps', steps_deg=[80., 120., 160.], phase_deg=15., direction=-1)
        self.assertTrue(self.e.commit('Неравномерная тестовая группа', groups=True))
        self.assertEqual([n.theta_deg for n in self.w.nozzles[:3]], [15., 295., 175.])
        for previous, current in zip(before, self.w.nozzles):
            self.assertEqual(current.id, previous.id)
            self.assertEqual(current.enabled, previous.enabled)
            self.assertEqual(current.sector_ml, previous.sector_ml)
            self.assertEqual(current.manifold_id, previous.manifold_id)
        self.assertEqual(len([h for h in self.e.handles
                              if h.object_handle and h.identity.startswith('N')]), 140)
        self.e.add_object('insert', [20., 275.])
        added = self.e.raw['objects'][-1]
        self.assertEqual(added['type'], 'insert')
        self.assertIsNone(added['parameters']['flow_kg_s'])
        self.assertIsNone(added['parameters']['penetration_mm'])

    def test_140_member_group_selection_and_normalized_angles(self):
        group = dict(id='all-nozzles', members=['N' + str(i) for i in range(1, 141)],
                     mode='uniform', direction=-1, phase_deg=450.)
        self.e.raw['groups'].append(group)
        self.assertTrue(self.e.commit('Большая угловая группа', groups=True))
        self.e.pages.setCurrentIndex(2)
        self.e.group_table.setCurrentCell(0, 0)
        selected = [h for h in self.e.handles if h.isSelected()]
        self.assertEqual(len(selected), 140)
        np.testing.assert_allclose([n.theta_deg for n in self.w.nozzles],
                                   (90 - np.arange(140) * 360 / 140) % 360)
        self.assertEqual(self.qt_errors, [])

    def test_longitudinal_view_uses_physical_profiles_and_selected_section_values(self):
        inner = next(p for p in self.e.raw['primitives'] if p['wall'] == 'inner')
        inner['start'][1] = 200.
        inner['end'][1] = 210.
        self.assertTrue(self.e.commit('Переменный внутренний профиль'))
        self.w.results = [calculate_plane(self.w.nozzles, self.w.cfg, z)
                          for z in self.w.cfg.z_mm]
        self.w.open_longitudinal()
        dialog = self.w.longitudinal_dialog
        self.app.processEvents()
        self.assertTrue(dialog.isVisible())
        top = dialog.figure.axes[0]
        np.testing.assert_allclose(top.lines[0].get_ydata(), [200., 210.])
        offsets = np.asarray(top.collections[0].get_offsets())
        expected = np.array([(p.z_mm, radius) for p in self.w.results
                             for radius in p.radial_centers_mm])
        np.testing.assert_allclose(offsets, expected)
        dialog.mode.setCurrentIndex(1)
        dialog.index.setValue(3)
        self.app.processEvents()
        self.assertTrue(dialog.index.isEnabled())
        self.assertEqual(dialog.index.value(), 3)
        self.assertEqual(self.qt_errors, [])
        dialog.close()

    def test_manual_section_order_and_outlet_deletion_honor_object_domain(self):
        self.e.pages.setCurrentIndex(3)
        before = [s['z_mm'] for s in self.e.raw['sections']]
        self.e.add_section()
        self.assertEqual([s['z_mm'] for s in self.e.raw['sections'][:2]], before)
        self.assertEqual(self.w.cfg.z_mm, [0., 30., 15.])
        row = len(self.e.raw['sections']) - 1
        identity = self.e.raw['sections'][row]['id']
        self.edit_number(self.e.section_table, identity, 1, '12.125')
        self.assertEqual(self.w.cfg.z_mm, [0., 30., 12.125])
        self.e.section_table.setCurrentCell(2, 0)
        self.e.move_section(-1)
        self.assertEqual(self.w.cfg.z_mm, [0., 12.125, 30.])
        outlet_row = next(i for i, s in enumerate(self.e.raw['sections']) if s['is_outlet'])
        geometry_before=deepcopy(self.e.raw)
        self.e.section_table.clearSelection()
        self.e.section_table.item(outlet_row, 0).setSelected(True)
        self.e.delete(self.e.section_table)
        self.assertEqual(self.w.cfg.z_mm, [0., 12.125, 30.])
        self.assertEqual(self.e.raw,geometry_before)
        self.assertIn('hole-test',self.e.status.text())

    def test_real_save_open_restores_geometry_and_unknown_fields(self):
        self.w.cfg._project_extra['future_project'] = {'x': [1, 2]}
        self.e.raw['future_geometry'] = {'identity': 'keep'}
        self.record('reference-arc')['future_primitive'] = [4, 5]
        self.e.raw['objects'][-1]['future_object'] = 'keep'
        self.assertTrue(self.e.commit('Неизвестные расширения'))
        expected = json.loads(json.dumps(project_to_json(self.w.cfg, self.w.nozzles)))
        with tempfile.TemporaryDirectory() as folder:
            path = str(Path(folder) / 'geometry_project.json')
            self.w.project_file = path
            self.w.save_project()
            data = json.loads(Path(path).read_text(encoding='utf-8'))
            self.assertEqual(data, expected)
            self.edit_number(self.e.primitive_table, 'reference-point', 4, '252')
            with patch('fuelfield.gui.QFileDialog.getOpenFileName', return_value=(path, 'JSON')):
                self.w.open_project()
            self.assertEqual(json.loads(json.dumps(project_to_json(self.w.cfg, self.w.nozzles))), expected)
            self.assertEqual(self.w.undo_stack.count(), 0)
            self.assertEqual(self.e.raw['future_geometry'], {'identity': 'keep'})
            self.assertEqual(self.record('reference-arc')['future_primitive'], [4, 5])
            self.assertEqual(self.e.raw['objects'][-1]['future_object'], 'keep')

    def test_wheel_zoom_middle_pan_and_fit_do_not_edit_project(self):
        state = project_to_json(self.w.cfg, self.w.nozzles)
        view = self.e.sketch
        original = view.transform().m11()
        pos = view.viewport().rect().center()
        wheel = QWheelEvent(QPointF(pos), QPointF(view.viewport().mapToGlobal(pos)),
                            QPoint(), QPoint(0, 120), Qt.NoButton,
                            Qt.NoModifier, Qt.NoScrollPhase, False)
        QApplication.sendEvent(view.viewport(), wheel)
        self.assertGreater(view.transform().m11(), original)
        old_scroll = (view.horizontalScrollBar().value(), view.verticalScrollBar().value())
        QTest.mousePress(view.viewport(), Qt.MiddleButton, pos=pos)
        move = QMouseEvent(QMouseEvent.MouseMove, QPointF(pos + QPoint(30, 20)),
                           Qt.NoButton, Qt.MiddleButton, Qt.NoModifier)
        QApplication.sendEvent(view.viewport(), move)
        QTest.mouseRelease(view.viewport(), Qt.MiddleButton, pos=pos + QPoint(30, 20))
        new_scroll = (view.horizontalScrollBar().value(), view.verticalScrollBar().value())
        self.assertNotEqual(old_scroll, new_scroll)
        self.assertIsNone(view._pan)
        self.e.fit()
        self.assertAlmostEqual(view.transform().m11(), original, delta=.02)
        self.assertEqual(project_to_json(self.w.cfg, self.w.nozzles), state)
        self.assertEqual(self.w.undo_stack.count(), 0)


if __name__ == '__main__':
    unittest.main()
