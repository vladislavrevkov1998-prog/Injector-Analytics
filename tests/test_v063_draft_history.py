"""Draft edits, accepted edits and reset share the application Undo timeline."""
from copy import deepcopy
import unittest
from unittest.mock import patch

from PyQt5.QtCore import Qt
from PyQt5.QtTest import QTest

from fuelfield.core import project_to_json
import test_v06_gui as previous


class TestDraftHistory(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        previous.TestV06GeometryUI.setUpClass()
        cls.app=previous.TestV06GeometryUI.app
    setUp=previous.TestV06GeometryUI.setUp
    tearDown=previous.TestV06GeometryUI.tearDown
    record=previous.TestV06GeometryUI.record
    row=previous.TestV06GeometryUI.row
    edit_number=previous.TestV06GeometryUI.edit_number

    def invalidate_wall(self,radius):
        self.edit_number(self.e.primitive_table,'inner-1',6,str(radius))
        self.assertTrue(self.e._draft_invalid)
        return deepcopy(self.e.raw)

    def test_each_incomplete_wall_edit_undoes_one_step_without_applying_it(self):
        canonical=project_to_json(self.w.cfg,self.w.nozzles)
        original=deepcopy(self.e.raw)
        first=self.invalidate_wall(self.w.cfg.r_out_mm+5.)
        second=self.invalidate_wall(self.w.cfg.r_out_mm+10.)
        self.assertEqual(self.w.undo_stack.count(),2)
        self.assertEqual(project_to_json(self.w.cfg,self.w.nozzles),canonical)
        self.w.undo_stack.undo();self.assertEqual(self.e.raw,first)
        self.assertTrue(self.e._draft_invalid)
        self.w.undo_stack.undo();self.assertEqual(self.e.raw,original)
        self.assertFalse(self.e._draft_invalid)
        self.w.undo_stack.redo();self.assertEqual(self.e.raw,first)
        self.w.undo_stack.redo();self.assertEqual(self.e.raw,second)
        self.assertEqual(project_to_json(self.w.cfg,self.w.nozzles),canonical)

    def test_reset_itself_is_undoable_and_redoable_by_main_shortcuts(self):
        canonical=project_to_json(self.w.cfg,self.w.nozzles)
        draft=self.invalidate_wall(self.w.cfg.r_out_mm+5.)
        self.e.reset_draft();self.assertFalse(self.e._draft_invalid)
        accepted=deepcopy(self.e.raw)
        self.assertEqual(self.w.undo_stack.count(),2)
        self.e.sketch.setFocus();QTest.keyClick(self.e.sketch,Qt.Key_Z,Qt.ControlModifier);self.app.processEvents()
        self.assertEqual(self.e.raw,draft);self.assertTrue(self.e._draft_invalid)
        QTest.keyClick(self.e.sketch,Qt.Key_Y,Qt.ControlModifier);self.app.processEvents()
        self.assertEqual(self.e.raw,accepted);self.assertFalse(self.e._draft_invalid)
        self.assertEqual(project_to_json(self.w.cfg,self.w.nozzles),canonical)

    def test_repair_undo_returns_incomplete_draft_and_previous_accepted_project(self):
        canonical=project_to_json(self.w.cfg,self.w.nozzles)
        original_radius=self.record('inner-1')['end'][1]
        invalid=self.invalidate_wall(self.w.cfg.r_out_mm+5.)
        self.edit_number(self.e.primitive_table,'inner-1',6,str(original_radius+1.))
        self.assertFalse(self.e._draft_invalid)
        accepted=project_to_json(self.w.cfg,self.w.nozzles)
        self.assertNotEqual(accepted,canonical)
        self.w.undo_stack.undo()
        self.assertEqual(self.e.raw,invalid)
        self.assertTrue(self.e._draft_invalid)
        self.assertEqual(project_to_json(self.w.cfg,self.w.nozzles),canonical)
        self.w.undo_stack.redo()
        self.assertFalse(self.e._draft_invalid)
        self.assertEqual(project_to_json(self.w.cfg,self.w.nozzles),accepted)

    def test_unparseable_cell_and_failed_radius_are_restored_on_redo(self):
        table=self.e.primitive_table;row=self.row(table,'reference-arc')
        self.edit_number(table,'reference-arc',9,'3')
        self.assertEqual(table.item(row,9).text(),'3')
        self.w.undo_stack.undo();self.assertFalse(self.e._draft_invalid)
        self.w.undo_stack.redo();self.assertTrue(self.e._draft_invalid)
        self.assertEqual(table.item(row,9).text(),'3')
        table.item(row,9).setText('не число')
        self.assertEqual(self.w.undo_stack.count(),2)
        self.w.undo_stack.undo();self.assertEqual(table.item(row,9).text(),'3')
        self.w.undo_stack.redo();self.assertEqual(table.item(row,9).text(),'не число')

    def test_valid_edit_draft_and_reset_keep_their_chronological_order(self):
        original=project_to_json(self.w.cfg,self.w.nozzles)
        self.edit_number(self.e.primitive_table,'reference-point',3,'14.125')
        accepted=project_to_json(self.w.cfg,self.w.nozzles)
        invalid=self.invalidate_wall(self.w.cfg.r_out_mm+5.)
        self.e.reset_draft();self.assertEqual(self.w.undo_stack.count(),3)
        self.w.undo_stack.undo();self.assertEqual(self.e.raw,invalid)
        self.assertEqual(project_to_json(self.w.cfg,self.w.nozzles),accepted)
        self.w.undo_stack.undo();self.assertFalse(self.e._draft_invalid)
        self.assertEqual(self.record('reference-point')['start'][0],14.125)
        self.assertEqual(project_to_json(self.w.cfg,self.w.nozzles),accepted)
        self.w.undo_stack.undo();self.assertEqual(project_to_json(self.w.cfg,self.w.nozzles),original)

    def test_pasting_multiple_invalid_cells_is_one_history_action(self):
        original=deepcopy(self.e.raw)
        table=self.e.primitive_table;row=self.row(table,'inner-1')
        table.setCurrentCell(row,5)
        table.paste_text('30\t'+str(self.w.cfg.r_out_mm+5.))
        self.assertTrue(self.e._draft_invalid)
        self.assertEqual(self.w.undo_stack.count(),1)
        invalid=deepcopy(self.e.raw)
        self.w.undo_stack.undo();self.assertEqual(self.e.raw,original)
        self.w.undo_stack.redo();self.assertEqual(self.e.raw,invalid)

    def test_undo_reset_from_another_tab_redraws_restored_draft_when_reopened(self):
        draft=self.invalidate_wall(self.w.cfg.r_out_mm+5.)
        expected=self.record('inner-1')['end'][1]
        self.e.reset_draft();self.w.tabs.setCurrentIndex(1);self.app.processEvents()
        self.w.undo_stack.undo();self.assertEqual(self.e.raw,draft)
        self.w.tabs.setCurrentIndex(0);self.w.geometry_pages.setCurrentIndex(1);self.app.processEvents()
        handle=next(h for h in self.e.handles if h.identity=='inner-1' and h.role=='end')
        self.assertAlmostEqual(-handle.pos().y(),expected)
        self.assertTrue(self.e._draft_invalid)

    def test_accepted_undo_uses_incremental_geometry_restore_and_no_second_rebuild(self):
        self.edit_number(self.e.primitive_table,'reference-point',3,'14.125')
        with patch.object(self.w,'_fill_controls',wraps=self.w._fill_controls) as controls,patch.object(self.e,'rebuild',wraps=self.e.rebuild) as rebuild:
            self.w.undo_stack.undo();self.w.undo_stack.redo()
        self.assertEqual(controls.call_count,0)
        self.assertEqual(rebuild.call_count,0)
        self.assertEqual(self.record('reference-point')['start'][0],14.125)

    def test_command_snapshots_remain_immutable_after_restore_and_further_edits(self):
        self.edit_number(self.e.primitive_table,'reference-point',3,'14.125')
        command=self.w.undo_stack.command(0)
        before=deepcopy(command.before);after=deepcopy(command.after)
        project_before=deepcopy(command.project_before);project_after=deepcopy(command.project_after)
        self.w.undo_stack.undo();self.w.undo_stack.redo()
        self.edit_number(self.e.primitive_table,'reference-point',3,'15.5')
        self.assertEqual(command.before,before);self.assertEqual(command.after,after)
        self.assertEqual(command.project_before,project_before);self.assertEqual(command.project_after,project_after)
        self.w.undo_stack.undo();self.assertEqual(self.record('reference-point')['start'][0],14.125)


if __name__=='__main__':unittest.main()
