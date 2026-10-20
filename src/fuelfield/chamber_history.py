"""One shared Undo timeline for accepted geometry and incomplete sketch drafts.

Snapshots are taken at the end of actions, never at drag frames. A draft can
therefore be undone without publishing invalid geometry to the calculation.
snapshot() and project_to_json() detach mutable editor data. Commands take
ownership of these snapshots; checkpoints only replace their reference and
must never mutate a saved snapshot. Restoration copies back into the editor.
"""
from copy import deepcopy

from PyQt5.QtWidgets import QUndoCommand

from .core import project_to_json


def snapshot(editor):
    return dict(raw=deepcopy(editor.raw), nozzles=deepcopy(editor.working_nozzles),
                invalid=editor._draft_invalid, selected=editor.selected,
                message=editor.status.text(), cells=deepcopy(getattr(editor,'_draft_cell_errors',{})))


def checkpoint(editor):
    editor._draft_checkpoint = snapshot(editor)


def restore_cells(editor):
    for (table_index,row,col),text in getattr(editor,'_draft_cell_errors',{}).items():
        table=editor.tables[table_index]
        if row<table.rowCount() and table.item(row,col) is not None:
            table.blockSignals(True)
            try:table.item(row,col).setText(text)
            finally:table.blockSignals(False)


def restore(editor, draft):
    editor.cancel_construction()
    matches=(editor.raw==draft['raw'] and editor.working_nozzles==draft['nozzles']
             and editor._draft_invalid==draft['invalid']
             and getattr(editor,'_draft_cell_errors',{})==draft['cells']
             and not editor._pending)
    if matches:
        # apply_geometry_state may already have updated the existing paths.
        # A second full 140-row rebuild would discard that incremental work.
        previous_loading=editor.loading;editor.loading=True
        try:
            editor.selected=draft['selected']
            for handle in editor.handles:
                handle.setSelected(handle.identity==editor.selected)
                handle.setZValue(10 if handle.identity==editor.selected else 8)
            editor.highlight_selection()
        finally:editor.loading=previous_loading
        editor.message(draft['message'],draft['invalid'])
        editor._draft_checkpoint=draft
        return
    editor.raw = deepcopy(draft['raw'])
    editor.working_nozzles = deepcopy(draft['nozzles'])
    editor._draft_invalid = draft['invalid']
    editor._draft_cell_errors = deepcopy(draft['cells'])
    editor.selected = draft['selected']
    editor._pending=True
    if editor.isVisible():editor.rebuild()
    editor.message(draft['message'], draft['invalid'])
    editor._draft_checkpoint=draft


class GeometryDraftEdit(QUndoCommand):
    """Includes both the calculable project and the independent visible draft."""
    def __init__(self, editor, before, after, project_before, project_after, label):
        super().__init__(label)
        self.editor = editor
        self.before, self.after = before, after
        self.project_before = project_before
        self.project_after = project_after
        self.first = True

    def _restore(self, project, draft):
        owner = self.editor.owner
        if project_to_json(owner.cfg, owner.nozzles) != project:
            owner.apply_geometry_state(project,self.text(),record_history=False)
        restore(self.editor, draft)

    def undo(self):
        self._restore(self.project_before, self.before)

    def redo(self):
        if self.first:
            self.first = False
        else:
            self._restore(self.project_after, self.after)


def record(editor, label, before=None, project_before=None):
    if before is None:
        before = editor._draft_checkpoint if hasattr(editor,'_draft_checkpoint') else snapshot(editor)
    after = snapshot(editor)
    project_after = project_to_json(editor.owner.cfg, editor.owner.nozzles)
    project_before = project_before if project_before is not None else project_after
    # Selection and status messages are presentation, not editing actions.
    changed = any(before[key] != after[key] for key in ('raw', 'nozzles', 'invalid', 'cells'))
    if changed or project_before != project_after:
        editor.owner.undo_stack.push(GeometryDraftEdit(editor, before, after,
            project_before, project_after, label))
    editor._draft_checkpoint=after
