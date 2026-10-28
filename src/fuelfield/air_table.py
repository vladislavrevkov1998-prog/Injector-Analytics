"""Excel-compatible cell editing with a single validated paste transaction."""
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QKeySequence
from PyQt5.QtWidgets import QApplication, QAbstractItemView, QComboBox, QStyledItemDelegate, QTableWidget
from .numeric_delegate import NumericDelegate


class AirChoices(QStyledItemDelegate):
    def __init__(self, choices, parent=None):
        super().__init__(parent); self.choices = choices

    def createEditor(self, parent, option, index):
        box = QComboBox(parent)
        for key, label in self.choices.items():
            box.addItem(label, key)
        if index.data() and box.findData(index.data()) < 0:
            box.addItem(str(index.data()), index.data())
        return box

    def setEditorData(self, editor, index):
        editor.setCurrentIndex(max(0, editor.findData(index.data())))

    def setModelData(self, editor, model, index):
        model.setData(index, editor.currentData())


class AirSourceTable(QTableWidget):
    batchFinished = pyqtSignal()
    pasteRejected = pyqtSignal(str)

    def __init__(self, *args):
        super().__init__(*args)
        self.in_batch = False
        self.batch_invalid = False
        self.validate_changes = None
        self.setItemDelegate(NumericDelegate(self))
        self.setSelectionBehavior(QAbstractItemView.SelectItems)
        self.setSelectionMode(QAbstractItemView.ExtendedSelection)

    def paste_text(self, raw):
        start = self.currentIndex()
        if not start.isValid() or not raw:
            return False
        cells = [line.split('\t') for line in raw.rstrip('\r\n').splitlines()]
        if len(cells) == 1 and len(cells[0]) == 1:
            targets = [(i.row(), i.column(), cells[0][0]) for i in self.selectedIndexes()]
        else:
            targets = [(start.row()+dr, start.column()+dc, text)
                       for dr, line in enumerate(cells) for dc, text in enumerate(line)]
        changes = []
        for row, col, text in targets:
            if row >= self.rowCount() or col >= self.columnCount():
                self.pasteRejected.emit('Вставка выходит за границы таблицы; значения не изменены.')
                return False
            item = self.item(row, col)
            editable = item is not None and bool(item.flags() & Qt.ItemIsEditable)
            checkable = item is not None and item.data(Qt.UserRole+3) == 'bool' and bool(item.flags() & Qt.ItemIsUserCheckable)
            if editable or checkable:
                changes.append((row, col, text.strip()))
        if not changes:
            self.pasteRejected.emit('Выбранные ячейки защищены от редактирования.')
            return False
        try:
            if self.validate_changes:
                self.validate_changes(changes)
        except (ValueError, TypeError, KeyError, OverflowError) as exc:
            self.batch_invalid = True
            self.pasteRejected.emit('Вставка отменена целиком: ' + str(exc))
            return False
        self.in_batch = True; self.batch_invalid = False
        blocked = self.blockSignals(True)
        self.setUpdatesEnabled(False)
        try:
            for row, col, text in changes:
                item = self.item(row, col)
                if item.data(Qt.UserRole+3) == 'bool':
                    item.setCheckState(Qt.Checked if text.lower() in ('1', 'true', 'yes', 'да', 'вкл') else Qt.Unchecked)
                else:
                    item.setText(text)
        finally:
            self.blockSignals(blocked); self.in_batch = False
            self.setUpdatesEnabled(True)
        self.batchFinished.emit()
        return True

    def keyPressEvent(self, event):
        if event.matches(QKeySequence.Copy):
            indexes = self.selectedIndexes()
            if indexes:
                selected = {(i.row(), i.column()) for i in indexes}
                rows = range(min(i.row() for i in indexes), max(i.row() for i in indexes)+1)
                columns = range(min(i.column() for i in indexes), max(i.column() for i in indexes)+1)
                text = []
                for row in rows:
                    values = []
                    for col in columns:
                        item = self.item(row, col)
                        value = ''
                        if item is not None and (row, col) in selected:
                            value = ('true' if item.checkState() == Qt.Checked else 'false') if item.data(Qt.UserRole+3) == 'bool' else item.text()
                        values.append(value)
                    text.append('\t'.join(values))
                QApplication.clipboard().setText('\n'.join(text))
            return
        if event.matches(QKeySequence.Paste):
            self.paste_text(QApplication.clipboard().text()); return
        super().keyPressEvent(event)
