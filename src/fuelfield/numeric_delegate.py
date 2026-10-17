"""Fine numeric editing without rounding values merely by opening an editor."""
import math
from PyQt5.QtWidgets import QStyledItemDelegate,QDoubleSpinBox
from PyQt5.QtCore import Qt

class NumericDelegate(QStyledItemDelegate):
    def createEditor(self,parent,option,index):
        try:value=float(str(index.data()).replace(',','.'))
        except (ValueError,TypeError):return super().createEditor(parent,option,index)
        if not math.isfinite(value):return super().createEditor(parent,option,index)
        editor=QDoubleSpinBox(parent);editor.setDecimals(12);editor.setRange(-1e15,1e15);editor.setSingleStep(.001)
        editor.setKeyboardTracking(False);editor._original_text=str(index.data());editor._original_value=value
        return editor
    def setEditorData(self,editor,index):
        if isinstance(editor,QDoubleSpinBox):editor.setValue(editor._original_value);editor._shown_value=editor.value()
        else:super().setEditorData(editor,index)
    def setModelData(self,editor,model,index):
        if isinstance(editor,QDoubleSpinBox):
            text=editor._original_text if editor.value()==editor._shown_value else format(editor.value(),'.12g')
            model.setData(index,text,Qt.EditRole)
        else:super().setModelData(editor,model,index)
