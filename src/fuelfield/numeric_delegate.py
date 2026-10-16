"""Fine numeric editing without rounding values merely by opening an editor."""
from .numeric_input import DecimalSpinBox
import math
from PyQt5.QtWidgets import QStyledItemDelegate,QDoubleSpinBox
from PyQt5.QtCore import Qt

class NumericDelegate(QStyledItemDelegate):
    def createEditor(self,parent,option,index):
        try:value=float(str(index.data()).replace(',','.'))
        except (ValueError,TypeError):return super().createEditor(parent,option,index)
        raw=index.data(Qt.UserRole)
        if isinstance(raw,(int,float)) and not isinstance(raw,bool):value=float(raw)
        elif isinstance(raw,(tuple,list)) and len(raw)==2 and isinstance(raw[1],(int,float)):value=float(raw[1])
        if not math.isfinite(value):return super().createEditor(parent,option,index)
        editor=DecimalSpinBox(parent);editor.setDecimals(12);editor.setRange(-1e15,1e15);editor.setSingleStep(.001)
        editor.setKeyboardTracking(False);editor._original_text=str(index.data());editor._original_value=value
        return editor
    def setEditorData(self,editor,index):
        if isinstance(editor,QDoubleSpinBox):editor.setValue(editor._original_value);editor._shown_value=editor.value()
        else:super().setEditorData(editor,index)
    def setModelData(self,editor,model,index):
        if isinstance(editor,QDoubleSpinBox):
            if editor.value()!=editor._shown_value:model.setData(index,format(editor.value(),'.15g'),Qt.EditRole)
        else:super().setModelData(editor,model,index)
