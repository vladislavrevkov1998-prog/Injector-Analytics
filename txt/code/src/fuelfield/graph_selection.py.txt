"""Explicit graph selection; counting remains available as a convenient shortcut."""
from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import QToolButton,QMenu

class GraphSelection(QToolButton):
    changed=pyqtSignal()
    def __init__(self,count,parent=None):
        super().__init__(parent);self.count=count;self.options=[];self.selected=[];self.manual=False;self.identity=None;self.order=[]
        self.setText('Выбрать графики');self.setPopupMode(QToolButton.InstantPopup)
        self.popup=QMenu(self);self.setMenu(self.popup);self.count.valueChanged.connect(self._resize)
    def configure(self,options,order=None,identity=None):
        if identity!=self.identity:self.selected=[];self.manual=False;self.identity=identity
        self.options=list(options);keys=[key for key,label in self.options];self.order=list(order or keys)
        self.count.blockSignals(True);self.count.setMaximum(max(1,len(keys)));self.count.blockSignals(False)
        if self.manual:self.selected=[key for key in self.selected if key in keys]
        else:self.selected=self.order[:self.count.value()]
        for key in self.order:
            if len(self.selected)>=self.count.value():break
            if key not in self.selected:self.selected.append(key)
        self.selected=self.selected[:self.count.value()];self._menu()
    def _menu(self):
        self.popup.clear()
        for key,label in self.options:
            action=self.popup.addAction(label);action.setCheckable(True);action.setChecked(key in self.selected)
            action.setEnabled(not (len(self.selected)==1 and key in self.selected))
            action.toggled.connect(lambda checked,k=key:self._choose(k,checked))
        self.setEnabled(bool(self.options))
        self.setToolTip('Выберите произвольные графики; число выбранных обновляется автоматически.')
    def _choose(self,key,checked):
        selected=list(self.selected)
        if checked and key not in selected:selected.append(key)
        elif not checked and key in selected and len(selected)>1:selected.remove(key)
        self.selected=[k for k,label in self.options if k in selected];self.manual=True
        self.count.blockSignals(True);self.count.setValue(len(self.selected));self.count.blockSignals(False)
        self._menu();self.changed.emit()
    def _resize(self,value):
        self.selected=self.selected[:value]
        for key in self.order:
            if len(self.selected)>=value:break
            if key not in self.selected:self.selected.append(key)
