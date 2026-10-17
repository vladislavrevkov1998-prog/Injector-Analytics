"""Compact decimal editing: no fixed zero padding occupying fractional positions."""
from PyQt5.QtCore import Qt,QEvent
from PyQt5.QtGui import QKeyEvent
from PyQt5.QtWidgets import QDoubleSpinBox

class DecimalSpinBox(QDoubleSpinBox):
    def textFromValue(self,value):
        text=f'{value:.{self.decimals()}f}'
        if '.' in text:text=text.rstrip('0').rstrip('.')
        if text=='-0':text='0'
        return text.replace('.',self.locale().decimalPoint())
    def _normalized(self,text):
        separator=self.locale().decimalPoint();return text.replace(',',separator).replace('.',separator)
    def validate(self,text,pos):
        text=self._normalized(text);separator=self.locale().decimalPoint()
        if separator in text:
            while len(text.split(separator,1)[1])>self.decimals() and text.endswith('0'):text=text[:-1]
        return super().validate(text,min(pos,len(text)))
    def valueFromText(self,text):return super().valueFromText(self._normalized(text))
    def keyPressEvent(self,event):
        edit=self.lineEdit();separator=self.locale().decimalPoint();text=edit.text();cursor=edit.cursorPosition()
        if event.text().isdigit() and not edit.hasSelectedText() and separator in text:
            position=text.index(separator)
            # Insertion into an already full fraction consumes only an insignificant zero.
            if cursor>position and len(text)-position-1>=self.decimals() and text.endswith('0'):
                edit.setText(text[:-1]);edit.setCursorPosition(min(cursor,len(text)-1))
        if event.text() in ('.',',') and event.text()!=separator:
            event=QKeyEvent(QEvent.KeyPress,event.key(),event.modifiers(),separator,event.isAutoRepeat(),event.count())
        super().keyPressEvent(event)
