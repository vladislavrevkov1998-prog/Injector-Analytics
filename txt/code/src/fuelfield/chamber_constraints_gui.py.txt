"""Explicit endpoint selection for joints, including overlapping scene points."""
from PyQt5.QtWidgets import QDialog,QVBoxLayout,QHBoxLayout,QLabel,QComboBox,QDialogButtonBox


class ConstraintDialog(QDialog):
    def __init__(self,raw,kind,parent=None):
        super().__init__(parent);self.kind=kind
        self.setWindowTitle('Совпадение точек' if kind=='coincident' else 'Касательность дуги и отрезка')
        self.resize(560,200);layout=QVBoxLayout(self)
        text=('Первая точка переместится ко второй. Зафиксированные кривые остаются на месте.' if kind=='coincident' else
              'Выберите конец дуги и конец отрезка. Отрезок останется на месте и будет зафиксирован; центр и радиус дуги подстроятся. Противоположный конец дуги сохраняется.')
        note=QLabel(text);note.setWordWrap(True);layout.addWidget(note);self.a=QComboBox();self.b=QComboBox()
        for box in (self.a,self.b):
            for p in raw['primitives']:
                if kind=='tangent' and p['kind'] not in ('line','arc'):continue
                for role,label in [('start','начало'),('end','конец')]:
                    if role in p:box.addItem(p['id']+' · '+label,dict(primitive_id=p['id'],endpoint=role))
        if self.b.count()>1:self.b.setCurrentIndex(self.b.count()-1)
        for label,box in [('Первая точка:',self.a),('Вторая точка:',self.b)]:
            row=QHBoxLayout();row.addWidget(QLabel(label));row.addWidget(box,1);layout.addLayout(row)
        buttons=QDialogButtonBox(QDialogButtonBox.Ok|QDialogButtonBox.Cancel);buttons.button(QDialogButtonBox.Ok).setText('Создать связь')
        buttons.accepted.connect(self.accept);buttons.rejected.connect(self.reject);layout.addWidget(buttons)
