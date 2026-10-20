"""Explicit endpoint selection for joints, including overlapping scene points."""
from PyQt5.QtWidgets import QDialog,QVBoxLayout,QHBoxLayout,QLabel,QComboBox,QDialogButtonBox,QCheckBox
from .chamber_constraints import tangent_references


class ConstraintDialog(QDialog):
    def __init__(self,raw,kind,parent=None):
        super().__init__(parent);self.kind=kind
        self.setWindowTitle('Совпадение точек' if kind=='coincident' else 'Касательность двух кривых')
        self.resize(560,200);layout=QVBoxLayout(self)
        text=('Первая точка переместится ко второй. Зафиксированные кривые остаются на месте.' if kind=='coincident' else
              'Выберите два объекта: дуга–отрезок или дуга–дуга. Общая связь совпадения определяет контакт, иначе используются ближайшие концы. Отрезок сохраняется и фиксируется. Для двух дуг первая подгоняется ко второй, опорной; зафиксированная дуга всегда опорная. Противоположный конец подгоняемой дуги сохраняется.')
        note=QLabel(text);note.setWordWrap(True);layout.addWidget(note);self.a=QComboBox();self.b=QComboBox()
        for box in (self.a,self.b):
            for p in raw['primitives']:
                if kind=='tangent' and p['kind'] not in ('line','arc'):continue
                if kind=='tangent':box.addItem(p['id']+' · '+('дуга' if p['kind']=='arc' else 'отрезок'),p['id']);continue
                for role,label in [('start','начало'),('end','конец')]:
                    if role in p:box.addItem(p['id']+' · '+label,dict(primitive_id=p['id'],endpoint=role))
        if self.b.count()>1:self.b.setCurrentIndex(self.b.count()-1)
        for label,box in [('Первая кривая:' if kind=='tangent' else 'Первая точка:',self.a),('Вторая кривая:' if kind=='tangent' else 'Вторая точка:',self.b)]:
            row=QHBoxLayout();row.addWidget(QLabel(label));row.addWidget(box,1);layout.addLayout(row)
        self.raw=raw
        if kind=='tangent':
            self.manual=QCheckBox('Указать контакт вручную (если общих концов несколько)');layout.addWidget(self.manual)
            self.a_endpoint=QComboBox();self.b_endpoint=QComboBox();row=QHBoxLayout()
            for name,box in [('Конец первой:',self.a_endpoint),('Конец второй:',self.b_endpoint)]:
                box.addItem('Начало','start');box.addItem('Конец','end');box.setEnabled(False)
                self.manual.toggled.connect(box.setEnabled);row.addWidget(QLabel(name));row.addWidget(box)
            layout.addLayout(row)
        buttons=QDialogButtonBox(QDialogButtonBox.Ok|QDialogButtonBox.Cancel);buttons.button(QDialogButtonBox.Ok).setText('Создать связь')
        buttons.accepted.connect(self.accept);buttons.rejected.connect(self.reject);layout.addWidget(buttons)
    def references(self):
        if self.kind!='tangent':return self.a.currentData(),self.b.currentData()
        if self.manual.isChecked():
            a=dict(primitive_id=self.a.currentData(),endpoint=self.a_endpoint.currentData());b=dict(primitive_id=self.b.currentData(),endpoint=self.b_endpoint.currentData())
            records={p['id']:p for p in self.raw['primitives']};first,second=records[a['primitive_id']],records[b['primitive_id']]
            if first['kind']==second['kind']=='arc' and first.get('fixed',False) and not second.get('fixed',False):a,b=b,a
            return a,b
        return tangent_references(self.raw,self.a.currentData(),self.b.currentData())
