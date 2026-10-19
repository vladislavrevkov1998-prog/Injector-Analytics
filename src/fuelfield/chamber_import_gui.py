"""Previewed point import. No project mutation until the Import button."""
from pathlib import Path
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (QDialog,QVBoxLayout,QHBoxLayout,QLabel,QComboBox,
    QSpinBox,QTableWidget,QTableWidgetItem,QDialogButtonBox)
from .chamber_import import read_point_tables,parse_points


class PointImportDialog(QDialog):
    def __init__(self,path,parent=None):
        super().__init__(parent);self.path=Path(path);self.tables=read_point_tables(path);self.points=[];self._loading=False
        self.setWindowTitle('Импорт точек XLSX');self.resize(800,580);layout=QVBoxLayout(self)
        note=QLabel('Точки будут добавлены в Sketcher. Существующие примитивы сохраняются. Точки можно использовать как опорные при построении линий и дуг.');note.setWordWrap(True);layout.addWidget(note)
        row=QHBoxLayout();self.sheet=QComboBox()
        for name,_ in self.tables:self.sheet.addItem(name)
        row.addWidget(QLabel('Лист:'));row.addWidget(self.sheet,1);self.first_row=QSpinBox();self.first_row.setRange(1,1048576);row.addWidget(QLabel('Первая строка данных:'));row.addWidget(self.first_row);layout.addLayout(row)
        row=QHBoxLayout();self.z_column=QComboBox();self.r_column=QComboBox();self.unit=QComboBox();self.unit.addItems(['mm','cm','m']);self.wall=QComboBox()
        for text,value in [('Опорные точки','reference'),('Внутренняя стенка','inner'),('Наружная стенка','outer')]:self.wall.addItem(text,value)
        for text,widget in [('z / x:',self.z_column),('R:',self.r_column),('Единицы файла:',self.unit),('Принадлежность:',self.wall)]:row.addWidget(QLabel(text));row.addWidget(widget)
        layout.addLayout(row);self.preview=QTableWidget(0,3);self.preview.setHorizontalHeaderLabels(['Строка Excel','z, мм','R, мм']);self.preview.horizontalHeader().setStretchLastSection(True);layout.addWidget(self.preview,1)
        self.status=QLabel();self.status.setWordWrap(True);layout.addWidget(self.status)
        self.buttons=QDialogButtonBox(QDialogButtonBox.Ok|QDialogButtonBox.Cancel);self.buttons.button(QDialogButtonBox.Ok).setText('Импортировать');self.buttons.accepted.connect(self.accept);self.buttons.rejected.connect(self.reject);layout.addWidget(self.buttons)
        self.sheet.currentIndexChanged.connect(self.change_sheet)
        for box in (self.z_column,self.r_column,self.unit,self.wall):box.currentIndexChanged.connect(self.refresh)
        self.first_row.valueChanged.connect(self.refresh);self.change_sheet()

    def change_sheet(self,*_):
        self._loading=True;rows=self.tables[self.sheet.currentIndex()][1];count=max(len(values) for _,values in rows);first_number,first=rows[0]
        for box in (self.z_column,self.r_column):
            box.clear()
            for col in range(count):
                index=col+1;letters=''
                while index:index,rem=divmod(index-1,26);letters=chr(65+rem)+letters
                value=first[col] if col<len(first) else ''
                box.addItem(f'{letters}: {value}' if value is not None else letters,col)
        def find(names):
            return next((i for i,v in enumerate(first) if isinstance(v,str) and v.strip().lower().replace(' ','').split(',')[0].split('[')[0] in names),None)
        z=find(('z','x','z_mm','x_mm'));r=find(('r','radius','r_mm','radius_mm','радиус'))
        self.z_column.setCurrentIndex(z if z is not None else 0);self.r_column.setCurrentIndex(r if r is not None else min(1,count-1))
        # An invalid numeric row is data, not an inferred header. Never skip it.
        header=z is not None and r is not None
        self.first_row.setValue(first_number+int(header));self._loading=False;self.refresh()

    def refresh(self,*_):
        if self._loading:return
        try:
            self.points=parse_points(self.tables[self.sheet.currentIndex()][1],self.z_column.currentIndex(),self.r_column.currentIndex(),self.first_row.value(),self.unit.currentText())
            self.preview.setRowCount(min(200,len(self.points)))
            for i,point in enumerate(self.points[:200]):
                for j,value in enumerate(point):
                    item=QTableWidgetItem(format(value,'.12g'));item.setFlags(Qt.ItemIsSelectable|Qt.ItemIsEnabled);self.preview.setItem(i,j,item)
            self.status.setText(f'Готово к импорту: {len(self.points)} точек. '+('Показаны первые 200.' if len(self.points)>200 else ''));self.status.setStyleSheet('');self.buttons.button(QDialogButtonBox.Ok).setEnabled(True)
        except ValueError as exc:
            self.points=[];self.preview.setRowCount(0);self.status.setText(str(exc));self.status.setStyleSheet('color:#a32138');self.buttons.button(QDialogButtonBox.Ok).setEnabled(False)

    def accept(self):
        self.refresh()
        if self.points:super().accept()
