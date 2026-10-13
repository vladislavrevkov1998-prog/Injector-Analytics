"""Qt editors for experimental atomizer characteristics and geometry templates."""
from copy import deepcopy
from pathlib import Path
from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import (QWidget,QVBoxLayout,QHBoxLayout,QFormLayout,QComboBox,
    QLineEdit,QPushButton,QLabel,QTableWidgetItem,QMessageBox,QFileDialog,QInputDialog)
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from .characteristics import ExperimentalCharacteristic,flow_to_kg_h,flow_from_kg_h
from .data_io import import_characteristic_file,write_characteristic_csv,_xlsx_sheets
from .v02 import GEOMETRY_KEYS,NOZZLE_KINDS,remove_atomizer,validate_catalog

PARAMETER_LABELS=dict(alpha_deg='α полный, °',r_nozzle_mm='R_сопла, мм',h0_mm='h0, мм',
    k_linear='k лин.',k_quadratic='k квадр.',power='p',z_ref_mm='z ref, мм',
    inner_angle_deg='α внутр., °',outer_angle_deg='α наруж., °',clock_deg='Clock, °',
    decay_length_mm='L затух., мм',diffusion_deg_sqrt_mm='σ / √мм',rotation_deg_per_mm='ω °/мм')


class CatalogEditor(QWidget):
    changed=pyqtSignal(object)
    def __init__(self,owner,table_class):
        super().__init__();self.owner=owner;self.loading=False;self.catalog=[];self.curve=None
        layout=QVBoxLayout(self);bar=QHBoxLayout()
        self.selector=QComboBox();bar.addWidget(QLabel('Тип распылителя:'));bar.addWidget(self.selector,1)
        for label,slot in [('Новый',self.new_type),('Удалить',self.delete_type)]:
            button=QPushButton(label);button.clicked.connect(slot);bar.addWidget(button)
        layout.addLayout(bar)
        form=QFormLayout();self.ident=QLineEdit();self.ident.setReadOnly(True);self.name=QLineEdit()
        form.addRow('ID',self.ident);form.addRow('Имя',self.name)
        self.kind=QComboBox();self.kind.addItem('Экспериментальная / ручная','experimental')
        self.kind.addItem('Дитякин — ожидается верификация источников','theoretical')
        self.kind.model().item(1).setEnabled(False)
        self.kind.setToolTip('Расчётный метод станет доступен после проверки уравнений и коэффициентов.')
        form.addRow('Характеристика',self.kind)
        self.unit=QComboBox();self.unit.addItem('кг/ч','kg/h');self.unit.addItem('л/мин','l/min')
        form.addRow('Единицы ввода Q',self.unit);layout.addLayout(form)
        body=QHBoxLayout();self.table=table_class(0,2)
        self.table.setHorizontalHeaderLabels(['Δp, кгс/см²','Q, кг/ч'])
        self.table.horizontalHeader().setStretchLastSection(True);body.addWidget(self.table,1)
        self.figure=Figure(figsize=(5,3),tight_layout=True);self.canvas=FigureCanvas(self.figure)
        body.addWidget(self.canvas,2);layout.addLayout(body)
        actions=QHBoxLayout()
        for label,slot in [('Добавить точку',self.add_point),('Удалить точки',self.delete_points),
                ('Импорт CSV/XLSX',self.import_file),('Экспорт CSV',self.export_file),
                ('Применить характеристику',self.apply)]:
            b=QPushButton(label);b.clicked.connect(slot);actions.addWidget(b)
        layout.addLayout(actions)
        self.status=QLabel('Введите измеренные точки. Изменения сохраняются кнопкой «Применить».')
        self.status.setWordWrap(True);layout.addWidget(self.status)
        self.selector.currentIndexChanged.connect(self.select_type)
        self.table.itemChanged.connect(self.preview)
        self.table.batchFinished.connect(self.preview)
        self.unit.currentIndexChanged.connect(self.change_unit)
        self._unit='kg/h'
        self._density=owner.cfg.density_kg_m3

    def load(self,catalog):
        selected=self.ident.text();self.loading=True
        self.catalog=deepcopy(catalog);self.selector.clear()
        for item in self.catalog:self.selector.addItem(f"{item['id']} · {item['name']}",item['id'])
        index=self.selector.findData(selected);self.selector.setCurrentIndex(max(0,index))
        self.loading=False;self.select_type()

    def select_type(self,*args):
        if self.loading:return
        selected=self.selector.currentData()
        item=next((a for a in self.catalog if a['id']==selected),None)
        self.loading=True
        if item is None:self.ident.clear();self.name.clear();self.table.setRowCount(0)
        else:
            self.ident.setText(item['id']);self.name.setText(item['name'])
            self.fill_points(item['points'])
        self.loading=False;self.preview()

    def fill_points(self,points):
        self._density=self.owner.cfg.density_kg_m3
        self.table.blockSignals(True);self.table.setRowCount(len(points))
        for row,(p,q) in enumerate(points):
            values=(p,flow_from_kg_h(q,self.unit.currentData(),self.owner.cfg.density_kg_m3))
            for col,value in enumerate(values):
                item=QTableWidgetItem(f'{value:.12g}')
                item.setData(256,(item.text(),float(value),float(q) if col==1 else float(p)))
                self.table.setItem(row,col,item)
        self.table.blockSignals(False)

    def read_curve(self,unit=None):
        unit=unit or self.unit.currentData();points=[]
        for row in range(self.table.rowCount()):
            values=[]
            for col in range(2):
                item=self.table.item(row,col)
                text=item.text().strip() if item else ''
                original=item.data(256) if item else None
                value=original[1] if original and original[0]==text else float(text.replace(',','.'))
                values.append(value)
            flow_item=self.table.item(row,1);original=flow_item.data(256) if flow_item else None
            canonical=original[2] if original and original[0]==flow_item.text().strip() else flow_to_kg_h(values[1],unit,self._density)
            points.append((values[0],canonical))
        return ExperimentalCharacteristic(points)

    def preview(self,*args):
        if self.loading or self.table.in_batch:return
        self.figure.clear();ax=self.figure.add_subplot(111)
        try:
            self.curve=self.read_curve()
            p,q=zip(*self.curve.points)
            q=[flow_from_kg_h(v,self.unit.currentData(),self.owner.cfg.density_kg_m3) for v in q]
            ax.plot(p,q,'o-',label='Измерения / линейная интерполяция')
            self.status.setText(f'{len(p)} измеренных точек; Δp {min(p):g}…{max(p):g} кгс/см². '
                               'Вне диапазона требуется явное разрешение экстраполяции.')
        except ValueError as exc:
            self.curve=None;self.status.setText(str(exc))
        ax.set_xlabel('Δp, кгс/см²');ax.set_ylabel('Q, '+self.unit.currentText());ax.grid(alpha=.25)
        self.canvas.draw_idle()

    def new_type(self):
        taken={a['id'] for a in self.catalog};i=1
        while f'atomizer-{i}' in taken:i+=1
        ident=f'atomizer-{i}'
        self.catalog.append(dict(id=ident,name='Новый тип',kind='experimental',points=[]))
        self.selector.addItem(ident,ident);self.selector.setCurrentIndex(self.selector.count()-1)

    def delete_type(self):
        ident=self.ident.text()
        if not ident:return
        try:
            catalog=remove_atomizer(self.owner.cfg.atomizer_types,self.owner.nozzles,ident)
            if any(a['id']==ident for a in self.owner.cfg.atomizer_types):self.changed.emit(catalog)
            else:self.load([a for a in self.catalog if a['id']!=ident])
        except ValueError as exc:QMessageBox.warning(self,'Удаление типа',str(exc))

    def add_point(self):
        row=self.table.rowCount();self.table.blockSignals(True);self.table.insertRow(row)
        self.table.setItem(row,0,QTableWidgetItem(''));self.table.setItem(row,1,QTableWidgetItem(''))
        self.table.blockSignals(False);self.table.setCurrentCell(row,0);self.preview()

    def delete_points(self):
        rows=sorted({i.row() for i in self.table.selectedIndexes()},reverse=True)
        for row in rows:self.table.removeRow(row)
        self.preview()

    def change_unit(self,*args):
        if self.loading:return
        try:
            if self.table.rowCount():curve=self.read_curve(self._unit)
            else:curve=None
        except ValueError as exc:
            self.unit.blockSignals(True);self.unit.setCurrentIndex(self.unit.findData(self._unit));self.unit.blockSignals(False)
            QMessageBox.warning(self,'Единицы',str(exc));return
        self._unit=self.unit.currentData()
        if curve:self.fill_points(curve.points)
        self.table.setHorizontalHeaderLabels(['Δp, кгс/см²','Q, '+self.unit.currentText()]);self.preview()

    def refresh_density(self):
        if self._density==self.owner.cfg.density_kg_m3:return
        try:curve=self.read_curve() if self.table.rowCount() else None
        except ValueError:
            self._density=self.owner.cfg.density_kg_m3
            self.status.setText('Плотность изменена: завершите и проверьте незавершённые точки.')
            return
        self._density=self.owner.cfg.density_kg_m3
        if curve:self.fill_points(curve.points)
        self.preview()

    def apply(self):
        try:
            curve=self.read_curve()
            if not self.ident.text() or not self.name.text().strip():raise ValueError('Создайте тип и укажите имя.')
            entry=dict(id=self.ident.text(),name=self.name.text().strip(),kind='experimental',points=[list(p) for p in curve.points])
            catalog=deepcopy(self.owner.cfg.atomizer_types)
            index=next((i for i,a in enumerate(catalog) if a['id']==entry['id']),None)
            if index is None:catalog.append(entry)
            else:catalog[index]=entry
            validate_catalog(catalog,self.owner.cfg.nozzle_presets)
            self.changed.emit(catalog)
        except ValueError as exc:QMessageBox.warning(self,'Характеристика',str(exc))

    def import_file(self):
        path,_=QFileDialog.getOpenFileName(self,'Расходная характеристика','','CSV (*.csv);;Excel (*.xlsx)')
        if not path:return
        try:
            sheet=None
            if Path(path).suffix.lower()=='.xlsx':
                names=[name for name,_ in _xlsx_sheets(Path(path))]
                if len(names)>1:
                    sheet,ok=QInputDialog.getItem(self,'Лист XLSX','Лист с Δp,Q:',names,0,False)
                    if not ok:return
            curve,_=import_characteristic_file(path,self.unit.currentData(),self.owner.cfg.density_kg_m3,sheet)
            self.fill_points(curve.points);self.preview()
        except (ValueError,OSError,KeyError) as exc:QMessageBox.warning(self,'Импорт',str(exc))

    def export_file(self):
        try:
            curve=self.read_curve()
            path,_=QFileDialog.getSaveFileName(self,'Экспорт характеристики','flow_curve.csv','CSV (*.csv)')
            if path:write_characteristic_csv(path,curve)
        except (ValueError,OSError) as exc:QMessageBox.warning(self,'Экспорт',str(exc))


class PresetEditor(QWidget):
    changed=pyqtSignal(object)
    def __init__(self,owner,table_class):
        super().__init__();self.owner=owner;layout=QVBoxLayout(self)
        self.selector=QComboBox()
        for key,label in NOZZLE_KINDS.items():self.selector.addItem(label,key)
        layout.addWidget(self.selector)
        self.table=table_class(len(GEOMETRY_KEYS),2)
        self.table.setHorizontalHeaderLabels(['Параметр','Значение шаблона'])
        self.table.horizontalHeader().setStretchLastSection(True);layout.addWidget(self.table)
        button=QPushButton('Применить шаблон');button.clicked.connect(self.apply);layout.addWidget(button)
        self.selector.currentIndexChanged.connect(self.load)
        note=QLabel('Начальные шаблоны демонстрационные. Изменения затрагивают только параметры без индивидуального override. '
                    'Для выбранных форсунок сброс overrides доступен на вкладке 02.')
        note.setWordWrap(True);layout.addWidget(note)

    def load(self,*args):
        from PyQt5.QtCore import Qt
        params=self.owner.cfg.nozzle_presets[self.selector.currentData()]
        for row,key in enumerate(GEOMETRY_KEYS):
            label=QTableWidgetItem(PARAMETER_LABELS[key]);label.setFlags(Qt.ItemIsEnabled)
            self.table.setItem(row,0,label)
            value=QTableWidgetItem(f'{params[key]:.12g}');value.setData(256,(value.text(),params[key]))
            self.table.setItem(row,1,value)

    def apply(self):
        try:
            params={}
            for row,key in enumerate(GEOMETRY_KEYS):
                item=self.table.item(row,1);original=item.data(256)
                params[key]=original[1] if original and original[0]==item.text() else float(item.text().replace(',','.'))
            presets=deepcopy(self.owner.cfg.nozzle_presets);presets[self.selector.currentData()]=params
            validate_catalog(self.owner.cfg.atomizer_types,presets);self.changed.emit(presets)
        except ValueError as exc:QMessageBox.warning(self,'Шаблон',str(exc))
