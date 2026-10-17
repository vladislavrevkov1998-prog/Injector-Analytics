"""Qt editors for measured/calculated atomizer characteristics and templates."""
from copy import deepcopy
from pathlib import Path
from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import (QWidget,QVBoxLayout,QHBoxLayout,QFormLayout,QComboBox,
    QLineEdit,QPushButton,QLabel,QTableWidgetItem,QMessageBox,QFileDialog,QInputDialog,QDoubleSpinBox)
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from .characteristics import ExperimentalCharacteristic,flow_to_kg_h,flow_from_kg_h
from .data_io import import_characteristic_file,write_characteristic_csv,_xlsx_sheets
from .v02 import GEOMETRY_KEYS,NOZZLE_KINDS,remove_atomizer,validate_catalog
from .dityakin import DityakinParameters,DityakinCharacteristic
import numpy as np

PARAMETER_LABELS=dict(z_tip_mm='z среза, мм',alpha_deg='α полный, °',r_nozzle_mm='R_сопла, мм',h0_mm='h0, мм',
    k_linear='k лин.',k_quadratic='k квадр.',power='p',z_ref_mm='z ref, мм',
    inner_angle_deg='α внутр., °',outer_angle_deg='α наруж., °',clock_deg='Clock, °',
    decay_length_mm='L затух., мм',diffusion_deg_sqrt_mm='σ / √мм',rotation_deg_per_mm='ω °/мм')

HYDRAULIC_LABELS=dict(outlet_diameter_mm='Диаметр сопла dс, мм',
    inlet_diameter_mm='Диаметр круглого входа dвх, мм',inlet_count='Число входов n',
    swirl_arm_mm='Плечо закручивания R, мм',chamber_height_mm='Высота камеры Lк, мм',
    inlet_length_mm='Длина входа Lвх, мм',outlet_length_mm='Длина сопла Lс, мм',
    cone_angle_deg='Угол входного конуса ψ, °',kinematic_viscosity_mm2_s='Кинематическая вязкость ν, мм²/с')

HYDRAULIC_RESULTS=dict(geometric_a='A геометрическая',deformation='ε деформации',
    theta='θ трения',a_active='Aд (расчёт)',a_equivalent='Aₑ (расчёт)',phi='φ заполнения',
    re_inlet='Re одного входа',re_equivalent='Re суммарного входа',friction='λк',
    loss_inlet='Δ входа',loss_chamber='Δ камеры',loss_outlet='Δ сопла',
    total_loss='ΔΣ',mu='μ расхода',film_mm='Толщина плёнки, мм',mass_kg_h='G, кг/ч')


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
        self.kind.addItem('Дитякин — простая короткая форсунка','theoretical')
        self.kind.setToolTip('Монография 1977 г., гл. 2. Круглые тангенциальные входы, короткая камера и сопло.')
        form.addRow('Характеристика',self.kind)
        self.unit=QComboBox();self.unit.addItem('кг/ч','kg/h');self.unit.addItem('л/мин','l/min')
        form.addRow('Единицы ввода Q',self.unit);layout.addLayout(form)
        body=QHBoxLayout();self.table=table_class(0,2)
        self.table.setHorizontalHeaderLabels(['Δp, кгс/см²','Q, кг/ч'])
        self.table.horizontalHeader().setStretchLastSection(True);body.addWidget(self.table,1)
        self.theory_panel=QWidget();theory_layout=QVBoxLayout(self.theory_panel)
        self.theory_table=table_class(len(HYDRAULIC_LABELS)+len(HYDRAULIC_RESULTS),2)
        self.theory_table.setHorizontalHeaderLabels(['Гидравлический параметр','Значение'])
        self.theory_table.horizontalHeader().setStretchLastSection(True)
        self.theory_table.setMinimumHeight(310)
        theory_layout.addWidget(self.theory_table)
        self.plot_min=QDoubleSpinBox();self.plot_max=QDoubleSpinBox()
        for spin,value in ((self.plot_min,2.5),(self.plot_max,30.)):
            spin.setDecimals(4);spin.setRange(.0001,100000.);spin.setValue(value)
        graph_form=QFormLayout();graph_form.addRow('График: Δp от, кгс/см²',self.plot_min)
        graph_form.addRow('График: Δp до, кгс/см²',self.plot_max);theory_layout.addLayout(graph_form)
        self.diagnostic_pressure=QDoubleSpinBox();self.diagnostic_pressure.setDecimals(5)
        self.diagnostic_pressure.setRange(.00001,100000.)
        self.diagnostic_pressure.setValue(owner.cfg.target_pressure_kgf_cm2)
        graph_form.addRow('Расчётные строки: Δp, кгс/см²',self.diagnostic_pressure)
        note=QLabel('Геометрия здесь относится к гидравлическому распылителю. '
            'R_сопла и h0 поля орошения настраиваются отдельно. '
            'Начальные параметры демонстрационные; ν задавайте для жидкости при температуре измерения. '
            'Коэффициенты ε и ξвх интерполируются по графикам книги. '
            'Выход за область методики запрещён, включая режим разрешённой экспериментальной экстраполяции.')
        note.setWordWrap(True);theory_layout.addWidget(note);body.addWidget(self.theory_panel,1)
        self.figure=Figure(figsize=(5,3),tight_layout=True);self.canvas=FigureCanvas(self.figure)
        body.addWidget(self.canvas,2);layout.addLayout(body)
        actions=QHBoxLayout()
        self.point_actions=[]
        for label,slot in [('Добавить точку',self.add_point),('Удалить точки',self.delete_points),
                ('Импорт CSV/XLSX',self.import_file),('Экспорт CSV',self.export_file),
                ('Применить характеристику',self.apply)]:
            b=QPushButton(label);b.clicked.connect(slot);actions.addWidget(b)
            if slot!=self.apply:self.point_actions.append(b)
        layout.addLayout(actions)
        self.status=QLabel('Введите измеренные точки. Изменения сохраняются кнопкой «Применить».')
        self.status.setWordWrap(True);layout.addWidget(self.status)
        self.selector.currentIndexChanged.connect(self.select_type)
        self.table.itemChanged.connect(self.preview)
        self.table.batchFinished.connect(self.preview)
        self.unit.currentIndexChanged.connect(self.change_unit)
        self.kind.currentIndexChanged.connect(self.change_kind)
        self.theory_table.itemChanged.connect(self.preview)
        self.theory_table.batchFinished.connect(self.preview)
        self.plot_min.valueChanged.connect(self.preview);self.plot_max.valueChanged.connect(self.preview)
        self.diagnostic_pressure.valueChanged.connect(self.preview)
        self._unit='kg/h'
        self._density=owner.cfg.density_kg_m3
        self.fill_parameters(DityakinParameters().to_dict());self.change_kind()

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
        if item is None:
            self.ident.clear();self.name.clear();self.table.setRowCount(0);self.kind.setCurrentIndex(0)
        else:
            self.ident.setText(item['id']);self.name.setText(item['name'])
            self.kind.setCurrentIndex(self.kind.findData('theoretical' if item['kind']=='theoretical' else 'experimental'))
            self.fill_points(item.get('points',[]))
            self.fill_parameters(item.get('parameters',DityakinParameters().to_dict()))
            low,high=item.get('plot_pressure_kgf_cm2',[2.5,30.])
            self.plot_min.setValue(float(low));self.plot_max.setValue(float(high))
        self.loading=False;self.change_kind()

    def change_kind(self,*args):
        theoretical=self.kind.currentData()=='theoretical'
        self.table.setVisible(not theoretical);self.theory_panel.setVisible(theoretical)
        for action in self.point_actions:action.setEnabled(not theoretical)
        if not self.loading:
            if not theoretical:
                try:
                    curve=self.read_curve(self._unit) if self.table.rowCount() else None
                    if curve:self.fill_points(curve.points)
                    self._unit=self.unit.currentData()
                    self.table.setHorizontalHeaderLabels(['Δp, кгс/см²','Q, '+self.unit.currentText()])
                except ValueError as exc:self.status.setText(str(exc))
            self.preview()

    def fill_parameters(self,params):
        from PyQt5.QtCore import Qt
        self.theory_table.blockSignals(True)
        for row,(key,label) in enumerate(HYDRAULIC_LABELS.items()):
            item=QTableWidgetItem(label);item.setFlags(Qt.ItemIsEnabled);item.setToolTip(label)
            self.theory_table.setItem(row,0,item)
            value=QTableWidgetItem(f'{params[key]:.12g}');value.setData(256,(value.text(),params[key]))
            self.theory_table.setItem(row,1,value)
        self.theory_table.blockSignals(False)
        self.theory_table.resizeColumnToContents(0)
        self.fill_results(None)

    def fill_results(self,state):
        from PyQt5.QtCore import Qt
        self.theory_table.blockSignals(True)
        for row,(key,label) in enumerate(HYDRAULIC_RESULTS.items(),start=len(HYDRAULIC_LABELS)):
            if state is None:value='—'
            else:
                derived=dict(geometric_a=state.a_active*self.curve.parameters.deformation,
                    deformation=self.curve.parameters.deformation,theta=state.a_active/state.a_equivalent-1,
                    total_loss=state.loss_inlet+state.loss_chamber+state.loss_outlet,mass_kg_h=state.mass_kg_s*3600)
                number=derived[key] if key in derived else getattr(state,key)
                value=f'{number:.9g}'
            for col,text in enumerate((label,value)):
                item=QTableWidgetItem(text);item.setFlags(Qt.ItemIsEnabled|Qt.ItemIsSelectable)
                item.setToolTip('Рассчитано при Δp='+str(self.diagnostic_pressure.value())+' кгс/см²')
                self.theory_table.setItem(row,col,item)
        self.theory_table.blockSignals(False)

    def read_theory(self):
        params={}
        for row,key in enumerate(HYDRAULIC_LABELS):
            item=self.theory_table.item(row,1);text=item.text().strip() if item else ''
            original=item.data(256) if item else None
            params[key]=original[1] if original and original[0]==text else float(text.replace(',','.'))
        if self.plot_min.value()>=self.plot_max.value():
            raise ValueError('График: минимальное давление должно быть меньше максимального.')
        return DityakinCharacteristic(DityakinParameters(**params),self.owner.cfg.density_kg_m3)

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
        if self.loading or self.table.in_batch or self.theory_table.in_batch:return
        self.figure.clear();ax=self.figure.add_subplot(111)
        try:
            if self.kind.currentData()=='theoretical':
                self.curve=self.read_theory()
                pressures=np.geomspace(self.plot_min.value(),self.plot_max.value(),48)
                values=[self.curve.evaluate(float(p)).flow_kg_h for p in pressures]
                values=[flow_from_kg_h(q,self.unit.currentData(),self.owner.cfg.density_kg_m3) for q in values]
                ax.plot(pressures,values,label='Дитякин (расчёт)')
                state=self.curve.hydraulics(self.diagnostic_pressure.value());self.fill_results(state)
                self.status.setText(f'Расчёт по Дитякину, гл. 2 (1977); ρ={self.owner.cfg.density_kg_m3:g} кг/м³. '
                    f'При Δp={self.diagnostic_pressure.value():g}: Aд={state.a_active:.4g}, Aₑ={state.a_equivalent:.4g}, '
                    f'φ={state.phi:.4g}, μ={state.mu:.4g}, λк={state.friction:.4g}, '
                    f'Reвх={state.re_inlet:.5g}; ΔΣ={state.loss_inlet+state.loss_chamber+state.loss_outlet:.4g}. '
                    'Потери в камере: численное интегрирование уравнения (41).')
                return
            self.curve=self.read_curve()
            p,q=zip(*self.curve.points)
            q=[flow_from_kg_h(v,self.unit.currentData(),self.owner.cfg.density_kg_m3) for v in q]
            ax.plot(p,q,'o-',label='Измерения / линейная интерполяция')
            self.status.setText(f'{len(p)} измеренных точек; Δp {min(p):g}…{max(p):g} кгс/см². '
                               'Вне диапазона требуется явное разрешение экстраполяции.')
        except ValueError as exc:
            self.curve=None;self.fill_results(None);self.status.setText(str(exc))
        finally:
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
        if self.kind.currentData()=='theoretical':
            self.preview();return
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
        if self.kind.currentData()=='theoretical':
            self.preview();return
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
            if not self.ident.text() or not self.name.text().strip():raise ValueError('Создайте тип и укажите имя.')
            entry=dict(id=self.ident.text(),name=self.name.text().strip(),kind=self.kind.currentData())
            if self.kind.currentData()=='theoretical':
                curve=self.read_theory()
                curve.evaluate(self.plot_min.value());curve.evaluate(self.plot_max.value())
                entry.update(method='dityakin_short_v1',parameters=curve.parameters.to_dict(),
                             plot_pressure_kgf_cm2=[self.plot_min.value(),self.plot_max.value()])
            else:
                curve=self.read_curve();entry.update(points=[list(p) for p in curve.points])
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
