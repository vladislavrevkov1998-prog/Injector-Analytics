"""Independent v0.5 session editor and worker-driven quasisteady results."""
import json,hashlib
from pathlib import Path
from copy import deepcopy
from dataclasses import replace
import numpy as np
from PyQt5.QtCore import Qt,QObject,QThread,pyqtSignal,pyqtSlot,QAbstractTableModel,QModelIndex
from PyQt5.QtGui import QColor,QKeySequence
from PyQt5.QtWidgets import (QWidget,QVBoxLayout,QHBoxLayout,QFormLayout,QLabel,QPushButton,QComboBox,QLineEdit,
    QSpinBox,QCheckBox,QFileDialog,QMessageBox,QTabWidget,QTableWidgetItem,QTableView,QHeaderView,QDialog,
    QDialogButtonBox,QProgressBar,QApplication,QSplitter,QGroupBox,QScrollArea)
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from .numeric_input import DecimalSpinBox
from .numeric_delegate import NumericDelegate
from .core import project_to_json
from .measurement import MeasurementSession,MeasurementStep,PressureSeries
from .measurement_io import read_tables,import_steps,import_pressures,load_session,save_session,write_report,boolean
from .transient import calculate_session,geometry_signature,CalculationCancelled
from .coordinates import plot_x,period,x_label,decorate_axes
from .plot_interaction import attach_interaction
from .axes_settings import open_axes_settings


def spin(value,low=-1e9,high=1e9,step=.001,decimals=9):
    s=DecimalSpinBox();s.setDecimals(decimals);s.setRange(low,high);s.setSingleStep(step);s.setValue(value);s.setKeyboardTracking(False);return s

class RowsModel(QAbstractTableModel):
    headers=['Коллектор','t, с (сырые часы)','Давление','Единицы','Вид давления','Достоверно','reference_id','Сдвиг часов, с']
    def __init__(self,parent=None):super().__init__(parent);self.rows=[]
    def rowCount(self,parent=QModelIndex()):return 0 if parent.isValid() else len(self.rows)
    def columnCount(self,parent=QModelIndex()):return 0 if parent.isValid() else len(self.headers)
    def data(self,index,role=Qt.DisplayRole):
        if not index.isValid():return None
        value=self.rows[index.row()][index.column()]
        if role==Qt.UserRole:return value
        if role==Qt.CheckStateRole and index.column()==5:return Qt.Checked if value else Qt.Unchecked
        if role in (Qt.DisplayRole,Qt.EditRole):return '' if index.column()==5 or value is None else format(value,'.12g') if isinstance(value,float) else str(value)
        if role==Qt.ToolTipRole:return self.headers[index.column()]
    def headerData(self,section,orientation,role=Qt.DisplayRole):
        if role==Qt.DisplayRole:return self.headers[section] if orientation==Qt.Horizontal else str(section+1)
    def flags(self,index):return Qt.ItemIsEnabled|Qt.ItemIsSelectable|(Qt.ItemIsUserCheckable if index.column()==5 else Qt.ItemIsEditable)
    def setData(self,index,value,role=Qt.EditRole):
        if not index.isValid():return False
        if index.column()==5:
            try:value=value==Qt.Checked if role==Qt.CheckStateRole else boolean(value)
            except ValueError:return False
        elif role!=Qt.EditRole:return False
        self.rows[index.row()][index.column()]=value;self.dataChanged.emit(index,index);return True
    def replace_rows(self,rows):self.beginResetModel();self.rows=rows;self.endResetModel()

class PressureView(QTableView):
    def keyPressEvent(self,event):
        if event.matches(QKeySequence.Copy):
            indices=self.selectedIndexes()
            if not indices:return
            rows=range(min(i.row() for i in indices),max(i.row() for i in indices)+1);columns=range(min(i.column() for i in indices),max(i.column() for i in indices)+1)
            QApplication.clipboard().setText('\n'.join('\t'.join(str(self.model().rows[r][c]) if self.model().rows[r][c] is not None else '' for c in columns) for r in rows));return
        if event.matches(QKeySequence.Paste):
            index=self.currentIndex()
            if not index.isValid():return
            for dr,row in enumerate(QApplication.clipboard().text().splitlines()):
                for dc,value in enumerate(row.split('\t')):
                    if index.row()+dr<self.model().rowCount() and index.column()+dc<self.model().columnCount():self.model().setData(self.model().index(index.row()+dr,index.column()+dc),value)
            return
        super().keyPressEvent(event)

class MappingDialog(QDialog):
    def __init__(self,tables,kind,parent):
        super().__init__(parent);self.tables=tables;self.kind=kind;self.setWindowTitle('Сопоставление столбцов: '+('остановки' if kind=='steps' else 'давление'));self.resize(750,600)
        layout=QVBoxLayout(self);self.sheet=QComboBox()
        for table in tables:self.sheet.addItem(table['name'])
        layout.addWidget(self.sheet);scroll=QScrollArea();scroll.setWidgetResizable(True);panel=QWidget();form=QFormLayout(panel);scroll.setWidget(panel);layout.addWidget(scroll,1);self.mapping={}
        if kind=='steps':
            keys=[('step_id','Индекс остановки (пусто = порядок строк)'),('theta_deg','Угол (пусто = номинальный шаг)'),('t_start_s','Начало окна / timestamp, с'),('t_end_s','Конец окна, с (пусто = timestamp)'),*[(f'T_R{r}_C',f'Температура R{r}, °C') for r in range(1,7)],*[(f'valid_R{r}',f'Достоверность R{r} (необязательно)') for r in range(1,7)],('note','Примечание')]
        else:keys=[('t_s','Время'),('pressure','Значение давления'),('manifold_id','Коллектор'),('unit','Единицы'),('pressure_kind','Вид давления'),('quality','Достоверность'),('reference_id','Общий ноль избыточного давления')]
        for key,label in keys:
            combo=QComboBox();self.mapping[key]=combo;form.addRow(label,combo)
        self.unit=QComboBox();self.unit.addItems(['kgf/cm2','bar','Pa','kPa','MPa'])
        self.pressure_kind=QComboBox()
        for label,key in [('Перепад топлива Δp','delta_p'),('Топливо: абсолютное','p_fuel_abs'),('Топливо: избыточное','p_fuel_gauge'),('Газ: абсолютное','p_gas_abs'),('Газ: избыточное','p_gas_gauge')]:self.pressure_kind.addItem(label,key)
        self.group=QLineEdit('A');self.reference=QLineEdit();self.offset=spin(0);self.duration=spin(0,0,1e8)
        self.time_scale=QComboBox()
        for label,factor in [('секунды',1.),('миллисекунды',.001),('микросекунды',.000001),('сутки (числа Excel)',86400.)]:self.time_scale.addItem(label,factor)
        form.addRow('Единица времени в файле',self.time_scale)
        if kind=='pressure':
            for label,widget in [('Единицы, если нет столбца',self.unit),('Физический вид, если нет столбца',self.pressure_kind),('Коллектор, если нет столбца',self.group),('reference_id, если нет столбца',self.reference),('Сдвиг часов в шкалу измерений, с',self.offset)]:form.addRow(label,widget)
        else:form.addRow('Длительность после timestamp, с (0 = мгновенный)',self.duration)
        note=QLabel('Пустые температуры сохраняются пропусками. Указанные единицы/отсчёт времени являются явной настройкой импорта. Сырые строки и сопоставление сохраняются.');note.setWordWrap(True);layout.addWidget(note)
        buttons=QDialogButtonBox(QDialogButtonBox.Ok|QDialogButtonBox.Cancel);buttons.accepted.connect(self.accept);buttons.rejected.connect(self.reject);layout.addWidget(buttons)
        self.sheet.currentIndexChanged.connect(self.populate);self.populate()
    def populate(self):
        headers=self.tables[self.sheet.currentIndex()]['headers']
        aliases={'t_start_s':['timestamp_s','t_s','time_s'],'t_s':['timestamp_s','time_s'],'pressure':['P','p'],'quality':['valid']}
        for key,combo in self.mapping.items():
            combo.clear();combo.addItem('Не задан / по умолчанию',None)
            for header in headers:combo.addItem(header,header)
            for guess in [key,*aliases.get(key,[])]:
                index=combo.findData(guess)
                if index>=0:combo.setCurrentIndex(index);break
    def values(self):return {key:combo.currentData() for key,combo in self.mapping.items()}

class SessionWorker(QObject):
    completed=pyqtSignal(object);failed=pyqtSignal(str);progress=pyqtSignal(int,int,str);finished=pyqtSignal();cancelled=pyqtSignal()
    def __init__(self,function):super().__init__();self.function=function
    def run(self):
        try:
            result=self.function(self)
            if QThread.currentThread().isInterruptionRequested():self.cancelled.emit()
            else:self.completed.emit(result)
        except CalculationCancelled:self.cancelled.emit()
        except Exception as exc:self.failed.emit(str(exc))
        finally:self.finished.emit()

class MeasurementPanel(QWidget):
    def __init__(self,owner,table_class):
        super().__init__();self.owner=owner;self.table_class=table_class;self.thread=None;self.result=None;self.operator=None;self._loaded=None;self._project_key=None;self.metadata={};self._pressure_meta={};self._step_meta={};self._point_delays={}
        layout=QVBoxLayout(self);top=QHBoxLayout();self.import_buttons=[];self.edit_buttons=[]
        for label,slot in [('JSON сессии / проекта',self.open_json),('Остановки CSV/XLSX',lambda:self.open_table('steps')),('Давление CSV/XLSX',lambda:self.open_table('pressure')),('JSON: сохранить',self.export_session)]:
            b=QPushButton(label);b.clicked.connect(slot);top.addWidget(b);self.import_buttons.append(b)
        top.addStretch();layout.addLayout(top)
        settings=QHBoxLayout();geometry=QGroupBox('Паспорт измерения');g=QFormLayout(geometry)
        self.z=spin(0,0,1e6);self.radii=QLineEdit();self.zero=spin(0,-360,360);self.direction=QComboBox();self.direction.addItem('Против часовой стрелки',1);self.direction.addItem('По часовой стрелке',-1);self.start_angle=spin(0,-360,360);self.nominal=QSpinBox();self.nominal.setRange(1,10000);self.nominal.setValue(140)
        for label,widget in [('z измерения, мм',self.z),('Радиусы R1…R6, мм (через ;)',self.radii),('Ноль сессии от +X против ЧС, °',self.zero),('Направление обхода',self.direction),('Первый номинальный угол сессии, °',self.start_angle),('Номинальных положений в окружности',self.nominal)]:g.addRow(label,widget)
        settings.addWidget(geometry,1);policy=QGroupBox('Время и давление');p=QFormLayout(policy)
        self.default_group=QLineEdit('A');self.delay=spin(0,0,1e9);self.gap=spin(0,0,1e9);self.rtol=spin(1e-6,1e-10,.01,1e-6,10);self.time_extrap=QCheckBox('Разрешить временную экстраполяцию');self.curve_extrap=QLabel()
        for label,widget in [('Коллектор по умолчанию',self.default_group),('Заданная транспортная задержка τ, с',self.delay),('Макс. разрыв давления, с (0 = не ограничен)',self.gap),('Относительная точность Q-интеграла',self.rtol)]:p.addRow(label,widget)
        p.addRow(self.time_extrap);self.curve_extrap.setWordWrap(True);p.addRow(self.curve_extrap)
        self.assumptions=QLabel('K фиксировано. Без измеренной задержки τ=0. Температура независима; её прогноз и подгонка не выполняются.');self.assumptions.setWordWrap(True);p.addRow(self.assumptions);settings.addWidget(policy,1);layout.addLayout(settings)
        self.pages=QTabWidget();layout.addWidget(self.pages,1)
        page=QWidget();v=QVBoxLayout(page);self.steps=table_class(0,18);self.steps.setHorizontalHeaderLabels(['step_id','θ сессии, °','t начала, с','t конца, с','Метод',*[f'T_R{i}, °C' for i in range(1,7)],*[f'valid R{i}' for i in range(1,7)],'Примечание']);v.addWidget(self.steps)
        bar=QHBoxLayout()
        for label,slot in [('Добавить остановку',self.add_step),('Удалить выбранные',self.remove_steps)]:b=QPushButton(label);b.clicked.connect(slot);bar.addWidget(b);self.edit_buttons.append(b)
        bar.addStretch();v.addLayout(bar);self.pages.addTab(page,'Остановки: 6 каналов')
        page=QWidget();v=QVBoxLayout(page);self.pressure_model=RowsModel(self);self.pressure_view=PressureView();self.pressure_view.setModel(self.pressure_model);self.pressure_view.setItemDelegate(NumericDelegate(self.pressure_view));self.pressure_view.horizontalHeader().setStretchLastSection(True);v.addWidget(self.pressure_view)
        bar=QHBoxLayout()
        for label,slot in [('Добавить отсчёт',self.add_pressure),('Удалить выбранные',self.remove_pressure)]:b=QPushButton(label);b.clicked.connect(slot);bar.addWidget(b);self.edit_buttons.append(b)
        bar.addStretch();v.addLayout(bar);self.pages.addTab(page,'Давление и газ')
        self.manifolds=table_class(140,3);self.manifolds.setHorizontalHeaderLabels(['ID форсунки','Коллектор (пусто = общий)','Заданная задержка форсунки, с']);self.manifolds.horizontalHeader().setStretchLastSection(True);self.pages.addTab(self.manifolds,'Коллекторы / задержки')
        self._build_results();self.pages.addTab(self.result_page,'Результаты')
        actions=QHBoxLayout();self.check_button=QPushButton('Проверить');self.check_button.clicked.connect(self.check);actions.addWidget(self.check_button)
        self.attach_button=QPushButton('Сохранить сессию в проект');self.attach_button.clicked.connect(self.attach);actions.addWidget(self.attach_button)
        self.run_button=QPushButton('Рассчитать сессию');self.run_button.clicked.connect(self.run);actions.addWidget(self.run_button)
        self.cancel_button=QPushButton('Отменить');self.cancel_button.setEnabled(False);self.cancel_button.clicked.connect(self.cancel);actions.addWidget(self.cancel_button)
        self.export_button=QPushButton('Отчёт CSV / XLSX');self.export_button.setEnabled(False);self.export_button.clicked.connect(self.export_report);actions.addWidget(self.export_button);actions.addStretch();layout.addLayout(actions)
        self.progress=QProgressBar();layout.addWidget(self.progress);self.status=QLabel('Загрузите протокол и давление. 140 остановок и 140 форсунок — разные индексы.');self.status.setWordWrap(True);layout.addWidget(self.status)
        for widget in (self.z,self.zero,self.start_angle,self.delay,self.gap,self.rtol):widget.editingFinished.connect(self.mark_dirty)
        for widget in (self.radii,self.default_group):widget.editingFinished.connect(self.mark_dirty)
        for widget in (self.direction,):widget.currentIndexChanged.connect(self.mark_dirty)
        self.nominal.valueChanged.connect(self.mark_dirty);self.time_extrap.toggled.connect(self.mark_dirty)
        self.steps.itemChanged.connect(self.mark_dirty);self.steps.batchFinished.connect(self.mark_dirty);self.manifolds.itemChanged.connect(self.mark_dirty);self.manifolds.batchFinished.connect(self.mark_dirty);self.pressure_model.dataChanged.connect(self.mark_dirty)
        self.load_project(force=True)
    def mark_dirty(self,*args):
        if self.result is not None:
            self.status.setText('Входные данные изменены. На графиках и в экспорте остаётся предыдущий расчётный снимок; выполните новый расчёт.')
            self.export_button.setText('Отчёт предыдущего снимка')
    def _build_results(self):
        self.result_page=QWidget();v=QVBoxLayout(self.result_page);row=QHBoxLayout();self.result_kind=QComboBox();self.result_kind.addItems(['Окружные эпюры: q и измеренная T','Матрица q по окнам','Измеренная T по окнам','История давления','Мгновенная матрица τ=0'])
        self.result_x=QComboBox();self.result_x.addItem('Градусы','angle');self.result_x.addItem('Номера окружных ячеек','cells');self.time=spin(0)
        for label,w in [('Вид:',self.result_kind),('X:',self.result_x),('t снимка, с:',self.time)]:row.addWidget(QLabel(label));row.addWidget(w)
        self.axes_button=QPushButton('Оси…');self.axes_button.clicked.connect(self.open_axes);row.addWidget(self.axes_button);row.addStretch();v.addLayout(row)
        self.balance=QLabel('Составная матрица: отдельное временное окно у каждого угла. Единый мгновенный баланс не определён.');self.balance.setWordWrap(True);v.addWidget(self.balance)
        self.figure=Figure(figsize=(10,5),tight_layout=True);self.canvas=FigureCanvasQTAgg(self.figure);v.addWidget(self.canvas,1);self.interaction=attach_interaction(self.canvas)
        self.result_kind.currentIndexChanged.connect(self.draw);self.result_x.currentIndexChanged.connect(self.draw);self.time.editingFinished.connect(self.draw)
    def _set_item(self,table,r,c,value,readonly=False,check=False):
        item=QTableWidgetItem('' if value is None or check else format(value,'.12g') if isinstance(value,float) else str(value))
        if isinstance(value,float):item.setData(Qt.UserRole,(item.text(),value))
        if check:item.setFlags(Qt.ItemIsEnabled|Qt.ItemIsSelectable|Qt.ItemIsUserCheckable);item.setCheckState(Qt.Checked if value else Qt.Unchecked)
        elif readonly:item.setFlags(Qt.ItemIsEnabled|Qt.ItemIsSelectable)
        table.setItem(r,c,item)
    def _fill_steps(self,steps):
        self.steps.setUpdatesEnabled(False);self.steps.setRowCount(len(steps));self._step_meta={s.step_id:deepcopy(s.metadata) for s in steps}
        for r,s in enumerate(steps):
            values=[s.step_id,s.theta_deg,s.t_start_s,s.t_end_s,s.metadata.get('sampling_method','window'),*s.temperatures_C,*s.valid,s.note]
            for c,value in enumerate(values):self._set_item(self.steps,r,c,value,check=11<=c<=16)
        self.steps.setUpdatesEnabled(True);self.steps.resizeColumnsToContents()
    def _fill_pressures(self,series):
        rows=[];self._pressure_meta={}
        for p in series:
            self._pressure_meta[(p.manifold_id,p.pressure_kind)]=deepcopy(p.metadata)
            for t,value,valid in zip(p.t_s,p.pressure,p.valid):rows.append([p.manifold_id,t,value,p.unit,p.pressure_kind,bool(valid),p.reference_id,p.metadata.get('time_offset_s',0.)])
        self.pressure_model.replace_rows(rows)
    def set_session(self,session):
        self.result=None;self._loaded=session;self.metadata=deepcopy(session.metadata);self.metadata.setdefault('raw_session',session.to_dict());self._point_delays=deepcopy(session.point_delays_s)
        self.z.setValue(session.z_mm);self.radii.setText('; '.join(format(v,'.12g') for v in session.radii_mm));self.zero.setValue(session.theta_zero_deg);self.direction.setCurrentIndex(self.direction.findData(session.theta_direction));self.start_angle.setValue(session.theta_start_deg);self.nominal.setValue(session.nominal_steps)
        self.default_group.setText(session.default_manifold);self.delay.setValue(session.transport_delay_s);self.time_extrap.setChecked(session.allow_time_extrapolation);self.gap.setValue(session.max_gap_s or 0);self.rtol.setValue(session.integration_rtol)
        self._fill_steps(session.steps);self._fill_pressures((*session.pressure_series,*session.gas_series))
        for r in range(140):
            for c,value in enumerate((r+1,session.nozzle_manifolds.get(str(r+1),''),session.nozzle_delays_s.get(str(r+1),0.))):self._set_item(self.manifolds,r,c,value,readonly=c==0)
        self.result=None;self.export_button.setEnabled(False);self.interaction.clear_artists();self.figure.clear();self.interaction.reset_axes();self.canvas.draw_idle();self._update_curve_policy()
    def _update_curve_policy(self):self.curve_extrap.setText('Экстраполяция Q(Δp): '+('разрешена' if self.owner.cfg.allow_extrapolation else 'запрещена')+'; меняется в настройках режима на вкладке 02. Границы Дитякина не отменяются.')
    @pyqtSlot(int)
    def history_changed(self,index):
        if not getattr(self.owner,"_closing",False):
            self.load_project();self._update_curve_policy()
            if self.result is not None and project_to_json(self.owner.cfg,self.owner.nozzles)!=json.loads(self.result.project_payload):self.mark_dirty()
    def load_project(self,force=False):
        raw=self.owner.cfg._project_extra.get('measurement_session');key=json.dumps(raw,sort_keys=True,ensure_ascii=False)
        if self.thread is not None or not force and key==self._project_key:return
        self._project_key=key
        if raw:
            try:self.set_session(MeasurementSession.from_dict(raw));self.status.setText('Из проекта загружена сессия с исходными временными данными.')
            except Exception as exc:
                self._empty_session();self.status.setText('Сессия сохранена в JSON, но требует исправления: '+str(exc))
        else:self._empty_session()
    def _empty_session(self):
        radii=self.owner.cfg.radial_centers_and_edges()[0]
        if len(radii)!=6:radii=np.linspace(self.owner.cfg.r_in_mm,self.owner.cfg.r_out_mm,8)[1:-1]
        self.set_session(MeasurementSession(tuple(radii),float(self.owner.cfg.z_mm[0]),(),(),theta_zero_deg=self.owner.cfg.theta_zero_deg,theta_direction=self.owner.cfg.theta_direction,nominal_steps=self.owner.cfg.n_angular))
    def read_session(self):
        def text(table,r,c):return table.item(r,c).text().strip() if table.item(r,c) else ''
        def number(value):return float(str(value).replace(',','.'))
        def cell(table,r,c):
            item=table.item(r,c);raw=item.data(Qt.UserRole) if item else None;shown=text(table,r,c)
            return raw[1] if raw and raw[0]==shown else number(shown)
        def exact(widget,key):
            value=widget.value();original=getattr(self._loaded,key,None)
            return original if original is not None and value==round(original,widget.decimals()) else value
        steps=[]
        for r in range(self.steps.rowCount()):
            ident=number(text(self.steps,r,0));metadata=deepcopy(self._step_meta.get(int(ident),{}));metadata['sampling_method']=text(self.steps,r,4) or 'window'
            angle=text(self.steps,r,1);temps=tuple(cell(self.steps,r,c) if text(self.steps,r,c) else None for c in range(5,11));valid=tuple(self.steps.item(r,c).checkState()==Qt.Checked for c in range(11,17))
            steps.append(MeasurementStep(ident,cell(self.steps,r,2),cell(self.steps,r,3),temps,None if not angle else cell(self.steps,r,1),valid,text(self.steps,r,17),metadata))
        grouped={}
        for row in self.pressure_model.rows:
            group,t,p,unit,kind,valid,reference,offset=row;key=(str(group),str(unit),str(kind),str(reference),number(offset));series=grouped.setdefault(key,[[],[],[]]);series[0].append(number(t));series[1].append(None if p is None or str(p).strip()=='' else number(p));series[2].append(bool(valid))
        fuel=[];gas=[]
        for (group,unit,kind,reference,offset),(times,values,valid) in grouped.items():
            metadata=deepcopy(self._pressure_meta.get((group,kind),{}));metadata['time_offset_s']=offset
            p=PressureSeries(group,tuple(times),tuple(values),unit,kind,tuple(valid),reference,metadata);(gas if kind.startswith('p_gas') else fuel).append(p)
        groups={str(r+1):text(self.manifolds,r,1) for r in range(140) if text(self.manifolds,r,1)};delays={str(r+1):cell(self.manifolds,r,2) for r in range(140) if text(self.manifolds,r,2) and cell(self.manifolds,r,2)!=0}
        radii=tuple(number(v.strip()) for v in self.radii.text().split(';') if v.strip())
        if self._loaded and self.radii.text()=='; '.join(format(v,'.12g') for v in self._loaded.radii_mm):radii=self._loaded.radii_mm
        return MeasurementSession(radii,exact(self.z,'z_mm'),tuple(steps),tuple(fuel),tuple(gas),exact(self.zero,'theta_zero_deg'),self.direction.currentData(),exact(self.start_angle,'theta_start_deg'),self.nominal.value(),self.default_group.text().strip(),groups,exact(self.delay,'transport_delay_s'),delays,deepcopy(self._point_delays),self.time_extrap.isChecked(),exact(self.gap,'max_gap_s') or None,exact(self.rtol,'integration_rtol'),deepcopy(self.metadata))
    def check(self):
        try:
            self.owner._record_edit();session=self.read_session();session.validate(self.owner.cfg,self.owner.nozzles)
            self.status.setText(f'Проверено: {len(session.steps)} остановок × 6 каналов, {len(session.pressure_series)} коллекторов. Радиусы привязаны к содержащим их контрольным площадкам.');return session
        except Exception as exc:self.status.setText(str(exc));return None
    def attach(self):
        try:
            self.owner._record_edit();session=self.read_session();session.validate(self.owner.cfg,self.owner.nozzles,require_temperatures=False)
            self.owner.cfg._project_extra['measurement_session']=session.to_dict();self._project_key=json.dumps(session.to_dict(),sort_keys=True,ensure_ascii=False);self.owner._record_edit(label='Измерительная сессия v0.5');self.status.setText('Сессия включена в проект; Ctrl+S сохраняет проект, Undo/Redo отменяет/повторяет включение.')
        except Exception as exc:self.status.setText(str(exc))
    def add_step(self):
        # A partially entered row must survive adding the next row.
        ids=[];ends=[]
        for r in range(self.steps.rowCount()):
            for c,values in ((0,ids),(3,ends)):
                item=self.steps.item(r,c)
                try:
                    value=float(item.text().replace(',','.'))
                    if np.isfinite(value):values.append(value)
                except (ValueError,AttributeError):pass
        ident=int(max(ids,default=-1))+1;start=max(ends,default=0.);r=self.steps.rowCount();self.steps.insertRow(r)
        for c,value in enumerate((ident,None,start,start+1,'window',*(None,)*6,*(False,)*6,'')):self._set_item(self.steps,r,c,value,check=11<=c<=16)
        self.mark_dirty()
    def remove_steps(self):
        for row in sorted({i.row() for i in self.steps.selectedIndexes()},reverse=True):self.steps.removeRow(row)
    def add_pressure(self):self.pressure_model.replace_rows([*self.pressure_model.rows,[self.default_group.text() or 'A',0.,None,'kgf/cm2','delta_p',False,'',0.]])
    def remove_pressure(self):
        selected={i.row() for i in self.pressure_view.selectedIndexes()};self.pressure_model.replace_rows([r for i,r in enumerate(self.pressure_model.rows) if i not in selected])
    def open_json(self):
        path,_=QFileDialog.getOpenFileName(self,'Сессия или проект с сессией','','JSON (*.json)')
        if path:
            try:self.set_session(load_session(path));self.status.setText('Сессия загружена; исходный файл не изменён. Для включения в проект нажмите «Сохранить сессию в проект».')
            except Exception as exc:self.status.setText(str(exc))
    def open_table(self,kind):
        path,_=QFileDialog.getOpenFileName(self,'Таблица остановок' if kind=='steps' else 'Давление','','Таблицы (*.csv *.xlsx)')
        if path:self._start(lambda worker:read_tables(path),lambda tables:self.import_table(tables,kind))
    def import_table(self,tables,kind,mapping=None,options=None):
        if mapping is None:
            dialog=MappingDialog(tables,kind,self)
            if dialog.exec_()!=QDialog.Accepted:return
            table=tables[dialog.sheet.currentIndex()];mapping=dialog.values();factor=dialog.time_scale.currentData()
            options=dict(duration_s=dialog.duration.value(),unit=dialog.unit.currentText(),pressure_kind=dialog.pressure_kind.currentData(),manifold_id=dialog.group.text(),reference_id=dialog.reference.text(),time_offset_s=dialog.offset.value(),time_scale_s=factor)
        else:table=tables[0];options=options or {}
        try:
            if kind=='steps':
                imported=import_steps(table,mapping,duration_s=options.get('duration_s'),time_scale_s=options.get('time_scale_s',1.));self._fill_steps(imported)
            else:
                series=import_pressures(table,mapping,**{k:v for k,v in options.items() if k!='duration_s'})
                # Replace only the imported physical channels; keep other collectors and gas data.
                keys={(p.manifold_id,p.pressure_kind) for p in series};previous=[row for row in self.pressure_model.rows if (str(row[0]),str(row[4])) not in keys];previous_meta=deepcopy(self._pressure_meta);self._fill_pressures(series);self.pressure_model.replace_rows(previous+self.pressure_model.rows);self._pressure_meta={**previous_meta,**self._pressure_meta}
            self.metadata.setdefault('raw_imports',[]).append(dict(kind=kind,mapping=deepcopy(mapping),options=deepcopy(options),source=deepcopy(table['source'])))
            self.status.setText('Импортировано. Проверьте паспорт сессии и привязку коллекторов; исходные строки сохранены в метаданных.')
        except Exception as exc:self.status.setText(str(exc))
    def _series_from_rows(self):
        if not self.pressure_model.rows:return []
        # This parser uses the current editable tables, without requiring a complete pressure/temperature protocol.
        s=self.read_session();return list(s.pressure_series)+list(s.gas_series)
    def export_session(self):
        path,_=QFileDialog.getSaveFileName(self,'Сохранить сессию','measurement_session.json','JSON (*.json)')
        if path:
            try:save_session(path,self.read_session());self.status.setText('JSON сессии сохранён.')
            except Exception as exc:self.status.setText(str(exc))
    def run(self):
        session=self.check()
        if session is None:return
        cfg=deepcopy(self.owner.cfg);nozzles=deepcopy(self.owner.nozzles);signature=geometry_signature(nozzles,cfg,session.z_mm)
        operator=self.operator if self.operator is not None and self.operator.signature==signature else None
        self._start(lambda worker:calculate_session(nozzles,cfg,session,cancel=lambda:QThread.currentThread().isInterruptionRequested(),progress=lambda done,total,label:worker.progress.emit(done,total,label),operator=operator),self._done)
    def _start(self,function,completed):
        if self.thread is not None:return
        self.thread=QThread(self);self.worker=SessionWorker(function);self.worker.moveToThread(self.thread);self.thread.started.connect(self.worker.run)
        self.worker.completed.connect(completed);self.worker.failed.connect(lambda message:self.status.setText(message));self.worker.cancelled.connect(lambda:self.status.setText('Расчёт отменён; предыдущий полный результат сохранён.'))
        self.worker.progress.connect(self._progress);self.worker.finished.connect(self.thread.quit);self.worker.finished.connect(self.worker.deleteLater);self.thread.finished.connect(self._cleanup)
        self.inputs=[*self.import_buttons,*self.edit_buttons,self.check_button,self.attach_button,self.run_button,self.z,self.radii,self.zero,self.direction,self.start_angle,self.nominal,self.default_group,self.delay,self.time_extrap,self.gap,self.rtol,self.steps,self.pressure_view,self.manifolds]
        for widget in self.inputs:widget.setEnabled(False)
        self.cancel_button.setEnabled(True);self.thread.start()
    def _progress(self,done,total,label):self.progress.setMaximum(total);self.progress.setValue(done);self.status.setText(f'{label}: {done}/{total}')
    def _cleanup(self):
        self.thread.deleteLater();self.thread=None;self.worker=None
        for widget in self.inputs:widget.setEnabled(True)
        self.cancel_button.setEnabled(False)
    def cancel(self):
        if self.thread is not None:self.thread.requestInterruption();self.status.setText('Отмена после текущего интеграционного участка…')
    def _done(self,result):
        self.result=result;self.operator=result.operator;self.export_button.setText('Отчёт CSV / XLSX');self.export_button.setEnabled(True);self.pages.setCurrentWidget(self.result_page);self.time.setValue(result.session.steps[0].t_start_s)
        self.status.setText(f'Готово: {len(result.session.steps)} остановок × 6 каналов; {result.elapsed_s:.3f} с; max баланс снимков {result.manifest()["maximum_snapshot_balance_error"]:.3g}. Исходная T не изменена.')
        if project_to_json(self.owner.cfg,self.owner.nozzles)!=json.loads(result.project_payload):self.mark_dirty()
        self.draw()
    def open_axes(self):self.axes_dialog=open_axes_settings(self.canvas,self)
    def draw(self,*args):
        if self.result is None:return
        r=self.result;self.interaction.clear_artists();self.figure.clear();kind=self.result_kind.currentIndex();mode=self.result_x.currentData();cfg=r.operator.cfg
        x=plot_x(r.global_angles,cfg,mode);order=np.argsort(x,kind='stable');self.axes_button.setEnabled(kind in (0,3));self.time.setEnabled(kind==4)
        self.balance.setText('Составная матрица: разные временные окна; единый мгновенный глобальный баланс не определён. q — среднее по содержащей датчик контрольной ячейке. T — независимые измерения.')
        try:
            if kind==0:
                for channel,radius in enumerate(r.session.radii_mm):
                    ax=self.figure.add_subplot(3,2,channel+1);ax.plot(x[order],r.q_corrected[channel,order],label='q топлива');ax.set_ylabel('кг/(м²·с)',fontsize=7);ax.set_xlabel(x_label(cfg,mode),fontsize=7);ax.set_title(f'R{channel+1}={radius:g} мм',fontsize=9);ax.tick_params(labelsize=7);ax.grid(alpha=.2);decorate_axes(ax,cfg,mode);ax._compare_y='кг/(м²·с)'
                    other=ax.twinx();temperatures=np.where(r.temperature_valid[channel],r.temperatures_C[channel],np.nan);other.plot(x[order],temperatures[order],'.-',color='#b35a24',ms=3,label='Измеренная T');other.set_ylabel('T измер., °C',fontsize=7,color='#b35a24');other.tick_params(labelsize=7,axis='y',colors='#b35a24');other._compare_x=ax._compare_x;other._compare_y='°C';other._compare_frame=ax._compare_frame
            elif kind in (1,2):
                ax=self.figure.add_subplot(111);values=r.q_corrected if kind==1 else np.ma.masked_where(~r.temperature_valid,r.temperatures_C);im=ax.imshow(values,aspect='auto',origin='lower',interpolation='nearest');ax.set_ylabel('Канал R1…R6');ax.set_yticks(range(6),labels=[f'R{i+1}={v:g} мм' for i,v in enumerate(r.session.radii_mm)]);ax.set_xlabel('Строка протокола: свой угол и временное окно');count=len(r.session.steps);ticks=np.unique(np.linspace(0,count-1,min(10,count)).astype(int));ax.set_xticks(ticks,labels=[str(r.session.steps[i].step_id) for i in ticks]);self.figure.colorbar(im,ax=ax,label='кг/(м²·с)' if kind==1 else 'T измер., °C')
            elif kind==3:
                ax=self.figure.add_subplot(111)
                for p in r.session.pressure_series:
                    times=p._times;values=[]
                    for t in times:
                        try:values.append(r.session.delta_p(p.manifold_id,float(t))[0])
                        except ValueError:values.append(np.nan)
                    ax.plot(times,values,'.-',ms=3,label='Δp '+p.manifold_id)
                for step in r.session.steps:ax.axvspan(step.t_start_s,step.t_end_s,alpha=.06,color='#76509c')
                ax.set_xlabel('Время в шкале измерений, с');ax.set_ylabel('Δp, кгс/см²');ax.legend(fontsize=8);ax.grid(alpha=.2)
            else:
                plane,flows=r.snapshot(self.time.value());ax=self.figure.add_subplot(111);im=ax.imshow(plane.flux_kg_m2_s,aspect='auto',origin='lower',interpolation='nearest');ax.set_xlabel('Окружная ячейка расчётной сетки');ax.set_ylabel('Радиальный уровень');self.figure.colorbar(im,ax=ax,label='кг/(м²·с)');self.balance.setText(f'Мгновенный снимок, t={self.time.value():g} с, τ=0: G={plane.injected_kg_s:.9g} кг/с; внутри={plane.inside_kg_s:.9g}; за внутренней={plane.inner_kg_s:.9g}; за наружной={plane.outer_kg_s:.9g}; относительная невязка={plane.rel_balance_error:.3g}.')
            if kind in (0,3):self.interaction.reset_axes()
            else:self.interaction.axes=[];self.interaction.original={}
            self.canvas.draw_idle()
        except Exception as exc:self.status.setText(str(exc));self.interaction.axes=[];self.canvas.draw_idle()
    def export_report(self):
        if self.result is None:return
        path,_=QFileDialog.getSaveFileName(self,'Отчёт временной сессии','measurement_report.xlsx','Excel (*.xlsx);;CSV (*.csv)')
        if path:
            try:outputs=write_report(path,self.result);self.status.setText(f'Отчёт сохранён: {len(outputs)} файлов. Указаны временные окна, исходная T, Δp, расходы и паспорт расчёта.')
            except Exception as exc:self.status.setText(str(exc))
