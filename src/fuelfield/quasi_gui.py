"""Pressure-only configuration window; drafts apply to the main project atomically."""
from pathlib import Path
from dataclasses import replace
from copy import deepcopy
import numpy as np
from PyQt5.QtCore import Qt,QTimer
from PyQt5.QtWidgets import (QDialog,QVBoxLayout,QHBoxLayout,QFormLayout,QLabel,QLineEdit,QComboBox,QPushButton,
    QTableWidget,QTableWidgetItem,QCheckBox,QFileDialog,QTextBrowser,QDialogButtonBox,QStyledItemDelegate,QSplitter,QWidget)
from .quasi import QuasiSettings,PressureChannel,PRESSURE_FACTORS,KINDS,finite
from .quasi_io import read_pressure_tables,import_pressure_table,load_settings,save_settings
from .pressure_charts import PressureCharts

KIND_LABELS={'delta_p':'Перепад Δp','p_fuel_abs':'Абсолютное топливо − абсолютный газ','p_fuel_gauge':'Избыточное топливо − избыточный газ'}

class ChoiceDelegate(QStyledItemDelegate):
    def __init__(self,choices,parent):super().__init__(parent);self.choices=choices
    def createEditor(self,parent,option,index):
        editor=QComboBox(parent)
        for key,label in self.choices:editor.addItem(label,key)
        return editor
    def setEditorData(self,editor,index):editor.setCurrentIndex(editor.findData(index.data()))
    def setModelData(self,editor,model,index):model.setData(index,editor.currentData(),Qt.EditRole)

class PressureImportDialog(QDialog):
    def __init__(self,mode,tables,parent):
        super().__init__(parent);self.mode=mode;self.tables=tables;self.setWindowTitle('Импорт характеристики давления');self.resize(760,500)
        layout=QVBoxLayout(self);form=QFormLayout();self.sheet=QComboBox();self.sheet.addItems([t['name'] for t in tables]);form.addRow('Лист',self.sheet);self.time=QComboBox();form.addRow('Столбец времени (необязательно)',self.time)
        self.time_unit=QComboBox()
        for label,factor in [('секунды',1.),('миллисекунды',.001),('сутки Excel',86400.)]:self.time_unit.addItem(label,factor)
        form.addRow('Единица времени файла',self.time_unit);self.adopt=QCheckBox('Принять начало и постоянный шаг времени из файла');self.adopt.setChecked(True);form.addRow(self.adopt);self.mapping={}
        for channel in mode.channels:
            p=QComboBox();gas=QComboBox() if channel.kind!='delta_p' else None;self.mapping[channel.id]=(p,gas);form.addRow(f'{channel.id}: давление [{channel.unit}]',p)
            if gas is not None:form.addRow(f'{channel.id}: газ [{channel.gas_unit}]',gas)
        layout.addLayout(form);note=QLabel('Одна строка — один шаг. Требуются ровно '+str(mode.n_steps)+' строк и постоянный временной шаг. Единицы давления задаются в таблице коллекторов; пропуски не заполняются.');note.setWordWrap(True);layout.addWidget(note)
        buttons=QDialogButtonBox(QDialogButtonBox.Ok|QDialogButtonBox.Cancel);buttons.accepted.connect(self.accept);buttons.rejected.connect(self.reject);layout.addWidget(buttons)
        self.sheet.currentIndexChanged.connect(self.populate);self.populate()
    def populate(self):
        headers=self.tables[self.sheet.currentIndex()]['headers']
        for combo in [self.time,*[c for pair in self.mapping.values() for c in pair if c is not None]]:
            combo.clear();combo.addItem('Не загружать / по настройкам',None)
            for h in headers:combo.addItem(h,h)
        for h in ('t_s','time_s','time'):
            index=self.time.findData(h)
            if index>=0:self.time.setCurrentIndex(index);break
        for key,(p,gas) in self.mapping.items():
            for h in (key,'P_'+key,'delta_p_'+key):
                i=p.findData(h)
                if i>=0:p.setCurrentIndex(i);break
            if gas is not None:
                for h in ('gas_'+key,'P_gas_'+key):
                    i=gas.findData(h)
                    if i>=0:gas.setCurrentIndex(i);break
    def result_settings(self):
        mappings={key:dict(pressure=p.currentData(),gas_pressure=gas.currentData() if gas else None) for key,(p,gas) in self.mapping.items()}
        return import_pressure_table(self.mode,self.tables[self.sheet.currentIndex()],mappings,self.time.currentData(),self.time_unit.currentData(),self.adopt.isChecked())

class QuasiDialog(QDialog):
    def __init__(self,owner,table_class):
        super().__init__(owner);self.owner=owner;self.table_class=table_class;self.pending_bindings={};self.metadata={};self.descriptors=[]
        self.setWindowTitle('Настройки квазистационарного режима');self.resize(1280,820)
        self.chart_timer=QTimer(self);self.chart_timer.setSingleShot(True);self.chart_timer.setInterval(120);self.chart_timer.timeout.connect(self.update_charts);self.charts_dirty=True
        layout=QVBoxLayout(self);top=QHBoxLayout();self.load_button=QPushButton('Загрузить *.quasi.json');self.load_button.clicked.connect(self.open_json);top.addWidget(self.load_button);self.save_button=QPushButton('Сохранить *.quasi.json');self.save_button.clicked.connect(self.save_json);top.addWidget(self.save_button)
        self.xlsx_button=QPushButton('Импорт XLSX / CSV');self.xlsx_button.clicked.connect(self.import_file);top.addWidget(self.xlsx_button);top.addStretch();self.guide_button=QPushButton('Инструкция');self.guide_button.clicked.connect(self.show_guide);top.addWidget(self.guide_button);layout.addLayout(top)
        note=QLabel('Матрица на 04 собирается по одному угловому столбцу за шаг; все радиальные ячейки — одновременно. Поле орошения показывает выбранное время. Температуры и давление воздуха на входе не нужны.');note.setWordWrap(True);layout.addWidget(note)
        form=QHBoxLayout();self.n_label=QLabel();self.start=QLineEdit('0');self.dt=QLineEdit('1');self.first=QLineEdit('0');self.direction=QComboBox();self.direction.addItem('Против ЧС',1);self.direction.addItem('По ЧС',-1);self.default=QLineEdit('A')
        for label,w in [('Шагов:',self.n_label),('t₀, с:',self.start),('Δt, с:',self.dt),('Первый угол, °:',self.first),('Обход:',self.direction),('Общий коллектор:',self.default)]:form.addWidget(QLabel(label));form.addWidget(w)
        for w in (self.start,self.dt,self.first,self.default):w.setMaximumWidth(105)
        layout.addLayout(form);self.channels=QTableWidget(0,5);self.channels.setHorizontalHeaderLabels(['Коллектор','Единицы топлива/Δp','Вид давления','Единицы газа','Общая база gauge (reference_id)']);self.channels.setMaximumHeight(150);self.channels.horizontalHeader().setStretchLastSection(True)
        for col,choices in [(1,[(u,u) for u in PRESSURE_FACTORS]),(2,[(k,KIND_LABELS[k]) for k in KINDS]),(3,[(u,u) for u in PRESSURE_FACTORS])]:self.channels.setItemDelegateForColumn(col,ChoiceDelegate(choices,self.channels))
        layout.addWidget(self.channels);bar=QHBoxLayout()
        for label,slot in [('Добавить коллектор',self.add_channel),('Удалить выбранные',self.remove_channel),('Обновить колонки давления',self.rebuild)]:b=QPushButton(label);b.clicked.connect(slot);bar.addWidget(b)
        bar.addStretch();layout.addLayout(bar);self.table=table_class(0,0);self.pressure_charts=PressureCharts(self)
        self.pressure_splitter=QSplitter(Qt.Horizontal);self.pressure_splitter.addWidget(self.table);self.pressure_splitter.addWidget(self.pressure_charts);self.pressure_splitter.setChildrenCollapsible(False);self.pressure_splitter.setSizes([650,590]);layout.addWidget(self.pressure_splitter,1)
        self.table.itemChanged.connect(self.queue_charts);self.table.batchFinished.connect(self.queue_charts);self.channels.itemChanged.connect(self.queue_charts)
        self.bindings_box=QCheckBox('Применить привязки форсунок из загруженного файла');self.bindings_box.hide();layout.addWidget(self.bindings_box)
        actions=QHBoxLayout();apply=QPushButton('Применить');apply.clicked.connect(self.apply);actions.addWidget(apply);close=QPushButton('Закрыть');close.clicked.connect(self.close);actions.addWidget(close);actions.addStretch();layout.addLayout(actions)
        self.status=QLabel('Примените настройки, затем выполните «Рассчитать сечения» на 04. Привязки форсунок редактируются на 02.');self.status.setWordWrap(True);layout.addWidget(self.status)
        raw=owner.cfg.quasi_settings
        try:mode=QuasiSettings.from_dict(raw) if raw else QuasiSettings(owner.cfg.n_angular,first_angle_deg=owner.cfg.grid_start_deg,channels=(PressureChannel('A',(None,)*owner.cfg.n_angular),))
        except ValueError:mode=QuasiSettings(owner.cfg.n_angular,first_angle_deg=owner.cfg.grid_start_deg,channels=(PressureChannel('A',(None,)*owner.cfg.n_angular),))
        self.set_mode(mode)
        for w in (self.start,self.dt,self.first):w.editingFinished.connect(self.update_coordinates)
        self.direction.currentIndexChanged.connect(self.update_coordinates)
    def item(self,table,row,col,value,readonly=False):
        shown='' if value is None else format(value,'.12g') if isinstance(value,(float,np.floating)) else str(value);item=QTableWidgetItem(shown)
        if isinstance(value,(float,np.floating)):item.setData(Qt.UserRole,(shown,float(value)))
        if readonly:item.setFlags(Qt.ItemIsEnabled|Qt.ItemIsSelectable)
        table.setItem(row,col,item)
    def value(self,table,row,col):
        item=table.item(row,col)
        if item is None or not item.text().strip():return None
        raw=item.data(Qt.UserRole)
        if raw and raw[0]==item.text():return raw[1]
        return finite(item.text().replace(',','.'),'Давление')
    def set_mode(self,mode):
        self.channels.blockSignals(True)
        self.mode=mode;self.metadata=deepcopy(mode.metadata);self.n_label.setText(str(mode.n_steps)+' (как на 01)');self.start.setText(repr(mode.t_start_s));self.dt.setText(repr(mode.dt_s));self.first.setText(repr(mode.first_angle_deg));self.direction.setCurrentIndex(self.direction.findData(mode.direction));self.default.setText(mode.default_manifold)
        self.channels.setRowCount(len(mode.channels))
        for row,c in enumerate(mode.channels):
            for col,value in enumerate((c.id,c.unit,c.kind,c.gas_unit,c.reference_id)):self.item(self.channels,row,col,value)
        self.channels.resizeColumnsToContents();self.channels.blockSignals(False);self.fill_pressure(mode)
    def fill_pressure(self,mode):
        self.table.blockSignals(True)
        self.descriptors=[];headers=['Шаг','t, с','Угол гребёнки, °']
        for i,c in enumerate(mode.channels):
            self.descriptors.append((i,'pressure'));headers.append(c.id+' · давление ['+c.unit+']')
            if c.kind!='delta_p':self.descriptors.append((i,'gas_pressure'));headers.append(c.id+' · газ ['+c.gas_unit+']')
        self.table.setColumnCount(len(headers));self.table.setRowCount(mode.n_steps);self.table.setHorizontalHeaderLabels(headers)
        for row in range(mode.n_steps):
            for col,value in enumerate((row,float(mode.times[row]),float(mode.angles[row]))):self.item(self.table,row,col,value,True)
            for col,(index,key) in enumerate(self.descriptors,3):
                values=getattr(mode.channels[index],key);self.item(self.table,row,col,values[row] if row<len(values) else None)
        self.table.resizeColumnsToContents();self.table.horizontalHeader().setStretchLastSection(True);self.table.blockSignals(False);self.queue_charts()
    def read_mode(self,tolerate_invalid=False):
        def text(r,c):return self.channels.item(r,c).text().strip() if self.channels.item(r,c) else ''
        channels=[]
        for row in range(self.channels.rowCount()):
            parts={}
            for key in ('pressure','gas_pressure'):
                pair=(row,key)
                values=[]
                if pair in self.descriptors:
                    for r in range(self.mode.n_steps):
                        try:values.append(self.value(self.table,r,self.descriptors.index(pair)+3))
                        except ValueError:
                            if not tolerate_invalid:raise
                            values.append(None)
                parts[key]=tuple(values) if pair in self.descriptors else (None,)*self.mode.n_steps
            if text(row,2)=='delta_p':parts['gas_pressure']=()
            channels.append(PressureChannel(text(row,0),parts['pressure'],text(row,1),text(row,2),parts['gas_pressure'],text(row,3),text(row,4)))
        return QuasiSettings(self.mode.n_steps,finite(self.dt.text().replace(',','.'),'Шаг времени'),finite(self.start.text().replace(',','.'),'Начало времени'),finite(self.first.text().replace(',','.'),'Первый угол'),self.direction.currentData(),self.default.text().strip(),tuple(channels),deepcopy(self.metadata))
    def update_coordinates(self,*args):
        if not hasattr(self,'mode'):return
        try:
            mode=self.read_mode()
            if mode.dt_s<=0:raise ValueError('Шаг времени должен быть >0')
            self.table.blockSignals(True)
            for row in range(mode.n_steps):
                self.item(self.table,row,1,float(mode.times[row]),True);self.item(self.table,row,2,float(mode.angles[row]),True)
            self.table.blockSignals(False);self.queue_charts()
        except ValueError as exc:self.status.setText(str(exc))

    def queue_charts(self,*args):
        self.charts_dirty=True
        if self.isVisible() and not self.table.in_batch:self.chart_timer.start()

    def update_charts(self):
        self.chart_timer.stop()
        if not self.isVisible():return
        try:
            mode=self.read_mode(tolerate_invalid=True)
            if mode.dt_s<=0:raise ValueError('Шаг времени должен быть >0')
            self.pressure_charts.set_mode(mode)
        except (ValueError,TypeError) as exc:self.pressure_charts.show_error(exc)
        self.charts_dirty=False

    def showEvent(self,event):
        super().showEvent(event)
        if self.charts_dirty:self.chart_timer.start()

    def hideEvent(self,event):
        self.chart_timer.stop();super().hideEvent(event)
    def rebuild(self):
        try:mode=self.read_mode();self.fill_pressure(mode);self.status.setText('Колонки обновлены. Для каждого коллектора заполните все '+str(mode.n_steps)+' шагов.')
        except ValueError as exc:self.status.setText(str(exc))
    def add_channel(self):
        try:
            mode=self.read_mode();used={c.id for c in mode.channels};key=next((chr(i) for i in range(65,91) if chr(i) not in used),'C'+str(len(used)+1));self.set_mode(replace(mode,channels=(*mode.channels,PressureChannel(key,(None,)*mode.n_steps))))
        except ValueError as exc:self.status.setText(str(exc))
    def remove_channel(self):
        try:
            mode=self.read_mode();rows={i.row() for i in self.channels.selectedIndexes()};self.set_mode(replace(mode,channels=tuple(c for i,c in enumerate(mode.channels) if i not in rows)))
        except ValueError as exc:self.status.setText(str(exc))
    def proposed_nozzles(self):
        nozzles=deepcopy(self.owner._read_nozzles())
        if self.bindings_box.isChecked():
            for n in nozzles:
                if str(n.id) in self.pending_bindings:n.manifold_id=self.pending_bindings[str(n.id)]
        return nozzles
    def apply(self):
        try:
            mode=self.read_mode().validate(self.owner._read_controls(),self.proposed_nozzles());bindings=self.pending_bindings if self.bindings_box.isChecked() else {}
            self.owner.apply_quasi_settings(mode.to_dict(),bindings);self.status.setText('Настройки применены к проекту. Ctrl+S сохранит проект; F5 пересчитает 04 в выбранном режиме.')
        except (ValueError,TypeError) as exc:self.status.setText(str(exc))
    def load_path(self,path):
        mode,bindings=load_settings(path);self.set_mode(mode);self.pending_bindings=bindings;self.bindings_box.setVisible(bool(bindings));self.bindings_box.setChecked(bool(bindings))
        counts={c.id:sum(v==c.id for v in bindings.values()) for c in mode.channels};self.status.setText('Загружено. Привязки из файла: '+str(counts)+'. Они изменят таблицу 02 только после «Применить».' if bindings else 'Загружено. Примените настройки к проекту.')
    def open_json(self):
        path,_=QFileDialog.getOpenFileName(self,'Настройки режима, не проект','','Настройки (*.quasi.json);;JSON (*.json)')
        if path:
            try:self.load_path(path)
            except Exception as exc:self.status.setText(str(exc))
    def save_json(self):
        try:
            mode=self.read_mode().validate(self.owner._read_controls(),self.proposed_nozzles());path,_=QFileDialog.getSaveFileName(self,'Отдельные настройки режима','pressure_mode.quasi.json','Настройки (*.quasi.json)')
            if path:
                saved=save_settings(path,mode,{str(n.id):mode.manifold(n) for n in self.proposed_nozzles()});self.status.setText('Сохранены настройки режима и привязки: '+str(saved))
        except Exception as exc:self.status.setText(str(exc))
    def import_file(self):
        path,_=QFileDialog.getOpenFileName(self,'Давление по шагам','','Характеристики (*.xlsx *.csv)')
        if path:
            try:
                dialog=PressureImportDialog(self.read_mode(),read_pressure_tables(path),self)
                if dialog.exec_()==QDialog.Accepted:self.set_mode(dialog.result_settings());self.status.setText('Характеристика импортирована. Проверьте единицы и примените настройки.')
            except Exception as exc:self.status.setText(str(exc))
    def show_guide(self):
        if hasattr(self,'guide_dialog'):self.guide_dialog.show();self.guide_dialog.raise_();return
        path=Path(__file__).resolve().parents[2]/'docs/QUASISTATIONARY_GUIDE.md'
        try:content=path.read_text(encoding='utf-8')
        except OSError as exc:self.status.setText(str(exc));return
        self.guide_dialog=QDialog(self);self.guide_dialog.setWindowTitle('Квазистационарный расчёт — инструкция');self.guide_dialog.resize(950,700);layout=QVBoxLayout(self.guide_dialog);browser=QTextBrowser();browser.setMarkdown(content);layout.addWidget(browser);close=QDialogButtonBox(QDialogButtonBox.Close);close.button(QDialogButtonBox.Close).setText('Закрыть');close.rejected.connect(self.guide_dialog.close);layout.addWidget(close);self.guide_dialog.show()
