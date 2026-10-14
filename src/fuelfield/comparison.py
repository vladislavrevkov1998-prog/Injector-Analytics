"""Independent comparison data and charts, synchronized by physical coordinates."""
import json
import textwrap
import numpy as np
from pathlib import Path
from PyQt5.QtCore import QObject,QThread,pyqtSignal
from PyQt5.QtWidgets import (QWidget,QVBoxLayout,QHBoxLayout,QComboBox,QPushButton,
    QLabel,QFileDialog,QCheckBox,QMessageBox,QSpinBox)
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from .core import project_from_json,calculate_plane
from .analysis import cell_values,matching_index,UNITS
from .analytics_io import read_chart_tables
from .plot_interaction import attach_interaction
from .graph_selection import GraphSelection

class ProjectWorker(QObject):
    completed=pyqtSignal(object)
    failed=pyqtSignal(str)
    finished=pyqtSignal()
    def __init__(self,cfg,nozzles):super().__init__();self.cfg=cfg;self.nozzles=nozzles
    def run(self):
        try:
            planes=[]
            for z in self.cfg.z_mm:
                if QThread.currentThread().isInterruptionRequested():return
                planes.append(calculate_plane(self.nozzles,self.cfg,z))
            self.completed.emit(planes)
        except Exception as exc:self.failed.emit(str(exc))
        finally:self.finished.emit()

class ComparisonPanel(QWidget):
    def __init__(self,owner):
        super().__init__(owner);self.owner=owner;self.planes=[];self.cfg=None;self.tables=[];self.thread=None
        self._loading_cfg=None;self._chart_key=None
        layout=QVBoxLayout(self);top=QHBoxLayout()
        top.addWidget(QLabel('Загрузить:'))
        self.load_button=QPushButton('JSON');self.excel_button=QPushButton('Excel');self.csv_button=QPushButton('CSV')
        self.load_buttons=(self.load_button,self.excel_button,self.csv_button)
        for button,file_filter in zip(self.load_buttons,('JSON (*.json)','Excel (*.xlsx)','CSV (*.csv)')):
            button.clicked.connect(lambda checked=False,f=file_filter:self.open_file(f));top.addWidget(button)
        self.sync=QCheckBox('Синхронизировать z и R');self.sync.setChecked(True);top.addWidget(self.sync);top.addStretch()
        layout.addLayout(top)
        controls=QHBoxLayout();self.plane=QComboBox();self.radius=QComboBox();self.kind=QComboBox()
        self.kind.addItems(['Линия','Точки','Столбцы'])
        controls.addWidget(QLabel('z / лист'));controls.addWidget(self.plane);controls.addWidget(QLabel('R'));controls.addWidget(self.radius);controls.addWidget(self.kind);controls.addStretch();layout.addLayout(controls)
        table_controls=QHBoxLayout();self.x=QComboBox();self.y=QComboBox();self.y2=QComboBox()
        for label,widget in [('X',self.x),('Y',self.y),('Y₂',self.y2)]:table_controls.addWidget(QLabel(label));table_controls.addWidget(widget)
        table_controls.addStretch()
        self.table_controls=QWidget();self.table_controls.setLayout(table_controls);layout.addWidget(self.table_controls);self.table_controls.hide()
        self.notice=QLabel('Загрузите второй проект или таблицу с заголовками и числовыми столбцами.');self.notice.setWordWrap(True);layout.addWidget(self.notice)
        grid_controls=QHBoxLayout();self.plot_count=QSpinBox();self.plot_count.setRange(1,140)
        self.graph_selection=GraphSelection(self.plot_count);self.graph_selection.changed.connect(self.draw)
        self.plot_columns=QSpinBox();self.plot_columns.setRange(1,6)
        self.plot_count.setToolTip("Для проекта: последовательные радиальные уровни от выбранного R. Для таблицы: последовательные числовые Y-столбцы; X и Y₂ исключаются.")
        for label,widget in (("Графиков:",self.plot_count),("Колонок:",self.plot_columns)):
            grid_controls.addWidget(QLabel(label));grid_controls.addWidget(widget)
        grid_controls.addWidget(self.graph_selection)
        reset=QPushButton("Исходный масштаб");reset.clicked.connect(lambda:self.interaction.reset_zoom());grid_controls.addWidget(reset);grid_controls.addStretch()
        layout.addLayout(grid_controls)
        self.figure=Figure(figsize=(6,5),tight_layout=True);self.canvas=FigureCanvasQTAgg(self.figure);self.canvas.setMinimumSize(160,120);layout.addWidget(self.canvas,1);self.interaction=attach_interaction(self.canvas)
        self.plane.currentIndexChanged.connect(self._plane_changed)
        for w in (self.radius,self.kind,self.x,self.y,self.y2):w.currentIndexChanged.connect(self.draw)
        self.plot_count.valueChanged.connect(self.draw);self.plot_columns.valueChanged.connect(self.draw)
        self.sync.toggled.connect(self.sync_from_primary)
    def open_file(self,file_filter="Данные (*.json *.csv *.xlsx)"):
        path,_=QFileDialog.getOpenFileName(self,'Данные для сравнения','',file_filter)
        if path:
            try:self.load_path(path)
            except Exception as exc:QMessageBox.warning(self,'Ошибка импорта',str(exc))
    def load_path(self,path):
        if self.thread is not None:raise ValueError('Дождитесь расчёта загруженного проекта.')
        if Path(path).suffix.lower()=='.json':
            cfg,ns=project_from_json(json.loads(Path(path).read_text(encoding='utf-8-sig')))
            cfg.validate();self._loading_cfg=cfg
            self.thread=QThread(self);self.worker=ProjectWorker(cfg,ns);self.worker.moveToThread(self.thread)
            self.thread.started.connect(self.worker.run);self.worker.completed.connect(self._project_done)
            self.worker.failed.connect(self._failed);self.worker.finished.connect(self.thread.quit)
            self.worker.finished.connect(self.worker.deleteLater);self.thread.finished.connect(self._cleanup)
            for button in self.load_buttons:button.setEnabled(False)
            self.notice.setText('Расчёт второго проекта…');self.thread.start()
        else:
            tables=read_chart_tables(path)
            self.tables=tables;self.planes=[];self.cfg=None
            self._populate([t[0] for t in tables]);self.table_controls.show();self.radius.setEnabled(False);self.sync.setEnabled(False);self._plane_changed()
    def _populate(self,labels):
        self.plane.blockSignals(True);self.plane.clear();self.plane.addItems(labels);self.plane.blockSignals(False)
    def _project_done(self,planes):
        self.planes=planes;self.cfg=self._loading_cfg;self.tables=[]
        self.table_controls.hide();self.radius.setEnabled(True);self.sync.setEnabled(True)
        self._populate([f'z={p.z_mm:g} мм' for p in planes]);self._plane_changed();self.sync_from_primary()
    def _failed(self,text):
        self.notice.setText('Не удалось загрузить проект: '+text)
    def _cleanup(self):
        self.thread.deleteLater();self.thread=None;self.worker=None;self._loading_cfg=None
        for button in self.load_buttons:button.setEnabled(True)
    def _plane_changed(self,*args):
        i=max(0,self.plane.currentIndex())
        if self.tables:
            _,headers,columns=self.tables[i]
            for widget in (self.x,self.y,self.y2):
                widget.blockSignals(True);widget.clear()
                if widget is self.y2:widget.addItem('Нет',None)
                for j in columns:widget.addItem(headers[j],j)
                widget.blockSignals(False)
            self.y.blockSignals(True);self.y.setCurrentIndex(1);self.y.blockSignals(False)
        elif self.planes:
            old=self.radius.currentData();self.radius.blockSignals(True);self.radius.clear()
            for r in self.planes[i].radial_centers_mm:self.radius.addItem(f'{r:g} мм',float(r))
            if old is not None:
                k=matching_index(self.planes[i].radial_centers_mm,old)
                if k is not None:self.radius.setCurrentIndex(k)
            self.radius.blockSignals(False)
        self.draw()
    def sync_from_primary(self,*args):
        if not self.planes:return
        if self.sync.isChecked() and self.owner.results:
            p=self.owner.results[max(0,self.owner.plane_select.currentIndex())]
            j=matching_index([v.z_mm for v in self.planes],p.z_mm)
            r=self.owner.compare_radius.currentData()
            if j is None:
                self.notice.setText('Несовместимые z: совпадающего физического сечения нет. Выберите справа вручную.');self.draw();return
            self.plane.setCurrentIndex(j)
            k=matching_index(self.planes[j].radial_centers_mm,r) if r is not None else None
            if k is None:
                self.notice.setText('Радиальные уровни различаются: совпадающего R нет. Выберите справа вручную.');self.draw();return
            self.radius.setCurrentIndex(k)
            compatible=(len(p.radial_centers_mm)==len(self.planes[j].radial_centers_mm) and np.allclose(p.radial_centers_mm,self.planes[j].radial_centers_mm,rtol=0,atol=1e-7) and len(p.theta_centers_deg)==len(self.planes[j].theta_centers_deg) and np.allclose(p.theta_centers_deg,self.planes[j].theta_centers_deg,rtol=0,atol=1e-7) and np.allclose(p.radial_edges_mm,self.planes[j].radial_edges_mm,rtol=0,atol=1e-7))
            self.notice.setText('z и R совпадают.' if compatible else 'z и R совпадают; сетки различаются. Расходы разных ячеек напрямую не сопоставимы.')
        elif not self.owner.results:self.notice.setText('Второй проект рассчитан. Рассчитайте основной проект для синхронизации.')
        else:self.notice.setText('Независимый выбор сечения и радиуса.')
        self.draw()
    def _series(self,ax,x,y,label,color=None):
        if self.kind.currentIndex()==1:ax.scatter(x,y,s=12,label=label,color=color)
        elif self.kind.currentIndex()==2:ax.bar(x,y,label=label,color=color)
        else:ax.plot(x,y,label=label,color=color)
    def _state_key(self):
        i=max(0,self.plane.currentIndex())
        if self.tables:source=('table',id(self.tables),i,self.x.currentData(),self.y.currentData(),self.y2.currentData())
        elif self.planes:source=('project',id(self.planes[i]),self.cfg.density_kg_m3,self.owner.metric.currentData(),self.owner.volume_interval.value(),self.radius.currentData())
        else:source=('empty',)
        return source+(self.kind.currentIndex(),self.plot_count.value(),self.plot_columns.value(),tuple(self.graph_selection.selected))

    def draw(self,*args):
        i=max(0,self.plane.currentIndex())
        if self.tables:
            _,headers,data=self.tables[i];xc,yc=self.x.currentData(),self.y.currentData();secondary=self.y2.currentData()
            choices=[yc]+[k for k in data if k not in (xc,yc,secondary)] if yc in data else []
            self.graph_selection.configure([(k,headers[k]) for k in choices],identity=('table',id(self.tables),i,xc,yc,secondary))
        elif self.planes:
            p=self.planes[i];start=max(0,self.radius.currentIndex());radii=p.radial_centers_mm
            self.graph_selection.configure([(float(r),f'R={r:g} мм') for r in radii],
                [float(radii[(start+k)%len(radii)]) for k in range(len(radii))],identity=('project',tuple(radii)))
        if self._state_key()==self._chart_key:return
        # One replacement of the axes resets annotations and zoom for changed data.
        self.interaction.clear_artists()
        self.figure.clear()
        i=max(0,self.plane.currentIndex())
        series=[];secondary=None
        if self.tables:
            name,headers,data=self.tables[i];xc,yc=self.x.currentData(),self.y.currentData()
            if xc not in data or yc not in data:
                self.interaction.reset_axes();self.canvas.draw_idle();return
            secondary=self.y2.currentData()
            choices=[yc]+[k for k in data if k not in (xc,yc,secondary)]
            maximum=len(choices)
            for key in self.graph_selection.selected:
                series.append((data[xc],data[key],headers[xc],headers[key],name+' · '+headers[key]))
            self.notice.setText('Исходные точки без сглаживания; единицы берутся из заголовков таблицы.')
        elif self.planes:
            p=self.planes[i];start=max(0,self.radius.currentIndex());metric=self.owner.metric.currentData()
            values=cell_values(p,metric,self.cfg.density_kg_m3,self.owner.volume_interval.value())
            maximum=len(p.radial_centers_mm);order=np.argsort(p.theta_centers_deg)
            for radius in self.graph_selection.selected:
                k=matching_index(p.radial_centers_mm,radius)
                series.append((p.theta_centers_deg[order],values[k,order],'Окружной угол, °',UNITS[metric],f'z={p.z_mm:g} мм; R={p.radial_centers_mm[k]:g} мм'))
        else:maximum=1
        self.plot_count.blockSignals(True);self.plot_count.setMaximum(maximum);self.plot_count.blockSignals(False)
        count=max(1,len(series));cols=min(count,self.plot_columns.value());rows=int(np.ceil(count/cols))
        if not series:self.figure.add_subplot(111)
        for slot,(x,y,xlabel,ylabel,title) in enumerate(series,1):
            ax=self.figure.add_subplot(rows,cols,slot);self._series(ax,x,y,ylabel)
            ax.set_xlabel(textwrap.fill(xlabel,28),fontsize=8);ax.set_ylabel(textwrap.fill(ylabel,18),fontsize=8);ax.set_title(title,fontsize=9);ax.grid(alpha=.25);ax.tick_params(labelsize=7)
            ax._compare_x='angle_deg' if not self.tables or xlabel in ('angle_deg','theta_deg','theta_center_deg') else None
            ax._compare_y=UNITS[metric] if not self.tables else ylabel[ylabel.find('[')+1:ylabel.find(']')] if '[' in ylabel and ']' in ylabel else ylabel
            if self.tables and secondary in data:
                other=ax.twinx();self._series(other,data[xc],data[secondary],headers[secondary],color='#ad4e21')
                other.set_ylabel(textwrap.fill(headers[secondary],18),fontsize=8);other.tick_params(axis='y',colors='#ad4e21',labelsize=7)
        self._chart_key=self._state_key()
        self.interaction.reset_axes();self.canvas.draw_idle()
