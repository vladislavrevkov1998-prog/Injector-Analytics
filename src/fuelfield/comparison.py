"""Independent comparison data and charts, synchronized by physical coordinates."""
import json
import numpy as np
from pathlib import Path
from PyQt5.QtCore import QObject,QThread,pyqtSignal
from PyQt5.QtWidgets import (QWidget,QVBoxLayout,QHBoxLayout,QComboBox,QPushButton,
    QLabel,QFileDialog,QCheckBox,QMessageBox)
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from .core import project_from_json,calculate_plane
from .analysis import cell_values,matching_index,UNITS
from .analytics_io import read_chart_tables

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
        self._loading_cfg=None
        layout=QVBoxLayout(self);top=QHBoxLayout()
        self.load_button=QPushButton('Сравнить: JSON / CSV / XLSX');self.load_button.clicked.connect(self.open_file);top.addWidget(self.load_button)
        self.sync=QCheckBox('Синхронизировать z и R');self.sync.setChecked(True);top.addWidget(self.sync)
        layout.addLayout(top)
        controls=QHBoxLayout();self.plane=QComboBox();self.radius=QComboBox();self.kind=QComboBox()
        self.kind.addItems(['Линия','Точки','Столбцы'])
        controls.addWidget(QLabel('z / лист'));controls.addWidget(self.plane);controls.addWidget(QLabel('R'));controls.addWidget(self.radius);controls.addWidget(self.kind);layout.addLayout(controls)
        table_controls=QHBoxLayout();self.x=QComboBox();self.y=QComboBox();self.y2=QComboBox()
        for label,widget in [('X',self.x),('Y топливо',self.y),('Y₂ температура',self.y2)]:table_controls.addWidget(QLabel(label));table_controls.addWidget(widget)
        self.table_controls=QWidget();self.table_controls.setLayout(table_controls);layout.addWidget(self.table_controls);self.table_controls.hide()
        self.notice=QLabel('Загрузите второй проект или таблицу с заголовками и числовыми столбцами.');self.notice.setWordWrap(True);layout.addWidget(self.notice)
        self.figure=Figure(figsize=(6,5),tight_layout=True);self.canvas=FigureCanvasQTAgg(self.figure);layout.addWidget(self.canvas)
        self.plane.currentIndexChanged.connect(self._plane_changed)
        for w in (self.radius,self.kind,self.x,self.y,self.y2):w.currentIndexChanged.connect(self.draw)
        self.sync.toggled.connect(self.sync_from_primary)
    def open_file(self):
        path,_=QFileDialog.getOpenFileName(self,'Данные для сравнения','','Данные (*.json *.csv *.xlsx)')
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
            self.load_button.setEnabled(False);self.notice.setText('Расчёт второго проекта…');self.thread.start()
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
        self.thread.deleteLater();self.thread=None;self.worker=None;self._loading_cfg=None;self.load_button.setEnabled(True)
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
    def draw(self,*args):
        self.figure.clear();ax=self.figure.add_subplot(111)
        i=max(0,self.plane.currentIndex())
        if self.tables:
            name,headers,cols=self.tables[i];xc,yc=self.x.currentData(),self.y.currentData()
            if xc not in cols or yc not in cols:return
            self._series(ax,cols[xc],cols[yc],headers[yc]);ax.set_xlabel(headers[xc]);ax.set_ylabel(headers[yc])
            secondary=self.y2.currentData()
            if secondary in cols:
                other=ax.twinx();self._series(other,cols[xc],cols[secondary],headers[secondary],color='#ad4e21');other.set_ylabel(headers[secondary]);other.tick_params(axis='y',colors='#ad4e21')
                for line in other.lines:line.set_color('#ad4e21')
            ax.set_title(name);self.notice.setText('Исходные точки без сглаживания; единицы берутся из заголовков таблицы.')
        elif self.planes:
            p=self.planes[i];k=max(0,self.radius.currentIndex());metric=self.owner.metric.currentData()
            values=cell_values(p,metric,self.cfg.density_kg_m3,self.owner.volume_interval.value())
            order=np.argsort(p.theta_centers_deg)
            self._series(ax,p.theta_centers_deg[order],values[k,order],'Сравниваемый проект')
            ax.set_xlabel('Окружной угол, °');ax.set_ylabel(UNITS[metric]);ax.set_title(f'z={p.z_mm:g} мм; R={p.radial_centers_mm[k]:g} мм')
        ax.grid(alpha=.25);self.canvas.draw_idle()
