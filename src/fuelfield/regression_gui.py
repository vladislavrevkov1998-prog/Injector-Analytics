"""Nonmodal staged identification. Workers operate on immutable snapshots."""
import json
from copy import deepcopy
import numpy as np
from PyQt5.QtCore import Qt,QObject,QThread,pyqtSignal
from PyQt5.QtWidgets import (QDialog,QVBoxLayout,QHBoxLayout,QLabel,QPushButton,QComboBox,QTableWidgetItem,
    QDoubleSpinBox,QSpinBox,QFileDialog,QProgressBar)
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from .core import GEOMETRY_MODELS,ANGULAR_MODELS,project_to_json
from .v02 import NOZZLE_KINDS
from .regression import RegressionSnapshot,ParameterSpec,active_parameters,PARAM_UNITS,QUANTITIES,fit
from .regression_io import load_target,write_fit_xlsx
from .editors import PARAMETER_LABELS

class RegressionWorker(QObject):
    completed=pyqtSignal(object);failed=pyqtSignal(str);progress=pyqtSignal(int,object);finished=pyqtSignal()
    def __init__(self,function):super().__init__();self.function=function
    def run(self):
        try:
            result=self.function(self)
            if QThread.currentThread().isInterruptionRequested() and not hasattr(result,'status'):return
            self.completed.emit(result)
        except Exception as exc:self.failed.emit(str(exc))
        finally:self.finished.emit()


def spin(value,low,high,decimals=6):
    widget=QDoubleSpinBox();widget.setDecimals(decimals);widget.setRange(low,high);widget.setValue(value);return widget

class RegressionDialog(QDialog):
    def __init__(self,owner,table_class):
        super().__init__(owner);self.owner=owner;self.thread=None;self.target=None;self.result=None;self.applied_hash=None;self._close_requested=False
        self.snapshot=RegressionSnapshot.capture(owner.cfg,owner.nozzles);self.source_cfg,self.source_ns=self.snapshot.restore()
        self.setWindowTitle('Идентификация параметров: топливо → топливо');self.resize(1120,900)
        layout=QVBoxLayout(self);note=QLabel('Подбор по топливным эталонам на совпадающих контрольных ячейках. Источник остаётся неизменным до «Применить».');note.setWordWrap(True);layout.addWidget(note)
        files=QHBoxLayout();self.load_button=QPushButton('Загрузить топливный JSON / CSV / XLSX');self.load_button.clicked.connect(self.open_target);files.addWidget(self.load_button)
        self.quantity=QComboBox()
        for key,label in [('flux','Плотность расхода, кг/(м²·с)'),('cell_mass','Расход ячейки, кг/с'),('normalized_ring','Индекс кольца'),('normalized_track','Индекс тракта')]:self.quantity.addItem(label,key)
        files.addWidget(QLabel('Величина:'));files.addWidget(self.quantity);files.addStretch();layout.addLayout(files)
        selection=QHBoxLayout();self.z=QComboBox();self.radius=QComboBox();self.angle_start=spin(0,0,360);self.angle_end=spin(360,0,360)
        for label,widget in [('z:',self.z),('R:',self.radius),('θ от:',self.angle_start),('до:',self.angle_end)]:selection.addWidget(QLabel(label));selection.addWidget(widget)
        selection.addStretch();layout.addLayout(selection)
        models=QHBoxLayout();self.geometry=QComboBox();self.angular=QComboBox();self.scope=QComboBox()
        for key,label in GEOMETRY_MODELS.items():self.geometry.addItem(label,key)
        for key,label in ANGULAR_MODELS.items():self.angular.addItem(label,key)
        self.geometry.setCurrentIndex(self.geometry.findData(owner.cfg.geometry_model));self.angular.setCurrentIndex(self.angular.findData(owner.cfg.angular_model))
        self.scope.addItem('Оба типа','all')
        for key,label in NOZZLE_KINDS.items():self.scope.addItem(label,key)
        for widget in (self.geometry,self.angular,self.scope):models.addWidget(widget);widget.currentIndexChanged.connect(self.rebuild_parameters)
        layout.addLayout(models)
        self.parameters=table_class(0,9);self.parameters.setHorizontalHeaderLabels(['Подбирать','Тип','Параметр','Единицы','Старт','min','max','Итог','Δ']);layout.addWidget(self.parameters)
        settings=QHBoxLayout();self.budget=QSpinBox();self.budget.setRange(20,3000);self.budget.setValue(120)
        self.seed=QSpinBox();self.seed.setRange(0,2147483647);self.penalty=spin(0,0,1e9);self.floor=spin(1e-9,1e-12,1e9,12)
        self.criterion=QComboBox();self.criterion.addItem('Абсолютная ошибка','absolute');self.criterion.addItem('Относительная ошибка','relative')
        for label,widget in [('Лимит:',self.budget),('Seed:',self.seed),('λ:',self.penalty),('Критерий:',self.criterion),('Порог y:',self.floor)]:settings.addWidget(QLabel(label));settings.addWidget(widget)
        layout.addLayout(settings);actions=QHBoxLayout()
        for label,attribute,slot in [('Проверить входные данные','check_button',self.check),('Запустить','run_button',self.run),('Отменить','cancel_button',self.cancel),
                                     ('Предпросмотр до/после','preview_button',self.preview),('Применить','apply_button',self.apply),('Вернуть исходное','rollback_button',self.rollback),('XLSX','export_button',self.export)]:
            button=QPushButton(label);setattr(self,attribute,button);button.clicked.connect(slot);actions.addWidget(button)
        layout.addLayout(actions);self.cancel_button.setEnabled(False)
        for widget in (self.apply_button,self.preview_button,self.rollback_button,self.export_button):widget.setEnabled(False)
        self.progress=QProgressBar();layout.addWidget(self.progress);self.status=QLabel('Выберите топливный эталон. Температурные таблицы не являются целями подбора.');self.status.setWordWrap(True);layout.addWidget(self.status)
        self.figure=Figure(figsize=(9,3),tight_layout=True);self.canvas=FigureCanvasQTAgg(self.figure);self.canvas.setMinimumSize(200,170);layout.addWidget(self.canvas)
        for widget in (self.z,self.radius):widget.currentIndexChanged.connect(self.rebuild_parameters)
        for widget in (self.angle_start,self.angle_end):widget.valueChanged.connect(self.rebuild_parameters)
    def open_target(self):
        path,_=QFileDialog.getOpenFileName(self,'Топливный эталон','','Топливные данные (*.json *.csv *.xlsx)')
        if path:self.load_path(path)
    def load_path(self,path):
        quantity=self.quantity.currentData();self._start(lambda worker:load_target(path,quantity,cancel=lambda:QThread.currentThread().isInterruptionRequested()),self._target_done)
    def _target_done(self,target):
        try:target.validate(self.source_cfg)
        except ValueError as exc:self.status.setText(str(exc));return
        self.target=deepcopy(target);self.result=None
        for widget in (self.apply_button,self.preview_button,self.export_button):widget.setEnabled(False)
        self.figure.clear();self.canvas.draw_idle()
        for widget,values,label in [(self.z,sorted(set(target.z_mm)),'Все z'),(self.radius,sorted(set(target.radius_mm)),'Все R')]:
            widget.blockSignals(True);widget.clear();widget.addItem(label,None)
            for value in values:widget.addItem(f'{value:g} мм',value)
            widget.blockSignals(False)
        self.quantity.setCurrentIndex(self.quantity.findData(target.quantity));self.rebuild_parameters()
        self.status.setText(f"Загружен топливный эталон: {target.metadata.get('source_file','')}; {len(target.y)} наблюдений; {target.unit}.")
    def selected_target(self):
        if self.target is None:raise ValueError('Загрузите топливный эталон')
        if self.target.quantity!=self.quantity.currentData():raise ValueError('При смене величины загрузите эталон заново, единицы не переименовываются')
        if self.angle_end.value()<=self.angle_start.value():raise ValueError('Укажите θ конца больше θ начала; полный диапазон 0–360°')
        return self.target.subset(self.z.currentData(),self.radius.currentData(),self.angle_start.value(),self.angle_end.value())
    def rebuild_parameters(self,*args):
        if not hasattr(self,'parameters') or self.target is None:return
        cfg=deepcopy(self.source_cfg);cfg.geometry_model=self.geometry.currentData();cfg.angular_model=self.angular.currentData()
        try:target=self.selected_target()
        except ValueError:return
        specs=[]
        for scope in NOZZLE_KINDS:
            if self.scope.currentData() not in ('all',scope):continue
            for key in active_parameters(cfg,self.source_ns,target,scope):specs.append((scope,key,cfg.nozzle_presets[scope][key]))
        self.parameters.setRowCount(len(specs))
        for row,(scope,key,value) in enumerate(specs):
            delta=max(abs(value)*.4,.001)
            if key in ('clock_deg','alpha_deg','inner_angle_deg','outer_angle_deg'):delta=15.
            if key=='z_tip_mm':delta=10.
            low=value-delta;high=value+delta
            if key in ('h0_mm','r_nozzle_mm','z_ref_mm','decay_length_mm'):low=max(1e-5,low)
            if key=='diffusion_deg_sqrt_mm':low=max(0,low)
            for col,text in enumerate(('',NOZZLE_KINDS[scope],PARAMETER_LABELS[key],PARAM_UNITS[key],f'{value:.9g}',f'{low:.9g}',f'{high:.9g}','—','—')):
                item=QTableWidgetItem(text)
                if col==0:item.setFlags(Qt.ItemIsEnabled|Qt.ItemIsUserCheckable);item.setCheckState(Qt.Unchecked);item.setData(Qt.UserRole,(scope,key))
                elif col not in (4,5,6):item.setFlags(Qt.ItemIsEnabled|Qt.ItemIsSelectable)
                if col in (4,5,6):item.setData(Qt.UserRole,(text,(value,low,high)[col-4]))
                self.parameters.setItem(row,col,item)
        self.parameters.resizeColumnsToContents()
        for col in range(4,9):self.parameters.setColumnWidth(col,max(110,self.parameters.columnWidth(col)))
    def specs(self):
        specs=[]
        for row in range(self.parameters.rowCount()):
            if self.parameters.item(row,0).checkState()!=Qt.Checked:continue
            scope,key=self.parameters.item(row,0).data(Qt.UserRole)
            def number(col):
                item=self.parameters.item(row,col);stored=item.data(Qt.UserRole)
                return stored[1] if stored and item.text()==stored[0] else float(item.text().replace(',','.'))
            initial,low,high=[number(c) for c in (4,5,6)]
            specs.append(ParameterSpec(key,scope,initial,low,high,unit=PARAM_UNITS[key]))
        if not specs:raise ValueError('Выберите параметры для подбора')
        return specs
    def check(self):
        try:
            target=self.selected_target();target.validate(self.source_cfg);specs=self.specs()
            cfg=deepcopy(self.source_cfg);cfg.geometry_model=self.geometry.currentData();cfg.angular_model=self.angular.currentData()
            for p in specs:
                if p.key not in active_parameters(cfg,self.source_ns,target,p.scope) or not p.minimum<=p.initial<=p.maximum or not p.minimum<p.maximum:raise ValueError('Проверьте активность параметров и min/start/max')
            self.status.setText(f'Входы проверены: {sum(target.arrays()[0])} используемых наблюдений, {len(specs)} коэффициентов.')
            return target,specs
        except ValueError as exc:self.status.setText(str(exc));return None
    def run(self):
        checked=self.check()
        if checked is None:return
        target,specs=checked
        kwargs=dict(budget=self.budget.value(),regularization=self.penalty.value(),seed=self.seed.value(),criterion=self.criterion.currentData(),relative_floor=self.floor.value(),geometry_model=self.geometry.currentData(),angular_model=self.angular.currentData())
        self.progress.setRange(0,kwargs['budget']);self.progress.setValue(0)
        self._start(lambda worker:fit(self.snapshot,deepcopy(target),specs,progress=lambda n,cost:worker.progress.emit(n,cost),cancel=lambda:QThread.currentThread().isInterruptionRequested(),**kwargs),self._fit_done)
    def _start(self,function,completed):
        if self.thread is not None:return
        self.thread=QThread(self);self.worker=RegressionWorker(function);self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run);self.worker.completed.connect(completed);self.worker.failed.connect(lambda message:self.status.setText(message))
        self.worker.progress.connect(lambda count,cost:(self.progress.setValue(count),self.status.setText(f'Прямых расчётов: {count}; лучший критерий: {cost}')))
        self.worker.finished.connect(self.thread.quit);self.worker.finished.connect(self.worker.deleteLater);self.thread.finished.connect(self._cleanup)
        self.mutable_controls=(self.quantity,self.z,self.radius,self.angle_start,self.angle_end,self.geometry,self.angular,self.scope,self.parameters,self.budget,self.seed,self.penalty,self.criterion,self.floor)
        for widget in (self.run_button,self.load_button,self.check_button,self.apply_button,*self.mutable_controls):widget.setEnabled(False)
        self.cancel_button.setEnabled(True);self.thread.start()
    def cancel(self):
        if self.thread is not None:self.thread.requestInterruption();self.status.setText('Отмена после текущего прямого расчёта…')
    def _cleanup(self):
        self.thread.deleteLater();self.thread=None;self.worker=None
        for widget in (self.run_button,self.load_button,self.check_button,*self.mutable_controls):widget.setEnabled(True)
        self.cancel_button.setEnabled(False)
        if self._close_requested:self.close()
    def _fit_done(self,result):
        self.result=result;self.preview_button.setEnabled(True);self.export_button.setEnabled(True);self.apply_button.setEnabled(result.status!='cancelled')
        for p,value in zip(result.parameters,result.fitted):
            for row in range(self.parameters.rowCount()):
                if self.parameters.item(row,0).data(Qt.UserRole)==(p.scope,p.key):
                    for col,text in [(7,f'{value:.9g}'),(8,f'{value-p.initial:+.9g}')]:self.parameters.item(row,col).setText(text)
        self.progress.setRange(0,max(1,result.evaluations));self.progress.setValue(result.evaluations)
        fmt=lambda value:'N/A' if value is None else f'{value:.6g}'
        m=result.metrics;self.status.setText(f"{result.status}; train RMSE={fmt(m['train']['RMSE'])}; R²={fmt(m['train']['R2'])}; holdout RMSE={fmt(m['holdout']['RMSE'])}; ранг={result.diagnostics['rank']}; cond={fmt(result.diagnostics['condition'])}; расчётов={result.evaluations}.  "+' '.join(result.warnings))
        self.preview()
    def preview(self):
        if self.result is None:return
        r=self.result;self.figure.clear();ax=self.figure.add_subplot(121);mask,weights,roles=r.target.arrays();use=mask&(weights>0)
        ax.plot(np.asarray(r.target.y)[use],label='Эталон');ax.plot(np.asarray(r.initial_prediction)[use],label='Старт (до подбора)',alpha=.7);ax.plot(np.asarray(r.prediction)[use],label='После',alpha=.7);ax.set_ylabel(r.target.unit);ax.set_xlabel('Наблюдение (z/R/θ)');ax.legend(fontsize=7)
        other=self.figure.add_subplot(122);angles=np.asarray(r.target.theta_deg)[use];residual=(np.asarray(r.prediction)-np.asarray(r.target.y))[use]
        other.scatter(angles,residual,c=np.asarray(r.target.z_mm)[use],s=8);other.set_xlabel('θ, °');other.set_ylabel('Остаток, '+r.target.unit);other.grid(alpha=.2);self.canvas.draw_idle()
    def apply(self):
        if self.result is None or self.result.status=='cancelled':return
        self.owner._record_edit()
        if RegressionSnapshot.capture(self.owner.cfg,self.owner.nozzles).sha256!=self.snapshot.sha256:self.status.setText('Рабочий проект изменён после снимка. Откройте новый подбор.');return
        self.owner.apply_project_state(json.loads(self.result.candidate_payload),'Применить топливную регрессию')
        self.applied_hash=RegressionSnapshot.capture(self.owner.cfg,self.owner.nozzles).sha256;self.rollback_button.setEnabled(True);self.apply_button.setEnabled(False)
        self.status.setText('Применено одной операцией Undo/Redo. Доступно полное восстановление исходного снимка.')
    def rollback(self):
        self.owner._record_edit()
        if self.applied_hash is None or RegressionSnapshot.capture(self.owner.cfg,self.owner.nozzles).sha256!=self.applied_hash:self.status.setText('После применения проект изменён. Используйте Undo для последовательного отката.');return
        self.owner.apply_project_state(json.loads(self.snapshot.payload),'Откат топливной регрессии');self.applied_hash=None;self.rollback_button.setEnabled(False)
        self.status.setText('Исходный снимок восстановлен.');self.apply_button.setEnabled(True)
    def export(self):
        if self.result is None:return
        path,_=QFileDialog.getSaveFileName(self,'Отчёт регрессии','fuel_fit.xlsx','Excel (*.xlsx)')
        if path:write_fit_xlsx(path,self.result)
    def closeEvent(self,event):
        if self.thread is not None:self._close_requested=True;self.cancel();event.ignore();return
        super().closeEvent(event)
