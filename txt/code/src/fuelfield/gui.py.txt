"""PyQt5 pilot for ring-field model. Launch with python main.py."""
from __future__ import annotations
import json, os, sys, traceback
from copy import deepcopy
from collections import OrderedDict
from pathlib import Path
import numpy as np
from PyQt5.QtCore import Qt, QThread, QObject, pyqtSignal, QTimer, QSize
from PyQt5.QtGui import QKeySequence, QFont, QPolygon, QColor, QTextDocument
from PyQt5.QtCore import QPoint
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QFormLayout,
    QTabWidget, QTableWidget, QTableWidgetItem, QLabel, QPushButton, QComboBox,
    QSpinBox, QDoubleSpinBox, QLineEdit, QFileDialog, QMessageBox, QSplitter,
    QGroupBox, QScrollArea, QPlainTextEdit, QHeaderView, QAbstractItemView,
    QStatusBar, QProgressBar, QMenu, QAction, QInputDialog, QCheckBox,
    QUndoStack, QUndoCommand, QStackedWidget, QStyledItemDelegate, QDialog, QToolButton, QSlider, QSizePolicy
)
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from .core import (
    Settings, Nozzle, default_nozzles, apply_default_positions,
    GEOMETRY_MODELS, ANGULAR_MODELS, RADIAL_MODELS,
    calculate_plane, project_to_json, project_from_json,
    source_flow_kg_s, evaluate_continuous_field, polar_epure_coordinates
)
from .data_io import import_sector_file, write_sector_csv, write_result_csv
from .visualization import draw_nozzle_markers, draw_detailed_map, draw_polar_nozzles_map, color_scale
from .v02 import (GEOMETRY_KEYS,NOZZLE_KINDS,effective_nozzle,flow_status,
                  validate_nozzles,follow_row,capture_report)
from .characteristics import flow_to_kg_h,flow_from_kg_h
from .editors import CatalogEditor,PresetEditor
from .analysis import METRICS,UNITS,EQUATIONS,cell_values,matching_index
from .plot_interaction import attach_interaction
from .numeric_delegate import NumericDelegate
from .numeric_input import DecimalSpinBox
from .coordinates import to_global,from_global,grid_angles,plot_x,period,x_label,decorate_axes,polar_ticks
from .axes_settings import open_axes_settings
from .analytics_io import write_epures_xlsx
from .comparison import ComparisonPanel,ResultCanvas
from .graph_selection import GraphSelection
from .graph_links import GraphLinks
from .regression_gui import RegressionDialog
from .quasi import calculate_quasi,QuasiSettings,CalculationCancelled,CalculationOutput
from .quasi_gui import QuasiDialog
from .transport_gui import TransportDialog
from .transport_settings import TransportSettings,MODES as TRANSPORT_MODES
from .quasi_plotting import QuasiFieldAnimator
from .quasi_io import write_report as write_quasi_report
from .chamber import chamber,sync_groups,sync_anchors,WALL_BOUND_NOTICE
from .chamber_gui import GeometryEditor
from .chamber_plotting import LongitudinalDialog
from . import __version__

FIELDS = [
    ('id','ID'),('enabled','Установлена'),('nozzle_kind','Тип форсунки'),
    ('manifold_id','Коллектор'),('atomizer_type_id','Имя распылителя'),('row','Ряд'),('manual_write','Ручная запись'),
    ('z_tip_mm','z среза, мм'),('theta_deg','θ форс., °'),('radius_mm','R форс., мм'),
    ('flow_kg_h','Gт, кг/ч'),('flow_mode','Расход из'),('effective_flow','Расход в режиме, кг/ч'),
    ('alpha_deg','α, полный °'),('r_nozzle_mm','R_сопла, мм'),('h0_mm','h0, мм'),
    ('k_linear','k лин.'),('k_quadratic','k квадр.'),('power','p степ.'),
    ('z_ref_mm','z ref, мм'),('inner_angle_deg','α внутр., °'),
    ('outer_angle_deg','α наруж., °'),('clock_deg','Clock, °'),
    ('decay_length_mm','L затух., мм'),('diffusion_deg_sqrt_mm','σ / √мм'),
    ('rotation_deg_per_mm','ω °/мм'),
]
INTCOL = {key:i for i,(key,_) in enumerate(FIELDS)}
READONLY={'id','row','sector_source','effective_flow','flow_status'}
NUMERICS = {f for f,_ in FIELDS} - READONLY - {'enabled','manual_write','row','flow_mode','atomizer_type_id','nozzle_kind','manifold_id'}
SINGLE_NOZZLE_PREVIEW_RESOLUTION = 400  # 2.78× the former 240² display cells


class UnitHeader(QHeaderView):
    def paintSection(self,painter,rect,index):
        super().paintSection(painter,rect,index)
        if index not in (INTCOL['flow_kg_h'],INTCOL['effective_flow']):return
        painter.save();painter.setClipRect(rect)
        painter.fillRect(rect.right()-17,rect.top()+2,15,rect.height()-4,self.palette().button())
        x=rect.right()-9;y=rect.center().y()
        painter.setPen(Qt.NoPen);painter.setBrush(self.palette().buttonText())
        painter.drawPolygon(QPolygon([QPoint(x-4,y-2),QPoint(x+4,y-2),QPoint(x,y+3)]))
        painter.restore()


class TypeDelegate(QStyledItemDelegate):
    def __init__(self,owner,key):super().__init__(owner);self.owner=owner;self.key=key
    def createEditor(self,parent,option,index):
        editor=QComboBox(parent);editor.setEditable(True)
        if self.key=='nozzle_kind':
            for key,label in NOZZLE_KINDS.items():editor.addItem(label,key)
        elif self.key=='atomizer_type_id':
            editor.addItem('','')
            for item in self.owner.cfg.atomizer_types:editor.addItem(item['name'],item['id'])
        else:
            for source in ('manual','sectors','curve'):editor.addItem(source,source)
        return editor
    def setEditorData(self,editor,index):
        value=index.data(Qt.UserRole) if self.key=='atomizer_type_id' else index.data();found=editor.findData(value)
        if found<0:found=editor.findText(value)
        if found>=0:editor.setCurrentIndex(found)
        else:editor.setEditText(value)
    def setModelData(self,editor,model,index):
        value=editor.currentData() if editor.currentIndex()>=0 and editor.currentText()==editor.itemText(editor.currentIndex()) else editor.currentText()
        model.setData(index,value,Qt.EditRole)


def _number(txt: str):
    return float(txt.strip().replace(',','.'))


def _dbl(value, low=-1e7, high=1e7, decimals=5, step=1.):
    c = DecimalSpinBox()
    c.setDecimals(decimals)
    c.setRange(low,high)
    c.setSingleStep(step)
    c.setValue(float(value))
    return c


def _integer(value, lo, hi):
    c=QSpinBox();c.setRange(lo,hi);c.setValue(value);return c


class PasteTable(QTableWidget):
    """Tab-separated copy/paste (Excel compatible)."""
    batchFinished=pyqtSignal()

    def __init__(self,*args):
        super().__init__(*args)
        self.in_batch=False
        self.batch_invalid=False
        self.setItemDelegate(NumericDelegate(self))

    def setItem(self,row,column,item):
        if not item.flags() & (Qt.ItemIsEditable | Qt.ItemIsUserCheckable):
            item.setBackground(QColor('#eee6f6'))
        super().setItem(row,column,item)

    def paste_text(self,raw):
        start=self.currentIndex()
        if not start.isValid() or not raw:return
        lines=raw.rstrip('\r\n').splitlines()
        cells=[line.split('\t') for line in lines]
        if len(cells)==1 and len(cells[0])==1:
            targets=[(i.row(),i.column(),cells[0][0]) for i in self.selectedIndexes()]
        else:
            targets=[(start.row()+dr,start.column()+dc,txt)
                     for dr,line in enumerate(cells) for dc,txt in enumerate(line)]
        self.in_batch=True
        self.batch_invalid=False
        self.setUpdatesEnabled(False)
        try:
            for row,col,txt in targets:
                if row>=self.rowCount() or col>=self.columnCount():continue
                item=self.item(row,col)
                if item is not None and item.flags() & Qt.ItemIsEditable:
                    item.setText(txt.strip())
        finally:
            self.in_batch=False
            self.setUpdatesEnabled(True)
            self.batchFinished.emit()

    def keyPressEvent(self, ev):
        if ev.matches(QKeySequence.Copy):
            sel=self.selectedIndexes()
            if not sel:return
            r0=min(i.row() for i in sel);r1=max(i.row() for i in sel)
            c0=min(i.column() for i in sel);c1=max(i.column() for i in sel)
            lines=[]
            for row in range(r0,r1+1):
                values=[]
                for col in range(c0,c1+1):
                    item=self.item(row,col)
                    values.append(item.text() if item else '')
                lines.append('\t'.join(values))
            QApplication.clipboard().setText('\n'.join(lines))
            return
        if ev.matches(QKeySequence.Paste):
            self.paste_text(QApplication.clipboard().text())
            return
        super().keyPressEvent(ev)


class ProjectEdit(QUndoCommand):
    """One validated project edit, including a whole paste or group operation."""
    def __init__(self,window,before,after,label):
        super().__init__(label)
        self.window=window
        self.before=deepcopy(before);self.after=deepcopy(after)
        self.first=True

    def redo(self):
        if self.first:self.first=False
        else:self.window._restore_edit(self.after)

    def undo(self):
        self.window._restore_edit(self.before)


class CalcWorker(QObject):
    progress=pyqtSignal(str)
    completed=pyqtSignal(object)
    failed=pyqtSignal(str)
    cancelled=pyqtSignal()
    finished=pyqtSignal()
    def __init__(self,settings,nozzles):
        super().__init__()
        self.settings=settings;self.nozzles=nozzles
    def run(self):
        try:
            if self.settings.quasistationary:
                self.quasi_result=calculate_quasi(self.nozzles,self.settings,cancel=lambda:QThread.currentThread().isInterruptionRequested(),progress=self.progress.emit)
                self.completed.emit(CalculationOutput([(p,self.quasi_result.smooth(i,0)) for i,p in enumerate(self.quasi_result.composites)],self.quasi_result));return
            result=[]
            for i,z in enumerate(self.settings.z_mm):
                if QThread.currentThread().isInterruptionRequested():self.cancelled.emit();return
                self.progress.emit(f'Сечение {i+1}/{len(self.settings.z_mm)}, z={z:g} мм…')
                plane=calculate_plane(self.nozzles,self.settings,z)
                if QThread.currentThread().isInterruptionRequested():self.cancelled.emit();return
                preview=evaluate_continuous_field(self.nozzles,self.settings,z)
                result.append((plane,preview))
            if QThread.currentThread().isInterruptionRequested():self.cancelled.emit();return
            self.completed.emit(result)
        except CalculationCancelled:self.cancelled.emit()
        except Exception:
            self.failed.emit(traceback.format_exc())
        finally:self.finished.emit()


class PlotStack(QStackedWidget):
    """Cached figure sizes must not push the result panel up on first display."""
    def sizeHint(self):return QSize(400,240)
    def minimumSizeHint(self):return QSize(0,0)


class MainWindow(QMainWindow):
    RENDER_CACHE_LIMIT=24
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Fuel Field Pilot — v'+__version__)
        self.resize(1480,930)
        self.cfg=Settings(pressure_mode='calibrated',flow_pressure_kgf_cm2=30.,sector_pressure_kgf_cm2=2.5)
        self.nozzles=default_nozzles(self.cfg)
        for n in self.nozzles:n.geometry_overrides=[]
        self.results=[]
        self.smooth_fields=[]
        self.quasi_result=None
        self.time_timer=QTimer(self);self.time_timer.setSingleShot(True);self.time_timer.setInterval(16);self.time_timer.timeout.connect(self._update_time_field)
        self.sector_id=None
        self.project_file=None
        self.thread=None
        self._restoring=True
        self._render_cache=OrderedDict()
        self._revision=0
        self.undo_stack=QUndoStack(self)
        self.undo_stack.setUndoLimit(100)
        self.preview_timer=QTimer(self)
        self.preview_timer.setSingleShot(True)
        self.preview_timer.timeout.connect(self._draw_single_preview)
        self.selection_timer=QTimer(self);self.selection_timer.setSingleShot(True)
        self.selection_timer.timeout.connect(self._apply_sector_selection)
        self._preview_key=None
        self.model_widgets={}
        self.geom_widgets={}
        self.mode_widgets={}
        self.tabs=QTabWidget()
        self.setCentralWidget(self.tabs)
        self._build_geometry_tab()
        self._build_nozzles_tab()
        self._build_models_tab()
        self._build_results_tab()
        self._build_menu()
        self.statusBar().showMessage('Начальные размеры и расход — демонстрационные/пустые. Задайте реальные значения перед расчётом.')
        self._fill_nozzles()
        self._fill_controls()
        self._restoring=False
        self._edit_state=project_to_json(self.cfg,self.nozzles)
        self._connect_edits()
        self.tabs.currentChanged.connect(self._main_tab_changed)
        self._queue_preview()

    def _build_menu(self):
        menu=self.menuBar().addMenu('Проект')
        for label,slot,shortcut in [
            ('Новый проект',self.new_project,'Ctrl+N'),
            ('Открыть проект JSON',self.open_project,'Ctrl+O'),
            ('Сохранить проект JSON',self.save_project,'Ctrl+S'),
            ('Сохранить как…',lambda:self.save_project(True),'Ctrl+Shift+S'),
        ]:
            act=QAction(label,self)
            act.setShortcut(shortcut);act.triggered.connect(slot);menu.addAction(act)
        menu=self.menuBar().addMenu('Расчёт')
        a=QAction('Рассчитать сечения',self);a.setShortcut('F5');a.triggered.connect(self.run_calc);menu.addAction(a)
        edit=self.menuBar().addMenu('Правка')
        undo=self.undo_stack.createUndoAction(self,'Отменить')
        undo.setShortcut('Ctrl+Z');edit.addAction(undo)
        redo=self.undo_stack.createRedoAction(self,'Повторить')
        redo.setShortcuts([QKeySequence('Ctrl+Y'),QKeySequence('Ctrl+Shift+Z')])
        edit.addAction(redo)
        menu=self.menuBar().addMenu('Настройки')
        action=QAction('Испытания и расстановка',self);action.triggered.connect(self.open_settings);menu.addAction(action)
        self.transport_settings_action=QAction('Перенос топлива внутри ЖТ…',self);self.transport_settings_action.triggered.connect(self.open_transport_settings);menu.addAction(self.transport_settings_action)

    def _build_geometry_tab(self):
        page=QWidget();layout=QVBoxLayout(page)
        warning=QLabel('ПРОВЕРОЧНЫЕ значения геометрии по умолчанию не являются реальными размерами ФУ. Введите свои.')
        warning.setStyleSheet('color:#b45309;font-weight:bold;padding:8px;')
        layout.addWidget(warning)
        scroll=QScrollArea();scroll.setWidgetResizable(True)
        panel=QWidget(); form=QFormLayout(panel)
        fields=[
            ('r_in_mm','Внутренний радиус тракта, мм',_dbl(200,0,10000)),
            ('r_out_mm','Наружный радиус тракта, мм',_dbl(300,0,10000)),
            ('row_inner_mm','Радиус расположения внутреннего ряда, мм',_dbl(230,0,10000)),
            ('row_outer_mm','Радиус расположения наружного ряда, мм',_dbl(270,0,10000)),
            ('nozzle_diameter_mm','Диаметр контура форсунки на схеме, мм',_dbl(16,0.1,500)),
            ('preview_resolution','Разрешение подробной карты (пикс./ось)',_integer(480,200,1000)),
            ('n_radial','Число радиальных уровней',_integer(6,1,40)),
            ('n_angular','Число окружных ячеек',_integer(140,8,1440)),
            ('theta_zero_deg','Положение 0° от физической оси +X против ЧС, °',_dbl(0,-360,360,8,.001)),
            ('theta_direction','Направление увеличения угла (движение гребёнки)',QComboBox()),
            ('grid_start_deg','Смещение первого центра матрицы от нуля, °',_dbl(0,-360,360)),
            ('z_mm','Координаты сечений z, мм (1–50)',QLineEdit('0; 20; 40; 60')),
            ('radial_centers_mm','Радиусы центров уровней, мм (пусто = равномерно)',QLineEdit('')),
        ]
        for key,label,w in fields:
            self.geom_widgets[key]=w
            if isinstance(w,QLineEdit):w.setMinimumWidth(380)
            if key=='theta_direction':w.addItem('Против часовой стрелки',1);w.addItem('По часовой стрелке',-1)
            form.addRow(label,w)
        frame_note=QLabel('Смена нуля/направления меняет отсчёт углов и порядок ячеек, сохраняя физические положения форсунок. Сектора 1–36 каждой форсунки сохраняют прежний отсчёт против ЧС.');frame_note.setWordWrap(True);form.addRow(frame_note)
        sections=QPushButton("Редактор сечений: добавить / удалить / порядок");sections.clicked.connect(self.open_sections_editor);form.addRow(sections)
        btnrow=QHBoxLayout()
        btn=QPushButton('Применить позиции двух рядов к 140 форсункам')
        btn.setToolTip('Изменяет только радиус и угол форсунок, не затрагивая включение, номер, расход и эпюру.')
        btn.clicked.connect(self.update_positions)
        btnrow.addWidget(btn)
        btn2=QPushButton('Восстановить равномерные радиальные уровни')
        btn2.clicked.connect(lambda:self.geom_widgets['radial_centers_mm'].setText(''))
        btnrow.addWidget(btn2);btnrow.addStretch()
        form.addRow(btnrow)
        explanation=QLabel('Пустая строка радиальных уровней → равномерные центры 6 полос. '
                           'Если задать радиусы вручную, ширины ячеек считаются по серединам между соседними центрами; '
                           'первая/последняя границы равны радиусам стенок. Поворот матрицы не поворачивает форсунки.')
        explanation.setWordWrap(True)
        form.addRow(explanation)
        scroll.setWidget(panel)
        editors=QTabWidget()
        self.catalog_editor=CatalogEditor(self,PasteTable)
        self.catalog_editor.changed.connect(self._catalog_changed)
        self.preset_editor=PresetEditor(self,PasteTable)
        self.preset_editor.changed.connect(self._presets_changed)
        editors.addTab(self.catalog_editor,'Типы распылителей и расходные характеристики')
        editors.addTab(self.preset_editor,'Шаблоны форсунок')
        form.addRow(editors)
        self.geometry_pages=QTabWidget();self.geometry_pages.addTab(scroll,'Сетка, измерительный отсчёт и распылители')
        self.geometry_editor=GeometryEditor(self,PasteTable);self.geometry_pages.addTab(self.geometry_editor,'Профили ЖТ: Sketcher и таблицы')
        layout.addWidget(self.geometry_pages)
        self.tabs.addTab(page,'01 · Геометрия и сетка')

    def open_longitudinal(self):
        self.longitudinal_dialog=LongitudinalDialog(self);self.longitudinal_dialog.show()

    def open_regression(self):
        self._record_edit()
        if TransportSettings.from_dict(self.cfg.fuel_transport).enabled:
            self._error('Идентификация параметров прежнего ядра не применима к инерционному переносу. Параметры τ, скоростей и флуктуаций задаются в «Настройки → Перенос топлива внутри ЖТ». Методика их идентификации требует отдельного топливного эталона.');return
        if self.cfg.quasistationary:self._error('Идентификация v0.4.1 использует стационарное ядро. Для неё выберите стационарный режим; настройки квазистационарного режима сохранятся.');return
        existing=getattr(self,'regression_dialog',None)
        if existing is not None and existing.thread is not None:existing.show();existing.raise_();return
        self.regression_dialog=RegressionDialog(self,PasteTable);self.regression_dialog.show()

    def apply_project_state(self,state,label):
        before=project_to_json(self.cfg,self.nozzles);cfg,nozzles=project_from_json(deepcopy(state))
        self._restoring=True
        try:self.cfg=cfg;self.nozzles=nozzles;self._fill_controls();self._fill_nozzles();self._edit_state=before
        finally:self._restoring=False
        self._record_edit(label=label)

    def apply_geometry_state(self,state,label,record_history=True):
        """A validated geometry edit needs neither a 140-row read-back nor
        rebuilding hidden hydraulic plots. It shares the same Undo snapshots.
        """
        cfg,nozzles=project_from_json(state);after=project_to_json(cfg,nozzles);before=self._edit_state
        if after==before:return
        previous_nozzles=self.nozzles;self._restoring=True
        try:
            self.cfg=cfg;self.nozzles=nozzles;self._edit_state=deepcopy(after)
            self.geom_widgets['z_mm'].setText('; '.join(f'{z:g}' for z in cfg.z_mm))
            self.mode_widgets['placement_mode'].setCurrentIndex(self.mode_widgets['placement_mode'].findData(cfg.placement_mode))
            active=bool(chamber(cfg))
            for key in ('r_in_mm','r_out_mm','z_mm'):
                self.geom_widgets[key].setEnabled(not active);self.geom_widgets[key].setToolTip('Задаётся в «Профили ЖТ: Sketcher и таблицы»' if active else '')
            if nozzles!=previous_nozzles:self._fill_nozzles()
            self.geometry_editor.load(cfg,nozzles)
        finally:self._restoring=False
        if record_history:self.undo_stack.push(ProjectEdit(self,before,after,label))
        self._invalidate_results();self._queue_preview()

    def open_sections_editor(self):
        if chamber(self.cfg):
            self.tabs.setCurrentIndex(0);self.geometry_pages.setCurrentIndex(1);self.geometry_editor.pages.setCurrentIndex(3);return
        dialog=QDialog(self);dialog.setWindowTitle('Осевые сечения: 1–50');layout=QVBoxLayout(dialog)
        layout.addWidget(QLabel('Глобальные координаты от начала ФУ, мм. Больше сечений — дольше расчёт.'))
        table=PasteTable(len(self.cfg.z_mm),1);table.setHorizontalHeaderLabels(['z, мм'])
        for row,z in enumerate(self.cfg.z_mm):table.setItem(row,0,QTableWidgetItem(repr(z)))
        layout.addWidget(table);buttons=QHBoxLayout()
        def add():
            if table.rowCount()>=50:return
            row=table.rowCount();table.insertRow(row);table.setItem(row,0,QTableWidgetItem(str(max([_number(table.item(i,0).text()) for i in range(row)]+[0])+10)))
        def remove():
            for row in sorted({i.row() for i in table.selectedIndexes()},reverse=True):
                if table.rowCount()>1:table.removeRow(row)
        def move(delta):
            row=table.currentRow();other=row+delta
            if row<0 or not 0<=other<table.rowCount():return
            a=table.takeItem(row,0);b=table.takeItem(other,0);table.setItem(row,0,b);table.setItem(other,0,a);table.setCurrentCell(other,0)
        def sort():
            try:values=sorted(_number(table.item(i,0).text()) for i in range(table.rowCount()))
            except ValueError:self._error('Введите числовые координаты.');return
            for i,z in enumerate(values):table.item(i,0).setText(repr(z))
        def apply():
            try:
                values=[_number(table.item(i,0).text()) for i in range(table.rowCount())];cfg=deepcopy(self.cfg);cfg.z_mm=values;cfg.validate()
            except ValueError as exc:self._error(exc);return
            self.geom_widgets['z_mm'].setText('; '.join(repr(z) for z in values));self._record_edit(label='Осевые сечения');dialog.accept()
        for label,slot in [('Добавить',add),('Удалить',remove),('↑',lambda:move(-1)),('↓',lambda:move(1)),('Упорядочить',sort),('Применить',apply)]:
            button=QPushButton(label);button.clicked.connect(slot);buttons.addWidget(button)
        layout.addLayout(buttons);dialog.resize(620,430);self.sections_dialog=dialog;self.sections_table=table;dialog.show()

    def open_settings(self):
        self.settings_page.show()
        self.settings_page.raise_()
        self.settings_page.activateWindow()

    def _quasi_toggled(self,enabled):
        if self._restoring:return
        self._record_edit(label='Режим расчёта')
        if enabled and self.cfg.quasistationary:self.open_quasi_settings()

    def open_quasi_settings(self):
        self._record_edit()
        existing=getattr(self,'quasi_dialog',None)
        if existing is not None and existing.isVisible():existing.raise_();existing.activateWindow();return
        self.quasi_dialog=QuasiDialog(self,PasteTable);self.quasi_dialog.show()

    def apply_quasi_settings(self,raw,bindings=None):
        self._record_edit();state=project_to_json(self.cfg,self.nozzles);state['settings']['quasi_settings']=deepcopy(raw)
        for n in state['nozzles']:
            if str(n['id']) in (bindings or {}):n['manifold_id']=bindings[str(n['id'])]
        self.apply_project_state(state,'Настройки квазистационарного режима')

    def open_transport_settings(self):
        self._record_edit()
        existing=getattr(self,'transport_dialog',None)
        if existing is not None and existing.isVisible():existing.raise_();existing.activateWindow();return
        self.transport_dialog=TransportDialog(self);self.transport_dialog.show()

    def apply_transport_settings(self,raw):
        self._record_edit();cfg=deepcopy(self.cfg)
        cfg.fuel_transport=TransportSettings.from_dict(raw).validate().to_dict()
        self.apply_project_state(project_to_json(cfg,self.nozzles),'Модель переноса топлива')

    def _build_nozzles_tab(self):
        page=QWidget();layout=QVBoxLayout(page);top=QHBoxLayout()
        for label,slot in [
            ('Импорт секторов',self.import_sectors),('Экспорт эпюр',self.export_sectors),
            ('Копировать эпюру',self.copy_sectors),('Расход из объёмов',self.sum_sectors_to_flow),
            ('ДЕМО: 10 кг/ч',self.assign_demo_flow)]:
            button=QPushButton(label);button.clicked.connect(slot);top.addWidget(button)
        top.addStretch();gear=QPushButton('⚙');gear.setToolTip('Настройки испытаний и расстановки')
        gear.setFixedWidth(36);gear.clicked.connect(self.open_settings);top.addWidget(gear);layout.addLayout(top)
        self.settings_page=QDialog(self);self.settings_page.setWindowTitle('Настройки испытаний и расстановки');settings=QVBoxLayout(self.settings_page);form=QFormLayout()
        self.duration_spin=_dbl(180,.01,1e6,3,10.)
        self.density_spin=_dbl(800,1,10000,2,10.)
        form.addRow('Время сбора, с',self.duration_spin);form.addRow('Плотность топлива, кг/м³',self.density_spin)
        pressure_mode=QComboBox();pressure_mode.addItem('Прежний режим проекта','legacy')
        pressure_mode.addItem('Пересчёт по характеристикам','calibrated')
        target=_dbl(30,.00001,100000,5)
        extrap=QCheckBox('Разрешить экспериментальную экстраполяцию (с предупреждением)')
        placement=QComboBox();placement.addItem('Позиции: ручные','manual')
        placement.addItem('Auto-follow обоих рядов по часовой стрелке','auto_follow')
        self.mode_widgets=dict(pressure_mode=pressure_mode,target_pressure_kgf_cm2=target,
            allow_extrapolation=extrap,placement_mode=placement,
            flow_pressure_kgf_cm2=_dbl(30,.00001,100000,5),
            sector_pressure_kgf_cm2=_dbl(2.5,.00001,100000,5))
        for key,label in [('pressure_mode','Режим расходов'),('target_pressure_kgf_cm2','Целевой Δp, кгс/см²'),
            ('flow_pressure_kgf_cm2','Δp измерения Gт, кгс/см²'),('sector_pressure_kgf_cm2','Δp секторов, кгс/см²'),
            ('allow_extrapolation','Экстраполяция'),('placement_mode','Расстановка')]:form.addRow(label,self.mode_widgets[key])
        settings.addLayout(form)
        mode_row=QHBoxLayout();self.quasi_check=QCheckBox('Квазистационарный (без галочки — стационарный)');self.mode_widgets['quasistationary']=self.quasi_check;mode_row.addWidget(self.quasi_check)
        self.quasi_settings_button=QPushButton('Настройки режима…');self.quasi_settings_button.clicked.connect(self.open_quasi_settings);mode_row.addWidget(self.quasi_settings_button);mode_row.addStretch();settings.addLayout(mode_row);self.quasi_check.toggled.connect(self._quasi_toggled)
        self.flow_notice=QLabel();self.flow_notice.setWordWrap(True);settings.addWidget(self.flow_notice)
        transport_row=QHBoxLayout();self.transport_mode_label=QLabel();self.transport_mode_label.setWordWrap(True);transport_row.addWidget(self.transport_mode_label,1)
        self.transport_settings_button=QPushButton('Перенос топлива…');self.transport_settings_button.clicked.connect(self.open_transport_settings);transport_row.addWidget(self.transport_settings_button);settings.addLayout(transport_row)
        close=QPushButton('Закрыть настройки');close.clicked.connect(self.settings_page.close)
        settings.addWidget(close);settings.addStretch()
        # Retain the group-operation API for existing scripts; assignment in the
        # visible UI is now performed directly in cells through editable lists.
        self.assign_type_selector=QComboBox(self);self.assign_type_selector.hide()
        self.assign_kind_selector=QComboBox(self);self.assign_kind_selector.hide()
        for key,label in NOZZLE_KINDS.items():self.assign_kind_selector.addItem(label,key)
        splitter=QSplitter(Qt.Vertical)
        self.nozzle_table=PasteTable(140,len(FIELDS));self.nozzle_table.setHorizontalHeader(UnitHeader(Qt.Horizontal,self.nozzle_table))
        self.nozzle_table.setHorizontalHeaderLabels([label for _,label in FIELDS])
        self.nozzle_table.setAlternatingRowColors(True)
        self.nozzle_table.setSelectionBehavior(QAbstractItemView.SelectItems)
        self.nozzle_table.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.nozzle_table.verticalHeader().setDefaultSectionSize(25)
        header=self.nozzle_table.horizontalHeader();header.setSectionsClickable(True);header.setDefaultSectionSize(112);header.setMinimumSectionSize(48)
        self.nozzle_table.setColumnWidth(INTCOL['id'],52)
        for key,width in [('nozzle_kind',130),('atomizer_type_id',145),('manual_write',125),('effective_flow',200)]:
            self.nozzle_table.setColumnWidth(INTCOL[key],width)
        for key in ('manifold_id','z_tip_mm','theta_deg','radius_mm','flow_kg_h','flow_mode'):
            self.nozzle_table.setColumnWidth(INTCOL[key],100)
        for key in ('manifold_id','flow_mode'):self.nozzle_table.setColumnWidth(INTCOL[key],96)
        for key in ('nozzle_kind','atomizer_type_id','flow_mode'):
            self.nozzle_table.setItemDelegateForColumn(INTCOL[key],TypeDelegate(self,key))
        self.nozzle_table.currentCellChanged.connect(self._sector_selection)
        # Opening on press also works when Qt scrolls a partly visible header
        # section before release (which otherwise cancels sectionClicked).
        header.sectionPressed.connect(self._toggle_flow_unit);splitter.addWidget(self.nozzle_table)
        box=QWidget();bottom=QVBoxLayout(box)
        self.sector_label=QLabel();self.sector_label.setWordWrap(True);bottom.addWidget(self.sector_label)
        self.sector_table=PasteTable(36,2);self.sector_table.setHorizontalHeaderLabels(['Сектор №','Объём, мл'])
        self.sector_table.horizontalHeader().setStretchLastSection(True);self.sector_table.setMaximumWidth(290)
        for k in range(36):
            item=QTableWidgetItem(str(k+1));item.setFlags(Qt.ItemIsEnabled);self.sector_table.setItem(k,0,item)
        self.sector_table.itemChanged.connect(self._sector_changed)
        row=QHBoxLayout();row.addWidget(self.sector_table)
        fine=QWidget();fine.setMaximumWidth(330);fine_layout=QVBoxLayout(fine)
        self.capture_label=QLabel();self.capture_label.setWordWrap(True);fine_layout.addWidget(self.capture_label)
        self.nozzle_status=QLabel();self.nozzle_status.setWordWrap(True);fine_layout.addWidget(self.nozzle_status)
        note=QLabel('Сектор 1 — к средней линии ФУ; 2–36 — против часовой стрелки со стороны выхода топлива. '
            'Gт хранится независимо от секторов. Положительное отклонение означает недобор; отрицательное — избыток.')
        note.setWordWrap(True);fine_layout.addWidget(note);fine_layout.addStretch();row.addWidget(fine)
        self.figpreview=Figure(figsize=(7,3),tight_layout=True);self.previewcanvas=FigureCanvas(self.figpreview)
        row.addWidget(self.previewcanvas,2);bottom.addLayout(row)
        splitter.addWidget(box);splitter.setStretchFactor(0,3);splitter.setStretchFactor(1,2)
        layout.addWidget(splitter);self.tabs.addTab(page,'02 · 140 форсунок и 36 секторов')

    def _build_models_tab(self):
        page=QWidget();lay=QVBoxLayout(page);form=QFormLayout()
        for key,label,vals in (
            ('geometry_model','Геометрия кольцевого факела',GEOMETRY_MODELS),
            ('radial_model','Распределение по толщине',RADIAL_MODELS),
            ('angular_model','Угловая неравномерность с удалением',ANGULAR_MODELS),
        ):
            cb=QComboBox()
            for value,name in vals.items():cb.addItem(name,value)
            self.model_widgets[key]=cb;form.addRow(label,cb)
        for key,label,value,lo,hi in (
            ('n_sample_radial','Интеграция: узлов по радиусу источника',80,8,1000),
            ('n_sample_angular','Интеграция: узлов по окружности источника',720,72,7200),
        ):
            c=_integer(value,lo,hi)
            self.model_widgets[key]=c;form.addRow(label,c)
        lay.addLayout(form)
        description=QPlainTextEdit();description.setReadOnly(True)
        description.setPlainText('''НАЧАЛЬНАЯ ПЛЁНКА
R_сопла — наружный радиус плёнки, начальная зона [R_сопла-h0, R_сопла].
Rc0 = R_сопла-h0/2 — производный средний радиус, не отдельный ввод.

МОДЕЛИ ГЕОМЕТРИИ (все расстояния и параметры задаются в мм)
Rc(z) = Rc0 + z·tan(α/2), где α — ПОЛНЫЙ угол раскрытия средней линии кольца.
  Постоянная толщина: h = h0.
  Линейная толщина: h = h0 + k_linear·z.
  Квадратичная толщина: h = h0 + k_linear·z + k_quadratic·z².
  Степенная толщина: h = h0·(1+z/z_ref)^p.
  Два угла: Rmin = Rc0-h0/2 + z·tan(αвнут), Rmax = Rc0+h0/2 + z·tan(αнаруж).
Границы кольца равны Rc(z) ± h(z)/2. Значения k и p редактируются отдельно для каждой форсунки.

РАДИАЛЬНЫЕ ПРОФИЛИ
  Равномерный: постоянная плотность по площади в кольце Rmin..Rmax.
  Треугольный: максимум на Rc и ноль на Rmin и Rmax.
  Гауссов: максимум на Rc, h — полная ширина на половине максимума (FWHM).
  Гауссов профиль не имеет резкой наружной/внутренней границы; интегрируется в пределах ±6σ.

УГЛОВЫЕ ПРОФИЛИ
  Сохранение: используется измеренная эпюра без дополнительного сглаживания.
  Линейное затухание: λ=max(0, 1-z/L), A(z)=1+λ·(A0-1).
  Экспоненциальное: λ=exp(-z/L), A(z)=1+λ·(A0-1).
  Круговое гауссово размытие: σφ(z) [град] = σ [град/√мм]·√z; FFT-круговая свёртка.
  Параметр ω поворачивает эпюру на ω·z градусов; Clock задаёт угол установки.

ЧИСЛЕННЫЙ МЕТОД И ФИЗИЧЕСКИЕ ОГРАНИЧЕНИЯ
  Интеграционная сетка строится по локальному радиусу/углу каждого факела.
  Массовый расход каждой форсунки нормируется на выбранной сетке точно.
  Пробы распределяются в ячейки матрицы или во внутреннюю/наружную области за стенками.
  Баланс: Σрасходов = Σячейки + за внутренней стенкой + за наружной стенкой.
  Выход за стенку — ГЕОМЕТРИЧЕСКАЯ классификация, НЕ модель осаждения или Wall Film.
  Суперпозиция линейная: взаимодействия капель/потоков соседних форсунок не учитываются.
  Пользовательская сетка 6×140 НЕ является интеграционной: она определяет разрешение выходных карт.''')
        lay.addWidget(description)
        formulas=Figure(figsize=(10,3),tight_layout=True)
        ax=formulas.add_subplot(111);ax.axis('off')
        equations=[
            r'$R_c(z)=R_{\mathrm{nozzle}}-h_0/2+z\tan(\alpha/2),\quad R_{\min,\max}=R_c\mp h/2$',
            r'$A_k=36V_k/\sum_{l=1}^{36}V_l,\quad \dot m_{\mathrm{sectors}}=10^{-6}\rho_l\sum_k V_k/T$',
            r'$q(x,y,z)=\sum_j\dot m_j G_j A_j/C_j\quad [\mathrm{kg}/(\mathrm{m}^2\,\mathrm{s})]$',
            r'$\sum_j\dot m_j=\sum_{m,n}q_{mn}S_{mn}+\dot m_{\mathrm{inner}}+\dot m_{\mathrm{outer}}$',
        ]
        for i,text in enumerate(equations):ax.text(.02,.92-i*.24,text,fontsize=13,va='top')
        lay.addWidget(FigureCanvas(formulas))
        lay.addWidget(QLabel('R_nozzle = R_сопла; R, h, z — мм; V — мл; T — с; ρₗ — кг/м³; ṁ — кг/с; '
                            'G, A — профили; C — нормировка; S — площадь ячейки [м²].'))
        self.tabs.addTab(page,'03 · Математические модели')

    def _build_results_tab(self):
        page=QWidget();lay=QVBoxLayout(page)
        actions=QHBoxLayout()
        self.calc_btn=QPushButton('Рассчитать сечения (F5)')
        self.calc_btn.clicked.connect(self.run_calc)
        actions.addWidget(self.calc_btn)
        self.cancel_calc_btn=QPushButton("Отменить");self.cancel_calc_btn.setEnabled(False);self.cancel_calc_btn.clicked.connect(lambda:self.thread.requestInterruption() if self.thread else None);actions.addWidget(self.cancel_calc_btn)
        self.plane_select=QComboBox();self.plane_select.setMinimumWidth(150);self.plane_select.currentIndexChanged.connect(self._plot_result)
        actions.addWidget(QLabel('Сечение:'));actions.addWidget(self.plane_select)
        for label,slot in [('XLSX: эпюры',self.export_xlsx),('CSV: матрицы',self.export_results),('PNG: текущее',self.export_png),
                            ('PNG: все z',self.export_all_png)]:
            b=QPushButton(label);b.clicked.connect(slot);actions.addWidget(b)
        regression=QPushButton("Идентификация параметров");regression.clicked.connect(self.open_regression);actions.addWidget(regression)
        actions.addStretch();lay.addLayout(actions)
        self.progress=QProgressBar();self.progress.setRange(0,4);self.progress.setValue(0)
        lay.addWidget(self.progress)
        view=QHBoxLayout()
        self.draw_outlines=QCheckBox('Контуры форсунок');self.draw_outlines.setChecked(True)
        self.draw_labels=QCheckBox('Номера форсунок');self.draw_labels.setChecked(True)
        self.draw_rose_over_field=QCheckBox('Полярные эпюры поверх подробного поля')
        self.draw_rose_over_field.setChecked(False)
        self.rose_scale=_dbl(8.0,0.1,100,1,1)
        for w in (self.draw_outlines,self.draw_labels,self.draw_rose_over_field):
            view.addWidget(w);w.stateChanged.connect(self._view_changed)
        view.addWidget(QLabel('Масштаб полярных эпюр, мм:'))
        view.addWidget(self.rose_scale);self.rose_scale.valueChanged.connect(self._view_changed)
        view.addStretch();lay.addLayout(view)
        scales=QHBoxLayout()
        self.scale_mode=QComboBox();self.scale_mode.addItem('Auto','auto');self.scale_mode.addItem('Manual','manual')
        self.scale_min=_dbl(0,decimals=8,step=.0001);self.scale_max=_dbl(1,decimals=8,step=.0001)
        self.scale_range=QComboBox();self.scale_range.addItem('Saturate','saturate');self.scale_range.addItem('Clip to range','clip')
        self.scale_shared=QCheckBox('Общая шкала всех z')
        for w in (QLabel('Цветовая шкала:'),self.scale_mode,QLabel('min'),self.scale_min,
                  QLabel('max'),self.scale_max,self.scale_range,self.scale_shared):scales.addWidget(w)
        scales.addStretch();lay.addLayout(scales)
        self.balance_label=QLabel('Результатов пока нет. Задайте исходные данные и нажмите «Рассчитать».')
        self.balance_label.setWordWrap(True);lay.addWidget(self.balance_label)
        note=QLabel('Матрица содержит контрольные ячейки для количественного анализа; подробное поле — визуализация. Баланс проверяется по матрице. '
                    'Представление применяется к матрице и окружным эпюрам; подробное поле всегда в кг/(м²·с). '
                    'Индексы при нулевом среднем не определены (пустые ячейки). '
                    'Полярная эпюра показывает угловую ИНТЕНСИВНОСТЬ, не физический контур факела. Контур форсунки — отдельная окружность.')
        note.setWordWrap(True);note.setStyleSheet('color:#525f71;');lay.addWidget(note)
        representations=QHBoxLayout();self.metric=QComboBox()
        for key,label in METRICS.items():self.metric.addItem(label,key)
        self.volume_interval=_dbl(180,.001,1e9,3,10.);self.volume_interval.setEnabled(False)
        self.compare_radius=QComboBox()
        for w in (QLabel('Величина:'),self.metric,QLabel('Интервал объёма, с:'),self.volume_interval,
                  QLabel('R для сравнения:'),self.compare_radius):representations.addWidget(w)
        representations.addStretch();lay.addLayout(representations)
        self.metric_equation=QLabel(EQUATIONS[self.metric.currentData()])
        self.metric_equation.setStyleSheet("color:#67458b;font-size:15px;")
        self.metric_equation.setToolTip("ṁ — расход контрольной ячейки [кг/с]; S — площадь [м²]; ρ — плотность [кг/м³]; t — интервал [с]. Средние взвешены по площади.")
        representations.insertWidget(2,self.metric_equation)
        axis_row=QHBoxLayout();self.x_axis=QComboBox();self.x_axis.addItem('Градусы: 0…360°','angle');self.x_axis.addItem('Окружные ячейки: 0…N','cells');axis_row.addWidget(QLabel('Ось X эпюр:'));axis_row.addWidget(self.x_axis)
        self.longitudinal_button=QPushButton('Продольный вид ЖТ');self.longitudinal_button.clicked.connect(self.open_longitudinal);axis_row.addWidget(self.longitudinal_button);axis_row.addStretch();lay.addLayout(axis_row)
        chart_controls=QHBoxLayout()
        self.plot_count=_integer(6,1,140);self.plot_columns=_integer(2,1,6)
        self.graph_selection=GraphSelection(self.plot_count);self.graph_selection.changed.connect(self._view_changed)
        self.plot_count.setEnabled(False);self.plot_columns.setEnabled(False)
        for label,widget in (("Графиков эпюр:",self.plot_count),("Колонок:",self.plot_columns)):
            chart_controls.addWidget(QLabel(label));chart_controls.addWidget(widget)
            widget.valueChanged.connect(self._view_changed)
        chart_controls.addWidget(self.graph_selection)
        self.fit_scale_button=QPushButton("Подогнать масштаб");self.fit_scale_button.setEnabled(False);self.fit_scale_button.clicked.connect(self._fit_plot_scales);chart_controls.addWidget(self.fit_scale_button)
        self.plot_count.setToolTip("Количество последовательных радиальных уровней, начиная с выбранного R; полный набор начинается с первого уровня.")
        reset=QPushButton("Исходный масштаб");self.reset_plot_button=reset;reset.setEnabled(False);reset.clicked.connect(self._reset_plot_zoom);chart_controls.addWidget(reset)
        self.axes_button=QPushButton("Оси…");self.axes_button.clicked.connect(lambda: self._open_axes_settings());chart_controls.addWidget(self.axes_button)
        chart_controls.addStretch()
        self.split_button=QToolButton();self.split_button.setCheckable(True)
        self.split_button.setText("◫");self.split_button.setToolTip("Split: показать / скрыть область сравнения")
        self.split_button.setToolButtonStyle(Qt.ToolButtonTextOnly);self.split_button.setFixedSize(32,28);self.split_button.setStyleSheet("font-size:18px;")
        self.split_button.setAccessibleName("Разделить рабочую область")
        self.split_button.toggled.connect(self._toggle_split);chart_controls.addWidget(self.split_button)
        lay.addLayout(chart_controls)
        self.metric.currentIndexChanged.connect(self._representation_changed)
        self.volume_interval.valueChanged.connect(self._representation_changed)
        plots=QTabWidget();self.plots=plots
        self.plot_stacks=[]
        self.figmap=Figure(figsize=(6,6),tight_layout=True)
        self.mapcanvas=ResultCanvas(self.figmap)
        stack=PlotStack();stack.addWidget(self.mapcanvas);self.plot_stacks.append(stack)
        plots.addTab(stack,'Матрица (консервативная)')
        self.figsmooth=Figure(figsize=(7,7),tight_layout=True)
        self.smoothcanvas=ResultCanvas(self.figsmooth)
        stack=PlotStack();stack.addWidget(self.smoothcanvas);self.plot_stacks.append(stack)
        field_page=QWidget();field_layout=QVBoxLayout(field_page);field_layout.setContentsMargins(0,0,0,0);field_layout.addWidget(stack,1)
        self.time_controls=QWidget();time_layout=QVBoxLayout(self.time_controls);time_layout.setContentsMargins(4,2,4,2);self.time_label=QLabel();self.time_label.setWordWrap(True);self.time_label.setMinimumHeight(self.time_label.fontMetrics().lineSpacing()*3+8);time_layout.addWidget(self.time_label);self.time_slider=QSlider(Qt.Horizontal);self.time_slider.setSingleStep(1);self.time_slider.setPageStep(1);time_layout.addWidget(self.time_slider);field_layout.addWidget(self.time_controls);self.time_controls.hide()
        self.time_slider.valueChanged.connect(self._schedule_time_field);self.time_slider.sliderReleased.connect(self._update_time_field)
        plots.addTab(field_page,'Подробное суммарное поле')
        self.figroses=Figure(figsize=(7,7),tight_layout=True)
        self.rosescanvas=ResultCanvas(self.figroses)
        stack=PlotStack();stack.addWidget(self.rosescanvas);self.plot_stacks.append(stack)
        plots.addTab(stack,'140 полярных эпюр на общей схеме')
        self.figcurves=Figure(figsize=(10,7),tight_layout=True)
        self.curvecanvas=ResultCanvas(self.figcurves)
        stack=PlotStack();stack.addWidget(self.curvecanvas);self.plot_stacks.append(stack)
        plots.addTab(stack,'Окружные эпюры радиальных уровней')
        plots.currentChanged.connect(self._plot_result)
        panels=QSplitter(Qt.Horizontal);self.result_panels=panels;panels.setChildrenCollapsible(False);panels.addWidget(plots)
        self.comparison=ComparisonPanel(self);panels.addWidget(self.comparison)
        self.comparison.hide();panels.setSizes([1400,0]);panels.setSizePolicy(QSizePolicy.Expanding,QSizePolicy.Expanding);lay.addWidget(panels,1)
        self.compare_radius.currentIndexChanged.connect(self._comparison_radius_changed)
        self.tabs.addTab(page,'04 · Поля и эпюры')
        self.graph_links=GraphLinks(self)

    def _fill_controls(self):
        restoring=self._restoring;self._restoring=True
        self.transport_mode_label.setText(TRANSPORT_MODES[TransportSettings.from_dict(self.cfg.fuel_transport).mode])
        for key,w in self.geom_widgets.items():
            value=getattr(self.cfg,key)
            if isinstance(w,QLineEdit):
                if key=='z_mm':w.setText('; '.join(f'{a:g}' for a in value))
                elif key=='radial_centers_mm':w.setText('' if value is None else '; '.join(f'{x:g}' for x in value))
            elif isinstance(w,QComboBox):w.setCurrentIndex(w.findData(value))
            else:w.setValue(value)
        self.x_axis.setCurrentIndex(self.x_axis.findData(self.cfg.display_x_axis))
        for key,w in self.model_widgets.items():
            val=getattr(self.cfg,key)
            if isinstance(w,QComboBox):w.setCurrentIndex(w.findData(val))
            else:w.setValue(val)
        self.duration_spin.setValue(self.cfg.duration_s)
        self.density_spin.setValue(self.cfg.density_kg_m3)
        self.scale_mode.setCurrentIndex(self.scale_mode.findData(self.cfg.color_scale_mode))
        self.scale_min.setValue(self.cfg.color_vmin);self.scale_max.setValue(self.cfg.color_vmax)
        self.scale_range.setCurrentIndex(self.scale_range.findData(self.cfg.color_out_of_range))
        self.scale_shared.setChecked(self.cfg.color_shared_z)
        for key,w in self.mode_widgets.items():
            value=getattr(self.cfg,key)
            if value is None and key in ('flow_pressure_kgf_cm2','sector_pressure_kgf_cm2'):
                value=getattr(self.nozzles[0],key)
            if isinstance(w,QComboBox):w.setCurrentIndex(w.findData(value))
            elif isinstance(w,QCheckBox):w.setChecked(value)
            else:w.setValue(value)
        self.catalog_editor.load(self.cfg.atomizer_types);self.preset_editor.load()
        self.geometry_editor.load(self.cfg,self.nozzles,force=True)
        active=bool(chamber(self.cfg))
        for key in ('r_in_mm','r_out_mm','z_mm'):
            self.geom_widgets[key].setEnabled(not active)
            self.geom_widgets[key].setToolTip('Задаётся в «Профили ЖТ: Sketcher и таблицы»' if active else '')
        selected=self.assign_type_selector.currentData()
        self.assign_type_selector.clear();self.assign_type_selector.addItem('Без типа распылителя','')
        for a in self.cfg.atomizer_types:self.assign_type_selector.addItem(a['name'],a['id'])
        self.assign_type_selector.setCurrentIndex(max(0,self.assign_type_selector.findData(selected)))
        self._restoring=restoring

    def _read_controls(self):
        cfg=deepcopy(self.cfg)
        for key,w in self.geom_widgets.items():
            if key in ('z_mm','radial_centers_mm'):
                text=w.text().strip().replace(',','.')
                if key=='radial_centers_mm' and not text:
                    setattr(cfg,key,None)
                else:
                    values=[float(x.strip()) for x in text.split(';') if x.strip()]
                    original=getattr(self.cfg,key)
                    if original is not None and values==[float(f'{x:g}') for x in original]:
                        values=list(original)
                    setattr(cfg,key,values)
            else:
                value=w.currentData() if isinstance(w,QComboBox) else w.value();original=getattr(self.cfg,key)
                if isinstance(w,QDoubleSpinBox) and value==round(original,w.decimals()):value=original
                setattr(cfg,key,value)
        for key,w in self.model_widgets.items():
            setattr(cfg,key,w.currentData() if isinstance(w,QComboBox) else w.value())
        for key,w in (('duration_s',self.duration_spin),('density_kg_m3',self.density_spin),
                      ('color_vmin',self.scale_min),('color_vmax',self.scale_max)):
            original=getattr(self.cfg,key);value=w.value()
            setattr(cfg,key,original if value==round(original,w.decimals()) else value)
        cfg.display_x_axis=self.x_axis.currentData()
        cfg.color_scale_mode=self.scale_mode.currentData()
        cfg.color_out_of_range=self.scale_range.currentData()
        cfg.color_shared_z=self.scale_shared.isChecked()
        for key,w in self.mode_widgets.items():
            value=w.currentData() if isinstance(w,QComboBox) else w.isChecked() if isinstance(w,QCheckBox) else w.value()
            original=getattr(self.cfg,key)
            if original is None and key in ('flow_pressure_kgf_cm2','sector_pressure_kgf_cm2'):
                if value==round(getattr(self.nozzles[0],key),w.decimals()):value=None
            elif isinstance(w,QDoubleSpinBox) and value==round(original,w.decimals()):value=original
            setattr(cfg,key,value)
        cfg.validate()
        return cfg

    def _fill_nozzles(self):
        tab=self.nozzle_table
        tab.blockSignals(True);tab.setRowCount(len(self.nozzles))
        unit=self.cfg.display_flow_unit
        tab.setHorizontalHeaderItem(INTCOL['flow_kg_h'],QTableWidgetItem('Gт, '+('кг/ч' if unit=='kg/h' else 'л/мин')))
        tab.setHorizontalHeaderItem(INTCOL['effective_flow'],QTableWidgetItem('Расход в режиме, '+('кг/ч' if unit=='kg/h' else 'л/мин')))
        for row,n in enumerate(self.nozzles):
            effective=effective_nozzle(n,self.cfg)
            if self.cfg.quasistationary:
                calculated='G(t): по шагам';label='Расход зависит от давления коллектора '+(n.manifold_id or 'по умолчанию')+'; значения по шагам — в отчёте 04.'
            else:
                try:
                    status=flow_status(n,self.cfg)
                    calculated=flow_from_kg_h(status.kg_s*3600,unit,self.cfg.density_kg_m3)
                    label=f'{status.source}; Δp={status.pressure_kgf_cm2:g}; '+('оценка' if status.estimated else 'измерено')
                    if status.notices:label+='; '+'; '.join(status.notices)
                except ValueError as exc:calculated='—';label=str(exc)
            for col,(key,_) in enumerate(FIELDS):
                if key=='effective_flow':v=calculated
                elif key=='theta_deg':v=float(from_global(n.theta_deg,self.cfg))
                elif key=='flow_status':v=label
                elif key=='flow_kg_h':v=flow_from_kg_h(n.flow_kg_h,unit,self.cfg.density_kg_m3)
                elif key=='manual_write':v=bool(n.geometry_overrides)
                elif key=='nozzle_kind':v=NOZZLE_KINDS[n.nozzle_kind]
                elif key=='row':v={'outer':'наружный','inner':'внутренний'}[n.row]
                elif key=='atomizer_type_id':v=next((a['name'] for a in self.cfg.atomizer_types if a['id']==n.atomizer_type_id),n.atomizer_type_id)
                else:v=getattr(effective if key in GEOMETRY_KEYS else n,key)
                if key in ('enabled','manual_write'):
                    it=QTableWidgetItem('');it.setFlags(Qt.ItemIsEnabled|Qt.ItemIsSelectable|Qt.ItemIsUserCheckable)
                    it.setCheckState(Qt.Checked if v else Qt.Unchecked)
                    if key=='manual_write':it.setToolTip('Разрешить ручную геометрию; снятие галочки возвращает текущий шаблон.')
                else:
                    val='' if v is None else f'{v:.12g}' if isinstance(v,float) else str(v)
                    it=QTableWidgetItem(val)
                    if key in READONLY or key in GEOMETRY_KEYS and not n.geometry_overrides:
                        it.setFlags(Qt.ItemIsEnabled|Qt.ItemIsSelectable)
                    it.setToolTip(val)
                    if key=='flow_status':it.setToolTip(label)
                    if key in GEOMETRY_KEYS:
                        it.setToolTip('Индивидуальный override' if key in n.geometry_overrides else 'Из шаблона: '+NOZZLE_KINDS[n.nozzle_kind])
                if key=='atomizer_type_id':
                    it.setData(Qt.UserRole,n.atomizer_type_id);it.setData(Qt.UserRole+1,str(v))
                tab.setItem(row,col,it)
        tab.blockSignals(False)
        notices=set();invalid=0
        for n in self.nozzles:
            if self.cfg.quasistationary:break
            if not n.enabled:continue
            try:
                status=flow_status(n,self.cfg)
                if status.kg_s>0:notices.update(status.notices)
            except ValueError:invalid+=1
        text='Legacy: давление и η не пересчитываются.' if self.cfg.pressure_mode=='legacy' else f'Целевой перепад: {self.cfg.target_pressure_kgf_cm2:g} кгс/см².'
        if self.cfg.quasistationary:text='Квазистационарный режим: используются перепады коллекторов на каждом шаге; целевой Δp выше относится к стационарному расчёту.'
        if invalid:text+=f' Требуют настройки: {invalid} форсунок.'
        if notices:text+=' '+ '; '.join(sorted(notices))+'.'
        self.flow_notice.setText(text)
        if self.sector_id is None:self.sector_id=1
        self._show_sectors(self.sector_id)

    def _read_nozzles(self,cfg=None):
        cfg=cfg or self.cfg
        nozzles=[];angle_edits={}
        for i,n in enumerate(self.nozzles):
            raw=vars(n).copy();raw['geometry_overrides']=list(n.geometry_overrides)
            effective=effective_nozzle(n,self.cfg);geometry_changes={}
            manual=self.nozzle_table.item(i,INTCOL['manual_write']).checkState()==Qt.Checked
            if manual and not n.geometry_overrides:
                values={key:getattr(effective,key) for key in GEOMETRY_KEYS}
                raw.update({key:value for key,value in values.items() if key!='r_nozzle_mm'})
                raw['rc0_mm']=values['r_nozzle_mm']-values['h0_mm']/2
                raw['geometry_overrides']=list(GEOMETRY_KEYS)
            elif not manual:raw['geometry_overrides']=[]
            for key in ('flow_pressure_kgf_cm2','sector_pressure_kgf_cm2'):
                if getattr(cfg,key) is not None:raw[key]=getattr(cfg,key)
            for col,(key,_) in enumerate(FIELDS):
                item=self.nozzle_table.item(i,col)
                if key in READONLY:continue
                if key=='manual_write':continue
                if key=='enabled':raw[key]=item.checkState()==Qt.Checked;continue
                val=item.text().strip()
                if key in NUMERICS:
                    original=getattr(effective if key in GEOMETRY_KEYS else n,key)
                    if key=='theta_deg':original=float(from_global(original,self.cfg))
                    if key=='flow_kg_h':original=flow_from_kg_h(original,self.cfg.display_flow_unit,self.cfg.density_kg_m3)
                    if key=='capture_efficiency' and val in ('','?','unknown'):
                        raw[key]=None;continue
                    displayed='' if original is None else f'{original:.12g}'
                    unchanged=val==displayed
                    try:value=original if unchanged else _number(val)
                    except ValueError:raise ValueError(f'Строка {i+1}, {FIELDS[col][1]}: неверное число «{val}».')
                    if not np.isfinite(value):raise ValueError(f'Строка {i+1}: число должно быть конечным.')
                    if key in GEOMETRY_KEYS:
                        if manual and not unchanged:geometry_changes[key]=value
                    elif key=='flow_kg_h':
                        raw[key]=n.flow_kg_h if unchanged else flow_to_kg_h(value,self.cfg.display_flow_unit,self.cfg.density_kg_m3)
                    elif key=='theta_deg':raw[key]=n.theta_deg if unchanged else float(to_global(value,self.cfg))
                    else:raw[key]=value
                    if key=='theta_deg' and raw[key]!=n.theta_deg:angle_edits[n.id]=raw[key]
                else:
                    if key=='nozzle_kind':
                        raw[key]=next((k for k,label in NOZZLE_KINDS.items() if val.lower() in (k,label.lower())),val.lower())
                    elif key=='atomizer_type_id':
                        old_name=next((a['name'] for a in self.cfg.atomizer_types if a['id']==n.atomizer_type_id),n.atomizer_type_id)
                        matches=[a['id'] for a in cfg.atomizer_types if val in (a['id'],a['name'])]
                        if val==item.data(Qt.UserRole+1):raw[key]=item.data(Qt.UserRole)
                        elif val==old_name:raw[key]=n.atomizer_type_id
                        elif len(matches)>1:raise ValueError('Неоднозначное имя распылителя; выберите его из списка.')
                        else:raw[key]=matches[0] if matches else val
                    elif key=='manifold_id':raw[key]=val.strip()
                    else:raw[key]=val.lower()
            if geometry_changes:
                values={k:getattr(effective,k) for k in GEOMETRY_KEYS};values.update(geometry_changes)
                raw.update({k:v for k,v in values.items() if k!='r_nozzle_mm'})
                raw['rc0_mm']=values['r_nozzle_mm']-values['h0_mm']/2
                raw['geometry_overrides']=sorted(set(raw['geometry_overrides'])|set(geometry_changes))
            nozzles.append(Nozzle(**raw))
        validate_nozzles(nozzles,cfg)
        if cfg.placement_mode=='auto_follow' and angle_edits:
            for new,old in zip(nozzles,self.nozzles):new.theta_deg=old.theta_deg
            for ident,theta in sorted(angle_edits.items()):follow_row(nozzles,ident,theta)
        if angle_edits and cfg.geometry:cfg.geometry=sync_groups(cfg.geometry,nozzles)
        return nozzles

    def _sync_nozzles(self):
        self.nozzles=self._read_nozzles()

    def _catalog_changed(self,catalog):
        self.cfg.atomizer_types=deepcopy(catalog)
        self._record_edit(label='Справочник распылителей');self._fill_controls()

    def _presets_changed(self,presets):
        self._sync_nozzles()
        self.cfg.nozzle_presets=deepcopy(presets)
        self._fill_nozzles()
        self._record_edit(label='Шаблон форсунки');self._fill_controls()

    def _toggle_flow_unit(self,column):
        if column not in (INTCOL['flow_kg_h'],INTCOL['effective_flow']):return
        menu=QMenu(self.nozzle_table);self.flow_unit_menu=menu
        for unit,label in (('kg/h','кг/ч'),('l/min','л/мин')):
            action=menu.addAction(label);action.setCheckable(True);action.setChecked(self.cfg.display_flow_unit==unit)
            action.triggered.connect(lambda checked,u=unit:self._set_flow_unit(u))
        header=self.nozzle_table.horizontalHeader()
        menu.popup(header.mapToGlobal(QPoint(header.sectionViewportPosition(column),header.height())))

    def _set_flow_unit(self,unit):
        if hasattr(self,'flow_unit_menu'):self.flow_unit_menu.close()
        if unit==self.cfg.display_flow_unit:return
        self._record_edit()
        self.cfg.display_flow_unit=unit
        self._fill_nozzles();self._record_edit(label='Единицы расхода')

    def assign_atomizer(self):
        ids=self._selected_ids()
        if not ids:self.statusBar().showMessage('Выделите целевые форсунки.',5000);return
        ident=self.assign_type_selector.currentData() or ''
        for n in self.nozzles:
            if n.id in ids:n.atomizer_type_id=ident
        self._fill_nozzles();self._record_edit(label='Назначение распылителя')

    def assign_nozzle_kind(self):
        ids=self._selected_ids()
        for n in self.nozzles:
            if n.id in ids:n.nozzle_kind=self.assign_kind_selector.currentData()
        self._fill_nozzles();self._record_edit(label='Тип форсунки')

    def reset_overrides(self):
        ids=self._selected_ids()
        for n in self.nozzles:
            if n.id in ids:n.geometry_overrides=[]
        self._fill_nozzles();self._record_edit(label='Сброс overrides')

    def _connect_edits(self):
        for table in (self.nozzle_table,self.sector_table):
            table.itemChanged.connect(self._record_edit)
            table.batchFinished.connect(lambda:self._record_edit(label='Вставка ячеек'))
        widgets=list(self.geom_widgets.values())+list(self.model_widgets.values())+list(self.mode_widgets.values())+[
            self.duration_spin,self.density_spin,self.scale_mode,self.scale_min,
            self.scale_max,self.scale_range,self.scale_shared,self.x_axis]
        for w in widgets:
            if isinstance(w,QComboBox):w.currentIndexChanged.connect(self._control_edited)
            elif isinstance(w,QCheckBox):w.toggled.connect(self._control_edited)
            else:w.editingFinished.connect(self._control_edited)

    def _control_edited(self,*args):self._record_edit(control_only=True)

    def _record_edit(self,*args,label='Изменение проекта',control_only=False):
        if self._restoring or self.nozzle_table.in_batch or self.sector_table.in_batch:return
        if self.sector_table.batch_invalid:
            self.sector_table.batch_invalid=False
            self._restore_edit(self._edit_state)
            self.statusBar().showMessage('Вставка отменена: объёмы должны быть конечными неотрицательными числами.',12000)
            return
        try:
            cfg=self._read_controls()
            if control_only and cfg==self.cfg:return
            nozzles=self._read_nozzles(cfg)
            if cfg.geometry:
                if chamber(cfg):cfg.geometry,nozzles=sync_anchors(cfg.geometry,nozzles,cfg,self.nozzles)
                previous_angles={n['id']:n['theta_deg'] for n in self._edit_state['nozzles']}
                if any(n.theta_deg!=previous_angles.get(n.id) for n in nozzles):cfg.geometry=sync_groups(cfg.geometry,nozzles)
            state=project_to_json(cfg,nozzles)
        except (ValueError,TypeError) as exc:
            self._restore_edit(self._edit_state)
            self.statusBar().showMessage(str(exc),12000)
            return
        if state==self._edit_state:return
        before=self._edit_state
        physical_before=deepcopy(before);physical_after=deepcopy(state)
        for data in (physical_before,physical_after):
            for key in list(data['settings']):
                if key.startswith(('color_','display_')):del data['settings'][key]
        self.cfg=cfg;self.nozzles=nozzles
        self.catalog_editor.refresh_density()
        self._edit_state=deepcopy(state)
        self._fill_nozzles()
        self.undo_stack.push(ProjectEdit(self,before,state,label))
        self.geometry_editor.load(self.cfg,self.nozzles)
        if physical_before!=physical_after:self._invalidate_results()
        else:
            if any(before['settings'].get(k)!=state['settings'].get(k) for k in state['settings'] if k.startswith('color_')):self._invalidate_render()
            self._plot_result();self.comparison.draw()
        self._queue_preview()

    def _restore_edit(self,state):
        # Undo preserves results for display-only changes, but invalidates physics edits.
        previous=deepcopy(self._edit_state)
        physical=lambda data: {**data,'settings':{k:v for k,v in data['settings'].items()
                                                 if not k.startswith(('color_','display_'))}}
        self._restoring=True
        try:
            self.cfg,self.nozzles=project_from_json(deepcopy(state))
            self._fill_controls();self._fill_nozzles()
            self._edit_state=deepcopy(state)
        finally:self._restoring=False
        if physical(previous)!=physical(state):self._invalidate_results()
        else:
            if any(previous['settings'].get(k)!=state['settings'].get(k) for k in state['settings'] if k.startswith('color_')):self._invalidate_render()
            self._plot_result();self.comparison.draw()
        self._queue_preview()

    def _invalidate_render(self):
        if hasattr(self,'graph_links'):self.graph_links.reset()
        self._render_cache.clear()
        names=[('figmap','mapcanvas'),('figsmooth','smoothcanvas'),('figroses','rosescanvas'),('figcurves','curvecanvas')]
        for index,stack in enumerate(self.plot_stacks):
            # Keep one empty canvas so a stale field never remains visible.
            while stack.count()>1:
                widget=stack.widget(stack.count()-1)
                stack.removeWidget(widget);widget.deleteLater()
            if stack.count():
                canvas=stack.widget(0);canvas.figure.clear()
                if canvas.isVisible():canvas.draw_idle()
                figure_name,canvas_name=names[index];setattr(self,figure_name,canvas.figure);setattr(self,canvas_name,canvas)

    def _invalidate_results(self):
        self.quasi_result=None;self.time_timer.stop();self.time_controls.hide()
        self._revision+=1
        self.results=[];self.smooth_fields=[]
        self._invalidate_render()
        self.plane_select.clear()
        self.compare_radius.clear()
        self.comparison.sync_from_primary()
        self.balance_label.setText('Исходные данные изменены. Нужен новый расчёт (F5).')

    def _queue_preview(self):
        if hasattr(self,'previewcanvas') and self.tabs.currentIndex()==1:self.preview_timer.start(120)

    def _main_tab_changed(self,index):
        # Hidden previews used to run after edits on every tab, including while
        # the result slider was painting. Build the preview when it is needed.
        if index==1:self._queue_preview()
        else:self.preview_timer.stop()
        if index!=3:self.time_timer.stop()
        elif self.results:self._plot_result()

    def _draw_single_preview(self):
        if self.sector_id is None:return
        try:
            cfg=self._read_controls()
            key=(self.sector_id,self._revision,cfg.color_scale_mode,cfg.color_vmin,cfg.color_vmax,cfg.color_out_of_range)
            if key==self._preview_key:return
            n=effective_nozzle(next(n for n in self.nozzles if n.id==self.sector_id),cfg)
            # Only one source is evaluated, at z=0, without a 140-source calculation.
            x,y=polar_epure_coordinates(n,n.z_tip_mm,cfg,8,n_angles=288)
            theta=np.deg2rad(n.theta_deg)
            x0=n.radius_mm*np.cos(theta);y0=n.radius_mm*np.sin(theta)
            from .core import geometry_at, _kernel_outer_limit
            rc,h=geometry_at(n,n.z_tip_mm,cfg);span=max(_kernel_outer_limit(rc,h,cfg.radial_model)*1.2,1)
            field=evaluate_continuous_field([n],cfg,n.z_tip_mm,resolution=SINGLE_NOZZLE_PREVIEW_RESOLUTION,
                bounds_xy=(x0-span,x0+span,y0-span,y0+span),mask_track=False)
            self.figpreview.clear()
            ax=self.figpreview.add_subplot(121,projection='polar')
            ax.plot(np.arctan2(y-y0,x-x0),np.hypot(x-x0,y-y0))
            ax.set_title(f'№{n.id}: эпюра, z лок.=0; z={n.z_tip_mm:g} мм',fontsize=9)
            ax=self.figpreview.add_subplot(122)
            data,norm,cmap=color_scale(field.masked_flux_kg_m2_s,cfg)
            # Interpolation is display-only. Strict clipping must not blend hidden pixels.
            interpolation='nearest' if cfg.color_scale_mode=='manual' and cfg.color_out_of_range=='clip' else 'bilinear'
            im=ax.imshow(data,origin='lower',extent=[field.x_mm[0],field.x_mm[-1],field.y_mm[0],field.y_mm[-1]],cmap=cmap,norm=norm,
                         interpolation=interpolation)
            ax.set(xlim=(x0-span,x0+span),ylim=(y0-span,y0+span),aspect='equal')
            ax.set_title(f'Поле одной форсунки: z={n.z_tip_mm:g} мм, z лок.=0',fontsize=9)
            ax.set_xlabel('X, мм');ax.set_ylabel('Y, мм')
            self.figpreview.colorbar(im,ax=ax,shrink=.8,label='кг/(м²·с)')
            self.previewcanvas.draw_idle()
            self._preview_key=key
        except (ValueError,TypeError) as exc:
            self.statusBar().showMessage(f'Превью: {exc}',5000)

    def _sector_selection(self,*args):
        self.preview_timer.stop()
        self.selection_timer.start(120)

    def _apply_sector_selection(self):
        idx=self.nozzle_table.currentIndex()
        if idx.isValid() and idx.row()<len(self.nozzles):
            new_id=self.nozzles[idx.row()].id
            if new_id!=self.sector_id:self._show_sectors(new_id)

    def _show_sectors(self,nozzle_id):
        self.sector_id=nozzle_id
        n=next(n for n in self.nozzles if n.id==nozzle_id)
        self.sector_table.blockSignals(True)
        for k,val in enumerate(n.sector_ml):
            self.sector_table.setItem(k,1,QTableWidgetItem(f'{val:.10g}'))
        self.sector_table.blockSignals(False)
        self._update_sector_label()
        self._queue_preview()

    def _update_sector_label(self):
        n=next(n for n in self.nozzles if n.id==self.sector_id)
        self.sector_label.setText(f'Форсунка №{n.id}, {dict(outer='наружный',inner='внутренний')[n.row]}: 36 секторных объёмов [мл]; '
                                  f'сумма = {sum(n.sector_ml):.4g} мл; {n.sector_source}')
        try:
            report=capture_report(n,self.cfg)
            self.capture_label.setText(f'Полнота сбора: {report.fraction*100:.3f}%\n'
                f'Отклонение от Gт: {report.deviation_percent:+.3f}%\n'
                f'Недостающий объём: {report.missing_ml:+.3f} мл\n'
                f'Ожидается: {report.expected_ml:.3f} мл\n'
                f'Собранный расход, приведённый к давлению Gт: {report.collected_at_measurement_kg_h:.4g} кг/ч')
        except ValueError as exc:self.capture_label.setText('Полнота сбора не определена: '+str(exc))
        try:
            status=flow_status(n,self.cfg)
            self.nozzle_status.setText(f'{status.source}; Δp={status.pressure_kgf_cm2:g} кгс/см²; '
                +('расчёт/оценка' if status.estimated else 'измерение')+'\n'+'; '.join(status.notices))
        except ValueError as exc:self.nozzle_status.setText(str(exc))

    def _sector_changed(self,item):
        if item.column()!=1 or self.sector_id is None:return
        n=next(n for n in self.nozzles if n.id==self.sector_id)
        try:
            value=_number(item.text())
            if value < 0 or not np.isfinite(value):raise ValueError('Неотрицательное число обязательно')
        except ValueError:
            if self.sector_table.in_batch:self.sector_table.batch_invalid=True
            self.sector_table.blockSignals(True)
            item.setText(f'{n.sector_ml[item.row()]:.10g}')
            self.sector_table.blockSignals(False)
            self.statusBar().showMessage('Объём сектора: введите неотрицательное число (мл). Предыдущее значение восстановлено.',12000)
            return
        if value==n.sector_ml[item.row()]:return
        n.sector_ml[item.row()]=value
        n.sector_source='ручное редактирование'
        self._update_sector_label()
        self._queue_preview()

    def _selected_ids(self):
        return sorted({self.nozzles[ind.row()].id for ind in self.nozzle_table.selectedIndexes()})

    def _error(self,description):
        QMessageBox.critical(self,'Ошибка',str(description))

    def update_positions(self):
        try:
            cfg=self._read_controls()
            self._sync_nozzles()
            apply_default_positions(self.nozzles,cfg)
            self._fill_nozzles()
            self._record_edit(label='Позиции двух рядов')
            self.statusBar().showMessage('Позиции форсунок перерасставлены. Флаги включения, расходы и секторные эпюры сохранены.',12000)
        except Exception as exc:self._error(exc)

    def import_sectors(self):
        path,_=QFileDialog.getOpenFileName(self,'Таблица секторных объёмов','','Excel (*.xlsx);;CSV (*.csv)')
        if not path:return
        try:
            self._sync_nozzles()
            data,sheet=import_sector_file(path)
            valid={n.id:n for n in self.nozzles}
            count=0;unknown=[]
            for nid,parts in data.items():
                if nid in valid:
                    n=valid[nid];n.sector_ml=[float(v) for v in parts]
                    n.sector_source=f'{Path(path).name}:{sheet}'
                    count+=1
                else:unknown.append(nid)
            self._fill_nozzles()
            self._record_edit(label='Импорт секторов')
            QMessageBox.information(self,'Импорт секторных объёмов',
                f'Загружены эпюры {count} форсунок, лист/источник: {sheet}. '
                f'Расходы в режиме manual НЕ изменены. '
                + (f'Не сопоставлены ID: {unknown[:15]}.' if unknown else ''))
        except Exception as exc:self._error(exc)

    def export_sectors(self):
        path,_=QFileDialog.getSaveFileName(self,'Экспорт эпюр','sector_data.csv','CSV (*.csv)')
        if not path:return
        try:self._sync_nozzles();write_sector_csv(path,self.nozzles)
        except Exception as exc:self._error(exc)

    def copy_sectors(self):
        ids=self._selected_ids()
        if not ids:
            QMessageBox.information(self,'Выделение','Выделите в верхней таблице одну или несколько целевых строк.')
            return
        source,ok=QInputDialog.getInt(self,'Копирование эпюры','ID форсунки-источника:',
                                      self.sector_id or 1,1,140)
        if not ok:return
        try:
            self._sync_nozzles()
            src=next(n for n in self.nozzles if n.id==source)
            for idx in ids:
                n=next(n for n in self.nozzles if n.id==idx)
                n.sector_ml=src.sector_ml.copy()
                n.sector_source=f'копия №{source}'
            self._fill_nozzles()
            self._record_edit(label='Копирование эпюры')
            self.statusBar().showMessage(f'Секторная эпюра №{source} назначена {len(ids)} позициям.',12000)
        except Exception as exc:self._error(exc)

    def sum_sectors_to_flow(self):
        ids=self._selected_ids()
        if not ids:
            QMessageBox.information(self,'Выделение','Выделите целевые позиции в верхней таблице. Расходы остальных не меняются.')
            return
        try:
            self._sync_nozzles()
            changed=0
            for id in ids:
                n=next(n for n in self.nozzles if n.id==id)
                if n.sector_source=='шаблон: равномерно':continue
                if sum(n.sector_ml)<=0:continue
                n.flow_mode='sectors'
                changed+=1
            self._fill_nozzles()
            self._record_edit(label='Расход из объёмов')
            self.statusBar().showMessage(f'Для {changed} форсунок задан расчёт расхода из суммы объёмов.',12000)
        except Exception as exc:self._error(exc)

    def assign_demo_flow(self):
        reply=QMessageBox.question(self,'Демонстрационный расход',
            'Установить всем 140 форсункам 10 кг/ч (только для проверки работы программы)? '
            'Измеренные расходы, если уже введены, будут заменены.',
            QMessageBox.Yes|QMessageBox.No)
        if reply!=QMessageBox.Yes:return
        try:
            self._sync_nozzles()
            for n in self.nozzles:
                n.flow_kg_h=10.0;n.flow_mode='manual'
            self._fill_nozzles()
            self._record_edit(label='Демонстрационный расход')
            self.statusBar().showMessage('ДЕМО расход назначен всем форсункам: 10 кг/ч. Это НЕ экспериментальные данные.',12000)
        except Exception as exc:self._error(exc)

    def _prepare(self):
        cfg=self._read_controls()
        nozzles=self._read_nozzles(cfg)
        validate_nozzles(nozzles,cfg)
        if cfg.quasistationary:
            QuasiSettings.from_dict(cfg.quasi_settings).validate(cfg,nozzles)
            return cfg,nozzles
        flowing=[n for n in nozzles if n.enabled and source_flow_kg_s(n,cfg)>0]
        if not flowing:
            raise ValueError('Нет ни одной установленной форсунки с положительным расходом. '
                             'Введите измеренный расход или задайте положительную оценку по секторам/характеристике.')
        return cfg,nozzles

    def run_calc(self):
        if self.thread is not None:return
        self._record_edit()
        try:
            cfg,nozzles=self._prepare()
            # Avoid live table mutations during calculation; immutable state snapshot.
            cfg,nozzles=project_from_json(project_to_json(cfg,nozzles))
            self.cfg=cfg;self.nozzles=nozzles
        except Exception as exc:
            self._error(exc);return
        self._calc_revision=self._revision
        self.progress.setRange(0,len(cfg.z_mm))
        self.progress.setValue(0)
        self.calc_btn.setEnabled(False);self.cancel_calc_btn.setEnabled(True)
        self.thread=QThread(self)
        self.worker=CalcWorker(cfg,nozzles)
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.progress.connect(self._progress)
        self.worker.completed.connect(self._done)
        self.worker.failed.connect(self._failed)
        self.worker.finished.connect(self.thread.quit)
        self.worker.cancelled.connect(lambda:self.statusBar().showMessage("Расчёт отменён; частичные поля не применены.",10000))
        self.thread.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self._cleanup_thread)
        self.thread.start()
        self.tabs.setCurrentIndex(3)

    def _progress(self,message):
        self.statusBar().showMessage(message)
        self.progress.setValue(min(self.progress.maximum(),self.progress.value()+1))

    def _done(self,planes):
        if hasattr(self,'_calc_revision') and self._calc_revision!=self._revision:
            self.statusBar().showMessage('Данные изменены во время расчёта. Повторите F5.',12000)
            return
        if hasattr(self,'_invalidate_render'):self._invalidate_render()
        self.results=[v[0] for v in planes]
        self.smooth_fields=[v[1] for v in planes]
        self.quasi_result=getattr(planes,'quasi_result',None)
        if hasattr(self,'time_controls'):
            self.time_controls.setVisible(self.quasi_result is not None);self.time_slider.blockSignals(True);self.time_slider.setRange(0,self.quasi_result.settings.n_steps-1 if self.quasi_result else 0);self.time_slider.setValue(0);self.time_slider.blockSignals(False)
        self.progress.setValue(len(planes))
        # The worker emits (PlaneResult, ContinuousField) pairs.
        # `self.results` already contains the unwrapped PlaneResult objects.
        # Keep signals blocked through the initial selection to avoid drawing
        # all four plots twice (once from currentIndexChanged, once below).
        self.plane_select.blockSignals(True)
        try:
            self.plane_select.clear()
            for idx, p in enumerate(self.results):
                self.plane_select.addItem(f'№{idx+1}: z={p.z_mm:g} мм', idx)
            self.plane_select.setCurrentIndex(0)
        finally:
            self.plane_select.blockSignals(False)
        self._plot_result()
        count=sum((self.quasi_result.flows>0).any(axis=0)) if self.quasi_result else sum(1 for n in self.nozzles if n.enabled and source_flow_kg_s(n,self.cfg)>0)
        self.statusBar().showMessage(f'Готово: {len(self.results)} '+('сечение' if len(self.results)%10==1 and len(self.results)%100!=11 else 'сечения' if len(self.results)%10 in (2,3,4) and len(self.results)%100 not in (12,13,14) else 'сечений')+f', {count} форсунок с расходом.',20000)

    def _failed(self,message):
        self._error('Расчёт не выполнен:\n'+message[-2200:])
        self.statusBar().showMessage('Ошибка расчёта.',10000)

    def _cleanup_thread(self):
        self.thread.deleteLater()
        self.thread=None
        self.worker=None
        self.calc_btn.setEnabled(True);self.cancel_calc_btn.setEnabled(False)

    @staticmethod
    def _balance_text(p,i):
        total=p.injected_kg_s*3600
        if getattr(p,'transport_notice',''):
            if p.is_composite:
                return (f'<b>Квазистационарное сечение {i+1}, z={p.z_mm:g} мм.</b> '
                    'Накопленная матрица полного потока (взвесь + плёнка); столбцы сняты в разное время. '
                    'Единый мгновенный баланс для неё не определён. Поле — мгновенный снимок. '
                    'Плёнка включена в граничные ячейки; это не концентрация взвеси.')
            phases=[float(np.sum(getattr(p,key)))*3600 for key in
                ('suspended_cell_mass_kg_s','film_inner_cell_mass_kg_s','film_outer_cell_mass_kg_s')]
            momentum=getattr(p,'transport_momentum',{}) or {}
            return (f'<b>Сечение {i+1}, z={p.z_mm:g} мм · перенос внутри ЖТ.</b> '
                f'Подано {total:.6g} кг/ч; взвесь {phases[0]:.6g}; плёнка внутренняя {phases[1]:.6g}; наружная {phases[2]:.6g} кг/ч. '
                f'Невязка массы {100*p.rel_balance_error:.3g}%; импульса {momentum.get("relative_residual",0):.3g}. '
                'Матрица и поле показывают полный поток, включая плёнку в граничных ячейках. '
                f'Максимальное время переноса {p.max_travel_time_s:.5g} с. Параметризованная модель; см. инструкцию v0.7.')
        if p.is_composite:
            text=f'<b>Квазистационарное сечение {i+1}, z={p.z_mm:g} мм.</b> Накопленная матрица: один угловой столбец на временной шаг. Эпюры — из этой матрицы. Поле ниже — мгновенный снимок; единый мгновенный баланс накопленной матрицы не определён.'
        else:text=(
            f'<b>Сечение {i+1}, z={p.z_mm:g} мм.</b> '
            f'Через сопла перед сечением {total:.5g} кг/ч; в тракте {p.inside_kg_s*3600:.5g} кг/ч '
            f'({100*p.inside_kg_s/max(p.injected_kg_s,1e-15):.2f}%); '
            f'за внутренней границей {p.inner_kg_s*3600:.5g} кг/ч; '
            f'за наружной {p.outer_kg_s*3600:.5g} кг/ч; '
            f'wall-bound {p.wall_bound_kg_s*3600:.5g} кг/ч; '
            f'невязка баланса {100*p.rel_balance_error:.3g}%.'
        )
        if p.geometry_notice:text+='<br>wall-bound: неразрешённое распределение вне стенок; перенос и плёнка не моделируются.'
        return text

    def _reserve_balance_height(self):
        # Different line wrapping must not resize and redraw cached Matplotlib
        # canvases every time a short (e.g. empty inlet) balance is selected.
        if not self.results:return
        width=max(1,self.balance_label.contentsRect().width()-2*self.balance_label.margin())
        key=(id(self.results),width,self.balance_label.font().toString())
        if key==getattr(self,'_balance_layout_key',None):return
        self._balance_layout_key=key
        document=QTextDocument();document.setDocumentMargin(0);document.setDefaultFont(self.balance_label.font());document.setTextWidth(width)
        height=0.
        for i,p in enumerate(self.results):
            document.setHtml(self._balance_text(p,i));height=max(height,document.size().height())
        self.balance_label.setMinimumHeight(int(np.ceil(height))+4)

    def resizeEvent(self,event):
        super().resizeEvent(event)
        if hasattr(self,'balance_label'):QTimer.singleShot(0,self._reserve_balance_height)

    def _plot_result(self,*args):
        for widget in (self.plot_count,self.plot_columns,self.reset_plot_button,self.axes_button):widget.setEnabled(self.plots.currentIndex()==3)
        if not self.results:return
        i=max(0,self.plane_select.currentIndex());p=self.results[i]
        self._reserve_balance_height();self.balance_label.setText(self._balance_text(p,i))
        if p.geometry_notice or getattr(p,'transport_notice',''):
            self.balance_label.setToolTip(getattr(p,'transport_notice','') or p.geometry_notice)
        else:self.balance_label.setToolTip('')
        self.plot_count.blockSignals(True);self.plot_count.setMaximum(len(p.radial_centers_mm));self.plot_count.blockSignals(False)
        previous=self.compare_radius.currentData()
        self.compare_radius.blockSignals(True);self.compare_radius.clear()
        for r in p.radial_centers_mm:self.compare_radius.addItem(f'{r:g} мм',float(r))
        k=matching_index(p.radial_centers_mm,previous) if previous is not None else None
        if k is not None:self.compare_radius.setCurrentIndex(k)
        self.compare_radius.blockSignals(False)
        total=len(p.radial_centers_mm);start=0 if self.plot_count.value()==total else max(0,self.compare_radius.currentIndex())
        self.graph_selection.configure([(float(r),f"Уровень {k+1}: R={r:g} мм") for k,r in enumerate(p.radial_centers_mm)],
            [float(p.radial_centers_mm[(start+k)%total]) for k in range(total)],identity=tuple(p.radial_centers_mm))
        self.graph_selection.setEnabled(self.plots.currentIndex()==3)
        self._render_view(self.plots.currentIndex(),i)
        self.comparison.sync_from_primary()

    def _toggle_split(self,enabled):
        self.fit_scale_button.setEnabled(bool(enabled and self.results and (self.comparison.planes or self.comparison.tables) and self.comparison.view.currentIndex()==0))
        self.comparison.setVisible(enabled)
        if enabled:
            self.result_panels.setSizes([max(1,self.result_panels.width()*3//5),max(1,self.result_panels.width()*2//5)])
            self.comparison.sync_from_primary()
        if hasattr(self,"graph_links"):QTimer.singleShot(0,self.graph_links.refresh)

    def _fit_plot_scales(self):
        if not self.results or not self.split_button.isChecked() or self.comparison.view.currentIndex()!=0:return
        self.plots.setCurrentIndex(3)
        self.result_panels.setSizes([self.result_panels.width()//2]*2)
        QApplication.processEvents()
        primary=self.curvecanvas;other=self.comparison.canvas
        axes=list(primary.figure.axes)+list(other.figure.axes)
        if not axes:return
        for canvas in (primary,other):canvas.draw()
        width=min(ax.bbox.width for ax in axes);height=min(ax.bbox.height for ax in axes)
        for ax in axes:
            bbox=ax.get_position();fw,fh=ax.figure.bbox.width,ax.figure.bbox.height
            ax.set_position([bbox.x0+(bbox.width-width/fw)/2,bbox.y0+(bbox.height-height/fh)/2,width/fw,height/fh])
        groups={}
        for ax in axes:
            if hasattr(ax,'_compare_y'):groups.setdefault((getattr(ax,'_compare_x',None),ax._compare_y),[]).append(ax)
        for (xunit,yunit),items in groups.items():
            if xunit is None or len(items)<2:continue
            xmin=min(ax.get_xlim()[0] for ax in items);xmax=max(ax.get_xlim()[1] for ax in items)
            ymin=min(ax.get_ylim()[0] for ax in items);ymax=max(ax.get_ylim()[1] for ax in items)
            for ax in items:ax.set_xlim(xmin,xmax);ax.set_ylim(ymin,ymax)
        # Keep explicit equal-size positions through subsequent paints.
        for canvas in (primary,other):canvas.figure.set_layout_engine(None);canvas.draw_idle()
        self.statusBar().showMessage('Размеры графиков выровнены; совместимые оси получили общий диапазон.',10000)

    def _reset_plot_zoom(self):
        canvas=self.plot_stacks[self.plots.currentIndex()].currentWidget()
        if hasattr(canvas,'interaction'):canvas.interaction.reset_zoom()

    def _comparison_radius_changed(self,*args):
        self._plot_result()

    def _representation_changed(self,*args):
        self.volume_interval.setEnabled(self.metric.currentData()=='volume')
        self.metric_equation.setText(EQUATIONS[self.metric.currentData()])
        self._plot_result();self.comparison.draw()

    def _render_view(self,view,i):
        metric=self.metric.currentData();interval=self.volume_interval.value() if metric=='volume' else None
        marker=(self.draw_outlines.isChecked(),self.draw_labels.isChecked())
        if view==0:options=marker+(metric,interval)
        elif view==1:options=marker+(self.draw_rose_over_field.isChecked(),self.rose_scale.value() if self.draw_rose_over_field.isChecked() else None)
        elif view==2:options=marker+(self.rose_scale.value(),)
        else:options=(metric,interval,self.compare_radius.currentData(),self.plot_count.value(),self.plot_columns.value(),tuple(self.graph_selection.selected),self.x_axis.currentData())
        key=(i,view)+options
        fig_name,canvas_name=[('figmap','mapcanvas'),('figsmooth','smoothcanvas'),
                              ('figroses','rosescanvas'),('figcurves','curvecanvas')][view]
        stack=self.plot_stacks[view]
        if key not in self._render_cache:
            fig=Figure(figsize=(10,7) if view==3 else (7,7),tight_layout=view!=3)
            canvas=ResultCanvas(fig)
            setattr(self,fig_name,fig);setattr(self,canvas_name,canvas)
            p=self.results[i]
            if view==0:self._draw_map(p)
            elif view==1:
                if self.quasi_result:
                    step=self.time_slider.value();self._draw_smooth(self.quasi_result.smooth(i,step),self.quasi_result.snapshot(i,step));canvas._quasi_step=step
                else:self._draw_smooth(self.smooth_fields[i],p)
            elif view==2:self._draw_roses(p)
            else:
                self._draw_curves(p);attach_interaction(canvas)
            stack.addWidget(canvas)
            self._render_cache[key]=canvas
        canvas=self._render_cache[key]
        self._render_cache.move_to_end(key)
        setattr(self,fig_name,canvas.figure);setattr(self,canvas_name,canvas)
        stack.setCurrentWidget(canvas)
        while len(self._render_cache)>self.RENDER_CACHE_LIMIT:
            _,old=self._render_cache.popitem(last=False)
            for old_stack in self.plot_stacks:
                if old_stack.indexOf(old)>=0:old_stack.removeWidget(old);break
            old.deleteLater()
        if view==1 and self.quasi_result:self._update_time_field(canvas,i)
        if hasattr(self,"graph_links"):QTimer.singleShot(0,self.graph_links.refresh)
        return canvas.figure

    def _open_axes_settings(self):
        self.axes_dialog=open_axes_settings(self.curvecanvas,self) if self.plots.currentIndex()==3 else None

    def _view_changed(self,*args):
        if self._restoring:return
        self._plot_result()

    def _draw_map(self,p):
        self.figmap.clear()
        ax=self.figmap.add_subplot(111,projection='polar')
        n=self.cfg.n_angular
        theta_edge=np.deg2rad(grid_angles(self.cfg,edges=True))
        th,r=np.meshgrid(theta_edge,p.radial_edges_mm)
        metric=self.metric.currentData()
        values=cell_values(p,metric,self.cfg.density_kg_m3,self.volume_interval.value())
        data,norm,cmap=color_scale(values,self.cfg,[cell_values(v,metric,self.cfg.density_kg_m3,self.volume_interval.value()) for v in self.results])
        pc=ax.pcolormesh(th,r,data,shading='flat',cmap=cmap,norm=norm)
        ax.set_theta_zero_location('E')
        ax.set_theta_direction(1);polar_ticks(ax,self.cfg)
        ax.set_ylim(0,p.radial_edges_mm[-1])
        ax.set_title(f'{METRICS[metric]} · z={p.z_mm:g} мм',pad=18,fontsize=11)
        self.figmap.colorbar(pc,ax=ax,pad=.14,shrink=.68,label=UNITS[metric])
        self._draw_nozzle_markers(ax,polar=True)
        self.mapcanvas.draw_idle()

    def _draw_nozzle_markers(self, ax, polar=False):
        draw_nozzle_markers(ax,self.cfg,self.nozzles,polar=polar,
                            outlines=self.draw_outlines.isChecked(),
                            labels=self.draw_labels.isChecked())

    def _draw_smooth(self, field, p):
        cfg=self.cfg
        nozzles=self.nozzles
        if self.quasi_result:
            from dataclasses import replace
            index=matching_index([op.z_mm for op in self.quasi_result.operators],p.z_mm);flows=self.quasi_result.flows[self.time_slider.value()]
            nozzles=[replace(n,flow_mode='manual',flow_kg_h=float(g*3600)) for n,g in zip(self.quasi_result.nozzles,flows)]
            cfg=deepcopy(cfg);cfg.pressure_mode='legacy'
        if self.quasi_result and cfg.color_scale_mode=='auto':
            cfg.color_scale_mode='manual';cfg.color_vmin=0.;cfg.color_vmax=self.quasi_result.color_max(index,cfg.color_shared_z)
        ax=draw_detailed_map(self.figsmooth,field,p,cfg,nozzles,
                          outlines=self.draw_outlines.isChecked(),
                          labels=self.draw_labels.isChecked(),
                          roses=self.draw_rose_over_field.isChecked(),
                          rose_scale_mm=self.rose_scale.value(),
                          shared_values=[f.masked_flux_kg_m2_s for f in self.smooth_fields])
        if self.quasi_result:
            ax.set_title(f'Мгновенное поле · z={p.z_mm:g} мм; время — на ползунке',fontsize=10);self.smoothcanvas.quasi_animator=QuasiFieldAnimator(self.smoothcanvas,ax,cfg)
        self.smoothcanvas.draw_idle()

    def _schedule_time_field(self,*args):
        # Throttle while dragging: restarting on every move would postpone
        # every frame until the user stops moving the handle.
        if not self.time_timer.isActive():self.time_timer.start()

    def _update_time_field(self,canvas=None,plane_index=None):
        self.time_timer.stop()
        if not self.quasi_result or not self.results:return
        i=max(0,self.plane_select.currentIndex()) if plane_index is None else plane_index;step=self.time_slider.value();result=self.quasi_result
        balance=result.balances[i*result.settings.n_steps+step]
        pressures='; '.join(f'{key}: Δp={values[step]:.7g} кгс/см²' for key,values in result.pressures.items())
        self.time_slider.setToolTip(f'Угол в системе 01: {result.settings.angles[step]:.8g}°; физический угол от +X против ЧС: {result.settings.global_angles(result.cfg)[step]:.8g}°. Изменяется интенсивность мгновенного поля; накопленная матрица сохраняется.')
        reference=result.balances[i*result.settings.n_steps]['injected_kg_s'];delta=f'{(balance["injected_kg_s"]/reference-1)*100:+.3f}%' if reference else '—'
        self.time_label.setText(
            f'Шаг {step}/{result.settings.n_steps-1} · t={result.settings.times[step]:.8g} с · угол гребёнки {result.settings.angles[step]:.8g}° · {pressures}\n'
            f'Мгновенный баланс: подано {balance["injected_kg_s"]*3600:.7g} кг/ч ({delta} к шагу 0); '
            f'внутри {balance["inside_kg_s"]*3600:.7g}; за внутренней {balance["inner_kg_s"]*3600:.7g}; '
            f'за наружной {balance["outer_kg_s"]*3600:.7g}; wall-bound {balance.get("wall_bound_kg_s",0)*3600:.7g}; невязка {balance["relative_error"]:.3g}.')
        if 'suspended_kg_s' in balance:
            self.time_label.setText(
                f'Шаг {step}/{result.settings.n_steps-1} · t={result.settings.times[step]:.8g} с · угол {result.settings.angles[step]:.8g}° · {pressures}\n'
                f'Подано {balance["injected_kg_s"]*3600:.7g} кг/ч ({delta}); взвесь {balance["suspended_kg_s"]*3600:.7g}; '
                f'плёнка внутренняя {balance["film_inner_kg_s"]*3600:.7g}; наружная {balance["film_outer_kg_s"]*3600:.7g} кг/ч; '
                f'невязка массы {balance["relative_error"]:.3g}. Скорость истечения задана отдельно от давления; задержка переноса пульсаций не моделируется.')
        if canvas is None:
            if self.plots.currentIndex()!=1:return
            canvas=self.plot_stacks[1].currentWidget()
        if hasattr(canvas,'quasi_animator') and getattr(canvas,'_quasi_step',None)!=step:
            field=result.smooth(i,step);canvas.quasi_animator.update(field);canvas._quasi_step=step

    def _draw_roses(self,p):
        cfg=self.cfg;nozzles=self.nozzles
        if self.quasi_result:
            from dataclasses import replace
            cfg=deepcopy(cfg);cfg.pressure_mode='legacy';nozzles=[replace(n,flow_mode='manual',flow_kg_h=float(g*3600)) for n,g in zip(self.quasi_result.nozzles,self.quasi_result.flows[self.time_slider.value()])]
        draw_polar_nozzles_map(self.figroses,p,cfg,nozzles,
                               outlines=self.draw_outlines.isChecked(),
                               labels=self.draw_labels.isChecked(),
                               rose_scale_mm=self.rose_scale.value())
        self.rosescanvas.draw_idle()

    def _draw_curves(self,p):
        self.figcurves.clear()
        total=len(p.radial_centers_mm);n=min(total,self.plot_count.value())
        start=0 if n==total else max(0,self.compare_radius.currentIndex())
        indices=[matching_index(p.radial_centers_mm,r) for r in self.graph_selection.selected]
        indices=[k for k in indices if k is not None];n=len(indices)
        cols=min(n,self.plot_columns.value())
        rows=int(np.ceil(n/cols))
        mode=self.x_axis.currentData();angles=plot_x(p.theta_centers_deg,self.cfg,mode);cycle=period(self.cfg,mode)
        order=np.argsort(angles)
        x=angles[order]
        for slot,k in enumerate(indices):
            ax=self.figcurves.add_subplot(rows,cols,slot+1)
            y=cell_values(p,self.metric.currentData(),self.cfg.density_kg_m3,self.volume_interval.value())[k][order]
            ax.plot(np.r_[x,x[0]+cycle],np.r_[y,y[0]],lw=1.2)
            selected=self.compare_radius.currentData() is not None and np.isclose(p.radial_centers_mm[k],self.compare_radius.currentData(),rtol=0,atol=1e-7)
            ax.set_title(f'Уровень {k+1}: R={p.radial_centers_mm[k]:.2f} мм'+(' · сравнение' if selected else ''),fontsize=9,color='#734ca0' if selected else '#222222')
            decorate_axes(ax,self.cfg,mode);ax._compare_y=UNITS[self.metric.currentData()]
            ax.set_xlim(0,cycle)
            ax.grid(alpha=.24)
            ax.tick_params(labelsize=7)
            ax.set_ylabel(UNITS[self.metric.currentData()],fontsize=7)
            ax.set_xlabel(x_label(self.cfg,mode),fontsize=7)
        self.figcurves.suptitle(f'Окружные эпюры · z={p.z_mm:g} мм',fontsize=11)
        self.figcurves.subplots_adjust(hspace=.65,wspace=.35,top=.92,bottom=.06)
        self.curvecanvas.draw_idle()

    def closeEvent(self,event):
        comparison=getattr(self,'comparison',None)
        regression=getattr(self,'regression_dialog',None)
        if self.thread is not None or comparison is not None and comparison.thread is not None or regression is not None and regression.thread is not None:
            self.statusBar().showMessage('Дождитесь завершения расчёта перед закрытием.',8000)
            event.ignore();return
        super().closeEvent(event)

    def export_xlsx(self):
        if not self.results:self._error('Сначала рассчитайте сечения.');return
        path,_=QFileDialog.getSaveFileName(self,'Окружные эпюры всех сечений','epures.xlsx','Excel (*.xlsx)')
        if path:
            try:
                if self.quasi_result:write_quasi_report(path,self.quasi_result,self.metric.currentData(),self.volume_interval.value())
                else:write_epures_xlsx(path,self.results,self.cfg,self.metric.currentData(),self.volume_interval.value())
            except Exception as exc:self._error(exc)

    def export_results(self):
        if not self.results:self._error('Сначала выполните расчёт.');return
        path,_=QFileDialog.getSaveFileName(self,'Матрицы всех сечений','fuel_fields.csv','CSV (*.csv)')
        if path:
            try:
                if self.quasi_result:write_quasi_report(path,self.quasi_result)
                else:write_result_csv(path,self.results,cfg=self.cfg)
            except Exception as exc:self._error(exc)

    def export_png(self):
        if not self.results:self._error('Сначала выполните расчёт.');return
        path,_=QFileDialog.getSaveFileName(self,'Сохранить изображение','field_map.png','PNG (*.png)')
        if path:
            try:
                if not path.lower().endswith('.png'):path+='.png'
                i=max(0,self.plane_select.currentIndex())
                for view in range(4):self._render_view(view,i)
                self.figmap.savefig(path,dpi=180,bbox_inches='tight')
                self.figcurves.savefig(path[:-4]+'_profiles.png',dpi=160,bbox_inches='tight')
                self.figsmooth.savefig(path[:-4]+'_detailed.png',dpi=180,bbox_inches='tight')
                self.figroses.savefig(path[:-4]+'_polar_nozzles.png',dpi=180,bbox_inches='tight')
                self._plot_result()
            except Exception as exc:self._error(exc)

    def export_all_png(self):
        if not self.results:self._error('Сначала выполните расчёт.');return
        folder=QFileDialog.getExistingDirectory(self,'Папка для всех сечений и графиков')
        if not folder:return
        try:
            current=self.plane_select.currentIndex()
            for i in range(len(self.results)):
                self.plane_select.setCurrentIndex(i)
                for view in range(4):self._render_view(view,i)
                self.figmap.savefig(os.path.join(folder,f'section_{i+1}_field.png'),dpi=180,bbox_inches='tight')
                self.figcurves.savefig(os.path.join(folder,f'section_{i+1}_profiles.png'),dpi=160,bbox_inches='tight')
                self.figsmooth.savefig(os.path.join(folder,f'section_{i+1}_detailed.png'),dpi=180,bbox_inches='tight')
                self.figroses.savefig(os.path.join(folder,f'section_{i+1}_polar_nozzles.png'),dpi=180,bbox_inches='tight')
            self.plane_select.setCurrentIndex(current)
            self.statusBar().showMessage(f'Сохранены {4*len(self.results)} PNG: карты и эпюры всех сечений.',15000)
        except Exception as exc:self._error(exc)

    def new_project(self):
        if self.thread is not None:self._error('Дождитесь завершения расчёта.');return
        reply=QMessageBox.question(self,'Новый проект','Сбросить все значения и эпюры?',QMessageBox.Yes|QMessageBox.No)
        if reply!=QMessageBox.Yes:return
        self._restoring=True
        self.cfg=Settings(pressure_mode='calibrated',flow_pressure_kgf_cm2=30.,sector_pressure_kgf_cm2=2.5)
        self.nozzles=default_nozzles(self.cfg);self.sector_id=1
        for n in self.nozzles:n.geometry_overrides=[]
        self.project_file=None;self.results=[];self.smooth_fields=[];self._fill_nozzles();self._fill_controls()
        self.plane_select.clear();self.balance_label.setText('Нет расчётных результатов.')
        self._reset_history()

    def save_project(self,save_as=False):
        try:
            self._record_edit()
            cfg=self._read_controls()
            nozzles=self._read_nozzles(cfg)
            path=self.project_file
            if not path or save_as:
                path,_=QFileDialog.getSaveFileName(self,'Сохранить проект','fuel_field_project.json','JSON (*.json)')
            if not path:return
            if not path.lower().endswith('.json'):path+='.json'
            if path.lower().endswith('.quasi.json'):raise ValueError('Расширение .quasi.json предназначено для настроек режима. Сохраните проект с именем *_project.json.')
            with open(path,'w',encoding='utf-8') as f:
                json.dump(project_to_json(cfg,nozzles),f,ensure_ascii=False,indent=2)
            self.project_file=path;self.cfg=cfg;self.nozzles=nozzles
            self.statusBar().showMessage(f'Сохранено: {path}',12000)
        except Exception as exc:self._error(exc)

    def open_project(self):
        if self.thread is not None:self._error('Дождитесь завершения расчёта.');return
        path,_=QFileDialog.getOpenFileName(self,'Открыть проект','','JSON (*.json)')
        if not path:return
        try:
            with open(path,'r',encoding='utf-8') as f:data=json.load(f)
            if data.get('format')=='fuel-quasistationary-settings':raise ValueError('Это настройки режима *.quasi.json. Загрузите их в окне настроек квазистационарного режима, а не как проект.')
            cfg,nozzles=project_from_json(data)
            self._restoring=True
            self.cfg,self.nozzles=cfg,nozzles
            self.project_file=path;self.results=[];self.smooth_fields=[];self.sector_id=1
            self._fill_controls();self._fill_nozzles()
            self.plane_select.clear();self.balance_label.setText('Загружены исходные данные. Нужен новый расчёт.')
            self.statusBar().showMessage(f'Загружено: {path}',12000)
            self._reset_history()
        except Exception as exc:self._error(exc)

    def _reset_history(self):
        self.quasi_result=None;self.time_timer.stop();self.time_controls.hide()
        existing=getattr(self,'quasi_dialog',None)
        if existing is not None:existing.close()
        existing=getattr(self,'transport_dialog',None)
        if existing is not None:existing.close()
        self.undo_stack.clear()
        self._edit_state=project_to_json(self.cfg,self.nozzles)
        self._restoring=False
        self._revision+=1
        self._invalidate_render()
        self._queue_preview()


def main():
    app=QApplication(sys.argv)
    app.setStyle('Fusion')
    win=MainWindow();win.show()
    sys.exit(app.exec_())
