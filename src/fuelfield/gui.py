"""PyQt5 pilot for ring-field model. Launch with python main.py."""
from __future__ import annotations
import json, os, sys, traceback
from pathlib import Path
import numpy as np
from PyQt5.QtCore import Qt, QThread, QObject, pyqtSignal
from PyQt5.QtGui import QKeySequence, QFont
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QFormLayout,
    QTabWidget, QTableWidget, QTableWidgetItem, QLabel, QPushButton, QComboBox,
    QSpinBox, QDoubleSpinBox, QLineEdit, QFileDialog, QMessageBox, QSplitter,
    QGroupBox, QScrollArea, QPlainTextEdit, QHeaderView, QAbstractItemView,
    QStatusBar, QProgressBar, QMenu, QAction, QInputDialog, QCheckBox
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
from .visualization import draw_nozzle_markers, draw_detailed_map, draw_polar_nozzles_map

FIELDS = [
    ('id','ID'),('enabled','Установлена'),('row','Ряд'),
    ('theta_deg','θ форс., °'),('radius_mm','R форс., мм'),
    ('flow_kg_h','Расход, кг/ч'),('flow_mode','Расход из'),
    ('alpha_deg','α, полный °'),('rc0_mm','Rc0, мм'),('h0_mm','h0, мм'),
    ('k_linear','k лин.'),('k_quadratic','k квадр.'),('power','p степ.'),
    ('z_ref_mm','z ref, мм'),('inner_angle_deg','α внутр., °'),
    ('outer_angle_deg','α наруж., °'),('clock_deg','Clock, °'),
    ('decay_length_mm','L затух., мм'),('diffusion_deg_sqrt_mm','σ / √мм'),
    ('rotation_deg_per_mm','ω °/мм'),('sector_source','Источник эпюры'),
]
INTCOL = {key:i for i,(key,_) in enumerate(FIELDS)}
NUMERICS = {f for f,_ in FIELDS} - {'id','enabled','row','flow_mode','sector_source'}


def _number(txt: str):
    return float(txt.strip().replace(',','.'))


def _dbl(value, low=-1e7, high=1e7, decimals=5, step=1.):
    c = QDoubleSpinBox()
    c.setDecimals(decimals)
    c.setRange(low,high)
    c.setSingleStep(step)
    c.setValue(float(value))
    return c


def _integer(value, lo, hi):
    c=QSpinBox();c.setRange(lo,hi);c.setValue(value);return c


class PasteTable(QTableWidget):
    """Tab-separated copy/paste (Excel compatible)."""
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
            raw=QApplication.clipboard().text().strip('\n')
            start=self.currentIndex()
            if not start.isValid():return
            for dr,line in enumerate(raw.splitlines()):
                for dc,txt in enumerate(line.split('\t')):
                    row=start.row()+dr;col=start.column()+dc
                    if row>=self.rowCount() or col>=self.columnCount():continue
                    item=self.item(row,col)
                    if item is not None and item.flags() & Qt.ItemIsEditable:
                        item.setText(txt.strip())
            return
        super().keyPressEvent(ev)


class CalcWorker(QObject):
    progress=pyqtSignal(str)
    completed=pyqtSignal(object)
    failed=pyqtSignal(str)
    def __init__(self,settings,nozzles):
        super().__init__()
        self.settings=settings;self.nozzles=nozzles
    def run(self):
        try:
            result=[]
            for i,z in enumerate(self.settings.z_mm):
                self.progress.emit(f'Сечение {i+1}/4, z={z:g} мм…')
                plane=calculate_plane(self.nozzles,self.settings,z)
                preview=evaluate_continuous_field(self.nozzles,self.settings,z)
                result.append((plane,preview))
            self.completed.emit(result)
        except Exception:
            self.failed.emit(traceback.format_exc())


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Fuel Field Pilot v0.1.2 — поле топлива фронтового устройства')
        self.resize(1480,930)
        self.cfg=Settings()
        self.nozzles=default_nozzles(self.cfg)
        self.results=[]
        self.smooth_fields=[]
        self.sector_id=None
        self.project_file=None
        self.thread=None
        self.model_widgets={}
        self.geom_widgets={}
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
        a=QAction('Рассчитать 4 сечения',self);a.setShortcut('F5');a.triggered.connect(self.run_calc);menu.addAction(a)

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
            ('grid_start_deg','Первый центр окружной матрицы: угол, °',_dbl(0,-360,360)),
            ('z_mm','Расстояния четырёх сечений z, мм',QLineEdit('0; 20; 40; 60')),
            ('radial_centers_mm','Радиусы центров уровней, мм (пусто = равномерно)',QLineEdit('')),
        ]
        for key,label,w in fields:
            self.geom_widgets[key]=w
            if isinstance(w,QLineEdit):w.setMinimumWidth(380)
            form.addRow(label,w)
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
        layout.addWidget(scroll)
        self.tabs.addTab(page,'01 · Геометрия и сетка')

    def _build_nozzles_tab(self):
        page=QWidget();layout=QVBoxLayout(page)
        top=QHBoxLayout()
        for label,slot in [
            ('Импорт 36 секторов: Excel / CSV',self.import_sectors),
            ('Экспорт эпюр CSV',self.export_sectors),
            ('Копировать эпюру на выбранные',self.copy_sectors),
            ('Расход из объёмов для выбранных',self.sum_sectors_to_flow),
            ('ДЕМО: 10 кг/ч всем',self.assign_demo_flow),
        ]:
            b=QPushButton(label);b.clicked.connect(slot);top.addWidget(b)
        layout.addLayout(top)
        opts=QHBoxLayout()
        self.duration_spin=_dbl(180,0.01,1e6,3,10.)
        self.density_spin=_dbl(800,1,10000,2,10.)
        opts.addWidget(QLabel('Время сбора, с:'));opts.addWidget(self.duration_spin)
        opts.addWidget(QLabel('Плотность топлива, кг/м³:'));opts.addWidget(self.density_spin)
        opts.addWidget(QLabel('Сумма V используется для расхода ТОЛЬКО в режиме «sectors».'))
        opts.addStretch();layout.addLayout(opts)
        splitter=QSplitter(Qt.Vertical)
        self.nozzle_table=PasteTable(140,len(FIELDS))
        self.nozzle_table.setHorizontalHeaderLabels([lab for _,lab in FIELDS])
        self.nozzle_table.setAlternatingRowColors(True)
        self.nozzle_table.setSelectionBehavior(QAbstractItemView.SelectItems)
        self.nozzle_table.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.nozzle_table.verticalHeader().setDefaultSectionSize(25)
        self.nozzle_table.horizontalHeader().setDefaultSectionSize(112)
        self.nozzle_table.setColumnWidth(INTCOL['id'],52)
        self.nozzle_table.setColumnWidth(INTCOL['sector_source'],170)
        self.nozzle_table.itemSelectionChanged.connect(self._sector_selection)
        splitter.addWidget(self.nozzle_table)
        box=QWidget();p=QVBoxLayout(box)
        self.sector_label=QLabel('Секторные объёмы форсунки: выберите строку выше. Данные в мл.')
        p.addWidget(self.sector_label)
        self.sector_table=PasteTable(36,2)
        self.sector_table.setHorizontalHeaderLabels(['Сектор №','Объём, мл'])
        self.sector_table.horizontalHeader().setStretchLastSection(True)
        self.sector_table.setMaximumWidth(360)
        for k in range(36):
            it=QTableWidgetItem(str(k+1));it.setFlags(Qt.ItemIsEnabled)
            self.sector_table.setItem(k,0,it)
        self.sector_table.itemChanged.connect(self._sector_changed)
        row=QHBoxLayout();row.addWidget(self.sector_table)
        notes=QLabel('Порядок секторов: №1 направлен к средней линии ФУ; №2–36 идут против часовой стрелки при взгляде со стороны выхода топлива.\n\n'
                     'При ручном редактировании эпюры изменяется форма распределения, но НЕ независимый расход форсунки (manual).\n\n'
                     'Ctrl+C / Ctrl+V копируют/вставляют данные из Excel. В верхней таблице колонка «Расход из» принимает manual или sectors.\n\n'
                     'Если используете «sectors», расход = сумма мл × плотность / время опыта. Этот подход НЕ учитывает капли вне бокса.')
        notes.setWordWrap(True);row.addWidget(notes);p.addLayout(row)
        splitter.addWidget(box);splitter.setStretchFactor(0,3);splitter.setStretchFactor(1,2)
        layout.addWidget(splitter)
        self.tabs.addTab(page,'02 · 140 форсунок и 36 секторов')

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
        description.setPlainText('''МОДЕЛИ ГЕОМЕТРИИ (все расстояния и параметры задаются в мм)
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
        self.tabs.addTab(page,'03 · Математические модели')

    def _build_results_tab(self):
        page=QWidget();lay=QVBoxLayout(page)
        actions=QHBoxLayout()
        self.calc_btn=QPushButton('Рассчитать четыре сечения (F5)')
        self.calc_btn.clicked.connect(self.run_calc)
        actions.addWidget(self.calc_btn)
        self.plane_select=QComboBox();self.plane_select.currentIndexChanged.connect(self._plot_result)
        actions.addWidget(QLabel('Сечение:'));actions.addWidget(self.plane_select)
        for label,slot in [('CSV: матрицы',self.export_results),('PNG: текущее',self.export_png),
                            ('PNG: все 4',self.export_all_png)]:
            b=QPushButton(label);b.clicked.connect(slot);actions.addWidget(b)
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
            view.addWidget(w);w.stateChanged.connect(self._plot_result)
        view.addWidget(QLabel('Масштаб полярных эпюр, мм:'))
        view.addWidget(self.rose_scale);self.rose_scale.valueChanged.connect(self._plot_result)
        view.addStretch();lay.addLayout(view)
        self.balance_label=QLabel('Результатов пока нет. Задайте исходные данные и нажмите «Рассчитать».')
        self.balance_label.setWordWrap(True);lay.addWidget(self.balance_label)
        note=QLabel('Важно: 6×140 — контрольные ячейки для количественного анализа; подробная карта — непрерывное отображение отдельных ядер. '
                    'Полярная эпюра показывает угловую ИНТЕНСИВНОСТЬ, не физический контур факела. Контур форсунки — отдельная окружность.')
        note.setWordWrap(True);note.setStyleSheet('color:#525f71;');lay.addWidget(note)
        plots=QTabWidget()
        self.figmap=Figure(figsize=(6,6),tight_layout=True)
        self.mapcanvas=FigureCanvas(self.figmap)
        plots.addTab(self.mapcanvas,'Матрица (консервативная)')
        self.figsmooth=Figure(figsize=(7,7),tight_layout=True)
        self.smoothcanvas=FigureCanvas(self.figsmooth)
        plots.addTab(self.smoothcanvas,'Подробное суммарное поле')
        self.figroses=Figure(figsize=(7,7),tight_layout=True)
        self.rosescanvas=FigureCanvas(self.figroses)
        plots.addTab(self.rosescanvas,'140 полярных эпюр на общей схеме')
        self.figcurves=Figure(figsize=(10,7),tight_layout=True)
        self.curvecanvas=FigureCanvas(self.figcurves)
        plots.addTab(self.curvecanvas,'Окружные эпюры радиальных уровней')
        lay.addWidget(plots)
        self.tabs.addTab(page,'04 · Поля и эпюры')

    def _fill_controls(self):
        for key,w in self.geom_widgets.items():
            value=getattr(self.cfg,key)
            if isinstance(w,QLineEdit):
                if key=='z_mm':w.setText('; '.join(f'{a:g}' for a in value))
                elif key=='radial_centers_mm':w.setText('' if value is None else '; '.join(f'{x:g}' for x in value))
            else:w.setValue(value)
        for key,w in self.model_widgets.items():
            val=getattr(self.cfg,key)
            if isinstance(w,QComboBox):w.setCurrentIndex(w.findData(val))
            else:w.setValue(val)
        self.duration_spin.setValue(self.cfg.duration_s)
        self.density_spin.setValue(self.cfg.density_kg_m3)

    def _read_controls(self):
        cfg=Settings(**{k:getattr(self.cfg,k) for k in vars(self.cfg)})
        for key,w in self.geom_widgets.items():
            if key in ('z_mm','radial_centers_mm'):
                text=w.text().strip().replace(',','.')
                if key=='radial_centers_mm' and not text:
                    setattr(cfg,key,None)
                else:
                    values=[float(x.strip()) for x in text.split(';') if x.strip()]
                    setattr(cfg,key,values)
            else:setattr(cfg,key,w.value())
        for key,w in self.model_widgets.items():
            setattr(cfg,key,w.currentData() if isinstance(w,QComboBox) else w.value())
        cfg.duration_s=self.duration_spin.value()
        cfg.density_kg_m3=self.density_spin.value()
        cfg.validate()
        return cfg

    def _fill_nozzles(self):
        tab=self.nozzle_table
        tab.blockSignals(True)
        tab.setRowCount(len(self.nozzles))
        for row,n in enumerate(self.nozzles):
            for col,(key,_) in enumerate(FIELDS):
                v=getattr(n,key)
                if key=='enabled':
                    it=QTableWidgetItem('')
                    it.setFlags(Qt.ItemIsEnabled|Qt.ItemIsSelectable|Qt.ItemIsUserCheckable)
                    it.setCheckState(Qt.Checked if v else Qt.Unchecked)
                else:
                    if isinstance(v,float): val=f'{v:.9g}'
                    else:val=str(v)
                    it=QTableWidgetItem(val)
                    if key in ('id','sector_source'):
                        it.setFlags(Qt.ItemIsEnabled|Qt.ItemIsSelectable)
                tab.setItem(row,col,it)
        tab.blockSignals(False)
        if self.sector_id is None:self.sector_id=1
        self._show_sectors(self.sector_id)

    def _read_nozzles(self):
        nozzles=[]
        for i,n in enumerate(self.nozzles):
            raw=vars(n).copy()
            for col,(key,_) in enumerate(FIELDS):
                item=self.nozzle_table.item(i,col)
                if key in ('id','sector_source'):continue
                if key=='enabled':raw[key]=item.checkState()==Qt.Checked
                else:
                    val=item.text().strip()
                    if key in NUMERICS:
                        try:raw[key]=_number(val)
                        except ValueError:raise ValueError(f'Строка {i+1}, столбец «{FIELDS[col][1]}»: неверное число «{val}».')
                    elif key=='flow_mode':
                        raw[key]=val.lower()
                        if raw[key] not in ('manual','sectors'):
                            raise ValueError(f'Строка {i+1}: «Расход из» должен быть manual либо sectors.')
                    else:raw[key]=val.lower()
            if raw['row'] not in ('inner','outer'):
                raise ValueError(f'Строка {i+1}: ряд должен быть inner либо outer.')
            if raw['flow_kg_h'] < 0:
                raise ValueError(f'Строка {i+1}: расход не может быть отрицательным.')
            nozzles.append(Nozzle(**raw))
        return nozzles

    def _sync_nozzles(self):
        self.nozzles=self._read_nozzles()

    def _sector_selection(self):
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

    def _update_sector_label(self):
        n=next(n for n in self.nozzles if n.id==self.sector_id)
        self.sector_label.setText(f'Форсунка №{n.id}, {n.row}: 36 секторных объёмов [мл]; '
                                  f'сумма = {sum(n.sector_ml):.4g} мл; {n.sector_source}')

    def _sector_changed(self,item):
        if item.column()!=1 or self.sector_id is None:return
        n=next(n for n in self.nozzles if n.id==self.sector_id)
        try:
            value=_number(item.text())
            if value < 0 or not np.isfinite(value):raise ValueError('Неотрицательное число обязательно')
        except ValueError:
            self.sector_table.blockSignals(True)
            item.setText(f'{n.sector_ml[item.row()]:.10g}')
            self.sector_table.blockSignals(False)
            self.statusBar().showMessage('Объём сектора: введите неотрицательное число (мл). Предыдущее значение восстановлено.',12000)
            return
        n.sector_ml[item.row()]=value
        n.sector_source='ручное редактирование'
        self._update_sector_label()
        self.nozzle_table.item(n.id-1,INTCOL['sector_source']).setText(n.sector_source)

    def _selected_ids(self):
        return sorted(set(ind.row()+1 for ind in self.nozzle_table.selectedIndexes()))

    def _error(self,description):
        QMessageBox.critical(self,'Ошибка',str(description))

    def update_positions(self):
        try:
            cfg=self._read_controls()
            self._sync_nozzles()
            apply_default_positions(self.nozzles,cfg)
            self._fill_nozzles()
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
                n=self.nozzles[idx-1]
                n.sector_ml=src.sector_ml.copy()
                n.sector_source=f'копия №{source}'
            self._fill_nozzles()
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
                n=self.nozzles[id-1]
                if n.sector_source=='шаблон: равномерно':continue
                if sum(n.sector_ml)<=0:continue
                n.flow_mode='sectors'
                changed+=1
            self._fill_nozzles()
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
            self.statusBar().showMessage('ДЕМО расход назначен всем форсункам: 10 кг/ч. Это НЕ экспериментальные данные.',12000)
        except Exception as exc:self._error(exc)

    def _prepare(self):
        cfg=self._read_controls()
        nozzles=self._read_nozzles()
        flowing=[n for n in nozzles if n.enabled and source_flow_kg_s(n,cfg)>0]
        if not flowing:
            raise ValueError('Нет ни одной установленной форсунки с положительным расходом. '
                             'Введите измеренный расход в кг/ч или выберите режим sectors.')
        return cfg,nozzles

    def run_calc(self):
        if self.thread is not None:return
        try:
            cfg,nozzles=self._prepare()
            # Avoid live table mutations during calculation; immutable state snapshot.
            cfg,nozzles=project_from_json(project_to_json(cfg,nozzles))
            self.cfg=cfg;self.nozzles=nozzles
        except Exception as exc:
            self._error(exc);return
        self.results=[]
        self.smooth_fields=[]
        self.progress.setValue(0)
        self.calc_btn.setEnabled(False)
        self.thread=QThread(self)
        self.worker=CalcWorker(cfg,nozzles)
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.progress.connect(self._progress)
        self.worker.completed.connect(self._done)
        self.worker.failed.connect(self._failed)
        self.worker.completed.connect(self.thread.quit)
        self.worker.failed.connect(self.thread.quit)
        self.thread.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self._cleanup_thread)
        self.thread.start()
        self.tabs.setCurrentIndex(3)

    def _progress(self,message):
        self.statusBar().showMessage(message)
        self.progress.setValue(min(3,self.progress.value()+1))

    def _done(self,planes):
        self.results=[v[0] for v in planes]
        self.smooth_fields=[v[1] for v in planes]
        self.progress.setValue(4)
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
        count=sum(1 for n in self.nozzles if n.enabled and source_flow_kg_s(n,self.cfg)>0)
        self.statusBar().showMessage(f'Готово: 4 сечения, {count} форсунок с расходом.',20000)

    def _failed(self,message):
        self._error('Расчёт не выполнен:\n'+message[-2200:])
        self.statusBar().showMessage('Ошибка расчёта.',10000)

    def _cleanup_thread(self):
        self.thread.deleteLater()
        self.thread=None
        self.worker=None
        self.calc_btn.setEnabled(True)

    def _plot_result(self,*args):
        if not self.results:return
        i=max(0,self.plane_select.currentIndex())
        p=self.results[i]
        total=p.injected_kg_s*3600
        self.balance_label.setText(
            f'<b>Сечение {i+1}, z={p.z_mm:g} мм.</b> '
            f'Введено {total:.5g} кг/ч; в тракте {p.inside_kg_s*3600:.5g} кг/ч '
            f'({100*p.inside_kg_s/max(p.injected_kg_s,1e-15):.2f}%); '
            f'за внутренней границей {p.inner_kg_s*3600:.5g} кг/ч; '
            f'за наружной {p.outer_kg_s*3600:.5g} кг/ч; '
            f'невязка баланса {100*p.rel_balance_error:.3g}%.'
        )
        self._draw_map(p)
        self._draw_smooth(self.smooth_fields[i], p)
        self._draw_roses(p)
        self._draw_curves(p)

    def _draw_map(self,p):
        self.figmap.clear()
        ax=self.figmap.add_subplot(111,projection='polar')
        n=self.cfg.n_angular
        theta_edge=np.deg2rad(self.cfg.grid_start_deg+(np.arange(n+1)-0.5)*360/n)
        th,r=np.meshgrid(theta_edge,p.radial_edges_mm)
        pc=ax.pcolormesh(th,r,p.flux_kg_m2_s,shading='flat',cmap='turbo')
        ax.set_theta_zero_location('E')
        ax.set_theta_direction(1)
        ax.set_ylim(0,self.cfg.r_out_mm)
        ax.set_title(f'Поле плотности расхода · z={p.z_mm:g} мм',pad=18,fontsize=11)
        self.figmap.colorbar(pc,ax=ax,pad=.14,shrink=.68,label='кг/(м²·с)')
        self._draw_nozzle_markers(ax,polar=True)
        self.mapcanvas.draw_idle()

    def _draw_nozzle_markers(self, ax, polar=False):
        draw_nozzle_markers(ax,self.cfg,self.nozzles,polar=polar,
                            outlines=self.draw_outlines.isChecked(),
                            labels=self.draw_labels.isChecked())

    def _draw_smooth(self, field, p):
        draw_detailed_map(self.figsmooth,field,p,self.cfg,self.nozzles,
                          outlines=self.draw_outlines.isChecked(),
                          labels=self.draw_labels.isChecked(),
                          roses=self.draw_rose_over_field.isChecked(),
                          rose_scale_mm=self.rose_scale.value())
        self.smoothcanvas.draw_idle()

    def _draw_roses(self,p):
        draw_polar_nozzles_map(self.figroses,p,self.cfg,self.nozzles,
                               outlines=self.draw_outlines.isChecked(),
                               labels=self.draw_labels.isChecked(),
                               rose_scale_mm=self.rose_scale.value())
        self.rosescanvas.draw_idle()

    def _draw_curves(self,p):
        self.figcurves.clear()
        n=len(p.radial_centers_mm)
        cols=2 if n>1 else 1
        rows=int(np.ceil(n/cols))
        angles=p.theta_centers_deg
        order=np.argsort(angles)
        x=angles[order]
        for k in range(n):
            ax=self.figcurves.add_subplot(rows,cols,k+1)
            y=p.flux_kg_m2_s[k][order]
            ax.plot(np.r_[x,x[0]+360],np.r_[y,y[0]],lw=1.2)
            ax.set_title(f'Уровень {k+1}: R={p.radial_centers_mm[k]:.2f} мм',fontsize=9)
            ax.set_xlim(0,360)
            ax.grid(alpha=.24)
            ax.tick_params(labelsize=7)
            ax.set_ylabel('кг/(м²·с)',fontsize=7)
            ax.set_xlabel('Окружной угол, °',fontsize=7)
        self.figcurves.suptitle(f'Окружные эпюры · z={p.z_mm:g} мм',fontsize=11)
        self.figcurves.subplots_adjust(hspace=.65,wspace=.35,top=.92,bottom=.06)
        self.curvecanvas.draw_idle()

    def export_results(self):
        if not self.results:self._error('Сначала выполните расчёт.');return
        path,_=QFileDialog.getSaveFileName(self,'Матрицы четырех сечений','fuel_fields.csv','CSV (*.csv)')
        if path:
            try:write_result_csv(path,self.results)
            except Exception as exc:self._error(exc)

    def export_png(self):
        if not self.results:self._error('Сначала выполните расчёт.');return
        path,_=QFileDialog.getSaveFileName(self,'Сохранить изображение','field_map.png','PNG (*.png)')
        if path:
            try:
                if not path.lower().endswith('.png'):path+='.png'
                self.figmap.savefig(path,dpi=180,bbox_inches='tight')
                self.figcurves.savefig(path[:-4]+'_profiles.png',dpi=160,bbox_inches='tight')
                self.figsmooth.savefig(path[:-4]+'_detailed.png',dpi=180,bbox_inches='tight')
                self.figroses.savefig(path[:-4]+'_polar_nozzles.png',dpi=180,bbox_inches='tight')
            except Exception as exc:self._error(exc)

    def export_all_png(self):
        if not self.results:self._error('Сначала выполните расчёт.');return
        folder=QFileDialog.getExistingDirectory(self,'Папка для четырех карт и 24 графиков')
        if not folder:return
        try:
            current=self.plane_select.currentIndex()
            for i in range(4):
                self.plane_select.setCurrentIndex(i)
                self.figmap.savefig(os.path.join(folder,f'section_{i+1}_field.png'),dpi=180,bbox_inches='tight')
                self.figcurves.savefig(os.path.join(folder,f'section_{i+1}_profiles.png'),dpi=160,bbox_inches='tight')
                self.figsmooth.savefig(os.path.join(folder,f'section_{i+1}_detailed.png'),dpi=180,bbox_inches='tight')
                self.figroses.savefig(os.path.join(folder,f'section_{i+1}_polar_nozzles.png'),dpi=180,bbox_inches='tight')
            self.plane_select.setCurrentIndex(current)
            self.statusBar().showMessage('Сохранены 16 PNG: матрицы, подробные поля, полярные эпюры и графики по радиальным уровням.',15000)
        except Exception as exc:self._error(exc)

    def new_project(self):
        reply=QMessageBox.question(self,'Новый проект','Сбросить все значения и эпюры?',QMessageBox.Yes|QMessageBox.No)
        if reply!=QMessageBox.Yes:return
        self.cfg=Settings();self.nozzles=default_nozzles(self.cfg);self.sector_id=1
        self.project_file=None;self.results=[];self.smooth_fields=[];self._fill_nozzles();self._fill_controls()
        self.plane_select.clear();self.balance_label.setText('Нет расчётных результатов.')

    def save_project(self,save_as=False):
        try:
            cfg=self._read_controls()
            nozzles=self._read_nozzles()
            path=self.project_file
            if not path or save_as:
                path,_=QFileDialog.getSaveFileName(self,'Сохранить проект','fuel_field_project.json','JSON (*.json)')
            if not path:return
            if not path.lower().endswith('.json'):path+='.json'
            with open(path,'w',encoding='utf-8') as f:
                json.dump(project_to_json(cfg,nozzles),f,ensure_ascii=False,indent=2)
            self.project_file=path;self.cfg=cfg;self.nozzles=nozzles
            self.statusBar().showMessage(f'Сохранено: {path}',12000)
        except Exception as exc:self._error(exc)

    def open_project(self):
        path,_=QFileDialog.getOpenFileName(self,'Открыть проект','','JSON (*.json)')
        if not path:return
        try:
            with open(path,'r',encoding='utf-8') as f:data=json.load(f)
            self.cfg,self.nozzles=project_from_json(data)
            self.project_file=path;self.results=[];self.smooth_fields=[];self.sector_id=1
            self._fill_controls();self._fill_nozzles()
            self.plane_select.clear();self.balance_label.setText('Загружены исходные данные. Нужен новый расчёт.')
            self.statusBar().showMessage(f'Загружено: {path}',12000)
        except Exception as exc:self._error(exc)


def main():
    app=QApplication(sys.argv)
    app.setStyle('Fusion')
    win=MainWindow();win.show()
    sys.exit(app.exec_())
