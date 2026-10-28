"""Air input draft and projections of the actual empirical jet trajectories.

Scene widths are one-sigma geometric envelopes of the same kernel inputs used
by the air matrices. They are not a sharp fluid boundary or a velocity vector.
"""
from copy import deepcopy
import math

import numpy as np
from PyQt5.QtCore import Qt, QPointF, QTimer
from PyQt5.QtGui import QColor, QPainterPath, QPen
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QFormLayout,
    QComboBox, QLineEdit, QLabel, QPushButton, QCheckBox, QScrollArea,
    QGraphicsPathItem, QGraphicsSimpleTextItem, QGraphicsItem)

from .air_sources import validate_air_model, normalize_source, resolve_sources
from .air_balance import solve_balance
from .air_distribution import cached_paths
from .air_jets import local_to_cartesian, cartesian_to_local


FIELDS = {
    'flow_kg_s': ('G, кг/с', 'Заданный расход одного элемента. Пусто — неизвестен.'),
    'share': ('Доля G КС', 'Фиксированная доля от суммарного расхода КС, 0…1.'),
    'area_mm2': ('F, мм²', 'Площадь прохода. Для area/effective_area требуется F > 0.'),
    'cd': ('Cd', 'Явный коэффициент расхода для effective_area; не определяется по P.'),
    'penetration_factor': ('Пробивная способность P', 'Безразмерный эмпирический параметр; не скорость и не импульс.'),
    'sigma_r_mm': ('σR, мм', 'Радиальная ширина ядра в начальном сечении; пунктир показывает ±1σ.'),
    'sigma_theta_deg': ('σθ, °', 'Окружная угловая ширина ядра; пунктир показывает ±1σ.'),
    'spread_mm_per_mm': ('Расширение, мм/мм', 'Приращение ширины на единицу длины пройденного пути.'),
    'mixed_fraction': ('Начальное вовлечение E', 'Доля расхода, уже отнесённая к смешанному воздуху, 0…1.'),
    'mixing_length_mm': ('Длина вовлечения, мм', 'Положительный масштаб роста E по пути. Пусто — постоянная начальная доля.'),
}
OPTIONAL = {'flow_kg_s', 'area_mm2', 'cd', 'mixing_length_mm'}


def project_points(points, angular=False):
    """Cartesian [z,x,y] -> Sketcher [horizontal, vertical], vertical inverted."""
    points = np.asarray(points, dtype=float)
    return np.column_stack((points[:, 1], -points[:, 2])) if angular else np.column_stack((points[:, 0], -np.hypot(points[:, 1], points[:, 2])))


def projected_envelopes(points, distance, source, angular=False, mixed=False):
    """Marginal one-sigma curves in the projection, using actual path s."""
    points = np.asarray(points, dtype=float)
    radius = np.hypot(points[:, 1], points[:, 2])
    spread = np.asarray(distance) * source['spread_mm_per_mm']
    multiplier = np.sqrt(2.) if mixed else 1.
    if angular:
        theta = np.arctan2(points[:, 2], points[:, 1])
        # Arc-length contribution of spreading to the angular sigma.
        sigma = multiplier*(np.deg2rad(source['sigma_theta_deg']) + np.divide(spread, radius, out=np.zeros_like(radius), where=radius > 0))
        curves = []
        for sign in (-1, 1):
            angle = theta + sign*sigma
            curves.append(np.column_stack((radius*np.cos(angle), -radius*np.sin(angle))))
            radial = np.maximum(0., radius+sign*multiplier*(source['sigma_r_mm']+spread))
            curves.append(np.column_stack((radial*np.cos(theta), -radial*np.sin(theta))))
        return curves
    sigma = multiplier*(source['sigma_r_mm'] + spread)
    return [np.column_stack((points[:, 0], -(radius+sign*sigma))) for sign in (-1, 1)]


class AirPathItem(QGraphicsPathItem):
    def __init__(self, panel, identity, path, role):
        super().__init__(path)
        self.panel = panel
        self.identity = identity
        self.setData(0, identity)
        self.setData(1, role)
        self.setCursor(Qt.PointingHandCursor)
        self.setZValue(12 if role == 'air-vector' else 9)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.panel.select_source(self.identity)
            event.accept()
            return
        super().mousePressEvent(event)


class AirGlyphItem(QGraphicsSimpleTextItem):
    def __init__(self,panel,identity,text):
        super().__init__(text);self.panel=panel;self.identity=identity
        self.setCursor(Qt.PointingHandCursor)

    def mousePressEvent(self,event):
        if event.button()==Qt.LeftButton:
            self.panel.select_source(self.identity);event.accept();return
        super().mousePressEvent(event)


class AirFlowPanel(QWidget):
    """A local, validated air-only draft; Apply emits one project transaction."""
    def __init__(self, editor):
        super().__init__(editor)
        self.editor = editor
        self._loading = False
        self._loaded = None
        self.draft = None
        self.items = []
        self.paths = None
        self.last_diagnostic = ''
        self._displayed = {}
        layout = QVBoxLayout(self)
        self.source_selector = QComboBox()
        self.source_selector.setMinimumContentsLength(14)
        self.source_selector.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self.source_selector.currentIndexChanged.connect(self.source_selected)
        layout.addWidget(QLabel('Воздушный элемент:'))
        layout.addWidget(self.source_selector)
        self.position = QLabel()
        self.position.setWordWrap(True)
        layout.addWidget(self.position)
        self.show_all = QCheckBox('Все потоки')
        self.show_all.setChecked(True)
        self.show_all.toggled.connect(self.draw_overlay)
        view_options=QHBoxLayout();view_options.addWidget(self.show_all)
        self.show_mixed = QCheckBox('Смешанный воздух (√2 σ)')
        self.show_mixed.toggled.connect(self.draw_overlay)
        view_options.addWidget(self.show_mixed);view_options.addStretch();layout.addLayout(view_options)
        self.enabled = QCheckBox('Источник включён')
        self.enabled.toggled.connect(self.schedule_change)
        form_widget = QWidget()
        form = QFormLayout(form_widget)
        form.addRow(self.enabled)
        self.method = QComboBox()
        for key, label in [('unknown', 'Неизвестен'), ('fixed', 'Заданный G'), ('fraction', 'Доля G КС'), ('area', 'По F'), ('effective_area', 'По CdF')]:
            self.method.addItem(label, key)
        self.method.currentIndexChanged.connect(self.schedule_change)
        form.addRow('Метод расхода', self.method)
        self.controls = {}
        for key, (label, tooltip) in FIELDS.items():
            widget = QLineEdit()
            widget.setToolTip(tooltip)
            widget.textEdited.connect(self.schedule_change)
            widget.editingFinished.connect(self.update_draft)
            self.controls[key] = widget
            form.addRow(label, widget)
        self.direction = [QLineEdit() for _ in range(3)]
        directions = QHBoxLayout()
        for label, widget in zip(('dz', 'dr', 'dθ'), self.direction):
            directions.addWidget(QLabel(label)); directions.addWidget(widget)
            widget.setMaximumWidth(80)
            widget.setToolTip('Компонента пространственного направления в локальном базисе [z,r,θ]; не скорость.')
            widget.textEdited.connect(self.schedule_change)
            widget.editingFinished.connect(self.update_draft)
        form.addRow('Направление', directions)
        self.kernel = QComboBox()
        for key, label in [('directed', 'Направленное'), ('uniform', 'Равномерное'), ('wall', 'Пристенный слой')]: self.kernel.addItem(label, key)
        self.kernel.currentIndexChanged.connect(self.schedule_change)
        form.addRow('Ядро', self.kernel)
        self.wall = QComboBox()
        for key, label in [('free', 'Свободно'), ('inner', 'Внутренняя'), ('outer', 'Наружная')]: self.wall.addItem(label, key)
        self.wall.currentIndexChanged.connect(self.schedule_change)
        form.addRow('Стенка', self.wall)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(form_widget)
        layout.addWidget(scroll, 1)
        self.vector_label = QLabel()
        self.vector_label.setWordWrap(True)
        layout.addWidget(self.vector_label)
        self.status = QLabel()
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        actions = QHBoxLayout()
        self.add_button = QPushButton('Для выбранного объекта')
        self.add_button.clicked.connect(self.add_selected_source)
        self.apply_button = QPushButton('Применить воздух')
        self.apply_button.clicked.connect(self.apply)
        actions.addWidget(self.add_button); actions.addWidget(self.apply_button)
        layout.addLayout(actions)
        self.reset_button = QPushButton('Отменить воздушный черновик')
        self.reset_button.clicked.connect(lambda:self.set_context(force=True))
        secondary_actions=QHBoxLayout();secondary_actions.addWidget(self.reset_button)
        self.balance_button = QPushButton('Баланс / этапы E(z)…')
        self.balance_button.clicked.connect(editor.owner.open_air_balance)
        secondary_actions.addWidget(self.balance_button);layout.addLayout(secondary_actions)
        self.timer = QTimer(self)
        self.timer.setSingleShot(True)
        self.timer.timeout.connect(self.update_draft)

    def set_context(self, force=False):
        cfg = self.editor.cfg
        current = deepcopy(cfg.air_model)
        signature = (current, deepcopy(cfg.nozzle_presets))
        if not force and self._loaded == signature:
            return
        identity = self.source_selector.currentData()
        self._loaded = signature
        self.draft = deepcopy(current)
        self.draft.setdefault('sources', [])
        self._loading = True
        self.source_selector.clear()
        for source in self.draft['sources']:
            binding = f" · форсунка {source['nozzle_id']}" if source.get('nozzle_id') is not None else f" · {source.get('object_id', source.get('kind', ''))}"
            self.source_selector.addItem(source['id'] + binding, source['id'])
        selected = self.source_selector.findData(identity)
        self.source_selector.setCurrentIndex(max(0, selected))
        self._loading = False
        self.source_selected()

    def selected_raw(self):
        identity = self.source_selector.currentData()
        return next((s for s in (self.draft or {}).get('sources', []) if s['id'] == identity), None)

    def effective_source(self, raw):
        try:
            from .air_templates import effective_air_source
        except ImportError:
            return normalize_source(raw)
        return normalize_source(effective_air_source(raw, self.editor.cfg, self.editor.owner.nozzles))

    def source_selected(self, *args):
        if self._loading:
            return
        self.timer.stop()
        raw = self.selected_raw()
        self._loading = True
        try:
            widgets = [self.enabled, self.method, self.kernel, self.wall, *self.controls.values(), *self.direction]
            for widget in widgets: widget.setEnabled(raw is not None)
            if raw is None:
                self.position.setText('Выберите объект на эскизе и добавьте воздушный источник либо задайте источники в «Баланс воздуха».')
                self._displayed = {}
            else:
                source = self.effective_source(raw)
                self.enabled.setChecked(source['enabled'])
                for control, key in [(self.method, 'method'), (self.kernel, 'kernel'), (self.wall, 'wall')]: control.setCurrentIndex(control.findData(source[key]))
                for key, control in self.controls.items(): control.setText('' if source[key] is None else format(source[key], '.12g'))
                for widget, value in zip(self.direction, source['direction']): widget.setText(format(value, '.12g'))
                self._displayed = self.read_controls()
                self.update_parameter_availability()
        except (ValueError, TypeError) as exc:
            self.status.setText(str(exc))
        finally:
            self._loading = False
        self.refresh_preview()

    def read_controls(self):
        values = dict(enabled=self.enabled.isChecked(), method=self.method.currentData(), kernel=self.kernel.currentData(), wall=self.wall.currentData())
        for key, widget in self.controls.items():
            text = widget.text().strip().replace(',', '.')
            values[key] = None if not text and key in OPTIONAL else float(text)
        values['direction'] = [float(w.text().strip().replace(',', '.')) for w in self.direction]
        return values

    def update_parameter_availability(self):
        uniform=self.kernel.currentData()=='uniform' or (self.draft or {}).get('distribution')=='uniform'
        for key in ('sigma_r_mm','sigma_theta_deg','spread_mm_per_mm'):
            self.controls[key].setEnabled(not uniform and self.selected_raw() is not None)
            self.controls[key].setToolTip('При равномерном распределении эти ширины не используются; пунктир показывает границы тракта.' if uniform else FIELDS[key][1])

    def schedule_change(self, *args):
        if not self._loading: self.timer.start(90)

    def update_draft(self):
        if self._loading or self.selected_raw() is None:
            return
        self.timer.stop()
        try:
            values = self.read_controls()
            changed = {key: value for key, value in values.items() if self._displayed.get(key) != value}
            if not changed:
                self.refresh_preview()
                return
            candidate = deepcopy(self.selected_raw())
            candidate.update(changed)
            if candidate.get('template_mode') == 'inherit':
                from .air_templates import AIR_TEMPLATE_FIELDS
                candidate['air_overrides'] = sorted(set(candidate.get('air_overrides', [])) | (set(changed) & set(AIR_TEMPLATE_FIELDS)))
            self.effective_source(candidate)
            self.selected_raw().update(candidate)
            self._displayed = deepcopy(values)
            self.update_parameter_availability()
            self.refresh_preview()
        except (ValueError, TypeError) as exc:
            self.status.setText('Параметры не применены: ' + str(exc))
            self.apply_button.setEnabled(False)

    def select_source(self, identity):
        index = self.source_selector.findData(identity)
        if index >= 0: self.source_selector.setCurrentIndex(index)

    def select_geometry_source(self, identity):
        obj = next((o for o in self.editor.raw['objects'] if o['id'] == identity), None)
        source = next((s for s in (self.draft or {}).get('sources', []) if s.get('object_id') == identity or obj and obj['type'] == 'nozzle' and s.get('nozzle_id') == obj.get('nozzle_id')), None)
        if source is not None: self.select_source(source['id'])

    def add_selected_source(self):
        obj = next((o for o in self.editor.raw['objects'] if o['id'] == self.editor.selected), None)
        if obj is None:
            self.status.setText('Сначала выберите форсунку, отверстие или вставку на эскизе.')
            return
        self.select_geometry_source(obj['id'])
        if any(s.get('object_id') == obj['id'] or obj['type'] == 'nozzle' and s.get('nozzle_id') == obj.get('nozzle_id') for s in self.draft['sources']):
            return
        identity = 'air:' + obj['id']
        source = dict(id=identity, kind='swirler' if obj['type'] == 'nozzle' else obj['type'], method='unknown', enabled=True, provenance={'source':'Sketcher: явно добавленный источник, расход неизвестен'})
        if obj['type'] == 'nozzle':
            source['nozzle_id'] = obj['nozzle_id']
            source.update(template_mode='inherit',air_overrides=[])
            nozzle=next(n for n in self.editor.owner.nozzles if n.id==obj['nozzle_id'])
            source['enabled']=nozzle.enabled
        else: source['object_id'] = obj['id']
        self.draft['sources'].append(source)
        self._loading = True
        self.source_selector.addItem(identity, identity)
        self._loading = False
        self.select_source(identity)

    def refresh_preview(self):
        if self.draft is None:
            return
        self.paths = None
        self.sources = []
        self.flows = {}
        self.last_diagnostic = ''
        try:
            model = validate_air_model(self.draft)
            self.sources = list(resolve_sources(self.draft, self.editor.cfg, self.editor.owner.nozzles))
            model['sources'] = self.sources
            balance = solve_balance(model)
            self.flows = balance.flows
            if model['enabled']:
                self.paths = cached_paths(self.sources, self.flows, self.editor.cfg, model)
                self.last_diagnostic = f'Известно {balance.assigned_kg_s:.6g} кг/с; не распределено {balance.unassigned_kg_s:.6g} кг/с.'
            else:
                self.last_diagnostic = 'Модель воздуха выключена. Показано только направление ввода, без расчётного потока.'
            self.apply_button.setEnabled(True)
        except (ValueError, TypeError, KeyError) as exc:
            self.last_diagnostic = 'Расчётные векторы недоступны: ' + str(exc)
            self.apply_button.setEnabled(not self.draft.get('enabled', False))
            # Invalid budgets do not manufacture a set of fictitious weights.
            try: self.sources = list(resolve_sources(self.draft, self.editor.cfg, self.editor.owner.nozzles))
            except (ValueError, TypeError, KeyError): self.sources = []
        legend='Пунктир — ±1σ геометрии ядра, не граница реального воздуха. Стрелки — направление, не скорость.'
        self.status.setText(self.last_diagnostic)
        self.status.setToolTip(legend)
        self.show_mixed.setToolTip(legend+' Для смешанного воздуха σ умножается на √2, как в матрицах.')
        self.draw_overlay()

    def clear_overlay(self):
        scene = self.editor.sketch.scene()
        for item in self.items:
            try:
                if item.scene() is scene: scene.removeItem(item)
            except RuntimeError:
                pass  # scene.clear() already destroyed it during view change.
        self.items = []

    def add_path(self, points, identity, role, color, dashed=False, width=1.5):
        if len(points) < 2 or not np.isfinite(points).all(): return
        path = QPainterPath(QPointF(*points[0]))
        for point in points[1:]: path.lineTo(*point)
        item = AirPathItem(self, identity, path, role)
        pen = QPen(QColor(color), width, Qt.DashLine if dashed else Qt.SolidLine)
        pen.setCosmetic(True)
        item.setPen(pen)
        item.setToolTip(identity + (' · ±1σ ядра' if role == 'air-envelope' else ' · расчётное направление' if role == 'air-vector' else ' · направление ввода, расход неизвестен/модель выключена' if role == 'air-input-preview' else ' · эмпирическая траектория'))
        self.editor.sketch.scene().addItem(item)
        self.items.append(item)

    def add_arrow(self, point, direction, identity, role, color, length, perpendicular=0., component='dz'):
        norm = float(np.linalg.norm(direction))
        if norm < 1e-10:
            glyph='⊙' if perpendicular>=0 else '⊗'
            label = AirGlyphItem(self,identity,glyph+(' '+identity if identity==self.source_selector.currentData() else ''))
            label.setFlag(QGraphicsItem.ItemIgnoresTransformations)
            label.setPos(QPointF(*point)); label.setBrush(QColor(color))
            label.setData(0, identity)
            label.setData(1, role); label.setData(2, float(perpendicular))
            label.setToolTip(f'{identity} · направление перпендикулярно проекции: ⊙ = +{component}, ⊗ = −{component}. Пространственные компоненты указаны в панели.')
            self.editor.sketch.scene().addItem(label); self.items.append(label)
            return
        vector = np.asarray(direction)/norm
        end = point + vector*length
        normal = np.array([-vector[1], vector[0]])
        head = min(4., length*.25)
        self.add_path([point, end, end-vector*head+normal*head*.5, end, end-vector*head-normal*head*.5], identity, role, color, dashed=role == 'air-input-preview', width=2.)

    def draw_overlay(self, *args):
        self.clear_overlay()
        if not self.editor.is_air_mode or not self.editor.isVisible():
            return
        angular = self.editor.angular.isChecked()
        selected = self.source_selector.currentData()
        from .chamber import chamber
        self._bounds_model=chamber(self.editor.cfg)
        self.vector_label.setText('')
        path_indices = {s['id']: i for i, s in enumerate(self.paths.sources)} if self.paths is not None else {}
        for source in self.sources:
            identity = source['id']
            if not self.show_all.isChecked() and identity != selected: continue
            active = identity in path_indices and self.flows.get(identity, 0.) > 0
            color = '#1686e0' if identity == selected else '#5ba3d9'
            if active:
                index = path_indices[identity]
                end = min(int(self.paths.stop_steps[index])+1, len(self.paths.path_mm))
                points = self.paths.positions[:end, index]
                distance = self.paths.path_mm[:end]
                # Display the physical Sketcher extent. Calculations retain the
                # configured full path horizon, independent of this clipping.
                inside = (points[:, 0] >= self.editor.raw['base_z_mm']-1e-7) & (points[:, 0] <= self.editor.raw['outlet_z_mm']+1e-7)
                keep = np.flatnonzero(inside)
                if not len(keep): continue
                keep = keep[np.unique(np.linspace(0, len(keep)-1, min(80, len(keep))).astype(int))]
                points = points[keep]; distance = distance[keep]
                projection = project_points(points, angular)
                self.add_path(projection, identity, 'air-trajectory', color, width=2.2 if identity == selected else 1.)
                for envelope in self.envelope_curves(points, distance, source, angular): self.add_path(envelope, identity, 'air-envelope', color, dashed=True, width=1.)
                middle = len(points)//2
                vector = self.paths.directions[keep[middle], index]
                theta = math.degrees(math.atan2(points[middle, 2], points[middle, 1]))
                local = cartesian_to_local(vector, theta)
                direction = np.array([vector[1], -vector[2]]) if angular else np.array([local[0], -local[1]])
                self.add_arrow(projection[middle], direction, identity, 'air-vector', color, 15., vector[0] if angular else local[2], 'dz' if angular else 'dθ')
                if identity == selected:
                    self.vector_label.setText(f'Результирующее направление [dz, dr, dθ] = [{local[0]:.5g}, {local[1]:.5g}, {local[2]:.5g}]\nG={self.flows[identity]:.6g} кг/с; остановка: {self.paths.stop_reasons[index]}.')
                    if np.linalg.norm(direction)<1e-10:self.vector_label.setText(self.vector_label.text()+f"\nНулевая проекция: ⊙ = +{'dz' if angular else 'dθ'}, ⊗ = −{'dz' if angular else 'dθ'}.")
            else:
                theta = math.radians(source['theta_deg'])
                origin = np.array([source['z_mm'], source['radius_mm']*math.cos(theta), source['radius_mm']*math.sin(theta)])
                vector = local_to_cartesian(source['direction'], source['theta_deg'])
                points = np.array([origin, origin+vector*20.])
                projection = project_points(points, angular)
                local=source['direction']
                direction=np.array([vector[1],-vector[2]]) if angular else np.array([local[0],-local[1]])
                self.add_arrow(projection[0], direction, identity, 'air-input-preview', '#b78125', 15., vector[0] if angular else local[2], 'dz' if angular else 'dθ')
                for envelope in self.envelope_curves(points, [0., 20.], source, angular): self.add_path(envelope, identity, 'air-envelope', '#b78125', dashed=True, width=1.)
                if identity == selected:
                    self.vector_label.setText('Только направление ввода [dz, dr, dθ] = ' + str(source['direction']) + '\nРасчётного расхода/результирующей струи нет.')
                    if np.linalg.norm(direction)<1e-10:self.vector_label.setText(self.vector_label.text()+f"\nНулевая проекция: ⊙ = +{'dz' if angular else 'dθ'}, ⊗ = −{'dz' if angular else 'dθ'}.")
            if identity == selected:
                self.position.setText(f"{source['kind']} · z={source['z_mm']:.6g} мм, R={source['radius_mm']:.6g} мм, θ={source['theta_deg']:.6g}°")
                self.position.setToolTip('Координаты геометрии заморожены. Пунктир показывает ±1σ ядра, а не границу реального воздуха; стрелки — направление, не скорость.')
                if source['kernel']=='uniform' or self.draft.get('distribution')=='uniform':self.position.setText(self.position.text()+' Равномерное ядро: пунктир показывает границы тракта.')
        if self.selected_raw() is not None and not self.selected_raw().get('enabled',True):
            self.vector_label.setText('Источник отключён; расчётного потока и векторов нет.')
            self.position.setText('Параметры отключённого источника сохраняются. Геометрия заморожена.')
        self.editor.sketch.viewport().update()

    def envelope_curves(self, points, distance, source, angular):
        """Clip sigma bounds to the same chamber domain as the field kernel."""
        if self._bounds_model is None:
            bounds=np.tile([self.editor.cfg.r_in_mm,self.editor.cfg.r_out_mm],(len(points),1))
        else:
            bounds=np.asarray([self._bounds_model.bounds(float(np.clip(point[0], self.editor.raw['base_z_mm'], self.editor.raw['outlet_z_mm']))) for point in points])
        if source['kernel'] == 'uniform' or self.draft.get('distribution')=='uniform':
            if angular:
                angle = np.linspace(0., 2*np.pi, 100)
                return [np.column_stack((radius*np.cos(angle), -radius*np.sin(angle)))
                        for radius in (bounds[:, 0].min(), bounds[:, 1].max())]
            return [np.column_stack((points[:, 0], -bounds[:, column])) for column in (0, 1)]
        curves = projected_envelopes(points, distance, source, angular, self.show_mixed.isChecked())
        for curve in curves:
            if angular:
                radius = np.hypot(curve[:, 0], curve[:, 1])
                clipped = np.clip(radius, bounds[:, 0], bounds[:, 1])
                curve *= np.divide(clipped, radius, out=np.ones_like(radius), where=radius > 0)[:, None]
            else:
                curve[:, 1] = -np.clip(-curve[:, 1], bounds[:, 0], bounds[:, 1])
        return curves

    def apply(self):
        self.update_draft()
        if not self.apply_button.isEnabled(): return
        self.editor.owner.apply_air_model(deepcopy(self.draft))
        self.set_context(force=True)
