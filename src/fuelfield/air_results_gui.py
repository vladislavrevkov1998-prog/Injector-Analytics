"""Air/alpha views derived from existing fuel results; never recompute fuel."""
from copy import deepcopy
import json

import numpy as np
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtWidgets import (QComboBox, QDialog, QDialogButtonBox, QFileDialog,
                             QHBoxLayout, QLabel, QMessageBox, QPushButton, QSlider,
                             QTabWidget, QTextBrowser, QVBoxLayout, QWidget, QTableWidget, QTableWidgetItem, QAbstractItemView)
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg

from .air_io import _plain, write_air_csv, write_air_xlsx
from .coordinates import from_global
from .mixture_field import compute_mixture
from .air_distribution import compute_air_field


METRICS = {
    'fuel_kg_s': ('Топливо, доступная взвесь', 'кг/с'),
    'air_total_kg_s': ('Полный учтённый воздух', 'кг/с'),
    'air_core_kg_s': ('Ядро струй', 'кг/с'),
    'air_mixed_kg_s': ('Условно вовлечённый воздух', 'кг/с'),
    'reverse_air_kg_s': ('Обратный воздух — вне положительной матрицы', 'кг/с'),
    'alpha_total': ('α по полному учтённому воздуху', 'безразмерно'),
    'alpha_mixed': ('α по условно вовлечённому воздуху', 'безразмерно'),
}


def _fmt(value):
    return f'{value:.7g}' if value is not None and np.isfinite(value) else 'не определён'


class AirResultsDialog(QDialog):
    def __init__(self, cfg, nozzles, planes, quasi_result=None, parent=None):
        super().__init__(parent)
        if not planes:
            raise ValueError('Сначала рассчитайте топливные сечения (F5).')
        if not getattr(cfg, 'air_model', {}).get('enabled', False):
            raise ValueError('Включите и примените модель воздуха в редакторе баланса.')
        self.cfg = deepcopy(cfg); self.nozzles = deepcopy(nozzles)
        self.planes = tuple(planes); self.quasi_result = quasi_result
        self._cache = {}; self._air_fields = {}; self.current_result = None; self.current_plane = None
        self.setWindowTitle('Воздух и расходные оценки α — v0.8'); self.resize(1120, 810)
        layout = QVBoxLayout(self)
        warning = QLabel('α — условное отношение соотнесённых расходов, без расчёта реакции или температуры. '
                         'Топливная матрица сохранена; стеночные плёнки показаны отдельно и исключены из знаменателя α.')
        warning.setWordWrap(True); layout.addWidget(warning)
        row = QHBoxLayout(); self.section = QComboBox(); self.metric = QComboBox()
        for i, p in enumerate(self.planes): self.section.addItem(f'z={p.z_mm:g} мм', i)
        for key, (label, unit) in METRICS.items(): self.metric.addItem(label + ' [' + unit + ']', key)
        self.scope = QComboBox()
        if quasi_result is not None:
            self.scope.addItem('Мгновенный снимок: один реальный шаг', 'instantaneous')
            self.scope.addItem('Составная карта: разные времена по столбцам', 'composite')
        else:
            self.scope.addItem('Стационарный расчёт', 'instantaneous')
        row.addWidget(QLabel('Сечение:')); row.addWidget(self.section)
        row.addWidget(self.scope, 1); row.addWidget(self.metric, 1); layout.addLayout(row)
        time_row = QHBoxLayout(); self.time_slider = QSlider(Qt.Horizontal); self.time_label = QLabel()
        self.time_slider.setRange(0, quasi_result.settings.n_steps - 1 if quasi_result is not None else 0)
        time_row.addWidget(QLabel('Шаг:')); time_row.addWidget(self.time_slider, 1); time_row.addWidget(self.time_label)
        self.time_panel = QWidget(); self.time_panel.setLayout(time_row); self.time_panel.setVisible(quasi_result is not None)
        layout.addWidget(self.time_panel)
        self.summary = QLabel(); self.summary.setWordWrap(True); layout.addWidget(self.summary)
        tabs = QTabWidget(); self.figure = Figure(figsize=(9, 5), tight_layout=True)
        self.canvas = FigureCanvasQTAgg(self.figure); tabs.addTab(self.canvas, 'Матрица')
        self.source_table = QTableWidget(0, 9)
        self.source_table.setHorizontalHeaderLabels(['Источник', 'Тип', 'Вход, кг/с', 'Прошёл, кг/с', 'Обратный, кг/с',
                                                   'Не пересёк, кг/с', 'Вне тракта, кг/с', 'Доля вовлечения', 'Состояние'])
        self.source_table.setEditTriggers(QAbstractItemView.NoEditTriggers); tabs.addTab(self.source_table, 'Расходы источников')
        self.details = QTextBrowser(); tabs.addTab(self.details, 'Происхождение и ограничения'); layout.addWidget(tabs, 1)
        self.status = QLabel(); self.status.setWordWrap(True); layout.addWidget(self.status)
        actions = QHBoxLayout()
        for label, handler in [('CSV: все z этого состояния', self.export_csv), ('XLSX: все z этого состояния', self.export_xlsx), ('PNG: текущая карта', self.export_png)]:
            button = QPushButton(label); button.clicked.connect(handler); actions.addWidget(button)
        actions.addStretch(); layout.addLayout(actions)
        buttons = QDialogButtonBox(QDialogButtonBox.Close); buttons.button(QDialogButtonBox.Close).setText('Закрыть')
        buttons.rejected.connect(self.reject); layout.addWidget(buttons)
        self.timer = QTimer(self); self.timer.setSingleShot(True); self.timer.setInterval(16); self.timer.timeout.connect(self.refresh)
        self.section.currentIndexChanged.connect(self.refresh); self.metric.currentIndexChanged.connect(self.refresh)
        self.scope.currentIndexChanged.connect(self.refresh); self.time_slider.valueChanged.connect(lambda: self.timer.start())
        self.refresh()

    def _state(self, index):
        composite = self.scope.currentData() == 'composite'
        step = self.time_slider.value()
        key = (index, None if composite or self.quasi_result is None else step)
        if key not in self._cache:
            plane = self.planes[index] if self.quasi_result is None or composite else self.quasi_result.snapshot(index, step)
            if index not in self._air_fields:
                self._air_fields[index] = compute_air_field(plane, self.cfg, self.nozzles)
            result = compute_mixture(plane, self.cfg, self.nozzles, air_result=self._air_fields[index])
            if result is None: raise ValueError('Модель воздуха отключена.')
            self._cache[key] = (plane, result)
            # Bound the retained time-series cache; exports compute other z lazily.
            if len(self._cache) > 32:
                oldest = next(iter(self._cache))
                if oldest != key: self._cache.pop(oldest)
        return self._cache[key]

    def refresh(self, *_):
        self.timer.stop()
        try:
            index = max(0, self.section.currentIndex()); plane, result = self._state(index)
            self.current_plane = plane; self.current_result = result
            is_composite = bool(getattr(plane, 'is_composite', False))
            self.time_slider.setEnabled(self.quasi_result is not None and not is_composite)
            if self.quasi_result is not None:
                step = self.time_slider.value(); stamp = self.quasi_result.settings.times[step]
                self.time_label.setText('разные времена по столбцам' if is_composite else f'{step + 1}/{self.quasi_result.settings.n_steps}; t={stamp:g} с')
            balance = result.balance
            prefix = 'Составная карта: единый мгновенный баланс и α сечения не определены. ' if is_composite else ''
            self.summary.setText(prefix + f'Воздух на сечении: {np.sum(result.air_total_kg_s):.7g} кг/с; '
                                 f'вовлечён: {np.sum(result.air_mixed_kg_s):.7g} кг/с. '
                                 f'Нераспределённый остаток: {balance.unassigned_kg_s:.7g} кг/с. '
                                 f'α сечения: {_fmt(None if is_composite else result.alpha_section_total)}; '
                                 f'α вовлечённого: {_fmt(None if is_composite else result.alpha_section_mixed)}. '
                                 f'Неопределённая площадь α: {100 * result.undefined_area_fraction:.5g}%.')
            info = dict(balance=_plain(balance), source_states=_plain(getattr(result, 'source_states', [])),
                        diagnostics=_plain(result.diagnostics), provenance=_plain(result.provenance),
                        scope='composite_different_times_no_global_balance' if is_composite else 'instantaneous',
                        fuel_mode=getattr(self.cfg, 'fuel_transport', {'mode': 'legacy'}),
                        film_inner_kg_s=float(np.sum(getattr(plane, 'film_inner_cell_mass_kg_s', None))) if getattr(plane, 'film_inner_cell_mass_kg_s', None) is not None else 0.,
                        film_outer_kg_s=float(np.sum(getattr(plane, 'film_outer_cell_mass_kg_s', None))) if getattr(plane, 'film_outer_cell_mass_kg_s', None) is not None else 0.)
            if self.quasi_result is not None and not is_composite:
                info['pressures_kgf_cm2'] = {key: float(values[self.time_slider.value()])
                                              for key, values in getattr(self.quasi_result, 'pressures', {}).items()}
            self.details.setPlainText(json.dumps(info, ensure_ascii=False, indent=2, allow_nan=False))
            states = getattr(result, 'source_states', []); self.source_table.setRowCount(len(states))
            labels = dict(forward='прямой', reverse='обратный', not_crossed='не пересёк', wall_exit='вышел через стенку',
                          multiple_crossings='неоднозначное пересечение')
            for row, state in enumerate(states):
                values = [state.get('id', ''), state.get('kind', ''),
                          *[_fmt(state.get(key)) for key in ('assigned_kg_s', 'crossed_kg_s', 'reverse_kg_s',
                                                            'not_crossed_kg_s', 'outside_kg_s', 'E')],
                          labels.get(state.get('status'), state.get('status', ''))]
                for col, value in enumerate(values): self.source_table.setItem(row, col, QTableWidgetItem(str(value)))
            self.source_table.resizeColumnsToContents()
            self._draw(plane, result)
            self.status.setText('Неопределённые значения α показаны серым: нет топлива, неизвестен L0 или локальная оценка недоступна. '
                                'Расходы воздуха соответствуют текущим заданным источникам; неизвестный остаток не создаёт искусственной струи.')
        except (ValueError, TypeError, KeyError) as exc:
            self.status.setText('Результат недоступен: ' + str(exc))

    def _draw(self, plane, result, *, export_light=False):
        self.figure.clear(); ax = self.figure.add_subplot(111)
        key = self.metric.currentData(); raw_values = getattr(result, key, None)
        values = np.zeros_like(result.fuel_kg_s) if raw_values is None else np.asarray(raw_values)
        label, unit = METRICS[key]
        cmap = __import__('matplotlib').colormaps['viridis'].with_extremes(bad='#a5a9b0')
        image = ax.pcolormesh(np.arange(values.shape[1] + 1), plane.radial_edges_mm,
                              np.ma.masked_invalid(values), shading='flat', cmap=cmap)
        indices = np.unique(np.linspace(0, values.shape[1] - 1, min(9, values.shape[1])).astype(int))
        measured = from_global(plane.theta_centers_deg, self.cfg)
        ax.set_xticks(indices + .5, labels=[f'{measured[j]:.5g}°' for j in indices])
        ax.set_xlabel('Окружной угол в измерительной системе; экспорт содержит также физический θ')
        ax.set_ylabel('Радиус, мм')
        ax.set_title(f'{label} [{unit}], z={plane.z_mm:g} мм' + (' — разные времена' if plane.is_composite else ''))
        self.figure.colorbar(image, ax=ax, label=unit)
        if export_light:
            self.figure.set_facecolor('white')
            for item in self.figure.axes:
                item.set_facecolor('white'); item.tick_params(colors='#20242a'); item.xaxis.label.set_color('#20242a')
                item.yaxis.label.set_color('#20242a'); item.title.set_color('#20242a')
                for spine in item.spines.values(): spine.set_edgecolor('#46505d')
        else:
            try:
                from .themes import style_canvas
                style_canvas(self.canvas)
            except ImportError:
                pass
        self.canvas.draw_idle()

    def export_bundle(self):
        """All axial planes for selected real step or an explicitly chosen composite."""
        planes, results = [], []
        for index in range(len(self.planes)):
            plane, result = self._state(index); planes.append(plane); results.append(result)
        times = None
        if self.quasi_result is not None and self.scope.currentData() != 'composite':
            times = [float(self.quasi_result.settings.times[self.time_slider.value()])] * len(planes)
        return results, planes, times

    def _export(self, kind):
        suffix = 'csv' if kind == 'csv' else 'xlsx'
        path, _ = QFileDialog.getSaveFileName(self, 'Экспорт воздуха и α', 'air_mixture.' + suffix, suffix.upper() + ' (*.' + suffix + ')')
        if not path: return
        try:
            results, planes, times = self.export_bundle()
            pressures = None
            if self.quasi_result is not None and self.scope.currentData() != 'composite':
                stamp = {key: float(values[self.time_slider.value()]) for key, values in self.quasi_result.pressures.items()}
                pressures = [stamp] * len(planes)
            (write_air_csv if kind == 'csv' else write_air_xlsx)(path, results, self.cfg, planes, times_s=times, pressures=pressures)
            self.status.setText('Сохранено: ' + path + '. Экспорт содержит только выбранное состояние во всех z.')
        except (OSError, ValueError, TypeError) as exc:
            QMessageBox.warning(self, 'Экспорт не выполнен', str(exc))

    def export_csv(self): self._export('csv')
    def export_xlsx(self): self._export('xlsx')

    def export_png(self):
        if self.current_result is None: return
        path, _ = QFileDialog.getSaveFileName(self, 'Карта воздуха / α', 'air_mixture.png', 'PNG (*.png)')
        if not path: return
        try:
            self._draw(self.current_plane, self.current_result, export_light=True)
            self.figure.savefig(path, dpi=160, facecolor='white')
            self.status.setText('Карта сохранена: ' + path)
        except (OSError, ValueError) as exc:
            QMessageBox.warning(self, 'PNG не сохранён', str(exc))
        finally:
            self._draw(self.current_plane, self.current_result)
