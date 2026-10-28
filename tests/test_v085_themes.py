"""Readable state colors are checked on the native widgets, not only CSS."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from PyQt5.QtCore import QSettings, Qt
from PyQt5.QtGui import QColor, QPalette, QPen
from PyQt5.QtWidgets import (QApplication, QCheckBox, QComboBox, QGraphicsScene,
    QGraphicsView, QLabel, QLineEdit, QPushButton, QTabWidget, QTableWidget,
    QTableWidgetItem, QVBoxLayout, QWidget, QStyle, QStyleOptionButton)
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.colors import to_rgba
from matplotlib.figure import Figure

from fuelfield.themes import (PALETTES, THEME_LABELS, apply_theme,
    current_theme, load_theme)


def contrast(a, b):
    def luminance(value):
        rgb = QColor(value).getRgbF()[:3]
        linear = [v/12.92 if v <= .04045 else ((v+.055)/1.055)**2.4 for v in rgb]
        return sum(w*v for w, v in zip((.2126, .7152, .0722), linear))
    first, second = sorted((luminance(a), luminance(b)))
    return (second+.05)/(first+.05)


class TestThreeReadableThemes(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.previous = current_theme()
        self.folder = tempfile.TemporaryDirectory()
        settings = QSettings(str(Path(self.folder.name)/'appearance.ini'), QSettings.IniFormat)
        self.settings = settings
        self.preference_patch = patch('fuelfield.themes.preferences', return_value=settings)
        self.preference_patch.start()
        self.widgets = []

    def tearDown(self):
        for widget in self.widgets:
            widget.close(); widget.deleteLater()
        self.app.processEvents()
        apply_theme(self.app, self.previous, persist=False)
        self.preference_patch.stop()
        self.folder.cleanup()

    def test_text_and_selection_contrast_and_original_standard_light(self):
        self.assertEqual(set(PALETTES), {'light', 'light_green', 'dark'})
        self.assertEqual(set(THEME_LABELS), set(PALETTES))
        original = dict(window='#f4f6fa', panel='#ffffff', base='#ffffff',
            alternate='#eef2f8', text='#202b3b', muted='#52617c', border='#c8d0dc',
            accent='#386ec5', guide='#357b72', pin='#a35721')
        self.assertEqual({key: PALETTES['light'][key] for key in original}, original)
        self.assertEqual(PALETTES['light_green']['accent'], '#4caf4f')
        self.assertEqual(PALETTES['light_green']['accent_strong'], '#237d31')
        self.assertEqual(PALETTES['light_green']['window'], '#f5f7fa')
        self.assertEqual(PALETTES['light_green']['text'], '#263238')
        self.assertEqual(PALETTES['dark']['accent'], '#ea1763')
        self.assertEqual(PALETTES['dark']['window'], '#000000')
        self.assertEqual(PALETTES['dark']['panel'], '#252525')
        for name, palette in PALETTES.items():
            for surface in ('base', 'panel', 'alternate', 'window'):
                self.assertGreaterEqual(contrast(palette['text'], palette[surface]), 7., (name, surface))
                self.assertGreaterEqual(contrast(palette['muted'], palette[surface]), 4.5, (name, surface))
            for foreground, background in (('selection_text', 'accent'),
                    ('tab_text', 'tab'), ('tab_selected_text', 'tab_selected'),
                    ('disabled_text', 'disabled_panel')):
                self.assertGreaterEqual(contrast(palette[foreground], palette[background]), 4.5,
                                        (name, foreground, background))
            self.assertGreaterEqual(contrast(palette['link'], palette['panel']), 4.5, name)
            self.assertGreaterEqual(contrast(palette['selection_text'], palette['disabled_checked']), 4.5, name)

    def test_native_tabs_have_visible_selected_inactive_and_disabled_surfaces(self):
        tabs = QTabWidget(); self.widgets.append(tabs)
        for label in ('Геометрия', 'Воздух', 'Недоступно'):
            tabs.addTab(QWidget(), label)
        tabs.setTabEnabled(2, False)
        tabs.resize(600, 260); tabs.show()
        for name in ('light_green', 'dark'):
            apply_theme(self.app, name, persist=False); self.app.processEvents()
            bar = tabs.tabBar(); image = bar.grab().toImage()
            backgrounds = []
            for index, key in enumerate(('tab_selected', 'tab', 'disabled_panel')):
                rect = bar.tabRect(index)
                # The sample is inside the actual native tab, away from text,
                # the accent underline, the rounded edges and the tab gap.
                ratio = bar.devicePixelRatioF()
                actual = image.pixelColor(round((rect.left()+5)*ratio), round((rect.top()+12)*ratio))
                expected = QColor(PALETTES[name][key])
                self.assertEqual(actual.name(), expected.name(), (name, index))
                backgrounds.append(actual.name())
                foreground = PALETTES[name][('tab_selected_text', 'tab_text', 'disabled_text')[index]]
                self.assertGreaterEqual(contrast(foreground, actual), 4.5)
            self.assertEqual(len(set(backgrounds)), 3)

    def test_table_selection_and_enabled_disabled_inputs_remain_readable(self):
        window = QWidget(); self.widgets.append(window); layout = QVBoxLayout(window)
        field = QLineEdit('Параметры воздушного потока'); layout.addWidget(field)
        disabled = QLineEdit('Координаты из геометрии'); disabled.setEnabled(False); layout.addWidget(disabled)
        combo = QComboBox(); combo.addItems(['Заданный расход', 'Доля']); layout.addWidget(combo)
        table = QTableWidget(2, 2); table.setHorizontalHeaderLabels(['Режим', 'G, кг/с'])
        for row in range(2):
            for column in range(2): table.setItem(row, column, QTableWidgetItem(str(row+column)))
        table.setCurrentCell(0, 0); table.selectRow(0); layout.addWidget(table)
        button = QPushButton('Применить'); layout.addWidget(button)
        window.resize(540, 300); window.show()
        for name in ('light_green', 'dark'):
            apply_theme(self.app, name, persist=False); self.app.processEvents()
            palette = PALETTES[name]
            self.assertEqual(table.palette().highlight().color().name(), palette['accent'])
            self.assertEqual(table.palette().highlightedText().color().name(), palette['selection_text'])
            self.assertGreaterEqual(contrast(field.palette().text().color(), field.palette().base().color()), 7.)
            self.assertGreaterEqual(contrast(disabled.palette().color(QPalette.Disabled, QPalette.Text),
                disabled.palette().color(QPalette.Disabled, QPalette.Base)), 4.5)
            self.assertTrue(button.isEnabled())

    def test_sketch_modes_survive_theme_changes_and_new_axes_match_theme(self):
        view = QGraphicsView(QGraphicsScene()); self.widgets.append(view)
        view.setStyleSheet('QGraphicsView { border:1px solid #d9dfeb; border-radius:4px; }')
        view.show()
        figure = Figure(); canvas = FigureCanvasQTAgg(figure); self.widgets.append(canvas)
        for mode, expected in (('geometry', '#28a35d'), ('air', '#2c85dc')):
            view.setProperty('ffSketchMode', mode)
            for name in ('dark', 'light_green', 'light'):
                apply_theme(self.app, name, persist=False); self.app.processEvents()
                self.assertIn('border:3px solid '+expected, view.styleSheet())
                self.assertEqual(view.backgroundBrush().color().name(), PALETTES[name]['base'])
                axes = figure.add_subplot(); axes.set_title('Поле α'); axes.plot([0., 1.], [2., 3.])
                canvas.draw()
                self.assertEqual(axes.get_facecolor(), to_rgba(PALETTES[name]['panel']))
                self.assertEqual(axes.xaxis.label.get_color(), PALETTES[name]['text'])
                figure.clear()

    def test_green_preference_roundtrip_uses_the_third_theme_key(self):
        apply_theme(self.app, 'light_green', persist=True)
        self.assertEqual(self.settings.value('appearance/theme'), 'light_green')
        apply_theme(self.app, 'light', persist=False)
        self.assertEqual(load_theme(self.app), 'light_green')

    def test_native_checkbox_keeps_a_visible_checkmark_and_boundary(self):
        checkbox=QCheckBox('Включить воздух'); self.widgets.append(checkbox)
        checkbox.resize(200,32); checkbox.show()
        for name in ('light_green','dark'):
            apply_theme(self.app,name,persist=False); checkbox.setChecked(True); self.app.processEvents()
            option=QStyleOptionButton(); option.initFrom(checkbox)
            rect=checkbox.style().subElementRect(QStyle.SE_CheckBoxIndicator,option,checkbox)
            image=checkbox.grab().toImage(); ratio=checkbox.devicePixelRatioF()
            target=QColor(PALETTES[name]['selection_text'])
            count=0
            for y in range(round(rect.top()*ratio),round((rect.bottom()+1)*ratio)):
                for x in range(round(rect.left()*ratio),round((rect.right()+1)*ratio)):
                    actual=image.pixelColor(x,y)
                    if max(abs(a-b) for a,b in zip(actual.getRgb()[:3],target.getRgb()[:3]))<8:count+=1
            self.assertGreater(count,8,(name,'checkmark missing'))
            checkbox.setChecked(False); self.app.processEvents()
            self.assertGreaterEqual(contrast(PALETTES[name]['accent_strong'],PALETTES[name]['panel']),3.)

    def test_scene_overlay_opacity_line_width_and_dash_survive_theme_switches(self):
        scene=QGraphicsScene(); view=QGraphicsView(scene); self.widgets.append(view)
        color=QColor('#316cb7'); color.setAlpha(96)
        pen=QPen(color, 3.5, Qt.DashLine)
        line=scene.addLine(0., 0., 100., 40., pen); view.show()
        apply_theme(self.app, 'dark', persist=False); self.app.processEvents()
        self.assertEqual(line.pen().color().alpha(), 96)
        self.assertEqual(line.pen().widthF(), 3.5)
        self.assertEqual(line.pen().style(), Qt.DashLine)
        apply_theme(self.app, 'light_green', persist=False); self.app.processEvents()
        self.assertEqual(line.pen().color(), color)
        self.assertEqual(line.pen().widthF(), 3.5)
        self.assertEqual(line.pen().style(), Qt.DashLine)

    def test_inherited_colored_warning_and_equation_labels_are_readable_in_dark(self):
        window=QWidget(); self.widgets.append(window); layout=QVBoxLayout(window)
        labels=[]
        for original in ('#b02838', '#67458b', '#0000ff', '#52617c'):
            label=QLabel('Проверка параметров'); label.setStyleSheet('color:'+original+';')
            layout.addWidget(label); labels.append((label, original))
        window.show(); apply_theme(self.app, 'dark', persist=False); self.app.processEvents()
        for label, original in labels:
            actual=label.palette().windowText().color()
            self.assertGreaterEqual(contrast(actual, PALETTES['dark']['alternate']), 4.5, original)
        apply_theme(self.app, 'light', persist=False); self.app.processEvents()
        for label, original in labels:
            self.assertEqual(label.palette().windowText().color().name(), original)


if __name__ == '__main__':
    unittest.main()
