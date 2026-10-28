"""Application appearance; preferences are separate from project/physics data.

Canvas hooks style newly created axes before rendering. Exports temporarily use
the light palette, preserving readable text and the original data colormaps.
"""
import colorsys
import os
from pathlib import Path
import tempfile
import weakref
import re

from PyQt5 import sip
from PyQt5.QtCore import QObject,QEvent,QSettings,QTimer,Qt
from PyQt5.QtGui import QColor,QPalette,QBrush,QPen,QImage,QPainter,QPainterPath
from PyQt5.QtWidgets import QApplication,QGraphicsView,QTableWidget,QWidget
from matplotlib import rcParams
from matplotlib.colors import to_rgba,to_hex
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg

PALETTES={
    'light':dict(window='#f4f6fa',panel='#ffffff',base='#ffffff',alternate='#eef2f8',text='#202b3b',muted='#52617c',border='#c8d0dc',accent='#386ec5',guide='#357b72',pin='#a35721'),
    'light_green':dict(window='#f5f7fa',panel='#ffffff',base='#ffffff',alternate='#e8f5e9',text='#263238',muted='#53645a',border='#a5d6a7',accent='#4caf4f',guide='#237d31',pin='#935824'),
    'dark':dict(window='#000000',panel='#252525',base='#151515',alternate='#252525',text='#ffffff',muted='#8a8a9d',border='#646464',accent='#ea1763',guide='#ea1763',pin='#ff87ad'),
}

THEME_LABELS={
    'light':'Стандартная светлая',
    'light_green':'Светлая с зелёными акцентами',
    'dark':'Тёмная с акцентами',
}

# State-specific colors keep tab labels readable even without hover. The
# standard light theme continues to use the existing native Fusion surfaces.
# Figma reference tokens, verified by the integration from high-fidelity data:
# i3d4YfBKo43oR2CBOfKzff / 1:15: BrandPrimary #4CAF4F, shade #237D31,
# text #263238, neutral #F5F7FA and tints #E8F5E9/#C8E6C9/#A5D6A7.
# DKhLCBE25R7iiKPyuMDE9n / 1:142: #EA1763, white, black, #252525,
# #646464 and #8A8A9D. Dense text uses a contrasting ink on exact brand fills.
PALETTES['light'].update(selection_text='#ffffff',tab='#eef2f8',tab_text='#202b3b',
    tab_selected='#ffffff',tab_selected_text='#202b3b',disabled_panel='#eef2f8',disabled_text='#52617c',hover='#e5edf9',
    accent_strong='#386ec5',link='#386ec5',disabled_checked='#52617c')
PALETTES['light_green'].update(selection_text='#263238',tab='#e8f5e9',tab_text='#263238',
    tab_selected='#c8e6c9',tab_selected_text='#263238',disabled_panel='#f5f7fa',disabled_text='#53645a',hover='#e8f5e9',
    accent_strong='#237d31',link='#237d31',disabled_checked='#c8e6c9')
PALETTES['dark'].update(selection_text='#000000',tab='#252525',tab_text='#ffffff',
    tab_selected='#ea1763',tab_selected_text='#000000',disabled_panel='#151515',disabled_text='#8a8a9d',hover='#353535',
    accent_strong='#ea1763',link='#ff6a9b',disabled_checked='#8a8a9d')

_indicator_folder=None
_indicator_cache={}


def _indicator_images(values):
    """Use Qt-rendered marks, keeping checkboxes clear on both new palettes."""
    global _indicator_folder
    key=values['selection_text']
    if key in _indicator_cache:return _indicator_cache[key]
    if _indicator_folder is None:_indicator_folder=tempfile.TemporaryDirectory(prefix='injector-theme-icons-')
    paths=[]
    for partial in (False,True):
        image=QImage(16,16,QImage.Format_ARGB32_Premultiplied);image.fill(Qt.transparent)
        painter=QPainter(image);painter.setRenderHint(QPainter.Antialiasing)
        painter.setPen(QPen(QColor(key),3.,Qt.SolidLine,Qt.RoundCap,Qt.RoundJoin))
        path=QPainterPath();path.moveTo(3,8)
        if partial:path.lineTo(13,8)
        else:path.lineTo(6,11);path.lineTo(13,4)
        painter.drawPath(path);painter.end()
        target=Path(_indicator_folder.name)/(key.lstrip('#')+('-partial.png' if partial else '-checked.png'))
        if not image.save(str(target)):raise OSError('Cannot create checkbox theme indicator')
        paths.append(target.as_posix())
    _indicator_cache[key]=tuple(paths)
    return _indicator_cache[key]


def _stylesheet(values, name):
    tooltip='QToolTip { color: %(text)s; background: %(panel)s; border:1px solid %(border)s; }' % values
    if name=='light':return tooltip
    checked,partial=_indicator_images(values)
    values=dict(values,checked_mark=checked,partial_mark=partial)
    return tooltip+'''
QTabWidget::pane { background: %(panel)s; border: 1px solid %(border)s; border-radius: 4px; top: -1px; }
QTabBar::tab { color: %(tab_text)s; background: %(tab)s; border: 1px solid %(border)s;
    border-bottom: 3px solid %(border)s; padding: 6px 11px; margin-right: 3px; }
QTabBar::tab:hover { background: %(hover)s; }
QTabBar::tab:selected { color: %(tab_selected_text)s; background: %(tab_selected)s;
    border-bottom: 3px solid %(accent)s; font-weight: bold; }
QTabBar::tab:disabled { color: %(disabled_text)s; background: %(disabled_panel)s; border-color: %(border)s; }
QHeaderView::section { color: %(text)s; background: %(tab)s; border: 0;
    border-right: 1px solid %(border)s; border-bottom: 1px solid %(border)s; padding: 4px; }
QTableCornerButton::section { background: %(tab)s; border: 1px solid %(border)s; }
QTableView, QTreeView, QListView { color: %(text)s; background: %(base)s;
    alternate-background-color: %(alternate)s; gridline-color: %(border)s;
    selection-background-color: %(accent)s; selection-color: %(selection_text)s; }
QTableView::item:selected, QTreeView::item:selected, QListView::item:selected {
    background: %(accent)s; color: %(selection_text)s; }
QLineEdit, QAbstractSpinBox, QComboBox, QTextEdit, QPlainTextEdit, QTextBrowser {
    color: %(text)s; background: %(base)s; border: 1px solid %(border)s;
    border-radius: 3px; padding: 2px; selection-background-color: %(accent)s;
    selection-color: %(selection_text)s; }
QLineEdit:focus, QAbstractSpinBox:focus, QComboBox:focus, QTextEdit:focus, QPlainTextEdit:focus {
    border: 2px solid %(accent_strong)s; }
QLineEdit:disabled, QAbstractSpinBox:disabled, QComboBox:disabled, QTextEdit:disabled {
    color: %(disabled_text)s; background: %(disabled_panel)s; }
QComboBox QAbstractItemView { color: %(text)s; background: %(panel)s;
    selection-background-color: %(accent)s; selection-color: %(selection_text)s; }
QPushButton, QToolButton { color: %(text)s; background: %(panel)s;
    border: 1px solid %(border)s; border-radius: 4px; padding: 4px 8px; }
QPushButton:hover, QToolButton:hover { background: %(hover)s; border-color: %(accent_strong)s; }
QPushButton:pressed, QToolButton:pressed, QPushButton:checked, QToolButton:checked {
    color: %(selection_text)s; background: %(accent)s; }
QPushButton:disabled, QToolButton:disabled { color: %(disabled_text)s; background: %(disabled_panel)s; }
QPushButton:focus, QToolButton:focus { border: 2px solid %(accent_strong)s; }
QMenu { color: %(text)s; background: %(panel)s; border: 1px solid %(border)s; }
QMenu::item:selected { color: %(selection_text)s; background: %(accent)s; }
QMenu::item:disabled { color: %(disabled_text)s; }
QMenu::separator { height: 1px; background: %(border)s; margin: 4px 7px; }
QGroupBox { color: %(text)s; border: 1px solid %(border)s; border-radius: 4px;
    margin-top: 10px; padding-top: 8px; }
QGroupBox::title { subcontrol-origin: margin; padding: 0 4px; color: %(text)s; }
QCheckBox, QRadioButton { color: %(text)s; spacing: 6px; }
QCheckBox:disabled, QRadioButton:disabled { color: %(disabled_text)s; }
QCheckBox::indicator { width: 14px; height: 14px; background: %(base)s;
    border: 1px solid %(accent_strong)s; border-radius: 2px; }
QCheckBox::indicator:checked { background: %(accent)s; image: url("%(checked_mark)s"); }
QCheckBox::indicator:indeterminate { background: %(accent)s; image: url("%(partial_mark)s"); }
QCheckBox::indicator:disabled { border-color: %(border)s; background: %(disabled_panel)s; }
QCheckBox::indicator:checked:disabled, QCheckBox::indicator:indeterminate:disabled { background: %(disabled_checked)s; }
QRadioButton::indicator { width: 14px; height: 14px; background: %(base)s;
    border: 1px solid %(accent_strong)s; border-radius: 8px; }
QRadioButton::indicator:checked { background: qradialgradient(cx:.5,cy:.5,radius:.5,
    stop:0 %(accent_strong)s,stop:.5 %(accent_strong)s,stop:.6 %(base)s,stop:1 %(base)s); }
QRadioButton::indicator:disabled { border-color: %(border)s; background: %(disabled_panel)s; }
QRadioButton::indicator:checked:disabled { background: qradialgradient(cx:.5,cy:.5,radius:.5,
    stop:0 %(muted)s,stop:.5 %(muted)s,stop:.6 %(disabled_panel)s,stop:1 %(disabled_panel)s); }
QSplitter::handle { background: %(border)s; }
''' % values


def preferences():
    settings=QSettings('InjectorAnalytics','FuelFieldPilot')
    # Managed cloud sessions have a read-only home. Keep the same QSettings INI
    # contract in their persistent shared workspace rather than a source JSON.
    parent=Path(settings.fileName()).parent
    try:parent.mkdir(parents=True,exist_ok=True)
    except OSError:
        root=Path('/workspace/shared') if Path('/workspace/shared').is_dir() else Path(tempfile.gettempdir())
        settings=QSettings(str(root/'injector-analytics-ui.ini'),QSettings.IniFormat)
    return settings


def current_theme(app=None):
    app=app or QApplication.instance()
    return getattr(app,'_ff_theme','light')


def colors(canvas=None):
    name=getattr(canvas,'_ff_export_theme',None) or current_theme()
    return PALETTES[name]


def _luminance(rgb):
    linear=[value/12.92 if value<=.04045 else ((value+.055)/1.055)**2.4 for value in rgb]
    return sum(value*weight for value,weight in zip(linear,(.2126,.7152,.0722)))


def _contrast(value,dark):
    try:r,g,b,a=to_rgba(value)
    except (ValueError,TypeError):return value
    h,s,v=colorsys.rgb_to_hsv(r,g,b)
    if dark:
        if s<.12:r,g,b=to_rgba(PALETTES['dark']['text'])[:3]
        else:r,g,b=colorsys.hsv_to_rgb(h,min(s,.7),max(v,.78))
        # Bright saturated blue/red still have low perceived luminance. Retain
        # their hue while moving toward the text color until inherited labels
        # and graph annotations meet normal-text contrast on dark surfaces.
        backdrop=max(_luminance(to_rgba(PALETTES['dark'][key])[:3]) for key in ('window','panel','base','alternate'))
        # A small margin retains >=4.5 after QColor's 8-bit rounding.
        target_ratio=4.6
        if (_luminance((r,g,b))+.05)/(backdrop+.05)<target_ratio:
            original=(r,g,b);text=to_rgba(PALETTES['dark']['text'])[:3];lo=0.;hi=1.
            for _ in range(12):
                mid=(lo+hi)/2.;candidate=tuple((1-mid)*c+mid*t for c,t in zip(original,text))
                if (_luminance(candidate)+.05)/(backdrop+.05)<target_ratio:lo=mid
                else:hi=mid
            r,g,b=((1-hi)*c+hi*t for c,t in zip(original,text))
    elif s<.12:r,g,b=to_rgba(PALETTES['light']['text'])[:3]
    elif .2126*r+.7152*g+.0722*b>.86:
        r,g,b=colorsys.hsv_to_rgb(h,s,min(v,.6))
    return (r,g,b,a)


def _qcolor(value,dark):
    rgba=value.getRgbF() if isinstance(value,QColor) else value
    return QColor.fromRgbF(*_contrast(rgba,dark))


def readable_color(value,canvas=None):return _contrast(value,colors(canvas) is PALETTES['dark'])


def _artist_color(artist,dark):
    if not hasattr(artist,'_ff_original_color'):artist._ff_original_color=artist.get_color()
    artist.set_color(_contrast(artist._ff_original_color,dark))


def style_canvas(canvas,theme=None):
    if sip.isdeleted(canvas):return
    palette=PALETTES[theme] if theme else colors(canvas)
    dark=palette is PALETTES['dark'];figure=canvas.figure
    figure.set_facecolor(palette['window'])
    for text in figure.texts:_artist_color(text,dark)
    for ax in figure.axes:
        ax.set_facecolor(palette['panel'])
        for spine in ax.spines.values():spine.set_edgecolor(palette['border'])
        ax.tick_params(colors=palette['text'],which='both')
        for axis in (ax.xaxis,ax.yaxis):
            axis.label.set_color(palette['text']);axis.get_offset_text().set_color(palette['text'])
            for grid in axis.get_gridlines():grid.set_color(palette['border'])
        for text in [ax.title,*ax.texts]:_artist_color(text,dark)
        for line in ax.lines:_artist_color(line,dark)
        legend=ax.get_legend()
        if legend is not None:
            legend.get_frame().set_facecolor(palette['panel']);legend.get_frame().set_edgecolor(palette['border'])
            for text in legend.get_texts():_artist_color(text,dark)
        for text in ax.texts:
            patch=text.get_bbox_patch()
            if patch is not None:patch.set_facecolor(palette['panel']);patch.set_edgecolor(palette['border'])


def _style_widget(widget,palette,visible_only=False):
    if sip.isdeleted(widget):return
    stylesheet=widget.styleSheet()
    if stylesheet!=getattr(widget,'_ff_styled_stylesheet',None):widget._ff_original_stylesheet=stylesheet
    original=getattr(widget,'_ff_original_stylesheet',stylesheet)
    def color_rule(match):
        value=match.group(1)
        return 'color:'+to_hex(_contrast(value,current_theme()=='dark'))+';'
    styled=re.sub(r'(?<![-\w])color\s*:\s*(#[0-9a-fA-F]{3,8})\s*;?',color_rule,original)
    if isinstance(widget,QGraphicsView):
        mode=widget.property('ffSketchMode')
        if mode in ('geometry','air'):
            outline='#28a35d' if mode=='geometry' else '#2c85dc'
            styled+='\nQGraphicsView { border:3px solid '+outline+'; border-radius:4px; }'
    widget._ff_styled_stylesheet=styled
    if styled!=stylesheet:widget.setStyleSheet(styled)
    if isinstance(widget,QGraphicsView):
        widget.setBackgroundBrush(QBrush(QColor(palette['base'])))
        widget.viewport().update()
        for item in widget.scene().items() if widget.scene() is not None else []:
            # QGraphicsTextItem / QGraphicsSimpleTextItem labels, not primitives.
            if hasattr(item,'setDefaultTextColor'):
                if not hasattr(item,'_ff_original_text_color'):item._ff_original_text_color=QColor(item.defaultTextColor())
                original=item._ff_original_text_color
                item.setDefaultTextColor(_qcolor(original,True) if current_theme()=='dark' else original)
            elif hasattr(item,'text') and hasattr(item,'setBrush'):
                if not hasattr(item,'_ff_original_text_brush'):item._ff_original_text_brush=QBrush(item.brush())
                original=item._ff_original_text_brush
                item.setBrush(QBrush(_qcolor(original.color(),True)) if current_theme()=='dark' else original)
            elif hasattr(item,'pen') and hasattr(item,'setPen'):
                pen=item.pen()
                if pen.style()!=0:
                    color=pen.color()
                    if not hasattr(item,'_ff_original_pen_color') or color!=getattr(item,'_ff_last_pen_color',color):item._ff_original_pen_color=QColor(color)
                    original=item._ff_original_pen_color
                    target=_qcolor(original,True) if current_theme()=='dark' else original
                    pen.setColor(target);item.setPen(pen);item._ff_last_pen_color=QColor(target)
    if isinstance(widget,QTableWidget):
        # Do not connect Python closures to the table's internal C++ model.
        # Those senders may emit while a long-lived window tree is being
        # destroyed. Fresh cells are styled at the viewport's next Paint.
        if visible_only:
            first_row=widget.rowAt(0);first_col=widget.columnAt(0)
            if first_row<0 or first_col<0:return
            last_row=widget.rowAt(max(0,widget.viewport().height()-1))
            last_col=widget.columnAt(max(0,widget.viewport().width()-1))
            rows=range(first_row,(last_row+1) if last_row>=0 else widget.rowCount())
            columns=range(first_col,(last_col+1) if last_col>=0 else widget.columnCount())
        else:rows=range(widget.rowCount());columns=range(widget.columnCount())
        blocked=widget.blockSignals(True)
        try:
            for row in rows:
                for col in columns:
                    item=widget.item(row,col)
                    if item is None:continue
                    # Appearance must not emit project itemChanged edits.
                    brush=item.background()
                    if brush.style()!=0:
                        if not hasattr(item,'_ff_original_background') or brush!=getattr(item,'_ff_last_background',brush):item._ff_original_background=brush
                        original=item._ff_original_background
                        if isinstance(original,QBrush):
                            color=original.color()
                            if current_theme()=='dark':color=QColor(palette['alternate'])
                            target=QBrush(color)
                            if item.background()!=target:item.setBackground(target)
                            item._ff_last_background=QBrush(target)
                    foreground=item.foreground()
                    if foreground.style()!=0:
                        if not hasattr(item,'_ff_original_foreground') or foreground!=getattr(item,'_ff_last_foreground',foreground):item._ff_original_foreground=QBrush(foreground)
                        original=item._ff_original_foreground
                        target=QBrush(_qcolor(original.color(),current_theme()=='dark'))
                        if foreground!=target:item.setForeground(target)
                        item._ff_last_foreground=QBrush(target)
        finally:
            widget.blockSignals(blocked)
    if isinstance(widget,FigureCanvasQTAgg):style_canvas(widget)


class ThemeWatcher(QObject):
    def eventFilter(self,widget,event):
        if event.type()==QEvent.DynamicPropertyChange and isinstance(widget,QGraphicsView):
            if bytes(event.propertyName())==b'ffSketchMode':_style_widget(widget,PALETTES[current_theme()])
        if event.type()==QEvent.Paint and isinstance(widget,QWidget):
            parent=widget.parentWidget()
            if isinstance(parent,QTableWidget):_style_widget(parent,PALETTES[current_theme()],visible_only=True)
        if event.type()==QEvent.Hide and isinstance(widget,FigureCanvasQTAgg):
            interaction=getattr(widget,'interaction',None)
            if interaction is not None:interaction.suspend()
        if event.type() in (QEvent.Show,QEvent.Polish) and isinstance(widget,QWidget):
            reference=weakref.ref(widget)
            def style_later():
                target=reference()
                if target is not None and not sip.isdeleted(target):_style_widget(target,PALETTES[current_theme()])
            QTimer.singleShot(0,style_later)
        return False


def _install_canvas_hooks():
    if getattr(FigureCanvasQTAgg,'_ff_theme_hooks',False):return
    original_draw=FigureCanvasQTAgg.draw
    original_print=FigureCanvasQTAgg.print_figure
    def draw(canvas,*args,**kwargs):
        style_canvas(canvas)
        return original_draw(canvas,*args,**kwargs)
    def print_figure(canvas,*args,**kwargs):
        canvas._ff_export_theme='light'
        try:
            style_canvas(canvas)
            return original_print(canvas,*args,**kwargs)
        finally:
            del canvas._ff_export_theme;style_canvas(canvas);canvas.draw_idle()
    FigureCanvasQTAgg.draw=draw;FigureCanvasQTAgg.print_figure=print_figure
    FigureCanvasQTAgg._ff_theme_hooks=True


def apply_theme(app,name,persist=True):
    if name not in PALETTES:raise ValueError('Unknown application theme: '+str(name))
    _install_canvas_hooks();app._ff_theme=name;values=PALETTES[name]
    app.setStyle('Fusion');palette=QPalette()
    roles={QPalette.Window:'window',QPalette.WindowText:'text',QPalette.Base:'base',QPalette.AlternateBase:'alternate',
           QPalette.Text:'text',QPalette.Button:'panel',QPalette.ButtonText:'text',QPalette.ToolTipBase:'panel',
           QPalette.ToolTipText:'text',QPalette.Highlight:'accent',QPalette.HighlightedText:'selection_text',QPalette.Link:'link',
           QPalette.Light:'border',QPalette.Mid:'border',QPalette.Dark:'border',QPalette.BrightText:'text'}
    for role,key in roles.items():palette.setColor(role,QColor(values[key]))
    for role in (QPalette.Text,QPalette.ButtonText,QPalette.WindowText):palette.setColor(QPalette.Disabled,role,QColor(values['disabled_text']))
    if name!='light':
        palette.setColor(QPalette.Disabled,QPalette.Base,QColor(values['disabled_panel']))
        palette.setColor(QPalette.Disabled,QPalette.Button,QColor(values['disabled_panel']))
    app.setPalette(palette)
    # Selectors override old inline label colors while preserving layout/font rules.
    app.setStyleSheet(_stylesheet(values,name))
    rcParams.update({'figure.facecolor':values['window'],'axes.facecolor':values['panel'],'text.color':values['text'],
                     'axes.labelcolor':values['text'],'axes.edgecolor':values['border'],'xtick.color':values['text'],
                     'ytick.color':values['text'],'grid.color':values['border'],'savefig.facecolor':'white'})
    if not hasattr(app,'_ff_theme_watcher'):
        app._ff_theme_watcher=ThemeWatcher(app);app.installEventFilter(app._ff_theme_watcher)
    for widget in app.allWidgets():
        if isinstance(widget,QWidget):
            _style_widget(widget,values)
            if isinstance(widget,FigureCanvasQTAgg):
                interaction=getattr(widget,'interaction',None)
                if interaction is not None:interaction.background=None;interaction.base_background=None
                widget.draw_idle()
            widget.update()
    if persist:
        settings=preferences();settings.setValue('appearance/theme',name);settings.sync()


def load_theme(app):
    name=preferences().value('appearance/theme','light')
    name=name if name in PALETTES else 'light'
    if getattr(app,'_ff_theme',None)!=name:apply_theme(app,name,persist=False)
    return current_theme(app)
