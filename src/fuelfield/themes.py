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
from PyQt5.QtCore import QObject,QEvent,QSettings,QTimer
from PyQt5.QtGui import QColor,QPalette,QBrush,QPen
from PyQt5.QtWidgets import QApplication,QGraphicsView,QTableWidget,QWidget
from matplotlib import rcParams
from matplotlib.colors import to_rgba,to_hex
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg

PALETTES={
    'light':dict(window='#f4f6fa',panel='#ffffff',base='#ffffff',alternate='#eef2f8',text='#202b3b',muted='#52617c',border='#c8d0dc',accent='#386ec5',guide='#357b72',pin='#a35721'),
    'dark':dict(window='#1b2230',panel='#222c3c',base='#17202e',alternate='#29364a',text='#e5ebf4',muted='#b4c2d7',border='#48566d',accent='#74a7f6',guide='#68c7b7',pin='#f4ae76'),
}


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


def _contrast(value,dark):
    try:r,g,b,a=to_rgba(value)
    except (ValueError,TypeError):return value
    h,s,v=colorsys.rgb_to_hsv(r,g,b)
    if dark:
        if s<.12:r,g,b=to_rgba(PALETTES['dark']['text'])[:3]
        else:r,g,b=colorsys.hsv_to_rgb(h,min(s,.7),max(v,.78))
    elif s<.12:r,g,b=to_rgba(PALETTES['light']['text'])[:3]
    elif .2126*r+.7152*g+.0722*b>.86:
        r,g,b=colorsys.hsv_to_rgb(h,s,min(v,.6))
    return (r,g,b,a)


def _qcolor(value,dark):return QColor.fromRgbF(*_contrast(value,dark))


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


def _style_widget(widget,palette):
    if sip.isdeleted(widget):return
    stylesheet=widget.styleSheet()
    if stylesheet!=getattr(widget,'_ff_styled_stylesheet',None):widget._ff_original_stylesheet=stylesheet
    original=getattr(widget,'_ff_original_stylesheet',stylesheet)
    def color_rule(match):
        value=match.group(1)
        return 'color:'+to_hex(_contrast(value,current_theme()=='dark'))+';'
    styled=re.sub(r'(?<![-\w])color\s*:\s*(#[0-9a-fA-F]{3,8})\s*;?',color_rule,original)
    widget._ff_styled_stylesheet=styled
    if styled!=stylesheet:widget.setStyleSheet(styled)
    if isinstance(widget,QGraphicsView):
        widget.setBackgroundBrush(QBrush(QColor(palette['base'])))
        widget.viewport().update()
        for item in widget.scene().items() if widget.scene() is not None else []:
            # QGraphicsTextItem / QGraphicsSimpleTextItem labels, not primitives.
            if hasattr(item,'setDefaultTextColor'):item.setDefaultTextColor(QColor(palette['text']))
            elif hasattr(item,'text') and hasattr(item,'setBrush'):item.setBrush(QBrush(QColor(palette['muted'])))
            elif hasattr(item,'pen') and hasattr(item,'setPen'):
                pen=item.pen()
                if pen.style()!=0:
                    if not hasattr(item,'_ff_original_pen'):item._ff_original_pen=QPen(pen)
                    pen=QPen(item._ff_original_pen);pen.setColor(_qcolor(pen.color().name(),current_theme()=='dark'));item.setPen(pen)
    if isinstance(widget,QTableWidget):
        if not getattr(widget,'_ff_theme_table_hook',False):
            widget._ff_theme_table_hook=True;reference=weakref.ref(widget)
            def dirty(*args):
                table=reference()
                if table is not None and not sip.isdeleted(table) and not getattr(table,'_ff_theme_styling',False):table._ff_theme_table_dirty=True
            for signal in (widget.model().dataChanged,widget.model().rowsInserted,widget.model().columnsInserted,widget.model().modelReset):signal.connect(dirty)
        widget._ff_theme_styling=True
        blocked=widget.blockSignals(True)
        try:
            for row in range(widget.rowCount()):
                for col in range(widget.columnCount()):
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
                        target=QBrush(_qcolor(original.color().name(),current_theme()=='dark'))
                        if foreground!=target:item.setForeground(target)
                        item._ff_last_foreground=QBrush(target)
        finally:
            widget.blockSignals(blocked);widget._ff_theme_styling=False;widget._ff_theme_table_dirty=False
    if isinstance(widget,FigureCanvasQTAgg):style_canvas(widget)


class ThemeWatcher(QObject):
    def eventFilter(self,widget,event):
        if event.type()==QEvent.Paint and isinstance(widget,QWidget):
            parent=widget.parentWidget()
            if isinstance(parent,QTableWidget) and getattr(parent,'_ff_theme_table_dirty',False):_style_widget(parent,PALETTES[current_theme()])
        if event.type()==QEvent.Hide and isinstance(widget,FigureCanvasQTAgg):
            interaction=getattr(widget,'interaction',None)
            if interaction is not None:interaction.leave(None)
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
           QPalette.ToolTipText:'text',QPalette.Highlight:'accent',QPalette.HighlightedText:'base',QPalette.Link:'accent',
           QPalette.Light:'border',QPalette.Mid:'border',QPalette.Dark:'border',QPalette.BrightText:'text'}
    for role,key in roles.items():palette.setColor(role,QColor(values[key]))
    for role in (QPalette.Text,QPalette.ButtonText,QPalette.WindowText):palette.setColor(QPalette.Disabled,role,QColor(values['muted']))
    app.setPalette(palette)
    # Selectors override old inline label colors while preserving layout/font rules.
    app.setStyleSheet('QToolTip { color: '+values['text']+'; background: '+values['panel']+'; border:1px solid '+values['border']+'; }')
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
