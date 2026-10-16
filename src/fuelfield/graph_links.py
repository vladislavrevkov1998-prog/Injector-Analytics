"""Visual links between independent charts; never changes physical source data."""
from dataclasses import dataclass
import math
import numpy as np
from PyQt5 import sip
from PyQt5.QtCore import Qt,QEvent,QPointF,QPoint,QRectF
from PyQt5.QtGui import QPainter,QPen,QColor,QPolygonF
from PyQt5.QtWidgets import QWidget,QApplication,QMenu,QDialog,QVBoxLayout,QLabel,QDialogButtonBox
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.collections import PathCollection


def series_data(ax,controller=None):
    """First displayed series only. Reject ambiguous/nonmonotonic X, preserve gaps."""
    ignored=set()
    if controller:
        ignored.update(controller.hover.get(ax,()))
        ignored.update(a for p in controller.points+controller.oblique for a in p.artists)
    for line in ax.lines:
        if line not in ignored and line.get_visible():
            data=np.ma.column_stack(line.get_data(orig=False)).filled(np.nan)
            if len(data)>1:return data[:,0],data[:,1]
    for collection in ax.collections:
        if isinstance(collection,PathCollection):
            data=np.ma.array(collection.get_offsets()).filled(np.nan)
            if len(data)>1:return data[:,0],data[:,1]
    for container in ax.containers:
        if hasattr(container,'patches'):
            return np.array([p.get_x()+p.get_width()/2 for p in container.patches]),np.array([p.get_height()+p.get_y() for p in container.patches])
    return np.array([]),np.array([])

class Curve:
    def __init__(self,ax,controller):
        x,y=series_data(ax,controller);self.x=np.asarray(x,float);self.y=np.asarray(y,float)
        finite=np.isfinite(self.x);dx=np.diff(self.x[finite]);self.valid=finite.all() and len(dx)>0 and (np.all(dx>0) or np.all(dx<0))
        if self.valid and np.all(dx<0):self.x=self.x[::-1];self.y=self.y[::-1]
    def at(self,x):
        if not self.valid or not np.isfinite(x) or x<self.x[0] or x>self.x[-1]:return float('nan')
        j=int(np.searchsorted(self.x,x))
        if j<len(self.x) and self.x[j]==x:return float(self.y[j])
        if j==0 or j==len(self.x) or not np.isfinite(self.y[j-1:j+1]).all():return float('nan')
        return float(np.interp(x,self.x[j-1:j+1],self.y[j-1:j+1]))


def comparison_statistics(a,b):
    if not a.valid or not b.valid:raise ValueError('X должен быть строго монотонным, без повторяющихся значений.')
    lo=max(a.x[0],b.x[0]);hi=min(a.x[-1],b.x[-1])
    x=np.unique(np.r_[a.x[(a.x>=lo)&(a.x<=hi)],b.x[(b.x>=lo)&(b.x<=hi)]])
    ya=np.array([a.at(v) for v in x]);yb=np.array([b.at(v) for v in x]);ok=np.isfinite(ya)&np.isfinite(yb);x,ya,yb=x[ok],ya[ok],yb[ok]
    if len(x)<3:raise ValueError('Нужно хотя бы 3 общие конечные точки X; экстраполяция не выполняется.')
    if np.ptp(ya)<=np.finfo(float).eps*max(abs(ya).max(),1e-300):raise ValueError('Первая кривая постоянна: наклон и Pearson r не определены.')
    slope=float(np.dot(ya-ya.mean(),yb-yb.mean())/np.dot(ya-ya.mean(),ya-ya.mean()));offset=float(yb.mean()-slope*ya.mean())
    predicted=slope*ya+offset;denom=float(np.sum((yb-yb.mean())**2))
    constant=np.ptp(yb)<=np.finfo(float).eps*max(abs(yb).max(),1e-300)
    r=None if constant else float(np.corrcoef(ya,yb)[0,1]);r2=None if constant else 1-float(np.sum((yb-predicted)**2))/denom
    return dict(x=x,first=ya,second=yb,slope=slope,offset=offset,r=r,r2=r2,n=len(x))

@dataclass(eq=False)
class Link:
    first: object
    second: object
    first_controller: object
    second_controller: object
    a: object
    b: object

class LinkOverlay(QWidget):
    def __init__(self,manager):
        super().__init__(manager.owner.result_panels.parentWidget());self.manager=manager
        self.setGeometry(manager.owner.result_panels.geometry())
        self.setAttribute(Qt.WA_TransparentForMouseEvents);self.setAttribute(Qt.WA_NoSystemBackground);self.show()
    def eventFilter(self,watched,event):return self.manager.eventFilter(watched,event)
    def paintEvent(self,event):
        m=self.manager;p=QPainter(self);p.setRenderHint(QPainter.Antialiasing);p.setPen(QPen(QColor('#76509c'),1.8))
        for ax,controller in m.sources():
            point=m.anchor(ax,controller,True);p.setBrush(Qt.NoBrush);p.drawEllipse(point,6,6)
        for link in m.links:
            if not m.visible(link):continue
            a=m.anchor(link.first,link.first_controller,True);b=m.anchor(link.second,link.second_controller,False)
            p.drawLine(a,b);delta=a-b;length=math.hypot(delta.x(),delta.y())
            if length:
                unit=delta/length;normal=QPointF(-unit.y(),unit.x());p.setBrush(QColor('#76509c'));p.drawPolygon(QPolygonF([b,b+unit*10+normal*4,b+unit*10-normal*4]))
            middle=(a+b)/2
            if m.hovered is link:
                p.setBrush(QColor('white'));p.drawRoundedRect(QRectF(middle.x()-10,middle.y()-10,20,20),3,3);p.drawText(int(middle.x()-5),int(middle.y()+5),'+')
        if m.pending is not None:
            ax,c=m.pending;p.setPen(QPen(QColor('#76509c'),1.5,Qt.DashLine));p.drawLine(m.anchor(ax,c,True),m.cursor)

class GraphLinks:
    def __init__(self,owner):
        self.owner=owner;self.links=[];self.controllers=[];self.pending=None;self.hovered=None;self.cursor=QPointF();self._syncing=False
        self.owner.result_panels.setMouseTracking(True)
        self.overlay=LinkOverlay(self);QApplication.instance().installEventFilter(self.overlay)
        # Keep a QObject event filter alive with the same lifetime as the window.

    def reset(self):
        self.links=[];self.pending=None;self.hovered=None
        for c in self.controllers:c.link_manager=None
        self.controllers=[]
        if not sip.isdeleted(self.overlay):self.overlay.update()
    def refresh(self):
        if sip.isdeleted(self.owner) or sip.isdeleted(self.overlay):return
        primary=getattr(self.owner,'curvecanvas',None);right=self.owner.comparison.canvas
        active=[]
        if primary is not None and not sip.isdeleted(primary) and hasattr(primary,'interaction') and self.owner.plots.currentIndex()==3:active.append(primary.interaction)
        if self.owner.comparison.view.currentIndex()==0:active.append(right.interaction)
        current_axes={ax for c in active for ax in c.axes if ax in c.canvas.figure.axes}
        self.links=[l for l in self.links if l.first in current_axes and l.second in current_axes]
        if self.pending is not None and self.pending[0] not in current_axes:self.pending=None
        for c in active:
            c.link_manager=self
            if not getattr(c,'_link_draw_hook',False):
                c.canvas.mpl_connect('draw_event',lambda event:self.overlay.update() if not sip.isdeleted(self.overlay) else None);c._link_draw_hook=True
            for ax in c.axes:
                if not getattr(ax,'_link_callbacks',False):
                    ax.callbacks.connect('xlim_changed',self.limits);ax.callbacks.connect('ylim_changed',self.limits);ax._link_callbacks=True
        self.controllers=active
        # Arrow midpoints may lie above a splitter, tab stack or panel padding.
        # Every underlying widget must deliver hover events, even without a button.
        for widget in [self.owner.result_panels,*self.owner.result_panels.findChildren(QWidget)]:widget.setMouseTracking(True)
        self.overlay.setGeometry(self.owner.result_panels.geometry());self.overlay.raise_();self.overlay.update()
    def sources(self):
        if not self.owner.split_button.isChecked() or self.owner.plots.currentIndex()!=3:return []
        canvas=self.owner.curvecanvas
        if not self.owner.results or sip.isdeleted(canvas) or not hasattr(canvas,'interaction') or not canvas.isVisible():return []
        return [(ax,canvas.interaction) for ax in canvas.interaction.axes if ax.name=='rectilinear']
    def anchor(self,ax,c,right):
        # Matplotlib display coordinates use a bottom-left origin; Qt uses top-left.
        ratio=c.canvas.device_pixel_ratio
        x=ax.bbox.x1/ratio+9 if right else ax.bbox.x0/ratio-9;y=(ax.bbox.y0+ax.bbox.y1)/(2*ratio)
        point=self.overlay.mapFromGlobal(c.canvas.mapToGlobal(QPoint(int(x),int(c.canvas.height()-y))))
        return QPointF(point)
    def visible(self,link):return not sip.isdeleted(link.first_controller.canvas) and not sip.isdeleted(link.second_controller.canvas) and link.first_controller.canvas.isVisible() and link.second_controller.canvas.isVisible() and self.owner.split_button.isChecked()
    def eventFilter(self,watched,event):
        kind=event.type()
        if kind in (QEvent.Resize,QEvent.Move,QEvent.Show,QEvent.Hide) and watched is self.owner.result_panels:
            self.overlay.setGeometry(self.owner.result_panels.geometry());self.overlay.update()
        if kind not in (QEvent.MouseMove,QEvent.MouseButtonPress,QEvent.MouseButtonRelease,QEvent.KeyPress):return False
        if kind==QEvent.KeyPress:
            if event.key()==Qt.Key_Escape:self.pending=None;self.overlay.update()
            return False
        if not hasattr(event,'globalPos'):return False
        pos=QPointF(self.overlay.mapFromGlobal(event.globalPos()));self.cursor=pos
        hovered=next((l for l in self.links if self.visible(l) and self.distance(pos,(self.anchor(l.first,l.first_controller,True)+self.anchor(l.second,l.second_controller,False))/2)<14),None)
        if hovered is not self.hovered or self.pending is not None:self.hovered=hovered;self.overlay.update()
        if kind not in (QEvent.MouseButtonPress,QEvent.MouseButtonRelease) or event.button()!=Qt.LeftButton:return False
        if hovered is not None and kind==QEvent.MouseButtonPress:
            menu=QMenu(self.owner);action=menu.addAction('Кривая связи: Pearson r, R² и уравнение');remove=menu.addAction('Удалить link')
            chosen=menu.exec_(event.globalPos())
            if chosen is action:self.analyze(hovered)
            elif chosen is remove:self.links.remove(hovered);self.overlay.update()
            return True
        if self.pending is not None:
            c=self.owner.comparison.interaction
            if c.canvas.isVisible() and self.owner.comparison.view.currentIndex()==0:
                local=c.canvas.mapFromGlobal(event.globalPos());pixel=(local.x()*c.canvas.device_pixel_ratio,(c.canvas.height()-local.y())*c.canvas.device_pixel_ratio)
                ax=next((ax for ax in c.axes if ax.bbox.contains(*pixel)),None)
                if ax is not None:
                    source,controller=self.pending
                    try:self.connect(source,controller,ax,c)
                    except ValueError as exc:self.owner.statusBar().showMessage(str(exc),10000)
                    self.pending=None;self.overlay.update();return True
        if kind!=QEvent.MouseButtonPress:return False
        for ax,c in self.sources():
            if self.distance(pos,self.anchor(ax,c,True))<=10:
                self.pending=(ax,c);self.owner.statusBar().showMessage('Link: нажмите нужную эпюру справа. Esc — отменить.',10000);self.overlay.update();return True
        return False
    @staticmethod
    def distance(a,b):return math.hypot(a.x()-b.x(),a.y()-b.y())
    def connect(self,ax,c,other,d):
        a,b=Curve(ax,c),Curve(other,d)
        if not a.valid or not b.valid:raise ValueError('Link требует монотонный X без повторов; исходную таблицу упорядочьте по X.')
        xa,xb=getattr(ax,'_compare_x',None),getattr(other,'_compare_x',None)
        if xa and xb and xa!=xb:raise ValueError('Разные единицы X: связь невозможна.')
        if max(a.x[0],b.x[0])>min(a.x[-1],b.x[-1]):raise ValueError('Нет общего диапазона X.')
        # A single partner per graph keeps point/scale synchronization unambiguous.
        self.links=[l for l in self.links if ax not in (l.first,l.second) and other not in (l.first,l.second)]
        link=Link(ax,other,c,d,a,b);self.links.append(link);c.link_manager=self;d.link_manager=self
        self.limits(ax);self.overlay.update()
        self.owner.statusBar().showMessage('Link создан. Сравнивается первая серия каждой эпюры; X сопоставляется численно, без экстраполяции.',10000)
        return link
    def partner(self,ax):
        for l in self.links:
            if ax is l.first:return l,l.second,l.second_controller,l.a,l.b
            if ax is l.second:return l,l.first,l.first_controller,l.b,l.a
        return None
    @staticmethod
    def same_y(a,b):return bool(getattr(a,'_compare_y',None)) and getattr(a,'_compare_y',None)==getattr(b,'_compare_y',None)
    def limits(self,ax):
        if self._syncing:return
        pair=self.partner(ax)
        if pair is None:return
        l,other,c,_,_=pair;self._syncing=True
        try:
            other.set_xlim(ax.get_xlim())
            if self.same_y(ax,other):other.set_ylim(ax.get_ylim())
            c.canvas.draw_idle()
        finally:self._syncing=False
    def pin(self,ax,x):
        pair=self.partner(ax)
        if pair:
            l,other,c,a,b=pair;y=b.at(x)
            if np.isfinite(y):c.add_point(other,x,y)
    def hover(self,ax,x):
        pair=self.partner(ax)
        if pair is None:return '', '#222222'
        l,other,c,a,b=pair;y,z=a.at(x),b.at(x)
        if not np.isfinite(y) or not np.isfinite(z):return 'Link: вне общей кривой / пропуск', '#666666'
        if not self.same_y(ax,other):return f'Link Y={z:.7g}; ΔY: разные единицы', '#666666'
        delta=y-z
        return f'Y кривой={y:.7g}; link Y={z:.7g}\nΔY (здесь − link)={delta:+.7g}', '#18833b' if delta>0 else '#c33333' if delta<0 else '#555555'
    def clear(self,controller):
        partners=set()
        for l in self.links:
            if controller is l.first_controller:partners.add(l.second_controller)
            elif controller is l.second_controller:partners.add(l.first_controller)
        for c in partners:c.clear_points(False)
    def analyze(self,link):
        dialog=QDialog(self.owner);dialog.setWindowTitle('Кривая связи двух эпюр');dialog.resize(720,520);layout=QVBoxLayout(dialog)
        try:
            if not self.same_y(link.first,link.second):raise ValueError('Разные единицы Y: кривая связи не строится.')
            result=comparison_statistics(link.a,link.b);fmt=lambda v:'N/A' if v is None else f'{v:.7g}'
            label=QLabel(f"Ysplit = {result['slope']:.7g} · Yосновн {result['offset']:+.7g}; Pearson r={fmt(result['r'])}; R²={fmt(result['r2'])}; n={result['n']}.");label.setWordWrap(True);layout.addWidget(label)
            note=QLabel('Описательное линейное сравнение Ysplit(Yосновн) на объединённой сетке общего X. Линейная интерполяция, без экстраполяции и пересечения пропусков. Это не физическая идентификация и не доказательство причинности.');note.setWordWrap(True);layout.addWidget(note)
            fig=Figure(tight_layout=True);canvas=FigureCanvasQTAgg(fig);ax=fig.add_subplot(111);ax.scatter(result['first'],result['second'],s=10);x=np.sort(result['first']);ax.plot(x,result['slope']*x+result['offset'],color='#76509c');ax.set_xlabel('Y основного графика ['+link.first._compare_y+']');ax.set_ylabel('Y split ['+link.second._compare_y+']');ax.grid(alpha=.2);layout.addWidget(canvas);dialog.statistics=result;canvas.draw_idle()
        except ValueError as exc:layout.addWidget(QLabel(str(exc)))
        close=QDialogButtonBox(QDialogButtonBox.Close);close.rejected.connect(dialog.close);layout.addWidget(close);self.analysis_dialog=dialog;dialog.show();return dialog
