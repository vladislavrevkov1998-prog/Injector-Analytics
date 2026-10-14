"""Shared epure interactions. Hover blits overlays; it never recalculates fields."""
from dataclasses import dataclass
import numpy as np
from matplotlib.backend_bases import MouseButton
from matplotlib.collections import PathCollection
from matplotlib.patches import Rectangle

@dataclass
class PinnedPoint:
    axes: object
    x: float
    y: float
    artists: list

class PlotInteraction:
    MAX_POINTS=10
    def __init__(self,canvas):
        self.canvas=canvas;self.points=[];self.axes=[];self.hover={};self.background=None;self.base_background=None
        self.pressed=set();self.drag=None;self.rectangle=None;self.original={}
        self.connections=[canvas.mpl_connect(name,handler) for name,handler in (
            ('draw_event',self._drawn),('motion_notify_event',self.motion),
            ('button_press_event',self.press),('button_release_event',self.release),
            ('figure_leave_event',self.leave),('key_press_event',self.key))]
        self.reset_axes()
    def clear_artists(self):
        """Detach overlays before a Figure clears/replaces its axes."""
        for point in self.points:
            for artist in point.artists:artist.remove()
        for artists in self.hover.values():
            for artist in artists:artist.remove()
        if self.rectangle is not None:self.rectangle.remove()
        self.points=[];self.hover={};self.drag=None;self.rectangle=None;self.background=None;self.base_background=None;self.pressed.clear()
    def reset_axes(self):
        """Call after replacing chart data. Annotations belong to that data/view."""
        self.clear_artists()
        self.axes=list(self.canvas.figure.axes)
        self.original={ax:(ax.get_xlim(),ax.get_ylim()) for ax in self.axes}
        for ax in self.axes:
            v=ax.axvline(0,color='#76509c',lw=.8,ls='--',visible=False,animated=True)
            h=ax.axhline(0,color='#76509c',lw=.8,ls='--',visible=False,animated=True)
            text=ax.text(.02,.98,'',transform=ax.transAxes,va='top',fontsize=8,
                         bbox=dict(facecolor='white',alpha=.9,edgecolor='#c9b4db'),visible=False,animated=True)
            self.hover[ax]=(v,h,text)
            xlim,ylim=self.original[ax];ax.set_xlim(xlim);ax.set_ylim(ylim)
    def _drawn(self,event):
        if event.canvas is not self.canvas:return
        if self.canvas.is_saving():self.background=None;self.base_background=None;return
        self.base_background=self.canvas.copy_from_bbox(self.canvas.figure.bbox)
        self._bake_points()
        self._paint_overlays()
    def _bake_points(self):
        if self.base_background is None:self.canvas.draw_idle();return
        self.canvas.restore_region(self.base_background)
        for point in self.points:
            for artist in point.artists:artist.axes.draw_artist(artist)
        self.background=self.canvas.copy_from_bbox(self.canvas.figure.bbox)

    def _blit(self):
        if self.background is None:
            self.canvas.draw_idle();return
        self.canvas.restore_region(self.background)
        self._paint_overlays()
        self.canvas.blit(self.canvas.figure.bbox)
    def _paint_overlays(self):
        for artists in self.hover.values():
            for artist in artists:
                if artist.get_visible():artist.axes.draw_artist(artist)
        if self.rectangle is not None and self.rectangle.get_visible():self.rectangle.axes.draw_artist(self.rectangle)
    def _hide_hover(self):
        for artists in self.hover.values():
            for artist in artists:artist.set_visible(False)
    def motion(self,event):
        self._hide_hover()
        if self.drag is not None:
            ax,x0,y0=self.drag
            x,y=self._coordinates(ax,event.x,event.y)
            self.rectangle.set_bounds(min(x0,x),min(y0,y),abs(x-x0),abs(y-y0));self._blit();return
        ax=event.inaxes
        if ax in self.hover and event.xdata is not None and event.ydata is not None:
            v,h,text=self.hover[ax];v.set_xdata([event.xdata,event.xdata]);h.set_ydata([event.ydata,event.ydata])
            label=f'x={event.xdata:.7g}; y={event.ydata:.7g}'
            if self.points:label+=f' · точек {len(self.points)}/{self.MAX_POINTS}'
            # With twinx show the physical coordinate of both Y axes.
            for other in self.axes:
                if other is not ax and np.allclose(other.bbox.bounds,ax.bbox.bounds):
                    _,y=other.transData.inverted().transform((event.x,event.y))
                    label+=f'\n{other.get_ylabel() or "Y"}={y:.7g}'
            text.set_text(label)
            for artist in (v,h,text):artist.set_visible(True)
        self._blit()
    def leave(self,event):
        self._hide_hover();self.pressed.clear()
        if self.drag is not None:self._cancel_drag()
        self._blit()
    def _coordinates(self,ax,x,y):
        pixel=(np.clip(x,ax.bbox.x0,ax.bbox.x1),np.clip(y,ax.bbox.y0,ax.bbox.y1))
        return ax.transData.inverted().transform(pixel)
    def nearest(self,event):
        """Closest line segment in screen pixels; preserve NaN gaps and twinx units."""
        candidates=[];cursor=np.array([event.x,event.y])
        for ax in self.axes:
            if event.inaxes is None or not np.allclose(ax.bbox.bounds,event.inaxes.bbox.bounds):continue
            ignored=set(self.hover.get(ax,()))
            ignored.update(a for point in self.points for a in point.artists)
            for line in ax.lines:
                if line in ignored or not line.get_visible():continue
                data=np.ma.column_stack(line.get_data(orig=False)).filled(np.nan)
                if len(data)==0:continue
                pixels=ax.transData.transform(data);valid=np.all(np.isfinite(pixels),axis=1)
                if valid.any():
                    d=np.sum((pixels[valid]-cursor)**2,axis=1);i=int(np.argmin(d))
                    candidates.append((float(d[i]),ax,*data[valid][i]))
                pairs=valid[:-1]&valid[1:]
                if pairs.any():
                    a=pixels[:-1][pairs];b=pixels[1:][pairs];delta=b-a
                    denom=np.sum(delta*delta,axis=1)
                    t=np.divide(np.sum((cursor-a)*delta,axis=1),denom,out=np.zeros_like(denom),where=denom>0).clip(0,1)
                    projected=a+t[:,None]*delta;dist=np.sum((projected-cursor)**2,axis=1);j=int(np.argmin(dist))
                    x,y=ax.transData.inverted().transform(projected[j]);candidates.append((float(dist[j]),ax,x,y))
            for collection in ax.collections:
                if not isinstance(collection,PathCollection) or not collection.get_visible():continue
                data=np.ma.array(collection.get_offsets()).filled(np.nan);pixels=ax.transData.transform(data)
                valid=np.all(np.isfinite(pixels),axis=1)
                if valid.any():
                    distances=np.sum((pixels[valid]-cursor)**2,axis=1);i=int(np.argmin(distances))
                    candidates.append((float(distances[i]),ax,*data[valid][i]))
            for container in ax.containers:
                if not hasattr(container,'patches'):continue
                for bar in container.patches:
                    if not bar.get_visible():continue
                    x=bar.get_x()+bar.get_width()/2;y=bar.get_y()+bar.get_height()
                    pixel=ax.transData.transform((x,y));distance=float(np.sum((pixel-cursor)**2))
                    if np.isfinite(distance):candidates.append((distance,ax,x,y))
        if not candidates:return None
        _,ax,x,y=min(candidates,key=lambda c:c[0]);return ax,float(x),float(y)
    def press(self,event):
        self.pressed.add(event.button)
        if {MouseButton.LEFT,MouseButton.RIGHT}.issubset(self.pressed):
            self._cancel_drag();self.clear_points();return
        if event.button==MouseButton.MIDDLE and event.inaxes in self.axes:
            self._hide_hover();ax=event.inaxes;x,y=self._coordinates(ax,event.x,event.y)
            self.drag=(ax,x,y);self.rectangle=Rectangle((x,y),0,0,facecolor='#b9a1d4',edgecolor='#76509c',alpha=.3,animated=True)
            ax.add_patch(self.rectangle);return
        if event.button==MouseButton.LEFT and event.dblclick and len(self.points)<self.MAX_POINTS:
            point=self.nearest(event)
            if point is None:return
            ax,x,y=point
            artists=[ax.axvline(x,color='#a35721',lw=.7,ls=':'),ax.axhline(y,color='#a35721',lw=.7,ls=':'),
                     ax.plot([x],[y],'o',color='#a35721',ms=4)[0],
                     ax.annotate(f'{len(self.points)+1}: ({x:.7g}, {y:.7g})',(x,y),xytext=(5,6),textcoords='offset points',fontsize=7,color='#803d15')]
            for artist in artists:artist.set_animated(True)
            self.points.append(PinnedPoint(ax,x,y,artists));self._bake_points();self._blit()
    def _cancel_drag(self):
        self.drag=None
        if self.rectangle is not None:self.rectangle.remove();self.rectangle=None
    def release(self,event):
        self.pressed.discard(event.button)
        if event.button!=MouseButton.MIDDLE or self.drag is None:return
        ax,x0,y0=self.drag;x,y=self._coordinates(ax,event.x,event.y)
        # Ignore accidental clicks or degenerate selection rectangles.
        start=ax.transData.transform((x0,y0));end=ax.transData.transform((x,y))
        if np.all(np.abs(end-start)>=5):
            low=np.minimum(start,end);high=np.maximum(start,end)
            limits=[]
            for other in self.axes:
                if not np.allclose(other.bbox.bounds,ax.bbox.bounds):continue
                a=other.transData.inverted().transform(low);b=other.transData.inverted().transform(high)
                limits.append((other,sorted((a[0],b[0]),reverse=other.xaxis_inverted()),sorted((a[1],b[1]),reverse=other.yaxis_inverted())))
            for other,xlim,ylim in limits:other.set_xlim(xlim);other.set_ylim(ylim)
        self._cancel_drag();self.canvas.draw_idle()
    def clear_points(self):
        for point in self.points:
            for artist in point.artists:artist.remove()
        self.points=[];self.background=self.base_background;self._blit()
    def reset_zoom(self):
        self._cancel_drag()
        for ax,(xlim,ylim) in self.original.items():ax.set_xlim(xlim);ax.set_ylim(ylim)
        self.canvas.draw_idle()
    def key(self,event):
        if event.key=='r':self.reset_zoom()
        elif event.key=='escape':self._cancel_drag();self._hide_hover();self._blit()


def attach_interaction(canvas):
    from PyQt5.QtCore import Qt
    canvas.setFocusPolicy(Qt.ClickFocus)
    controller=PlotInteraction(canvas);canvas.interaction=controller
    canvas.setToolTip('Наведение: прицел и x/y. Двойной ЛКМ: точка на эпюре (до 10). '\
                      'ЛКМ + ПКМ: удалить точки. Средняя кнопка + выделение: масштаб. R: исходный масштаб.')
    return controller
