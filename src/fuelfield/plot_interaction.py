"""Shared epure interactions. Hover blits overlays; it never recalculates fields."""
from dataclasses import dataclass
import numpy as np
from matplotlib.backend_bases import MouseButton
from matplotlib.collections import PathCollection
from matplotlib.patches import Rectangle
from .plot_history import PlotHistory,locator_state,restore_locator
from .themes import colors,readable_color

@dataclass
class PinnedPoint:
    axes: object
    x: float
    y: float
    artists: list
    slope: object = None

class PlotInteraction:
    MAX_POINTS=10
    def __init__(self,canvas):
        self.canvas=canvas;self.points=[];self.axes=[];self.hover={};self.background=None;self.base_background=None
        self.pressed=set();self.drag=None;self.rectangle=None;self.original={}
        self.oblique=[];self.slope_drag=None;self.link_manager=None
        self._local_history=PlotHistory();self._slope_before=None;self._zoom_before=None;self._axis_generation=0
        canvas.destroyed.connect(lambda *_:self.suspend())
        self.connections=[canvas.mpl_connect(name,handler) for name,handler in (
            ('draw_event',self._drawn),('motion_notify_event',self.motion),
            ('button_press_event',self.press),('button_release_event',self.release),
            ('figure_leave_event',self.leave),('key_press_event',self.key),('resize_event',self._resized))]
        self.reset_axes()
    def history(self):
        owner=self.canvas.window()
        return getattr(owner,'graph_history',self._local_history)
    def history_controllers(self):
        controllers=[self]
        if self.link_manager is not None:
            for link in self.link_manager.links:
                if self in (link.first_controller,link.second_controller):
                    controllers.extend((link.first_controller,link.second_controller))
        return list(dict.fromkeys(controllers))
    def snapshot(self):
        points=tuple((self.axes.index(p.axes),p.x,p.y) for p in self.points)
        oblique=tuple((self.axes.index(p.axes),p.x,p.y,None if np.isinf(p.slope) else float(p.slope)) for p in self.oblique)
        views=tuple((index,tuple(float(v) for v in ax.get_xlim()),tuple(float(v) for v in ax.get_ylim()),
                     tuple(locator_state(getattr(axis,'get_'+kind+'_locator')()) for axis in (ax.xaxis,ax.yaxis) for kind in ('major','minor')))
                    for index,ax in enumerate(self.axes))
        return dict(generation=self._axis_generation,axes_count=len(self.axes),points=points,oblique=oblique,views=views,signature=(points,oblique,views))
    def accepts_snapshot(self,state):
        from PyQt5 import sip
        return not sip.isdeleted(self.canvas) and self._axis_generation==state['generation'] and len(self.axes)==state['axes_count'] and all(ax in self.canvas.figure.axes for ax in self.axes)
    def restore_snapshot(self,state):
        if not self.accepts_snapshot(state):return
        self._cancel_drag();self._hide_hover()
        for point in self.points+self.oblique:
            for artist in point.artists:artist.remove()
        self.points=[];self.oblique=[]
        for index,xlim,ylim,locators in state['views']:
            ax=self.axes[index]
            ax.set_xlim(xlim);ax.set_ylim(ylim)
            for axis,pair in zip((ax.xaxis,ax.yaxis),(locators[:2],locators[2:])):
                axis.set_major_locator(restore_locator(pair[0]));axis.set_minor_locator(restore_locator(pair[1]))
        for index,x,y in state['points']:self._create_point(self.axes[index],x,y)
        for index,x,y,slope in state['oblique']:self._create_oblique(self.axes[index],x,y,np.inf if slope is None else slope)
        self.background=None;self.base_background=None;self.canvas.draw_idle()
    def clear_artists(self):
        """Detach overlays before a Figure clears/replaces its axes."""
        self._axis_generation+=1
        for point in self.points+self.oblique:
            for artist in point.artists:artist.remove()
        for artists in self.hover.values():
            for artist in artists:artist.remove()
        if self.rectangle is not None:self.rectangle.remove()
        self.points=[];self.oblique=[];self.slope_drag=None;self._slope_before=None;self._zoom_before=None;self.hover={};self.drag=None;self.rectangle=None;self.background=None;self.base_background=None;self.pressed.clear()
    def reset_axes(self):
        """Call after replacing chart data. Annotations belong to that data/view."""
        self.clear_artists()
        self.axes=list(self.canvas.figure.axes)
        self.original={ax:(ax.get_xlim(),ax.get_ylim()) for ax in self.axes}
        for ax in self.axes:
            v=ax.axvline(0,color='#76509c',lw=.8,ls='--',visible=False,animated=True)
            h=ax.axhline(0,color='#76509c',lw=.8,ls='--',visible=False,animated=True)
            text=ax.text(.02,.98,'',transform=ax.transAxes,va='top',fontsize=8,
                         color=colors(self.canvas)['text'],bbox=dict(facecolor=colors(self.canvas)['panel'],alpha=.9,edgecolor=colors(self.canvas)['border']),visible=False,animated=True)
            self.hover[ax]=(v,h,text)
            xlim,ylim=self.original[ax];ax.set_xlim(xlim);ax.set_ylim(ylim)
            ax.callbacks.connect('xlim_changed',self._limits_changed)
            ax.callbacks.connect('ylim_changed',self._limits_changed)
    def _limits_changed(self,ax):
        self.background=None;self.base_background=None
        for point in self.oblique:
            if point.axes is ax:self._update_oblique(point,point.slope)
    def _resized(self,event):
        self._finish_slope(cancel=True);self.background=None;self.base_background=None
    def _create_oblique(self,ax,x,y,slope):
        # A normal Line2D uses set_data on every frame. AxLine.set_slope left
        # cached transforms unchanged on older supported Matplotlib versions.
        line=ax.plot([],[],color=colors(self.canvas)['guide'],lw=1,animated=True)[0]
        label=ax.annotate('',(x,y),xytext=(8,-18),textcoords='offset points',color=colors(self.canvas)['guide'],fontsize=8,animated=True)
        point=PinnedPoint(ax,x,y,[line,label],slope);self.oblique.append(point)
        self._update_oblique(point,slope)
        return point
    def _update_oblique(self,point,slope):
        ax=point.axes;x0,y0=point.x,point.y;point.slope=float(slope)
        line,label=point.artists
        direction=np.array([0.,1.]) if np.isinf(slope) else np.array([1.,slope])/np.hypot(1.,slope)
        low=-np.inf;high=np.inf
        for value,delta,bounds in zip((x0,y0),direction,(sorted(ax.get_xlim()),sorted(ax.get_ylim()))):
            if abs(delta)<1e-15:
                if value<bounds[0] or value>bounds[1]:low=1.;high=0.;break
            else:
                a,b=sorted(((bounds[0]-value)/delta,(bounds[1]-value)/delta));low=max(low,a);high=min(high,b)
        if low<=high:
            endpoints=np.array([x0,y0])+np.array([low,high])[:,None]*direction;line.set_data(endpoints[:,0],endpoints[:,1])
        else:line.set_data([],[])
        label.set_text(f'x = {x0:.7g}' if np.isinf(slope) else f'y = {slope:.7g} x {y0-slope*x0:+.7g}')
    def _drawn(self,event):
        if event.canvas is not self.canvas:return
        if self.canvas.is_saving():self.background=None;self.base_background=None;return
        if tuple(self.axes)!=tuple(self.canvas.figure.axes):
            self.reset_axes()
            changed=self.history().changed
            if changed:changed()
        self.base_background=self.canvas.copy_from_bbox(self.canvas.figure.bbox)
        self._bake_points()
        self._paint_overlays()
    def _bake_points(self):
        if self.base_background is None:self.canvas.draw_idle();return
        self.canvas.restore_region(self.base_background)
        for point in self.points+self.oblique:
            for artist in point.artists:artist.axes.draw_artist(artist)
        self.background=self.canvas.copy_from_bbox(self.canvas.figure.bbox)

    def _blit(self):
        from PyQt5 import sip
        if sip.isdeleted(self.canvas) or not self.canvas.isVisible():return
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
        if not self.axes:return
        self._hide_hover()
        if self.drag is not None:
            ax,x0,y0=self.drag
            x,y=self._coordinates(ax,event.x,event.y)
            self.rectangle.set_bounds(min(x0,x),min(y0,y),abs(x-x0),abs(y-y0));self._blit();return
        if self.slope_drag is not None:
            ax,x0,y0,px,py,point=self.slope_drag
            if point is None and np.hypot(event.x-px,event.y-py)>6:
                x,y=ax.transData.inverted().transform((event.x,event.y))
                slope=(y-y0)/(x-x0) if abs(x-x0)>max(abs(x0),1)*1e-12 else np.inf
                point=self._create_oblique(ax,x0,y0,slope)
                self.slope_drag=(ax,x0,y0,px,py,point)
            if point is not None:
                x,y=ax.transData.inverted().transform((event.x,event.y))
                self._update_oblique(point,(y-y0)/(x-x0) if abs(x-x0)>max(abs(x0),1)*1e-12 else np.inf)
                self._bake_points();self._blit();return
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
            text.set_color(colors(self.canvas)['text'])
            if self.link_manager is not None:
                extra,color=self.link_manager.hover(ax,event.xdata)
                if extra:label+='\n'+extra;text.set_color(readable_color(color,self.canvas))
            text.set_text(label)
            for artist in (v,h,text):artist.set_visible(True)
        self._blit()
    def leave(self,event):
        self._finish_slope(cancel=True);self._hide_hover();self.pressed.clear()
        if self.drag is not None:self._cancel_drag()
        self._blit()
    def suspend(self):
        """Cancel a hidden/dying widget's draft without painting or Qt calls.

        A Qt Hide also occurs during deletion. Restoring a chart snapshot here
        would enqueue draw_idle on a widget whose paint device is going away.
        Completed guides and the history remain untouched.
        """
        if self.slope_drag is not None:
            point=self.slope_drag[-1]
            if point is not None and point in self.oblique:
                self.oblique.remove(point)
                for artist in point.artists:artist.remove()
        self.slope_drag=None;self._slope_before=None;self._zoom_before=None
        self._cancel_drag();self._hide_hover();self.pressed.clear()
        self.background=None;self.base_background=None
    def _coordinates(self,ax,x,y):
        pixel=(np.clip(x,ax.bbox.x0,ax.bbox.x1),np.clip(y,ax.bbox.y0,ax.bbox.y1))
        return ax.transData.inverted().transform(pixel)
    def nearest(self,event):
        """Closest line segment in screen pixels; preserve NaN gaps and twinx units."""
        candidates=[];cursor=np.array([event.x,event.y])
        for ax in self.axes:
            if event.inaxes is None or not np.allclose(ax.bbox.bounds,event.inaxes.bbox.bounds):continue
            ignored=set(self.hover.get(ax,()))
            ignored.update(a for point in self.points+self.oblique for a in point.artists)
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
            self._finish_slope(cancel=True);self._cancel_drag();self.clear_points();return
        if event.button==MouseButton.MIDDLE and (any(k in (event.key or '').lower() for k in ('ctrl','control')) or set(getattr(event,'modifiers',())) & {'ctrl','control'}):
            self.reset_zoom();return
        if event.button==MouseButton.MIDDLE and event.inaxes in self.axes:
            self._zoom_before=self.history().capture(self.history_controllers())
            self._hide_hover();ax=event.inaxes;x,y=self._coordinates(ax,event.x,event.y)
            self.drag=(ax,x,y);self.rectangle=Rectangle((x,y),0,0,facecolor='#b9a1d4',edgecolor='#76509c',alpha=.3,animated=True)
            ax.add_patch(self.rectangle);return
        if event.button==MouseButton.LEFT and event.dblclick and len(self.points)<self.MAX_POINTS:
            point=self.nearest(event)
            if point is None:return
            ax,x,y=point
            self._finish_slope()
            with self.history().action('Точка на графике',self.history_controllers()):
                self.add_point(ax,x,y)
                if self.link_manager is not None:self.link_manager.pin(ax,x)
        elif event.button==MouseButton.LEFT and not event.dblclick and event.inaxes in self.axes:
            point=self.nearest(event)
            if point is not None:
                ax,x,y=point
                if sum(p.axes is ax for p in self.oblique)<10:
                    self._slope_before=self.history().capture(self.history_controllers())
                    self.slope_drag=(ax,x,y,event.x,event.y,None)
    def add_point(self,ax,x,y):
        if len(self.points)>=self.MAX_POINTS:return
        with self.history().action('Точка на графике',self.history_controllers()):
            self._create_point(ax,x,y);self._bake_points();self._blit()
    def _create_point(self,ax,x,y):
        color=colors(self.canvas)['pin']
        artists=[ax.axvline(x,color=color,lw=.7,ls=':'),ax.axhline(y,color=color,lw=.7,ls=':'),
                 ax.plot([x],[y],'o',color=color,ms=4)[0],
                 ax.annotate(f'{len(self.points)+1}: ({x:.7g}, {y:.7g})',(x,y),xytext=(5,6),textcoords='offset points',fontsize=7,color=color)]
        for artist in artists:artist.set_animated(True)
        self.points.append(PinnedPoint(ax,x,y,artists))
    def clear_oblique(self):
        self._finish_slope()
        with self.history().action('Удаление наклонных осей',self.history_controllers()):
            for point in self.oblique:
                for artist in point.artists:artist.remove()
            self.oblique=[];self._bake_points();self._blit()
    def _finish_slope(self,cancel=False):
        before=self._slope_before;self._slope_before=None;self.slope_drag=None
        if before is not None:
            if cancel:
                for reference,state in before:
                    controller=reference()
                    if controller is not None:controller.restore_snapshot(state)
            else:self.history().record('Наклонная ось',before,self.history_controllers())
    def _cancel_drag(self):
        self.slope_drag=None;self.drag=None
        if self.rectangle is not None:self.rectangle.remove();self.rectangle=None
    def release(self,event):
        self.pressed.discard(event.button)
        if event.button==MouseButton.LEFT:self._finish_slope()
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
        if self._zoom_before is not None:
            self.history().record('Масштаб графика',self._zoom_before,self.history_controllers());self._zoom_before=None
    def clear_points(self,propagate=True):
        with self.history().action('Удаление точек',self.history_controllers()):
            for point in self.points:
                for artist in point.artists:artist.remove()
            self.points=[];self._bake_points();self._blit()
            if propagate and self.link_manager is not None:self.link_manager.clear(self)
    def reset_zoom(self):
        with self.history().action('Исходный масштаб графика',self.history_controllers()):
            self._cancel_drag()
            for ax,(xlim,ylim) in self.original.items():ax.set_xlim(xlim);ax.set_ylim(ylim)
            self.canvas.draw_idle()
    def key(self,event):
        if event.key=='r':self.reset_zoom()
        elif event.key=='escape':self._finish_slope(cancel=True);self._cancel_drag();self._hide_hover();self._blit()
        elif event.key=='delete':self.clear_oblique()
        elif event.key in ('ctrl+z','control+z'):self.history().undo()
        elif event.key in ('ctrl+y','ctrl+shift+z','control+y'):self.history().redo()


def attach_interaction(canvas):
    from PyQt5.QtCore import Qt
    canvas.setFocusPolicy(Qt.ClickFocus)
    controller=PlotInteraction(canvas);canvas.interaction=controller
    canvas.setToolTip('Наведение: прицел и x/y. Двойной ЛКМ: точка на эпюре (до 10). '\
                      'Удерживать и перемещать ЛКМ: наклонная ось (до 10 на эпюру); Delete: удалить наклонные оси. ЛКМ + ПКМ: удалить точки. Средняя кнопка + выделение: масштаб. Ctrl + средняя кнопка или R: исходный масштаб.')
    return controller
