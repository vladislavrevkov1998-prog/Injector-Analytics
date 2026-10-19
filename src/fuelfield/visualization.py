"""Matplotlib-only display routines (no PyQt dependency).

Quantitative conservation is exclusively computed by core.calculate_plane().
The dense, angular-interpolated previews are visual diagnostics, not substitutes
for the controlled 6×140 collection cell averages.
"""
from __future__ import annotations
import numpy as np
from matplotlib.patches import Circle
from matplotlib.lines import Line2D
from matplotlib import patheffects
from matplotlib.colors import Normalize
from matplotlib import colormaps
from .core import polar_epure_coordinates, source_flow_kg_s
from .v02 import effective_nozzle


def color_scale(values, cfg, shared_values=None):
    """Return display data, norm and transparent colormap without mutating input.

    Existing physical/wall masks are preserved. Clipping hides only the field.
    """
    data = np.ma.masked_invalid(np.ma.array(values, copy=True))
    if cfg.color_scale_mode == 'manual':
        low, high = cfg.color_vmin, cfg.color_vmax
        if cfg.color_out_of_range == 'clip':
            data = np.ma.masked_outside(data, low, high)
    else:
        arrays = shared_values if cfg.color_shared_z and shared_values is not None else [data]
        finite = [np.ma.masked_invalid(a).compressed() for a in arrays]
        finite = [a for a in finite if a.size]
        low = min(float(a.min()) for a in finite) if finite else 0.0
        high = max(float(a.max()) for a in finite) if finite else 1.0
        if high <= low:
            high = low + max(1.0, abs(low)*0.01)
    cmap = colormaps['turbo'].with_extremes(bad=(0, 0, 0, 0))
    return data, Normalize(low, high, clip=True), cmap


def draw_nozzle_markers(ax, cfg, nozzles, *, polar=False, outlines=True, labels=True):
    """Projected circular injector mouths; labels remain tied to position IDs."""
    if not (outlines or labels):
        return
    phase = np.linspace(0, 2*np.pi, 49)
    stroke = [patheffects.withStroke(linewidth=1.35, foreground='#141923')]
    for n in nozzles:
        angle = np.deg2rad(n.theta_deg)
        x0, y0 = n.radius_mm*np.cos(angle), n.radius_mm*np.sin(angle)
        color = '#ffffff' if n.enabled else '#888888'
        if outlines:
            xx = x0+cfg.nozzle_diameter_mm/2*np.cos(phase)
            yy = y0+cfg.nozzle_diameter_mm/2*np.sin(phase)
            args = (np.arctan2(yy,xx),np.hypot(xx,yy)) if polar else (xx,yy)
            ax.plot(*args,color=color,alpha=.72 if n.enabled else .4,
                    lw=.7,ls='-' if n.enabled else '--',zorder=10)
        if labels:
            kw=dict(ha='center',va='center',fontsize=5,color='#fff' if n.enabled else '#aaa',
                    zorder=11,path_effects=stroke)
            if polar:
                tip=effective_nozzle(n,cfg).z_tip_mm
                ax.text(angle,n.radius_mm,str(n.id)+(f'\nzₛ={tip:g}' if tip else ''),**kw)
            else:
                tip=effective_nozzle(n,cfg).z_tip_mm
                ax.text(x0,y0,str(n.id)+(f'\nzₛ={tip:g}' if tip else ''),**kw)


def draw_polar_nozzle_profiles(ax, cfg, nozzles, z_mm, scale_mm, *, fill=False):
    """Relative angular intensity roses; NOT spray edge coordinates."""
    for n in nozzles:
        if not n.enabled or source_flow_kg_s(n,cfg)<=0:
            continue
        x,y=polar_epure_coordinates(n,z_mm,cfg,scale_mm,n_angles=144)
        colour='#f59e0b' if n.row=='outer' else '#22d3ee'
        ax.plot(x,y,color=colour,lw=.7,alpha=.9 if not fill else .65,zorder=8)
        if fill:
            ax.fill(x,y,color=colour,alpha=.075,zorder=7)


def draw_detailed_map(fig, field, plane, cfg, nozzles, *, outlines=True,
                      labels=True, roses=False, rose_scale_mm=8.0, shared_values=None):
    fig.clear()
    ax=fig.add_subplot(111)
    rin,lim=plane.radial_edges_mm[[0,-1]]
    data,norm,cmap=color_scale(field.masked_flux_kg_m2_s,cfg,shared_values)
    im=ax.imshow(data,origin='lower',interpolation='nearest',
                 extent=[-lim,lim,-lim,lim],cmap=cmap,norm=norm,aspect='equal')
    for radius in (rin,lim):
        ax.add_patch(Circle((0,0),radius,fill=False,lw=1.,edgecolor='#cbd5e1',zorder=9))
    if roses:
        draw_polar_nozzle_profiles(ax,cfg,nozzles,plane.z_mm,rose_scale_mm)
    draw_nozzle_markers(ax,cfg,nozzles,polar=False,outlines=outlines,labels=labels)
    ax.set(xlim=(-lim,lim),ylim=(-lim,lim),xlabel='X, мм',ylabel='Y, мм')
    ax.set_title(f'Подробное поле без агрегирования 6×140 · z={plane.z_mm:g} мм',fontsize=10)
    ax.grid(False)
    fig.colorbar(im,ax=ax,shrink=.8,label='кг/(м²·с)')
    return ax


def draw_polar_nozzles_map(fig, plane, cfg, nozzles, *, outlines=True,labels=True,
                           rose_scale_mm=8.0):
    fig.clear()
    ax=fig.add_subplot(111)
    rin,lim=plane.radial_edges_mm[[0,-1]]
    for radius in (rin,lim):
        ax.add_patch(Circle((0,0),radius,fill=False,edgecolor='#7c8491',lw=1))
    draw_polar_nozzle_profiles(ax,cfg,nozzles,plane.z_mm,rose_scale_mm,fill=True)
    draw_nozzle_markers(ax,cfg,nozzles,polar=False,outlines=outlines,labels=labels)
    ax.set(aspect='equal',xlim=(-lim,lim),ylim=(-lim,lim),xlabel='X, мм',ylabel='Y, мм')
    ax.set_title(f'Индивидуальные полярные эпюры · z={plane.z_mm:g} мм',fontsize=11)
    ax.legend(handles=[Line2D([0],[0],color='#f59e0b',lw=2,label='Наружный ряд'),
                       Line2D([0],[0],color='#22d3ee',lw=2,label='Внутренний ряд')],
              loc='upper right',fontsize=8)
    ax.text(.02,.02,'Радиус полярной эпюры ∝ относительной интенсивности, НЕ геометрии факела',
            transform=ax.transAxes,fontsize=7,va='bottom',color='#777777')
    return ax
