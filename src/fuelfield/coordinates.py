"""Canonical geometry is +X/CCW; the measurement frame is an explicit transform."""
import numpy as np

def to_global(angle,cfg):
    return (cfg.theta_zero_deg+cfg.theta_direction*np.asarray(angle,dtype=float))%360

def from_global(angle,cfg):
    return (cfg.theta_direction*(np.asarray(angle,dtype=float)-cfg.theta_zero_deg))%360

def grid_angles(cfg,edges=False):
    indices=np.arange(cfg.n_angular+int(edges))-(.5 if edges else 0)
    # Do not wrap edges: the polar pcolormesh must keep their continuous order.
    angles=cfg.theta_zero_deg+cfg.theta_direction*(cfg.grid_start_deg+indices*360/cfg.n_angular)
    return angles if edges else angles%360

def angular_indices(x,y,cfg):
    delta=2*np.pi/cfg.n_angular
    first=np.deg2rad(cfg.theta_zero_deg+cfg.theta_direction*cfg.grid_start_deg)
    angles=(cfg.theta_direction*(np.arctan2(y,x)-first)+delta/2)%(2*np.pi)
    return np.floor(angles/delta).astype(int).clip(0,cfg.n_angular-1)

def plot_x(global_angles,cfg,mode='angle'):
    angles=from_global(global_angles,cfg)
    return angles*cfg.n_angular/360 if mode=='cells' else angles

def period(cfg,mode='angle'):return cfg.n_angular if mode=='cells' else 360

def x_label(cfg,mode='angle'):
    return f'Окружная ячейка (0…{cfg.n_angular}; шаг {360/cfg.n_angular:.7g}°)' if mode=='cells' else 'Окружной угол, °'

def decorate_axes(ax,cfg,mode='angle'):
    ax._compare_x='cell_index' if mode=='cells' else 'angle_deg'
    ax._compare_frame=(float(cfg.theta_zero_deg)%360,int(cfg.theta_direction),int(cfg.n_angular) if mode=='cells' else None)

def polar_ticks(ax,cfg):
    labels=np.arange(0,360,45);ax.set_thetagrids(to_global(labels,cfg),labels=[f'{v}°' for v in labels])
