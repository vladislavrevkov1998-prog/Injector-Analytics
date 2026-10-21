"""Conservative polar trajectory bins and a smooth display-only XY preview.

The polar bins retain every transported kg/s, including a separate film on
each wall. A small conservative blur and Cartesian bilinear interpolation
improve display resolution only. Neither operation changes trajectories,
the audited section matrix, momentum, or the separately reported film flow.
"""
from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np


DISPLAY_NOTICE=('Отображение: мелкая полярная гистограмма тех же траекторий, '
    'визуальное сглаживание и интерполяция в XY. Расчётная матрица и баланс '
    'используют исходные консервативные веса; плёнка отдельно сохранена в пристенных ячейках. '
    'Интеграл интерполированной XY-карты приближённый и не является проверкой баланса.')


@dataclass
class PolarTransportPreview:
    radial_edges_mm: np.ndarray
    angular_centers_rad: np.ndarray
    suspended_cell_mass_kg_s: np.ndarray
    film_inner_cell_mass_kg_s: np.ndarray
    film_outer_cell_mass_kg_s: np.ndarray

    @property
    def cell_mass_kg_s(self):
        return (self.suspended_cell_mass_kg_s+self.film_inner_cell_mass_kg_s
                +self.film_outer_cell_mass_kg_s)

    @property
    def cell_area_m2(self):
        return np.repeat((np.pi/len(self.angular_centers_rad)*
            np.diff((self.radial_edges_mm*.001)**2))[:,None],
            len(self.angular_centers_rad),axis=1)

    @property
    def flux_kg_m2_s(self):return self.cell_mass_kg_s/self.cell_area_m2


def _display_smoothing(mass,film=False):
    from scipy.ndimage import gaussian_filter,gaussian_filter1d
    total=float(mass.sum())
    if total==0:return mass.copy()
    if film:
        result=gaussian_filter1d(mass,1.2,axis=1,mode='wrap')
    else:result=gaussian_filter(mass,(1.2,1.2),mode=('reflect','wrap'))
    # Keep integral conservation explicit even across SciPy versions and
    # floating-point reduction order. No particle trajectory is modified.
    result*=total/float(result.sum())
    return result


def build_polar_preview(nozzles,cfg,z_mm,*,radial_cells=128,angular_cells=720,
                        unit_mass=False,smooth=True):
    """Return a fine polar mass grid from the same physical parcel histories.

    Angular bins are fixed to global +X/CCW, independent of measurement-frame
    presentation. Film occupies the adjacent inner/outer radial row and is
    blurred only in angle. Suspended mass is blurred with reflecting radial
    boundaries. All three phase integrals remain unchanged.
    """
    from .core import confined_cell_masses,source_flow_kg_s
    from .fuel_transport import confined_samples,validate_transport_source,transport_model
    from .v02 import effective_nozzle
    if (isinstance(radial_cells,bool) or not isinstance(radial_cells,int)
            or not 4<=radial_cells<=600 or isinstance(angular_cells,bool)
            or not isinstance(angular_cells,int) or not 36<=angular_cells<=2880):
        raise ValueError('Неверная дискретизация полярного поля переноса топлива')
    rin,rout=cfg.bounds_at(z_mm);edges=np.linspace(rin,rout,radial_cells+1)
    bins=SimpleNamespace(n_radial=radial_cells,n_angular=angular_cells,
        grid_start_deg=0.,theta_zero_deg=0.,theta_direction=1)
    phases=[np.zeros((radial_cells,angular_cells)) for _ in range(3)]
    model=transport_model(cfg)
    for nozzle in nozzles:
        n=effective_nozzle(nozzle,cfg)
        if not n.enabled:continue
        if not unit_mass and source_flow_kg_s(n,cfg)<=0:continue
        validate_transport_source(n,cfg,model)
        if z_mm<n.z_tip_mm:continue
        samples=confined_samples(n,cfg,z_mm,unit_mass=unit_mass)
        for target,addition in zip(phases,confined_cell_masses(samples,bins,edges)):
            target+=addition
    if smooth:phases=[_display_smoothing(values,index>0) for index,values in enumerate(phases)]
    return PolarTransportPreview(edges,np.arange(angular_cells)*2*np.pi/angular_cells,*phases)


def evaluate_transport_field(nozzles,cfg,z_mm,resolution=None,bounds_xy=None,
                              mask_track=True,*,unit_mass=False):
    """Interpolate the fine conservative polar display onto a regular XY grid."""
    from .core import SmoothField
    dim=int(cfg.preview_resolution if resolution is None else resolution)
    if not 64<=dim<=1200:raise ValueError('Неверное разрешение карты.')
    rin,rout=cfg.bounds_at(z_mm)
    if bounds_xy is None:bounds_xy=(-rout,rout,-rout,rout)
    xmin,xmax,ymin,ymax=bounds_xy
    if not all(np.isfinite(v) for v in bounds_xy) or xmin>=xmax or ymin>=ymax:
        raise ValueError('Некорректные границы карты.')
    # A boundary-cell film thinner than one Cartesian pixel can disappear
    # entirely between pixel centres. Match the fine radial bins to XY pixel
    # spacing, including a narrow profiled outlet, rather than oversampling r.
    pixel=max(xmax-xmin,ymax-ymin)/(dim-1)
    radial_cells=max(4,min(384,int((rout-rin)/pixel)))
    polar=build_polar_preview(nozzles,cfg,z_mm,
        radial_cells=radial_cells,angular_cells=max(180,min(1440,dim*2)),
        unit_mass=unit_mass)
    x_axis=np.linspace(xmin,xmax,dim);y_axis=np.linspace(ymin,ymax,dim)
    x,y=np.meshgrid(x_axis,y_axis);radius=np.hypot(x,y)
    valid=(radius>=rin)&(radius<=rout)
    nr,nt=polar.suspended_cell_mass_kg_s.shape
    position=np.clip((radius-rin)/(rout-rin)*nr-.5,0,nr-1)
    r0=np.floor(position).astype(int);r1=np.minimum(r0+1,nr-1);rw=position-r0
    position=(np.arctan2(y,x)%(2*np.pi))*nt/(2*np.pi)
    a0=np.floor(position).astype(int)%nt;a1=(a0+1)%nt;aw=position-np.floor(position)
    density=polar.flux_kg_m2_s
    low=(1-aw)*density[r0,a0]+aw*density[r0,a1]
    high=(1-aw)*density[r1,a0]+aw*density[r1,a1]
    flux=np.where(valid,(1-rw)*low+rw*high,0.)
    masked=np.ma.masked_where(~valid,flux) if mask_track else np.ma.array(flux,mask=False)
    return SmoothField(x_axis,y_axis,flux,masked,float(z_mm),display_notice=DISPLAY_NOTICE)
