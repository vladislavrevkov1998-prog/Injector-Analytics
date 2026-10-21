"""One-way inertial fuel parcels in a prescribed axisymmetric carrier.

Each quadrature parcel carries a mass *flow*. Geometry integration never
depends on the output bins, observed sectors or source flow. Sidewall capture
and prescribed forward film transport are explicit engineering closures.
"""
from collections import OrderedDict
from dataclasses import dataclass
import json
import threading
import numpy as np
from .chamber import Chamber, chamber, legacy_geometry
from .carrier import build_carrier
from .transport_settings import TransportSettings
from .transport_contacts import first_contacts, wall_radius, film_velocity, wall_arc_length

NOTICE=('Инерционный перенос: заданный идеальный носитель; однонаправленное влияние на капли. '
        'Оси источников направлены вдоль +z; наклоны объектов не используются. '
        'При первом контакте 100% расхода переходит в отдельно учитываемую плёнку. '
        'Это параметризованная модель, без расчёта воздуха, отрыва, дробления, внутренних тел/их следов и обратного влияния топлива.')
_CACHE=OrderedDict()
_CACHE_BYTES=0
_LOCK=threading.RLock()
_LIMIT=96*1024**2


def transport_enabled(cfg):
    return (cfg.fuel_transport or {}).get('mode','legacy')=='confined_inertial'


def transport_model(cfg):
    model=chamber(cfg)
    if model:return model
    raw=legacy_geometry(cfg);raw['enabled']=True
    return Chamber(raw)


def validate_transport_source(n,cfg,model=None):
    """Validate a working source even if every requested plane precedes it.

    A valid downstream source is unreached before its tip. A source outside
    the physical tract must never disappear silently from all flow balances.
    """
    from .v02 import effective_nozzle
    n=effective_nozzle(n,cfg);model=model or transport_model(cfg)
    if not np.isfinite(n.z_tip_mm) or not model.base-1e-9<=n.z_tip_mm<=model.outlet+1e-9:
        raise ValueError(f'Форсунка {n.id}: срез сопла должен находиться внутри осевой области переноса')
    rin,rout=model.bounds(n.z_tip_mm)
    if not np.isfinite(n.radius_mm) or not rin-1e-8<=n.radius_mm<=rout+1e-8:
        raise ValueError(f'Форсунка {n.id}: источник топлива находится вне закрытого тракта')
    return n


def clear_transport_cache():
    global _CACHE_BYTES
    with _LOCK:_CACHE.clear();_CACHE_BYTES=0


def transport_cache_info():
    with _LOCK:return dict(entries=len(_CACHE),bytes=_CACHE_BYTES,limit_bytes=_LIMIT)


@dataclass(frozen=True)
class TransportSamples:
    x_mm: np.ndarray
    y_mm: np.ndarray
    weights_kg_s: np.ndarray
    phase: np.ndarray
    velocity_m_s: np.ndarray
    source_velocity_m_s: np.ndarray
    carrier_delta_v_m_s: np.ndarray
    wall_delta_v_m_s: np.ndarray
    travel_time_s: np.ndarray
    notice: str=NOTICE


def _time_for_distance(v0,ug,dz_m,tau):
    """Positive time for the exact frozen-carrier axial relaxation update."""
    if np.any(v0<=1e-10) or np.any(ug< -1e-10):
        raise ValueError('Инерционный перенос: обратная/нулевая осевая скорость; осевое марширование неприменимо')
    ug=np.maximum(ug,0.)
    if np.any((ug<=1e-12)&(dz_m>=tau*v0)):
        raise ValueError('Инерционный перенос: частица останавливается до следующего шага; уменьшите шаг или измените физические параметры')
    lo=np.zeros_like(v0)
    hi=np.where(ug>1e-12,dz_m/np.maximum(np.minimum(v0,ug),1e-300),
                -tau*np.log1p(-np.minimum(dz_m/(tau*v0),1-1e-15)))
    t=np.minimum(dz_m/v0,hi)
    for _ in range(64):
        e=np.exp(-t/tau)
        f=ug*t+(v0-ug)*tau*(-np.expm1(-t/tau))-dz_m
        done=np.abs(f)<max(1e-14,dz_m*1e-11)
        if np.all(done):return t
        lo=np.where(f<0,t,lo);hi=np.where(f>=0,t,hi)
        trial=t-f/(ug+(v0-ug)*e)
        t=np.where(done,t,np.where((trial>lo)&(trial<hi),trial,(lo+hi)/2))
    raise ValueError('Не сошлось определение времени переноса; результат не применён')


def _emission(n,cfg,p):
    from .core import angular_profile
    if cfg.geometry_model=='two_angles':lo,hi=float(n.inner_angle_deg),float(n.outer_angle_deg)
    else:
        if not np.isfinite(n.alpha_deg) or not 0<=n.alpha_deg<179.8:
            raise ValueError(f'Форсунка {n.id}: полный угол истечения должен быть 0…179.8°')
        lo,hi=max(0.,n.alpha_deg/2-p.cone_half_width_deg),n.alpha_deg/2+p.cone_half_width_deg
    if not 0<=lo<=hi<89.9:raise ValueError(f'Форсунка {n.id}: для прямого переноса нужны 0 ≤ углы истечения < 89.9°')
    nr,na=p.emission_radial_samples,p.emission_angular_samples
    cosb=np.cos(np.deg2rad(lo))+(np.arange(nr)+.5)/nr*(np.cos(np.deg2rad(hi))-np.cos(np.deg2rad(lo)))
    # Subdivide each observed 10-degree bin independently. A global midpoint
    # grid for odd multiples of 36 lands exactly on sector boundaries and
    # floating-point floor can duplicate/omit a measured sector.
    per_sector=na//36
    sector_index=np.repeat(np.arange(36),per_sector)
    phi=np.deg2rad(sector_index*10.-5.+np.tile((np.arange(per_sector)+.5)*10./per_sector,36))
    direction=phi+np.deg2rad((180. if n.row=='outer' else 0.)+n.clock_deg)
    v=np.empty((nr,na,3));v[:,:,0]=p.injection_speed_m_s*cosb[:,None]
    v[:,:,1]=p.injection_speed_m_s*np.sqrt(1-cosb[:,None]**2)*np.cos(direction)
    v[:,:,2]=p.injection_speed_m_s*np.sqrt(1-cosb[:,None]**2)*np.sin(direction)
    sector=angular_profile(n,n.z_tip_mm,cfg)
    ap=sector[sector_index]
    ap=ap/ap.sum()
    return v.reshape(-1,3),np.tile(ap/nr,nr)


def _trace(n,cfg,p,model,source,targets):
    carrier=build_carrier(model.raw,p.carrier_inlet_velocity_m_s,max_step_mm=p.max_step_mm,
                          radial_cells=p.carrier_radial_cells,wall_tolerance_mm=p.wall_tolerance_mm)
    count=len(source);xy=np.zeros((count,2));xy[:,0]=n.radius_mm
    velocity=source.copy();drag=np.zeros_like(velocity);travel=np.zeros(count)
    phase=np.zeros(count,dtype=np.uint8);hit_z=np.zeros(count);hit_theta=np.zeros(count)
    hit_time=np.zeros(count);hit_drag=np.zeros_like(velocity)
    rng=np.random.default_rng(np.random.SeedSequence([p.seed,int(n.id)]))
    noise=rng.normal(0,p.turbulence_rms_m_s,(count,2)) if p.turbulence_rms_m_s else np.zeros((count,2))
    stations=np.r_[n.z_tip_mm,carrier.z_mm[carrier.z_mm>n.z_tip_mm+1e-10]]
    output={};cursor=0

    def snapshot(z,positions,vel,cd,times,ph):
        nonlocal cursor
        wd=vel-source-cd
        arrays=(positions.copy(),vel.copy(),cd.copy(),wd.copy(),times.copy(),ph.copy())
        for value in arrays:value.setflags(write=False)
        output[float(z)]=arrays;cursor+=1

    if targets and abs(targets[0]-n.z_tip_mm)<1e-10:snapshot(targets[0],xy,velocity,drag,travel,phase)
    for z0,z1 in zip(stations,stations[1:]):
        if cursor==len(targets):break
        active=np.flatnonzero(phase==0);next_xy=xy.copy();next_v=velocity.copy();next_drag=drag.copy();next_time=travel.copy()
        fraction=np.full(count,np.inf);contacts=np.zeros(count,dtype=np.uint8)
        if len(active):
            a=active;ug=carrier.velocity(z0,xy[a,0],xy[a,1]);ug[:,1:]+=noise[a]
            dt=_time_for_distance(velocity[a,0],ug[:,0],(z1-z0)*.001,p.relaxation_time_s)
            exp=np.exp(-dt/p.relaxation_time_s)
            next_v[a]=ug+(velocity[a]-ug)*exp[:,None]
            next_xy[a]=xy[a]+1000*(ug[:,1:]*dt[:,None]+(velocity[a,1:]-ug[:,1:])*p.relaxation_time_s*(-np.expm1(-dt/p.relaxation_time_s))[:,None])
            f,w=first_contacts(model,float(z0),float(z1),xy[a],next_xy[a]);fraction[a]=f;contacts[a]=w
            next_drag[a]=drag[a]+next_v[a]-velocity[a];next_time[a]=travel[a]+dt
            impacted=np.flatnonzero(w>0)
            if len(impacted):
                ids=a[impacted];fac=f[impacted]
                hit_z[ids]=z0+(z1-z0)*fac
                point=xy[ids]+fac[:,None]*(next_xy[ids]-xy[ids])
                hit_theta[ids]=np.arctan2(point[:,1],point[:,0])
                hit_time[ids]=travel[ids]+fac*dt[impacted]
                vc=ug[impacted]+(velocity[ids]-ug[impacted])*np.exp(-fac*dt[impacted]/p.relaxation_time_s)[:,None]
                hit_drag[ids]=drag[ids]+vc-velocity[ids]
        while cursor<len(targets) and targets[cursor]<=z1+1e-10:
            z=targets[cursor];fout=(z-z0)/(z1-z0)
            pos=xy+fout*(next_xy-xy);vel=velocity+fout*(next_v-velocity)
            cd=drag+fout*(next_drag-drag);times=travel+fout*(next_time-travel)
            if len(active):
                a=active;vel[a]=ug+(velocity[a]-ug)*np.exp(-fout*dt/p.relaxation_time_s)[:,None]
                cd[a]=drag[a]+vel[a]-velocity[a]
            ph=phase.copy();new=(contacts>0)&(fraction<=fout+1e-12);ph[new]=contacts[new]
            film=np.flatnonzero(ph>0)
            if len(film):
                r=wall_radius(model,ph[film],z)
                pos[film]=np.c_[r*np.cos(hit_theta[film]),r*np.sin(hit_theta[film])]
                vel[film]=film_velocity(model,ph[film],z,hit_theta[film],p.film_speed_m_s)
                cd[film]=hit_drag[film]
                times[film]=hit_time[film]+wall_arc_length(model,ph[film],hit_z[film],z)*.001/p.film_speed_m_s
            snapshot(z,pos,vel,cd,times,ph)
        new=contacts>0;phase[new]=contacts[new]
        xy,velocity,drag,travel=next_xy,next_v,next_drag,next_time
        if len(active) and p.turbulence_rms_m_s:
            corr=np.exp(-dt/p.turbulence_time_s)
            noise[active]=corr[:,None]*noise[active]+p.turbulence_rms_m_s*np.sqrt(-np.expm1(-2*dt/p.turbulence_time_s))[:,None]*rng.normal(size=(len(active),2))
    if cursor!=len(targets):raise ValueError('Сечение за пределами области переноса')
    return output


def _rotate(a,angle):
    result=a.copy();c,s=np.cos(angle),np.sin(angle)
    result[:, -2]=c*a[:,-2]-s*a[:,-1];result[:,-1]=s*a[:,-2]+c*a[:,-1]
    return result


def confined_samples(n,cfg,z_mm,*,unit_mass=False):
    from .v02 import effective_nozzle
    from .core import source_flow_kg_s
    global _CACHE_BYTES
    n=effective_nozzle(n,cfg);p=TransportSettings.from_dict(cfg.fuel_transport).validate()
    if not p.enabled:raise ValueError('Выберите инерционную модель переноса топлива')
    z_mm=float(z_mm)
    if not np.isfinite(z_mm):raise ValueError('Нужна конечная координата сечения')
    model=transport_model(cfg)
    validate_transport_source(n,cfg,model)
    if not model.base-1e-9<=z_mm<=model.outlet+1e-9:raise ValueError('Сечение вне закрытого тракта')
    if z_mm<n.z_tip_mm:
        empty=np.empty(0);vectors=np.empty((0,3))
        return TransportSamples(empty,empty,empty,empty.astype(np.uint8),vectors,vectors,vectors,vectors,empty)
    source,weights=_emission(n,cfg,p)
    targets=tuple(sorted({float(z) for z in (*cfg.z_mm,z_mm) if n.z_tip_mm-1e-10<=z<=model.outlet+1e-10}))
    physical={key:model.raw.get(key) for key in ('base_z_mm','outlet_z_mm','primitives')}
    params=p.to_dict();params.pop('provenance',None)
    # Flow, sector observations, output radial/angular grids and global rotation
    # cannot alter these one-way trajectories. Nozzle-specific OU streams can.
    key=json.dumps((physical,params,n.radius_mm,n.z_tip_mm,n.row,n.alpha_deg,n.clock_deg,
                    cfg.geometry_model,n.inner_angle_deg,n.outer_angle_deg,n.id if p.turbulence_rms_m_s else 0,targets),sort_keys=True,allow_nan=False)
    with _LOCK:result=_CACHE.get(key)
    if result is None:
        try:result=_trace(n,cfg,p,model,source,targets)
        except ValueError as exc:raise ValueError(f'Форсунка {n.id}: {exc}') from exc
        size=sum(a.nbytes for arrays in result.values() for a in arrays)
        if size<_LIMIT:
            with _LOCK:
                if key not in _CACHE:
                    while _CACHE and _CACHE_BYTES+size>_LIMIT:
                        _,removed=_CACHE.popitem(last=False);_CACHE_BYTES-=sum(a.nbytes for arrays in removed.values() for a in arrays)
                    _CACHE[key]=result;_CACHE_BYTES+=size
    else:
        with _LOCK:
            if key in _CACHE:_CACHE.move_to_end(key)
    xy,v,cd,wd,t,phase=result[z_mm];angle=np.deg2rad(n.theta_deg)
    xy=_rotate(xy,angle)
    scale=1. if unit_mass else source_flow_kg_s(n,cfg)
    return TransportSamples(xy[:,0],xy[:,1],weights*scale,phase,_rotate(v,angle),
                            _rotate(source,angle),_rotate(cd,angle),_rotate(wd,angle),t)
