"""Conservative fuel-flux superposition. Units: geometry mm, flows kg/h.

The source-polar midpoint quadrature has strictly normalized discrete mass weights.
It is an engineering quadrature, not a DPM or interaction model.
"""
from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Optional
import numpy as np
from .v02 import (GEOMETRY_KEYS, default_presets, validate_catalog, effective_nozzle,
                  flow_status, validate_nozzles)

GEOMETRY_MODELS = {
    'constant': 'Конус: постоянная толщина кольца',
    'linear': 'Конус: линейный рост толщины',
    'quadratic': 'Конус: квадратичный рост толщины',
    'power': 'Конус: степенной рост толщины',
    'two_angles': 'Независимые углы внутренней / наружной границ',
}
ANGULAR_MODELS = {
    'preserve': 'Сохранение измеренной эпюры',
    'linear': 'Линейное затухание к равномерной',
    'exponential': 'Экспоненциальное затухание к равномерной',
    'diffusion': 'Круговое гауссово размытие',
}
RADIAL_MODELS = {
    'uniform': 'Равномерно по толщине кольца',
    'triangular': 'Треугольный профиль',
    'gaussian': 'Гауссов профиль (толщина = FWHM)',
}

@dataclass
class Nozzle:
    id: int
    row: str = 'outer'
    theta_deg: float = 0.0
    radius_mm: float = 270.0
    enabled: bool = True
    flow_kg_h: float = 0.0
    flow_mode: str = 'manual'  # manual | sectors
    sector_ml: list[float] = field(default_factory=lambda: [1.0] * 36)
    sector_source: str = 'шаблон: равномерно'
    alpha_deg: float = 45.0        # full opening angle of peak-ring centre
    rc0_mm: float = 10.0
    h0_mm: float = 4.0
    k_linear: float = 0.08        # mm thickness per mm axial distance
    k_quadratic: float = 0.0004  # thickness mm per z_mm**2
    power: float = 0.7
    z_ref_mm: float = 50.0
    inner_angle_deg: float = 10.0
    outer_angle_deg: float = 30.0
    clock_deg: float = 0.0
    decay_length_mm: float = 70.0
    diffusion_deg_sqrt_mm: float = 1.0
    rotation_deg_per_mm: float = 0.0
    atomizer_type_id: str = ''
    flow_pressure_kgf_cm2: float = 30.0
    sector_pressure_kgf_cm2: float = 2.5
    capture_efficiency: Optional[float] = None
    nozzle_kind: str = 'uncooled'
    geometry_overrides: list[str] = field(default_factory=lambda:list(GEOMETRY_KEYS))

    @property
    def r_nozzle_mm(self):
        return self.rc0_mm+self.h0_mm/2

@dataclass
class Settings:
    r_in_mm: float = 200.0
    r_out_mm: float = 300.0
    row_inner_mm: float = 230.0
    row_outer_mm: float = 270.0
    z_mm: list[float] = field(default_factory=lambda: [0.0, 20.0, 40.0, 60.0])
    n_radial: int = 6
    n_angular: int = 140
    grid_start_deg: float = 0.0  # angular position of CENTER of first matrix cell
    radial_centers_mm: Optional[list[float]] = None
    geometry_model: str = 'linear'
    angular_model: str = 'preserve'
    radial_model: str = 'gaussian'
    n_sample_radial: int = 80
    n_sample_angular: int = 720
    duration_s: float = 180.0
    density_kg_m3: float = 800.0
    nozzle_diameter_mm: float = 16.0  # drawn nominal outline; user-adjustable
    preview_resolution: int = 480  # Cartesian detail-map resolution
    # Display preferences only; never used by the physical kernels.
    color_scale_mode: str = 'auto'
    color_vmin: float = 0.0
    color_vmax: float = 1.0
    color_out_of_range: str = 'saturate'
    color_shared_z: bool = False
    atomizer_types: list[dict] = field(default_factory=list)
    nozzle_presets: dict = field(default_factory=default_presets)
    target_pressure_kgf_cm2: float = 30.0
    pressure_mode: str = 'legacy'  # legacy preserves imported v0.1 sector flow
    allow_extrapolation: bool = False
    placement_mode: str = 'manual'
    display_flow_unit: str = 'kg/h'

    def validate(self):
        validate_catalog(self.atomizer_types,self.nozzle_presets)
        if not isinstance(self.allow_extrapolation,bool):raise ValueError('Разрешение экстраполяции должно быть логическим значением.')
        if not all(np.isfinite(v) for v in (self.r_in_mm,self.r_out_mm,self.row_inner_mm,
                    self.row_outer_mm,self.duration_s,self.density_kg_m3,self.grid_start_deg)):
            raise ValueError('Размеры, время и плотность должны быть конечными.')
        if not np.isfinite(self.target_pressure_kgf_cm2) or self.target_pressure_kgf_cm2<=0:
            raise ValueError('Целевой перепад давления должен быть положительным.')
        if self.pressure_mode not in ('legacy','calibrated') or self.placement_mode not in ('manual','auto_follow'):
            raise ValueError('Неизвестный режим пересчёта или расстановки.')
        if self.display_flow_unit not in ('kg/h','l/min'):raise ValueError('Неизвестные единицы расхода.')
        if self.color_scale_mode not in ('auto', 'manual'):
            raise ValueError('Шкала должна быть Auto или Manual.')
        if self.color_out_of_range not in ('saturate', 'clip'):
            raise ValueError('Режим шкалы должен быть Saturate или Clip to range.')
        if not (np.isfinite(self.color_vmin) and np.isfinite(self.color_vmax)
                and self.color_vmin < self.color_vmax):
            raise ValueError('Шкала: нужны конечные min < max.')
        if not (0 < self.r_in_mm < self.r_out_mm):
            raise ValueError('Требуется 0 < Rвнутр < Rнаруж.')
        if len(self.z_mm) != 4 or not all(np.isfinite(z) for z in self.z_mm) or min(self.z_mm) < 0:
            raise ValueError('Нужны 4 неотрицательных расстояния z [мм].')
        if not 1 <= self.n_radial <= 40 or not 8 <= self.n_angular <= 1440:
            raise ValueError('Число уровней 1..40, окружных ячеек 8..1440.')
        if self.n_sample_radial < 8 or self.n_sample_angular < 72:
            raise ValueError('Слишком мало интеграционных узлов. Минимум 8 и 72.')
        if self.duration_s <= 0 or self.density_kg_m3 <= 0:
            raise ValueError('Время опыта и плотность должны быть положительными.')
        if not (0 < self.nozzle_diameter_mm <= 500):
            raise ValueError('Диаметр контура форсунки должен быть положительным и не больше 500 мм.')
        if not 200 <= self.preview_resolution <= 1000:
            raise ValueError('Разрешение подробной карты: 200..1000.')
        for value, pool, label in ((self.geometry_model, GEOMETRY_MODELS, 'геометрия'),
                                   (self.angular_model, ANGULAR_MODELS, 'неравномерность'),
                                   (self.radial_model, RADIAL_MODELS, 'радиальный профиль')):
            if value not in pool:
                raise ValueError(f'Неизвестная модель «{label}»: {value}')
        self.radial_centers_and_edges()

    def radial_centers_and_edges(self):
        if self.radial_centers_mm is None:
            centers = self.r_in_mm + (np.arange(self.n_radial) + 0.5) * (
                (self.r_out_mm - self.r_in_mm) / self.n_radial)
        else:
            centers = np.asarray(self.radial_centers_mm, dtype=float)
            if len(centers) != self.n_radial:
                raise ValueError('Число заданных радиусов должно равняться числу радиальных уровней.')
            if (not np.all(np.isfinite(centers)) or
                np.any(centers <= self.r_in_mm) or np.any(centers >= self.r_out_mm) or
                np.any(np.diff(centers) <= 0)):
                raise ValueError('Радиальные уровни должны возрастать и находиться строго внутри тракта.')
        edges = np.concatenate(([self.r_in_mm], (centers[:-1] + centers[1:]) / 2,
                                [self.r_out_mm]))
        return centers, edges


def default_nozzles(settings: Settings) -> list[Nozzle]:
    """70 outer at theta=0+5.142857*k, 70 inner at +2.571429 degrees."""
    rows = []
    for k in range(70):
        rows.append(Nozzle(id=k+1, row='outer', theta_deg=k*360/70,
                           radius_mm=settings.row_outer_mm))
    for k in range(70):
        rows.append(Nozzle(id=k+71, row='inner', theta_deg=(k+0.5)*360/70,
                           radius_mm=settings.row_inner_mm))
    return rows


def apply_default_positions(nozzles: list[Nozzle], settings: Settings):
    """Change *only* nozzle coordinates, never enable flags/measurements/ids."""
    outer = [n for n in nozzles if n.row == 'outer']
    inner = [n for n in nozzles if n.row == 'inner']
    for k, n in enumerate(outer):
        n.theta_deg = k*360/len(outer) if outer else 0
        n.radius_mm = settings.row_outer_mm
    for k, n in enumerate(inner):
        n.theta_deg = (k+0.5)*360/len(inner) if inner else 0
        n.radius_mm = settings.row_inner_mm


def source_flow_kg_s(n: Nozzle, cfg: Settings) -> float:
    return flow_status(n,cfg).kg_s


def geometry_at(n: Nozzle, z_mm: float, cfg: Settings):
    n=effective_nozzle(n,cfg)
    if n.h0_mm <= 0 or n.rc0_mm < 0:
        raise ValueError(f'Форсунка {n.id}: Rc0 >=0 и h0 >0 обязательны.')
    z = float(z_mm)
    if cfg.geometry_model == 'two_angles':
        lo = n.rc0_mm - n.h0_mm/2 + z*np.tan(np.deg2rad(n.inner_angle_deg))
        hi = n.rc0_mm + n.h0_mm/2 + z*np.tan(np.deg2rad(n.outer_angle_deg))
        rc, h = (lo+hi)/2, hi-lo
    else:
        if abs(n.alpha_deg) >= 179:
            raise ValueError(f'Форсунка {n.id}: некорректный угол факела.')
        rc = n.rc0_mm + z*np.tan(np.deg2rad(n.alpha_deg/2))
        if cfg.geometry_model == 'constant':
            h = n.h0_mm
        elif cfg.geometry_model == 'linear':
            h = n.h0_mm + n.k_linear*z
        elif cfg.geometry_model == 'quadratic':
            h = n.h0_mm + n.k_linear*z + n.k_quadratic*z*z
        else:
            if n.z_ref_mm <= 0:
                raise ValueError(f'Форсунка {n.id}: z_ref должно быть >0.')
            h = n.h0_mm*(1+z/n.z_ref_mm)**n.power
    if not np.isfinite(rc+h) or rc < 0 or h <= 0 or rc > 1e5 or h > 1e5:
        raise ValueError(f'Форсунка {n.id}, z={z}: геометрия не имеет физического смысла.')
    return float(rc), float(h)


def angular_profile(n: Nozzle, z_mm: float, cfg: Settings):
    n=effective_nozzle(n,cfg)
    a = np.asarray(n.sector_ml, dtype=float)
    if len(a) != 36 or not np.isfinite(a).all() or np.min(a) < 0:
        raise ValueError(f'Форсунка {n.id}: нужны 36 неотрицательных измеренных объёмов.')
    if np.sum(a) <= 0:
        # Running a nozzle with no sector observations falls back to uniform:
        a = np.ones(36, dtype=float)
    a = 36.0 * a / np.sum(a)  # mean == 1, irrespective of collected volume
    z = max(0.0, z_mm)
    if cfg.angular_model == 'linear':
        lam = max(0, 1-z/max(1e-12, n.decay_length_mm))
        a = 1+lam*(a-1)
    elif cfg.angular_model == 'exponential':
        lam = np.exp(-z/max(1e-12, n.decay_length_mm))
        a = 1+lam*(a-1)
    elif cfg.angular_model == 'diffusion':
        sigma = np.deg2rad(max(0.0, n.diffusion_deg_sqrt_mm)*np.sqrt(z))
        if sigma:
            harmonics = np.arange(19, dtype=float)
            coefficients = np.fft.rfft(a)
            coefficients *= np.exp(-0.5*(harmonics*sigma)**2)
            a = np.fft.irfft(coefficients, n=36)
            a = np.maximum(a, 0)
            a *= 36/a.sum()
    return a


def sample_kernel(n: Nozzle, z_mm: float, cfg: Settings):
    """Return source-position points [mm] and *exactly* normalized flow weights [kg/s].

    Sector 1 is centered toward the track middle line; sectors 2..36 are CCW
    in the chosen XY view from the fuel outlet. Rotation parameter in degrees.
    """
    rc, h = geometry_at(n, z_mm, cfg)
    if cfg.radial_model == 'gaussian':
        sigma = h/(2*np.sqrt(2*np.log(2)))  # FWHM of Gaussian profile = h
        rlo, rhi = max(0, rc-6*sigma), rc+6*sigma
    else:
        rlo, rhi = max(0, rc-h/2), rc+h/2
    if rhi <= rlo:
        raise ValueError(f'Форсунка {n.id}: нулевая ширина факела.')
    r = rlo + (np.arange(cfg.n_sample_radial)+0.5)*(rhi-rlo)/cfg.n_sample_radial
    if cfg.radial_model == 'uniform':
        radial = np.ones_like(r)
    elif cfg.radial_model == 'triangular':
        radial = np.maximum(0, 1 - np.abs(r-rc)/(h/2))
    else:
        radial = np.exp(-0.5*((r-rc)/sigma)**2)
    wr = radial*r  # area Jacobian in local polar coordinates
    wr = wr/wr.sum()
    phi = (np.arange(cfg.n_sample_angular)+0.5)*2*np.pi/cfg.n_sample_angular
    sector = angular_profile(n,z_mm,cfg)
    # Sector measurements are 10-degree BIN INTEGRALS, not pointwise samples.
    # Piecewise-constant angular density therefore preserves the 36 measured
    # sector shares at z=0 without inventing extra smoothing of adjacent bins.
    # Sector 1 is centered at 0 deg, hence boundaries at +/- 5 degrees.
    sector_idx = np.floor((phi*36/(2*np.pi) + 0.5)%36).astype(int)
    ap = sector[sector_idx]
    ap = ap/ap.sum()  # exact conservation even for non-multiple-of-36 grids
    # Global polar direction to first sector: inner outward, outer inward.
    base_deg = n.theta_deg + (180.0 if n.row == 'outer' else 0.0)
    facing = np.deg2rad(base_deg + n.clock_deg + n.rotation_deg_per_mm*z_mm)
    direction = phi+facing
    xpos = (n.radius_mm*np.cos(np.deg2rad(n.theta_deg)) +
            r[:,None]*np.cos(direction)[None,:])
    ypos = (n.radius_mm*np.sin(np.deg2rad(n.theta_deg)) +
            r[:,None]*np.sin(direction)[None,:])
    weights = (source_flow_kg_s(n,cfg)*wr[:,None]*ap[None,:]).reshape(-1)
    return xpos.reshape(-1), ypos.reshape(-1), weights

@dataclass
class PlaneResult:
    z_mm: float
    cell_mass_kg_s: np.ndarray
    cell_area_m2: np.ndarray
    flux_kg_m2_s: np.ndarray
    radial_centers_mm: np.ndarray
    radial_edges_mm: np.ndarray
    theta_centers_deg: np.ndarray
    injected_kg_s: float
    inside_kg_s: float
    inner_kg_s: float
    outer_kg_s: float

    @property
    def rel_balance_error(self):
        denominator = max(self.injected_kg_s, 1e-15)
        return (self.inside_kg_s+self.inner_kg_s+self.outer_kg_s-self.injected_kg_s)/denominator


def calculate_plane(nozzles: list[Nozzle], cfg: Settings, z_mm: float) -> PlaneResult:
    cfg.validate()
    centers, edges = cfg.radial_centers_and_edges()
    nrad, nang = cfg.n_radial, cfg.n_angular
    dtheta = 2*np.pi/nang
    theta0 = np.deg2rad(cfg.grid_start_deg)
    mass = np.zeros((nrad,nang))
    inner = outer = injected = 0.0
    for n in nozzles:
        n=effective_nozzle(n,cfg)
        if not n.enabled: continue
        flow = source_flow_kg_s(n,cfg)
        if flow <= 0: continue
        if not np.isfinite(flow):
            raise ValueError(f'Форсунка {n.id}: некорректный расход.')
        injected += flow
        xp,yp,w = sample_kernel(n,z_mm,cfg)
        radial = np.hypot(xp,yp)
        ir = np.searchsorted(edges,radial,side='right')-1
        mask_in = (ir >= 0)&(ir<nrad)
        inner += float(np.sum(w[ir<0]))
        outer += float(np.sum(w[ir>=nrad]))
        if np.any(mask_in):
            theta = (np.arctan2(yp[mask_in],xp[mask_in])-theta0+dtheta/2)%(2*np.pi)
            ith = np.floor(theta/dtheta).astype(int).clip(0,nang-1)
            cell_id = ir[mask_in]*nang+ith
            mass += np.bincount(cell_id,weights=w[mask_in],minlength=nrad*nang).reshape(nrad,nang)
    area = (np.pi/nang)*(np.diff((edges*1e-3)**2))
    area = np.repeat(area[:,None],nang,axis=1)
    flux = mass/area
    th_centers = (cfg.grid_start_deg+np.arange(nang)*360/nang)%360
    return PlaneResult(float(z_mm),mass,area,flux,centers,edges,th_centers,
                       injected,float(mass.sum()),inner,outer)


def project_to_json(settings: Settings, nozzles: list[Nozzle]):
    return {'format':'fuel-field-pilot-v1','version':'0.2','settings':asdict(settings),
            'nozzles':[{**asdict(n),'r_nozzle_mm':n.r_nozzle_mm} for n in nozzles]}


def project_from_json(data):
    if data.get('format') != 'fuel-field-pilot-v1':
        raise ValueError('Неподдерживаемый формат проекта')
    cfg = Settings(**data['settings'])
    nozzles=[]
    for item in data['nozzles']:
        raw=dict(item)
        radius=raw.pop('r_nozzle_mm',None)
        if radius is not None:
            derived=radius-raw.get('h0_mm',4.)/2
            if 'rc0_mm' in raw and not np.isclose(derived,raw['rc0_mm'],rtol=1e-12,atol=1e-12):
                raise ValueError('Несогласованные R_сопла и legacy Rc0.')
            raw.setdefault('rc0_mm',derived)
        nozzles.append(Nozzle(**raw))
    cfg.validate()
    if len(nozzles) != 140 or set(n.id for n in nozzles)!=set(range(1,141)):
        raise ValueError('В v0.1 проект должен содержать 140 уникальных форсунок')
    validate_nozzles(nozzles,cfg)
    return cfg,nozzles


@dataclass
class SmoothField:
    """Continuous visual preview, not the mass-audited 6×140 output.

    flux_kg_m2_s is obtained by evaluating the independently normalized
    per-nozzle physical kernels on a finer Cartesian grid. A periodic linear
    interpolation of the measured 36-bin angular averages is used for display;
    it preserves their angular mean, but NOT the exact 10-degree bin integrals.
    The conservative numeric output remains calculate_plane().
    """
    x_mm: np.ndarray
    y_mm: np.ndarray
    flux_kg_m2_s: np.ndarray
    masked_flux_kg_m2_s: np.ma.MaskedArray
    z_mm: float


def _radial_density(r, rc, h, model):
    if model == 'uniform':
        return ((r >= max(0.,rc-h/2)) & (r <= rc+h/2)).astype(float)
    if model == 'triangular':
        return np.maximum(0.,1.-np.abs(r-rc)/(h/2))
    sigma = h/(2*np.sqrt(2*np.log(2)))
    return np.exp(-0.5*((r-rc)/sigma)**2)


def _kernel_outer_limit(rc,h,model):
    if model == 'gaussian':
        return rc + 6*h/(2*np.sqrt(2*np.log(2)))
    return rc + h/2


def evaluate_continuous_field(nozzles: list[Nozzle], cfg: Settings,
                              z_mm: float, resolution: Optional[int] = None) -> SmoothField:
    """High-resolution XY preview of superposed local polar source kernels.

    Uses analytic-area density per nozzle, sampled on a regular XY grid. This
    is NOT fed back into conservation reports, which remain based on exact
    normalized source-quadrature weights in calculate_plane().
    """
    cfg.validate()
    dim = int(cfg.preview_resolution if resolution is None else resolution)
    if not 64 <= dim <= 1200:
        raise ValueError('Неверное разрешение карты.')
    extent = float(cfg.r_out_mm)
    axis = np.linspace(-extent,extent,dim)
    flux = np.zeros((dim,dim),dtype=np.float64)
    # Do not construct a full dim×dim intermediate for every nozzle. A bounded
    # window is sufficient because outside 6 sigma the Gaussian is negligible.
    for n in nozzles:
        n=effective_nozzle(n,cfg)
        if not n.enabled:
            continue
        m_dot = source_flow_kg_s(n,cfg)
        if m_dot <= 0:
            continue
        rc,h = geometry_at(n,z_mm,cfg)
        rhi = _kernel_outer_limit(rc,h,cfg.radial_model)
        if rhi <= 0:
            continue
        xp = n.radius_mm*np.cos(np.deg2rad(n.theta_deg))
        yp = n.radius_mm*np.sin(np.deg2rad(n.theta_deg))
        ix0 = max(0,int(np.searchsorted(axis,xp-rhi,side='left')))
        ix1 = min(dim,int(np.searchsorted(axis,xp+rhi,side='right')))
        iy0 = max(0,int(np.searchsorted(axis,yp-rhi,side='left')))
        iy1 = min(dim,int(np.searchsorted(axis,yp+rhi,side='right')))
        if ix0 >= ix1 or iy0 >= iy1:
            continue
        xx = axis[None,ix0:ix1]-xp
        yy = axis[iy0:iy1,None]-yp
        rr = np.hypot(xx,yy)
        radial = _radial_density(rr,rc,h,cfg.radial_model)
        if not np.any(radial):
            continue
        # Radial normalization: integral_0^rhi f(r)*r*dr [mm²].
        rs = np.linspace(0,rhi,1501)
        integrand = rs*_radial_density(rs,rc,h,cfg.radial_model)
        integral = float(np.sum((integrand[1:]+integrand[:-1])*np.diff(rs)/2))
        if integral <= 0:
            continue
        # Sector 1 faces the middle of the annulus, CCW sector numbering.
        sector = angular_profile(n,z_mm,cfg)
        facing = np.deg2rad(n.theta_deg + (180 if n.row=='outer' else 0)
                               + n.clock_deg+n.rotation_deg_per_mm*z_mm)
        angle = (np.arctan2(yy,xx)-facing)%(2*np.pi)
        phase = angle*36/(2*np.pi)
        indices = np.floor(phase).astype(np.int32)
        weight = phase-indices
        # Periodic piecewise-linear interpolation between bin CENTER values.
        # Since the 36 values have mean 1, the circular integral equals 2π.
        ang = sector[indices%36]*(1-weight) + sector[(indices+1)%36]*weight
        flux[iy0:iy1,ix0:ix1] += (m_dot*1e6/(2*np.pi*integral))*radial*ang
    xx,yy = np.meshgrid(axis,axis)
    r_global = np.hypot(xx,yy)
    valid = (r_global >= cfg.r_in_mm)&(r_global <= cfg.r_out_mm)
    return SmoothField(axis,axis,flux,np.ma.masked_where(~valid,flux),float(z_mm))


def polar_epure_coordinates(n: Nozzle, z_mm: float, cfg: Settings,
                             relative_radius_mm: float, include_flow: bool = False,
                             flow_reference_kg_s: float = 1.0, n_angles: int = 144):
    """XY polar 'rose' of one injector's measured angular *intensity*.

    A plot glyph: radial distance is NOT the real boundary of the spray.
    The baseline is dimensionless a(phi); optional amplitude includes flow.
    """
    n=effective_nozzle(n,cfg)
    phi = np.linspace(0,2*np.pi,n_angles+1)
    amp36 = angular_profile(n,z_mm,cfg)
    phase = (phi%(2*np.pi))*36/(2*np.pi)
    left = np.floor(phase).astype(int)
    w = phase-left
    amp = amp36[left%36]*(1-w)+amp36[(left+1)%36]*w
    amp = amp*(source_flow_kg_s(n,cfg)/max(1e-16,flow_reference_kg_s) if include_flow else 1)
    facing = np.deg2rad(n.theta_deg + (180 if n.row=='outer' else 0)
                           + n.clock_deg + n.rotation_deg_per_mm*z_mm)
    x0 = n.radius_mm*np.cos(np.deg2rad(n.theta_deg))
    y0 = n.radius_mm*np.sin(np.deg2rad(n.theta_deg))
    return x0+relative_radius_mm*amp*np.cos(phi+facing), y0+relative_radius_mm*amp*np.sin(phi+facing)
