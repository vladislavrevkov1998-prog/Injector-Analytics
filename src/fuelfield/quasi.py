"""Fixed-step fuel scanning and instantaneous fields. No temperatures or Qt."""
from dataclasses import dataclass,field,asdict,replace
from copy import deepcopy
import json,hashlib,time,warnings
import numpy as np
from scipy.sparse import csc_matrix
from .core import sample_kernel,PlaneResult,SmoothField,evaluate_continuous_field,project_to_json
from .v02 import flow_status,curve_for,bench_pressure,effective_nozzle,validate_nozzles
from .characteristics import ExtrapolationWarning
from .coordinates import angular_indices,grid_angles,to_global
from .chamber import chamber,WALL_BOUND_NOTICE

FORMAT='fuel-quasistationary-settings'
PRESSURE_FACTORS={'kgf/cm2':1.,'кгс/см²':1.,'bar':1e5/98066.5,'Pa':1/98066.5,'kPa':1000/98066.5,'MPa':1e6/98066.5}
KINDS=('delta_p','p_fuel_abs','p_fuel_gauge')

class CalculationCancelled(Exception):pass

class CalculationOutput(list):
    """Own completed data independently of an already deleted Qt worker."""
    def __init__(self,values,quasi_result=None,fields=()):
        super().__init__(values);self.quasi_result=quasi_result;self.fields=list(fields)

def check_cancel(cancel):
    if cancel and cancel():raise CalculationCancelled('Расчёт отменён; частичный результат не применяется')

def finite(v,label):
    try:v=float(v)
    except (ValueError,TypeError):raise ValueError(label+': нужно число') from None
    if not np.isfinite(v):raise ValueError(label+': нужно конечное число')
    return v

@dataclass(frozen=True)
class PressureChannel:
    id: str
    pressure: tuple
    unit: str='kgf/cm2'
    kind: str='delta_p'
    gas_pressure: tuple=()
    gas_unit: str='kgf/cm2'
    reference_id: str=''
    def __post_init__(self):
        object.__setattr__(self,'pressure',tuple(self.pressure));object.__setattr__(self,'gas_pressure',tuple(self.gas_pressure))
    def delta_p(self,count):
        if not isinstance(self.id,str) or not self.id.strip() or self.id!=self.id.strip():raise ValueError('Укажите непустое имя коллектора без крайних пробелов')
        if self.kind not in KINDS or self.unit not in PRESSURE_FACTORS or self.gas_unit not in PRESSURE_FACTORS:raise ValueError(f'{self.id}: неверный вид или единицы давления')
        if len(self.pressure)!=count:raise ValueError(f'{self.id}: требуется {count} значений давления, по одному на каждый шаг')
        p=np.array([finite(v,f'{self.id}, шаг {i}: давление') for i,v in enumerate(self.pressure)])*PRESSURE_FACTORS[self.unit]
        if self.kind=='delta_p':
            if self.gas_pressure:raise ValueError(f'{self.id}: для готового Δp газ повторно не вычитается; очистите газовый ряд')
            result=p
        else:
            if len(self.gas_pressure)!=count:raise ValueError(f'{self.id}: для давления подачи нужны {count} значений давления газа в том же базисе')
            gas=np.array([finite(v,f'{self.id}, шаг {i}: давление газа') for i,v in enumerate(self.gas_pressure)])*PRESSURE_FACTORS[self.gas_unit]
            if self.kind=='p_fuel_abs' and (np.any(p<0) or np.any(gas<0)):raise ValueError(f'{self.id}: абсолютное давление не может быть отрицательным')
            if self.kind=='p_fuel_gauge' and not str(self.reference_id).strip():raise ValueError(f'{self.id}: укажите reference_id общей базы избыточных давлений топлива и газа')
            result=p-gas
        if np.any(result<=0):raise ValueError(f'{self.id}: физический Δp должен быть >0 на каждом шаге')
        return result

@dataclass(frozen=True)
class QuasiSettings:
    n_steps: int=140
    dt_s: float=1.
    t_start_s: float=0.
    first_angle_deg: float=0.
    direction: int=1  # +1/-1 relative to the positive measurement angle on 01.
    default_manifold: str='A'
    channels: tuple=()
    metadata: dict=field(default_factory=dict)
    def __post_init__(self):
        for key in ('dt_s','t_start_s','first_angle_deg'):object.__setattr__(self,key,finite(getattr(self,key),key))
        object.__setattr__(self,'channels',tuple(self.channels));object.__setattr__(self,'metadata',deepcopy(self.metadata))
    @property
    def times(self):return self.t_start_s+np.arange(self.n_steps)*self.dt_s
    @property
    def angles(self):return (self.first_angle_deg+self.direction*np.arange(self.n_steps)*360/self.n_steps)%360
    def global_angles(self,cfg):return to_global(self.angles,cfg)
    def validate(self,cfg=None,nozzles=None):
        if isinstance(self.n_steps,bool) or not isinstance(self.n_steps,int) or not 8<=self.n_steps<=1440:raise ValueError('Число шагов: целое 8…1440, равно числу окружных ячеек на 01')
        if finite(self.dt_s,'Шаг времени')<=0:raise ValueError('Шаг времени должен быть положительным и постоянным')
        finite(self.t_start_s,'Начало времени');finite(self.first_angle_deg,'Первый угол')
        if self.direction not in (-1,1) or isinstance(self.direction,bool):raise ValueError('Направление: +1 вперёд или −1 обратно по углам системы на 01')
        if not isinstance(self.default_manifold,str) or not self.default_manifold.strip():raise ValueError('Укажите коллектор по умолчанию')
        ids=[c.id for c in self.channels]
        if not ids or len(ids)!=len(set(ids)):raise ValueError('Нужны коллекторы с уникальными именами')
        for channel in self.channels:channel.delta_p(self.n_steps)
        if cfg is not None:
            if self.n_steps!=cfg.n_angular:raise ValueError(f'В режиме {self.n_steps} шагов, на вкладке 01 {cfg.n_angular} окружных ячеек; числа должны совпадать')
            self.column_indices(cfg)
        if nozzles is not None:
            for n in nozzles:
                if n.enabled and self.manifold(n) not in ids:raise ValueError(f'Форсунка {n.id}: нет давления коллектора «{self.manifold(n)}»; проверьте таблицу 02')
        return self
    def manifold(self,n):return n.manifold_id.strip() or self.default_manifold
    def column_indices(self,cfg):
        raw=((self.angles-cfg.grid_start_deg)%360)*cfg.n_angular/360;nearest=np.rint(raw)
        if not np.allclose(raw,nearest,rtol=0,atol=1e-7):raise ValueError('Первый угол обхода должен совпадать с центром окружной ячейки сетки на 01')
        result=nearest.astype(int)%cfg.n_angular
        if len(set(result))!=cfg.n_angular:raise ValueError('Каждая окружная ячейка должна быть измерена ровно один раз')
        return result
    def pressures(self):return {c.id:c.delta_p(self.n_steps) for c in self.channels}
    def to_dict(self,bindings=None):
        raw=asdict(self);raw.update(format=FORMAT,version=1)
        if bindings is not None:raw['nozzle_manifolds']={str(k):str(v) for k,v in bindings.items()}
        return raw
    @classmethod
    def from_dict(cls,raw):
        if not isinstance(raw,dict) or raw.get('format')!=FORMAT or raw.get('version')!=1:raise ValueError('Нужен JSON настроек квазистационарного режима (*.quasi.json), а не JSON проекта/измерительной сессии')
        data=deepcopy(raw)
        for key in ('format','version','nozzle_manifolds'):data.pop(key,None)
        data['channels']=tuple(PressureChannel(**{**c,'pressure':tuple(c['pressure']),'gas_pressure':tuple(c.get('gas_pressure',()))}) for c in data.get('channels',[]))
        try:return cls(**data)
        except TypeError as exc:raise ValueError('Неверная структура настроек режима: '+str(exc)) from None

class FlowEvaluator:
    """Existing Q curves; manual/sectors references scaled once, without a guessed law."""
    def __init__(self,cfg):self.cfg=cfg;self.curves={};self.values={};self.bases={};self.notices=set()
    def curve_value(self,n,pressure):
        key=(n.atomizer_type_id,float(pressure))
        if key not in self.values:
            if n.atomizer_type_id not in self.curves:self.curves[n.atomizer_type_id]=curve_for(n,self.cfg)
            with warnings.catch_warnings():
                warnings.simplefilter('ignore',ExtrapolationWarning);estimate=self.curves[n.atomizer_type_id].evaluate(pressure,allow_extrapolation=self.cfg.allow_extrapolation)
            if estimate.extrapolated:self.notices.add('Разрешённая экстраполяция расходной характеристики')
            self.values[key]=estimate.flow_kg_h/3600
        return self.values[key]
    def base(self,n):
        if n.id not in self.bases:
            reference=bench_pressure(n,self.cfg,'flow' if n.flow_mode=='manual' else 'sector')
            status=flow_status(n,replace(self.cfg,target_pressure_kgf_cm2=reference));self.notices.update(status.notices)
            self.bases[n.id]=(status.kg_s,reference)
        return self.bases[n.id]
    def at(self,n,pressure):
        if not n.enabled:return 0.
        if n.flow_mode=='curve':return self.curve_value(n,pressure)
        base,reference=self.base(n)
        if base==0 or pressure==reference:return base
        ref=self.curve_value(n,reference)
        if ref<=0:raise ValueError(f'Форсунка {n.id}: Q при опорном давлении должна быть >0')
        return base*self.curve_value(n,pressure)/ref

@dataclass
class Influence:
    cfg: object
    z_mm: float
    weights: np.ndarray
    inner: np.ndarray
    outer: np.ndarray
    reached: np.ndarray
    area: np.ndarray
    radii: np.ndarray
    edges: np.ndarray
    angles: np.ndarray
    wall_inner: object=None
    wall_outer: object=None
    preview_weights: object=None
    preview_axis: object=None
    preview_mask: object=None
    upper_field: object=None
    suspended_weights: object=None
    film_inner_weights: object=None
    film_outer_weights: object=None
    momentum_weights: object=None
    travel_time_s: object=None
    transport_notice: str=''
    def plane(self,flows):
        mass=(self.weights@flows).reshape(self.area.shape)
        wi=float(self.wall_inner@flows) if self.wall_inner is not None else 0.;wo=float(self.wall_outer@flows) if self.wall_outer is not None else 0.
        transport={}
        if self.suspended_weights is not None:
            momentum={key:np.asarray(flows@value) for key,value in self.momentum_weights.items()}
            residual=momentum['outgoing_N']-momentum['incoming_N']-momentum['carrier_N']-momentum['wall_N']
            denominator=max(*(np.linalg.norm(value) for value in momentum.values()),1e-15)
            momentum={**{key:value.tolist() for key,value in momentum.items()},
                'residual_N':residual.tolist(),'relative_residual':float(np.linalg.norm(residual)/denominator)}
            transport=dict(suspended_cell_mass_kg_s=(self.suspended_weights@flows).reshape(self.area.shape),
                film_inner_cell_mass_kg_s=(self.film_inner_weights@flows).reshape(self.area.shape),
                film_outer_cell_mass_kg_s=(self.film_outer_weights@flows).reshape(self.area.shape),
                transport_momentum=momentum,transport_notice=self.transport_notice,
                max_travel_time_s=float(np.max(self.travel_time_s[np.asarray(flows)>0],initial=0.)))
        return PlaneResult(self.z_mm,mass,self.area,mass/self.area,self.radii,self.edges,self.angles,float(self.reached@flows),float(mass.sum()),float(self.inner@flows),float(self.outer@flows),
            wall_bound_kg_s=wi+wo,wall_inner_kg_s=wi,wall_outer_kg_s=wo,
            geometry_notice=self.transport_notice or (WALL_BOUND_NOTICE if self.wall_inner is not None else ''),**transport)
    def smooth(self,flows):
        if self.preview_weights is None:raise ValueError('Подробное поле не было подготовлено')
        dim=len(self.preview_axis);values=(self.preview_weights@flows).reshape(dim,dim)
        return SmoothField(self.preview_axis,self.preview_axis,values,np.ma.masked_where(self.preview_mask,values),self.z_mm)

def build_influence(nozzles,cfg,z,flows,preview=True,cancel=None):
    from .fuel_transport import transport_enabled
    confined=transport_enabled(cfg)
    radii,edges=cfg.radial_centers_and_edges(z);weights=np.zeros((cfg.n_radial*cfg.n_angular,len(nozzles)));inner=np.zeros(len(nozzles));outer=inner.copy();reached=inner.copy()
    model=chamber(cfg);variable=bool(model and model.variable)
    if confined:
        from .fuel_transport import confined_samples,validate_transport_source,transport_model
        from .core import confined_cell_masses,confined_momentum
        phase_weights=[np.zeros_like(weights) for _ in range(3)]
        momentum_weights={key:np.zeros((len(nozzles),3)) for key in ('incoming_N','outgoing_N','carrier_N','wall_N')}
        travel=np.zeros(len(nozzles));transport_notices=set()
        source_model=transport_model(cfg)
    for j,n in enumerate(nozzles):
        check_cancel(cancel)
        if confined and n.enabled and np.any(flows[:,j]>0):validate_transport_source(n,cfg,source_model)
        if not n.enabled or z<effective_nozzle(n,cfg).z_tip_mm:continue
        if confined:
            if not np.any(flows[:,j]>0):continue
            reached[j]=1.;samples=confined_samples(n,cfg,z,unit_mass=True)
            phases=confined_cell_masses(samples,cfg,edges)
            for target,values in zip(phase_weights,phases):target[:,j]=values.reshape(-1)
            weights[:,j]=sum(values[:,j] for values in phase_weights)
            momentum=confined_momentum(samples)
            for key in momentum_weights:momentum_weights[key][j]=momentum[key]
            travel[j]=float(np.max(samples.travel_time_s,initial=0.))
            if samples.notice:transport_notices.add(samples.notice)
            continue
        reached[j]=1.;xp,yp,w=sample_kernel(n,z,cfg,unit_mass=True);ir=np.searchsorted(edges,np.hypot(xp,yp),side='right')-1;inside=(ir>=0)&(ir<cfg.n_radial)
        inner[j]=w[ir<0].sum();outer[j]=w[ir>=cfg.n_radial].sum()
        ith=angular_indices(xp[inside],yp[inside],cfg)
        weights[:,j]=np.bincount(ir[inside]*cfg.n_angular+ith,weights=w[inside],minlength=weights.shape[0])
    if np.max(np.abs(weights.sum(axis=0)+inner+outer-reached))>1e-12:raise ValueError('Не замкнута нормировка единичных пространственных влияний')
    area=np.repeat((np.pi/cfg.n_angular*np.diff((edges*.001)**2))[:,None],cfg.n_angular,axis=1)
    op=Influence(deepcopy(cfg),float(z),weights,inner,outer,reached,area,radii,edges,grid_angles(cfg))
    if confined:
        op.suspended_weights,op.film_inner_weights,op.film_outer_weights=phase_weights
        op.momentum_weights=momentum_weights;op.travel_time_s=travel;op.transport_notice=' '.join(sorted(transport_notices))
    elif variable:op.wall_inner=inner.copy();op.wall_outer=outer.copy();op.inner=np.zeros_like(inner);op.outer=np.zeros_like(outer)
    if preview:
        dim=int(cfg.preview_resolution);columns=[];data=[];pointers=[0]
        for j,n in enumerate(nozzles):
            check_cancel(cancel)
            if n.enabled and reached[j] and np.any(flows[:,j]>0):
                field=evaluate_continuous_field([n],cfg,z,mask_track=False,unit_mass=True);flat=field.flux_kg_m2_s.reshape(-1);indices=np.flatnonzero(flat)
                columns.append(indices.astype(np.int32));data.append(flat[indices]);pointers.append(pointers[-1]+len(indices))
            else:pointers.append(pointers[-1])
        indices=np.concatenate(columns) if columns else np.array([],dtype=np.int32);values=np.concatenate(data) if data else np.array([],dtype=float)
        op.preview_weights=csc_matrix((values,indices,np.array(pointers)),shape=(dim*dim,len(nozzles))).tocsr()
        rin,rout=cfg.bounds_at(z);op.preview_axis=np.linspace(-rout,rout,dim);x,y=np.meshgrid(op.preview_axis,op.preview_axis);r=np.hypot(x,y);op.preview_mask=(r<rin)|(r>rout)
        op.upper_field=op.smooth(np.max(flows,axis=0)).masked_flux_kg_m2_s
    return op

@dataclass
class QuasiResult:
    cfg: object
    nozzles: tuple
    settings: QuasiSettings
    flows: np.ndarray
    pressures: dict
    operators: tuple
    composites: tuple
    balances: tuple
    notices: tuple
    elapsed_s: float
    project_payload: str
    @property
    def maximum_balance_error(self):return max((abs(b['relative_error']) for b in self.balances),default=0.)
    def snapshot(self,plane_index,step):return self.operators[plane_index].plane(self.flows[step])
    def smooth(self,plane_index,step):return self.operators[plane_index].smooth(self.flows[step])
    def color_max(self,plane_index,shared=None):
        ops=self.operators if (self.cfg.color_shared_z if shared is None else shared) else (self.operators[plane_index],)
        return max(1e-15,max(float(op.upper_field.max()) if op.upper_field.count() else 0. for op in ops))
    def manifest(self):
        confined=any(op.suspended_weights is not None for op in self.operators)
        transport={}
        if confined:
            maximum=max((plane.max_travel_time_s for plane in self.composites),default=0.)
            transport=dict(max_travel_time_s=maximum,scan_dt_s=self.settings.dt_s,
                travel_time_to_scan_step_ratio=maximum/self.settings.dt_s,
                pressure_response='Only prescribed source mass flow changes with pressure; injection velocity and trajectory parameters stay fixed; no transport delay or time-history response')
        return dict(format='fuel-quasistationary-result',version=1,project_sha256=hashlib.sha256(self.project_payload.encode()).hexdigest(),settings=self.settings.to_dict(),
            composite='One angular column per uniform timestamp; all radial cells simultaneously; density kg/(m2 s), not accumulated mass',
            composite_global_balance=None,maximum_instantaneous_balance_error=self.maximum_balance_error,
            spatial_model=('fixed unit inertial trajectories in a prescribed carrier; suspended fuel and two captured wall films; source mass flow follows pressure' if confined else 'fixed unit conservative kernels; no pressure-dependent shape; tau=0'),
            preview=('conservative per-source parcel reconstruction; sparse cached linear superposition at selected timestamp' if confined else 'analytic per-source kernels; sparse cached linear superposition at selected timestamp'),
            wall_bound=WALL_BOUND_NOTICE if not confined and chamber(self.cfg) and chamber(self.cfg).variable else None,
            color_auto='fixed across time; upper envelope from per-nozzle maximum flows',notices=list(self.notices),elapsed_s=self.elapsed_s,**transport)

def calculate_quasi(nozzles,cfg,*,preview=True,cancel=None,progress=None):
    started=time.perf_counter();cfg=deepcopy(cfg);nozzles=tuple(sorted(deepcopy(nozzles),key=lambda n:n.id));cfg.validate();validate_nozzles(nozzles,cfg)
    model=chamber(cfg)
    if model:model.validate_nozzle_anchors(nozzles,cfg)
    if len(nozzles)!=140 or {n.id for n in nozzles}!=set(range(1,141)):raise ValueError('Нужны 140 фиксированных позиций форсунок')
    settings=QuasiSettings.from_dict(cfg.quasi_settings).validate(cfg,nozzles);check_cancel(cancel);pressures=settings.pressures();flow=FlowEvaluator(cfg);g=np.zeros((settings.n_steps,len(nozzles)))
    for j,n in enumerate(nozzles):
        check_cancel(cancel)
        if n.enabled:
            for step,p in enumerate(pressures[settings.manifold(n)]):
                check_cancel(cancel);g[step,j]=flow.at(n,float(p))
    if not np.any(g>0):raise ValueError('Нет работающих форсунок с положительным расходом на шагах режима')
    if not np.isfinite(g).all() or np.any(g<0):raise ValueError('Получены некорректные расходы форсунок')
    cols=settings.column_indices(cfg);operators=[];planes=[];balances=[]
    for i,z in enumerate(cfg.z_mm):
        check_cancel(cancel)
        if progress:progress(f'Квазистационарный: сечение {i+1}/{len(cfg.z_mm)}, z={z:g} мм — влияния и кэш поля…')
        op=build_influence(nozzles,cfg,z,g,preview,cancel);operators.append(op)
        mass=np.empty(op.area.shape);times=np.empty(cfg.n_angular);steps=np.empty(cfg.n_angular,dtype=int);pcolumns=[None]*cfg.n_angular
        phase_mass=[np.empty(op.area.shape) for _ in range(3)] if op.suspended_weights is not None else None
        for step,col in enumerate(cols):
            check_cancel(cancel);mass[:,col]=op.weights.reshape(cfg.n_radial,cfg.n_angular,-1)[:,col,:]@g[step]
            if phase_mass is not None:
                for target,unit in zip(phase_mass,(op.suspended_weights,op.film_inner_weights,op.film_outer_weights)):
                    target[:,col]=unit.reshape(cfg.n_radial,cfg.n_angular,-1)[:,col,:]@g[step]
            times[col]=settings.times[step];steps[col]=step;pcolumns[col]={key:float(v[step]) for key,v in pressures.items()}
            snapshot=op.plane(g[step]);error=snapshot.rel_balance_error
            if abs(error)>1e-11:raise ValueError('Не замкнут баланс мгновенного сечения')
            balance=dict(z_mm=float(z),step=step,t_s=float(settings.times[step]),injected_kg_s=snapshot.injected_kg_s,inside_kg_s=snapshot.inside_kg_s,inner_kg_s=snapshot.inner_kg_s,outer_kg_s=snapshot.outer_kg_s,wall_bound_kg_s=snapshot.wall_bound_kg_s,wall_inner_kg_s=snapshot.wall_inner_kg_s,wall_outer_kg_s=snapshot.wall_outer_kg_s,relative_error=error)
            if phase_mass is not None:
                balance.update(suspended_kg_s=float(snapshot.suspended_cell_mass_kg_s.sum()),
                    film_inner_kg_s=float(snapshot.film_inner_cell_mass_kg_s.sum()),
                    film_outer_kg_s=float(snapshot.film_outer_cell_mass_kg_s.sum()),
                    momentum_relative_residual=snapshot.transport_momentum['relative_residual'],max_travel_time_s=snapshot.max_travel_time_s)
                for key in ('incoming_N','outgoing_N','carrier_N','wall_N','residual_N'):
                    for axis,value in zip(('z','x','y'),snapshot.transport_momentum[key]):
                        balance[key.removesuffix('_N')+'_'+axis+'_N']=float(value)
            balances.append(balance)
        transport={}
        if phase_mass is not None:
            transport=dict(suspended_cell_mass_kg_s=phase_mass[0],film_inner_cell_mass_kg_s=phase_mass[1],
                film_outer_cell_mass_kg_s=phase_mass[2],transport_notice=op.transport_notice,
                max_travel_time_s=float(np.max(op.travel_time_s[np.any(g>0,axis=0)],initial=0.)))
        planes.append(PlaneResult(float(z),mass,op.area,mass/op.area,op.radii,op.edges,op.angles,float('nan'),float(mass.sum()),float('nan'),float('nan'),True,times,steps,tuple(pcolumns),
            wall_bound_kg_s=0. if phase_mass is not None else float('nan'),
            geometry_notice=op.transport_notice or (WALL_BOUND_NOTICE if op.wall_inner is not None else ''),**transport))
    notices=set(flow.notices);notices.add('Накопленная матрица содержит разные времена; единый мгновенный глобальный баланс не определён.')
    if any(op.suspended_weights is not None for op in operators):
        maximum=max((p.max_travel_time_s for p in planes),default=0.)
        notices.update(op.transport_notice for op in operators if op.transport_notice)
        notices.add('Давление изменяет только расход источника; скорость истечения, параметры релаксации и траектории заданы независимо от давления. Задержка переноса и предыстория давления не учитываются.')
        notices.add(f'Максимальное время переноса {maximum:.6g} с; шаг сканирования {settings.dt_s:.6g} с; отношение {maximum/settings.dt_s:.6g}. Для квазистационарного приближения время переноса должно быть мало относительно масштаба изменения давления; быстрое изменение давления требует нестационарной модели.')
    else:notices.add('Фиксированная форма ядер, τ=0; температура и воздух не являются входными данными.')
    payload=json.dumps(project_to_json(cfg,list(nozzles)),ensure_ascii=False,sort_keys=True,allow_nan=False)
    return QuasiResult(cfg,nozzles,settings,g,pressures,tuple(operators),tuple(planes),tuple(balances),tuple(sorted(notices)),time.perf_counter()-started,payload)
