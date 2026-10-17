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
    preview_weights: object=None
    preview_axis: object=None
    preview_mask: object=None
    upper_field: object=None
    def plane(self,flows):
        mass=(self.weights@flows).reshape(self.area.shape)
        return PlaneResult(self.z_mm,mass,self.area,mass/self.area,self.radii,self.edges,self.angles,float(self.reached@flows),float(mass.sum()),float(self.inner@flows),float(self.outer@flows))
    def smooth(self,flows):
        if self.preview_weights is None:raise ValueError('Подробное поле не было подготовлено')
        dim=len(self.preview_axis);values=(self.preview_weights@flows).reshape(dim,dim)
        return SmoothField(self.preview_axis,self.preview_axis,values,np.ma.masked_where(self.preview_mask,values),self.z_mm)

def build_influence(nozzles,cfg,z,flows,preview=True,cancel=None):
    radii,edges=cfg.radial_centers_and_edges();weights=np.zeros((cfg.n_radial*cfg.n_angular,len(nozzles)));inner=np.zeros(len(nozzles));outer=inner.copy();reached=inner.copy()
    for j,n in enumerate(nozzles):
        check_cancel(cancel)
        if not n.enabled or z<effective_nozzle(n,cfg).z_tip_mm:continue
        reached[j]=1.;xp,yp,w=sample_kernel(n,z,cfg,unit_mass=True);ir=np.searchsorted(edges,np.hypot(xp,yp),side='right')-1;inside=(ir>=0)&(ir<cfg.n_radial)
        inner[j]=w[ir<0].sum();outer[j]=w[ir>=cfg.n_radial].sum()
        ith=angular_indices(xp[inside],yp[inside],cfg)
        weights[:,j]=np.bincount(ir[inside]*cfg.n_angular+ith,weights=w[inside],minlength=weights.shape[0])
    if np.max(np.abs(weights.sum(axis=0)+inner+outer-reached))>1e-12:raise ValueError('Не замкнута нормировка единичных пространственных влияний')
    area=np.repeat((np.pi/cfg.n_angular*np.diff((edges*.001)**2))[:,None],cfg.n_angular,axis=1)
    op=Influence(deepcopy(cfg),float(z),weights,inner,outer,reached,area,radii,edges,grid_angles(cfg))
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
        op.preview_axis=np.linspace(-cfg.r_out_mm,cfg.r_out_mm,dim);x,y=np.meshgrid(op.preview_axis,op.preview_axis);r=np.hypot(x,y);op.preview_mask=(r<cfg.r_in_mm)|(r>cfg.r_out_mm)
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
        return dict(format='fuel-quasistationary-result',version=1,project_sha256=hashlib.sha256(self.project_payload.encode()).hexdigest(),settings=self.settings.to_dict(),
            composite='One angular column per uniform timestamp; all radial cells simultaneously; density kg/(m2 s), not accumulated mass',
            composite_global_balance=None,maximum_instantaneous_balance_error=self.maximum_balance_error,spatial_model='fixed unit conservative kernels; no pressure-dependent shape; tau=0',
            preview='analytic per-source kernels; sparse cached linear superposition at selected timestamp',color_auto='fixed across time; upper envelope from per-nozzle maximum flows',notices=list(self.notices),elapsed_s=self.elapsed_s)

def calculate_quasi(nozzles,cfg,*,preview=True,cancel=None,progress=None):
    started=time.perf_counter();cfg=deepcopy(cfg);nozzles=tuple(sorted(deepcopy(nozzles),key=lambda n:n.id));cfg.validate();validate_nozzles(nozzles,cfg)
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
        for step,col in enumerate(cols):
            check_cancel(cancel);mass[:,col]=op.weights.reshape(cfg.n_radial,cfg.n_angular,-1)[:,col,:]@g[step]
            times[col]=settings.times[step];steps[col]=step;pcolumns[col]={key:float(v[step]) for key,v in pressures.items()}
            snapshot=op.plane(g[step]);error=snapshot.rel_balance_error
            if abs(error)>1e-11:raise ValueError('Не замкнут баланс мгновенного сечения')
            balances.append(dict(z_mm=float(z),step=step,t_s=float(settings.times[step]),injected_kg_s=snapshot.injected_kg_s,inside_kg_s=snapshot.inside_kg_s,inner_kg_s=snapshot.inner_kg_s,outer_kg_s=snapshot.outer_kg_s,relative_error=error))
        planes.append(PlaneResult(float(z),mass,op.area,mass/op.area,op.radii,op.edges,op.angles,float('nan'),float(mass.sum()),float('nan'),float('nan'),True,times,steps,tuple(pcolumns)))
    notices=set(flow.notices);notices.add('Накопленная матрица содержит разные времена; единый мгновенный глобальный баланс не определён.');notices.add('Фиксированная форма ядер, τ=0; температура и воздух не являются входными данными.')
    payload=json.dumps(project_to_json(cfg,list(nozzles)),ensure_ascii=False,sort_keys=True,allow_nan=False)
    return QuasiResult(cfg,nozzles,settings,g,pressures,tuple(operators),tuple(planes),tuple(balances),tuple(sorted(notices)),time.perf_counter()-started,payload)
