"""Versioned sequential six-channel measurements and pressure time series (no Qt)."""
from dataclasses import dataclass,field,asdict
from copy import deepcopy
import math
import numpy as np

PRESSURE_FACTORS={'kgf/cm2':1.,'кгс/см²':1.,'bar':1e5/98066.5,'Pa':1/98066.5,'kPa':1000/98066.5,'MPa':1e6/98066.5}
KINDS={'delta_p','p_fuel_abs','p_fuel_gauge','p_gas_abs','p_gas_gauge'}

def finite(value,label):
    try:value=float(value)
    except (ValueError,TypeError):raise ValueError(f'{label}: нужно число') from None
    if not math.isfinite(value):raise ValueError(f'{label}: нужно конечное число')
    return value

@dataclass(frozen=True)
class PressureSeries:
    manifold_id: str
    t_s: tuple
    pressure: tuple
    unit: str='kgf/cm2'
    pressure_kind: str='delta_p'
    valid: tuple=()
    reference_id: str=''
    metadata: dict=field(default_factory=dict)
    def __post_init__(self):
        object.__setattr__(self,'t_s',tuple(finite(t,'Время давления') for t in self.t_s))
        object.__setattr__(self,'pressure',tuple(None if p is None else finite(p,'Давление') for p in self.pressure))
        object.__setattr__(self,'valid',tuple(self.valid) if self.valid else tuple(p is not None for p in self.pressure))
        object.__setattr__(self,'metadata',deepcopy(self.metadata))
        if not isinstance(self.manifold_id,str) or not self.manifold_id.strip() or self.unit not in PRESSURE_FACTORS or self.pressure_kind not in KINDS:raise ValueError('Давление: укажите коллектор, единицы и физический вид давления')
        times=np.asarray(self.t_s)+finite(self.metadata.get('time_offset_s',0.),'Сдвиг часов давления');times.flags.writeable=False;object.__setattr__(self,'_times',times)
        if len(self.t_s)<2 or len(self.pressure)!=len(self.t_s) or len(self.valid)!=len(self.t_s):raise ValueError('Давление: нужны не менее двух согласованных временных отсчётов')
        if np.any(np.diff(self.t_s)<=0):raise ValueError('Время давления должно строго возрастать, без повторов; исходные записи не сортируются молча')
        if any(not isinstance(v,(bool,np.bool_)) for v in self.valid):raise ValueError('Качество давления: valid должен быть логическим')
        for p,v in zip(self.pressure,self.valid):
            if v and p is None:raise ValueError('Нет значения достоверного давления')
            if v and (self.pressure_kind=='delta_p' and p<=0 or self.pressure_kind.endswith('_abs') and p<0):raise ValueError('Перепад должен быть >0, абсолютное давление ≥0')
    def value(self,t,allow_extrapolation=False,max_gap_s=None):
        t=finite(t,'Время интерполяции');times=self._times;j=int(np.searchsorted(times,t))
        if j<len(times) and times[j]==t:
            if not self.valid[j]:raise ValueError(f'{self.manifold_id}: недостоверное давление при t={t:g} с')
            return self.pressure[j]*PRESSURE_FACTORS[self.unit],False
        outside=j==0 or j==len(times)
        if outside and not allow_extrapolation:raise ValueError(f'{self.manifold_id}: время {t:g} с вне записи давления; временная экстраполяция запрещена')
        right=min(max(j,1),len(times)-1);left=right-1
        if not self.valid[left] or not self.valid[right]:raise ValueError(f'{self.manifold_id}: пропуск/недостоверный интервал давления')
        if max_gap_s is not None and times[right]-times[left]>max_gap_s:raise ValueError(f'{self.manifold_id}: интервал между отсчётами больше допустимого max_gap_s')
        f=(t-times[left])/(times[right]-times[left]);p=self.pressure[left]+f*(self.pressure[right]-self.pressure[left])
        return float(p*PRESSURE_FACTORS[self.unit]),outside
    def to_dict(self):return asdict(self)

@dataclass(frozen=True)
class MeasurementStep:
    step_id: int
    t_start_s: float
    t_end_s: float
    temperatures_C: tuple
    theta_deg: object=None  # angle in this session's measurement frame; None = nominal position
    valid: tuple=()
    note: str=''
    metadata: dict=field(default_factory=dict)
    def __post_init__(self):
        if isinstance(self.step_id,bool) or int(self.step_id)!=self.step_id or self.step_id<0:raise ValueError('step_id: нужен неотрицательный целый индекс измерения, не ID форсунки')
        object.__setattr__(self,'step_id',int(self.step_id))
        for key in ('t_start_s','t_end_s'):object.__setattr__(self,key,finite(getattr(self,key),key))
        if self.t_end_s<self.t_start_s or self.t_end_s==self.t_start_s and self.metadata.get('sampling_method')!='instantaneous':raise ValueError('Окно: t_end > t_start; равенство допустимо только для явно мгновенного замера')
        if self.theta_deg is not None:object.__setattr__(self,'theta_deg',finite(self.theta_deg,'Угол измерения'))
        temperatures=tuple(None if v is None else finite(v,'Температура') for v in self.temperatures_C)
        valid=tuple(self.valid) if self.valid else tuple(v is not None for v in temperatures)
        if len(temperatures)!=6 or len(valid)!=6:raise ValueError('В одной остановке нужны ровно шесть одновременных каналов')
        if any(not isinstance(v,(bool,np.bool_)) for v in valid):raise ValueError('Температура: valid должен быть логическим')
        if any(ok and value is None for ok,value in zip(valid,temperatures)):raise ValueError('Достоверный канал не может содержать пустую температуру')
        object.__setattr__(self,'temperatures_C',temperatures);object.__setattr__(self,'valid',valid);object.__setattr__(self,'metadata',deepcopy(self.metadata))
    def to_dict(self):return asdict(self)

@dataclass(frozen=True)
class MeasurementSession:
    radii_mm: tuple
    z_mm: float
    steps: tuple
    pressure_series: tuple
    gas_series: tuple=()
    theta_zero_deg: float=0.
    theta_direction: int=1
    theta_start_deg: float=0.
    nominal_steps: int=140
    default_manifold: str='A'
    nozzle_manifolds: dict=field(default_factory=dict)
    transport_delay_s: float=0.
    nozzle_delays_s: dict=field(default_factory=dict)
    point_delays_s: dict=field(default_factory=dict)  # "step_id:channel(1..6):nozzle_id"
    allow_time_extrapolation: bool=False
    max_gap_s: object=None
    integration_rtol: float=1e-6
    metadata: dict=field(default_factory=dict)
    def __post_init__(self):
        object.__setattr__(self,'radii_mm',tuple(finite(v,'Радиус датчика') for v in self.radii_mm))
        object.__setattr__(self,'steps',tuple(self.steps));object.__setattr__(self,'pressure_series',tuple(self.pressure_series));object.__setattr__(self,'gas_series',tuple(self.gas_series))
        for name in ('z_mm','theta_zero_deg','theta_start_deg','transport_delay_s','integration_rtol'):object.__setattr__(self,name,finite(getattr(self,name),name))
        for name in ('nozzle_manifolds','nozzle_delays_s','point_delays_s','metadata'):object.__setattr__(self,name,deepcopy(getattr(self,name)))
        object.__setattr__(self,'nozzle_manifolds',{str(k):str(v) for k,v in self.nozzle_manifolds.items()})
        object.__setattr__(self,'nozzle_delays_s',{str(k):finite(v,'Задержка форсунки') for k,v in self.nozzle_delays_s.items()})
        object.__setattr__(self,'point_delays_s',{str(k):finite(v,'Задержка площадки') for k,v in self.point_delays_s.items()})
        if len(self.radii_mm)!=6 or len(set(self.radii_mm))!=6:raise ValueError('Нужны шесть различных радиусов датчиков, порядок каналов сохраняется')
        if self.z_mm<0 or self.theta_direction not in (-1,1) or isinstance(self.theta_direction,bool):raise ValueError('Сессия: неверные z или направление обхода')
        if isinstance(self.nominal_steps,bool) or int(self.nominal_steps)!=self.nominal_steps or not 1<=self.nominal_steps<=10000:raise ValueError('Номинальное число окружных положений: 1…10000')
        object.__setattr__(self,'nominal_steps',int(self.nominal_steps));object.__setattr__(self,'theta_direction',int(self.theta_direction))
        if not isinstance(self.allow_time_extrapolation,bool):raise ValueError('Флаг временной экстраполяции должен быть логическим')
        if self.max_gap_s is not None:
            object.__setattr__(self,'max_gap_s',finite(self.max_gap_s,'max_gap_s'))
            if self.max_gap_s<=0:raise ValueError('max_gap_s должен быть >0 или null')
        if not 1e-10<=self.integration_rtol<=1e-2:raise ValueError('Точность интегрирования rtol: 1e-10…1e-2')
        if self.transport_delay_s<0 or any(v<0 for v in (*self.nozzle_delays_s.values(),*self.point_delays_s.values())):raise ValueError('Задержки не могут быть отрицательными')
        ids=[s.step_id for s in self.steps]
        if len(set(ids))!=len(ids):raise ValueError('Повторные step_id; одинаковые углы в разные времена допустимы с разными индексами')
        windows=sorted((s.t_start_s,s.t_end_s) for s in self.steps)
        if any(b[0]<a[1] for a,b in zip(windows,windows[1:])):raise ValueError('Окна разных последовательных остановок перекрываются')
        for key in set(self.nozzle_manifolds)|set(self.nozzle_delays_s):
            if not key.isdigit() or not 1<=int(key)<=140:raise ValueError('Привязка к коллекторам/задержкам: ID форсунки 1…140')
        for key in self.point_delays_s:
            try:step,radial,nozzle=(int(v) for v in key.split(':'))
            except ValueError:raise ValueError('Ключ точечной задержки: step_id:канал(1…6):nozzle_id') from None
            if step not in ids or not 1<=radial<=6 or not 1<=nozzle<=140:raise ValueError('Точечная задержка ссылается на отсутствующее измерение/канал/форсунку')
        for collection in (self.pressure_series,self.gas_series):
            groups=[p.manifold_id for p in collection]
            if len(groups)!=len(set(groups)):raise ValueError('Повторный ряд давления одного коллектора')
        if any(p.pressure_kind.startswith('p_gas') for p in self.pressure_series) or any(not p.pressure_kind.startswith('p_gas') for p in self.gas_series):raise ValueError('Ряды топлива и газа должны иметь соответствующий физический вид давления')
    def angle(self,step):return step.theta_deg if step.theta_deg is not None else (self.theta_start_deg+step.step_id*360/self.nominal_steps)%360
    def global_angle(self,step):return (self.theta_zero_deg+self.theta_direction*self.angle(step))%360
    def manifold(self,nozzle):return self.nozzle_manifolds.get(str(nozzle.id),self.default_manifold)
    def delay(self,step,channel,nozzle):return self.transport_delay_s+self.nozzle_delays_s.get(str(nozzle.id),0.)+self.point_delays_s.get(f'{step.step_id}:{channel+1}:{nozzle.id}',0.)
    def fuel(self,manifold):
        channel=next((p for p in self.pressure_series if p.manifold_id==manifold),None)
        if channel is None:raise ValueError(f'Нет временного ряда топлива коллектора {manifold}')
        return channel
    def gas(self,manifold):return next((p for p in self.gas_series if p.manifold_id==manifold),next((p for p in self.gas_series if p.manifold_id=='*'),None))
    def delta_p(self,manifold,t):
        fuel=self.fuel(manifold);fp,extrapolated=fuel.value(t,self.allow_time_extrapolation,self.max_gap_s)
        if fuel.pressure_kind.endswith('_abs') and fp<0:raise ValueError('Абсолютное давление топлива стало отрицательным при экстраполяции')
        if fuel.pressure_kind=='delta_p':delta=fp
        else:
            gas=self.gas(manifold)
            if gas is None:raise ValueError(f'{manifold}: для давления подачи нужен отдельный ряд давления газа')
            expected='p_gas_abs' if fuel.pressure_kind=='p_fuel_abs' else 'p_gas_gauge'
            if gas.pressure_kind!=expected:raise ValueError(f'{manifold}: несовместимые абсолютный/избыточный базисы топлива и газа')
            if expected=='p_gas_gauge' and (not fuel.reference_id or fuel.reference_id!=gas.reference_id):raise ValueError(f'{manifold}: для избыточных давлений нужен одинаковый явный reference_id')
            gp,extra=gas.value(t,self.allow_time_extrapolation,self.max_gap_s)
            if gas.pressure_kind.endswith("_abs") and gp<0:raise ValueError("Абсолютное давление газа стало отрицательным при экстраполяции")
            extrapolated|=extra;delta=fp-gp
        if delta<=0 or not np.isfinite(delta):raise ValueError(f'{manifold}: физический Δp при t={t:g} должен быть >0')
        return delta,extrapolated
    def time_nodes(self,manifold,start,end):
        nodes=[start,end];channels=[self.fuel(manifold)]
        if channels[0].pressure_kind!='delta_p':
            gas=self.gas(manifold)
            if gas is None:raise ValueError(f'{manifold}: нет давления газа')
            channels.append(gas)
        for channel in channels:nodes.extend(float(t) for t in channel._times if start<t<end)
        return np.unique(nodes)
    def validate(self,cfg,nozzles,require_temperatures=True):
        if not self.steps:raise ValueError('Сессия не содержит остановок')
        if any(not cfg.r_in_mm<r<cfg.r_out_mm for r in self.radii_mm):raise ValueError('Радиусы датчиков должны находиться строго внутри тракта; совпадение с центрами сетки не требуется')
        if require_temperatures and not any(v for step in self.steps for v in step.valid):raise ValueError('В сессии нет достоверных температур; пустые T не заменяются нулями')
        windows=set()
        for n in nozzles:
            if n.enabled:
                group=self.manifold(n);self.fuel(group)
                for step in self.steps:
                    for delay in {0.,*(self.delay(step,r,n) for r in range(6))}:windows.add((group,step.t_start_s-delay,step.t_end_s-delay))
        for group,start,end in windows:
            for t in self.time_nodes(group,start,end):self.delta_p(group,t)
        return self
    def to_dict(self):return {'format':'fuel-measurement-session','version':'0.5',**asdict(self)}
    @classmethod
    def from_dict(cls,data):
        if data.get('format')!='fuel-measurement-session':raise ValueError('Ожидается MeasurementSession формата fuel-measurement-session')
        if data.get('version')!='0.5':raise ValueError('Неизвестная версия MeasurementSession')
        for key in ('radii_mm','z_mm','steps','pressure_series'):
            if key not in data:raise ValueError('В сессии отсутствует '+key)
        for p in [*data.get('pressure_series',[]),*data.get('gas_series',[])]:
            if any(key not in p for key in ('unit','pressure_kind','manifold_id','t_s','pressure')):raise ValueError('Каждый ряд давления должен явно задавать единицы, физический вид и коллектор')
        raw=deepcopy(data);raw.pop('format');raw.pop('version',None)
        raw['steps']=tuple(MeasurementStep(**s) for s in raw.get('steps',[]))
        for key in ('pressure_series','gas_series'):raw[key]=tuple(PressureSeries(**p) for p in raw.get(key,[]))
        return cls(**raw)
