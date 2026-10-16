"""Quasisteady fuel superposition for time-stamped scanning. No temperature model, no Qt."""
from dataclasses import dataclass,field,replace
from copy import deepcopy
import hashlib,json,time,warnings
import numpy as np
from scipy.integrate import quad,IntegrationWarning
from .core import sample_kernel,PlaneResult,project_to_json
from .coordinates import grid_angles,angular_indices
from .v02 import flow_status,curve_for,bench_pressure,effective_nozzle,FlowStatus,validate_nozzles,GEOMETRY_KEYS
from .characteristics import ExtrapolationWarning
from .measurement import MeasurementSession

class CalculationCancelled(Exception):pass

def check_cancel(cancel):
    if cancel and cancel():raise CalculationCancelled('Расчёт сессии отменён; частичный результат не применяется')

class FlowEvaluator:
    """Curve and window caches are scoped to an immutable calculation snapshot."""
    def __init__(self,cfg,nozzles):
        self.cfg=deepcopy(cfg);self.nozzles=deepcopy(nozzles);self.curves={};self.values={};self.bases={};self.notices=set()
    def curve_value(self,n,pressure):
        key=(n.atomizer_type_id,float(pressure))
        if key not in self.values:
            if n.atomizer_type_id not in self.curves:self.curves[n.atomizer_type_id]=curve_for(n,self.cfg)
            with warnings.catch_warnings():
                warnings.simplefilter('ignore',ExtrapolationWarning)
                estimate=self.curves[n.atomizer_type_id].evaluate(pressure,allow_extrapolation=self.cfg.allow_extrapolation)
            if estimate.extrapolated:self.notices.add('Разрешённая экстраполяция расходной характеристики')
            self.values[key]=estimate.flow_kg_h/3600
        return self.values[key]
    def base(self,n):
        if n.id not in self.bases:
            reference=bench_pressure(n,self.cfg,'flow' if n.flow_mode=='manual' else 'sector')
            status=flow_status(n,replace(self.cfg,target_pressure_kgf_cm2=reference))
            self.notices.update(status.notices);self.bases[n.id]=(status.kg_s,reference,status.source)
        return self.bases[n.id]
    def at(self,n,pressure):
        if not np.isfinite(pressure) or pressure<=0:raise ValueError('Физический перепад должен быть положительным конечным числом')
        if not n.enabled:return 0.
        if n.flow_mode=='curve':return self.curve_value(n,pressure)
        base,reference,_=self.base(n)
        if base==0 or pressure==reference:return base
        ref=self.curve_value(n,reference)
        if ref<=0:raise ValueError(f'Форсунка {n.id}: Q при опорном давлении должна быть >0')
        return base*self.curve_value(n,pressure)/ref

def flow_at_pressure(nozzle,cfg,delta_p):
    """Physical Δp, absolute curve once, or original measured reference × Q(P)/Q(Pref)."""
    evaluator=FlowEvaluator(cfg,[nozzle]);value=evaluator.at(nozzle,float(delta_p))
    if not nozzle.enabled:return FlowStatus(0.,float(delta_p),'выключена',False)
    if nozzle.flow_mode=='curve':source='расходная характеристика';estimated=True
    else:
        base,reference,source=evaluator.base(nozzle);estimated=nozzle.flow_mode!='manual' or float(delta_p)!=reference
        if float(delta_p)!=reference:source+=' → Q(Δp)/Q(Δp_ref)'
    return FlowStatus(value,float(delta_p),source,estimated,tuple(sorted(evaluator.notices)))

def geometry_signature(nozzles,cfg,z):
    keys=('r_in_mm','r_out_mm','n_radial','n_angular','radial_centers_mm','grid_start_deg','theta_zero_deg','theta_direction','geometry_model','angular_model','radial_model','n_sample_radial','n_sample_angular')
    geometry=[]
    for n in sorted(nozzles,key=lambda n:n.id):
        e=effective_nozzle(n,cfg);geometry.append({**{k:getattr(e,k) for k in GEOMETRY_KEYS},**{k:getattr(e,k) for k in ('id','enabled','row','theta_deg','radius_mm','sector_ml')}})
    payload=json.dumps(dict(z=z,settings={k:getattr(cfg,k) for k in keys},nozzles=geometry),sort_keys=True,separators=(',',':'),allow_nan=False)
    return hashlib.sha256(payload.encode()).hexdigest()

@dataclass
class InfluenceMatrix:
    z_mm: float
    cfg: object
    nozzle_ids: tuple
    mass_weights: np.ndarray  # cells × sources, dimensionless, independent of G_ref
    area_m2: np.ndarray
    inner: np.ndarray
    outer: np.ndarray
    upstream: np.ndarray
    radii: np.ndarray
    radial_edges: np.ndarray
    angles: np.ndarray
    normalization_error: float
    signature: str
    @property
    def flux_weights(self):return self.mass_weights/self.area_m2.reshape(-1,1)
    def plane(self,flows):
        flows=np.asarray(flows,float)
        if flows.shape!=(len(self.nozzle_ids),) or not np.isfinite(flows).all() or np.any(flows<0):raise ValueError('Некорректный вектор расходов источников')
        mass=(self.mass_weights@flows).reshape(self.area_m2.shape);injected=float(self.upstream@flows)
        return PlaneResult(self.z_mm,mass,self.area_m2.copy(),mass/self.area_m2,self.radii.copy(),self.radial_edges.copy(),self.angles.copy(),injected,float(mass.sum()),float(self.inner@flows),float(self.outer@flows))
    def cell_indices(self,radii,angles):
        radial=np.searchsorted(self.radial_edges,radii,side='right')-1
        angle=np.deg2rad(angles);angular=angular_indices(np.cos(angle),np.sin(angle),self.cfg)
        if np.any(radial<0) or np.any(radial>=self.cfg.n_radial):raise ValueError('Площадка датчика вне расчётного тракта')
        return radial[:,None]*self.cfg.n_angular+angular[None,:]

def build_influence(nozzles,cfg,z_mm,cancel=None,progress=None):
    cfg.validate();radii,edges=cfg.radial_centers_and_edges();ordered=sorted(nozzles,key=lambda n:n.id)
    weights=np.zeros((cfg.n_radial*cfg.n_angular,len(ordered)));inner=np.zeros(len(ordered));outer=inner.copy();upstream=inner.copy()
    for j,original in enumerate(ordered):
        check_cancel(cancel);n=effective_nozzle(original,cfg)
        if n.enabled and z_mm>=n.z_tip_mm:
            xp,yp,w=sample_kernel(n,z_mm,cfg,unit_mass=True);upstream[j]=1.
            ir=np.searchsorted(edges,np.hypot(xp,yp),side='right')-1;inside=(ir>=0)&(ir<cfg.n_radial)
            inner[j]=w[ir<0].sum();outer[j]=w[ir>=cfg.n_radial].sum()
            if inside.any():
                angular=angular_indices(xp[inside],yp[inside],cfg);weights[:,j]=np.bincount(ir[inside]*cfg.n_angular+angular,weights=w[inside],minlength=weights.shape[0])
        if progress:progress(j+1,len(ordered),'Пространственные влияния')
    error=float(np.max(np.abs(weights.sum(axis=0)+inner+outer-upstream)))
    if error>1e-12:raise ValueError('Не замкнута нормировка единичных пространственных весов')
    area=np.repeat((np.pi/cfg.n_angular*np.diff((edges*.001)**2))[:,None],cfg.n_angular,axis=1)
    return InfluenceMatrix(float(z_mm),deepcopy(cfg),tuple(n.id for n in ordered),weights,area,inner,outer,upstream,radii,edges,grid_angles(cfg),error,geometry_signature(nozzles,cfg,float(z_mm)))

class WindowIntegrator:
    def __init__(self,session,evaluator,cancel=None):
        self.session=session;self.evaluator=evaluator;self.cancel=cancel;self.pressures={};self.curve_integrals={};self.means={};self.query_times=set();self.notices=set();self.max_error=0.
    def pressure_window(self,group,start,end):
        key=(group,start,end)
        if key not in self.pressures:
            nodes=self.session.time_nodes(group,start,end);pairs=[self.session.delta_p(group,float(t)) for t in nodes];values=np.array([v for v,extra in pairs])
            if any(extra for v,extra in pairs):self.notices.add('Разрешена временная экстраполяция давления')
            mean=float(values[0]) if end==start else float(np.sum((values[:-1]+values[1:])*.5*np.diff(nodes))/(end-start))
            self.pressures[key]=(nodes,values,(float(values.min()),float(values.max()),mean))
        return self.pressures[key]
    def mean(self,n,group,start,end):
        key=(n.id,group,start,end)
        if key in self.means:return self.means[key]
        check_cancel(self.cancel);nodes,pressures,stats=self.pressure_window(group,start,end)
        if not n.enabled:value,error=0.,0.
        elif np.ptp(pressures)==0:value,error=self.evaluator.at(n,float(pressures[0])),0.
        elif n.flow_mode!='curve' and self.evaluator.base(n)[0]==0:value,error=0.,0.
        else:
            # No curve: a varying pressure cannot be converted by an invented root law.
            if n.atomizer_type_id not in self.evaluator.curves:self.evaluator.curves[n.atomizer_type_id]=curve_for(n,self.evaluator.cfg)
            curve=self.evaluator.curves[n.atomizer_type_id];curve_key=(n.atomizer_type_id,group,start,end)
            if curve_key not in self.curve_integrals:
                total=error=0.
                for a,b,pa,pb in zip(nodes[:-1],nodes[1:],pressures[:-1],pressures[1:]):
                    check_cancel(self.cancel)
                    boundaries=[float(a),float(b)]
                    # Experimental Q is piecewise linear. Crossing a measured pressure knot is a time knot.
                    if hasattr(curve,'points') and pb!=pa:
                        boundaries.extend(float(a+(p-pa)*(b-a)/(pb-pa)) for p,q in curve.points if min(pa,pb)<p<max(pa,pb))
                    boundaries=sorted(boundaries)
                    def instantaneous(t):
                        check_cancel(self.cancel);self.query_times.add(float(t));p,extra=self.session.delta_p(group,float(t))
                        if extra:self.notices.add('Разрешена временная экстраполяция давления')
                        return self.evaluator.curve_value(n,p)
                    for left,right in zip(boundaries,boundaries[1:]):
                        if hasattr(curve,'points'):
                            integral=(instantaneous(left)+instantaneous(right))*.5*(right-left);estimate=0.
                        else:
                            with warnings.catch_warnings():
                                warnings.simplefilter('error',IntegrationWarning)
                                integral,estimate=quad(instantaneous,left,right,epsabs=1e-12*(right-left),epsrel=self.session.integration_rtol,limit=100)
                        total+=integral;error+=estimate
                self.curve_integrals[curve_key]=(total/(end-start),error/(end-start))
            value,error=self.curve_integrals[curve_key]
            if n.flow_mode!='curve':
                base,reference,_=self.evaluator.base(n);reference_flow=self.evaluator.curve_value(n,reference)
                if reference_flow<=0:raise ValueError(f'Форсунка {n.id}: Q при опорном давлении должна быть >0')
                factor=base/reference_flow;value*=factor;error*=abs(factor)
        self.max_error=max(self.max_error,float(error));self.means[key]=float(value);return float(value)

@dataclass
class TransientResult:
    session: MeasurementSession
    project_payload: str
    project_sha256: str
    operator: InfluenceMatrix
    nozzles: tuple
    global_angles: np.ndarray
    cells: np.ndarray
    q_corrected: np.ndarray
    temperatures_C: np.ndarray
    temperature_valid: np.ndarray
    mean_flows: np.ndarray  # [step, 140], unshifted common window
    point_flows: np.ndarray  # [channel, step, 140], explicit shifted windows
    pressure_windows: tuple
    balances: tuple
    notices: tuple
    elapsed_s: float
    temporal_error_kg_s: float
    _evaluator: object=field(repr=False)
    def snapshot(self,t):
        flows=np.array([self._evaluator.at(n,self.session.delta_p(self.session.manifold(n),float(t))[0]) if n.enabled else 0. for n in self.nozzles])
        plane=self.operator.plane(flows)
        if abs(plane.rel_balance_error)>1e-11:raise ValueError('Не замкнут баланс мгновенного сечения')
        return plane,flows
    def manifest(self):
        return dict(version='0.5',project_sha256=self.project_sha256,session=self.session.to_dict(),spatial_model='fixed source kernels; conservative containing-cell average',
            temperature_model=None,footprint='radial control band × angular control sector containing actual sensor coordinate',
            time_windows='[start,end); continuous signals share endpoints of measure zero',interpolation='linear pressure, no bridging invalid samples',
            integration='exact trapezoids split at experimental Q knots; adaptive QUADPACK for Dityakin',integration_rtol=self.session.integration_rtol,
            maximum_quadrature_error_kg_s=self.temporal_error_kg_s,spatial_normalization_error=self.operator.normalization_error,
            maximum_snapshot_balance_error=max((abs(b['relative_error']) for b in self.balances),default=0.),
            composite_global_balance=None,instantaneous_snapshot_delay_s=0.,elapsed_s=self.elapsed_s,notices=list(self.notices))

def calculate_session(nozzles,cfg,session,cancel=None,progress=None,operator=None):
    started=time.perf_counter();session=MeasurementSession.from_dict(session.to_dict());cfg=deepcopy(cfg);nozzles=tuple(sorted(deepcopy(nozzles),key=lambda n:n.id))
    cfg.validate();validate_nozzles(nozzles,cfg)
    if len(nozzles)!=140 or {n.id for n in nozzles}!=set(range(1,141)):raise ValueError("Нужны 140 фиксированных позиций форсунок")
    session.validate(cfg,nozzles);check_cancel(cancel)
    payload=json.dumps(project_to_json(cfg,list(nozzles)),ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False)
    op=build_influence(nozzles,cfg,session.z_mm,cancel,progress) if operator is None else operator
    if op.signature!=geometry_signature(nozzles,cfg,session.z_mm):raise ValueError('Кэш пространственных влияний не соответствует расчёту')
    angles=np.array([session.global_angle(s) for s in session.steps]);cells=op.cell_indices(np.asarray(session.radii_mm),angles);w=op.flux_weights[cells]
    evaluator=FlowEvaluator(cfg,nozzles);integrator=WindowIntegrator(session,evaluator,cancel);count=len(session.steps)
    means=np.zeros((count,len(nozzles)));point_flows=np.zeros((6,count,len(nozzles)));pressure_records=[]
    for i,step in enumerate(session.steps):
        check_cancel(cancel)
        for j,n in enumerate(nozzles):
            if not n.enabled:continue
            group=session.manifold(n);means[i,j]=integrator.mean(n,group,step.t_start_s,step.t_end_s)
            for channel in range(6):
                delay=session.delay(step,channel,n);a,b=step.t_start_s-delay,step.t_end_s-delay
                point_flows[channel,i,j]=integrator.mean(n,group,a,b)
        if progress:progress(i+1,count,'Временные окна остановок')
    q=np.sum(w*point_flows,axis=2)
    for (group,a,b),(nodes,values,stats) in sorted(integrator.pressures.items()):
        pressure_records.append(dict(manifold_id=group,t_start_s=float(a),t_end_s=float(b),min_delta_p=stats[0],max_delta_p=stats[1],mean_delta_p=stats[2]))
    temperatures=np.array([[np.nan if s.temperatures_C[r] is None else s.temperatures_C[r] for s in session.steps] for r in range(6)])
    valid=np.array([[s.valid[r] for s in session.steps] for r in range(6)],dtype=bool)
    notices={'Топливо — средний массовый поток по контрольной площадке, содержащей координату датчика; температура — независимое исходное измерение.',
             'Составная матрица содержит разные временные окна: единый мгновенный глобальный баланс для неё не определён.',
             'Без заданной задержки принято τ=0: поле мгновенно следует за давлением; время пролёта не идентифицировано.'}
    if session.transport_delay_s or session.nozzle_delays_s or session.point_delays_s:notices.add('Задержки заданы пользователем; снимки полного сечения и их баланс относятся к τ=0. Инерция термопар не корректируется.')
    result=TransientResult(session,payload,hashlib.sha256(payload.encode()).hexdigest(),op,nozzles,angles,cells,q,temperatures,valid,means,point_flows,tuple(pressure_records),(),(),0.,integrator.max_error,evaluator)
    # Every displayed/audited instantaneous field has its own balance, independent of the composite matrix.
    times=set()
    for step in session.steps:
        times.update((step.t_start_s,(step.t_start_s+step.t_end_s)/2,step.t_end_s))
        for group in {session.manifold(n) for n in nozzles if n.enabled}:times.update(session.time_nodes(group,step.t_start_s,step.t_end_s))
    balances=[]
    for t in sorted(times):
        check_cancel(cancel);plane,flows=result.snapshot(t)
        balances.append(dict(t_s=float(t),injected_kg_s=plane.injected_kg_s,inside_kg_s=plane.inside_kg_s,inner_kg_s=plane.inner_kg_s,outer_kg_s=plane.outer_kg_s,relative_error=plane.rel_balance_error))
    result.balances=tuple(balances);result.notices=tuple(sorted(notices|evaluator.notices|integrator.notices));result.elapsed_s=time.perf_counter()-started
    return result
