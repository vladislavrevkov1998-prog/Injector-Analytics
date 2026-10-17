"""Bounded fuel-to-fuel identification, independent of Qt and working projects."""
from dataclasses import dataclass,field,replace
from copy import deepcopy
import hashlib,json,time
import numpy as np
from scipy.optimize import least_squares
from .core import project_from_json,project_to_json,calculate_plane,geometry_at,source_flow_kg_s
from .v02 import GEOMETRY_KEYS,NOZZLE_KINDS,effective_nozzle,validate_nozzles
from .analysis import cell_values,matching_index

QUANTITIES={'flux':'кг/(м²·с)','cell_mass':'кг/с','normalized_ring':'1','normalized_track':'1'}
METRIC={'flux':'flux','normalized_ring':'ring','normalized_track':'track'}
PARAM_UNITS={k:('мм' if k.endswith('_mm') else '°' if 'angle' in k or k in ('alpha_deg','clock_deg') else '°/мм' if k=='rotation_deg_per_mm' else '°/√мм' if k=='diffusion_deg_sqrt_mm' else '1') for k in GEOMETRY_KEYS}
PARAM_UNITS.update(k_quadratic='1/мм',diffusion_deg_sqrt_mm='°/√мм',rotation_deg_per_mm='°/мм')

@dataclass(frozen=True)
class RegressionSnapshot:
    payload: str
    sha256: str
    @classmethod
    def capture(cls,cfg,nozzles):
        payload=json.dumps(project_to_json(cfg,nozzles),ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False)
        return cls(payload,hashlib.sha256(payload.encode()).hexdigest())
    def restore(self):
        if hashlib.sha256(self.payload.encode()).hexdigest()!=self.sha256:raise ValueError('Снимок повреждён')
        return project_from_json(json.loads(self.payload))

@dataclass(frozen=True)
class RegressionTarget:
    quantity: str
    z_mm: tuple
    radius_mm: tuple
    theta_deg: tuple
    y: tuple
    unit: str
    mask: tuple=()
    weights: tuple=()
    split: tuple=()
    area_m2: tuple=()
    metadata: dict=field(default_factory=dict)
    def __post_init__(self):
        for key in ("z_mm","radius_mm","theta_deg","y","mask","weights","split","area_m2"):
            value=getattr(self,key);object.__setattr__(self,key,tuple(value) if value is not None else ())
        object.__setattr__(self,"metadata",deepcopy(self.metadata))
    def arrays(self):
        n=len(self.y)
        return (np.asarray(self.mask or (True,)*n,dtype=bool),
                np.asarray(self.weights or (1.,)*n,dtype=float),
                np.asarray(self.split or ('train',)*n))
    def validate(self,cfg):
        if self.quantity not in QUANTITIES or self.unit!=QUANTITIES.get(self.quantity):raise ValueError('Нужен топливный эталон с явной величиной и единицами; температура не поддерживается.')
        if self.metadata.get('source_kind') not in ('fuel','synthetic','reference_model'):raise ValueError('Укажите source_kind: fuel, synthetic или reference_model.')
        if self.metadata.get('theta_direction','ccw')!='ccw' or self.metadata.get('theta_zero_deg',0)!=0:raise ValueError('Эталон должен задавать глобальные углы против часовой стрелки от +X')
        n=len(self.y)
        if not n or any(len(values)!=n for values in (self.z_mm,self.radius_mm,self.theta_deg)):raise ValueError('Несогласованные размеры топливного эталона')
        if any(values and len(values)!=n for values in (self.mask,self.weights,self.split,self.area_m2)):raise ValueError('Несогласованные маски/веса/роль/площади')
        mask,weights,split=self.arrays()
        if not np.all(np.isfinite(weights)) or np.any(weights<0):raise ValueError('Веса должны быть конечными и неотрицательными')
        if not set(split).issubset({'train','holdout','test'}):raise ValueError('Роль наблюдения: train / holdout / test')
        if not np.all(np.isfinite(np.asarray(self.y)[mask])) or np.any(np.asarray(self.y)[mask]<0):raise ValueError('Используемые топливные значения должны быть конечными и неотрицательными')
        radii,edges=cfg.radial_centers_and_edges();angles=(cfg.grid_start_deg+np.arange(cfg.n_angular)*360/cfg.n_angular)%360
        source_grid=self.metadata.get('model_settings')
        if source_grid:
            for key in ('r_in_mm','r_out_mm','n_radial','n_angular'):
                if source_grid.get(key)!=getattr(cfg,key):raise ValueError('Контрольные области эталона и проекта различаются: '+key)
            source_r=source_grid.get('radial_centers_mm')
            if source_r is None:source_r=source_grid['r_in_mm']+(np.arange(source_grid['n_radial'])+.5)*(source_grid['r_out_mm']-source_grid['r_in_mm'])/source_grid['n_radial']
            if not np.allclose(source_r,radii,rtol=0,atol=1e-7):raise ValueError('Радиальные контрольные области различаются')
        seen=set();mapping=[]
        for i,(z,r,theta) in enumerate(zip(self.z_mm,self.radius_mm,self.theta_deg)):
            if not all(np.isfinite(v) for v in (z,r,theta)) or z<0:raise ValueError('Неверные физические координаты эталона')
            ri=matching_index(radii,r);ai=matching_index(angles,theta%360)
            if ri is None or ai is None:raise ValueError(f'Наблюдение {i+1}: z/R/θ должны соответствовать физическим центрам контрольной сетки; интерполяция не подставляется.')
            key=(round(z,7),ri,ai)
            if key in seen:raise ValueError('Повторное наблюдение одной координаты; train/holdout должны быть независимы')
            seen.add(key);mapping.append((float(z),ri,ai))
            if self.quantity=='cell_mass' or self.area_m2:
                area=np.pi/cfg.n_angular*((edges[ri+1]*1e-3)**2-(edges[ri]*1e-3)**2)
                if not self.area_m2 or not np.isclose(self.area_m2[i],area,rtol=1e-9,atol=1e-14):raise ValueError('Расход ячейки требует совпадающей площади интегрирования')
        if not np.any(mask&(weights>0)&(split=='train')):raise ValueError('Нет действительных train-наблюдений')
        return mapping
    @classmethod
    def from_project(cls,cfg,nozzles,quantity='flux',source_kind='reference_model',cancel=None):
        zvals=[];rs=[];angles=[];ys=[];areas=[];mask=[];roles=[]
        for z in cfg.z_mm:
            if cancel and cancel():raise ValueError('Загрузка эталона отменена')
            p=calculate_plane(nozzles,cfg,z)
            values=p.cell_mass_kg_s if quantity=='cell_mass' else cell_values(p,METRIC[quantity],cfg.density_kg_m3)
            for k,r in enumerate(p.radial_centers_mm):
                for j,angle in enumerate(p.theta_centers_deg):
                    v=values[k,j];valid=not np.ma.is_masked(v)
                    zvals.append(z);rs.append(float(r));angles.append(float(angle));ys.append(float(v) if valid else np.nan)
                    mask.append(valid);areas.append(float(p.cell_area_m2[k,j]));roles.append('holdout' if j%5==0 else 'train')
        return cls(quantity,tuple(zvals),tuple(rs),tuple(angles),tuple(ys),QUANTITIES[quantity],tuple(mask),split=tuple(roles),area_m2=tuple(areas),metadata={'source_kind':source_kind,'model_settings':project_to_json(cfg,[])['settings']})
    def subset(self,z=None,radius=None,angle_start=0,angle_end=360):
        angles=np.asarray(self.theta_deg)%360;mask=np.asarray(self.mask or (True,)*len(self.y),dtype=bool)
        if z is not None:mask&=np.isclose(self.z_mm,z,rtol=0,atol=1e-7)
        if radius is not None:mask&=np.isclose(self.radius_mm,radius,rtol=0,atol=1e-7)
        if angle_end-angle_start<360:
            mask&=((angles-angle_start)%360)<=((angle_end-angle_start)%360)
        return RegressionTarget(**{**vars(self),'mask':tuple(mask)})

@dataclass(frozen=True)
class ParameterSpec:
    key: str
    scope: str
    initial: float
    minimum: float
    maximum: float
    fixed: bool=False
    unit: str=''

@dataclass(frozen=True)
class FitResult:
    snapshot: RegressionSnapshot
    parameters: tuple
    fitted: tuple
    candidate_payload: str
    initial_prediction: tuple
    prediction: tuple
    target: RegressionTarget
    metrics: dict
    initial_cost: float
    cost: float
    evaluations: int
    status: str
    warnings: tuple
    diagnostics: dict
    settings: dict
    elapsed_s: float
    def candidate(self):return project_from_json(json.loads(self.candidate_payload))

def active_parameters(cfg,nozzles,target,scope):
    if scope not in NOZZLE_KINDS:raise ValueError('v0.4 подбирает только параметры типа форсунки')
    members=[effective_nozzle(n,cfg) for n in nozzles if n.enabled and n.nozzle_kind==scope and source_flow_kg_s(n,cfg)>0]
    if not members:return []
    mask,weights,_=target.arrays();z_values=np.asarray(target.z_mm)[mask&(weights>0)]
    if not len(z_values):return []
    positive=any(z>n.z_tip_mm for z in z_values for n in members)
    anisotropic=any(np.ptp(n.sector_ml)>1e-12 for n in members)
    reached=any(z>=n.z_tip_mm for z in z_values for n in members)
    keys={'h0_mm','r_nozzle_mm','z_tip_mm'} if reached else {'z_tip_mm'}
    if positive:
        if cfg.geometry_model=='two_angles':keys.update(('inner_angle_deg','outer_angle_deg'))
        else:keys.add('alpha_deg')
        if cfg.geometry_model in ('linear','quadratic'):keys.add('k_linear')
        if cfg.geometry_model=='quadratic':keys.add('k_quadratic')
        if cfg.geometry_model=='power':keys.update(('power','z_ref_mm'))
    if anisotropic and reached:
        keys.add('clock_deg')
        if positive:
            keys.add('rotation_deg_per_mm')
            if cfg.angular_model in ('linear','exponential'):keys.add('decay_length_mm')
            if cfg.angular_model=='diffusion':keys.add('diffusion_deg_sqrt_mm')
    return [k for k in GEOMETRY_KEYS if k in keys and any(n.enabled and n.nozzle_kind==scope and k not in n.geometry_overrides for n in nozzles)]

def _metrics(y,pred,mask):
    a=np.asarray(y)[mask];b=np.asarray(pred)[mask]
    if not len(a):return {'n':0,'RMSE':None,'MAE':None,'R2':None}
    res=b-a;denom=float(np.sum((a-a.mean())**2));scale=max(np.finfo(float).tiny,float(np.sum(a*a)))
    return {'n':len(a),'RMSE':float(np.sqrt(np.mean(res*res))),'MAE':float(np.mean(np.abs(res))),
            'R2':None if denom<=np.finfo(float).eps*scale else float(1-np.sum(res*res)/denom)}

class _Stopped(Exception):pass


def fit(snapshot,target,parameters,*,budget=120,regularization=0.,criterion='absolute',relative_floor=1e-9,seed=0,
        geometry_model=None,angular_model=None,progress=None,cancel=None):
    started=time.perf_counter();cfg,ns=snapshot.restore()
    if geometry_model is not None:cfg.geometry_model=geometry_model
    if angular_model is not None:cfg.angular_model=angular_model
    cfg.validate();mapping=target.validate(cfg);parameters=tuple(replace(p,unit=p.unit or PARAM_UNITS.get(p.key,"")) for p in parameters)
    if not np.isfinite(regularization) or regularization<0 or criterion not in ('absolute','relative') or not np.isfinite(relative_floor) or relative_floor<=0:raise ValueError('Неверные настройки критерия/регуляризации')
    active=[i for i,p in enumerate(parameters) if not p.fixed]
    if not active or budget<2*len(active)+10:raise ValueError('Выберите активные параметры и достаточный лимит вычислений')
    seen=set()
    for p in parameters:
        if p.key in PARAM_UNITS and p.unit!=PARAM_UNITS[p.key]:raise ValueError('Неверные единицы параметра: '+p.key)
        if (p.scope,p.key) in seen:raise ValueError('Параметр выбран дважды')
        seen.add((p.scope,p.key))
        if p.key not in active_parameters(cfg,ns,target,p.scope):raise ValueError(f'{p.scope}/{p.key}: параметр не активен при этих моделях/сечениях или зафиксирован override')
        if not all(np.isfinite(v) for v in (p.initial,p.minimum,p.maximum)) or not p.minimum<p.maximum or not p.minimum<=p.initial<=p.maximum:raise ValueError('Нужны конечные min < max и initial внутри границ')
    mask,weights,roles=target.arrays();used=mask&(weights>0);train=used&(roles=='train');holdout=used&(roles!='train')
    if train.sum()<=len(active):raise ValueError('Недостаточно независимых train-наблюдений для числа коэффициентов')
    z_used=sorted(set(z for i,(z,r,a) in enumerate(mapping) if used[i]));y=np.asarray(target.y)
    lo=np.array([parameters[i].minimum for i in active]);span=np.array([parameters[i].maximum-parameters[i].minimum for i in active]);initial=np.array([parameters[i].initial for i in active]);x0=(initial-lo)/span
    cache={};evaluations=0;invalid=0;best=None;initial_pred=None;initial_cost=None;status='budget';notes=['Чувствительность оценена конечными шагами консервативной квадратуры; требуется проверка сеточной сходимости.']
    optimize_limit=budget-2*len(active)
    base_cfg=deepcopy(cfg);base_ns=deepcopy(ns)
    def candidate(x):
        c=deepcopy(base_cfg);n=deepcopy(base_ns);values=[p.initial for p in parameters]
        for k,i in enumerate(active):values[i]=float(lo[k]+span[k]*x[k])
        for p,value in zip(parameters,values):c.nozzle_presets[p.scope][p.key]=value
        c.validate();validate_nozzles(n,c)
        for nozzle in n:
            if not nozzle.enabled:continue
            effective=effective_nozzle(nozzle,c)
            for z in set(c.z_mm)|set(z_used):
                if z>=effective.z_tip_mm:geometry_at(effective,z,c)
        return c,n,values
    def evaluate(x,diagnostic=False):
        nonlocal evaluations,invalid,best
        if cancel and cancel():raise _Stopped('cancelled')
        key=tuple(float(v) for v in x)
        if key in cache:return cache[key]
        if evaluations>=(budget if diagnostic else optimize_limit):raise _Stopped('budget')
        evaluations+=1
        try:
            c,n,values=candidate(x);planes={}
            for z in z_used:
                if cancel and cancel():raise _Stopped('cancelled')
                planes[z]=calculate_plane(n,c,z)
            if any(abs(p.rel_balance_error)>1e-10 for p in planes.values()):raise ValueError('Массовый баланс кандидата не замкнут')
            prediction=np.full(len(y),np.nan)
            fields={z:(p.cell_mass_kg_s if target.quantity=='cell_mass' else cell_values(p,METRIC[target.quantity],c.density_kg_m3)) for z,p in planes.items()}
            for i,(z,r,a) in enumerate(mapping):
                if used[i]:prediction[i]=fields[z][r,a]
            if not np.isfinite(prediction[used]).all():raise ValueError('Индекс при нулевом среднем не определён')
            res=(prediction[train]-y[train])*np.sqrt(weights[train])
            if criterion=='relative':res=res/np.maximum(np.abs(y[train]),relative_floor)
            residual=np.r_[res,np.sqrt(regularization)*(x-x0)]
            cost=float(np.dot(residual,residual));cache[key]=(residual,prediction,cost)
            if best is None or cost<best[2]:best=(np.array(x,copy=True),prediction.copy(),cost)
        except (ValueError,FloatingPointError):
            invalid+=1;penalty=(1e6 if initial_cost is None else 1e3*max(1.,np.sqrt(initial_cost)))*(1+np.linalg.norm(x-x0))
            cache[key]=(np.full(int(train.sum())+len(active),penalty),None,float('inf'))
        if progress:progress(evaluations,None if best is None else best[2])
        return cache[key]
    try:
        _,initial_pred,initial_cost=evaluate(x0)
        if initial_pred is None:raise ValueError('Начальные параметры не дают физически допустимого сопоставимого поля')
        # Conservative histograms are piecewise smooth. Deterministic coarse
        # exploration prevents tiny numerical derivatives from faking inactivity.
        if len(active)==1:
            seeds=[np.array([v]) for v in np.linspace(0,1,17)]
        else:
            seeds=[np.clip(x0+np.random.default_rng(seed).normal(0,.15,len(active)),0,1)]
            for i in range(len(active)):
                for sign in (-1,1):
                    x=x0.copy();x[i]=np.clip(x[i]+sign*.15,0,1);seeds.append(x)
        for x in seeds:evaluate(x)
        def jac(x):
            out=[]
            for i in range(len(active)):
                a=x.copy();b=x.copy();a[i]=max(0,x[i]-.003);b[i]=min(1,x[i]+.003)
                out.append((evaluate(b)[0]-evaluate(a)[0])/(b[i]-a[i]))
            return np.asarray(out).T
        result=least_squares(lambda x:evaluate(x)[0],best[0],jac=jac,bounds=(np.zeros(len(active)),np.ones(len(active))),
                             max_nfev=budget,xtol=1e-7,ftol=1e-7,gtol=1e-7)
        status='converged' if result.success else 'budget'
    except _Stopped as stopped:status=str(stopped)
    if best is None:
        if status=='cancelled':raise ValueError('Отмена до первого прямого расчёта')
        raise ValueError('Нет допустимого кандидата')
    x,prediction,cost=best;diagnostics={'rank':None,'condition':None,'standard_error':None,'invalid_candidates':invalid}
    if status!='cancelled':
        try:
            columns=[]
            for i in range(len(active)):
                a=x.copy();b=x.copy();a[i]=max(0,x[i]-.003);b[i]=min(1,x[i]+.003)
                ra=evaluate(a,True)[0][:int(train.sum())];rb=evaluate(b,True)[0][:int(train.sum())]
                columns.append((rb-ra)/(b[i]-a[i]))
            matrix=np.asarray(columns).T;singular=np.linalg.svd(matrix,compute_uv=False);rank=int(np.linalg.matrix_rank(matrix))
            condition=float(singular[0]/singular[-1]) if rank==len(active) and singular[-1]>0 else None
            diagnostics.update(rank=rank,condition=condition)
            if rank==len(active) and train.sum()>len(active):
                errors=(prediction[train]-y[train])*np.sqrt(weights[train])
                if criterion=='relative':errors/=np.maximum(np.abs(y[train]),relative_floor)
                variance=float(np.dot(errors,errors)/(train.sum()-len(active)))
                covariance=np.linalg.pinv(matrix.T@matrix)*variance
                diagnostics['standard_error']=(np.sqrt(np.maximum(0,np.diag(covariance)))*span).tolist()
            if rank<len(active) or condition is None or condition>1e6:notes.append('Параметры плохо идентифицируемы: ранг/обусловленность требуют проверки.')
        except _Stopped:notes.append('Лимит вычислений исчерпан; диагностика чувствительности неполная.')
    if status!='converged':notes.append('Оптимизация '+status+'; сходимость не подтверждена.')
    if train.sum()<10*len(active):notes.append('Мало независимых наблюдений относительно числа параметров.')
    if not holdout.any():notes.append('Holdout отсутствует: независимая проверка недоступна.')
    if invalid:notes.append(f'Отклонено физически недопустимых кандидатов: {invalid}.')
    if status=='cancelled':c,n= snapshot.restore();values=[p.initial for p in parameters]
    else:c,n,values=candidate(x)
    bounds=[p.key for p,v in zip(parameters,values) if np.isclose(v,p.minimum,rtol=0,atol=1e-5*(p.maximum-p.minimum)) or np.isclose(v,p.maximum,rtol=0,atol=1e-5*(p.maximum-p.minimum))]
    diagnostics['active_bounds']=bounds
    if bounds:notes.append('Достигнуты границы параметров: '+', '.join(bounds))
    metrics={'initial_train':_metrics(y,initial_pred if initial_pred is not None else prediction,train),
             'train':_metrics(y,prediction,train),'holdout':_metrics(y,prediction,holdout)}
    if metrics['train']['R2'] is None:notes.append('R² не определён для постоянного топливного эталона.')
    return FitResult(snapshot,parameters,tuple(values),json.dumps(project_to_json(c,n),ensure_ascii=False,allow_nan=False),
                     tuple(initial_pred if initial_pred is not None else prediction),tuple(prediction),target,metrics,
                     float(initial_cost or 0),cost,evaluations,status,tuple(notes),diagnostics,
                     dict(budget=budget,regularization=regularization,criterion=criterion,relative_floor=relative_floor,seed=seed,
                          geometry_model=c.geometry_model,angular_model=c.angular_model,algorithm='bounded least_squares + conservative coarse exploration'),time.perf_counter()-started)


def predict_snapshot(snapshot,target,cancel=None):
    """Original project prediction at target coordinates for an honest before/after view."""
    cfg,nozzles=snapshot.restore();mapping=target.validate(cfg);mask,weights,_=target.arrays();used=mask&(weights>0)
    fields={};prediction=np.full(len(target.y),np.nan)
    for z in sorted({z for i,(z,_,_) in enumerate(mapping) if used[i]}):
        if cancel and cancel():raise ValueError('Предпросмотр исходного проекта отменён')
        plane=calculate_plane(nozzles,cfg,z)
        fields[z]=plane.cell_mass_kg_s if target.quantity=='cell_mass' else cell_values(plane,METRIC[target.quantity],cfg.density_kg_m3)
    for i,(z,r,a) in enumerate(mapping):
        if used[i]:prediction[i]=fields[z][r,a]
    return tuple(prediction)
