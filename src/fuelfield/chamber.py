"""Analytic axisymmetric chamber geometry. z=x [mm], global theta=+X/CCW.

No transport/air/film law lives here. Raw records preserve unknown user fields.
Only drawing uses chord approximations; radius and normals remain analytic.
"""
from copy import deepcopy
from functools import lru_cache
import json
import math
import numpy as np

UNITS={'mm':1.,'cm':10.,'m':1000.}
TOL=1e-7
WALL_BOUND_NOTICE=('wall-bound — геометрически неразрешённый расход вне локального '
    'тракта. Это не расчёт столкновения, плёнки или движения вдоль стенки; '
    'предыдущие пересечения и последующий возврат пока не моделируются.')


def number(value,label='Размер'):
    if isinstance(value,bool):raise ValueError(label+': требуется конечное число')
    try:value=float(value)
    except (TypeError,ValueError):raise ValueError(label+': требуется конечное число') from None
    if not math.isfinite(value):raise ValueError(label+': требуется конечное число')
    return value


class Primitive:
    def __init__(self,record):
        if not isinstance(record,dict):raise ValueError('Примитив должен быть объектом JSON')
        self.raw=deepcopy(record);self.id=record.get('id');self.wall=record.get('wall');self.kind=record.get('kind')
        if not isinstance(self.id,str) or not self.id.strip():raise ValueError('Нужен стабильный ID примитива')
        if self.wall not in ('inner','outer','reference'):raise ValueError(self.id+': стенка inner/outer/reference')
        if self.kind not in ('point','line','arc'):raise ValueError(self.id+': тип point/line/arc')
        self.unit=record.get('unit','mm')
        if self.unit not in UNITS:raise ValueError(self.id+': единицы mm/cm/m')
        factor=UNITS[self.unit]
        def point(key):
            value=record.get(key)
            if not isinstance(value,(list,tuple)) or len(value)!=2:raise ValueError(self.id+': '+key+' — пара z/R')
            return np.array([number(v,self.id+' '+key)*factor for v in value])
        self.start=point('start');self.end=self.start.copy() if self.kind=='point' else point('end')
        self.lo,self.hi=sorted((float(self.start[0]),float(self.end[0])))
        if min(self.start[1],self.end[1])<=0:raise ValueError(self.id+': радиус должен быть положительным')
        if self.kind=='point':return
        if self.hi-self.lo<=TOL:raise ValueError(self.id+': вертикальная/нулевая стенка не задаёт R(z)')
        if self.kind=='arc':
            self.center=point('center');self.radius=float(np.linalg.norm(self.start-self.center))
            if self.radius<=TOL:raise ValueError(self.id+': нулевой радиус дуги')
            if not np.isclose(np.linalg.norm(self.end-self.center),self.radius,rtol=1e-10,atol=TOL):raise ValueError(self.id+': концы не лежат на одной окружности')
            if 'radius' in record and not np.isclose(number(record['radius'])*factor,self.radius,rtol=1e-10,atol=TOL):raise ValueError(self.id+': заданный радиус не согласован с центром/концами')
            clockwise=record.get('clockwise',False)
            if not isinstance(clockwise,bool):raise ValueError(self.id+': clockwise должен быть логическим')
            self.angle=math.atan2(*(self.start-self.center)[::-1]);end=math.atan2(*(self.end-self.center)[::-1])
            self.sweep=-((self.angle-end)%(2*math.pi)) if clockwise else ((end-self.angle)%(2*math.pi))
            if abs(self.sweep)<1e-12:raise ValueError(self.id+': полная/нулевая окружность не является R(z)')
            lower,upper=sorted((self.angle,self.angle+self.sweep))
            if any(lower+1e-10<k*math.pi<upper-1e-10 for k in range(-4,5)):raise ValueError(self.id+': дуга поворачивает по z (самоперекрытие профиля)')
            self.branch=1 if math.sin(self.angle+self.sweep/2)>0 else -1
            if self.range(self.lo,self.hi)[0]<=0:raise ValueError(self.id+': дуга достигает неположительного R')

    def at(self,z):
        values=np.asarray(z,dtype=float)
        if np.any(~np.isfinite(values)) or np.any(values<self.lo-TOL) or np.any(values>self.hi+TOL):raise ValueError(self.id+': z вне примитива')
        if self.kind=='point':return np.zeros_like(values)+self.start[1]
        if self.kind=='line':return self.start[1]+(values-self.start[0])*(self.end[1]-self.start[1])/(self.end[0]-self.start[0])
        return self.center[1]+self.branch*np.sqrt(np.maximum(0,self.radius**2-(values-self.center[0])**2))

    def range(self,a,b):
        samples=[a,b]
        if self.kind=='arc' and a<self.center[0]<b:samples.append(self.center[0])
        values=self.at(np.asarray(samples));return float(values.min()),float(values.max())

    def discretize(self,error_mm=.01):
        error=number(error_mm,'Погрешность хорды')
        if error<=0:raise ValueError('Погрешность хорды должна быть >0')
        if self.kind=='point':return self.start[None,:].copy()
        if self.kind=='line':return np.array([self.start,self.end])
        step=2*math.acos(max(-1.,1-min(error/self.radius,2.)))
        count=max(1,math.ceil(abs(self.sweep)/max(step,1e-8)))
        if count>1000000:raise ValueError('Погрешность требует более миллиона сегментов дуги')
        angles=self.angle+np.linspace(0,self.sweep,count+1)
        points=self.center+self.radius*np.column_stack((np.cos(angles),np.sin(angles)))
        points[0]=self.start;points[-1]=self.end;return points

    def normal(self,z):
        radius=float(self.at(z))
        if self.kind=='point':raise ValueError('У отдельной точки нет нормали')
        if self.kind=='line':
            vector=np.array([-(self.end[1]-self.start[1])/(self.end[0]-self.start[0]),1.])
        else:vector=np.array([float(z)-self.center[0],radius-self.center[1]])*self.branch
        vector/=np.linalg.norm(vector)
        return vector*(-1 if self.wall=='outer' else 1)


class Chamber:
    def __init__(self,raw):
        if not isinstance(raw,dict) or type(raw.get('schema_version',1)) is not int or raw.get('schema_version',1)!=1:raise ValueError('Неподдерживаемая схема геометрии')
        self.raw=deepcopy(raw)
        if not isinstance(raw.get('enabled',False),bool):raise ValueError('enabled геометрии должен быть логическим')
        self.enabled=raw.get('enabled',False)
        self.base=number(raw.get('base_z_mm',0),'Базовая плоскость');self.outlet=number(raw.get('outlet_z_mm',60),'Выход ЖТ')
        if self.base<0 or self.outlet<=self.base:raise ValueError('Требуется 0 ≤ базовая плоскость < выход ЖТ')
        self.error=number(raw.get('chord_error_mm',.01));self.step=number(raw.get('internal_step_mm',1.))
        if self.error<=0 or self.step<=0:raise ValueError('Погрешность хорды и внутренний шаг должны быть >0')
        primitives=raw.get('primitives',[])
        if not isinstance(primitives,list):raise ValueError('Примитивы: список')
        self.primitives=tuple(Primitive(p) for p in primitives)
        ids=[p.id for p in self.primitives]
        if len(ids)!=len(set(ids)):raise ValueError('Повторный ID примитива')
        self.profiles={wall:sorted((p for p in self.primitives if p.wall==wall and p.kind!='point'),key=lambda p:p.lo) for wall in ('inner','outer')}
        for wall,profile in self.profiles.items():
            if not profile:raise ValueError('Не задан профиль '+wall)
            if abs(profile[0].lo-self.base)>TOL or abs(profile[-1].hi-self.outlet)>TOL:raise ValueError(wall+': профиль должен покрывать базовую плоскость…выход')
            for left,right in zip(profile,profile[1:]):
                if abs(left.hi-right.lo)>TOL:raise ValueError(wall+': разрыв/перекрытие по z')
                if abs(float(left.at(left.hi)-right.at(right.lo)))>TOL:raise ValueError(wall+': концы примитивов не сопряжены')
        knots=sorted({self.base,self.outlet,*[p.lo for p in self.primitives if p.kind!='point'],*[p.hi for p in self.primitives if p.kind!='point']})
        for a,b in zip(knots,knots[1:]):
            if a<self.base or b>self.outlet:continue
            self._certify_gap(self.primitive_at('inner',(a+b)/2),self.primitive_at('outer',(a+b)/2),a,b)
        self.sections=deepcopy(raw.get('sections',[]));self._validate_sections()
        types=raw.get('types',{})
        if not isinstance(types,dict) or any(not isinstance(k,str) or not k.strip() or not isinstance(v,dict) for k,v in types.items()):raise ValueError('Пользовательские типы: словарь имён и определений')
        self.objects=deepcopy(raw.get('objects',[]));self.groups=deepcopy(raw.get('groups',[]));self._validate_objects()
        self.variable=any(p.kind=='arc' or abs(p.start[1]-p.end[1])>TOL for p in self.primitives if p.wall in self.profiles and p.kind!='point')

    def _certify_gap(self,inner,outer,a,b,depth=0):
        # H is smooth on each common primitive interval. Its only interior
        # extrema solve R'out=R'in. Lines/circular arcs give closed forms;
        # independent interval bounds would explode for near-parallel walls.
        candidates=[a,b]
        if inner.kind==outer.kind=='arc':
            for sign in (-1,1):
                denominator=outer.radius-sign*inner.radius
                if denominator!=0:
                    candidates.append((outer.radius*inner.center[0]-sign*inner.radius*outer.center[0])/denominator)
        elif inner.kind=='arc' or outer.kind=='arc':
            arc,line=(inner,outer) if inner.kind=='arc' else (outer,inner)
            slope=(line.end[1]-line.start[1])/(line.end[0]-line.start[0])
            candidates.append(arc.center[0]-arc.branch*slope*arc.radius/math.sqrt(1+slope*slope))
        for x in candidates:
            if not a<=x<=b:continue
            if float(outer.at(x)-inner.at(x))<=TOL:raise ValueError('Стенки пересекаются/соприкасаются: Rout ≤ Rin')

    def primitive_at(self,wall,z):
        z=number(z,'Координата сечения')
        if z<self.base-TOL or z>self.outlet+TOL:raise ValueError('Сечение вне покрытого профилем интервала')
        for p in self.profiles[wall]:
            if p.lo-TOL<=z<=p.hi+TOL:return p
        raise ValueError('Профиль не покрывает z='+str(z))

    def bounds(self,z):return tuple(float(self.primitive_at(wall,z).at(z)) for wall in ('inner','outer'))

    def grid(self,z,n):
        if isinstance(n,bool) or not isinstance(n,(int,np.integer)) or n<1:raise ValueError('Число поясов должно быть целым >0')
        rin,rout=self.bounds(z);edges=np.linspace(rin,rout,n+1);return (edges[:-1]+edges[1:])/2,edges

    def stations(self):
        count=math.ceil((self.outlet-self.base)/self.step)
        if count>100000:raise ValueError('Внутренняя сетка геометрии превышает 100000 шагов')
        return np.linspace(self.base,self.outlet,count+1)

    def _validate_sections(self):
        if not isinstance(self.sections,list) or not 1<=len(self.sections)<=50:raise ValueError('Нужны 1–50 ручных сечений с выходным')
        ids=[];positions=[];outlets=[]
        for section in self.sections:
            if not isinstance(section,dict) or not isinstance(section.get('id'),str) or not section['id']:raise ValueError('Нужен ID сечения')
            ids.append(section['id']);z=number(section.get('z_mm'),'Сечение');self.bounds(z);positions.append(z)
            if not isinstance(section.get('is_outlet',False),bool):raise ValueError('Признак выходного сечения должен быть логическим')
            if section.get('is_outlet',False):outlets.append(z)
        if len(set(ids))!=len(ids) or any(abs(a-b)<=TOL for i,a in enumerate(positions) for b in positions[:i]):raise ValueError('Повторный ID/координата сечения')
        if len(outlets)!=1 or abs(outlets[0]-self.outlet)>TOL:raise ValueError('Обязательно ровно одно выходное сечение ЖТ')

    def check_section_list(self,z_values):
        requested=[number(v) for v in z_values];stored=[number(s['z_mm']) for s in self.sections]
        if len(requested)!=len(stored) or any(abs(a-b)>TOL for a,b in zip(requested,stored)):raise ValueError('Список z и ручные сечения геометрии различаются; примените редактор сечений')

    def _validate_objects(self):
        if not isinstance(self.objects,list) or not isinstance(self.groups,list):raise ValueError('Объекты и группы: списки')
        ids=[];nozzle_ids=[];primitives={p.id:p for p in self.primitives}
        for obj in self.objects:
            if not isinstance(obj,dict) or not isinstance(obj.get('id'),str) or not obj['id']:raise ValueError('Нужен стабильный ID объекта')
            ids.append(obj['id'])
            if not isinstance(obj.get('type'),str) or not obj['type']:raise ValueError('Нужен тип объекта')
            if obj['type']=='nozzle':
                nid=obj.get('nozzle_id')
                if isinstance(nid,bool) or not isinstance(nid,int) or not 1<=nid<=140:raise ValueError('Объект форсунки ссылается на ID 1…140')
                nozzle_ids.append(nid)
            else:
                z=number(obj.get('z_mm'),'z объекта');radius=number(obj.get('radius_mm'),'R объекта');self.bounds(z)
                if radius<=0:raise ValueError('R объекта должен быть >0')
                theta=number(obj.get('theta_deg',0),'θ объекта')
                if not 0<=theta<360:raise ValueError('Угол объекта нормализуйте в [0,360)')
            wall=obj.get('wall','free')
            if wall not in ('inner','outer','free'):raise ValueError('Стенка объекта: inner/outer/free')
            bound=obj.get('primitive_id','')
            if not isinstance(bound,str):raise ValueError('Привязка к примитиву: строка ID или пустая строка')
            if bound and (bound not in primitives or primitives[bound].wall!=wall):raise ValueError('Не найдена согласованная привязка объекта к примитиву')
            if bound and obj['type']!='nozzle':primitives[bound].at(obj['z_mm'])
            for key in ('parameters','units','provenance'):
                if not isinstance(obj.get(key,{}),dict):raise ValueError(key+' объекта: словарь')
            params=obj.get('parameters',{})
            for key in ('flow_kg_s','penetration_mm'):
                if params.get(key) is not None and number(params[key],key)<0:raise ValueError(key+' должен быть ≥0 или неизвестен (null)')
            for key in ('inclination_deg','azimuth_deg'):
                if params.get(key) is not None:number(params[key],key)
        if len(ids)!=len(set(ids)) or len(nozzle_ids)!=len(set(nozzle_ids)):raise ValueError('Повторный объект/ссылка на форсунку')
        owners=set();group_ids=[]
        for group in self.groups:
            if not isinstance(group,dict):raise ValueError('Группа должна быть объектом JSON')
            group_ids.append(group.get('id'));members=group.get('members',[])
            if not isinstance(group.get('id'),str) or not group['id']:raise ValueError('Нужен ID группы')
            if not isinstance(members,list) or not members or any(not isinstance(m,str) or not m for m in members) or len(members)!=len(set(members)):raise ValueError('Нужны уникальные строковые ID участников группы')
            if not set(members).issubset(ids):raise ValueError('Группа ссылается на отсутствующие объекты')
            if owners.intersection(members):raise ValueError('Объект не может принадлежать двум угловым группам')
            owners.update(members);group_angles(group)
        if len(group_ids)!=len(set(group_ids)):raise ValueError('Повторный ID группы')

    def object_position(self,obj,nozzles,cfg=None):
        if obj['type']=='nozzle':
            n=next(n for n in nozzles if n.id==obj['nozzle_id'])
            from .v02 import effective_nozzle
            n=effective_nozzle(n,cfg) if cfg is not None else n
            z,radius,theta=n.z_tip_mm,n.radius_mm,n.theta_deg
        else:z,radius,theta=obj['z_mm'],obj['radius_mm'],obj.get('theta_deg',0)
        wall=obj.get('wall','free')
        if wall!='free':
            p=next((p for p in self.primitives if p.id==obj.get('primitive_id')),None) or self.primitive_at(wall,z)
            radius=float(p.at(z))+number(obj.get('radial_offset_mm',0))
        return z,radius,theta

    def resolve_nozzle_anchors(self,nozzles,cfg):
        updated=deepcopy(nozzles);lookup={n.id:n for n in updated}
        for obj in self.objects:
            if obj['type']=='nozzle' and obj['nozzle_id'] in lookup and obj.get('wall','free')!='free':
                _,radius,_=self.object_position(obj,updated,cfg)
                if radius<=0:raise ValueError(obj['id']+': привязка даёт неположительный радиус')
                lookup[obj['nozzle_id']].radius_mm=radius
        return updated

    def validate_nozzle_anchors(self,nozzles,cfg):
        lookup={n.id:n for n in nozzles}
        for obj in self.objects:
            if obj['type']=='nozzle' and obj['nozzle_id'] in lookup and obj.get('wall','free')!='free':
                _,radius,_=self.object_position(obj,nozzles,cfg)
                if radius<=0 or abs(lookup[obj['nozzle_id']].radius_mm-radius)>TOL:raise ValueError(obj['id']+': координата форсунки не согласована с привязкой к стенке')


def group_angles(group):
    members=group.get('members',[]);n=len(members)
    if not n:raise ValueError('Пустая угловая группа')
    direction=group.get('direction',-1)
    if isinstance(direction,bool) or direction not in (-1,1):raise ValueError('Направление группы ±1')
    phase=number(group.get('phase_deg',0))+number(group.get('row_offset_deg',0));mode=group.get('mode','uniform')
    if mode=='uniform':relative=np.arange(n)*360/n
    elif mode=='individual':
        relative=np.array([number(v,'Угол') for v in group.get('angles_deg',[])])
        if len(relative)!=n:raise ValueError('Количество индивидуальных углов должно совпадать с количеством объектов')
    elif mode=='steps':
        steps=np.array([number(v,'Угловой шаг') for v in group.get('steps_deg',[])])
        if len(steps)!=n or np.any(steps<=0) or not np.isclose(steps.sum(),360,atol=1e-7,rtol=0):raise ValueError('Нужны N положительных шагов с суммой 360°')
        relative=np.r_[0,np.cumsum(steps[:-1])]
    else:raise ValueError('Режим группы: uniform/individual/steps')
    result=(phase+direction*relative)%360
    if n>1 and np.min(np.diff(np.r_[np.sort(result),np.sort(result)[0]+360]))<=1e-7:raise ValueError('У группы повторяющиеся углы')
    return result


def legacy_geometry(cfg,nozzles=()):
    outlet=max(max(cfg.z_mm),1.);sections=[dict(id=f'S{i+1}',z_mm=float(z),is_outlet=abs(z-outlet)<=TOL) for i,z in enumerate(cfg.z_mm)]
    if not any(s['is_outlet'] for s in sections):sections.append(dict(id='outlet',z_mm=outlet,is_outlet=True))
    return dict(schema_version=1,enabled=False,base_z_mm=0.,outlet_z_mm=outlet,chord_error_mm=.01,internal_step_mm=1.,
        primitives=[dict(id=wall+'-1',wall=wall,kind='line',start=[0.,r],end=[outlet,r],unit='mm') for wall,r in [('inner',cfg.r_in_mm),('outer',cfg.r_out_mm)]],
        sections=sections,objects=[dict(id='N'+str(n.id),type='nozzle',nozzle_id=n.id,wall='free',parameters={},units={},provenance={}) for n in nozzles],groups=[],types={})


@lru_cache(maxsize=64)
def _cached(payload):return Chamber(json.loads(payload))


def chamber(cfg):
    raw=cfg.geometry
    if not isinstance(raw,dict):raise ValueError('Геометрия должна быть объектом JSON')
    if not raw:return None
    model=_cached(json.dumps(raw,sort_keys=True,ensure_ascii=False,allow_nan=False))
    return model if model.enabled else None


def apply_groups(raw,nozzles):
    """Explicit action: change positions only, never nozzle data/identity/flow."""
    data=deepcopy(raw);model=Chamber(data);updated=deepcopy(nozzles);lookup={n.id:n for n in updated};objects={o['id']:o for o in data['objects']}
    for group in data['groups']:
        for member,angle in zip(group['members'],group_angles(group)):
            obj=objects[member]
            if obj['type']=='nozzle':lookup[obj['nozzle_id']].theta_deg=float(angle)
            else:obj['theta_deg']=float(angle)
    return data,updated


def sync_groups(raw,nozzles):
    """An explicit edit outside a group becomes its individual-angle record."""
    data=deepcopy(raw)
    if not data:return data
    lookup={n.id:n for n in nozzles};objects={o['id']:o for o in data.get('objects',[])}
    for group in data.get('groups',[]):
        physical=np.array([lookup[objects[m]['nozzle_id']].theta_deg if objects[m]['type']=='nozzle' else objects[m].get('theta_deg',0) for m in group['members']])
        if not np.allclose((physical-group_angles(group)+180)%360-180,0,rtol=0,atol=1e-7):
            group['mode']='individual';group['angles_deg']=((physical-number(group.get('phase_deg',0))-number(group.get('row_offset_deg',0)))*group.get('direction',-1)%360).tolist()
    return data


def sync_anchors(raw,nozzles,cfg,previous):
    """Explicit source-R edit changes offset; wall/z edit moves its anchor."""
    data=deepcopy(raw)
    if not data:return data,nozzles
    model=Chamber(data);old={n.id:n for n in previous};lookup={n.id:n for n in nozzles}
    for obj in data.get('objects',[]):
        if obj['type']!='nozzle' or obj.get('wall','free')=='free':continue
        n=lookup[obj['nozzle_id']]
        if n.radius_mm!=old[n.id].radius_mm:
            _,expected,_=model.object_position(obj,nozzles,cfg)
            obj['radial_offset_mm']=number(obj.get('radial_offset_mm',0))+n.radius_mm-expected
    return data,Chamber(data).resolve_nozzle_anchors(nozzles,cfg)
