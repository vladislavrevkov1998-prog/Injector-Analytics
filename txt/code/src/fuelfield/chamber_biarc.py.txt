"""Limited two-circle fallback for an arc pair between existing end tangents.

This is a geometric biarc interpolant, not a general CAD optimizer. The two
outer endpoints and tangents remain exact; the common, non-fixed joint may
move. Fixed members of its coincidence component are never displaced.
"""
from copy import deepcopy
import math
import numpy as np
from scipy.optimize import minimize_scalar
from .chamber import Primitive,UNITS,TOL
from .chamber_editing import move_endpoint


def _key(ref):return ref['primitive_id'],ref['endpoint']
def _point(p,role):return np.array(p[role],dtype=float)*UNITS[p.get('unit','mm')]
def _rotate(v):return np.array([-v[1],v[0]])


def _normal_and_tangent(raw,p,role,join_role):
    records={q['id']:q for q in raw['primitives']};point=_point(p,role)
    normal=point-_point(p,'center')
    for c in raw.get('constraints',[]):
        if c['kind']!='tangent' or not c.get('enabled',True):continue
        a,b=_key(c['a']),_key(c['b'])
        target=b if a==(p['id'],role) else a if b==(p['id'],role) else None
        if target is None:continue
        reference=records[target[0]]
        if reference['kind']=='line':normal=_rotate(_point(reference,'end')-_point(reference,'start'))
        elif reference['kind']=='arc':normal=_point(reference,target[1])-_point(reference,'center')
        break
    length=np.linalg.norm(normal)
    if length<=TOL:raise ValueError('Не определена внешняя касательная дуги '+p['id'])
    normal=normal/length;tangent=_rotate(normal)
    delta=_point(p,join_role)-point
    if abs(tangent[0])>1e-12:
        if tangent[0]*delta[0]<0:tangent=-tangent
    else:
        old=_rotate(point-_point(p,'center'))*(-1 if p.get('clockwise',False) else 1)*(1 if role=='start' else -1)
        if np.dot(tangent,old)<0:tangent=-tangent
    return normal,tangent


def _join(p0,p1,t0,t1,ratio):
    # Biarc construction: |v-d0*t0-d1*t1|=d0+d1, d0=ratio*d1.
    # In chord-length units this gives a*d1²+b*d1-1=0; the positive root
    # determines the common tangent join between the two virtual tangent ends.
    v=p1-p0;scale=float(np.linalg.norm(v))
    if scale<=TOL:return None
    v=v/scale;a=2*ratio*max(0.,1-float(np.dot(t0,t1)));b=2*float(np.dot(v,ratio*t0+t1))
    if a<1e-13:
        if b<=1e-13:return None
        d1=1/b
    else:
        root=math.sqrt(b*b+4*a)
        d1=2/(b+root) if b>=0 else (-b+root)/(2*a)
    d0=ratio*d1
    return p0+scale*(d0*v+d0*d1*(t0-t1))/(d0+d1)


def _fit(p,outer_role,joint,normal,tangent):
    result=deepcopy(p);contact=_point(p,outer_role);delta=joint-contact;height=float(np.dot(delta,normal))
    if np.linalg.norm(delta)<=TOL or abs(height)<=TOL:raise ValueError('Нужна конечная ненулевая дуга')
    factor=UNITS[p.get('unit','mm')];join_role='end' if outer_role=='start' else 'start'
    result[join_role]=(joint/factor).tolist();result['center']=((contact+float(np.dot(delta,delta))/(2*height)*normal)/factor).tolist();result.pop('radius',None)
    for clockwise in (p.get('clockwise',False),not p.get('clockwise',False)):
        result['clockwise']=clockwise
        try:primitive=Primitive(result)
        except ValueError:continue
        direction=_rotate(contact-primitive.center)/primitive.radius*(-1 if clockwise else 1)*(1 if outer_role=='start' else -1)
        if float(np.dot(direction,tangent))>1-1e-8:return result
    raise ValueError(p['id']+': biarc не задаёт однозначный профиль R(z)')


def candidates(raw,constraint):
    """Yield exact admissible pair shapes closest to the proposed common joint."""
    records={p['id']:p for p in raw['primitives']};a,b=(_key(constraint[k]) for k in ('a','b'));first,second=records[a[0]],records[b[0]]
    if first['kind']!='arc' or second['kind']!='arc' or first.get('fixed',False) or second.get('fixed',False):return
    outer0='end' if a[1]=='start' else 'start';outer1='end' if b[1]=='start' else 'start'
    p0,p1=_point(first,outer0),_point(second,outer1)
    try:n0,t0=_normal_and_tangent(raw,first,outer0,a[1]);n1,tback=_normal_and_tangent(raw,second,outer1,b[1])
    except (ValueError,KeyError):return
    t1=-tback;desired=(_point(first,a[1])+_point(second,b[1]))/2
    graph={}
    for c in raw.get('constraints',[]):
        if c.get('enabled',True):
            x,y=(_key(c[k]) for k in ('a','b'));graph.setdefault(x,set()).add(y);graph.setdefault(y,set()).add(x)
    component={a,b};queue=list(component)
    while queue:
        for neighbor in graph.get(queue.pop(),()):
            if neighbor not in component:component.add(neighbor);queue.append(neighbor)
    def objective(logratio):
        joint=_join(p0,p1,t0,t1,math.exp(logratio))
        return float(np.dot(joint-desired,joint-desired)) if joint is not None else 1e100
    optimum=minimize_scalar(objective,bounds=(-8.,8.),method='bounded',options={'xatol':1e-10})
    ratios=[math.exp(float(optimum.x)),1.,.25,.5,2.,4.]
    shapes=[]
    for ratio in ratios:
        joint=_join(p0,p1,t0,t1,ratio)
        if joint is None or any(records[i].get('fixed',False) and np.linalg.norm(_point(records[i],r)-joint)>TOL for i,r in component):continue
        try:
            fit0=_fit(first,outer0,joint,n0,t0);fit1=_fit(second,outer1,joint,n1,tback)
            candidate=deepcopy(raw);updated={p['id']:p for p in candidate['primitives']}
            for fit in (fit0,fit1):updated[fit['id']].clear();updated[fit['id']].update(fit)
            for identity,role in component:
                if identity in (a[0],b[0]):continue
                p=updated[identity]
                if np.linalg.norm(_point(p,role)-joint)<=TOL:continue
                moved=move_endpoint(p,role,joint/UNITS[p.get('unit','mm')]);p.clear();p.update(moved)
            shapes.append((float(np.linalg.norm(joint-desired)),candidate))
        except ValueError:continue
    for _,shape in sorted(shapes,key=lambda pair:pair[0]):yield shape
