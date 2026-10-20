"""Small exact endpoint/line-circle constraints, independent of Qt and flow math.

Coordinates are solved in physical millimetres. There is no iterative CAD
optimizer: incompatible fixed anchors or tangencies produce a diagnostic.
"""
from collections import defaultdict
from copy import deepcopy
import math
import numpy as np
from .chamber import Primitive, UNITS, TOL
from .chamber_editing import arc_circle, next_id


def _key(ref):
    if not isinstance(ref,dict):raise ValueError('Связь: ссылка на точку должна быть объектом')
    identity,role=ref.get('primitive_id'),ref.get('endpoint')
    if not isinstance(identity,str) or role not in ('start','end'):raise ValueError('Связь: нужны ID примитива и start/end')
    return identity,role


def _records(raw):return {p['id']:p for p in raw.get('primitives',[])}
def _point(p,role):return np.array(p[role],dtype=float)*UNITS[p.get('unit','mm')]
def _same(a,b):return np.allclose(a,b,rtol=0,atol=TOL)


def constraint_records(raw):
    """Old projects retain their former shared joints until first explicit edit."""
    if 'constraints' in raw:return deepcopy(raw['constraints'])
    result=[];seen=[]
    for p in raw.get('primitives',[]):
        if p['kind']=='point':continue
        for role in ('start','end'):
            match=next(((q,r) for q,r in seen if q['wall']==p['wall'] and q['id']!=p['id'] and _same(_point(q,r),_point(p,role))),None)
            if match:
                q,r=match
                identity=next_id({**raw,'groups':raw.get('groups',[])+result},'C')
                result.append(dict(id=identity,kind='coincident',a=dict(primitive_id=q['id'],endpoint=r),b=dict(primitive_id=p['id'],endpoint=role)))
            seen.append((p,role))
    return result


def materialize_constraints(raw):
    if 'constraints' not in raw:raw['constraints']=constraint_records(raw)
    raw['schema_version']=2
    return raw


def _structure(raw):
    records=_records(raw);entries=raw.get('constraints',[])
    for p in records.values():
        if not isinstance(p.get('fixed',False),bool):raise ValueError(p['id']+': «Зафиксировать кривую» должен быть логическим')
    if not isinstance(entries,list):raise ValueError('Связи: требуется список')
    ids=set()
    for c in entries:
        if not isinstance(c,dict) or not isinstance(c.get('id'),str) or not c['id'] or c['id'] in ids:raise ValueError('Связи: нужен уникальный ID')
        ids.add(c['id'])
        if c.get('kind') not in ('coincident','tangent'):raise ValueError(c['id']+': неизвестный вид связи')
        if not isinstance(c.get('enabled',True),bool):raise ValueError(c['id']+': enabled должен быть логическим')
        for name in ('a','b'):
            identity,role=_key(c.get(name));p=records.get(identity)
            if p is None or role not in p or (p['kind']=='point' and role!='start'):raise ValueError(c['id']+': не найдена точка '+identity+'.'+role)
        if _key(c['a'])[0]==_key(c['b'])[0]:raise ValueError(c['id']+': выберите точки разных примитивов')
        if c['kind']=='tangent' and {records[_key(c[k])[0]]['kind'] for k in ('a','b')}!={'arc','line'}:raise ValueError(c['id']+': касательность задаётся между дугой и отрезком')
    return records,entries


def _arc_line(c,records):
    a,b=_key(c['a']),_key(c['b'])
    return (a,b) if records[a[0]]['kind']=='arc' else (b,a)


def _circle_branch(arc):
    errors=[]
    for clockwise in (arc.get('clockwise',False),not arc.get('clockwise',False)):
        candidate=deepcopy(arc);candidate['clockwise']=clockwise
        try:Primitive(candidate);return candidate
        except ValueError as exc:errors.append(str(exc))
    raise ValueError(arc['id']+': касательная окружность не задаёт однозначный R(z); '+errors[0])


def tangent_arc(arc,arc_endpoint,line,line_endpoint):
    """Fit one exact circle through the opposite end and a fixed line contact."""
    if arc.get('kind')!='arc' or line.get('kind')!='line':raise ValueError('Для касания нужны дуга и отрезок')
    if arc_endpoint not in ('start','end') or line_endpoint not in ('start','end'):raise ValueError('Выберите концы дуги и отрезка')
    contact=_point(line,line_endpoint);opposite='end' if arc_endpoint=='start' else 'start';other=_point(arc,opposite)
    direction=_point(line,'end')-_point(line,'start');length=np.linalg.norm(direction)
    if length<=TOL:raise ValueError('Касательность: нулевой отрезок')
    normal=np.array([-direction[1],direction[0]])/length;delta=other-contact
    tolerance=max(TOL,64*np.finfo(float).eps*float(np.linalg.norm(delta)))
    height=float(np.dot(delta,normal))
    if np.linalg.norm(delta)<=tolerance or abs(height)<=tolerance:raise ValueError('Касательность невозможна: для заданных концов нет конечной окружности')
    offset=float(np.dot(delta,delta))/(2*height);center=contact+offset*normal
    factor=UNITS[arc.get('unit','mm')];result=deepcopy(arc)
    result[arc_endpoint]=(contact/factor).tolist();result['center']=(center/factor).tolist();result.pop('radius',None)
    result=_circle_branch(result)
    if arc.get('fixed',False) and not _geometry_same(arc,result):raise ValueError(arc['id']+': дуга зафиксирована; снимите флажок перед касанием')
    return result


def _geometry_same(a,b):
    if a['kind']!=b['kind']:return False
    for key in ('start','end','center'):
        if (key in a)!=(key in b):return False
        if key in a and not _same(_point(a,key),_point(b,key)):return False
    return a.get('clockwise',False)==b.get('clockwise',False)


def _check_fixed(before,after):
    updated=_records(after)
    for identity,p in _records(before).items():
        q=updated.get(identity)
        if p.get('fixed',False) and (q is None or q.get('fixed',False) and not _geometry_same(p,q)):
            raise ValueError(identity+': кривая зафиксирована; сначала снимите флажок')


def validate_constraints(raw):
    records,entries=_structure(raw)
    for c in entries:
        if not c.get('enabled',True):continue
        a,b=(_key(c[k]) for k in ('a','b'))
        if not _same(_point(records[a[0]],a[1]),_point(records[b[0]],b[1])):raise ValueError(c['id']+': точки связи не совпадают')
        if c['kind']=='tangent':
            arc_ref,line_ref=_arc_line(c,records);arc,line=records[arc_ref[0]],records[line_ref[0]]
            contact=_point(line,line_ref[1]);radius=contact-_point(arc,'center');direction=_point(line,'end')-_point(line,'start')
            denominator=np.linalg.norm(radius)*np.linalg.norm(direction)
            if denominator<=TOL*TOL or abs(float(np.dot(radius,direction)))/denominator>1e-9:raise ValueError(c['id']+': касательность не выполнена')


def apply_constraints(raw,drivers=(),baseline=None):
    result=deepcopy(raw)
    result['schema_version']=2
    if 'constraints' not in result:
        result['constraints']=constraint_records(baseline or raw)
        result['schema_version']=2
    records,entries=_structure(result);drivers=set(map(tuple,drivers));parent={}
    def root(key):
        parent.setdefault(key,key)
        ancestor=key
        while parent[ancestor]!=ancestor:ancestor=parent[ancestor]
        while parent[key]!=key:
            following=parent[key];parent[key]=ancestor;key=following
        return ancestor
    active=[c for c in entries if c.get('enabled',True)];line_anchors=set();tangents=defaultdict(list)
    for c in active:
        a,b=_key(c['a']),_key(c['b']);parent[root(b)]=root(a)
        if c['kind']=='tangent':
            arc,line=_arc_line(c,records);line_anchors.add(line);tangents[arc[0]].append(c)
    groups=defaultdict(list)
    for key in parent:groups[root(key)].append(key)
    changes=defaultdict(dict)
    for members in groups.values():
        pinned=[key for key in members if records[key[0]].get('fixed',False) or key in line_anchors]
        moving=[key for key in members if key in drivers]
        anchor=(pinned or moving or members)[0];target=_point(records[anchor[0]],anchor[1])
        for key in pinned+moving:
            if not _same(_point(records[key[0]],key[1]),target):raise ValueError('Несовместимая связь: точка '+key[0]+'.'+key[1]+' закреплена или задана в другом месте')
        for identity,role in members:
            if not _same(_point(records[identity],role),target):changes[identity][role]=target/UNITS[records[identity].get('unit','mm')]
    for identity,points in changes.items():
        p=records[identity];old=deepcopy(p)
        for role,point in points.items():p[role]=point.tolist()
        p.pop('radius',None)
        if p['kind']=='arc' and identity not in tangents:
            *_,sweep=arc_circle(old);bulge=math.tan(sweep/4);a,b=np.array(p['start']),np.array(p['end']);chord=b-a
            if np.linalg.norm(chord)<=TOL/UNITS[p.get('unit','mm')]:raise ValueError(identity+': связь совмещает оба конца дуги')
            p['center']=((a+b)/2+(1-bulge*bulge)/(4*bulge)*np.array([-chord[1],chord[0]])).tolist()
    for identity,constraints in tangents.items():
        arc_ref,line_ref=_arc_line(constraints[0],records)
        fitted=tangent_arc(records[identity],arc_ref[1],records[line_ref[0]],line_ref[1])
        records[identity].clear();records[identity].update(fitted)
    _check_fixed(raw,result)
    if baseline is not None:_check_fixed(baseline,result)
    validate_constraints(result)
    return result


def add_constraint(raw,kind,a,b):
    result=deepcopy(raw);materialize_constraints(result)
    pair={_key(a),_key(b)}
    if any(c['kind']==kind and {_key(c['a']),_key(c['b'])}==pair for c in result['constraints']):raise ValueError('Такая связь уже существует')
    result['constraints'].append(dict(id=next_id({**result,'groups':result.get('groups',[])+result['constraints']},'C'),kind=kind,a=deepcopy(a),b=deepcopy(b)))
    records,entries=_structure(result)
    driver=_key(b)
    if kind=='tangent':_,driver=_arc_line(entries[-1],records)
    return apply_constraints(result,drivers=(driver,),baseline=raw)


def remove_constraints(raw,ids):
    result=deepcopy(raw);materialize_constraints(result);ids=set(ids)
    pairs=[{_key(c['a']),_key(c['b'])} for c in result['constraints'] if c['id'] in ids and c['kind']=='coincident']
    result['constraints']=[c for c in result['constraints'] if c['id'] not in ids and not(c['kind']=='tangent' and {_key(c['a']),_key(c['b'])} in pairs)]
    return result
