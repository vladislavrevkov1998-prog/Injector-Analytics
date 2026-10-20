"""Derive the outlet from the farthest section, preserving upstream profiles."""
from copy import deepcopy
import math
import numpy as np
from .chamber import Primitive,UNITS,TOL,number
from .chamber_editing import next_id,WALL_PREFIX
from .chamber_constraints import materialize_constraints,add_constraint


def _wall_terminal(raw,wall,old,z):
    """Locate the actual end of a continuous, manually edited wall.

    A draft can already contain the downstream geometry when its last section
    is moved.  The old outlet is then only stale bookkeeping; extending from
    it would overlap (or fail to find) the geometry the user just drew.
    """
    profile=sorted((Primitive(p) for p in raw['primitives']
                    if p['wall']==wall and p['kind']!='point'),key=lambda p:p.lo)
    if not profile or abs(profile[0].lo-number(raw['base_z_mm']))>TOL:
        raise ValueError(wall+': сначала согласуйте начало профиля с базовой плоскостью')
    for left,right in zip(profile,profile[1:]):
        if abs(left.hi-right.lo)>TOL or abs(float(left.at(left.hi)-right.at(right.lo)))>TOL:
            raise ValueError(wall+': сначала согласуйте конец профиля: разрыв/перекрытие или несопряжённые участки')
    terminal=profile[-1]
    if terminal.hi<old-TOL:
        raise ValueError(wall+': сначала согласуйте конец профиля с текущим выходом')
    if terminal.hi>z+TOL:
        raise ValueError(wall+': профиль продолжается дальше последнего сечения; переместите выходное сечение к концу профиля')
    return terminal


def sync_section_outlet(raw):
    result=deepcopy(raw)
    if not result.get('sections'):raise ValueError('Нужно хотя бы одно расчётное сечение')
    z=max(number(s['z_mm'],'z сечения') for s in result['sections']);old=number(result['outlet_z_mm'])
    if z<=number(result['base_z_mm'])+TOL:raise ValueError('Выход должен быть правее базы')
    if abs(z-old)>TOL:
        materialize_constraints(result)
        if z>old:
            for wall in ('inner','outer'):
                primitive=_wall_terminal(result,wall,old,z)
                if abs(primitive.hi-z)<=TOL:continue
                start=primitive.hi;p=next(p for p in result['primitives'] if p['id']==primitive.id)
                role='end' if primitive.end[0]>primitive.start[0] else 'start';radius=float(primitive.at(start))
                if p['kind']=='line':slope=(primitive.end[1]-primitive.start[1])/(primitive.end[0]-primitive.start[0])
                else:
                    distance=primitive.radius**2-(start-primitive.center[0])**2
                    if distance<=TOL*TOL:raise ValueError(p['id']+': вертикальную касательную нельзя продолжить как R(z); добавьте следующий участок вручную')
                    slope=-primitive.branch*(start-primitive.center[0])/math.sqrt(distance)
                identity=next_id(result,WALL_PREFIX[wall]);result['primitives'].append(dict(id=identity,wall=wall,kind='line',start=[start,radius],end=[z,radius+(z-start)*slope],unit='mm',provenance=dict(source_kind='constructed',operation='extend_outlet_tangent',from_primitive_id=p['id'],previous_outlet_mm=old)))
                result=add_constraint(result,'coincident',dict(primitive_id=identity,endpoint='start'),dict(primitive_id=p['id'],endpoint=role))
        else:
            removed=set();changed=set();kept=[]
            for p in result['primitives']:
                if p['wall']=='reference' or p['kind']=='point':kept.append(p);continue
                primitive=Primitive(p)
                if primitive.hi<=z+TOL:kept.append(p);continue
                if p.get('fixed',False):raise ValueError(p['id']+': сокращение выхода затрагивает зафиксированную кривую')
                if primitive.lo>=z-TOL:removed.add(p['id']);continue
                role='end' if primitive.end[0]>primitive.start[0] else 'start';p[role]=[z/UNITS[primitive.unit],float(primitive.at(z))/UNITS[primitive.unit]];p.pop('radius',None);changed.add((p['id'],role));kept.append(p)
            result['primitives']=kept
            result['constraints']=[c for c in result['constraints'] if not any(c[k]['primitive_id'] in removed or (c[k]['primitive_id'],c[k]['endpoint']) in changed for k in ('a','b'))]
    result['outlet_z_mm']=z
    for s in result['sections']:s['is_outlet']=abs(number(s['z_mm'],'z сечения')-z)<=TOL
    return result


def set_section_outlet(raw,z):
    result=deepcopy(raw);z=number(z,'Выход ЖТ');last=max(result['sections'],key=lambda s:number(s['z_mm']))
    if any(s is not last and number(s['z_mm'])>=z-TOL for s in result['sections']):raise ValueError('Выход должен быть правее остальных сечений; сначала измените их координаты')
    last['z_mm']=z
    return sync_section_outlet(result)
