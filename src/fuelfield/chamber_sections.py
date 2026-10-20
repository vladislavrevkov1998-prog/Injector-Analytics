"""Derive the outlet from the farthest section, preserving upstream profiles."""
from copy import deepcopy
import math
import numpy as np
from .chamber import Primitive,UNITS,TOL,number
from .chamber_editing import next_id,WALL_PREFIX
from .chamber_constraints import materialize_constraints,add_constraint


def sync_section_outlet(raw):
    result=deepcopy(raw)
    if not result.get('sections'):raise ValueError('Нужно хотя бы одно расчётное сечение')
    z=max(number(s['z_mm'],'z сечения') for s in result['sections']);old=number(result['outlet_z_mm'])
    if z<=number(result['base_z_mm'])+TOL:raise ValueError('Выход должен быть правее базы')
    if abs(z-old)>TOL:
        materialize_constraints(result)
        if z>old:
            for wall in ('inner','outer'):
                terminals=[p for p in result['primitives'] if p['wall']==wall and p['kind']!='point' and abs(Primitive(p).hi-old)<=TOL]
                if len(terminals)!=1:raise ValueError(wall+': сначала согласуйте конец профиля с текущим выходом')
                p=terminals[0];primitive=Primitive(p);role='end' if primitive.end[0]>primitive.start[0] else 'start';radius=float(primitive.at(old))
                if p['kind']=='line':slope=(primitive.end[1]-primitive.start[1])/(primitive.end[0]-primitive.start[0])
                else:
                    distance=primitive.radius**2-(old-primitive.center[0])**2
                    if distance<=TOL*TOL:raise ValueError(p['id']+': вертикальную касательную нельзя продолжить как R(z); добавьте следующий участок вручную')
                    slope=-primitive.branch*(old-primitive.center[0])/math.sqrt(distance)
                identity=next_id(result,WALL_PREFIX[wall]);result['primitives'].append(dict(id=identity,wall=wall,kind='line',start=[old,radius],end=[z,radius+(z-old)*slope],unit='mm',provenance=dict(source_kind='constructed',operation='extend_outlet_tangent',from_primitive_id=p['id'],previous_outlet_mm=old)))
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
    for s in result['sections']:s['is_outlet']=abs(s['z_mm']-z)<=TOL
    return result


def set_section_outlet(raw,z):
    result=deepcopy(raw);z=number(z,'Выход ЖТ');last=max(result['sections'],key=lambda s:number(s['z_mm']))
    if any(s is not last and number(s['z_mm'])>=z-TOL for s in result['sections']):raise ValueError('Выход должен быть правее остальных сечений; сначала измените их координаты')
    last['z_mm']=z
    return sync_section_outlet(result)
