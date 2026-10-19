"""Exact, unconstrained editing gestures; profile admissibility is checked later."""
from copy import deepcopy
import math
import re
import numpy as np
from .chamber import Primitive, UNITS, TOL, number

WALL_PREFIX = {'inner': 'ВН', 'outer': 'НР', 'reference': 'ОП'}


def next_id(raw, prefix):
    """Readable IDs are allocated without reindexing existing user records."""
    used = {entry['id'] for key in ('primitives', 'objects', 'groups', 'sections')
            for entry in raw.get(key, [])}
    numbers = [int(match.group(1)) for identity in used
               if (match := re.fullmatch(re.escape(prefix) + r'(\d+)', identity))]
    value = max(numbers, default=0) + 1
    while prefix + str(value) in used: value += 1
    return prefix + str(value)


def arc_circle(raw):
    tolerance=TOL/UNITS[raw.get('unit','mm')]
    a, b, center = (np.asarray(raw[key], dtype=float) for key in ('start', 'end', 'center'))
    if any(p.shape != (2,) or not np.isfinite(p).all() for p in (a, b, center)):
        raise ValueError('Для дуги нужны конечные координаты двух концов и центра')
    radius = float(np.linalg.norm(a-center))
    if radius <= tolerance or np.linalg.norm(b-a) <= tolerance:
        raise ValueError('Для дуги нужны разные концы и ненулевой радиус')
    if not np.isclose(np.linalg.norm(b-center), radius, rtol=1e-10, atol=tolerance):
        raise ValueError('Концы дуги должны лежать на одной окружности')
    angle = math.atan2(*(a-center)[::-1]); end = math.atan2(*(b-center)[::-1])
    sweep = -((angle-end) % (2*math.pi)) if raw.get('clockwise', False) else ((end-angle) % (2*math.pi))
    if abs(sweep) < 1e-12: raise ValueError('Нужна ненулевая дуга')
    return a, b, center, radius, angle, sweep


def move_endpoint(raw, role, point):
    result = deepcopy(raw)
    if raw['kind'] == 'arc':
        *_, sweep = arc_circle(raw)
        bulge = math.tan(sweep/4)
        result[role] = list(map(float, point))
        a, b = np.array(result['start']), np.array(result['end']); chord = b-a
        if np.linalg.norm(chord) <= TOL/UNITS[raw.get('unit','mm')]: raise ValueError('Концы дуги не должны совпадать')
        center = (a+b)/2 + (1-bulge*bulge)/(4*bulge)*np.array([-chord[1], chord[0]])
        result['center'] = center.tolist()
    else:
        result[role] = list(map(float, point))
    result.pop('radius', None)
    return result


def translate(raw, delta):
    result = deepcopy(raw); delta = np.asarray(delta)
    for key in ('start', 'end', 'center'):
        if key in result: result[key] = (np.array(result[key])+delta).tolist()
    return result


def resize_arc(raw, radius):
    tolerance=TOL/UNITS[raw.get('unit','mm')]
    radius = number(radius, 'Радиус дуги')
    a, b = np.array(raw['start'], dtype=float), np.array(raw['end'], dtype=float)
    chord = b-a; length = float(np.linalg.norm(chord))
    if length <= tolerance or radius < length/2 or radius <= 0:
        raise ValueError(f'Радиус дуги должен быть не меньше половины хорды ({length/2:g})')
    normal = np.array([-chord[1], chord[0]])/length; midpoint = (a+b)/2
    side = np.dot(np.array(raw['center'])-midpoint, normal)
    if abs(side)<=tolerance:side=-1 if raw.get('clockwise',False) else 1
    center = midpoint + (-1 if side < 0 else 1)*math.sqrt(max(0., radius*radius-length*length/4))*normal
    result = deepcopy(raw); result['center'] = center.tolist(); result['radius'] = radius
    return result


def preview(raw, error_mm):
    """Invalid drafts stay visible; never silently drop an invalid primitive."""
    try:
        return Primitive(raw).discretize(error_mm), ''
    except (ValueError, TypeError, KeyError) as exc:
        message = str(exc)
    factor = UNITS.get(raw.get('unit', 'mm'), 1.)
    try:
        if raw['kind'] == 'arc':
            a, b, center, radius, angle, sweep = arc_circle(raw)
            step = 2*math.acos(max(-1., 1-min(error_mm/(radius*factor), 2.)))
            count = min(10000, max(2, math.ceil(abs(sweep)/max(step, 1e-6))))
            angles = np.linspace(angle, angle+sweep, count+1)
            points = center + radius*np.column_stack((np.cos(angles), np.sin(angles)))
            points[0], points[-1] = a, b
        else:
            points = np.array([raw['start'], raw.get('end', raw['start'])], dtype=float)
    except (ValueError, TypeError, KeyError):
        points = np.array([raw[key] for key in ('start', 'center', 'end') if key in raw], dtype=float)
    return points*factor, message
