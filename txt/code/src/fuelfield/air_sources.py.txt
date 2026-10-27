"""Versioned air-source input, without Qt or inferred measurements.

Lengths are mm, angles degrees and every flow is kg/s. Unknown JSON members
survive normalization. Numerical kernel/trajectory parameters are empirical
model choices, never pressure, velocity, density or physical momentum.
"""
from copy import deepcopy
import numpy as np

METHODS = ('fixed', 'fraction', 'area', 'effective_area', 'unknown')


def default_air_model():
    return dict(schema_version=1, enabled=False, total_kg_s=None, l0=None,
                area_allocation_fraction=1., area_hypothesis_confirmed=False,
                distribution='directed', interaction_strength=0.,
                interaction_length_mm=30., step_mm=1., max_path_mm=1000., sources=[])


def finite(value, label, minimum=None, maximum=None):
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(label + ': требуется конечное число')
    try:
        value = float(value)
    except (TypeError, ValueError):
        raise ValueError(label + ': требуется конечное число') from None
    if not np.isfinite(value) or minimum is not None and value < minimum or maximum is not None and value > maximum:
        raise ValueError(label + ': число вне допустимого диапазона')
    return value


def normalize_source(raw):
    if not isinstance(raw, dict):
        raise ValueError('Воздушный источник должен быть объектом JSON')
    source = dict(id='', kind='other', enabled=True, group='', method='unknown',
                  flow_kg_s=None, share=0., area_mm2=None, cd=None, nozzle_id=None,
                  object_id=None, z_mm=0., radius_mm=250., theta_deg=0., wall='free',
                  direction=[1., 0., 0.], penetration_factor=1., kernel='directed',
                  sigma_r_mm=10., sigma_theta_deg=10., spread_mm_per_mm=0.,
                  mixed_fraction=0., mixing_length_mm=None, entrainment_points=[],
                  entrainment_stages=[],
                  provenance={}, confidence='unknown')
    source.update(deepcopy(raw))
    identity = source['id']
    if not isinstance(identity, str) or not identity.strip():
        raise ValueError('Нужен устойчивый непустой ID воздушного источника')
    if not isinstance(source['kind'], str) or not source['kind'].strip():
        raise ValueError(identity + ': нужен тип воздушного источника')
    if type(source['enabled']) is not bool:
        raise ValueError(identity + ': enabled должен быть логическим')
    if source['method'] not in METHODS:
        raise ValueError(identity + ': неизвестный метод расхода')
    if source['wall'] not in ('free', 'inner', 'outer') or source['kernel'] not in ('directed', 'uniform', 'wall'):
        raise ValueError(identity + ': неверная стенка или пространственное ядро')
    if source['nozzle_id'] is not None and (type(source['nozzle_id']) is not int or not 1 <= source['nozzle_id'] <= 140):
        raise ValueError(identity + ': ID форсунки должен быть 1…140')
    if source['kind'] == 'swirler' and source['nozzle_id'] is None:
        raise ValueError(identity + ': завихритель требует nozzle_id')
    if source['object_id'] is not None and (not isinstance(source['object_id'], str) or not source['object_id']):
        raise ValueError(identity + ': object_id должен быть непустой строкой')
    for name in ('flow_kg_s', 'area_mm2', 'cd', 'mixing_length_mm'):
        if source[name] is not None:
            source[name] = finite(source[name], identity + ' ' + name, 0.)
    for name in ('z_mm', 'radius_mm', 'penetration_factor', 'sigma_r_mm', 'sigma_theta_deg', 'spread_mm_per_mm'):
        source[name] = finite(source[name], identity + ' ' + name, 0.)
    for name in ('share', 'mixed_fraction'):
        source[name] = finite(source[name], identity + ' ' + name, 0., 1.)
    source['theta_deg'] = finite(source['theta_deg'], identity + ' theta_deg') % 360
    if source['radius_mm'] <= 0:
        raise ValueError(identity + ': радиус должен быть >0')
    if source['mixing_length_mm'] == 0:
        raise ValueError(identity + ': длина вовлечения должна быть >0 или null')
    direction = source['direction']
    if not isinstance(direction, (list, tuple)) or len(direction) != 3:
        raise ValueError(identity + ': direction = [d_z,d_r,d_theta]')
    vector = np.asarray([finite(v, identity + ' direction') for v in direction])
    scale = float(np.max(np.abs(vector)))
    if scale == 0:
        raise ValueError(identity + ': нулевой направляющий вектор')
    scaled = vector/scale
    source['direction'] = (scaled / np.linalg.norm(scaled)).tolist()
    if source['enabled']:
        if source['method'] == 'fixed' and source['flow_kg_s'] is None:
            raise ValueError(identity + ': фиксированный расход не задан')
        if source['method'] in ('area', 'effective_area') and (source['area_mm2'] is None or source['area_mm2'] <= 0):
            raise ValueError(identity + ': метод площади требует F >0')
        if source['method'] == 'effective_area' and (source['cd'] is None or source['cd'] <= 0):
            raise ValueError(identity + ': метод CdF требует явно заданный Cd >0')
    points = source['entrainment_points']
    if not isinstance(points, list):
        raise ValueError(identity + ': точки вовлечения должны быть списком [z,E]')
    parsed = []
    for point in points:
        if not isinstance(point, (list, tuple)) or len(point) != 2:
            raise ValueError(identity + ': точка вовлечения — [z_mm,E]')
        parsed.append([finite(point[0], 'z вовлечения', 0.), finite(point[1], 'Доля вовлечения', 0., 1.)])
    if any(b[0] <= a[0] or b[1] < a[1] for a, b in zip(parsed, parsed[1:])):
        raise ValueError(identity + ': координаты вовлечения возрастают, E не убывает')
    source['entrainment_points'] = parsed
    stages = source['entrainment_stages']
    if not isinstance(stages, list):
        raise ValueError(identity + ': порции вовлечения должны быть списком')
    parsed_stages = []
    for index, stage in enumerate(stages):
        if not isinstance(stage, dict):
            raise ValueError(identity + ': порция вовлечения должна быть объектом JSON')
        record = deepcopy(stage)
        record.setdefault('id', identity + ':portion:' + str(index+1))
        record['z_start_mm'] = finite(record.get('z_start_mm'), 'Начало порции', 0.)
        record['z_end_mm'] = finite(record.get('z_end_mm'), 'Конец порции', record['z_start_mm'])
        record['fraction'] = finite(record.get('fraction'), 'Доля порции', 0., 1.)
        if not isinstance(record['id'], str) or not record['id']:
            raise ValueError(identity + ': нужен ID порции')
        parsed_stages.append(record)
    if parsed_stages:
        if parsed:
            raise ValueError(identity + ': выберите либо точки E(z), либо отдельные порции')
        if abs(sum(p['fraction'] for p in parsed_stages)-1.) > 1e-10:
            raise ValueError(identity + ': сумма долей порций должна быть 1')
        if len({p['id'] for p in parsed_stages}) != len(parsed_stages):
            raise ValueError(identity + ': повторный ID порции вовлечения')
        if any(b['z_start_mm'] < a['z_start_mm'] for a, b in zip(parsed_stages, parsed_stages[1:])):
            raise ValueError(identity + ': порции упорядочиваются по началу участка')
    source['entrainment_stages'] = parsed_stages
    if not isinstance(source['provenance'], dict):
        raise ValueError(identity + ': provenance должен быть словарём')
    return source


def validate_air_model(raw):
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise ValueError('air_model должен быть объектом JSON')
    model = default_air_model()
    model.update(deepcopy(raw))
    if type(model['schema_version']) is not int or model['schema_version'] != 1:
        raise ValueError('Неподдерживаемая схема air_model')
    if type(model['enabled']) is not bool or type(model['area_hypothesis_confirmed']) is not bool:
        raise ValueError('Признаки воздушной модели должны быть логическими')
    if model['total_kg_s'] is not None:
        model['total_kg_s'] = finite(model['total_kg_s'], 'Расход воздуха через КС', 0.)
    elif model['enabled']:
        raise ValueError('Задайте суммарный расход воздуха через КС [кг/с]')
    if model['l0'] is not None:
        model['l0'] = finite(model['l0'], 'L0', 0.)
    model['area_allocation_fraction'] = finite(model['area_allocation_fraction'], 'Доля распределяемого остатка', 0., 1.)
    for name in ('interaction_strength',):
        model[name] = finite(model[name], name, 0., 10.)
    for name in ('interaction_length_mm', 'step_mm', 'max_path_mm'):
        model[name] = finite(model[name], name, 1e-6)
    if model['max_path_mm'] / model['step_mm'] > 100000:
        raise ValueError('Воздушные траектории требуют более 100000 шагов')
    if model['distribution'] not in ('directed', 'uniform'):
        raise ValueError('Распределение воздуха: directed или uniform')
    if not isinstance(model['sources'], list):
        raise ValueError('Реестр воздуха должен быть списком')
    model['sources'] = [normalize_source(source) for source in model['sources']]
    ids = [s['id'] for s in model['sources']]
    if len(ids) != len(set(ids)):
        raise ValueError('Повторный ID воздушного источника')
    for key, label in (('nozzle_id', 'завихрителя/форсунки'), ('object_id', 'геометрического объекта')):
        values = [s[key] for s in model['sources'] if s[key] is not None]
        if len(values) != len(set(values)):
            raise ValueError('Повторное списание воздушного источника ' + label)
    return model


def resolve_sources(raw, cfg, nozzles):
    """Resolve coordinates, never duplicate a geometry nozzle as a fuel source."""
    from .chamber import chamber
    from .v02 import effective_nozzle
    model = validate_air_model(raw)
    geometry = chamber(cfg)
    lookup = {n.id: n for n in nozzles}
    objects = {obj['id']: obj for obj in geometry.objects} if geometry else {}
    result = []
    physical = set()
    for original in model['sources']:
        source = deepcopy(original)
        if not source['enabled']:
            continue
        if source['nozzle_id'] is not None:
            if source['nozzle_id'] not in lookup:
                raise ValueError(source['id'] + ': отсутствует связанная форсунка')
            nozzle = effective_nozzle(lookup[source['nozzle_id']], cfg)
            if not nozzle.enabled:
                raise ValueError(source['id'] + ': связанная форсунка не установлена; отключите воздушный источник явно')
            source.update(z_mm=float(nozzle.z_tip_mm), radius_mm=float(nozzle.radius_mm), theta_deg=float(nozzle.theta_deg))
        object_id = source['object_id']
        if object_id is not None:
            if object_id not in objects:
                raise ValueError(source['id'] + ': отсутствует геометрический объект ' + object_id)
            obj = objects[object_id]
            if obj['type'] == 'nozzle':
                if source['kind'] != 'swirler' or source['nozzle_id'] != obj['nozzle_id']:
                    raise ValueError(source['id'] + ': объект nozzle связывается только с его завихрителем')
                key = ('nozzle', obj['nozzle_id'])
            else:
                key = ('object', object_id)
            z, radius, theta = geometry.object_position(obj, nozzles, cfg)
            source.update(z_mm=float(z), radius_mm=float(radius), theta_deg=float(theta), wall=obj.get('wall', 'free'))
        else:
            key = ('nozzle', source['nozzle_id']) if source['nozzle_id'] is not None else ('registry', source['id'])
            if source['wall'] != 'free':
                index = 0 if source['wall'] == 'inner' else 1
                source['radius_mm'] = float(cfg.bounds_at(source['z_mm'])[index])
        if key in physical:
            raise ValueError('Двойное списание физического воздушного источника: ' + source['id'])
        physical.add(key)
        rin, rout = cfg.bounds_at(source['z_mm'])
        if source['kernel'] == 'wall' and source['wall'] == 'free':
            source['wall'] = 'inner' if source['radius_mm'] <= (rin+rout)/2 else 'outer'
            source['radius_mm'] = rin if source['wall'] == 'inner' else rout
        if source['wall'] != 'free':
            wall = source['wall']
            normal_zr = geometry.primitive_at(wall, source['z_mm']).normal(source['z_mm']) if geometry else np.array([0., 1. if wall == 'inner' else -1.])
            normal = np.array([normal_zr[0], normal_zr[1], 0.])
            direction = np.asarray(source['direction'])
            dot = float(direction@normal)
            if source['kernel'] == 'wall':
                tangent = direction-dot*normal
                norm = float(np.linalg.norm(tangent))
                if norm <= 1e-12:
                    raise ValueError(source['id'] + ': пристеночная пелена требует касательного направления, не чистой нормали')
                source['direction'] = (tangent/norm).tolist()
            elif dot < -1e-10:
                raise ValueError(source['id'] + ': направление отверстия обращено из тракта; проверьте знак относительно нормали стенки')
            source['normal_local'] = normal.tolist()
        if source['radius_mm'] < rin - 1e-7 or source['radius_mm'] > rout + 1e-7:
            raise ValueError(source['id'] + ': воздушный источник вне локального тракта')
        source['physical_key'] = list(key)
        if geometry:
            for stage in source['entrainment_stages']:
                if stage['z_start_mm'] < geometry.base or stage['z_end_mm'] > geometry.outlet:
                    raise ValueError(source['id'] + ': порция вовлечения вне профиля ЖТ')
        result.append(source)
    return tuple(sorted(result, key=lambda s: s['id']))
