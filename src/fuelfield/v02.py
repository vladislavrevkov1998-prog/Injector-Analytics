"""Project-level v0.2 flow modes, geometry presets and row placement."""
from copy import deepcopy
from dataclasses import dataclass, replace
import math
from .dityakin import characteristic_for_type

GEOMETRY_DEFAULTS = dict(z_tip_mm=0.,alpha_deg=45., r_nozzle_mm=12., h0_mm=4., k_linear=.08,
    k_quadratic=.0004, power=.7, z_ref_mm=50., inner_angle_deg=10., outer_angle_deg=30.,
    clock_deg=0., decay_length_mm=70., diffusion_deg_sqrt_mm=1., rotation_deg_per_mm=0.)
GEOMETRY_KEYS = tuple(GEOMETRY_DEFAULTS)
NOZZLE_KINDS = {'uncooled': 'Неохлаждаемая', 'cooled': 'Охлаждаемая'}


def default_presets():
    # Demonstration defaults, editable independently; not calibrated type data.
    return {kind: dict(GEOMETRY_DEFAULTS) for kind in NOZZLE_KINDS}


def validate_catalog(catalog, presets):
    ids=set()
    for item in catalog:
        if not isinstance(item,dict):
            raise ValueError('Тип распылителя должен быть объектом с ID и параметрами.')
        if not isinstance(item.get('id'),str) or not item['id'].strip() or item['id'] in ids:
            raise ValueError('Типы распылителей: нужны уникальные непустые строковые ID.')
        ids.add(item['id'])
        if not isinstance(item.get('name'),str) or not item['name'].strip():
            raise ValueError('Укажите имя типа распылителя.')
        if item.get('kind') not in ('experimental','manual','theoretical'):
            raise ValueError('Неизвестный вид расходной характеристики.')
        characteristic_for_type(item,800.)
        if item['kind']=='theoretical':
            interval=item.get('plot_pressure_kgf_cm2',[2.5,30.])
            if (not isinstance(interval,(list,tuple)) or len(interval)!=2 or
                    not all(math.isfinite(float(p)) for p in interval) or
                    not 0<float(interval[0])<float(interval[1])):
                raise ValueError('Дитякин: нужен конечный положительный диапазон давления для графика.')
    if set(presets)!=set(NOZZLE_KINDS):
        raise ValueError('Нужны шаблоны охлаждаемой и неохлаждаемой форсунок.')
    for params in presets.values():
        if set(params)!=set(GEOMETRY_KEYS) or not all(math.isfinite(float(v)) for v in params.values()):
            raise ValueError('Шаблон: нужны все конечные параметры геометрии.')
        if params['h0_mm']<=0 or params['r_nozzle_mm']<params['h0_mm']/2 or abs(params['alpha_deg'])>=179:
            raise ValueError('Шаблон: неверные радиус, толщина или угол.')
        if params['z_ref_mm']<=0 or params['decay_length_mm']<=0 or params['diffusion_deg_sqrt_mm']<0:
            raise ValueError('Шаблон: масштабы должны быть положительны, диффузия неотрицательна.')


def effective_nozzle(n,cfg):
    """Resolve inherited geometry without modifying a nozzle or its measurements."""
    if n.nozzle_kind not in cfg.nozzle_presets:
        raise ValueError(f'Форсунка {n.id}: неизвестный тип форсунки.')
    if set(n.geometry_overrides)==set(GEOMETRY_KEYS):
        return n
    params=cfg.nozzle_presets[n.nozzle_kind]
    values={key:getattr(n,key) if key in n.geometry_overrides else params[key] for key in GEOMETRY_KEYS}
    kwargs={k:v for k,v in values.items() if k!='r_nozzle_mm'}
    kwargs['rc0_mm']=values['r_nozzle_mm']-values['h0_mm']/2
    # Mark resolved values so a second call does not resolve them again.
    kwargs['geometry_overrides']=list(GEOMETRY_KEYS)
    return replace(n,**kwargs)


@dataclass(frozen=True)
class FlowStatus:
    kg_s: float
    pressure_kgf_cm2: float
    source: str
    estimated: bool
    notices: tuple[str,...]=()


def curve_for(n,cfg):
    item=next((a for a in cfg.atomizer_types if a['id']==n.atomizer_type_id),None)
    if item is None:
        raise ValueError(f'Форсунка {n.id}: назначьте тип с расходной характеристикой.')
    return characteristic_for_type(item,cfg.density_kg_m3)


def flow_status(n,cfg):
    if n.flow_mode not in ('manual','sectors','curve'):
        raise ValueError(f'Форсунка {n.id}: неизвестный источник расхода.')
    notices=[]
    if n.flow_mode=='curve':
        curve=curve_for(n,cfg)
        estimate=curve.evaluate(cfg.target_pressure_kgf_cm2,allow_extrapolation=cfg.allow_extrapolation)
        if estimate.extrapolated:notices.append('Экстраполяция характеристики')
        source='Дитякин (расчёт)' if hasattr(curve,'hydraulics') else 'экспериментальная характеристика'
        return FlowStatus(estimate.flow_kg_h/3600,cfg.target_pressure_kgf_cm2,source,True,tuple(notices))
    reference=bench_pressure(n,cfg,'flow' if n.flow_mode=='manual' else 'sector')
    if not math.isfinite(reference) or reference<=0:
        raise ValueError(f'Форсунка {n.id}: перепад измерения должен быть положительным.')
    if n.flow_mode=='manual':
        base=n.flow_kg_h/3600
        source='измеренный расход'
    else:
        parts=n.sector_ml
        if len(parts)!=36 or any(not math.isfinite(v) or v<0 for v in parts):
            raise ValueError(f'Форсунка {n.id}: нужны 36 конечных неотрицательных объёмов.')
        eta=n.capture_efficiency
        if cfg.pressure_mode=='legacy':eta=1.
        elif eta is None:
            eta=1.
            try:capture_report(n,cfg)
            except ValueError:notices.append('Полнота сбора неизвестна: принято η=1')
            else:notices.append('Полнота сбора сопоставлена с Gт; расход без компенсации недобора (η=1)')
        else:notices.append(f'Сохранённая поправка η={eta:g}')
        if not math.isfinite(eta) or not 0<eta<=1:
            raise ValueError(f'Форсунка {n.id}: полнота сбора должна быть в (0,1].')
        base=sum(parts)*1e-6*cfg.density_kg_m3/cfg.duration_s/eta
        source='оценка по секторам'
    if not math.isfinite(base) or base<0:
        raise ValueError(f'Форсунка {n.id}: расход должен быть конечным и неотрицательным.')
    target=cfg.target_pressure_kgf_cm2
    estimated=n.flow_mode!='manual'
    if cfg.pressure_mode=='calibrated' and target!=reference and base>0:
        curve=curve_for(n,cfg)
        start=curve.evaluate(reference,allow_extrapolation=cfg.allow_extrapolation)
        end=curve.evaluate(target,allow_extrapolation=cfg.allow_extrapolation)
        if start.flow_kg_h<=0:raise ValueError(f'Форсунка {n.id}: Q при опорном давлении должна быть >0.')
        base*=end.flow_kg_h/start.flow_kg_h
        estimated=True;source+=(' → пересчёт по Дитякину' if hasattr(curve,'hydraulics')
                                else ' → пересчёт по характеристике')
        if start.extrapolated or end.extrapolated:notices.append('Экстраполяция характеристики')
    else:target=reference
    return FlowStatus(base,target,source,estimated,tuple(notices))


def validate_nozzles(nozzles,cfg):
    ids={a['id'] for a in cfg.atomizer_types}
    for n in nozzles:
        if n.atomizer_type_id and n.atomizer_type_id not in ids:
            raise ValueError(f'Форсунка {n.id}: ссылка на отсутствующий тип {n.atomizer_type_id}.')
        if n.nozzle_kind not in NOZZLE_KINDS or n.row not in ('inner','outer'):
            raise ValueError(f'Форсунка {n.id}: неизвестный тип или ряд.')
        if n.row!=('outer' if n.id<=70 else 'inner'):
            raise ValueError('ID 1–70 принадлежат наружному ряду, 71–140 — внутреннему.')
        if not set(n.geometry_overrides)<=set(GEOMETRY_KEYS):
            raise ValueError(f'Форсунка {n.id}: неизвестный override.')
        effective=effective_nozzle(n,cfg)
        if not all(math.isfinite(getattr(effective,key)) for key in GEOMETRY_KEYS):
            raise ValueError(f'Форсунка {n.id}: параметры геометрии должны быть конечными.')
        if not math.isfinite(n.theta_deg) or not math.isfinite(n.radius_mm) or n.radius_mm<0:
            raise ValueError(f'Форсунка {n.id}: неверные координаты.')
        if effective.h0_mm<=0 or effective.rc0_mm<0:
            raise ValueError(f'Форсунка {n.id}: нужны h0>0 и R_сопла≥h0/2.')
        for value in (n.flow_pressure_kgf_cm2,n.sector_pressure_kgf_cm2):
            if not math.isfinite(value) or value<=0:raise ValueError('Давления измерений должны быть положительными.')
        if n.capture_efficiency is not None and (not math.isfinite(n.capture_efficiency) or not 0<n.capture_efficiency<=1):
            raise ValueError('Полнота сбора должна быть в (0,1] или неизвестна.')
        if not math.isfinite(n.flow_kg_h) or n.flow_kg_h<0:raise ValueError('Неверный измеренный расход.')
        if n.flow_mode not in ('manual','sectors','curve'):raise ValueError('Неизвестный источник расхода.')
        if len(n.sector_ml)!=36 or any(not math.isfinite(v) or v<0 for v in n.sector_ml):
            raise ValueError('Нужны 36 конечных неотрицательных объёмов.')


def follow_row(nozzles,nozzle_id,new_theta):
    """Align both full clockwise rows to an anchor, preserving the half-pitch stagger."""
    new_theta=float(new_theta)
    if not math.isfinite(new_theta):raise ValueError('Угол должен быть конечным.')
    source=next(n for n in nozzles if n.id==nozzle_id)
    index=source.id-1 if source.row=='outer' else source.id-71+.5
    phase=new_theta+index*360/70
    for n in nozzles:
        index=n.id-1 if n.row=='outer' else n.id-71+.5
        n.theta_deg=(phase-index*360/70)%360


def bench_pressure(n,cfg,kind):
    key='flow_pressure_kgf_cm2' if kind=='flow' else 'sector_pressure_kgf_cm2'
    common=getattr(cfg,key)
    return getattr(n,key) if common is None else common


@dataclass(frozen=True)
class CaptureReport:
    collected_ml: float
    expected_ml: float
    deviation_percent: float
    missing_ml: float
    fraction: float
    collected_at_measurement_kg_h: float


def capture_report(n,cfg):
    """Independent comparison, never a circular replacement of sector flow by Gт."""
    if n.flow_kg_h<=0:raise ValueError('Для полноты сбора нужен положительный измеренный Gт.')
    measured_pressure=bench_pressure(n,cfg,'flow');sector_pressure=bench_pressure(n,cfg,'sector')
    ratio=1.
    if measured_pressure!=sector_pressure:
        curve=curve_for(n,cfg)
        qm=curve.evaluate(measured_pressure,allow_extrapolation=cfg.allow_extrapolation).flow_kg_h
        qs=curve.evaluate(sector_pressure,allow_extrapolation=cfg.allow_extrapolation).flow_kg_h
        if qm<=0 or qs<=0:raise ValueError('Для полноты сбора нужна положительная Q при обоих давлениях испытаний.')
        ratio=qm/qs
    collected=sum(n.sector_ml)
    collected_at_measurement=collected*1e-6*cfg.density_kg_m3/cfg.duration_s*3600*ratio
    expected=n.flow_kg_h/ratio/3600/cfg.density_kg_m3*cfg.duration_s*1e6
    fraction=collected/expected
    return CaptureReport(collected,expected,100*(1-fraction),expected-collected,
                         fraction,collected_at_measurement)


def remove_atomizer(catalog,nozzles,type_id):
    if any(n.atomizer_type_id==type_id for n in nozzles):
        raise ValueError('Тип используется форсунками. Сначала переназначьте ссылки.')
    return deepcopy([a for a in catalog if a['id']!=type_id])
