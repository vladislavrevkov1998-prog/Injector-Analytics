"""Air parameters inherited from a nozzle-kind template, never from fuel flow."""
from copy import deepcopy

AIR_TEMPLATE_FIELDS = (
    'method', 'flow_kg_s', 'share', 'area_mm2', 'cd', 'direction',
    'penetration_factor', 'kernel', 'sigma_r_mm', 'sigma_theta_deg',
    'spread_mm_per_mm', 'mixed_fraction', 'mixing_length_mm',
    'entrainment_points', 'entrainment_stages', 'confidence',
)


def default_air_template():
    from .air_sources import normalize_source
    normalized = normalize_source({'id': '__template__'})
    return {key: deepcopy(normalized[key]) for key in AIR_TEMPLATE_FIELDS}


def validate_air_template(raw):
    from .air_sources import normalize_source
    if not isinstance(raw, dict):
        raise ValueError('Воздушный шаблон должен быть объектом JSON.')
    values = default_air_template()
    values.update(deepcopy(raw))
    normalized = normalize_source(dict(values, id='__template__', kind='other'))
    # Retain future metadata, just as the surrounding project does.
    return {key: deepcopy(normalized[key] if key in AIR_TEMPLATE_FIELDS else value)
            for key, value in values.items()}


def effective_air_source(source, cfg, nozzles):
    """Return effective parameters; keep raw values and override metadata intact."""
    from .air_sources import normalize_source
    raw = deepcopy(source)
    mode = raw.get('template_mode', 'manual')
    if mode not in ('manual', 'inherit'):
        raise ValueError('Режим воздушных параметров: manual или inherit.')
    overrides = raw.get('air_overrides', [])
    if not isinstance(overrides, list) or not all(isinstance(k, str) for k in overrides):
        raise ValueError('air_overrides должен быть списком имён параметров.')
    if mode == 'inherit':
        if raw.get('kind') != 'swirler' or raw.get('nozzle_id') is None:
            raise ValueError('Воздушный шаблон наследуется только связанным завихрителем.')
        lookup = {n.id: n for n in nozzles}
        nozzle = lookup.get(raw['nozzle_id'])
        if nozzle is None:
            raise ValueError(str(raw.get('id', '')) + ': отсутствует связанная форсунка')
        template = validate_air_template(cfg.nozzle_presets[nozzle.nozzle_kind].get('air', {}))
        for key in AIR_TEMPLATE_FIELDS:
            if key not in overrides:
                raw[key] = deepcopy(template[key])
    return normalize_source(raw)


def effective_air_model(raw, cfg, nozzles):
    from .air_sources import validate_air_model
    model = deepcopy(raw or {})
    model['sources'] = [effective_air_source(s, cfg, nozzles) for s in model.get('sources', [])]
    return validate_air_model(model)


def inherited_swirlers(raw, cfg, nozzles, *, force=False):
    """Add missing registry entries only after an explicit template/add action.

    A template merely declares inputs. It never enables the physical air model.
    Existing manual sources, including historical measured values, are untouched.
    """
    model = deepcopy(raw or {})
    if not force and not any('air' in p for p in cfg.nozzle_presets.values()):
        return model
    sources = model.setdefault('sources', [])
    linked = {s.get('nozzle_id') for s in sources if s.get('nozzle_id') is not None}
    identities = {s.get('id') for s in sources}
    for nozzle in nozzles:
        if nozzle.id in linked:
            continue
        identity = f'S{nozzle.id}'
        if identity in identities:
            identity = f'swirler-{nozzle.id}'
            while identity in identities:
                identity += '-air'
        sources.append(dict(id=identity, kind='swirler', nozzle_id=nozzle.id,
                            enabled=nozzle.enabled, z_mm=nozzle.z_tip_mm,
                            radius_mm=nozzle.radius_mm, theta_deg=nozzle.theta_deg,
                            method='unknown', flow_kg_s=None,
                            template_mode='inherit', air_overrides=[],
                            provenance={'origin': 'nozzle-kind air template; flow not inferred from fuel'}))
        identities.add(identity)
    return model
