"""Conservative air ledger. Assigned inlet flow is never silently normalized."""
from dataclasses import dataclass
from collections import defaultdict
import math
from .air_sources import validate_air_model


@dataclass
class BalanceResult:
    total_kg_s: float
    flows: dict
    fixed_kg_s: float
    assigned_kg_s: float
    unassigned_kg_s: float
    categories: dict
    diagnostics: tuple = ()

    @property
    def rel_error(self):
        return (math.fsum(self.flows.values()) + self.unassigned_kg_s - self.total_kg_s) / max(self.total_kg_s, 1e-15)


def solve_balance(raw):
    model = validate_air_model(raw)
    if model['total_kg_s'] is None:
        raise ValueError('Суммарный расход воздуха через КС неизвестен')
    total = model['total_kg_s']
    sources = [s for s in model['sources'] if s['enabled']]
    flows = {s['id']: 0. for s in sources}
    for source in sources:
        if source['method'] == 'fixed':
            flows[source['id']] = source['flow_kg_s']
        elif source['method'] == 'fraction':
            flows[source['id']] = total * source['share']
    try:
        fixed = math.fsum(flows.values())
    except OverflowError:
        raise ValueError('Заданные расходы превышают конечный суммарный расход воздуха через КС') from None
    tolerance = max(total * 1e-12, 1e-14)
    if fixed > total + tolerance:
        raise ValueError('Заданные расходы/доли превышают суммарный расход воздуха через КС; значения не нормированы')
    variable = [s for s in sources if s['method'] in ('area', 'effective_area')]
    if variable and not model['area_hypothesis_confirmed']:
        raise ValueError('Подтвердите гипотезу распределения остатка по площади при сходных гидравлических условиях')
    remainder = max(0., total - fixed)
    if variable:
        weights = [s['area_mm2'] * (s['cd'] if s['method'] == 'effective_area' else 1.) for s in variable]
        if any(not math.isfinite(value) or value <= 0 for value in weights):
            raise ValueError('Эффективная площадь CdF должна быть конечной и положительной')
        try:
            denominator = math.fsum(weights)
        except OverflowError:
            scale = max(weights)
            weights = [value/scale for value in weights]
            denominator = math.fsum(weights)
        assigned = remainder * model['area_allocation_fraction']
        for source, weight in zip(variable, weights):
            flows[source['id']] = assigned * (weight / denominator)
    assigned = math.fsum(flows.values())
    unassigned = max(0., total - assigned)
    categories = defaultdict(float)
    for source in sources:
        categories[source['kind']] += flows[source['id']]
    categories['unassigned'] = unassigned
    diagnostics = []
    if unassigned > tolerance:
        diagnostics.append('Нераспределённый остаток остаётся в балансе и не создаёт искусственную струю.')
    if any(s['method'] == 'unknown' for s in sources):
        diagnostics.append('Расходы источников с методом unknown неизвестны; в матрицы они не включены.')
    return BalanceResult(total, flows, fixed, assigned, unassigned, dict(categories), tuple(diagnostics))


def didenko_reference():
    """Independent 1D oracle, Didenko 2012 part III, printed pp. 8–13.

    This is the book's conditional involvement bookkeeping, not a validated
    spatial jet model. Cooling typo '.5' on p10 is resolved by five × .05.
    """
    return dict(provenance='А.А. Диденко, Теория и расчёт камер сгорания ВРД, ч.III, 2012, с.8–13',
                group_shares=dict(front=.25, active=.4038, mixing=.0962, cooling=.25),
                mixed_cumulative=[.25, .29692, .35384, .45460, .53844, .63920,
                                  .72304, .82380, .89228, .97, 1.])


def didenko_mixed_cumulative():
    """Recompute the eleven book ledger entries from independent inlet shares."""
    front, active, mixing, cooling = .25, .4038 / 3., .0962, .05
    # One front source, three active jets, one mixing jet and five wall sources.
    increments = [front, active*.2 + cooling*.4, active*.2 + cooling*.6,
                  active*.6 + cooling*.4, active*.4 + cooling*.6,
                  active*.6 + cooling*.4, active*.4 + cooling*.6,
                  active*.6 + cooling*.4, mixing*.4 + cooling*.6,
                  mixing*.6 + cooling*.4, cooling*.6]
    # Rounded inlet shares in the printed table produce the reported values
    # directly; the three active jets are each 0.1346, not a new fit.
    output=[]
    for value in increments:
        output.append(math.fsum(increments[:len(output)+1]))
    return output
