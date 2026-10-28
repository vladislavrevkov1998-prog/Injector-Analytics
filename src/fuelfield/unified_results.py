"""Air and fuel results share one calculation mode and one spatial grid.

Air inputs are stationary in v0.8.5. Quasi snapshots associate that same air
field with the fuel at the selected real time; composite columns retain their
own scan times. This module does not depend on Qt or alter fuel calculations.
"""
from collections import OrderedDict
from copy import deepcopy
import numpy as np
from .air_distribution import compute_air_field
from .mixture_field import compute_mixture

AIR_METRICS = {
    'air_flux': ('Воздух: плотность расхода, кг/(м²·с)', 'кг/(м²·с)'),
    'air_total_kg_s': ('Воздух: полный учтённый расход ячейки, кг/с', 'кг/с'),
    'air_core_kg_s': ('Воздух: ядро струй, кг/с', 'кг/с'),
    'air_mixed_kg_s': ('Воздух: условно вовлечённый, кг/с', 'кг/с'),
    'reverse_air_kg_s': ('Воздух: обратный расход отдельно, кг/с', 'кг/с'),
    'alpha_total': ('Коэффициент избытка воздуха α_total', '1'),
    'alpha_mixed': ('Коэффициент избытка вовлечённого воздуха α_mixed', '1'),
}
AIR_EQUATIONS = {
    'air_flux': 'q_air = G_air / S  [кг/(м²·с)]',
    'air_total_kg_s': 'G_air,total = G_core + G_mixed  [кг/с]',
    'air_core_kg_s': 'G_core = (1 − E) · G_crossed  [кг/с]',
    'air_mixed_kg_s': 'G_mixed = E · G_crossed  [кг/с]',
    'reverse_air_kg_s': 'G_reverse: отдельный обратный вклад, не положительный воздух  [кг/с]',
    'alpha_total': 'α_total = G_air,total / (L0 · G_fuel,suspended); это расходное отношение',
    'alpha_mixed': 'α_mixed = G_air,mixed / (L0 · G_fuel,suspended); это расходное отношение',
}


def air_cell_values(plane, mixture, metric):
    if metric not in AIR_METRICS:
        raise ValueError('Неизвестная величина воздуха')
    if mixture is None:
        raise ValueError('Нет совместного расчёта воздуха. Включите воздух и нажмите F5.')
    if metric == 'air_flux':
        values = np.asarray(mixture.air_total_kg_s) / np.asarray(plane.cell_area_m2)
    else:
        values = getattr(mixture, metric)
    return np.ma.masked_invalid(np.array(values, dtype=float, copy=True))


class AirResultSet:
    """Own immutable calculation inputs and bounded derived time snapshots."""
    def __init__(self, cfg, nozzles, planes, air_fields, mixtures, quasi_result=None):
        self.cfg = deepcopy(cfg)
        self.nozzles = deepcopy(nozzles)
        self.planes = tuple(planes)
        self.air_fields = tuple(air_fields)
        self.mixtures = tuple(mixtures)
        self.quasi_result = quasi_result
        self._snapshots = OrderedDict()

    def state(self, index, step=None):
        if step is None or self.quasi_result is None:
            return self.planes[index], self.mixtures[index]
        if not 0 <= step < self.quasi_result.settings.n_steps:
            raise ValueError('Шаг вне выбранного квазистационарного протокола')
        key = (index, int(step))
        if key not in self._snapshots:
            plane = self.quasi_result.snapshot(index, step)
            result = compute_mixture(plane, self.cfg, self.nozzles,
                                     air_result=self.air_fields[index])
            self._snapshots[key] = plane, result
        self._snapshots.move_to_end(key)
        while len(self._snapshots) > 32:
            self._snapshots.popitem(last=False)
        return self._snapshots[key]


def build_air_results(cfg, nozzles, planes, quasi_result=None, cancel=None, progress=None):
    if not getattr(cfg, 'air_model', {}).get('enabled', False):
        return None
    from .quasi import check_cancel
    air_fields, mixtures = [], []
    for i, plane in enumerate(planes):
        check_cancel(cancel)
        if progress:
            progress(f'Воздух и α: сечение {i+1}/{len(planes)}, z={plane.z_mm:g} мм…')
        field = compute_air_field(plane, cfg, nozzles)
        mixture = compute_mixture(plane, cfg, nozzles, air_result=field)
        air_fields.append(field)
        mixtures.append(mixture)
    check_cancel(cancel)
    return AirResultSet(cfg, nozzles, planes, air_fields, mixtures, quasi_result)


def draw_air_reference(ax, cfg, nozzles, scan_angles=None):
    """Concentric nozzle locations and actual comb pass positions in +X/CCW."""
    from .coordinates import grid_angles
    phase = np.linspace(0., 2*np.pi, 361)
    for radius in sorted({float(n.radius_mm) for n in nozzles}):
        line, = ax.plot(phase, np.full_like(phase, radius), color='#8097a8',
                        lw=.65, ls='--', alpha=.75, zorder=6)
        line.set_gid('nozzle-row-ring')
    angles = grid_angles(cfg) if scan_angles is None else np.asarray(scan_angles)
    outer = ax.get_ylim()[1]
    markers = ax.scatter(np.deg2rad(angles), np.full(len(angles), outer),
                         s=7, color='#6f8b9e', clip_on=False, zorder=12)
    markers.set_gid('comb-pass-positions')


def air_summary(plane, mixture):
    if mixture is None:
        return 'Воздух не рассчитан. Примените баланс воздуха и нажмите F5.'
    def fmt(value):
        return f'{value:.7g}' if np.isfinite(value) else 'не определён'
    scope = ('Составная матрица: столбцы относятся к разным временам; '
             'единый мгновенный α сечения не определён. ' if plane.is_composite else '')
    return (scope + f'Воздух {np.sum(mixture.air_total_kg_s):.7g} кг/с; '
            f'вовлечено {np.sum(mixture.air_mixed_kg_s):.7g} кг/с; '
            f'остаток не назначен {mixture.balance.unassigned_kg_s:.7g} кг/с. '
            f'α_total сечения: {fmt(mixture.alpha_section_total)}; '
            f'α_mixed: {fmt(mixture.alpha_section_mixed)}. '
            f'Неопределённая площадь α: {100*mixture.undefined_area_fraction:.5g}%.')
