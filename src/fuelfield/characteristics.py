"""Experimental flow-curve primitives for v0.2 preparation.

Pressure: kgf/cm². Canonical flow: kg/h. No theoretical atomizer model and no
implicit activation in the v0.1.x physical kernels or project format.
"""
from __future__ import annotations
from bisect import bisect_left
from dataclasses import dataclass
import math
import warnings


class ExtrapolationWarning(UserWarning):
    """The estimate is outside the measured pressure interval."""


class PressureOutOfRange(ValueError):
    """Extrapolation must be explicitly requested by the caller."""


def _finite(value, label):
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f'{label}: требуется конечное число.')
    return value


def _density(value):
    value = _finite(value, 'Плотность')
    if value <= 0:
        raise ValueError('Плотность должна быть положительной [кг/м³].')
    return value


def flow_to_kg_h(value, unit, density_kg_m3):
    """Convert UI units to the canonical mass flow without modifying measurements."""
    value = _finite(value, 'Расход')
    density = _density(density_kg_m3)
    if value < 0:
        raise ValueError('Расход должен быть неотрицательным.')
    if unit == 'kg/h':
        return value
    if unit == 'l/min':
        return _finite(value * density * 0.06, 'Массовый расход')
    raise ValueError(f'Неизвестная единица расхода: {unit}')


def flow_from_kg_h(value, unit, density_kg_m3):
    """Convert a canonical flow for display; 1 l/min = 0.06*rho kg/h."""
    value = _finite(value, 'Расход')
    density = _density(density_kg_m3)
    if value < 0:
        raise ValueError('Расход должен быть неотрицательным.')
    if unit == 'kg/h':
        return value
    if unit == 'l/min':
        return _finite(value / density / 0.06, 'Объёмный расход')
    raise ValueError(f'Неизвестная единица расхода: {unit}')


@dataclass(frozen=True)
class FlowEstimate:
    pressure_kgf_cm2: float
    flow_kg_h: float
    extrapolated: bool


@dataclass(frozen=True)
class ExperimentalCharacteristic:
    """Immutable, sorted measured pairs; piecewise-linear interpolation only.

    One point is valid as a measurement, but cannot establish a slope.
    Non-monotonic measured flows are allowed and never smoothed or replaced.
    """
    points: tuple[tuple[float, float], ...]

    def __post_init__(self):
        measured = []
        for pressure, flow in self.points:
            pressure = _finite(pressure, 'Перепад давления')
            flow = _finite(flow, 'Расход')
            if pressure <= 0 or flow < 0:
                raise ValueError('Нужны Δp > 0 [кгс/см²] и Q ≥ 0 [кг/ч].')
            measured.append((pressure, flow))
        if not measured:
            raise ValueError('Нужна хотя бы одна измеренная точка.')
        measured.sort(key=lambda pair: pair[0])
        if any(a[0] == b[0] for a, b in zip(measured, measured[1:])):
            raise ValueError('Перепады давления должны быть уникальными.')
        object.__setattr__(self, 'points', tuple(measured))

    def evaluate(self, pressure_kgf_cm2, *, allow_extrapolation=False):
        pressure = _finite(pressure_kgf_cm2, 'Перепад давления')
        if pressure <= 0:
            raise ValueError('Перепад давления должен быть положительным.')
        pressures = [p for p, _ in self.points]
        index = bisect_left(pressures, pressure)
        if index < len(self.points) and pressures[index] == pressure:
            return FlowEstimate(pressure, self.points[index][1], False)
        outside = index == 0 or index == len(self.points)
        if outside and not allow_extrapolation:
            raise PressureOutOfRange('Давление вне измеренного диапазона; экстраполяция не разрешена.')
        if len(self.points) < 2:
            raise ValueError('Для интерполяции/экстраполяции нужны хотя бы две точки.')
        right = min(max(index, 1), len(self.points)-1)
        p0, q0 = self.points[right-1]
        p1, q1 = self.points[right]
        flow = _finite(q0 + (q1-q0)*((pressure-p0)/(p1-p0)), 'Оценка расхода')
        if flow < 0:
            raise ValueError('Линейная экстраполяция дала отрицательный расход; оценка отклонена.')
        if outside:
            warnings.warn('Расход экстраполирован вне измеренного диапазона давления.',
                          ExtrapolationWarning, stacklevel=2)
        return FlowEstimate(pressure, flow, outside)

    def to_dict(self):
        """Standalone curve data; not yet a new project-schema declaration."""
        return {'kind': 'experimental', 'pressure_unit': 'kgf/cm2',
                'flow_unit': 'kg/h', 'points': [list(pair) for pair in self.points]}

    @classmethod
    def from_dict(cls, data):
        if (data.get('kind') != 'experimental' or
                data.get('pressure_unit') != 'kgf/cm2' or data.get('flow_unit') != 'kg/h'):
            raise ValueError('Ожидается экспериментальная характеристика в кгс/см² и кг/ч.')
        return cls(data['points'])
