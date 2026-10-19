"""Dityakin et al. (1977), ch. 2: a simple, short pressure-swirl atomizer.

The source equations, graph readings, units and supported geometry are recorded
in docs/DITYAKIN_METHOD.md. Chamber losses integrate eq. (41), with the same
negligible radial-velocity approximation as eqs. (39), (40), (44).
"""
from dataclasses import dataclass, asdict
from functools import lru_cache
import math
import numpy as np
from .characteristics import FlowEstimate, _finite, _density, PressureOutOfRange

KGF_CM2_TO_PA = 98066.5
# Readings of the curves, not the scattered experimental markers. The abscissa
# of fig. 21 is log10(Re) for ONE inlet; fig. 24 uses inlet radius / swirl arm.
INLET_LOSS_CURVE = ((2.125,2.65),(2.25,2.12),(2.5,1.60),(2.75,1.26),
                    (3.,1.10),(3.25,.97),(3.5,.88),(4.,.76),(4.5,.68),(4.7,.66))
DEFORMATION_CURVE = ((0.,1.),(.05,.90),(.10,.844),(.15,.809),(.20,.790),
                     (.30,.766),(.40,.755),(.50,.750),(.70,.750))
_NODES, _WEIGHTS = np.polynomial.legendre.leggauss(64)


def graph_value(points, x):
    if not points[0][0]-1e-12 <= x <= points[-1][0]+1e-12:
        raise PressureOutOfRange('Дитякин: значение вне диапазона исходного графика; расчёт отклонён.')
    return float(np.interp(x, *zip(*points)))


def chamber_friction(re_equivalent):
    """p. 51: Re uses the diameter of the TOTAL inlet area (d*sqrt(n))."""
    re_equivalent = _finite(re_equivalent, 'Число Рейнольдса')
    if re_equivalent <= 0:
        raise ValueError('Число Рейнольдса должно быть положительным.')
    return (24.6/re_equivalent**.75 if re_equivalent <= 2300
            else 1.22/re_equivalent**.36)


def filling_coefficient(a_equivalent):
    """Unique root of A = sqrt(2)*(1-phi)/phi**1.5; eq. (9), p. 47."""
    a_equivalent = _finite(a_equivalent, 'Aₑ')
    if a_equivalent <= 0:
        raise ValueError('Aₑ должна быть положительной.')
    low, high = 0., 1.
    for _ in range(60):
        phi = (low+high)/2
        if math.sqrt(2)*(1-phi) > a_equivalent*phi**1.5:
            low = phi
        else:
            high = phi
    return (low+high)/2


def chamber_loss(a_active, friction, chamber_ratio):
    """Dimensionless integral of p. 45 eq. (41), normalized as eq. (43).

    x=r/r_c; V_theta=Q*A_d/(pi*r_c*r)/(1+lambda*A_d*(C_k-x)/2).
    The quadrature is independent of Q after normalization.
    """
    x = 1 + (_NODES+1)*(chamber_ratio-1)/2
    denominator = 1 + friction*a_active*(chamber_ratio-x)/2
    integral = (chamber_ratio-1)/2*float(np.dot(_WEIGHTS, 1/(x*x*denominator**3)))
    return friction*a_active**3*integral


@dataclass(frozen=True)
class DityakinParameters:
    outlet_diameter_mm: float = 1.
    inlet_diameter_mm: float = .4
    inlet_count: int = 3
    swirl_arm_mm: float = 1.5
    chamber_height_mm: float = .4
    inlet_length_mm: float = .8
    outlet_length_mm: float = .6
    cone_angle_deg: float = 90.
    kinematic_viscosity_mm2_s: float = 1.7

    def __post_init__(self):
        for key, value in asdict(self).items():
            number = _finite(value, key)
            if number <= 0:
                raise ValueError('Дитякин: все геометрические размеры и вязкость должны быть > 0.')
            object.__setattr__(self,key,number)
            if key.endswith('_mm') and not 1e-6 <= number <= 1e5:
                raise ValueError('Дитякин: размер вне численного диапазона 10⁻⁶…10⁵ мм.')
        if self.inlet_count != int(self.inlet_count) or not 2 <= self.inlet_count <= 6:
            raise ValueError('Дитякин: нужно целое число круглых входов от 2 до 6.')
        object.__setattr__(self,'inlet_count',int(self.inlet_count))
        d, di, arm = self.outlet_diameter_mm, self.inlet_diameter_mm, self.swirl_arm_mm
        if not 1.25 <= 2*arm/d <= 5:
            raise ValueError('Дитякин: поддерживается 1.25 ≤ R/rс ≤ 5.')
        if not 1/.7 <= 2*arm/di <= 16:
            raise ValueError('Дитякин: плечо/радиус входа вне диапазона графика деформации или формулы (39).')
        if not di <= self.chamber_height_mm <= 3*di:
            raise ValueError('Дитякин: нужна короткая камера, dвх ≤ Lк ≤ 3 dвх.')
        if not 1.5 <= self.inlet_length_mm/di <= 4:
            raise ValueError('Дитякин: нужна длина входа от 1.5 до 4 его диаметров.')
        if not .25 <= self.outlet_length_mm/d <= 1:
            raise ValueError('Дитякин: нужна длина сопла от 0.25 до 1 его диаметра.')
        if not 90 <= self.cone_angle_deg <= 120:
            raise ValueError('Дитякин: коэффициент сопла подтверждён для угла конуса 90…120°.')
        if not 1.5 <= self.active_characteristic <= 9.5:
            raise ValueError('Дитякин: Aд вне диапазона 1.5…9.5, использованного для коэффициента трения.')

    @property
    def deformation(self):
        return graph_value(DEFORMATION_CURVE, self.inlet_diameter_mm/(2*self.swirl_arm_mm))

    @property
    def active_characteristic(self):
        return (2*self.swirl_arm_mm/self.outlet_diameter_mm /
                (self.deformation*self.inlet_count*(self.inlet_diameter_mm/self.outlet_diameter_mm)**2))

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, data):
        # Persist all inputs: a future default change must not change a saved curve.
        if not isinstance(data,dict) or set(data) != set(cls.__dataclass_fields__):
            raise ValueError('Дитякин: нужны все параметры геометрии и кинематическая вязкость.')
        return cls(**data)


@dataclass(frozen=True)
class HydraulicState:
    mass_kg_s: float
    pressure_pa: float
    re_inlet: float
    re_equivalent: float
    friction: float
    a_active: float
    a_equivalent: float
    phi: float
    loss_inlet: float
    loss_chamber: float
    loss_outlet: float
    mu: float
    film_mm: float


@dataclass(frozen=True)
class DityakinCharacteristic:
    parameters: DityakinParameters
    density_kg_m3: float

    def __post_init__(self):
        object.__setattr__(self,'density_kg_m3',_density(self.density_kg_m3))

    def state_for_mass(self, mass_kg_s):
        p = self.parameters
        mass = _finite(mass_kg_s, 'Массовый расход')
        if mass <= 0:
            raise ValueError('Дитякин: расход должен быть положительным.')
        rc, ri = p.outlet_diameter_mm/2000, p.inlet_diameter_mm/2000
        nu = p.kinematic_viscosity_mm2_s*1e-6
        volume = mass/self.density_kg_m3
        velocity = volume/(p.inlet_count*math.pi*ri**2)
        re_inlet = velocity*2*ri/nu
        re_equivalent = re_inlet*math.sqrt(p.inlet_count)
        if re_equivalent > 1e5*(1+1e-12):
            raise PressureOutOfRange('Дитякин: Re суммарного входа > 10⁵; вне диапазона рис.28.')
        friction = chamber_friction(re_equivalent)
        # p. 45 gives the accuracy conditions of the reduced moment equation.
        if friction > .2*(1+1e-12):
            raise PressureOutOfRange('Дитякин: λк > 0.2; вне области приближения короткой камеры.')
        a = p.active_characteristic
        ck = (p.swirl_arm_mm+p.inlet_diameter_mm/2)/(p.outlet_diameter_mm/2)
        ae = a/(1+friction*a*(ck-1)/2)
        phi = filling_coefficient(ae)
        xi_inlet = graph_value(INLET_LOSS_CURVE,math.log10(re_inlet))
        # Eq. (29) contains GEOMETRIC A, not A_d or A_e.
        loss_inlet = xi_inlet*(rc*rc/(p.inlet_count*ri*ri))**2
        loss_chamber = chamber_loss(a,friction,ck)
        xi_outlet = .11 + (p.cone_angle_deg-90)*(.16-.11)/30
        loss_outlet = xi_outlet/phi**2
        mu = 1/math.sqrt(ae*ae/(1-phi)+1/phi**2+loss_inlet+loss_chamber+loss_outlet)
        pressure_pa = .5*self.density_kg_m3*(volume/(math.pi*rc*rc*mu))**2
        film_mm = p.outlet_diameter_mm/2*(1-math.sqrt(1-phi))
        return HydraulicState(mass,pressure_pa,re_inlet,re_equivalent,friction,a,ae,phi,
                              loss_inlet,loss_chamber,loss_outlet,mu,film_mm)

    @lru_cache(maxsize=1024)
    def hydraulics(self, pressure_kgf_cm2):
        pressure = _finite(pressure_kgf_cm2,'Перепад давления')
        if pressure <= 0:
            raise ValueError('Перепад давления должен быть положительным.')
        target = pressure*KGF_CM2_TO_PA
        p = self.parameters
        # Solve only within the original inlet graph AND the lambda <= .2 range.
        re_min = max(10**INLET_LOSS_CURVE[0][0],(24.6/.2)**(1/.75)/math.sqrt(p.inlet_count))
        re_max = min(10**INLET_LOSS_CURVE[-1][0],1e5/math.sqrt(p.inlet_count))
        scale = self.density_kg_m3*math.pi*(p.inlet_diameter_mm/2000)**2*p.inlet_count
        scale *= p.kinematic_viscosity_mm2_s*1e-6/(p.inlet_diameter_mm/1000)
        low, high = re_min*scale, re_max*scale
        left, right = self.state_for_mass(low), self.state_for_mass(high)
        if not left.pressure_pa <= target <= right.pressure_pa:
            raise PressureOutOfRange('Дитякин: давление вне диапазона исходных коэффициентов для этой геометрии и жидкости.')
        state = left
        for _ in range(90):
            state = self.state_for_mass((low+high)/2)
            if abs(state.pressure_pa-target)/target < 1e-10:
                break
            if state.pressure_pa < target:
                low = state.mass_kg_s
            else:
                high = state.mass_kg_s
        else:
            raise ValueError('Дитякин: расход не сошёлся; возможна граница режимов течения.')
        if p.outlet_length_mm < 4*state.film_mm:
            raise PressureOutOfRange('Дитякин: сопло короче 4 толщин плёнки; критическое сечение не сформировано.')
        return state

    def evaluate(self, pressure_kgf_cm2, *, allow_extrapolation=False):
        # Experimental extrapolation permission never overrides physical validity.
        state = self.hydraulics(pressure_kgf_cm2)
        return FlowEstimate(float(pressure_kgf_cm2),state.mass_kg_s*3600,False)


def characteristic_for_type(item, density_kg_m3):
    if item['kind'] == 'theoretical':
        if item.get('method') != 'dityakin_short_v1':
            raise ValueError('Неизвестная версия расчётной методики Дитякина.')
        return DityakinCharacteristic(DityakinParameters.from_dict(item.get('parameters',{})),density_kg_m3)
    from .characteristics import ExperimentalCharacteristic
    return ExperimentalCharacteristic(item['points'])
