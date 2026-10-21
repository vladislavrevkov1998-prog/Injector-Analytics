"""Explicit parameters of the one-way confined-fuel model (no guessed data)."""
from copy import deepcopy
from dataclasses import dataclass,field,fields
import math

MODES={'legacy':'Независимые сечения — прежняя модель',
       'confined_inertial':'Перенос топлива внутри ЖТ — инерционная модель'}
FLOAT_FIELDS=('carrier_inlet_velocity_m_s','injection_speed_m_s','relaxation_time_s',
    'film_speed_m_s','cone_half_width_deg','turbulence_rms_m_s','turbulence_time_s',
    'max_step_mm','wall_tolerance_mm')
INTEGER_FIELDS=('carrier_radial_cells','emission_radial_samples','emission_angular_samples','seed')
LABELS={
    'carrier_inlet_velocity_m_s':'Скорость условного носителя на входе, м/с',
    'injection_speed_m_s':'Скорость истечения топлива, м/с',
    'relaxation_time_s':'Эффективное время релаксации капли τ, с',
    'film_speed_m_s':'Скорость переноса плёнки вдоль стенки, м/с',
    'cone_half_width_deg':'Полуширина углового распределения, °',
    'turbulence_rms_m_s':'СКО поперечных флуктуаций носителя, м/с',
    'turbulence_time_s':'Время корреляции флуктуаций, с',
    'max_step_mm':'Максимальный осевой шаг, мм',
    'wall_tolerance_mm':'Погрешность аппроксимации стенки, мм',
    'carrier_radial_cells':'Радиальных ячеек носителя',
    'emission_radial_samples':'Узлов по углу раскрытия',
    'emission_angular_samples':'Окружных узлов источника (кратно 36)',
    'seed':'Seed воспроизводимой квадратуры'}


def _float(value,name):
    if value is None:return None
    if isinstance(value,bool):raise ValueError(LABELS[name]+': требуется число, не флажок')
    try:result=float(str(value).strip().replace(',','.'))
    except (ValueError,TypeError):raise ValueError(LABELS[name]+': требуется число') from None
    if not math.isfinite(result):raise ValueError(LABELS[name]+': требуется конечное число')
    return result


@dataclass
class TransportSettings:
    schema_version: int=1
    mode: str='legacy'
    carrier_inlet_velocity_m_s: object=None
    injection_speed_m_s: object=None
    relaxation_time_s: object=None
    film_speed_m_s: object=None
    cone_half_width_deg: object=None
    turbulence_rms_m_s: float=0.
    turbulence_time_s: object=None
    wall_model: str='capture_film'
    max_step_mm: float=1.
    carrier_radial_cells: int=16
    wall_tolerance_mm: float=.02
    emission_radial_samples: int=4
    emission_angular_samples: int=144
    seed: int=0
    _extra: dict=field(default_factory=dict,repr=False)

    @property
    def enabled(self):return self.mode=='confined_inertial'

    @classmethod
    def from_dict(cls,raw):
        if raw is None:raw={}
        if not isinstance(raw,dict):raise ValueError('Настройки переноса топлива должны быть объектом JSON')
        known={entry.name for entry in fields(cls) if not entry.name.startswith('_')}
        data={key:deepcopy(value) for key,value in raw.items() if key in known}
        for name in FLOAT_FIELDS:
            if name in data:data[name]=_float(data[name],name)
        return cls(**data,_extra={key:deepcopy(value) for key,value in raw.items() if key not in known})

    def to_dict(self):
        return {**deepcopy(self._extra),**{entry.name:deepcopy(getattr(self,entry.name))
            for entry in fields(self) if not entry.name.startswith('_')}}

    def validate(self):
        if isinstance(self.schema_version,bool) or not isinstance(self.schema_version,int) or self.schema_version!=1:raise ValueError('Неподдерживаемая версия настроек переноса топлива')
        if not isinstance(self.mode,str) or self.mode not in MODES:raise ValueError('Неизвестная модель переноса топлива')
        if self.wall_model!='capture_film':raise ValueError('Поддерживается только явно заданный захват топлива в стеночную плёнку')
        positive=('carrier_inlet_velocity_m_s','injection_speed_m_s','relaxation_time_s','film_speed_m_s')
        for name in FLOAT_FIELDS:
            value=_float(getattr(self,name),name)
            setattr(self,name,value)
            required=self.enabled and name in (*positive,'cone_half_width_deg')
            if value is None:
                if required:raise ValueError(LABELS[name]+': задайте параметр; измеренное значение неизвестно')
                if name in ('max_step_mm','wall_tolerance_mm','turbulence_rms_m_s'):raise ValueError(LABELS[name]+': требуется число')
                continue
            if name in (*positive,'max_step_mm','wall_tolerance_mm','turbulence_time_s') and value<=0:raise ValueError(LABELS[name]+': значение должно быть >0')
            if name in ('cone_half_width_deg','turbulence_rms_m_s') and value<0:raise ValueError(LABELS[name]+': значение должно быть ≥0')
        if self.cone_half_width_deg is not None and self.cone_half_width_deg>=90:raise ValueError('Полуширина углового распределения должна быть меньше 90°')
        if self.enabled and self.turbulence_rms_m_s>0 and self.turbulence_time_s is None:raise ValueError('Для ненулевых флуктуаций задайте время корреляции, с')
        ranges={'carrier_radial_cells':(4,128),'emission_radial_samples':(1,32),
                'emission_angular_samples':(36,1440),'seed':(0,2**32-1)}
        for name,(lo,hi) in ranges.items():
            value=getattr(self,name)
            if isinstance(value,bool) or not isinstance(value,int) or not lo<=value<=hi:raise ValueError(LABELS[name]+f': требуется целое {lo}…{hi}')
        if self.emission_angular_samples%36:raise ValueError('Число окружных узлов источника должно быть кратно 36')
        if not 1e-4<=self.max_step_mm<=1000:raise ValueError('Осевой шаг переноса: 10⁻⁴…1000 мм')
        if not 1e-6<=self.wall_tolerance_mm<=5:raise ValueError('Погрешность стенки: 10⁻⁶…5 мм')
        return self


def synthetic_settings():
    """Clearly labelled runnable assumptions, never measured nozzle properties."""
    return TransportSettings.from_dict(dict(mode='confined_inertial',
        carrier_inlet_velocity_m_s=20.,injection_speed_m_s=40.,relaxation_time_s=.005,
        film_speed_m_s=1.,cone_half_width_deg=5.,turbulence_rms_m_s=0.,turbulence_time_s=.002,
        provenance={'source_kind':'synthetic','description':'Учебные параметры; не данные испытаний'})).validate()
