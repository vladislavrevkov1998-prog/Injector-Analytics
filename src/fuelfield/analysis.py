"""Quantitative representations of integrated control cells, without mutation."""
import numpy as np

METRICS = {'flux':'Плотность расхода, кг/(м²·с)', 'kg_h':'Расход ячейки, кг/ч',
           'l_min':'Расход ячейки, л/мин', 'volume':'Объём ячейки, мл',
           'ring':'Индекс относительно кольца', 'track':'Индекс относительно тракта'}
UNITS = {'flux':'кг/(м²·с)', 'kg_h':'кг/ч', 'l_min':'л/мин', 'volume':'мл', 'ring':'1', 'track':'1'}

def cell_values(plane, metric='flux', density=800., interval=180.):
    if metric not in METRICS:raise ValueError('Неизвестное представление')
    if not np.isfinite(density) or density<=0:raise ValueError('Плотность должна быть положительной')
    if not np.isfinite(interval) or interval<=0:raise ValueError('Интервал должен быть положительным')
    mass=np.asarray(plane.cell_mass_kg_s);area=np.asarray(plane.cell_area_m2)
    if metric=='flux':values=plane.flux_kg_m2_s
    elif metric=='kg_h':values=mass*3600
    elif metric=='l_min':values=mass*60000/density
    elif metric=='volume':values=mass*interval*1e6/density
    else:
        axes=1 if metric=='ring' else None
        mean=mass.sum(axis=axes,keepdims=True)/area.sum(axis=axes,keepdims=True)
        values=np.divide(plane.flux_kg_m2_s,mean,out=np.full_like(mass,np.nan),where=mean>0)
    return np.ma.masked_invalid(np.array(values,copy=True))


def matching_index(values, target):
    matches=np.flatnonzero(np.isclose(values,target,rtol=0,atol=1e-7))
    return int(matches[0]) if len(matches)==1 else None

# Conversion constants assume canonical ṁ [kg/s], ρ [kg/m³], t [s], S [m²].
EQUATIONS = {
    'flux':'qᵢⱼ = ṁᵢⱼ / Sᵢⱼ  [кг/(м²·с)]',
    'kg_h':'Gᵢⱼ = 3600 · ṁᵢⱼ  [кг/ч]',
    'l_min':'Qᵢⱼ = 60000 · ṁᵢⱼ / ρ  [л/мин]',
    'volume':'Vᵢⱼ = 10⁶ · ṁᵢⱼ · t / ρ  [мл]',
    'ring':'Iᵢⱼ = qᵢⱼ / ⟨q⟩ᵢ;  ⟨q⟩ᵢ = Σⱼṁᵢⱼ / ΣⱼSᵢⱼ',
    'track':'Iᵢⱼ = qᵢⱼ / ⟨q⟩;  ⟨q⟩ = Σᵢⱼṁᵢⱼ / ΣᵢⱼSᵢⱼ',
}
