"""Reproduce an explicitly synthetic v0.6 geometry demonstration."""
from pathlib import Path
import sys,json
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from fuelfield.core import Settings,default_nozzles,project_to_json
from fuelfield.chamber import legacy_geometry,apply_groups
from fuelfield.quasi import QuasiSettings,PressureChannel


def build():
    cfg=Settings(z_mm=[0.,17.375,50.,100.,73.2,180.],n_sample_radial=40,n_sample_angular=360,preview_resolution=480,
        pressure_mode='calibrated',placement_mode='manual',flow_pressure_kgf_cm2=30.,sector_pressure_kgf_cm2=2.5)
    cfg.atomizer_types=[dict(id='synthetic',name='Синтетическая характеристика для demo',kind='experimental',points=[[2.5,3.],[30.,10.],[40.,12.]],provenance=dict(source_kind='synthetic'))]
    nozzles=default_nozzles(cfg)
    for n in nozzles:
        n.flow_kg_h=float(8+.8*np.sin(n.id*.2));n.atomizer_type_id='synthetic';n.geometry_overrides=[]
        n.manifold_id='A' if n.id<=70 else 'B';n.sector_ml=(4+np.sin(np.arange(36)*np.pi/18+n.id/10)).tolist();n.sector_source='СИНТЕТИЧЕСКИЕ объёмы, не испытание'
    nozzles[44].enabled=False
    raw=legacy_geometry(cfg,nozzles);raw.update(enabled=True,outlet_z_mm=180.,internal_step_mm=2.,chord_error_mm=.01,
        provenance=dict(source_kind='synthetic',note='Вся геометрия, расходы, сектора и давления в этом проекте демонстрационные.'))
    raw['primitives']=[
        dict(id='inner-line-1',wall='inner',kind='line',start=[0.,200.],end=[40.,205.],unit='mm'),
        dict(id='inner-arc',wall='inner',kind='arc',start=[40.,205.],end=[80.,215.],center=[40.,290.],clockwise=False,radius=85.,unit='mm'),
        dict(id='inner-line-2',wall='inner',kind='line',start=[80.,215.],end=[180.,235.],unit='mm'),
        dict(id='outer-line-1',wall='outer',kind='line',start=[0.,300.],end=[40.,310.],unit='mm'),
        dict(id='outer-arc',wall='outer',kind='arc',start=[40.,310.],end=[100.,330.],center=[40.,410.],clockwise=False,radius=100.,unit='mm'),
        dict(id='outer-line-2',wall='outer',kind='line',start=[100.,330.],end=[180.,345.],unit='mm'),
        dict(id='datum',wall='reference',kind='point',start=[17.375,270.],unit='mm')]
    steps=360/70*(1+.12*np.sin(np.arange(70)*2*np.pi/70));steps[-1]=360-float(steps[:-1].sum())
    raw['groups']=[dict(id='outer-uneven',members=['N'+str(i) for i in range(1,71)],mode='steps',steps_deg=steps.tolist(),phase_deg=0.,direction=-1,row_offset_deg=0.),
        dict(id='inner-staggered',members=['N'+str(i) for i in range(71,141)],mode='uniform',phase_deg=0.,direction=-1,row_offset_deg=-360/140)]
    for i in range(16):
        raw['objects'].append(dict(id=f'hole-{i+1}',type='hole',wall='outer',primitive_id='outer-arc',z_mm=60.,radius_mm=312.0204102887,theta_deg=i*360/16,
            parameters=dict(inclination_deg=30.,azimuth_deg=0.,flow_kg_s=None,penetration_mm=None),units=dict(length='mm',angle='deg',flow='kg/s'),provenance=dict(source_kind='synthetic',note='Расход неизвестен; воздушная физика не рассчитывается')))
    raw['groups'].append(dict(id='hole-ring',members=[f'hole-{i+1}' for i in range(16)],mode='uniform',phase_deg=0.,direction=1,row_offset_deg=0.))
    for i,theta in enumerate((10.,85.,190.,277.)):
        raw['objects'].append(dict(id=f'insert-{i+1}',type='insert',wall='inner',primitive_id='inner-line-2',z_mm=120.,radius_mm=223.,theta_deg=theta,radial_offset_mm=8.,
            parameters=dict(inclination_deg=None,azimuth_deg=None,flow_kg_s=None,penetration_mm=8.),units=dict(length='mm',angle='deg',flow='kg/s'),provenance=dict(source_kind='synthetic')))
    raw['objects'].append(dict(id='custom-probe',type='probe',wall='free',z_mm=150.,radius_mm=280.,theta_deg=22.,parameters=dict(flow_kg_s=None),units=dict(length='mm',angle='deg'),provenance=dict(source_kind='synthetic')))
    raw['types']={'probe':dict(label='Пользовательский условный зонд',properties={'purpose':'Расширяемый тип; физический источник не задан'})}
    cfg.geometry,nozzles=apply_groups(raw,nozzles)
    phase=np.arange(cfg.n_angular)*2*np.pi/cfg.n_angular
    mode=QuasiSettings(cfg.n_angular,dt_s=.5,channels=(PressureChannel('A',tuple(30*(1+.07*np.sin(phase)))),PressureChannel('B',tuple(28*(1+.04*np.cos(phase))))),metadata=dict(source_kind='synthetic'))
    cfg.quasi_settings=mode.to_dict();cfg.validate()
    project=project_to_json(cfg,nozzles);project['title']='СИНТЕТИЧЕСКАЯ v0.6: линии, дуги, объекты, неравномерные группы';project['provenance']={'source_kind':'synthetic'}
    return project


if __name__=='__main__':
    out=Path(__file__).resolve().parents[1]/'examples/demo_project_v06_geometry.json'
    out.write_text(json.dumps(build(),ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8');print(out)
