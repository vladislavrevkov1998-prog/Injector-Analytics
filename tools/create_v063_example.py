"""A wholly synthetic S-wall profile, representative of the reported sketch."""
from copy import deepcopy
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from fuelfield.chamber import Chamber
from fuelfield.core import project_from_json,project_to_json
from fuelfield.chamber_constraints import materialize_constraints,add_object_tangent


def profile():
    def line(identity,wall,a,b):return dict(id=identity,kind='line',wall=wall,start=a,end=b,unit='mm')
    def arc(identity,wall,a,b,c,cw):return dict(id=identity,kind='arc',wall=wall,start=a,end=b,center=c,clockwise=cw,unit='mm')
    primitives=[
        arc('НР1','outer',[0,290],[10,300],[10,290],True),
        line('НР2','outer',[10,300],[60,300]),
        arc('НР3','outer',[60,300],[80,280],[60,280],True),
        arc('НР4','outer',[80,280],[100,260],[100,280],False),
        line('НР5','outer',[100,260],[160,260]),
        arc('ВН1','inner',[0,210],[10,200],[10,210],False),
        line('ВН2','inner',[10,200],[60,200]),
        arc('ВН3','inner',[60,200],[90,230],[60,230],False),
        arc('ВН4','inner',[90,230],[110,250],[110,230],True),
        line('ВН5','inner',[110,250],[160,250]),
    ]
    return dict(schema_version=2,enabled=True,base_z_mm=0.,outlet_z_mm=160.,internal_step_mm=1.,chord_error_mm=.01,primitives=primitives,objects=[],groups=[],sections=[dict(id=f'S{i+1}',z_mm=z,is_outlet=z==160) for i,z in enumerate((0.,20.,40.,60.,160.))],provenance=dict(source_kind='synthetic',description='Illustrative S-profile; not measured geometry'))


if __name__=='__main__':
    source=json.loads((ROOT/'examples/demo_project_v062_constraints.json').read_text(encoding='utf-8'))
    cfg,nozzles=project_from_json(source);geometry=profile()
    geometry['objects']=deepcopy(cfg.geometry['objects']);geometry['groups']=deepcopy(cfg.geometry['groups'])
    for obj in geometry['objects']:
        obj.pop('primitive_id',None);obj['wall']='free'
    materialize_constraints(geometry)
    for a,b in (('НР1','НР2'),('НР3','НР2'),('НР4','НР3'),('НР4','НР5'),('ВН1','ВН2'),('ВН3','ВН2'),('ВН4','ВН3'),('ВН4','ВН5')):
        geometry=add_object_tangent(geometry,a,b)
    for p in geometry['primitives']:
        if p['kind']=='line':p['fixed']=True
    cfg.geometry=geometry;cfg.z_mm=[s['z_mm'] for s in geometry['sections']]
    Chamber(geometry);project_from_json(project_to_json(cfg,nozzles))
    (ROOT/'examples/demo_project_v063_s_profile.json').write_text(json.dumps(project_to_json(cfg,nozzles),ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
