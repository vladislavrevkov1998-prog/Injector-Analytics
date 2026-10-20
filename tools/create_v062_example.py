"""Reproduce a wholly synthetic constraint demo from the synthetic v0.6 demo."""
import json
from pathlib import Path
import sys
root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root/'src'))
from fuelfield.core import project_from_json,project_to_json
from fuelfield.chamber import Chamber
from fuelfield.chamber_constraints import materialize_constraints,add_constraint

cfg,nozzles=project_from_json(json.loads((root/'examples/demo_project_v06_geometry.json').read_text(encoding='utf-8')))
geometry=cfg.geometry;materialize_constraints(geometry)
for wall in ('inner','outer'):
    geometry=add_constraint(geometry,'tangent',dict(primitive_id=wall+'-arc',endpoint='start'),dict(primitive_id=wall+'-line-1',endpoint='end'))
    next(p for p in geometry['primitives'] if p['id']==wall+'-line-1')['fixed']=True
geometry['provenance_v062']=dict(source_kind='synthetic',description='Both inlet lines fixed; each first arc tangent to its inlet line. All geometry, flows, sectors and pressures are synthetic.')
cfg.geometry=geometry;nozzles=Chamber(geometry).resolve_nozzle_anchors(nozzles,cfg)
output=project_to_json(cfg,nozzles);project_from_json(output)
path=root/'examples/demo_project_v062_constraints.json';path.write_text(json.dumps(output,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');print(path)
