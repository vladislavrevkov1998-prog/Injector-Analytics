"""Generate an explicitly synthetic identification example from user geometry.

The supplied measurement file is read only; targets are computed fields,
not a new experiment. Re-run from the repository root.
"""
import sys,json
from copy import deepcopy
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from fuelfield.core import project_from_json,project_to_json
from fuelfield.regression import RegressionTarget
from fuelfield.regression_io import write_target
root=Path(__file__).resolve().parents[1]
cfg,ns=project_from_json(json.loads((root/'examples/demo_project_user_field.json').read_text()))
cfg.z_mm=[0.,10.,20.,30.,40.,50.,60.];cfg.geometry_model='linear'
cfg.nozzle_presets['cooled']['z_tip_mm']=10.;cfg.nozzle_presets['uncooled']['z_tip_mm']=5.
cfg._project_extra['data_kind']='synthetic'
cfg._project_extra['description']='Synthetic v0.4 example: user geometry/measurement inputs retained, invented tip coordinates and computed fuel target. Not an experiment.'
(root/'examples/demo_project_v04_synthetic.json').write_text(json.dumps(project_to_json(cfg,ns),ensure_ascii=False,indent=2)+'\n')
truth=deepcopy(cfg);truth.nozzle_presets['cooled']['alpha_deg']=66.
target=RegressionTarget.from_project(truth,ns,source_kind='synthetic')
target.metadata.update(description='Computed synthetic fuel reference, not experimental',known_parameter={'cooled/alpha_deg':66.})
write_target(root/'examples/demo_regression_v04_synthetic.json',target)
print('Synthetic examples generated; original user project unchanged.')
