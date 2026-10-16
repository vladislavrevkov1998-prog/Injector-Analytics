"""Headless v0.5 session calculation and CSV/XLSX report from a project snapshot."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from fuelfield.core import project_from_json
from fuelfield.measurement import MeasurementSession
from fuelfield.measurement_io import load_session,write_report
from fuelfield.transient import calculate_session
parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('project',type=Path);parser.add_argument('--session',type=Path);parser.add_argument('--out',type=Path,required=True);args=parser.parse_args()
cfg,ns=project_from_json(json.loads(args.project.read_text(encoding='utf-8-sig')))
session=load_session(args.session) if args.session else MeasurementSession.from_dict(cfg._project_extra.get('measurement_session',{}))
result=calculate_session(ns,cfg,session);outputs=write_report(args.out,result)
print(json.dumps(dict(files=[str(p) for p in outputs],steps=len(session.steps),elapsed_s=result.elapsed_s,max_balance=result.manifest()['maximum_snapshot_balance_error'],composite_global_balance=None),ensure_ascii=False))
