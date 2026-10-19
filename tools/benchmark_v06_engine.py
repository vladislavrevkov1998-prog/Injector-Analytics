"""Isolate repeatable legacy quasi engine timings from cold Qt rendering.

Run sequentially for baseline/current with identical BLAS thread settings.
"""
import argparse,hashlib,json,sys,time,gc
from pathlib import Path
import numpy as np
parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--source-root',type=Path,default=Path(__file__).resolve().parents[1]);parser.add_argument('--out',type=Path,required=True);args=parser.parse_args();root=args.source_root.resolve();sys.path.insert(0,str(root/'src'))
from fuelfield.core import project_from_json
from fuelfield.quasi import calculate_quasi
from fuelfield.quasi_io import load_settings
cfg,ns=project_from_json(json.loads((root/'examples/demo_project_user_field.json').read_text()));mode,bindings=load_settings(root/'examples/demo_user_field_140_two_collectors.quasi.json');cfg.quasistationary=True;cfg.quasi_settings=mode.to_dict()
for n in ns:
    if str(n.id) in bindings:n.manifold_id=bindings[str(n.id)]
samples=[];digests=[];errors=[]
for i in range(4):
    started=time.perf_counter();result=calculate_quasi(ns,cfg);elapsed=(time.perf_counter()-started)*1000
    assert result.maximum_balance_error<1e-11
    digest=hashlib.sha256(b''.join(p.cell_mass_kg_s.tobytes() for p in result.composites)).hexdigest();errors.append(result.maximum_balance_error)
    if i:samples.append(elapsed);digests.append(digest)
    del result;gc.collect()
assert len(set(digests))==1
report=dict(source_root=str(root),method='pure quasi engine, 4 z,140nozzles,140timestamps,720²; one warmup then3measured; sequential processes, OPENBLAS_NUM_THREADS=1',samples_ms=samples,median_ms=float(np.median(samples)),maximum_error=max(errors),composite_arrays_sha256=digests[0])
args.out.parent.mkdir(parents=True,exist_ok=True);args.out.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
