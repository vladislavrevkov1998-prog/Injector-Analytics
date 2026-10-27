"""Compare legacy fuel arrays with an archived pre-v0.8 checkout.

Run with the same installed dependencies as the application::

    python -B tools/compare_v08_legacy.py --out /tmp/v08-legacy-comparison

The output directory retains both NPZ files, the worker and RESULT.json. This
checks numerical compatibility; concurrent load makes the timings unsuitable
for a performance claim. No Git checkout or tracked input file is modified.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile

import numpy as np


WORKER = '''from pathlib import Path
import json,sys,time
import numpy as np
from fuelfield.core import project_from_json,calculate_plane,evaluate_continuous_field
from fuelfield.quasi import calculate_quasi
start=time.perf_counter()
root=Path(sys.argv[1]);out=Path(sys.argv[2]);arrays={}
cfg,ns=project_from_json(json.loads((root/'examples/demo_project_user_field.json').read_text()))
for k,z in enumerate(cfg.z_mm):
    p=calculate_plane(ns,cfg,z)
    for field in ('cell_mass_kg_s','cell_area_m2','flux_kg_m2_s','radial_centers_mm','radial_edges_mm','theta_centers_deg'):
        arrays['legacy_'+str(k)+'_'+field]=np.asarray(getattr(p,field))
    arrays['legacy_'+str(k)+'_balances']=np.array([p.injected_kg_s,p.inside_kg_s,p.inner_kg_s,p.outer_kg_s,p.wall_bound_kg_s])
p=evaluate_continuous_field(ns,cfg,cfg.z_mm[-1],resolution=200)
arrays['legacy_preview_flux']=p.flux_kg_m2_s
arrays['legacy_preview_mask']=np.ma.getmaskarray(p.masked_flux_kg_m2_s)
cfg,ns=project_from_json(json.loads((root/'examples/fuel_field_project_quasi.json').read_text()))
# Stored source values and scan protocol; omit only the costly visual cache.
q=calculate_quasi(ns,cfg,preview=False)
arrays['quasi_flows']=q.flows
for k,p in enumerate(q.composites):
    for field in ('cell_mass_kg_s','cell_area_m2','flux_kg_m2_s','scan_times_s','scan_steps'):
        arrays['quasi_'+str(k)+'_'+field]=np.asarray(getattr(p,field))
    for step in (0,q.settings.n_steps-1):
        snap=q.snapshot(k,step)
        arrays['quasi_'+str(k)+'_snapshot_'+str(step)]=snap.cell_mass_kg_s
arrays['quasi_balances']=np.asarray([list(x.values()) for x in q.balances],dtype=float)
# Schema 3 synthetic transport: stored sources, one actual outlet section.
cfg,ns=project_from_json(json.loads((root/'examples/demo_project_v07_confined_synthetic.json').read_text()))
p=calculate_plane(ns,cfg,cfg.z_mm[-1])
for field in ('cell_mass_kg_s','cell_area_m2','flux_kg_m2_s','suspended_cell_mass_kg_s','film_inner_cell_mass_kg_s','film_outer_cell_mass_kg_s'):
    arrays['confined_'+field]=np.asarray(getattr(p,field))
for field,value in p.transport_momentum.items():
    arrays['confined_momentum_'+field]=np.asarray(value)
np.savez_compressed(out,**arrays)
print(json.dumps({'runtime_s':time.perf_counter()-start,'arrays':len(arrays),'user_project_air_enabled':getattr(cfg,'air_model',{}).get('enabled',False)}))
'''


def run_worker(worker, root, output):
    """A fresh interpreter prevents cross-imports from the two source trees."""
    env = os.environ.copy()
    env.update(PYTHONPATH=str(root / 'src'), OPENBLAS_NUM_THREADS='1',
               MPLCONFIGDIR=str(output.parent / 'mpl-cache'),
               QT_QPA_PLATFORM='offscreen')
    try:
        completed = subprocess.run(
            [sys.executable, '-B', str(worker), str(root), str(output)],
            cwd=output.parent, env=env, capture_output=True, text=True,
            check=True)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError('Comparison worker failed for ' + str(root) + '\n' +
                           exc.stdout[-10000:] + '\n' + exc.stderr[-10000:]) from exc
    try:
        return json.loads(completed.stdout.strip().splitlines()[-1])
    except (IndexError, json.JSONDecodeError) as exc:
        raise RuntimeError('Comparison worker did not return its JSON summary:\n' +
                           completed.stdout[-10000:] + '\n' + completed.stderr[-10000:]) from exc


def compare(root, baseline_ref, out):
    root, out = Path(root).resolve(), Path(out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    # Resolve to an actual commit before archive; an option-like ref cannot be
    # interpreted as a git-archive flag or select arbitrary filesystem content.
    baseline_sha = subprocess.check_output(
        ['git', 'rev-parse', '--verify', '--end-of-options', baseline_ref + '^{commit}'],
        cwd=root, text=True).strip()
    worker = out / 'worker.py'
    worker.write_text(WORKER, encoding='utf-8')
    results = {}
    with tempfile.TemporaryDirectory(prefix='baseline-', dir=out) as temporary:
        temporary = Path(temporary)
        archive = temporary / 'source.tar'
        subprocess.run(['git', 'archive', '--format=tar', '--output', str(archive),
                        baseline_sha], cwd=root, check=True)
        checkout = temporary / 'checkout'
        checkout.mkdir()
        # data rejects absolute paths, escaping links and non-data devices.
        # The archive is extracted into a new directory, never an old checkout.
        with tarfile.open(archive) as bundle:
            bundle.extractall(checkout, filter='data')
        results['baseline'] = run_worker(worker, checkout, out / 'baseline.npz')
        print(json.dumps(dict(label='baseline', **results['baseline'])), flush=True)
    results['current'] = run_worker(worker, root, out / 'current.npz')
    print(json.dumps(dict(label='current', **results['current'])), flush=True)
    errors, digests = [], []
    with np.load(out / 'baseline.npz', allow_pickle=False) as old, np.load(
            out / 'current.npz', allow_pickle=False) as new:
        count = len(old.files)
        errors.extend('extra:' + name for name in new.files if name not in old.files)
        for name in old.files:
            if (name not in new.files or old[name].dtype != new[name].dtype or
                    old[name].shape != new[name].shape or
                    old[name].tobytes() != new[name].tobytes()):
                errors.append(name)
            # Deliberately stable ordering and bytes match the original v0.8
            # acceptance evidence, including dtype/NaN bit patterns per array.
            digests.append(name.encode() + old[name].tobytes())
    result = dict(
        baseline_sha=baseline_sha, baseline_ref=baseline_ref,
        current_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'],
                                               cwd=root, text=True).strip(),
        current_worktree_dirty=bool(subprocess.check_output(
            ['git', 'status', '--porcelain'], cwd=root, text=True).strip()),
        array_count=count, byte_identical=not errors, differences=errors,
        aggregate_sha256=hashlib.sha256(b''.join(digests)).hexdigest(),
        runtimes=results,
        user_json_sha256=hashlib.sha256(
            (root / 'examples/demo_project_user_field.json').read_bytes()).hexdigest(),
        timing_scope='Concurrent load may vary; timings are not a performance comparison.')
    (out / 'RESULT.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline-ref', default='944f1f9',
                        help='Pre-v0.8 Git commit/ref (default: 944f1f9).')
    parser.add_argument('--out', type=Path, required=True,
                        help='Directory for both NPZ files and the JSON evidence.')
    args = parser.parse_args()
    result = compare(Path(__file__).resolve().parents[1], args.baseline_ref, args.out)
    print(json.dumps(result))
    if not result['byte_identical']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
