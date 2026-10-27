"""Run the complete unittest discovery with a fresh process per module."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import unittest

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--jobs', type=int, default=2)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    assert args.jobs > 0
    sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'tests')]
    loader = unittest.TestLoader()
    expected = loader.discover(str(ROOT / 'tests')).countTestCases()
    assert not loader.errors, loader.errors
    modules = {p.stem: loader.loadTestsFromName(p.stem).countTestCases()
               for p in sorted((ROOT / 'tests').glob('test*.py'))}
    modules = {name: count for name, count in modules.items() if count}
    assert not loader.errors and sum(modules.values()) == expected
    args.out.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()

    def run(name, count):
        folder = args.out / name
        folder.mkdir(exist_ok=True)
        env = dict(os.environ, QT_QPA_PLATFORM='offscreen',
                   OPENBLAS_NUM_THREADS='1', PYTHONPATH=os.pathsep.join(
                       [str(ROOT / 'src'), str(ROOT / 'tests')]),
                   XDG_CONFIG_HOME=str(folder / 'config'))
        with (folder / 'suite.log').open('w') as log:
            result = subprocess.run([sys.executable, '-X', 'faulthandler', '-B',
                                     '-m', 'unittest', name, '-v'], cwd=ROOT,
                                    env=env, stdout=log, stderr=subprocess.STDOUT,
                                    timeout=900)
        content = (folder / 'suite.log').read_text()
        found = re.findall(r'Ran (\d+) tests? in ([\d.]+)s', content)
        actual, seconds = found[-1] if found else (0, 0.)
        skipped = len(re.findall(r'\.\.\. skipped', content))
        passed = result.returncode == 0 and int(actual) == count and not skipped
        return dict(module=name, expected=count, tests=int(actual),
                    seconds=float(seconds), exit_code=result.returncode,
                    skipped=skipped, status='PASS' if passed else 'FAIL')

    records = []
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures = [pool.submit(run, name, count) for name, count in modules.items()]
        for future in as_completed(futures):
            record = future.result()
            records.append(record)
            print(json.dumps(record), flush=True)
    report = dict(status='PASS' if all(r['status'] == 'PASS' for r in records) else 'FAIL',
                  method='Complete discovery; fresh subprocess per module', jobs=args.jobs,
                  expected_tests=expected, tests=sum(r['tests'] for r in records),
                  skipped=sum(r['skipped'] for r in records),
                  wall_seconds=time.monotonic()-start,
                  modules=sorted(records, key=lambda r: r['module']))
    (args.out / 'summary.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({key: value for key, value in report.items() if key != 'modules'}), flush=True)
    raise SystemExit(0 if report['status'] == 'PASS' and report['tests'] == expected else 1)


if __name__ == '__main__':
    main()
