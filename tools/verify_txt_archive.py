"""Verify an already built TXT ZIP, restore it, and run its actual runtime.

Run with the application's installed dependencies. The Unicode extraction
folder is retained for inspection; no source is imported from the repository.
"""
import argparse
import contextlib
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]

SMOKE = r'''
import json
import sys
import time
sys.path.insert(0, 'src')
import numpy as np
from PyQt5.QtCore import QUrl
from PyQt5.QtTest import QTest
from PyQt5.QtWidgets import QApplication, QTextBrowser
from fuelfield.gui import MainWindow
from fuelfield.help_text import read_help
from fuelfield.runtime_help import HELP, IMAGES
from fuelfield.core import Settings, default_nozzles, calculate_plane, project_to_json
from fuelfield.mixture_field import compute_mixture
app = QApplication([])
w = MainWindow(); w.show(); app.processEvents()
assert len(w.nozzles) == 140
assert len(HELP) == 7 and len(IMAGES) == 5
for name in HELP:
    assert len(read_help(name)) > 2000, name
w.open_air_balance(); w.air_dialog.show_help()
assert not w.air_dialog.enabled.isChecked()
browser = w.air_dialog.guide_dialog.findChild(QTextBrowser)
assert 'Баланс воздуха' in browser.toPlainText()
browser.anchorClicked.emit(QUrl('V0.8_IMPLEMENTED_MODEL.md'))
assert 'Инструкция отсутствует' not in browser.toPlainText()
assert len(browser.toPlainText()) > 2000
w.air_dialog.guide_dialog.close(); w.air_dialog.close()
cfg = Settings(n_radial=6, n_angular=16, z_mm=[20.])
cfg.preview_resolution = 200
ns = default_nozzles(cfg)
for n in ns:
    n.flow_kg_h = 360. / 140
    n.alpha_deg = 0.; n.k_linear = 0.; n.k_quadratic = 0.
cfg.air_model = dict(enabled=True, total_kg_s=2., l0=2., distribution='uniform',
    max_path_mm=40., sources=[dict(id='SYNTHETIC_SMOKE', kind='hole',
        method='fixed', flow_kg_s=2., z_mm=0., radius_mm=250., theta_deg=0.,
        direction=[1.,0.,0.], mixed_fraction=.25)])
plane = calculate_plane(ns, cfg, 20.)
fuel = plane.cell_mass_kg_s.copy()
mixture = compute_mixture(plane, cfg, ns)
np.testing.assert_array_equal(fuel, plane.cell_mass_kg_s)
np.testing.assert_allclose(mixture.air_total_kg_s.sum(), 2., rtol=1e-12)
np.testing.assert_allclose(mixture.air_mixed_kg_s.sum(), .5, rtol=1e-12)
np.testing.assert_allclose(mixture.alpha_section_total, 2./(2.*fuel.sum()), rtol=1e-12)
np.testing.assert_allclose(mixture.alpha_section_mixed, .5/(2.*fuel.sum()), rtol=1e-12)
# Exercise the new workflow from this restored copy, including its actual
# background worker and common circular matrix; never import the source tree.
errors = []
w._error = errors.append
w.apply_project_state(project_to_json(cfg, ns), 'SYNTHETIC portable acceptance')
w.run_calc()
deadline = time.monotonic() + 60
while w.thread is not None and time.monotonic() < deadline:
    QTest.qWait(10)
assert w.thread is None and not errors, errors
assert w.air_results is not None and len(w.results) == 1
assert not hasattr(w, 'air_results_button')
state_before = project_to_json(w.cfg, w.nozzles)
fuel_before = w.results[0].cell_mass_kg_s.copy()
w.tabs.setCurrentIndex(3)
w.metric.setCurrentIndex(w.metric.findData('alpha_total'))
assert set(w.theme_actions) == {'light', 'light_green', 'dark'}
for theme in w.theme_actions:
    w.set_theme(theme); app.processEvents()
    assert project_to_json(w.cfg, w.nozzles) == state_before
    np.testing.assert_array_equal(w.results[0].cell_mass_kg_s, fuel_before)
assert w.figmap.axes[0].name == 'polar'
assert w.geometry_editor.mode_selector.findData('air') >= 0
w.close(); app.processEvents()
print('PORTABLE_V08_OK ' + json.dumps(dict(embedded_guides=len(HELP),
    embedded_images=len(IMAGES), fuel_kg_s=float(fuel.sum()),
    air_kg_s=float(mixture.air_total_kg_s.sum()),
    alpha_total=float(mixture.alpha_section_total),
    joint_f5=True, circular_alpha=True, themes=3)))
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('archive', type=Path)
    parser.add_argument('--extract', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()
    archive = args.archive.resolve()
    with zipfile.ZipFile(archive) as bundle:
        names = bundle.namelist()
        assert bundle.testzip() is None
        for name in names:
            path = Path(name)
            assert not path.is_absolute() and '..' not in path.parts
            assert not set(path.parts) & {'docs', 'tests', 'examples', '__pycache__', '.git'}
            assert path.suffix == '.txt', name
        bundle.extractall(args.extract)
    base = next(args.extract.iterdir())
    scope = {'__name__': 'portable_verify'}
    exec((base / 'ВОССТАНОВИТЬ_ПРОГРАММУ.txt').read_text(encoding='utf-8'), scope)
    with contextlib.redirect_stdout(io.StringIO()):
        destination = scope['restore'](base)
        assert scope['restore'](base) == destination
    for relative, checksum in scope['FILES'].items():
        assert hashlib.sha256((destination / relative).read_bytes()).hexdigest() == checksum
    runtime = list((ROOT / 'src/fuelfield').rglob('*.py'))
    for source in runtime + [ROOT / 'main.py', ROOT / 'requirements.txt']:
        assert source.read_bytes() == (destination / source.relative_to(ROOT)).read_bytes(), source
    env = dict(os.environ, QT_QPA_PLATFORM='offscreen', PYTHONPATH='')
    result = subprocess.run([sys.executable, '-B', '-c', SMOKE], cwd=destination,
                            env=env, capture_output=True, text=True, timeout=90)
    assert result.returncode == 0, result.stdout + result.stderr
    details = json.loads(next(line.split(' ', 1)[1] for line in result.stdout.splitlines()
                             if line.startswith('PORTABLE_V08_OK ')))
    report = dict(status='PASS', archive=archive.name,
                  sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
                  source_commit=scope['SOURCE_COMMIT'], version=scope['VERSION'],
                  archive_entries=len(names), restored_files=len(scope['FILES']),
                  runtime_modules_compared=len(runtime), restored_outside_repository=True,
                  platform=sys.platform, python=sys.version, smoke=details,
                  windows_spyder='NOT RUN')
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
