"""Measure first visits and cached plane switching on the demo project (Qt offscreen)."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import statistics
import sys
import time

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
os.environ.setdefault('MPLCONFIGDIR', '/tmp/injector-mpl')
root = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--source', type=Path, default=root/'src/fuelfield/gui.py',
                    help='GUI source in a src/fuelfield directory, optionally from v0.1.2')
args = parser.parse_args()
# Load the complete matching core/visualization as well, for an honest baseline.
sys.path.insert(0, str(args.source.resolve().parents[1]))
from PyQt5.QtWidgets import QApplication
from fuelfield.core import project_from_json, calculate_plane, evaluate_continuous_field

app = QApplication([])
spec = importlib.util.spec_from_file_location('fuelfield.benchmark_gui', args.source)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
window = module.MainWindow()
cfg, nozzles = project_from_json(json.loads(
    (root/'examples/demo_project_140.json').read_text(encoding='utf-8')))
window.cfg, window.nozzles = cfg, nozzles
if hasattr(window, '_restoring'):
    window._restoring = True
window._fill_controls()
window._fill_nozzles()
if hasattr(window, '_reset_history'):
    window._reset_history()
# The timed region excludes calculation, but includes Qt/Matplotlib painting.
pairs = [(calculate_plane(nozzles, cfg, z), evaluate_continuous_field(nozzles, cfg, z))
         for z in cfg.z_mm]
window.tabs.setCurrentIndex(3)
window.show()
app.processEvents()
window._done(pairs)
app.processEvents()


def measure(sequence):
    elapsed = []
    for index in sequence:
        start = time.perf_counter()
        window.plane_select.setCurrentIndex(index)
        app.processEvents()
        elapsed.append((time.perf_counter()-start)*1000)
    return elapsed


cold = measure((1, 2, 3))
warm = measure((0, 1, 2, 3, 0, 1, 2, 3))
print(json.dumps({'cold_ms': cold, 'warm_ms': warm,
                  'cold_median_ms': statistics.median(cold),
                  'warm_median_ms': statistics.median(warm)}))
window.close()
