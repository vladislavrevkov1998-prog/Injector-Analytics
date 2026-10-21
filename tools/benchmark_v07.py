"""Native v0.7 UI/worker measurements on unchanged projects and synthetic data.

Run with QT_QPA_PLATFORM=offscreen and --out /tmp/v07-benchmark.json.  Wall
times are observations, not machine-dependent pass/fail thresholds.  No UI
timer is disabled.  The original example is never edited on disk.
"""
from copy import deepcopy
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))

from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage
from PyQt5.QtTest import QTest
from PyQt5.QtWidgets import QApplication

from fuelfield.core import (calculate_plane, evaluate_continuous_field,
                            project_from_json, project_to_json)
from fuelfield.carrier import clear_carrier_cache
from fuelfield.fuel_transport import clear_transport_cache, transport_cache_info
from fuelfield.gui import MainWindow


def stats(values):
    values = list(values)
    if not values:
        return dict(samples=0, median_ms=None, p95_ms=None, max_ms=None)
    return dict(samples=len(values), median_ms=round(float(np.median(values)), 3),
                p95_ms=round(float(np.percentile(values, 95)), 3),
                max_ms=round(float(max(values)), 3))


def pixels(canvas):
    image = canvas.grab().toImage().convertToFormat(QImage.Format_RGBA8888)
    buffer = image.bits()
    buffer.setsize(image.byteCount())
    return np.frombuffer(buffer, dtype=np.uint8).reshape(
        image.height(), image.bytesPerLine()//4, 4)[:, :image.width()].copy()


def wait_until(condition, timeout_s=180.):
    deadline = time.monotonic()+timeout_s
    while not condition() and time.monotonic() < deadline:
        QTest.qWait(5)
    if not condition():
        raise AssertionError('Timed out waiting for actual native Qt state')


def settle(app):
    for _ in range(3):
        app.processEvents()
    QTest.qWait(2)
    app.processEvents()


def open_project(window, path):
    with patch('fuelfield.gui.QFileDialog.getOpenFileName', return_value=(str(path), '')):
        window.open_project()


def worker_measurement(window, app):
    errors = []
    beats = []
    stages = {}
    real_done = window._done

    def observed_done(values):
        stages['completed_slot_entered'] = time.perf_counter()
        real_done(values)
        stages['completed_slot_returned'] = time.perf_counter()

    timer = QTimer()
    timer.setInterval(5)
    timer.timeout.connect(lambda: beats.append(time.perf_counter()))
    timer.start()
    QTest.qWait(15)
    started = time.perf_counter()
    with patch.object(window, '_error', side_effect=lambda value: errors.append(str(value))), \
         patch.object(window, '_done', side_effect=observed_done):
        # Use the same QAction shortcut as the user; calculation is real.
        window.activateWindow()
        QApplication.setActiveWindow(window)
        window.setFocus()
        QTest.qWait(10)
        QTest.keyClick(window, Qt.Key_F5)
        if window.thread is None and not errors:
            raise AssertionError('F5 did not start the calculation worker')
        wait_until(lambda: window.thread is None)
        settle(app)
    elapsed = 1000*(time.perf_counter()-started)
    timer.stop()
    if errors:
        raise AssertionError('\n'.join(errors))
    if not window.results:
        raise AssertionError('Worker finished without applied results')
    completed = stages['completed_slot_entered']
    worker_beats = [value for value in beats if started <= value < completed]
    return dict(worker_and_initial_render_ms=round(elapsed, 3),
                worker_until_completed_slot_ms=round(1000*(completed-started), 3),
                completion_to_settled_render_ms=round(elapsed-1000*(completed-started), 3),
                worker_qt_heartbeat_ms=stats(np.diff(worker_beats)*1000),
                qt_heartbeat_ms=stats(np.diff(beats)*1000),
                event_loop_ticks=len(beats), planes=len(window.results))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=Path('/tmp/injector-v07-benchmark.json'))
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    app = QApplication.instance() or QApplication([])
    app.setStyle('Fusion')
    original_path = ROOT/'examples/demo_project_user_field.json'
    synthetic_path = ROOT/'examples/demo_project_v07_confined_synthetic.json'
    hashes = {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
              for path in (original_path, synthetic_path)}
    report = dict(method=('Linux native Qt offscreen, real F5 QThread workers, '
        'QTest tab/slider controls, RGBA QWidget pixels, active 5-ms heartbeat; '
        'no UI timers disabled; synthetic assumptions are explicit; timings '
        'are measurements, not assertions about other computers'),
        projects_sha256=hashes, screenshots=[])

    print('Checking unchanged legacy example and explicit legacy mode…', flush=True)
    raw = json.loads(original_path.read_text())
    legacy_cfg, legacy_ns = project_from_json(raw)
    state = project_to_json(legacy_cfg, legacy_ns)
    implicit, explicit = [], []
    started = time.perf_counter()
    for z in legacy_cfg.z_mm:
        implicit.append(calculate_plane(legacy_ns, legacy_cfg, z))
    report['legacy_original_planes_ms'] = round(1000*(time.perf_counter()-started), 3)
    legacy_copy = deepcopy(legacy_cfg)
    legacy_copy.fuel_transport = {'mode': 'legacy'}
    for z in legacy_copy.z_mm:
        explicit.append(calculate_plane(legacy_ns, legacy_copy, z))
    for a, b in zip(implicit, explicit):
        np.testing.assert_array_equal(a.cell_mass_kg_s, b.cell_mass_kg_s)
    assert project_to_json(legacy_cfg, legacy_ns) == state
    report['legacy_implicit_equals_explicit_exactly'] = True

    window = MainWindow()
    window.resize(1480, 930)
    window.show()

    def screenshot(suffix, widget=window):
        path = args.out.with_name(args.out.stem+'-'+suffix+'.png')
        assert widget.grab().save(str(path))
        report['screenshots'].append(str(path))

    try:
        # Opt in on an in-memory copy of the original user project as well as
        # exercising the new curved demonstration.  No data are saved back.
        open_project(window, original_path)
        user_before = project_to_json(window.cfg, window.nozzles)
        window.open_transport_settings()
        user_dialog = window.transport_dialog
        settle(app)
        QTest.mouseClick(user_dialog.example_button, Qt.LeftButton)
        assert project_to_json(window.cfg, window.nozzles) == user_before
        QTest.mouseClick(user_dialog.apply_button, Qt.LeftButton)
        assert window.cfg.fuel_transport['provenance']['source_kind'] == 'synthetic'
        assert project_to_json(window.cfg, window.nozzles)['nozzles'] == user_before['nozzles']
        user_dialog.close()
        clear_transport_cache()
        clear_carrier_cache()
        print('Opting the unchanged user-project copy into explicit synthetic transport…', flush=True)
        report['user_copy_synthetic_first'] = worker_measurement(window, app)
        report['user_copy_new_mode_mass_residual_max'] = max(
            abs(plane.rel_balance_error) for plane in window.results)
        assert report['user_copy_new_mode_mass_residual_max'] < 1e-12
        report['user_copy_nozzles_unchanged'] = True

        open_project(window, synthetic_path)
        baseline = project_to_json(window.cfg, window.nozzles)
        # Native settings opening/apply must preserve all explicit values and
        # must not mark them as measured when the example provenance is kept.
        window.open_transport_settings()
        dialog = window.transport_dialog
        settle(app)
        assert dialog.mode.currentData() == 'confined_inertial'
        assert 'синтетические' in dialog.provenance.text().lower()
        screenshot('settings', dialog)
        QTest.mouseClick(dialog.apply_button, Qt.LeftButton)
        assert project_to_json(window.cfg, window.nozzles) == baseline
        dialog.close()
        clear_transport_cache()
        clear_carrier_cache()
        print('Computing exact synthetic 140-nozzle S-profile in the worker…', flush=True)
        report['confined_first'] = worker_measurement(window, app)
        snapshot = [plane.cell_mass_kg_s.copy() for plane in window.results]
        assert all(abs(plane.rel_balance_error) < 1e-12 for plane in window.results)
        assert all(abs(plane.inner_kg_s)+abs(plane.outer_kg_s) == 0. for plane in window.results)
        report['transport_cache_after_first'] = transport_cache_info()
        print('Repeating the same worker calculation with warm trajectories…', flush=True)
        report['confined_warm'] = worker_measurement(window, app)
        for plane, expected in zip(window.results, snapshot):
            np.testing.assert_array_equal(plane.cell_mass_kg_s, expected)

        warm_plane, warm_preview = [], []
        for _ in range(3):
            started = time.perf_counter()
            calculate_plane(window.nozzles, window.cfg, window.cfg.z_mm[-1])
            warm_plane.append(1000*(time.perf_counter()-started))
            started = time.perf_counter()
            evaluate_continuous_field(window.nozzles, window.cfg, window.cfg.z_mm[-1])
            warm_preview.append(1000*(time.perf_counter()-started))
        report['warm_last_plane_math_ms'] = stats(warm_plane)
        report['warm_last_plane_preview_math_ms'] = stats(warm_preview)
        # The upstream wider section provides a substantial visible field;
        # the 10-mm outlet annulus naturally occupies only a few pixels.
        window.plane_select.setCurrentIndex(2)
        window.tabs.setCurrentIndex(3)
        before = project_to_json(window.cfg, window.nozzles)
        canvases, cold, warm = {}, [], []
        bar = window.plots.tabBar()
        for cycle in range(6):
            for view in range(4):
                started = time.perf_counter()
                QTest.mouseClick(bar, Qt.LeftButton, pos=bar.tabRect(view).center())
                settle(app)
                (cold if cycle == 0 else warm).append(1000*(time.perf_counter()-started))
                canvas = window.plot_stacks[view].currentWidget()
                if cycle == 0:
                    canvases[view] = canvas
                else:
                    assert canvas is canvases[view], 'Warm navigation rebuilt an existing figure'
        assert project_to_json(window.cfg, window.nozzles) == before
        report['first_result_view_visits_ms'] = cold
        report['warm_result_view_switch_ms'] = stats(warm)
        window.plots.setCurrentIndex(0)
        settle(app)
        screenshot('matrix')
        window.plots.setCurrentIndex(1)
        settle(app)
        screenshot('field')
        window.plots.setCurrentIndex(3)
        settle(app)
        screenshot('epures')

        print('Computing the 140-step/two-collector quasi-static scan…', flush=True)
        window.open_settings()
        settle(app)
        QTest.mouseClick(window.quasi_check, Qt.LeftButton)
        assert window.cfg.quasistationary
        window.quasi_dialog.close()
        window.settings_page.close()
        report['quasi_worker'] = worker_measurement(window, app)
        assert window.quasi_result is not None
        window.plane_select.setCurrentIndex(min(2,len(window.results)-1))
        window.plots.setCurrentIndex(1)
        settle(app)
        canvas = window.smoothcanvas
        window.smoothcanvas.draw()
        before_pixels = pixels(canvas)
        arrays = [plane.cell_mass_kg_s.copy() for plane in window.results]
        native_frames = []
        full_draws = canvas.quasi_animator.draw_count
        for step in (35, 70, 105, 0, 35, 70, 105, 0):
            started = time.perf_counter()
            window.time_slider.setValue(step)
            wait_until(lambda: canvas._quasi_step == step, 5.)
            settle(app)
            native_frames.append(1000*(time.perf_counter()-started))
        window.time_slider.setValue(35)
        wait_until(lambda: canvas._quasi_step == 35, 5.)
        settle(app)
        after_pixels = pixels(canvas)
        changed = int(np.any(np.abs(before_pixels[:, :, :3].astype(int)
                                   -after_pixels[:, :, :3].astype(int)) > 2, axis=2).sum())
        assert changed > 100, 'Changing timestamp did not change the native field pixels'
        for plane, expected in zip(window.results, arrays):
            np.testing.assert_array_equal(plane.cell_mass_kg_s, expected)
        report['quasi_slider_ms'] = stats(native_frames)
        report['quasi_slider_changed_pixels'] = changed
        report['quasi_slider_full_draws'] = canvas.quasi_animator.draw_count-full_draws
        report['quasi_composite_navigation_unchanged'] = True
        screenshot('quasi-field')

        # The new mode is opt-in; both repository source files remain intact.
        assert all(hashlib.sha256((ROOT/name).read_bytes()).hexdigest() == digest
                   for name, digest in hashes.items())
        report['source_projects_unchanged'] = True
        args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
        print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    finally:
        if window.thread is not None:
            window.thread.requestInterruption()
            wait_until(lambda: window.thread is None)
        window.close()
        window.deleteLater()
        app.processEvents()


if __name__ == '__main__':
    main()
