"""Reproduce v0.8.5 native Qt acceptance without importing the test suite.

Run acceptance, then refresh screenshots after palette changes. Run performance
separately on an idle machine; it is an observation, not an invented SLA::

    QT_QPA_PLATFORM=offscreen python tools/verify_v085_ui.py --phase acceptance --skip-screens
    QT_QPA_PLATFORM=offscreen python tools/verify_v085_ui.py --phase screenshots
    QT_QPA_PLATFORM=offscreen python tools/verify_v085_ui.py --phase performance

The source examples are read only. All changed inputs and generated files go to
--out. Acceptance uses the explicitly synthetic air example, 8 scan positions,
140 nozzles and two z planes. Performance preserves the real user's nozzle
geometry, sectors, fuel flows and type assignments, adding explicitly synthetic
air inputs and the existing synthetic 140-step two-collector pressure protocol.
Linux/offscreen evidence does not claim Windows or physical-model validation.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import resource
import sys
import time
import traceback
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

import matplotlib
import numpy as np
import scipy
from PyQt5.QtCore import Qt, QTimer, QThread, QPoint, QPointF, QEvent, QT_VERSION_STR, PYQT_VERSION_STR
from PyQt5.QtGui import QImage, QMouseEvent
from PyQt5.QtTest import QTest
from PyQt5.QtWidgets import QApplication, QFileDialog, QComboBox, QStyle, QStyleOptionSlider, QTabWidget, QScrollArea, QTableWidgetSelectionRange

from fuelfield import __version__
from fuelfield.air_gui import AirBalanceDialog, COLUMNS
from fuelfield.core import project_from_json, project_to_json
from fuelfield.gui import MainWindow, CalcWorker
from fuelfield.quasi import QuasiSettings, PressureChannel
from fuelfield.themes import THEME_LABELS, PALETTES, apply_theme
from fuelfield.unified_results import air_cell_values
import fuelfield.unified_results as unified


def sha_file(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def array_digest(window):
    """Hash every completed fuel/air/alpha array, including undefined masks."""
    digest = hashlib.sha256()
    arrays = [p.cell_mass_kg_s for p in window.results]
    if window.air_results:
        for item in window.air_results.mixtures:
            arrays.extend(getattr(item, key) for key in ('air_total_kg_s', 'air_core_kg_s', 'air_mixed_kg_s', 'alpha_total', 'alpha_mixed'))
    for array in arrays:
        data = np.ma.array(array)
        digest.update(str(data.shape).encode())
        digest.update(np.asarray(data.data).tobytes())
        digest.update(np.ma.getmaskarray(data).tobytes())
    return digest.hexdigest()


def stats(values):
    values = list(values)
    if not values:
        return dict(samples=0, median_ms=None, p95_ms=None, max_ms=None)
    return dict(samples=len(values), median_ms=float(np.median(values)),
                p95_ms=float(np.percentile(values, 95)), max_ms=float(max(values)))


class Proof:
    def __init__(self, args):
        self.args = args
        args.out.mkdir(parents=True, exist_ok=True)
        # Preference writes belong to this run, never the developer's settings.
        os.environ['XDG_CONFIG_HOME'] = str(args.out / 'settings')
        self.app = QApplication.instance() or QApplication([])
        self.app.setStyle('Fusion')
        self.window = MainWindow()
        self.window.resize(1480, 1060)
        self.window.setWindowTitle('Injector Analytics v' + __version__ + ' · SYNTHETIC UI proof — not measurements')
        self.window.show()
        self.errors = []
        self.callback_exceptions = []
        self.previous_excepthook = sys.excepthook
        def callback_exception(kind, value, tb):
            message = ''.join(traceback.format_exception(kind,value,tb))
            self.callback_exceptions.append(message)
            print(message,file=sys.stderr,flush=True)
        sys.excepthook = callback_exception
        self.window._error = lambda message: self.errors.append(str(message))
        self.source_paths = [ROOT / 'examples/demo_project_user_field.json', ROOT / 'examples/demo_project_v08_air_synthetic.json', ROOT / 'examples/demo_user_field_140_two_collectors.quasi.json']
        self.source_hashes = {str(path.relative_to(ROOT)): sha_file(path) for path in self.source_paths}
        self.report = dict(version=__version__, phase=args.phase,
            captured_utc=datetime.now(timezone.utc).isoformat(),
            timing_context=args.timing_context,
            tool_sha256=sha_file(Path(__file__)),
            runtime_source_sha256={path.name:sha_file(path) for path in sorted((ROOT/'src/fuelfield').glob('*.py'))},
            environment=dict(python=platform.python_version(), platform=platform.platform(),
                numpy=np.__version__, scipy=scipy.__version__, matplotlib=matplotlib.__version__,
                pyqt=PYQT_VERSION_STR, qt=QT_VERSION_STR, qpa=self.app.platformName(),
                device_pixel_ratio=self.window.devicePixelRatioF()),
            method='Real MainWindow, F5 QAction and CalcWorker QThread; native Qt event loop; no test imports; timings are observations without SLA thresholds.',
            limitations=['Linux offscreen only; Windows/Spyder NOT RUN', 'Air and pressure overrides are explicitly synthetic; no physical-model validation'],
            checks={}, screenshots=[])

    def settle(self, delay=25):
        self.app.processEvents()
        QTest.qWait(delay)

    def record(self, name, callback):
        started = time.perf_counter()
        print('START', name, flush=True)
        try:
            details = callback() or {}
            self.report['checks'][name] = dict(pass_check=True, elapsed_s=time.perf_counter()-started, **details)
        except Exception as error:
            self.report['checks'][name] = dict(pass_check=False, elapsed_s=time.perf_counter()-started,
                error=str(error), traceback=traceback.format_exc())
            print(traceback.format_exc(), flush=True)
        self.save()
        print('END', name, self.report['checks'][name]['pass_check'], flush=True)

    def save(self):
        self.report['errors'] = self.errors
        self.report['unexpected_qt_callback_exceptions'] = self.callback_exceptions
        destination = self.args.out / ('report-' + self.args.phase + '.json')
        destination.write_text(json.dumps(self.report, ensure_ascii=False, indent=2, allow_nan=False) + '\n')

    def inputs(self, quasi=False, performance=False):
        source_path = ROOT / ('examples/demo_project_user_field.json' if performance else 'examples/demo_project_v08_air_synthetic.json')
        cfg, nozzles = project_from_json(json.loads(source_path.read_text()))
        original_nozzles = project_to_json(cfg, nozzles)['nozzles']
        cfg.z_mm = [30., 60.]
        cfg.preview_resolution = 200
        cfg.n_radial = 6
        cfg.n_angular = 140 if performance else 8
        cfg.quasistationary = quasi
        if performance:
            # Preserve all measured fuel/sector/type data. These air values are
            # independent UI loads, not inferred measurements or nozzle flows.
            cfg.air_model = dict(enabled=True, total_kg_s=1., l0=15., distribution='directed',
                interaction_strength=0., max_path_mm=80., step_mm=4.,
                provenance={'origin': 'SYNTHETIC v0.8.5 UI performance input; not measured'},
                sources=[dict(id='SYNTHETIC_S' + str(n.id), kind='swirler', nozzle_id=n.id,
                    method='fixed', flow_kg_s=.005, direction=[1., .04, .10],
                    sigma_r_mm=12., sigma_theta_deg=5., spread_mm_per_mm=.04,
                    mixed_fraction=.2, mixing_length_mm=60., confidence='synthetic') for n in nozzles])
            mode = json.loads((ROOT / 'examples/demo_user_field_140_two_collectors.quasi.json').read_text())
            for nozzle in nozzles:
                nozzle.manifold_id = mode['nozzle_manifolds'][str(nozzle.id)]
            cfg.quasi_settings = mode
            before = [{k:v for k,v in n.items() if k != 'manifold_id'} for n in original_nozzles]
            after = [{k:v for k,v in n.items() if k != 'manifold_id'} for n in project_to_json(cfg,nozzles)['nozzles']]
            assert before == after, 'Real user nozzle geometry/flows/sectors/types changed'
        else:
            cfg.air_model.update(max_path_mm=80., step_mm=4., interaction_strength=0.)
            cfg.air_model['provenance'] = {'origin': 'SYNTHETIC v0.8.5 UI acceptance; 8 comb positions and two z planes; not measurements'}
            template = dict(method='fixed', flow_kg_s=.003, direction=[1., .04, .10],
                sigma_r_mm=12., sigma_theta_deg=5., spread_mm_per_mm=.04,
                mixed_fraction=.2, mixing_length_mm=60., confidence='synthetic')
            for preset in cfg.nozzle_presets.values():
                preset['air'] = deepcopy(template)
            for source in cfg.air_model['sources']:
                if source['kind'] == 'swirler':
                    source.update(template_mode='inherit', air_overrides=[])
            if quasi:
                cfg.pressure_mode = 'calibrated'
                cfg.atomizer_types = [dict(id='SYNTHETIC_UI_Q', name='SYNTHETIC UI Q(Δp); not measurements',
                    kind='manual', points=[[15.,.75],[30.,1.],[60.,1.4]], provenance={'source_kind':'synthetic UI acceptance'})]
                for nozzle in nozzles:
                    nozzle.atomizer_type_id = 'SYNTHETIC_UI_Q'
                    nozzle.manifold_id = 'A' if nozzle.id <= 70 else 'B'
                cfg.quasi_settings = QuasiSettings(8, .25, 5., 0., 1, 'A',
                    (PressureChannel('A', tuple(np.linspace(24.,38.,8))),
                     PressureChannel('B', tuple(np.linspace(35.,22.,8)))),
                    metadata={'origin':'SYNTHETIC two-collector UI acceptance; not measurements'}).to_dict()
        state = project_to_json(cfg, nozzles)
        fixture = self.args.out / ('synthetic-input-' + ('140' if performance else '8') + ('-quasi' if quasi else '-stationary') + '.json')
        fixture.write_text(json.dumps(state, ensure_ascii=False, indent=2) + '\n')
        return state, dict(source=str(source_path.relative_to(ROOT)), synthetic_input=str(fixture),
            overrides='two z=[30,60] mm; 200px preview; explicit synthetic air with capped 80mm paths/4mm step, no interaction; 6 radial levels; scan/pressure inputs synthetic',
            nozzle_count=len(nozzles), air_source_count=len(cfg.air_model['sources']),
            angular_positions=cfg.n_angular, z_mm=cfg.z_mm,
            user_nozzle_geometry_flows_sectors_types_preserved=True if performance else None)

    def f5(self, quasi=False, performance=False):
        window = self.window
        state, description = self.inputs(quasi, performance)
        window.apply_project_state(state, 'SYNTHETIC acceptance input')
        window.preview_timer.stop()
        window.selection_timer.stop()
        window.tabs.setCurrentIndex(3)
        window.metric.setCurrentIndex(window.metric.findData('alpha_total'))
        self.settle()
        assert window.thread is None
        beats = [time.perf_counter()]
        timer = QTimer(); timer.setInterval(10); timer.timeout.connect(lambda: beats.append(time.perf_counter()))
        threads = []
        old_run = CalcWorker.run
        def observed_run(worker):
            threads.append(QThread.currentThread() is not self.app.thread())
            return old_run(worker)
        errors_before = len(self.errors)
        with patch.object(CalcWorker, 'run', observed_run), patch.object(unified, 'compute_air_field', wraps=unified.compute_air_field) as compute:
            timer.start(); started = time.perf_counter()
            QTest.keyClick(window, Qt.Key_F5)
            assert window.thread is not None, 'F5 QAction did not launch a real worker'
            deadline = time.monotonic() + self.args.timeout
            while window.thread is not None and time.monotonic() < deadline:
                QTest.qWait(5)
            elapsed = time.perf_counter()-started
            timer.stop(); beats.append(time.perf_counter())
            if window.thread is not None:
                window.thread.requestInterruption()
                raise RuntimeError('F5 worker timeout; interruption requested')
            calls = compute.call_count
        assert len(self.errors) == errors_before, self.errors[errors_before:]
        assert threads == [True], 'CalcWorker did not execute on its actual QThread'
        assert len(window.results) == 2 and window.air_results is not None
        assert bool(window.quasi_result is not None) == quasi
        assert calls == 2, 'Static air must be computed once per z in the joint worker'
        assert all(p.cell_mass_kg_s.shape == (6, 140 if performance else 8) for p in window.results)
        assert all(np.isfinite(p.cell_mass_kg_s).all() for p in window.results)
        assert project_to_json(window.cfg,window.nozzles)==state, 'F5 changed its source inputs'
        description.update(f5_worker_and_initial_render_ms=elapsed*1000,
            qt_heartbeat_gap_ms=stats(np.diff(beats)*1000), worker_on_qthread=True,
            atomic_fuel_and_air=True, air_compute_calls=calls, source_inputs_unchanged=project_to_json(window.cfg,window.nozzles)==state,
            physical_arrays_sha256=array_digest(window), peak_rss_mb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)
        if quasi:
            description.update(quasi_steps=window.quasi_result.settings.n_steps,
                fuel_engine_ms=window.quasi_result.elapsed_s*1000,
                maximum_fuel_balance_error=window.quasi_result.maximum_balance_error)
        return description

    def metrics_and_exports(self):
        window = self.window
        result = window.quasi_result
        assert result is not None
        snapshot = project_to_json(window.cfg,window.nozzles)
        original = array_digest(window)
        window.plane_select.setCurrentIndex(1)
        window.metric.setCurrentIndex(window.metric.findData('alpha_total'))
        window.plots.setCurrentIndex(0); self.settle()
        assert window.figmap.axes[0].name == 'polar'
        rings = [line for line in window.figmap.axes[0].lines if line.get_gid() == 'nozzle-row-ring']
        assert {float(line.get_ydata()[0]) for line in rings} == {n.radius_mm for n in window.nozzles}
        comb = next(item for item in window.figmap.axes[0].collections if item.get_gid() == 'comb-pass-positions')
        np.testing.assert_allclose(np.rad2deg(comb.get_offsets()[:,0]), result.settings.global_angles(window.cfg))
        window.plots.setCurrentIndex(3); self.settle()
        assert window.figcurves.axes and all(ax.get_ylabel() == '1' for ax in window.figcurves.axes)
        exports = {}
        for name, view in [('instantaneous',1), ('composite',0)]:
            window.plots.setCurrentIndex(view)
            window.time_slider.setValue(result.settings.n_steps-1)
            self.settle(40)
            path = self.args.out / ('air-' + name + '-' + str(result.settings.n_steps) + '.csv')
            with patch.object(QFileDialog, 'getSaveFileName', return_value=(str(path),'')):
                window.export_results()
            with path.open(encoding='utf-8-sig') as stream:
                rows = list(csv.DictReader(stream, delimiter=';'))
            assert len(rows) == len(window.results)*window.cfg.n_radial*window.cfg.n_angular
            for row in rows:
                plane_index = window.cfg.z_mm.index(float(row['z_mm']))
                column, radial = int(row['theta_node'])-1, int(row['radial_level'])-1
                step = result.settings.n_steps-1 if view==1 else int(window.results[plane_index].scan_steps[column])
                assert float(row['time_s']) == float(result.settings.times[step])
                assert json.loads(row['pressures_kgf_cm2_json']) == {key:float(values[step]) for key,values in result.pressures.items()}
                plane, mixture = window.air_results.state(plane_index, step if view==1 else None)
                expected = mixture.alpha_total[radial,column]
                assert (row['alpha_total']=='') == (not np.isfinite(expected))
                if np.isfinite(expected):
                    np.testing.assert_allclose(float(row['alpha_total']), expected, rtol=1e-13)
                assert row['scope'] == ('instantaneous' if view==1 else 'composite_different_times_no_global_balance')
            if view==0:
                section_path = path.with_name(path.stem+'_Air_sections.csv')
                with section_path.open(encoding='utf-8-sig') as stream:
                    sections = list(csv.DictReader(stream,delimiter=';'))
                assert all(not row['alpha_section_total'] and not row['alpha_section_mixed'] for row in sections)
            exports[name] = dict(path=str(path), cells_verified=len(rows), actual_time_and_pressure=True,
                no_fictitious_composite_section_alpha=view==0)
        assert project_to_json(window.cfg,window.nozzles) == snapshot
        assert array_digest(window) == original
        assert not hasattr(window,'air_results_button') and getattr(window,'air_results_dialog',None) is None
        assert not any(type(widget).__name__ == 'AirResultsDialog' and widget.isVisible() for widget in self.app.topLevelWidgets())
        return dict(exports=exports, physical_arrays_unchanged=True, input_project_unchanged=True,
            circular_alpha=True, epures_alpha=True, true_nozzle_rings_and_comb_angles=True, separate_air_results_window=False)

    def slider(self):
        window = self.window
        result = window.quasi_result
        snapshot = project_to_json(window.cfg,window.nozzles)
        original = array_digest(window)
        fields = tuple(id(field) for field in window.air_results.air_fields)
        window.metric.setCurrentIndex(window.metric.findData('alpha_total'))
        window.plots.setCurrentIndex(1)
        window.time_slider.setValue(0); self.settle(50)
        canvas = window.smoothcanvas
        mesh = canvas._air_mesh
        first = np.ma.array(mesh.get_array(),copy=True)
        canvas.draw(); before_pixels = np.asarray(canvas.buffer_rgba()).copy()
        beats = [time.perf_counter()]
        timer = QTimer(); timer.setInterval(10); timer.timeout.connect(lambda:beats.append(time.perf_counter()))
        updates, value_events = [], []
        window.time_slider.valueChanged.connect(value_events.append)
        old_set_array = mesh.set_array
        def observe(values):
            updates.append(time.perf_counter())
            return old_set_array(values)
        with patch.object(unified,'compute_air_field',wraps=unified.compute_air_field) as compute, patch.object(mesh,'set_array',observe):
            timer.start(); started = time.perf_counter()
            # A burst verifies the 16ms timer coalesces changing values.
            for i in range(200):
                window.time_slider.setValue((i*13)%result.settings.n_steps)
            burst_pending = window.time_timer.isActive()
            burst_updates_before_wait = len(updates)
            self.settle(40)
            burst_updates = len(updates)
            # Native held-button events verify intermediate frames and release.
            window.time_slider.setValue(0); self.settle(35)
            slider = window.time_slider
            option = QStyleOptionSlider(); slider.initStyleOption(option)
            handle = slider.style().subControlRect(QStyle.CC_Slider,option,QStyle.SC_SliderHandle,slider).center()
            drag_initial = len(updates)
            QTest.mousePress(slider,Qt.LeftButton,Qt.NoModifier,handle)
            for x in np.linspace(handle.x(),slider.width()-8,200):
                event = QMouseEvent(QEvent.MouseMove,QPointF(float(x),slider.height()/2),Qt.NoButton,Qt.LeftButton,Qt.NoModifier)
                QApplication.sendEvent(slider,event)
                QTest.qWait(1)
            QTest.mouseRelease(slider,Qt.LeftButton,Qt.NoModifier,QPoint(slider.width()-8,slider.height()//2))
            self.settle(50)
            elapsed = time.perf_counter()-started
            timer.stop(); beats.append(time.perf_counter())
            drag_updates = len(updates)-drag_initial
            assert compute.call_count == 0, 'Slider retraced/recomputed stationary air'
        assert burst_pending and burst_updates_before_wait == 0 and burst_updates <= 1
        assert drag_updates > 1, 'No intermediate frames during native drag'
        assert canvas is window.smoothcanvas and mesh is window.smoothcanvas._air_mesh
        assert canvas._air_step == window.time_slider.value() == result.settings.n_steps-1
        expected = air_cell_values(*window.air_results.state(window.plane_select.currentIndex(),canvas._air_step),'alpha_total')
        np.testing.assert_allclose(np.ma.filled(mesh.get_array(),np.nan).reshape(expected.shape),np.ma.filled(expected,np.nan),equal_nan=True)
        assert not np.ma.allequal(first,mesh.get_array())
        canvas.draw(); after_pixels = np.asarray(canvas.buffer_rgba()).copy()
        assert np.count_nonzero(before_pixels != after_pixels)>0
        x0,y0,x1,y1 = map(int,canvas._air_axes.bbox.extents)
        height = before_pixels.shape[0]
        changed_plot = int(np.count_nonzero(before_pixels[height-y1:height-y0,x0:x1] != after_pixels[height-y1:height-y0,x0:x1]))
        assert changed_plot>0, 'Rendered alpha cells did not change inside the plot'
        assert tuple(id(field) for field in window.air_results.air_fields)==fields
        assert array_digest(window)==original and project_to_json(window.cfg,window.nozzles)==snapshot
        return dict(native_drag_motion_events=200, burst_value_assignments=200, value_change_events=len(value_events),
            burst_updates_before_timer=burst_updates_before_wait, burst_updates_after_timer=burst_updates,
            displayed_drag_updates=drag_updates, displayed_total_updates=len(updates), elapsed_ms=elapsed*1000,
            qt_heartbeat_gap_ms=stats(np.diff(beats)*1000), intervals_between_updates_ms=stats(np.diff(updates)*1000),
            static_air_compute_calls=0, same_static_air_objects=True, same_mesh_and_canvas=True,
            final_step=canvas._air_step, final_mesh_matches_physics=True, changed_rgba=int(np.count_nonzero(before_pixels!=after_pixels)),
            changed_rgba_inside_plot=changed_plot,
            physical_arrays_unchanged=True, input_project_unchanged=True,
            note='8-step acceptance has only 8 unique slider positions; 140-step performance exercises the full scan.')

    def balance_table(self):
        window = self.window
        original = project_to_json(window.cfg,window.nozzles)
        dialog = AirBalanceDialog(window.cfg,window.nozzles,window)
        dialog.show(); self.settle()
        table = dialog.table
        columns = {key:i for i,(key,_,_) in enumerate(COLUMNS)}
        col = columns['sigma_r_mm']
        table.setCurrentCell(0,col)
        table.setRangeSelected(QTableWidgetSelectionRange(0,col,2,col),True)
        self.app.clipboard().setText('13.25')
        QTest.keyClick(table,Qt.Key_V,Qt.ControlModifier); self.settle(60)
        assert all(table.item(row,col).text()=='13.25' for row in range(3))
        method_col = columns['method']
        table.clearSelection(); table.setCurrentCell(0,method_col)
        table.scrollToItem(table.item(0,method_col)); self.settle()
        table.editItem(table.item(0,method_col)); self.settle()
        editors = [box for box in table.findChildren(QComboBox) if box.isVisible()]
        assert editors and {editors[-1].itemData(i) for i in range(editors[-1].count())} >= {'fixed','fraction','unknown','area','effective_area'}
        QTest.keyClick(editors[-1],Qt.Key_Escape)
        dialog.reject(); self.settle()
        assert project_to_json(window.cfg,window.nozzles)==original
        return dict(table_rows=table.rowCount(), selected_cells_pasted=3, actual_ctrl_v=True, actual_cell_dropdown=True, cancelled_draft_preserves_project=True)

    def geometry_freeze(self):
        window = self.window
        window.tabs.setCurrentIndex(0); window.geometry_pages.setCurrentIndex(1)
        editor = window.geometry_editor
        editor.mode_selector.setCurrentIndex(0); self.settle()
        assert editor.sketch.property('ffSketchMode') == 'geometry'
        assert '#28a35d' in editor.sketch.styleSheet()
        candidate = project_to_json(window.cfg,window.nozzles)
        candidate['settings']['r_out_mm'] += 1.
        window.apply_project_state(candidate,'SYNTHETIC reversible geometry history')
        editor.mode_selector.setCurrentIndex(1); self.settle()
        assert editor.is_air_mode and editor.sketch.property('ffSketchMode')=='air'
        assert '#2c85dc' in editor.sketch.styleSheet()
        frozen = project_to_json(window.cfg,window.nozzles)
        index = window.undo_stack.index()
        for tab in (0,1,2):
            window.tabs.setCurrentIndex(tab); self.settle()
            assert not window.undo_action.isEnabled() and not window.redo_action.isEnabled()
            QTest.keyClick(window,Qt.Key_Z,Qt.ControlModifier)
            QTest.keyClick(window,Qt.Key_Y,Qt.ControlModifier)
            assert window.undo_stack.index()==index
            assert project_to_json(window.cfg,window.nozzles)==frozen
        rejected = deepcopy(frozen); rejected['nozzles'][0]['radius_mm'] += 1.
        window.apply_project_state(rejected,'SYNTHETIC forbidden geometry')
        assert project_to_json(window.cfg,window.nozzles)==frozen
        window.tabs.setCurrentIndex(0)
        panel = editor.air_panel
        panel.source_selector.setCurrentIndex(panel.source_selector.findData('H1'))
        panel.show_all.setChecked(False)
        self.settle(75)
        before = [item.path().boundingRect() for item in panel.items if item.data(1)=='air-envelope']
        control = panel.controls['sigma_r_mm']
        control.setFocus(); control.selectAll(); QTest.keyClicks(control,'19.5'); QTest.keyClick(control,Qt.Key_Return)
        self.settle(125)
        after = [item.path().boundingRect() for item in panel.items if item.data(1)=='air-envelope']
        assert before and after and before!=after, 'Dashed envelopes did not update after air input'
        assert project_to_json(window.cfg,window.nozzles)==frozen, 'Unapplied air draft changed project'
        assert all(table.editTriggers()==table.NoEditTriggers for table in editor.tables)
        editor.mode_selector.setCurrentIndex(0); self.settle()
        assert all(table.editTriggers()!=table.NoEditTriggers for table in editor.tables)
        return dict(geometry_border='#28a35d', air_border='#2c85dc', geometry_frozen=True,
            project_undo_redo_blocked_tabs=[1,2,3], coordinate_mutation_rejected=True,
            air_draft_updates_dashed_geometry=True, unapplied_draft_preserves_project=True,
            geometry_editing_restored_after_mode_switch=True)

    def capture(self, widget, name):
        self.settle(40)
        image = widget.grab().toImage().convertToFormat(QImage.Format_RGBA8888)
        path = self.args.out / (name+'.png')
        assert image.save(str(path))
        self.report['screenshots'].append(dict(path=str(path),width=image.width(),height=image.height(),sha256=sha_file(path)))

    def screenshots(self):
        window = self.window
        before, inputs = array_digest(window), project_to_json(window.cfg,window.nozzles)
        editor = window.geometry_editor
        timings = {}
        for theme in THEME_LABELS:
            started = time.perf_counter(); apply_theme(self.app,theme,persist=False); self.settle(75)
            timings[theme] = (time.perf_counter()-started)*1000
            editor.mode_selector.setCurrentIndex(0)
            window.tabs.setCurrentIndex(3)
            window.metric.setCurrentIndex(window.metric.findData('alpha_total'))
            window.plots.setCurrentIndex(0); self.capture(window,'04-alpha-circular-'+theme)
            window.plots.setCurrentIndex(3); self.capture(window,'04-alpha-epures-'+theme)
            dialog = AirBalanceDialog(window.cfg,window.nozzles,window)
            dialog.show(); self.capture(dialog,'air-balance-table-'+theme); dialog.close()
            window.tabs.setCurrentIndex(0); window.geometry_pages.setCurrentIndex(0)
            # Reveal the actual templates page inside its enclosing scroll area.
            parent = window.preset_editor.parentWidget()
            while parent is not None:
                if isinstance(parent,QTabWidget) and parent.indexOf(window.preset_editor)>=0:
                    parent.setCurrentWidget(window.preset_editor)
                parent = parent.parentWidget()
            for medium in (0,1):
                window.preset_editor.medium_tabs.setCurrentIndex(medium)
                scroll = window.preset_editor.parentWidget()
                while scroll is not None and not isinstance(scroll,QScrollArea):scroll=scroll.parentWidget()
                if scroll:scroll.verticalScrollBar().setValue(scroll.verticalScrollBar().maximum())
                assert window.preset_editor.isVisible()
                self.capture(window,'01-nozzle-template-'+('fuel' if medium==0 else 'air')+'-'+theme)
            window.geometry_pages.setCurrentIndex(1)
            for mode in (0,1):
                editor.mode_selector.setCurrentIndex(mode)
                if mode:
                    panel=editor.air_panel
                    panel.source_selector.setCurrentIndex(panel.source_selector.findData('H1'))
                    panel.show_all.setChecked(False)
                for angular in (False,True):
                    editor.angular.setChecked(angular);editor.fit()
                    self.capture(window,'sketcher-'+('air' if mode else 'geometry')+'-'+('circumferential' if angular else 'meridional')+'-'+theme)
        editor.mode_selector.setCurrentIndex(0)
        assert array_digest(window)==before
        assert project_to_json(window.cfg,window.nozzles)==inputs
        return dict(theme_switch_ms=timings,themes=list(THEME_LABELS),palette_sha256=hashlib.sha256(json.dumps(PALETTES,sort_keys=True).encode()).hexdigest(),
            physical_arrays_unchanged=True,input_project_unchanged=True,screenshot_count=len(self.report['screenshots']))

    def finish(self):
        if self.window.thread is not None:
            self.window.thread.requestInterruption()
            deadline=time.monotonic()+60.
            while self.window.thread is not None and time.monotonic()<deadline:
                QTest.qWait(5)
        if self.window.thread is None:
            self.window.close();self.window.deleteLater();self.app.processEvents()
        changed = {str(path.relative_to(ROOT)):sha_file(path) for path in self.source_paths}
        same = changed==self.source_hashes
        self.report['checks']['source_files_read_only'] = dict(pass_check=same,sha256_before=self.source_hashes,sha256_after=changed)
        self.report['pass_check'] = all(check['pass_check'] for check in self.report['checks'].values()) and not self.errors and not self.callback_exceptions
        self.save()
        sys.excepthook = self.previous_excepthook
        print(json.dumps(self.report,ensure_ascii=False,indent=2,allow_nan=False),flush=True)
        return 0 if self.report['pass_check'] else 1


def main():
    parser=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--phase',choices=('acceptance','screenshots','performance'),default='acceptance')
    parser.add_argument('--out',type=Path,default=Path('/workspace/artifacts/v085/ui-proof'))
    parser.add_argument('--timeout',type=float,default=240.,help='Operational worker timeout, not a performance SLA')
    parser.add_argument('--skip-screens',action='store_true')
    parser.add_argument('--timing-context',default='External process load unspecified; timings are observations, not a benchmark comparison.')
    args=parser.parse_args()
    proof=Proof(args)
    if args.phase=='acceptance':
        proof.record('stationary_f5',lambda:proof.f5())
        proof.record('quasi_8_f5',lambda:proof.f5(quasi=True))
        proof.record('main_metrics_and_csv_export',proof.metrics_and_exports)
        proof.record('native_time_slider',proof.slider)
        proof.record('air_balance_native_editing',proof.balance_table)
        if not args.skip_screens:proof.record('three_theme_screenshots',proof.screenshots)
        proof.record('sketcher_air_geometry_freeze',proof.geometry_freeze)
    elif args.phase=='screenshots':
        proof.record('quasi_8_f5',lambda:proof.f5(quasi=True))
        proof.record('three_theme_screenshots',proof.screenshots)
    else:
        proof.record('stationary_140_f5',lambda:proof.f5(performance=True))
        proof.record('quasi_140_f5',lambda:proof.f5(quasi=True,performance=True))
        proof.record('main_metrics_and_csv_export',proof.metrics_and_exports)
        proof.record('native_time_slider',proof.slider)
    return proof.finish()


if __name__=='__main__':
    sys.exit(main())
