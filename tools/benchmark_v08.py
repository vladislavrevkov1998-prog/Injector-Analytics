"""Run a real Qt worker, air matrices, UI timings and optional evidence images.

Use an idle machine for comparable timing. Synthetic inputs only; no pass/fail
threshold pretends to be a physical or platform validation.
"""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import platform
import resource
import sys
import time
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from PyQt5.QtCore import QTimer,QT_VERSION_STR,PYQT_VERSION_STR
from PyQt5.QtWidgets import QApplication
from PyQt5.QtTest import QTest
import matplotlib
import scipy
from fuelfield.gui import MainWindow
from fuelfield.core import project_to_json,project_from_json
from fuelfield.air_gui import AirBalanceDialog
from fuelfield.air_results_gui import AirResultsDialog
from fuelfield.mixture_field import compute_mixture
from fuelfield.quasi import QuasiSettings,PressureChannel
from fuelfield.themes import apply_theme

ROOT=Path(__file__).resolve().parents[1]


def digest(planes):
    return hashlib.sha256(b''.join(np.asarray(p.cell_mass_kg_s).tobytes() for p in planes)).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True);parser.add_argument('--screens',type=Path)
    args=parser.parse_args();app=QApplication.instance() or QApplication([]);app.setStyle('Fusion')
    win=MainWindow();win.show();errors=[];win._error=lambda message:errors.append(str(message))
    raw=json.loads((ROOT/'examples/demo_project_v08_air_synthetic.json').read_text())
    win.apply_project_state(raw,'SYNTHETIC benchmark');win.tabs.setCurrentIndex(3)
    app.processEvents();beats=[];timer=QTimer();timer.setInterval(10);timer.timeout.connect(lambda:beats.append(time.perf_counter()))
    timer.start();start=time.perf_counter();win.run_calc();deadline=start+240.
    while win.thread is not None and time.perf_counter()<deadline:QTest.qWait(5)
    elapsed=time.perf_counter()-start;timer.stop();static_beats=beats.copy()
    if win.thread is not None:win.thread.requestInterruption();raise RuntimeError('Worker timeout')
    if len(win.results)!=len(win.cfg.z_mm):raise RuntimeError('\n'.join(errors) or win.statusBar().currentMessage())
    print('static_worker_seconds',elapsed,flush=True)
    snapshot=project_to_json(win.cfg,win.nozzles);before=digest(win.results)
    timings=[];air=[]
    for plane in win.results:
        start=time.perf_counter();air.append(compute_mixture(plane,win.cfg,win.nozzles));timings.append(time.perf_counter()-start)
    print('air_plane_seconds',timings,flush=True)
    editor=AirBalanceDialog(win.cfg,win.nozzles,win);editor.show();app.processEvents()
    moves=[];slider=editor.groups['hole'][0]
    for i in range(200):
        start=time.perf_counter();slider.setValue(2000+i);app.processEvents();moves.append(time.perf_counter()-start)
    print('group_slider_200_seconds',sum(moves),flush=True)
    result=AirResultsDialog(win.cfg,win.nozzles,win.results,parent=win);result.show()
    result.section.setCurrentIndex(len(win.results)-1);result.metric.setCurrentIndex(result.metric.findData('alpha_total'))
    theme_times={}
    for name in ('light','dark'):
        start=time.perf_counter();apply_theme(app,name,persist=False);app.processEvents()
        theme_times[name]=time.perf_counter()-start
        if args.screens:
            args.screens.mkdir(parents=True,exist_ok=True)
            for key,widget in [('main',win),('balance',editor),('alpha',result)]:
                widget.grab().save(str(args.screens/f'{key}-{name}.png'))
            win.plots.setCurrentIndex(3);app.processEvents();win.grab().save(str(args.screens/f'epures-{name}.png'))
            win.plots.setCurrentIndex(0);win.tabs.setCurrentIndex(0);app.processEvents()
            win.grab().save(str(args.screens/f'sketcher-{name}.png'));win.tabs.setCurrentIndex(3)
    if project_to_json(win.cfg,win.nozzles)!=snapshot or digest(win.results)!=before:
        raise AssertionError('UI modified physical inputs or completed fuel matrices')
    editor.close();result.close()
    cfg,nozzles=project_from_json(deepcopy(snapshot));cfg.quasistationary=True
    cfg.pressure_mode='calibrated'
    cfg.atomizer_types=[dict(id='SYNTHETIC_BENCH',name='SYNTHETIC Q(delta p), not measurements',
        kind='manual',points=[[25.,.9],[30.,1.],[35.,1.1]],provenance={'source_kind':'synthetic benchmark'})]
    for nozzle in nozzles:nozzle.atomizer_type_id='SYNTHETIC_BENCH'
    pressures=tuple(float(30.+5.*np.sin(2*np.pi*i/140)) for i in range(140))
    cfg.quasi_settings=QuasiSettings(channels=(PressureChannel('A',pressure=pressures),)).to_dict()
    win.apply_project_state(project_to_json(cfg,nozzles),'SYNTHETIC 140-step benchmark')
    beats.clear();timer.start();start=time.perf_counter();win.run_calc();deadline=start+240.
    while win.thread is not None and time.perf_counter()<deadline:QTest.qWait(5)
    quasi_elapsed=time.perf_counter()-start;timer.stop()
    if win.thread is not None:win.thread.requestInterruption();raise RuntimeError('Quasi worker timeout')
    if win.quasi_result is None:raise RuntimeError('\n'.join(errors) or win.statusBar().currentMessage())
    print('quasi_worker_seconds',quasi_elapsed,flush=True)
    result=AirResultsDialog(win.cfg,win.nozzles,win.results,win.quasi_result,win)
    result.show();app.processEvents();quasi_snapshot=project_to_json(win.cfg,win.nozzles);quasi_hash=digest(win.results)
    quasi_moves=[]
    for i in range(200):
        start=time.perf_counter();result.time_slider.setValue(i%140);result.refresh();app.processEvents()
        quasi_moves.append(time.perf_counter()-start)
    print('quasi_air_slider_200_seconds',sum(quasi_moves),flush=True)
    assert project_to_json(win.cfg,win.nozzles)==quasi_snapshot and digest(win.results)==quasi_hash
    # A separate 1000-source draft exercises the table and a real allocation edit.
    stress=deepcopy(cfg);stress.air_model['sources']=[dict(id=f'stress-{i}',kind='other',method='fraction',
        share=.0001,z_mm=0.,radius_mm=250.,provenance={'origin':'SYNTHETIC UI stress'}) for i in range(1000)]
    start=time.perf_counter();large=AirBalanceDialog(stress,nozzles,win);large.show();app.processEvents()
    large.groups['other'][0].setValue(2000);app.processEvents();stress_elapsed=time.perf_counter()-start
    assert large.table.rowCount()==1000 and large.balance is not None
    assert abs(large.balance.rel_error)<1e-12
    output=dict(python=sys.version,platform=platform.platform(),numpy=np.__version__,scipy=scipy.__version__,
        matplotlib=matplotlib.__version__,qt=QT_VERSION_STR,pyqt=PYQT_VERSION_STR,
        qt_platform=app.platformName(),device_pixel_ratio=win.devicePixelRatioF(),
        worker_140_nozzles_6x140_seconds=elapsed,worker_planes=len(win.results),heartbeat_count=len(static_beats),
        heartbeat_max_gap_seconds=float(np.diff(static_beats).max()) if len(static_beats)>1 else None,
        air_sources=len(win.cfg.air_model['sources']),air_seconds_per_plane=timings,
        air_matrix_balance_max_error=max(abs(x.air_total_kg_s.sum()-sum(s['crossed_kg_s'] for s in x.source_states)) for x in air),
        group_slider_events=200,group_slider_seconds_median=float(np.median(moves)),
        group_slider_seconds_p95=float(np.percentile(moves,95)),theme_seconds=theme_times,
        quasi_140_steps_worker_seconds=quasi_elapsed,quasi_heartbeat_count=len(beats),
        quasi_heartbeat_max_gap_seconds=float(np.diff(beats).max()) if len(beats)>1 else None,
        quasi_air_slider_events=200,quasi_air_slider_seconds_median=float(np.median(quasi_moves)),
        quasi_air_slider_seconds_p95=float(np.percentile(quasi_moves,95)),
        quasi_maximum_instantaneous_balance_error=win.quasi_result.maximum_balance_error,
        quasi_composites_unchanged_by_air_slider=True,
        thousand_source_editor_and_edit_seconds=stress_elapsed,
        peak_process_rss_mb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
        unchanged_fuel_sha256=before,ui_preserves_project=True,
        note='Synthetic model, Linux Qt offscreen; timings depend on concurrent load; not physical validation')
    args.out.parent.mkdir(parents=True,exist_ok=True);args.out.write_text(json.dumps(output,indent=2)+'\n')
    print(json.dumps(output,indent=2));large.close();result.close();win.close()


if __name__=='__main__':main()
