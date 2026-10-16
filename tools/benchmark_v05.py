"""v0.5 worker/Qt heartbeat and reuse on the original user's 139 running sources."""
import argparse,json,sys,time,platform
from pathlib import Path
from dataclasses import replace
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import QTimer,Qt,QPoint
from PyQt5.QtTest import QTest
from matplotlib.backend_bases import MouseEvent,MouseButton
from fuelfield.gui import MainWindow
from fuelfield.core import project_from_json,project_to_json
from fuelfield.measurement import MeasurementSession,PressureSeries
from fuelfield import __version__
root=Path(__file__).resolve().parents[1];parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,required=True);args=parser.parse_args()
app=QApplication([]);w=MainWindow();w._restoring=True;w.cfg,w.nozzles=project_from_json(json.loads((root/'examples/demo_project_user_field.json').read_text()));w._fill_controls();w._fill_nozzles();w._reset_history();w.preview_timer.stop();w.selection_timer.stop();w.show()
def wait(panel):
 deadline=time.monotonic()+120
 while panel.thread is not None and time.monotonic()<deadline:QTest.qWait(5)
 assert panel.thread is None,'Worker timeout'
def stats(values):return dict(samples=len(values),median_ms=round(float(np.median(values)),3),p95_ms=round(float(np.percentile(values,95)),3),max_ms=round(float(max(values)),3))
w.run_calc();wait(w);before=[p.cell_mass_kg_s.copy() for p in w.results];identity=w.results;project_before=json.dumps(project_to_json(w.cfg,w.nozzles),ensure_ascii=False,sort_keys=True)
base=MeasurementSession.from_dict(json.loads((root/'examples/demo_session_v05_synthetic.json').read_text()))
session=replace(base,z_mm=30.,theta_zero_deg=0.,theta_direction=1,pressure_series=(PressureSeries('A',(0.,280.),(30.,30.)),),gas_series=(),nozzle_manifolds={},metadata={'source_kind':'synthetic','description':'Synthetic constant pressure/time/T protocol over unchanged user project; not new measured data'})
panel=w.measurement_panel;panel.set_session(session);w.tabs.setCurrentWidget(panel);beats=[];timer=QTimer();timer.setInterval(10);timer.timeout.connect(lambda:beats.append(time.perf_counter()))
timer.start();started=time.perf_counter();panel.run();wait(panel);first_wall=(time.perf_counter()-started)*1000;timer.stop();result=panel.result;assert result is not None,panel.status.text();operator=result.operator
p=w.results[2];expected=p.flux_kg_m2_s.reshape(-1)[result.cells];np.testing.assert_allclose(result.q_corrected,expected,rtol=3e-13,atol=1e-12)
started=time.perf_counter();panel.run();wait(panel);reuse_wall=(time.perf_counter()-started)*1000;assert panel.operator is operator
panel.canvas.draw();app.processEvents();controller=panel.interaction;ax=panel.figure.axes[0];hover=[]
for i in range(105):
 x=ax.get_xlim()[0]+(.1+.8*(i%100)/100)*np.ptp(ax.get_xlim());y=np.mean(ax.get_ylim());px,py=ax.transData.transform((x,y));point=QPoint(round(px/panel.canvas.device_pixel_ratio),round(panel.canvas.height()-py/panel.canvas.device_pixel_ratio))
 started=time.perf_counter();QTest.mouseMove(panel.canvas,point,delay=0);app.processEvents()
 if i>=5:hover.append((time.perf_counter()-started)*1000)
assert any(items[2].get_visible() for items in controller.hover.values()),'Real Qt hover did not reach the session epure'
started=time.perf_counter();panel.result_kind.setCurrentIndex(4);app.processEvents();snapshot_ms=(time.perf_counter()-started)*1000
panel.result_kind.setCurrentIndex(0);app.processEvents();previous=panel.result;started=time.perf_counter();panel.run();panel.cancel();wait(panel);cancel_ms=(time.perf_counter()-started)*1000;assert panel.result is previous
assert w.results is identity;assert json.dumps(project_to_json(w.cfg,w.nozzles),ensure_ascii=False,sort_keys=True)==project_before
for plane,old in zip(w.results,before):np.testing.assert_array_equal(plane.cell_mass_kg_s,old)
report=dict(version=__version__,python=platform.python_version(),platform=platform.system(),project='examples/demo_project_user_field.json',protocol='140 synthetic windows, 6 channels, constant physical Δp=30; source project unchanged',method='Linux offscreen; true QTest mouseMove delay=0 and Qt painting; no concurrent tests',first_engine_ms=round(result.elapsed_s*1000,3),first_worker_and_render_ms=round(first_wall,3),cached_engine_ms=round(panel.result.elapsed_s*1000,3),cached_worker_and_render_ms=round(reuse_wall,3),qt_heartbeat_gap_ms=stats(np.diff(beats)*1000) if len(beats)>1 else None,hover_six_dual_axes=stats(hover),instant_snapshot_render_ms=round(snapshot_ms,3),cancel_and_ui_check_ms=round(cancel_ms,3),max_static_field_difference=float(np.max(np.abs(result.q_corrected-expected))),max_snapshot_balance=result.manifest()['maximum_snapshot_balance_error'],original_static_balance=max(abs(p.rel_balance_error) for p in w.results),physics_unchanged=True)
args.out.parent.mkdir(parents=True,exist_ok=True);args.out.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n');print(json.dumps(report,ensure_ascii=False,indent=2));w.close()
