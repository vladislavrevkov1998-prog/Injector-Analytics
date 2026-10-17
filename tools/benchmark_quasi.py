"""Actual user project, two collectors and 140 steps: worker, slider and fidelity."""
import sys,json,time,hashlib,platform,resource,argparse
from pathlib import Path
from copy import deepcopy
from dataclasses import replace
from unittest.mock import patch
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from PyQt5.QtWidgets import QApplication,QStyle,QStyleOptionSlider
from PyQt5.QtCore import Qt,QTimer,QPoint,QPointF,QEvent
from PyQt5.QtGui import QMouseEvent
from PyQt5.QtTest import QTest
from fuelfield.gui import MainWindow
from fuelfield.core import project_to_json,calculate_plane,evaluate_continuous_field
from fuelfield import __version__
root=Path(__file__).resolve().parents[1];parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,required=True);args=parser.parse_args()
app=QApplication([]);app.setStyle('Fusion');w=MainWindow();w.show();source=root/'examples/demo_project_user_field.json';original=hashlib.sha256(source.read_bytes()).hexdigest()
with patch('fuelfield.gui.QFileDialog.getOpenFileName',return_value=(str(source),'')):w.open_project()
w.preview_timer.stop();w.selection_timer.stop();w.quasi_check.setChecked(True);d=w.quasi_dialog;d.load_path(root/'examples/demo_user_field_140_two_collectors.quasi.json');d.apply();d.close()
def wait(owner):
 deadline=time.monotonic()+180
 while owner.thread is not None and time.monotonic()<deadline:QTest.qWait(5)
 assert owner.thread is None,'Worker timeout'
def stats(a):return dict(samples=len(a),median_ms=round(float(np.median(a)),3),p95_ms=round(float(np.percentile(a,95)),3),max_ms=round(float(max(a)),3))
beats=[];timer=QTimer();timer.setInterval(10);timer.timeout.connect(lambda:beats.append(time.perf_counter()));timer.start();started=time.perf_counter();w.run_calc();wait(w);wall=(time.perf_counter()-started)*1000;timer.stop();result=w.quasi_result;assert result is not None
w.plane_select.setCurrentIndex(2);w.plots.setCurrentIndex(1);app.processEvents();QTest.qWait(60);w.smoothcanvas.draw();QTest.qWait(30);canvas=w.smoothcanvas;anim=canvas.quasi_animator;draws=anim.draw_count;identity=w.results;arrays=[p.cell_mass_kg_s.copy() for p in w.results];snapshot=json.dumps(project_to_json(w.cfg,w.nozzles),sort_keys=True);operator=result.operators[2].preview_weights;values=[]
for i in range(145):
 step=(i*13)%140;w.time_slider.blockSignals(True);w.time_slider.setValue(step);w.time_slider.blockSignals(False);started=time.perf_counter();w._update_time_field();app.processEvents()
 if i>=5:values.append((time.perf_counter()-started)*1000)
callback=stats(values);full_draws=anim.draw_count-draws;values=[]
for i in range(65):
 key=Qt.Key_Right if w.time_slider.value()<139 else Qt.Key_Home;expected=w.time_slider.value()+1 if key==Qt.Key_Right else 0;started=time.perf_counter();QTest.keyClick(w.time_slider,key)
 deadline=time.monotonic()+2
 while canvas._quasi_step!=expected and time.monotonic()<deadline:QTest.qWait(1)
 assert canvas._quasi_step==expected,'Qt slider did not display selected timestamp'
 if i>=5:values.append((time.perf_counter()-started)*1000)
real_qt=stats(values)
# QTest press/release with held-button Qt moves (offscreen mouseMove omits buttons).
slider=w.time_slider;opt=QStyleOptionSlider();slider.initStyleOption(opt);handle=slider.style().subControlRect(QStyle.CC_Slider,opt,QStyle.SC_SliderHandle,slider).center();before_updates=anim.update_count
started=time.perf_counter();QTest.mousePress(slider,Qt.LeftButton,Qt.NoModifier,handle)
for x in np.linspace(10,slider.width()-8,100):
 event=QMouseEvent(QEvent.MouseMove,QPointF(float(x),slider.height()/2),Qt.NoButton,Qt.LeftButton,Qt.NoModifier)
 QApplication.sendEvent(slider,event);QTest.qWait(1)
QTest.mouseRelease(slider,Qt.LeftButton,Qt.NoModifier,QPoint(slider.width()-8,slider.height()//2));QTest.qWait(25);drag_ms=(time.perf_counter()-started)*1000;assert slider.value()>=137,(slider.value(),slider.size());assert canvas._quasi_step==slider.value();assert anim.update_count-before_updates>1,'Dragging must display intermediate frames'
assert w.results is identity;assert result.operators[2].preview_weights is operator
for p,old in zip(w.results,arrays):np.testing.assert_array_equal(p.cell_mass_kg_s,old)
assert json.dumps(project_to_json(w.cfg,w.nozzles),sort_keys=True)==snapshot
# Independent v0.4.1 solver for selected columns, with each manifold's own target P.
max_column=0.;max_field=0.
for step in (0,37,139):
 expected=np.zeros_like(result.composites[2].cell_mass_kg_s);field=np.zeros_like(result.smooth(2,step).flux_kg_m2_s)
 for key,pressures in result.pressures.items():
  subset=[n for n in result.nozzles if result.settings.manifold(n)==key];cfg=replace(result.cfg,target_pressure_kgf_cm2=float(pressures[step]),quasistationary=False)
  expected+=calculate_plane(subset,cfg,30.).cell_mass_kg_s;field+=evaluate_continuous_field(subset,cfg,30.).flux_kg_m2_s
 col=result.settings.column_indices(result.cfg)[step];max_column=max(max_column,float(np.max(np.abs(result.composites[2].cell_mass_kg_s[:,col]-expected[:,col]))));max_field=max(max_field,float(np.max(np.abs(result.smooth(2,step).flux_kg_m2_s-field))))
 np.testing.assert_allclose(result.composites[2].cell_mass_kg_s[:,col],expected[:,col],rtol=3e-13,atol=1e-14);np.testing.assert_allclose(result.smooth(2,step).flux_kg_m2_s,field,rtol=3e-13,atol=1e-12)
previous=w.quasi_result;started=time.perf_counter();w.run_calc();w.thread.requestInterruption();wait(w);cancel_ms=(time.perf_counter()-started)*1000;assert w.quasi_result is previous
assert hashlib.sha256(source.read_bytes()).hexdigest()==original
report=dict(version=__version__,python=platform.python_version(),platform=platform.system(),project='examples/demo_project_user_field.json',mode='synthetic 140 steps, two collectors; real user geometry/flows/sectors/types unchanged; portable demonstration bindings',method='Linux offscreen; no concurrent tests; 5 warmups excluded; direct callback with Qt painting + QTest slider keys/press/release and Qt held-button mouse moves',engine_ms=round(result.elapsed_s*1000,3),worker_and_initial_render_ms=round(wall,3),qt_heartbeat_gap_ms=stats(np.diff(beats)*1000),frame_callback_ms=callback,real_qt_slider_ms=real_qt,full_canvas_draws_during_140_callback_frames=full_draws,mouse_drag=dict(movements=100,displayed_updates=anim.update_count-before_updates,elapsed_ms=round(drag_ms,3),last_step=canvas._quasi_step),cancel_ms=round(cancel_ms,3),preview_cache_mb=round(sum((op.preview_weights.data.nbytes+op.preview_weights.indices.nbytes+op.preview_weights.indptr.nbytes)/2**20 for op in result.operators),3),peak_process_rss_mb=round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,3),maximum_instantaneous_balance_error=result.maximum_balance_error,independent_v041_column_max_absolute_difference_kg_s=max_column,independent_v041_detailed_field_max_absolute_difference=max_field,accumulated_matrices_unchanged_by_slider=True,project_unchanged_by_slider=True,user_source_file_unchanged=True)
args.out.parent.mkdir(parents=True,exist_ok=True);args.out.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n');print(json.dumps(report,ensure_ascii=False,indent=2));w.close()
