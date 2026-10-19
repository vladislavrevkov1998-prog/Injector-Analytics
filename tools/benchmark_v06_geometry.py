"""Qt geometry editing, native drag frames, section changes and quasi snapshots.

Run after the test suite, with no competing benchmarks. No timers disabled.
"""
import argparse,hashlib,json,sys,time
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch
import numpy as np
parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--out',type=Path,required=True);args=parser.parse_args()
root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root/'src'))
from PyQt5.QtCore import Qt,QPointF
from PyQt5.QtGui import QMouseEvent,QImage
from PyQt5.QtTest import QTest
from PyQt5.QtWidgets import QApplication
from fuelfield.gui import MainWindow
from fuelfield.core import project_to_json
from fuelfield.chamber import Chamber


def stats(values):return dict(samples=len(values),median_ms=round(float(np.median(values)),3),p95_ms=round(float(np.percentile(values,95)),3),max_ms=round(float(max(values)),3))
def wait(condition):
    deadline=time.monotonic()+90
    while not condition() and time.monotonic()<deadline:QTest.qWait(2)
    assert condition(),'UI calculation timed out'
def pixels(canvas):
    im=canvas.grab().toImage().convertToFormat(QImage.Format_RGBA8888);data=im.bits();data.setsize(im.byteCount())
    return np.frombuffer(data,dtype=np.uint8).reshape(im.height(),im.bytesPerLine()//4,4)[:,:im.width()].copy()


app=QApplication([]);app.setStyle('Fusion');w=MainWindow();w.resize(1480,930);w.show()
project=root/'examples/demo_project_v06_geometry.json';digest=hashlib.sha256(project.read_bytes()).hexdigest();args.out.parent.mkdir(parents=True,exist_ok=True)
images=args.out.parent/'images/v0.6';images.mkdir(parents=True,exist_ok=True)
with patch('fuelfield.gui.QFileDialog.getOpenFileName',return_value=(str(project),'')):w.open_project()
began=time.perf_counter();w.geometry_pages.setCurrentIndex(1);app.processEvents();cold_ms=(time.perf_counter()-began)*1000;e=w.geometry_editor;e.fit();app.processEvents()
model=Chamber(w.cfg.geometry);assert len(w.nozzles)==140;original=project_to_json(w.cfg,w.nozzles);timings=[]
for i in range(18):
    row=next(k for k,p in enumerate(e.raw['primitives']) if p['id']=='datum');start=time.perf_counter();e.primitive_table.item(row,4).setText(str(270+(i+1)*.005));app.processEvents()
    assert w.cfg.geometry['primitives'][-1]['start'][1]==270+(i+1)*.005
    if i>=3:timings.append((time.perf_counter()-start)*1000)
row=next(k for k,p in enumerate(e.raw['primitives']) if p['id']=='datum');e.primitive_table.item(row,4).setText('270');app.processEvents();assert project_to_json(w.cfg,w.nozzles)==original
e.select('datum');handle=next(h for h in e.handles if h.identity=='datum');view=e.sketch;position=view.mapFromScene(handle.pos());w.undo_stack.clear();QTest.mousePress(view.viewport(),Qt.LeftButton,pos=position);frames=[]
for i in range(45):
    end=view.mapFromScene(QPointF(17.375,-(270+(i+1)*.2)));start=time.perf_counter()
    event=QMouseEvent(QMouseEvent.MouseMove,QPointF(end),QPointF(view.viewport().mapToGlobal(end)),Qt.NoButton,Qt.LeftButton,Qt.NoModifier);app.sendEvent(view.viewport(),event);app.processEvents()
    if i>=5:frames.append((time.perf_counter()-start)*1000)
start=time.perf_counter();QTest.mouseRelease(view.viewport(),Qt.LeftButton,pos=end);app.processEvents();release_ms=(time.perf_counter()-start)*1000;assert w.undo_stack.count()==1;w.undo_stack.undo();app.processEvents();assert project_to_json(w.cfg,w.nozzles)==original
e.select('inner-arc');e.splitter.setSizes([600,850]);app.processEvents();e.fit();app.processEvents();w.grab().save(str(images/'sketcher_table.png'));e.angular.setChecked(True);e.fit();app.processEvents();w.grab().save(str(images/'angular_groups.png'));e.angular.setChecked(False)
layouts=[]
for width in (1280,1480,1920):
    w.resize(width,930);app.processEvents();assert w.width()==width
    layouts.append(dict(width=width,sketch_width=e.sketch.width(),table_width=e.primitive_table.width(),splitter=e.splitter.sizes()))
w.resize(1480,930);w.geometry_pages.setCurrentIndex(0);w.tabs.setCurrentIndex(3);start=time.perf_counter();w.run_calc();wait(lambda:w.thread is None);static_ms=(time.perf_counter()-start)*1000
assert len(w.results)==6;assert all(abs(p.rel_balance_error)<1e-12 for p in w.results);assert any(p.wall_bound_kg_s>0 for p in w.results)
switch=[]
for i in range(30):
    start=time.perf_counter();w.plane_select.setCurrentIndex((w.plane_select.currentIndex()+1)%6);app.processEvents()
    if i>=6:switch.append((time.perf_counter()-start)*1000)
w.plane_select.setCurrentIndex(3);app.processEvents();w.grab().save(str(images/'variable_matrix.png'));w.open_longitudinal();ld=w.longitudinal_dialog;ld.canvas.draw();ld.grab().save(str(images/'longitudinal.png'));ld.close()
w.quasi_check.setChecked(True);d=w.quasi_dialog;d.close();start=time.perf_counter();w.run_calc();wait(lambda:w.thread is None);quasi_ms=(time.perf_counter()-start)*1000;assert w.quasi_result is not None
w.plane_select.setCurrentIndex(3);w.plots.setCurrentIndex(1);app.processEvents();w.smoothcanvas.draw();QTest.qWait(30)
before=pixels(w.smoothcanvas);identity=w.results;matrices=[p.cell_mass_kg_s.copy() for p in identity];w.time_slider.setValue(35);wait(lambda:getattr(w.smoothcanvas,'_quasi_step',None)==35);app.processEvents();after=pixels(w.smoothcanvas)
changed=int(np.any(np.abs(before[:,:,:3].astype(int)-after[:,:,:3].astype(int))>2,axis=2).sum());assert changed>100
slider=[]
for i in range(45):
    expected=w.time_slider.value()+1;start=time.perf_counter();QTest.keyClick(w.time_slider,Qt.Key_Right);wait(lambda:w.smoothcanvas._quasi_step==expected);app.processEvents()
    if i>=5:slider.append((time.perf_counter()-start)*1000)
for p,array in zip(identity,matrices):np.testing.assert_array_equal(p.cell_mass_kg_s,array)
assert w.results is identity;quasi_error=w.quasi_result.maximum_balance_error;w.grab().save(str(images/'quasi_variable_field.png'));w.open_quasi_settings();d=w.quasi_dialog;app.processEvents();d.pressure_charts.canvas.draw();d.grab().save(str(images/'pressure_charts.png'));d.close()
# Large geometry group consists only of unmodelled holes, never extra nozzles.
w.tabs.setCurrentIndex(0);w.geometry_pages.setCurrentIndex(1);e.angular.setChecked(True);state=project_to_json(w.cfg,w.nozzles);raw=state['settings']['geometry'];members=[]
for i in range(1000):
    identity=f'large-{i}';members.append(identity);raw['objects'].append(dict(id=identity,type='hole',wall='free',z_mm=100.,radius_mm=250.,theta_deg=i*360/1000,parameters={'flow_kg_s':None},units={'flow_kg_s':'kg/s'},provenance={'source_kind':'synthetic'}))
raw['groups'].append(dict(id='large-group',mode='uniform',members=members,phase_deg=0.,direction=1,row_offset_deg=0.))
start=time.perf_counter();w.apply_project_state(state,'Large geometry benchmark');app.processEvents();large_ms=(time.perf_counter()-start)*1000;assert len(w.nozzles)==140 and len(e.raw['objects'])==1161
assert hashlib.sha256(project.read_bytes()).hexdigest()==digest
report=dict(method='Linux Qt offscreen; QTest mouse/slider, actual paint events, no disabled timers; warmups excluded; original fixture hash checked',project=project.name,source_gui_sha256=hashlib.sha256((root/'src/fuelfield/chamber_gui.py').read_bytes()).hexdigest(),source_project_sha256=digest,
    geometry_cold_open_ms=round(cold_ms,3),table_edit_ms=stats(timings),drag_frame_ms=stats(frames),drag_release_commit_ms=round(release_ms,3),one_drag_one_undo=True,layouts=layouts,
    stationary_worker_and_render_ms=round(static_ms,3),six_section_switch_ms=stats(switch),quasi_worker_and_render_ms=round(quasi_ms,3),variable_field_slider_ms=stats(slider),native_changed_pixels=changed,quasi_maximum_relative_balance_error=quasi_error,
    large_objects=dict(nozzles=140,other_objects=1021,new_group_members=1000,one_commit_and_first_render_ms=round(large_ms,3)),navigation_preserves_matrices=True)
args.out.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n');print(json.dumps(report,indent=2));w.close();w.deleteLater();app.processEvents()
