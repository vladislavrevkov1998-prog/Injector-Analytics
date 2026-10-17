"""Native Qt frames, ordinary tab/focus activity, and live pressure plots.

Run sequentially on the old and new source roots; no UI timers are disabled.
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from unittest.mock import patch
import numpy as np

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--source-root',type=Path,default=Path(__file__).resolve().parents[1])
parser.add_argument('--out',type=Path,required=True)
parser.add_argument('--frame-zero',type=float,default=0.,help='Physical +X/CCW position of measurement zero, degrees')
parser.add_argument('--clockwise',action='store_true',help='Positive measurement direction is clockwise')
parser.add_argument('--cells',action='store_true',help='Display epures in circumferential cells')
args=parser.parse_args();root=args.source_root.resolve();sys.path.insert(0,str(root/'src'))
from PyQt5.QtCore import Qt,QTimer
from PyQt5.QtGui import QImage
from PyQt5.QtTest import QTest
from PyQt5.QtWidgets import QApplication
from fuelfield.gui import MainWindow
from fuelfield.core import project_to_json


def stats(values):
    return dict(samples=len(values),median_ms=round(float(np.median(values)),3),p95_ms=round(float(np.percentile(values,95)),3),max_ms=round(float(max(values)),3))


def wait_until(condition,seconds=60):
    deadline=time.monotonic()+seconds
    while not condition() and time.monotonic()<deadline:QTest.qWait(2)
    assert condition(),'Timed out waiting for actual UI state'


def pixels(canvas):
    image=canvas.grab().toImage().convertToFormat(QImage.Format_RGBA8888)
    data=image.bits();data.setsize(image.byteCount())
    return np.frombuffer(data,dtype=np.uint8).reshape(image.height(),image.bytesPerLine()//4,4)[:,:image.width()].copy()


app=QApplication([]);app.setStyle('Fusion');w=MainWindow();w.show()
source=root/'examples/demo_project_user_field.json';original=hashlib.sha256(source.read_bytes()).hexdigest()
with patch('fuelfield.gui.QFileDialog.getOpenFileName',return_value=(str(source),'')):w.open_project()
if args.frame_zero or args.clockwise or args.cells:
    w.geom_widgets['theta_zero_deg'].setValue(args.frame_zero);w.geom_widgets['theta_zero_deg'].editingFinished.emit()
    w.geom_widgets['theta_direction'].setCurrentIndex(w.geom_widgets['theta_direction'].findData(-1 if args.clockwise else 1))
    w.x_axis.setCurrentIndex(w.x_axis.findData('cells' if args.cells else 'angle'))
w.quasi_check.setChecked(True);d=w.quasi_dialog;d.load_path(root/'examples/demo_user_field_140_two_collectors.quasi.json');d.apply();d.close()
started=time.perf_counter();w.run_calc();wait_until(lambda:w.thread is None);worker_ms=(time.perf_counter()-started)*1000
assert w.quasi_result is not None;w.tabs.setCurrentIndex(3);w.plane_select.setCurrentIndex(2);w.plots.setCurrentIndex(1);app.processEvents();w.smoothcanvas.draw();QTest.qWait(40)
canvas=w.smoothcanvas;anim=canvas.quasi_animator;identity=w.results;matrices=[p.cell_mass_kg_s.copy() for p in w.results];state=project_to_json(w.cfg,w.nozzles)
before=pixels(canvas);w.time_slider.setValue(35);wait_until(lambda:canvas._quasi_step==35);app.processEvents();after=pixels(canvas)
changed=int(np.any(np.abs(before[:,:,:3].astype(int)-after[:,:,:3].astype(int))>2,axis=2).sum());assert changed>100,'Field must change on the native Qt widget'
frames=[];draws=anim.draw_count
for i in range(65):
    expected=w.time_slider.value()+1 if w.time_slider.value()<139 else 0
    started=time.perf_counter();QTest.keyClick(w.time_slider,Qt.Key_Right if expected else Qt.Key_Home)
    wait_until(lambda:canvas._quasi_step==expected,2);app.processEvents()
    if i>=5:frames.append((time.perf_counter()-started)*1000)
full_draws=anim.draw_count-draws
# Every tab is visited before measuring warm navigation; first visits stay separate.
cold=[];warm=[];bar=w.tabs.tabBar()
for cycle in range(9):
    for index in (0,1,2,3):
        started=time.perf_counter();QTest.mouseClick(bar,Qt.LeftButton,pos=bar.tabRect(index).center());app.processEvents();QTest.qWait(2)
        (cold if cycle==0 else warm).append((time.perf_counter()-started)*1000)
w.tabs.setCurrentIndex(0);QTest.qWait(200);focus=[];spin=w.geom_widgets['r_in_mm']
for i in range(35):
    spin.setFocus();app.processEvents();started=time.perf_counter();spin.editingFinished.emit();app.processEvents()
    if i>=5:focus.append((time.perf_counter()-started)*1000)
assert w.results is identity
for p,array in zip(w.results,matrices):np.testing.assert_array_equal(p.cell_mass_kg_s,array)
assert project_to_json(w.cfg,w.nozzles)==state
# A physical edit on 01 must not trigger expensive hidden preview work on 02.
beats=[];timer=QTimer();timer.setInterval(5);timer.timeout.connect(lambda:beats.append(time.perf_counter()));timer.start();QTest.qWait(15)
spin.setValue(spin.value()+.01);started=time.perf_counter();spin.editingFinished.emit();edit_ms=(time.perf_counter()-started)*1000;QTest.qWait(220);timer.stop()
edit_heartbeat=stats(np.diff(beats)*1000);w.undo_stack.undo();QTest.qWait(220);assert project_to_json(w.cfg,w.nozzles)==state
w.open_quasi_settings();d=w.quasi_dialog;charts=getattr(d,'pressure_charts',None);pressure=None
if charts is not None:
    wait_until(lambda:len(charts.axes)==2);QTest.qWait(30);count=charts.update_count;started=time.perf_counter()
    for i in range(100):d.table.item(i,3).setText(str(29+i/100))
    wait_until(lambda:charts.update_count>count);app.processEvents();pressure=dict(edited_cells=100,elapsed_ms=round((time.perf_counter()-started)*1000,3),chart_updates=charts.update_count-count,graphs=len(charts.axes))
    d.grab().save(str(args.out.with_suffix('.png')))
d.close();assert hashlib.sha256(source.read_bytes()).hexdigest()==original
report=dict(source_root=str(root),source_gui_sha256=hashlib.sha256((root/'src/fuelfield/gui.py').read_bytes()).hexdigest(),project=source.name,method='Linux offscreen; QTest tab/slider actions, native QWidget pixels and Qt editingFinished focus signals; no UI timers disabled; 5 warmups for slider/focus; sequential source runs',worker_and_initial_render_ms=round(worker_ms,3),native_field_changed_pixels=changed,slider_ms=stats(frames),slider_full_draws=full_draws,first_visits_ms=cold,warm_tab_click_ms=stats(warm),unchanged_control_focus_ms=stats(focus),physical_edit_sync_ms=round(edit_ms,3),physical_edit_qt_heartbeat_ms=edit_heartbeat,pressure_chart_batch=pressure,source_project_and_results_unchanged_by_navigation=True)
report['measurement_frame']=dict(zero_deg=getattr(w.cfg,'theta_zero_deg',0),direction=getattr(w.cfg,'theta_direction',1),x_axis=getattr(w.cfg,'display_x_axis','angle'))
args.out.parent.mkdir(parents=True,exist_ok=True);args.out.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n');print(json.dumps(report,indent=2));w.close()
