"""Actual Qt paint latency for linked epures + ten pins/oblique lines, user project."""
import argparse,json,sys,time,platform,tempfile
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from PyQt5.QtWidgets import QApplication
from PyQt5.QtTest import QTest
from PyQt5.QtCore import QPoint
from matplotlib.backend_bases import MouseEvent,MouseButton
from fuelfield.gui import MainWindow
from fuelfield.core import project_from_json
from fuelfield.regression import RegressionSnapshot
from fuelfield import __version__
root=Path(__file__).resolve().parents[1];parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,required=True);args=parser.parse_args()
app=QApplication([]);w=MainWindow();w._restoring=True;w.cfg,w.nozzles=project_from_json(json.loads((root/'examples/demo_project_user_field.json').read_text()))
w._fill_controls();w._fill_nozzles();w._reset_history();w.preview_timer.stop();w.selection_timer.stop();w.show()
def wait(condition,timeout=90):
 deadline=time.monotonic()+timeout
 while condition() and time.monotonic()<deadline:QTest.qWait(10)
 assert not condition(),'Worker timed out'
start=time.perf_counter();w.run_calc();wait(lambda:w.thread is not None);primary_ms=(time.perf_counter()-start)*1000
w.plane_select.setCurrentIndex(2);w.plots.setCurrentIndex(3);w.split_button.setChecked(True)
start=time.perf_counter();w.comparison.load_path(root/'examples/demo_project_user_field.json');wait(lambda:w.comparison.thread is not None);load_ms=(time.perf_counter()-start)*1000
w.comparison.plot_count.setValue(6);w.comparison.plot_columns.setValue(2);app.processEvents();w.curvecanvas.draw();w.comparison.canvas.draw();w.graph_links.refresh()
identity=w.results;before=[p.cell_mass_kg_s.copy() for p in w.results];snapshot=RegressionSnapshot.capture(w.cfg,w.nozzles).sha256

def stats(values):return dict(samples=len(values),median_ms=round(float(np.median(values)),3),p95_ms=round(float(np.percentile(values,95)),3),max_ms=round(float(max(values)),3))
def event(c,ax,name,fraction=.5,yfraction=.5,button=None,double=False):
 xlim,ylim=ax.get_xlim(),ax.get_ylim();xy=ax.transData.transform((xlim[0]+fraction*np.ptp(xlim),ylim[0]+yfraction*np.ptp(ylim)))
 return MouseEvent(name,c,*xy,button=button,dblclick=double)
def measure(label):
 a=w.curvecanvas;b=w.comparison.canvas;c=a.interaction;d=b.interaction;ax=a.figure.axes[0];bx=b.figure.axes[0]
 c.clear_points();d.clear_points();c.clear_oblique();d.clear_oblique();a.draw();b.draw();w.graph_links.refresh();manager=w.graph_links;link=manager.connect(ax,c,bx,d);app.processEvents()
 values=[]
 for i in range(10):
  start=time.perf_counter();c.press(event(a,ax,'button_press_event',.1+.08*i,button=MouseButton.LEFT,double=True));c.release(event(a,ax,'button_release_event',button=MouseButton.LEFT));app.processEvents();values.append((time.perf_counter()-start)*1000)
 assert len(c.points)==10 and len(d.points)==10
 out={'linked_pin':stats(values)}
 for i in range(10):
  c.press(event(a,ax,'button_press_event',.2,button=MouseButton.LEFT));c.motion(event(a,ax,'motion_notify_event',.7,.6,MouseButton.LEFT));c.release(event(a,ax,'button_release_event',button=MouseButton.LEFT))
 assert len(c.oblique)==10
 app.processEvents();values=[]
 for i in range(205):
  start=time.perf_counter();c.motion(event(a,ax,'motion_notify_event',.1+.8*(i%200)/200));app.processEvents()
  if i>=5:values.append((time.perf_counter()-start)*1000)
 out['hover_10_pins_10_oblique']=stats(values)
 values=[]
 for i in range(55):
  ev=event(a,ax,'motion_notify_event',.15+.7*(i%50)/50);point=QPoint(round(ev.x/a.device_pixel_ratio),round(a.height()-ev.y/a.device_pixel_ratio))
  start=time.perf_counter();QTest.mouseMove(a,point,delay=0);app.processEvents()
  if i>=5:values.append((time.perf_counter()-start)*1000)
 assert c.hover[ax][2].get_visible(),'Real Qt hover did not reach the epure'
 out['real_qt_hover_10_pins_10_oblique']=stats(values)
 values=[]
 for i in range(10):
  c.press(event(a,ax,'button_press_event',.2,.2,MouseButton.MIDDLE));c.motion(event(a,ax,'motion_notify_event',.8,.8,MouseButton.MIDDLE));app.processEvents();start=time.perf_counter();c.release(event(a,ax,'button_release_event',.8,.8,MouseButton.MIDDLE));app.processEvents();values.append((time.perf_counter()-start)*1000)
  c.reset_zoom();app.processEvents()
 out['linked_zoom_both_canvases']=stats(values)
 start=time.perf_counter();dialog=manager.analyze(link);app.processEvents();out['pearson_dialog_ms']=round((time.perf_counter()-start)*1000,3);dialog.close();app.processEvents()
 return out
report=dict(version=__version__,project='examples/demo_project_user_field.json',python=platform.python_version(),platform=platform.system(),method='Linux offscreen; Matplotlib events + Qt painting, 5 hover warmups excluded; extra real QTest mouseMove (delay=0), no concurrent tests',primary_four_sections_ms=round(primary_ms,3),split_four_sections_and_fields_ms=round(load_ms,3))
report['json_six_epures']=measure('json')
# Real XLSX parser path, compatible physical units and a different radial series.
from fuelfield.analytics_io import write_workbook
with tempfile.TemporaryDirectory() as temp:
 p=w.results[2];r=2;order=np.argsort(p.theta_centers_deg);rows=[['theta_deg','q [кг/(м²·с)]']]+[[float(p.theta_centers_deg[j]),float(p.flux_kg_m2_s[r,j])] for j in order]
 path=Path(temp)/'radial_comparison.xlsx';write_workbook(path,[('different_R',rows)]);w.comparison.load_path(path);app.processEvents();w.comparison.canvas.draw();w.graph_links.refresh()
 report['xlsx_different_radius']=measure('xlsx')
assert w.results is identity
assert RegressionSnapshot.capture(w.cfg,w.nozzles).sha256==snapshot
for p,old in zip(w.results,before):np.testing.assert_array_equal(p.cell_mass_kg_s,old)
report['physics_unchanged']=True;report['max_relative_balance']=max(abs(p.rel_balance_error) for p in w.results)
args.out.parent.mkdir(parents=True,exist_ok=True);args.out.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n');print(json.dumps(report,ensure_ascii=False,indent=2));w.close()
