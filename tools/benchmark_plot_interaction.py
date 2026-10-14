"""Measure GUI interaction latency on the user's project, including Qt painting.

Run QT_QPA_PLATFORM=offscreen MPLCONFIGDIR=/tmp/injector-mpl python -B
 tools/benchmark_plot_interaction.py --out /tmp/plot_latency.json
The full-redraw reference is a measured alternative, not a historical version.
"""
import argparse,json,sys,time,platform
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from PyQt5.QtWidgets import QApplication
from PyQt5.QtTest import QTest
from matplotlib.backend_bases import MouseEvent,MouseButton
from fuelfield.gui import MainWindow
from fuelfield import __version__
from fuelfield.core import project_from_json

root=Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--out',type=Path,required=True);args=parser.parse_args()
app=QApplication([]);w=MainWindow();w._restoring=True
w.cfg,w.nozzles=project_from_json(json.loads((root/'examples/demo_project_user_field.json').read_text()))
w._fill_controls();w._fill_nozzles();w._reset_history();w.show();w.run_calc()
deadline=time.monotonic()+60
while w.thread is not None and time.monotonic()<deadline:QTest.qWait(20)
assert len(w.results)==4,'primary calculation did not complete'
w.plane_select.setCurrentIndex(2);w.plots.setCurrentIndex(3);w.split_button.setChecked(True)
w.comparison.load_path(root/'examples/demo_project_user_field.json')
while w.comparison.thread is not None and time.monotonic()<deadline:QTest.qWait(20)
assert len(w.comparison.planes)==4,'comparison calculation did not complete'
w.comparison.plot_count.setValue(6);w.comparison.plot_columns.setValue(2);app.processEvents()
before=[p.cell_mass_kg_s.copy() for p in w.results];result_identity=w.results;revision=w._revision

def stats(values):
    return {'samples':len(values),'median_ms':round(float(np.median(values)),3),
            'p95_ms':round(float(np.percentile(values,95)),3),'max_ms':round(float(max(values)),3)}

def measure(canvas):
    canvas.draw();app.processEvents();c=canvas.interaction;ax=canvas.figure.axes[0]
    def event(name,fraction=.5,button=None,double=False):
        xlim,ylim=ax.get_xlim(),ax.get_ylim();x=xlim[0]+fraction*(xlim[1]-xlim[0]);y=ylim[0]+.5*(ylim[1]-ylim[0])
        pixel=ax.transData.transform((x,y));return MouseEvent(name,canvas,*pixel,button=button,dblclick=double)
    def hover(samples,full=False):
        values=[]
        for i in range(samples+5):
            started=time.perf_counter();c.motion(event('motion_notify_event',.1+.8*(i%samples)/samples))
            if full:canvas.draw()
            app.processEvents()
            if i>=5:values.append((time.perf_counter()-started)*1000)
        return stats(values)
    out={'axes':len(canvas.figure.axes),'hover_0_points':hover(150)}
    pins=[]
    for i in range(10):
        started=time.perf_counter();c.press(event('button_press_event',.1+.08*i,MouseButton.LEFT,True))
        c.release(event('button_release_event',.1+.08*i,MouseButton.LEFT));app.processEvents();pins.append((time.perf_counter()-started)*1000)
    assert len(c.points)==10
    out['pin_point']=stats(pins);out['hover_10_points']=hover(150);out['full_redraw_reference_10_points']=hover(20,True)
    zoom=[]
    for _ in range(10):
        xlim,ylim=ax.get_xlim(),ax.get_ylim()
        start=ax.transData.transform((xlim[0]+.2*(xlim[1]-xlim[0]),ylim[0]+.2*(ylim[1]-ylim[0])))
        end=ax.transData.transform((xlim[0]+.8*(xlim[1]-xlim[0]),ylim[0]+.8*(ylim[1]-ylim[0])))
        c.press(MouseEvent('button_press_event',canvas,*start,button=MouseButton.MIDDLE))
        started=time.perf_counter();c.motion(MouseEvent('motion_notify_event',canvas,*end,button=MouseButton.MIDDLE));app.processEvents()
        highlight=(time.perf_counter()-started)*1000
        started=time.perf_counter();c.release(MouseEvent('button_release_event',canvas,*end,button=MouseButton.MIDDLE));app.processEvents()
        zoom.append((time.perf_counter()-started)*1000);c.reset_zoom();app.processEvents()
    out['zoom_release']=stats(zoom);out['last_selection_motion_ms']=round(highlight,3)
    started=time.perf_counter();c.press(event('button_press_event',button=MouseButton.LEFT));c.press(event('button_press_event',button=MouseButton.RIGHT));app.processEvents()
    out['clear_10_points_ms']=round((time.perf_counter()-started)*1000,3)
    c.release(event('button_release_event',button=MouseButton.LEFT));c.release(event('button_release_event',button=MouseButton.RIGHT))
    assert not c.points
    return out

report={'version':__version__,'python':platform.python_version(),'platform':platform.system(),'project':'examples/demo_project_user_field.json','window_px':[w.width(),w.height()],
        'method':'Matplotlib events + Qt processEvents; offscreen Linux; warmup excluded',
        'primary_six_epures':measure(w.curvecanvas),'comparison_six_epures':measure(w.comparison.canvas)}
w.comparison.load_path(root/'examples/demo_comparison.csv');w.comparison.y2.setCurrentIndex(w.comparison.y2.findData(2));app.processEvents()
report['external_two_y']=measure(w.comparison.canvas)
assert w.results is result_identity and w._revision==revision
for p,expected in zip(w.results,before):np.testing.assert_array_equal(p.cell_mass_kg_s,expected)
report['physics_unchanged']=True;report['max_relative_balance']=max(abs(p.rel_balance_error) for p in w.results)
args.out.parent.mkdir(parents=True,exist_ok=True);args.out.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(report,ensure_ascii=False,indent=2));w.close()
