"""Run identical native Qt P0 checks against a selected source checkout.

Example: QT_QPA_PLATFORM=offscreen PYTHONPATH=src python tools/verify_v08_ui_p0.py
  --source-root /tmp/injector-before --output-prefix /tmp/ui-before
The baseline intentionally returns a failure for U01. U02 may already pass on
newer Matplotlib; that is evidence of non-reproduction, not proof about Windows.
"""
import argparse
import json
import os
from pathlib import Path
import platform
import sys
import time
import traceback

parser=argparse.ArgumentParser();parser.add_argument('--source-root',type=Path,default=Path(__file__).resolve().parents[1]);parser.add_argument('--output-prefix',type=Path,required=True);parser.add_argument('--only',choices=('U01','U02'))
arguments=parser.parse_args();sys.path.insert(0,str(arguments.source_root/'src'))
import numpy as np
import scipy
import matplotlib
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from PyQt5.QtCore import Qt,QPoint,QPointF,PYQT_VERSION_STR,QT_VERSION_STR
from PyQt5.QtGui import QMouseEvent,QImage
from PyQt5.QtWidgets import QApplication
from PyQt5.QtTest import QTest
from fuelfield.plot_interaction import attach_interaction
from fuelfield.gui import MainWindow,INTCOL
from fuelfield.core import project_from_json,project_to_json

app=QApplication.instance() or QApplication([])
report=dict(environment=dict(python=platform.python_version(),platform=platform.platform(),matplotlib=matplotlib.__version__,numpy=np.__version__,scipy=scipy.__version__,pyqt=PYQT_VERSION_STR,qt=QT_VERSION_STR,backend=matplotlib.get_backend(),qpa=os.environ.get('QT_QPA_PLATFORM'),scale=os.environ.get('QT_SCALE_FACTOR','1'),source_root=str(arguments.source_root.resolve())),checks={})


def position(canvas,ax,x,y):
    px,py=ax.transData.transform((x,y));ratio=canvas.device_pixel_ratio
    return QPoint(round(px/ratio),round(canvas.height()-py/ratio))


def frame(canvas,name):
    image=canvas.grab().toImage().convertToFormat(QImage.Format_RGBA8888)
    image.save(str(arguments.output_prefix)+'-'+name+'.png');data=image.bits();data.setsize(image.byteCount())
    return np.frombuffer(data,np.uint8).reshape(image.height(),image.bytesPerLine()//4,4)[:,:image.width()].copy()


def run_u01():
    window=MainWindow();window._restoring=True
    demo=arguments.source_root/'examples/demo_project_user_field.json'
    window.cfg,window.nozzles=project_from_json(json.loads(demo.read_text()))
    window.cfg.z_mm=[30.];window._fill_controls();window._fill_nozzles();window._reset_history()
    window.nozzle_table.item(0,INTCOL['flow_kg_h']).setText('9.125')
    window.tabs.setCurrentIndex(3);window.show();app.processEvents()
    started=time.perf_counter();QTest.keyClick(window,Qt.Key_F5)
    deadline=time.monotonic()+90
    while window.thread is not None and time.monotonic()<deadline:QTest.qWait(20)
    if window.thread is not None:raise AssertionError('F5 did not finish in 90s')
    if not window.results:raise AssertionError('F5 produced no results')
    window.plots.setCurrentIndex(3);window.plot_count.setValue(1);app.processEvents()
    canvas=window.curvecanvas;canvas.draw();canvas.setFocus();controller=canvas.interaction;ax=canvas.figure.axes[0]
    before=window.results;physical_before=project_to_json(window.cfg,window.nozzles);before_mass=before[0].cell_mass_kg_s.copy()
    source_index=window.undo_stack.index();line=ax.lines[0];x=np.asarray(line.get_xdata());y=np.asarray(line.get_ydata());index=len(x)//2
    point=position(canvas,ax,float(x[index]),float(y[index]))
    QTest.mouseDClick(canvas,Qt.LeftButton,pos=point);QTest.mouseRelease(canvas,Qt.LeftButton,pos=point);app.processEvents()
    point_count=len(controller.points);frame(canvas,'U01-pinned')
    QTest.keyClick(canvas,Qt.Key_Z,Qt.ControlModifier);app.processEvents()
    same_results=window.results is before and bool(window.results) and np.array_equal(window.results[0].cell_mass_kg_s,before_mass)
    same_project=project_to_json(window.cfg,window.nozzles)==physical_before
    outcome=dict(pass_check=bool(point_count==1 and not controller.points and same_results and same_project),points_before=point_count,points_after=len(controller.points),result_count_after=len(window.results),same_result_identity_and_values=bool(same_results),same_project=bool(same_project),source_index_before=source_index,source_index_after=window.undo_stack.index(),f5_and_check_s=time.perf_counter()-started,device_pixel_ratio=canvas.device_pixel_ratio)
    window.close();window.deleteLater();app.processEvents();return outcome


def run_u02():
    figure=Figure(figsize=(6,4));canvas=FigureCanvasQTAgg(figure);canvas.resize(600,400);ax=figure.add_subplot();ax.plot([0,10],[0,20]);ax.set_xlim(0,10);ax.set_ylim(0,20)
    controller=attach_interaction(canvas);canvas.show();app.processEvents();canvas.draw();events=[]
    for kind in ('button_press_event','motion_notify_event','button_release_event','draw_event'):
        canvas.mpl_connect(kind,lambda event,name=kind:events.append(dict(type=name,x=getattr(event,'x',None),y=getattr(event,'y',None),button=str(getattr(event,'button',None)))))
    start=position(canvas,ax,3,6);target=position(canvas,ax,7,16);other=position(canvas,ax,7,3)
    before=frame(canvas,'U02-before');QTest.mousePress(canvas,Qt.LeftButton,pos=start);frames=[];slopes=[];green_counts=[]
    draw_count=[0];original_draw=canvas.draw
    def counted_draw(*args,**kwargs):draw_count[0]+=1;return original_draw(*args,**kwargs)
    canvas.draw=counted_draw
    for name,point in (('first-move',target),('second-move',other)):
        event=QMouseEvent(QMouseEvent.MouseMove,QPointF(point),QPointF(canvas.mapToGlobal(point)),Qt.NoButton,Qt.LeftButton,Qt.NoModifier)
        QApplication.sendEvent(canvas,event);app.processEvents();pixels=frame(canvas,'U02-'+name);frames.append(pixels)
        if not controller.oblique:raise AssertionError('No oblique guide before release')
        line=controller.oblique[0].artists[0]
        transformed=ax.transData.inverted().transform(line.get_transform().transform(line.get_path().vertices));delta=np.diff(transformed,axis=0)[0]
        slopes.append(float(delta[1]/delta[0]) if delta[0]!=0 else None)
        anchor=controller.oblique[0];k=slopes[-1];probe=position(canvas,ax,5,anchor.y+k*(5-anchor.x));ratio=canvas.device_pixel_ratio;px,py=round(probe.x()*ratio),round(probe.y()*ratio)
        roi=pixels[py-6:py+7,px-6:px+7,:3].astype(int)
        green_counts.append(int(np.count_nonzero((roi[:,:,1]>roi[:,:,0]+15)&(roi[:,:,1]>roi[:,:,2]+4))))
    before_release=controller.slope_drag is not None
    QTest.mouseRelease(canvas,Qt.LeftButton,pos=other);app.processEvents();frame(canvas,'U02-release')
    changed=[int(np.count_nonzero(frames[0]!=before)),int(np.count_nonzero(frames[1]!=frames[0]))]
    result=dict(pass_check=bool(slopes[0]>2.3 and slopes[1]<-.5 and all(green_counts) and min(changed)>200 and before_release and draw_count[0]==0),slopes=slopes,green_pixels_at_predicted_line=green_counts,changed_rgba=changed,draw_count_during_moves=draw_count[0],created_before_release=before_release,device_pixel_ratio=canvas.device_pixel_ratio,events=events)
    canvas.close();canvas.deleteLater();app.processEvents();return result


for name,check in (('U01',run_u01),('U02',run_u02)):
    if arguments.only and arguments.only!=name:continue
    try:report['checks'][name]=check()
    except Exception as error:report['checks'][name]=dict(pass_check=False,error=str(error),traceback=traceback.format_exc())
output=Path(str(arguments.output_prefix)+'.json');output.write_text(json.dumps(report,ensure_ascii=False,indent=2))
print(json.dumps(report,ensure_ascii=False,indent=2));sys.exit(0 if all(r['pass_check'] for r in report['checks'].values()) else 1)
