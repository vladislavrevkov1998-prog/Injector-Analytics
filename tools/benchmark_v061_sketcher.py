"""Real arc drag frames on the unchanged 140-nozzle user project."""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
import time
from unittest.mock import patch
import numpy as np
from PyQt5.QtCore import Qt,QPointF
from PyQt5.QtGui import QMouseEvent,QFont
from PyQt5.QtTest import QTest
from PyQt5.QtWidgets import QApplication

root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root/'src'))
from fuelfield.gui import MainWindow
from fuelfield.chamber import Primitive
from fuelfield.core import project_to_json

parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--out',type=Path,required=True);parser.add_argument('--image-subdir',default='v0.6.1');parser.add_argument('--linked',action='store_true');args=parser.parse_args()
def stats(values):return dict(samples=len(values),median_ms=round(float(np.median(values)),3),p95_ms=round(float(np.percentile(values,95)),3),max_ms=round(max(values),3))
app=QApplication([]);app.setStyle('Fusion');w=MainWindow();w.resize(1480,930);w.show()
source=root/'examples/demo_project_user_field.json';digest=hashlib.sha256(source.read_bytes()).hexdigest()
with patch('fuelfield.gui.QFileDialog.getOpenFileName',return_value=(str(source),'')):w.open_project()
w.geometry_pages.setCurrentIndex(1);app.processEvents();e=w.geometry_editor
radius=(w.cfg.r_in_mm+w.cfg.r_out_mm)/2
e.raw['primitives'].append(dict(id='ОП1',wall='reference',kind='arc',start=[10,radius],end=[20,radius],center=[15,radius+10],clockwise=False,unit='mm'))
assert e.commit('Synthetic benchmark arc');e.fit();app.processEvents();original=project_to_json(w.cfg,w.nozzles);reports={}
for role in ('start','end','center','body'):
    samples=[];releases=[]
    for repeat in range(5):
        e.selected=None;w.undo_stack.clear();raw=next(p for p in e.raw['primitives'] if p['id']=='ОП1');p=Primitive(raw)
        point=np.array(raw[role]) if role!='body' else np.array([15,float(p.at(15))])
        view=e.sketch;start=view.mapFromScene(QPointF(point[0],-point[1]));QTest.mousePress(view.viewport(),Qt.LeftButton,pos=start)
        assert e.selected=='ОП1',(role,e.selected)
        assert view.scene().mouseGrabberItem().role==role,role
        for frame in range(30):
            end=start+view.mapFromScene(QPointF(2*(frame+1)/30,-1.5*(frame+1)/30))-view.mapFromScene(QPointF(0,0))
            began=time.perf_counter();event=QMouseEvent(QMouseEvent.MouseMove,QPointF(end),QPointF(view.viewport().mapToGlobal(end)),Qt.NoButton,Qt.LeftButton,Qt.NoModifier)
            app.sendEvent(view.viewport(),event);app.processEvents()
            if frame>=5:samples.append((time.perf_counter()-began)*1000)
        began=time.perf_counter();QTest.mouseRelease(view.viewport(),Qt.LeftButton,pos=end);app.processEvents();releases.append((time.perf_counter()-began)*1000)
        assert w.undo_stack.count()==1 and not e._draft_invalid
        assert e.raw!=original['settings']['geometry'];w.undo_stack.undo();app.processEvents();assert project_to_json(w.cfg,w.nozzles)==original
    reports[role]=dict(frame_ms=stats(samples),release_ms=stats(releases))
if args.linked:
    e.raw['primitives'].append(dict(id='ОП2',wall='reference',kind='line',start=[20,radius],end=[30,radius+5],unit='mm'))
    assert e.commit('Synthetic contact line')
    assert e.create_constraint('tangent',dict(primitive_id='ОП1',endpoint='end'),dict(primitive_id='ОП2',endpoint='start'))
    linked_original=project_to_json(w.cfg,w.nozzles);samples=[];releases=[]
    for repeat in range(5):
        w.undo_stack.clear();handle=next(h for h in e.handles if h.identity=='ОП1' and h.role=='start');view=e.sketch;start=view.mapFromScene(handle.pos())
        QTest.mousePress(view.viewport(),Qt.LeftButton,pos=start);assert view.scene().mouseGrabberItem().role=='start'
        for frame in range(30):
            end=start+view.mapFromScene(QPointF(2*(frame+1)/30,-1.5*(frame+1)/30))-view.mapFromScene(QPointF(0,0))
            began=time.perf_counter();event=QMouseEvent(QMouseEvent.MouseMove,QPointF(end),QPointF(view.viewport().mapToGlobal(end)),Qt.NoButton,Qt.LeftButton,Qt.NoModifier);app.sendEvent(view.viewport(),event);app.processEvents()
            if frame>=5:samples.append((time.perf_counter()-began)*1000)
        began=time.perf_counter();QTest.mouseRelease(view.viewport(),Qt.LeftButton,pos=end);app.processEvents();releases.append((time.perf_counter()-began)*1000)
        assert w.undo_stack.count()==1 and not e._draft_invalid
        from fuelfield.chamber_constraints import validate_constraints
        validate_constraints(e.raw);w.undo_stack.undo();app.processEvents();assert project_to_json(w.cfg,w.nozzles)==linked_original
    reports['tangent_free_end']=dict(frame_ms=stats(samples),release_ms=stats(releases))
images=args.out.parent/'images'/args.image_subdir;images.mkdir(parents=True,exist_ok=True);layouts=[]
for font_size in (10,13):
    app.setFont(QFont(app.font().family(),font_size));w.resize(1280,800);app.processEvents();e.fit();app.processEvents()
    layouts.append(dict(font_size=font_size,window_width=w.width(),window_height=w.height(),minimum_width=w.minimumSizeHint().width(),split=e.splitter.sizes()))
    assert w.width()==1280
    e.select('ОП1');w.grab().save(str(images/f'sketcher_1280_font{font_size}.png'))
assert hashlib.sha256(source.read_bytes()).hexdigest()==digest
report=dict(project=source.name,sha256=digest,method=f'Linux Qt offscreen, real first-click press/move/release and repaint, {25 if args.linked else 20} gestures, {625 if args.linked else 500} measured frames after warmup, 140 nozzles, no timers disabled',arcs=reports,layouts=layouts,one_gesture_one_undo=True,original_project_unchanged=True)
args.out.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n');print(json.dumps(report,ensure_ascii=False,indent=2));w.close()
