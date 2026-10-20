"""Measure new draft history and arc–arc dragging on the original user project."""
from copy import deepcopy
import hashlib,json,sys,time
from pathlib import Path
from unittest.mock import patch
import numpy as np
from PyQt5.QtCore import Qt,QPointF
from PyQt5.QtGui import QMouseEvent
from PyQt5.QtTest import QTest
from PyQt5.QtWidgets import QApplication
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from fuelfield.gui import MainWindow
from fuelfield.chamber_constraints import tangent_references,validate_constraints
from fuelfield.core import project_to_json


def stats(values):return dict(samples=len(values),median_ms=round(float(np.median(values)),3),p95_ms=round(float(np.percentile(values,95)),3),max_ms=round(max(values),3))


if __name__=='__main__':
    app=QApplication([]);app.setStyle('Fusion');w=MainWindow();w.resize(1280,800);w.show()
    source=ROOT/'examples/demo_project_user_field.json';digest=hashlib.sha256(source.read_bytes()).hexdigest()
    with patch('fuelfield.gui.QFileDialog.getOpenFileName',return_value=(str(source),'')):w.open_project()
    w.geometry_pages.setCurrentIndex(1);app.processEvents();e=w.geometry_editor
    e.raw['primitives'].extend([
        dict(id='A1',kind='arc',wall='reference',start=[30.,260.],end=[40.,250.],center=[30.,250.],clockwise=True,unit='mm'),
        dict(id='A2',kind='arc',wall='reference',start=[40.,250.],end=[50.,240.],center=[50.,250.],clockwise=False,unit='mm')])
    assert e.commit('Synthetic benchmark arcs');a,b=tangent_references(e.raw,'A1','A2');assert e.create_constraint('tangent',a,b)
    e.fit();app.processEvents();baseline=project_to_json(w.cfg,w.nozzles);frames=[];releases=[];undo=[];redo=[]
    def measure(action,values):
        start=time.perf_counter();action();app.processEvents();values.append((time.perf_counter()-start)*1000)
    for repeat in range(8):
        w.undo_stack.clear();view=e.sketch;handle=next(h for h in e.handles if h.identity=='A1' and h.role=='start');start=view.mapFromScene(handle.pos())
        QTest.mousePress(view.viewport(),Qt.LeftButton,pos=start);assert view.scene().mouseGrabberItem().identity=='A1'
        for frame in range(30):
            end=start+view.mapFromScene(QPointF((frame+1)/30,-(frame+1)/30))-view.mapFromScene(QPointF(0,0))
            event=QMouseEvent(QMouseEvent.MouseMove,QPointF(end),QPointF(view.viewport().mapToGlobal(end)),Qt.NoButton,Qt.LeftButton,Qt.NoModifier)
            measure(lambda:app.sendEvent(view.viewport(),event),frames)
        measure(lambda:QTest.mouseRelease(view.viewport(),Qt.LeftButton,pos=end),releases)
        assert not e._draft_invalid and w.undo_stack.count()==1;validate_constraints(e.raw)
        measure(w.undo_stack.undo,undo);assert project_to_json(w.cfg,w.nozzles)==baseline
        measure(w.undo_stack.redo,redo);validate_constraints(e.raw);w.undo_stack.undo();app.processEvents()
    invalid=[];reset=[];reset_undo=[]
    inner=next(p for p in e.raw['primitives'] if p['wall']=='inner' and p['kind']=='line')['id']
    for repeat in range(12):
        w.undo_stack.clear();p=next(p for p in e.raw['primitives'] if p['id']==inner);p['end'][1]=w.cfg.r_out_mm+5+repeat
        measure(lambda:e.commit('Synthetic invalid wall'),invalid);assert e._draft_invalid
        draft=deepcopy(e.raw);measure(e.reset_draft,reset);assert not e._draft_invalid
        measure(w.undo_stack.undo,reset_undo);assert e.raw==draft and e._draft_invalid
        w.undo_stack.undo();app.processEvents();assert project_to_json(w.cfg,w.nozzles)==baseline
    assert hashlib.sha256(source.read_bytes()).hexdigest()==digest
    report=dict(project=source.name,sha256=digest,method='Linux Qt offscreen; real native drag events, 140 original nozzles, synthetic reference arcs/invalid drafts; all times include processEvents; no timers disabled',arc_arc_frames=stats(frames),arc_arc_release=stats(releases),accepted_undo=stats(undo),accepted_redo=stats(redo),invalid_commit=stats(invalid),reset_draft=stats(reset),undo_reset=stats(reset_undo))
    (ROOT/'docs/V0.6.3_DRAFT_LATENCY.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n');print(json.dumps(report,ensure_ascii=False,indent=2))
    demo=ROOT/'examples/demo_project_v063_s_profile.json'
    with patch('fuelfield.gui.QFileDialog.getOpenFileName',return_value=(str(demo),'')):w.open_project()
    w.geometry_pages.setCurrentIndex(1);app.processEvents();w.geometry_editor.fit();app.processEvents()
    w.grab().save(str(ROOT/'docs/images/v0.6.3/s-profile.png'));w.close()
