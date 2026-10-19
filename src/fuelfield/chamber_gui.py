"""Peer Sketcher/table editor. Valid edits commit through the main Undo stack.

The editable draft may temporarily have an open joint; it never enters a
calculation until validated. Dragging commits once, at mouse release.
"""
from copy import deepcopy
import json,math,uuid
from pathlib import Path
import numpy as np
from PyQt5.QtCore import Qt,QPointF,QRectF
from PyQt5.QtGui import QColor,QPen,QBrush,QPainterPath,QPainter
from PyQt5.QtWidgets import (QWidget,QVBoxLayout,QHBoxLayout,QSplitter,QTabWidget,
    QTableWidgetItem,QPushButton,QLabel,QComboBox,QCheckBox,QGraphicsView,
    QGraphicsScene,QGraphicsPathItem,QGraphicsEllipseItem,QGraphicsItem,
    QStyledItemDelegate,QInputDialog,QDoubleSpinBox,QDialog,QTextBrowser,QDialogButtonBox)
from .chamber import Chamber,Primitive,legacy_geometry,apply_groups,sync_groups,group_angles,UNITS,TOL,WALL_BOUND_NOTICE
from .core import project_to_json
from .numeric_input import DecimalSpinBox
from .numeric_delegate import NumericDelegate

WALLS={'inner':'Внутренняя','outer':'Наружная','reference':'Опорная','free':'Свободно'}
KINDS={'point':'Точка','line':'Отрезок','arc':'Дуга'}
MODES={'uniform':'Равномерно','individual':'Индивидуальные углы','steps':'Массив шагов'}
PRIMITIVE_COLUMNS=['ID','Стенка','Тип','z₁','R₁','z₂','R₂','z центра','R центра','Радиус','По ЧС','Единицы']
OBJECT_COLUMNS=['ID','Тип','Стенка','Примитив','z, мм','R, мм','θ, °','Группа','Параметры (JSON)','Единицы (JSON)','Происхождение (JSON)']
GROUP_COLUMNS=['ID','Объекты (через ;)','Режим','Фаза, °','Направление ±1','Смещение ряда, °','Углы/шаги, ° (через ;)']


class Choices(QStyledItemDelegate):
    def __init__(self,values,parent):super().__init__(parent);self.values=values
    def createEditor(self,parent,option,index):
        box=QComboBox(parent)
        for key,label in self.values.items():box.addItem(label,key)
        return box
    def setEditorData(self,editor,index):editor.setCurrentIndex(max(0,editor.findText(index.data())))
    def setModelData(self,editor,model,index):model.setData(index,editor.currentText())


class Handle(QGraphicsEllipseItem):
    def __init__(self,editor,identity,role,point,color,object_handle=False):
        super().__init__(-4,-4,8,8);self.editor=editor;self.identity=identity;self.role=role;self.object_handle=object_handle
        self.setBrush(QBrush(QColor(color)));self.setPen(QPen(QColor('#ffffff'),1));self.setZValue(8)
        self.setFlags(QGraphicsItem.ItemIsMovable|QGraphicsItem.ItemIsSelectable|QGraphicsItem.ItemIgnoresTransformations)
        self.setPos(QPointF(*point));self.setData(0,identity);self.setToolTip(identity+' · '+role)
    def mousePressEvent(self,event):
        self.before=deepcopy(self.editor.raw);self.before_nozzles=deepcopy(self.editor.working_nozzles)
        self.editor.select(self.identity,self.object_handle);super().mousePressEvent(event)
    def mouseMoveEvent(self,event):
        super().mouseMoveEvent(event)
        self.editor.preview_drag(self.identity,self.role,self.pos(),self.object_handle)
    def mouseReleaseEvent(self,event):
        super().mouseReleaseEvent(event)
        self.editor.finish_drag(self.identity,self.role,self.pos(),self.object_handle,self.before,self.before_nozzles)


class Sketch(QGraphicsView):
    def __init__(self,editor):
        super().__init__(QGraphicsScene(editor),editor);self.editor=editor;self.setRenderHint(QPainter.Antialiasing)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse);self.setMinimumSize(220,220)
        self._pan=None;self.setBackgroundBrush(QColor('#fafbfe'))
    def wheelEvent(self,event):self.scale(1.18 if event.angleDelta().y()>0 else 1/1.18,1.18 if event.angleDelta().y()>0 else 1/1.18);event.accept()
    def drawForeground(self,painter,rect):
        painter.save();painter.resetTransform();painter.setPen(QColor('#334155'))
        painter.drawText(QPointF(8,18),'X/Y [мм], θ от +X против ЧС' if self.editor.angular.isChecked() else 'z=x → [мм]     R ↑ [мм]');painter.restore()
    def mousePressEvent(self,event):
        if event.button()==Qt.MiddleButton:self._pan=event.pos();self.setCursor(Qt.ClosedHandCursor);event.accept();return
        super().mousePressEvent(event)
    def mouseMoveEvent(self,event):
        if self._pan is not None:
            delta=event.pos()-self._pan;self._pan=event.pos();self.horizontalScrollBar().setValue(self.horizontalScrollBar().value()-delta.x());self.verticalScrollBar().setValue(self.verticalScrollBar().value()-delta.y());event.accept();return
        super().mouseMoveEvent(event)
    def mouseReleaseEvent(self,event):
        if event.button()==Qt.MiddleButton:self._pan=None;self.unsetCursor();event.accept();return
        super().mouseReleaseEvent(event)
    def mouseDoubleClickEvent(self,event):
        if event.button()==Qt.LeftButton and not self.editor.angular.isChecked():self.editor.sketch_add(self.mapToScene(event.pos()));event.accept();return
        super().mouseDoubleClickEvent(event)


class GeometryEditor(QWidget):
    def __init__(self,owner,table_class):
        super().__init__(owner);self.owner=owner;self.loading=True;self.selected=None;self._pending=True;self.paths={};self.handles=[];self._first=None;self.batch_mode=False;self.pending_batch={}
        layout=QVBoxLayout(self);top=QHBoxLayout();self.enabled=QCheckBox('Переменные профили ЖТ');top.addWidget(self.enabled)
        self.base=self.spin(0);self.outlet=self.spin(60);self.step=self.spin(1,.0001);self.error=self.spin(.01,.000001)
        for label,widget in [('База z=x, мм:',self.base),('Выход, мм:',self.outlet),('Внутр. шаг, мм:',self.step),('Хорда ≤, мм:',self.error)]:top.addWidget(QLabel(label));top.addWidget(widget)
        layout.addLayout(top)
        note=QLabel('z=x [мм]; θ — от физической +X против ЧС. Изменения сразу применяются после проверки. Несогласованный черновик не участвует в расчёте.');note.setWordWrap(True);layout.addWidget(note)
        controls=QHBoxLayout();self.angular=QCheckBox('Вид по окружности');controls.addWidget(self.angular)
        fit=QPushButton('Вписать');fit.clicked.connect(self.fit);controls.addWidget(fit)
        self.tool=QComboBox();self.tool.addItems(['Выбор','Точка','Отрезок (2 точки)','Дуга (2 точки)','Отверстие','Вставка']);controls.addWidget(QLabel('Двойной ЛКМ:'));controls.addWidget(self.tool);controls.addStretch()
        guide=QPushButton('Инструкция');guide.clicked.connect(self.show_guide);controls.addWidget(guide)
        undo=QPushButton('↶');redo=QPushButton('↷');undo.clicked.connect(owner.undo_stack.undo);redo.clicked.connect(owner.undo_stack.redo);controls.addWidget(undo);controls.addWidget(redo);layout.addLayout(controls)
        self.splitter=QSplitter(Qt.Horizontal);self.sketch=Sketch(self);self.splitter.addWidget(self.sketch);self.pages=QTabWidget();self.splitter.addWidget(self.pages);self.splitter.setSizes([550,650]);layout.addWidget(self.splitter,1)
        self.primitive_table=table_class(0,len(PRIMITIVE_COLUMNS));self.object_table=table_class(0,len(OBJECT_COLUMNS));self.group_table=table_class(0,len(GROUP_COLUMNS));self.section_table=table_class(0,3)
        self.tables=[self.primitive_table,self.object_table,self.group_table,self.section_table]
        for tab,title,headers in zip(self.tables,['Примитивы','Объекты','Угловые группы','Сечения'],[PRIMITIVE_COLUMNS,OBJECT_COLUMNS,GROUP_COLUMNS,['ID','z=x, мм','Выход ЖТ']]):
            panel=QWidget();box=QVBoxLayout(panel);tab.setHorizontalHeaderLabels(headers);tab.setSelectionBehavior(tab.SelectItems);tab.setMinimumWidth(300);box.addWidget(tab);buttons=QHBoxLayout();box.addLayout(buttons);self.pages.addTab(panel,title)
            if tab is self.primitive_table:
                for title,callback in [('Точка',lambda:self.add_primitive('point')),('Отрезок',lambda:self.add_primitive('line')),('Дуга',lambda:self.add_primitive('arc')),('Разбить',self.split_primitive)]:self.button(buttons,title,callback)
            elif tab is self.object_table:
                for kind,title in [('hole','Отверстие'),('insert','Вставка'),('custom','Другой тип')]:self.button(buttons,title,lambda checked=False,k=kind:self.add_object(k))
            elif tab is self.group_table:self.button(buttons,'Добавить группу',self.add_group)
            else:
                self.button(buttons,'Добавить сечение',self.add_section);self.button(buttons,'↑',lambda:self.move_section(-1));self.button(buttons,'↓',lambda:self.move_section(1))
            self.button(buttons,'Удалить',lambda checked=False,t=tab:self.delete(t));buttons.addStretch()
            tab.itemChanged.connect(lambda item,t=tab:self.table_edited(t,item))
            tab.batchFinished.connect(lambda t=tab:self.batch_finished(t))
            tab.currentCellChanged.connect(lambda row,col,oldrow,oldcol,t=tab:self.table_selected(t,row))
        for column,values in ((1,{k:v for k,v in WALLS.items() if k!='free'}),(2,KINDS),(11,{k:k for k in UNITS})):
            self.primitive_table.setItemDelegateForColumn(column,Choices(values,self.primitive_table))
        self.object_table.setItemDelegateForColumn(2,Choices({k:v for k,v in WALLS.items() if k!='reference'},self.object_table));self.group_table.setItemDelegateForColumn(2,Choices(MODES,self.group_table))
        numeric=NumericDelegate(self)
        for column in range(3,10):self.primitive_table.setItemDelegateForColumn(column,numeric)
        for column in (4,5,6):self.object_table.setItemDelegateForColumn(column,numeric)
        self.section_table.setItemDelegateForColumn(1,numeric)
        self.status=QLabel();self.status.setWordWrap(True);layout.addWidget(self.status)
        self.enabled.toggled.connect(self.config_edited)
        for widget in (self.base,self.outlet,self.step,self.error):widget.editingFinished.connect(self.config_edited)
        self.angular.toggled.connect(lambda:self.draw_scene(True));self.sketch.scene().selectionChanged.connect(self.scene_selected)
        self.loading=False

    @staticmethod
    def spin(value,minimum=0):
        spin=DecimalSpinBox();spin.setDecimals(8);spin.setRange(minimum,1e6);spin.setSingleStep(.001);spin.setValue(value);spin.setMaximumWidth(115);return spin
    @staticmethod
    def button(layout,title,callback):
        button=QPushButton(title);button.clicked.connect(callback);layout.addWidget(button);return button
    @staticmethod
    def uid(prefix):return prefix+'-'+uuid.uuid4().hex[:8]
    def load(self,cfg,nozzles,force=False):
        candidate=cfg.geometry or legacy_geometry(cfg,nozzles)
        if not force and hasattr(self,'base_nozzles') and nozzles==self.base_nozzles and candidate==self.loaded_raw and cfg.nozzle_presets==self.loaded_presets:
            self.cfg=cfg;return
        incremental=False
        if not force and hasattr(self,'loaded_raw') and self.isVisible() and not self._pending and nozzles==self.base_nozzles and cfg.nozzle_presets==self.loaded_presets:
            old=self.loaded_raw
            incremental=({k:v for k,v in old.items() if k!='primitives'}=={k:v for k,v in candidate.items() if k!='primitives'} and
                [(p['id'],p['kind']) for p in old['primitives']]==[(p['id'],p['kind']) for p in candidate['primitives']])
        if incremental:
            old=self.loaded_raw;self.cfg=cfg;self.loaded_raw=deepcopy(candidate);self.raw=deepcopy(candidate)
            self.refresh_primitives(old);return
        self.cfg=cfg;self.loaded_raw=deepcopy(candidate);self.loaded_presets=deepcopy(cfg.nozzle_presets);self.working_nozzles=deepcopy(nozzles);self.base_nozzles=deepcopy(nozzles);self.raw=deepcopy(candidate);self._pending=True
        self.message('Геометрия загружена. Несогласованный черновик не участвует в расчёте.')
        if self.isVisible():self.rebuild()
    def showEvent(self,event):
        super().showEvent(event)
        if self._pending and hasattr(self,'raw'):self.rebuild()
    def message(self,text,error=False):self.status.setText(text);self.status.setStyleSheet('color:#a32138' if error else 'color:#52617c')
    def show_guide(self):
        self.guide_dialog=QDialog(self);self.guide_dialog.setWindowTitle('Геометрия ЖТ v0.6 — инструкция');self.guide_dialog.resize(900,750)
        layout=QVBoxLayout(self.guide_dialog);browser=QTextBrowser();path=Path(__file__).resolve().parents[2]/'docs/V0.6_GUIDE.md'
        try:browser.setMarkdown(path.read_text(encoding='utf-8'))
        except OSError:browser.setPlainText('Таблица и Sketcher редактируют одну геометрию. Валидные изменения входят в Undo/Redo; неподтверждённый черновик не участвует в расчёте. z=x [мм].')
        layout.addWidget(browser);buttons=QDialogButtonBox(QDialogButtonBox.Close);buttons.rejected.connect(self.guide_dialog.close);layout.addWidget(buttons);self.guide_dialog.show()
    def rebuild(self):
        self.loading=True
        try:
            self.enabled.setChecked(self.raw.get('enabled',False));self.base.setValue(self.raw['base_z_mm']);self.outlet.setValue(self.raw['outlet_z_mm']);self.step.setValue(self.raw.get('internal_step_mm',1));self.error.setValue(self.raw.get('chord_error_mm',.01))
            self.fill_tables();self.draw_scene(not bool(self.paths));self._pending=False
        finally:self.loading=False
    def item(self,table,row,column,value,editable=True,check=None):
        item=QTableWidgetItem(format(value,'.12g') if isinstance(value,(int,float)) and not isinstance(value,bool) else str(value))
        if isinstance(value,(int,float)) and not isinstance(value,bool):item.setData(Qt.UserRole,value)
        if check is not None:item.setText('');item.setFlags(Qt.ItemIsEnabled|Qt.ItemIsSelectable|Qt.ItemIsUserCheckable);item.setCheckState(Qt.Checked if check else Qt.Unchecked)
        elif not editable:item.setFlags(item.flags()&~Qt.ItemIsEditable);item.setBackground(QColor('#eee8f6'))
        table.setItem(row,column,item)
    def fill_primitive_row(self,row,p):
        start=p['start'];end=p.get('end',['','']);center=p.get('center',['','']);kind=p['kind']
        radius=p.get('radius',float(np.linalg.norm(np.array(start)-center)) if kind=='arc' else '')
        values=[p['id'],WALLS.get(p['wall'],p['wall']),KINDS.get(kind,kind),*start,*end,*center,radius,'',p.get('unit','mm')]
        for col,v in enumerate(values):self.item(self.primitive_table,row,col,v,col!=0 and (kind!='point' or col not in (5,6)) and (kind=='arc' or col not in (7,8,9,10)),check=p.get('clockwise',False) if col==10 and kind=='arc' else None)
    def refresh_primitives(self,previous):
        # Coordinate edits keep the existing scene and unrelated table rows.
        # Rebuilding 140 nozzle/property rows on every keystroke caused lag.
        self.loading=True;self.primitive_table.blockSignals(True);self.object_table.blockSignals(True)
        try:
            changed_walls=set()
            for row,(old,raw) in enumerate(zip(previous['primitives'],self.raw['primitives'])):
                if old==raw:continue
                self.fill_primitive_row(row,raw);p=Primitive(raw);changed_walls.update((old['wall'],p.wall))
                if p.id in self.paths:
                    points=p.discretize(self.raw.get('chord_error_mm',.01));path=QPainterPath(QPointF(points[0,0],-points[0,1]))
                    for z,r in points[1:]:path.lineTo(z,-r)
                    self.paths[p.id].setPath(path)
                    self.paths[p.id].setPen(QPen(QColor({'inner':'#316cb7','outer':'#bb6740','reference':'#8a73b3'}[p.wall]),0))
                coordinates={'start':p.start,'end':p.end}
                if p.kind=='arc':coordinates['center']=p.center
                for h in self.handles:
                    if not h.object_handle and h.identity==p.id:
                        h.setPos(QPointF(coordinates[h.role][0],-coordinates[h.role][1]));h.setBrush(QBrush(QColor({'inner':'#316cb7','outer':'#bb6740','reference':'#8a73b3'}[p.wall])))
            model=Chamber(self.raw)
            for section in self.raw['sections']:
                identity=section['id'];z=section['z_mm'];rin,rout=model.bounds(z)
                if identity in self.section_lines:self.section_lines[identity].setLine(z,-rin,z,-rout)
                if identity in self.section_labels:self.section_labels[identity].setPos(z,-rout)
            for row,obj in enumerate(self.raw['objects']):
                if obj.get('wall','free') not in changed_walls:continue
                z,r,theta=model.object_position(obj,self.working_nozzles,self.cfg)
                for col,value in zip((4,5,6),(z,r,theta)):self.item(self.object_table,row,col,value,col!=5 or obj.get('wall','free')=='free')
                point=(r*math.cos(math.radians(theta)),-r*math.sin(math.radians(theta))) if self.angular.isChecked() else (z,-r)
                for h in self.handles:
                    if h.object_handle and h.identity==obj['id']:h.setPos(QPointF(*point))
            rect=self.sketch.scene().itemsBoundingRect();current=self.sketch.sceneRect()
            if not current.contains(rect):self.sketch.scene().setSceneRect(current.united(rect).adjusted(-3,-3,3,3))
            self.sketch.viewport().update()
        finally:self.primitive_table.blockSignals(False);self.object_table.blockSignals(False);self.loading=False
    def fill_tables(self):
        for table in self.tables:table.blockSignals(True)
        try:
            table=self.primitive_table;table.setRowCount(len(self.raw['primitives']))
            for row,p in enumerate(self.raw['primitives']):
                self.fill_primitive_row(row,p)
            table=self.object_table;table.setRowCount(len(self.raw['objects']));model=None
            try:model=Chamber(self.raw)
            except ValueError:pass
            for row,obj in enumerate(self.raw['objects']):
                if obj['type']=='nozzle':
                    n=next(n for n in self.working_nozzles if n.id==obj['nozzle_id']);from .v02 import effective_nozzle
                    effective=effective_nozzle(n,self.cfg);position=(effective.z_tip_mm,n.radius_mm,n.theta_deg)
                else:
                    try:position=model.object_position(obj,self.working_nozzles,self.cfg) if model else (obj['z_mm'],obj['radius_mm'],obj.get('theta_deg',0))
                    except ValueError:position=(obj['z_mm'],obj['radius_mm'],obj.get('theta_deg',0))
                group=next((g['id'] for g in self.raw['groups'] if obj['id'] in g['members']),'')
                values=[obj['id'],obj['type'],WALLS.get(obj.get('wall','free'),obj.get('wall')) ,obj.get('primitive_id',''),*position,group,*[json.dumps(obj.get(key,{}),ensure_ascii=False) for key in ('parameters','units','provenance')]]
                for col,v in enumerate(values):self.item(table,row,col,v,col not in (0,7) and not (obj['type']=='nozzle' and col==1) and not (col==5 and obj.get('wall','free')!='free'))
            table=self.group_table;table.setRowCount(len(self.raw['groups']))
            for row,g in enumerate(self.raw['groups']):
                values=[g['id'],';'.join(g['members']),MODES.get(g['mode'],g['mode']),g.get('phase_deg',0),g.get('direction',-1),g.get('row_offset_deg',0),';'.join(format(v,'.12g') for v in g.get('steps_deg' if g['mode']=='steps' else 'angles_deg',[]))]
                for col,v in enumerate(values):self.item(table,row,col,v,col!=0 and not(col==6 and g['mode']=='uniform'))
            table=self.section_table;table.setRowCount(len(self.raw['sections']))
            for row,s in enumerate(self.raw['sections']):
                self.item(table,row,0,s['id'],False);self.item(table,row,1,s['z_mm']);self.item(table,row,2,'',check=s.get('is_outlet',False))
            for table in self.tables:table.resizeColumnsToContents()
        finally:
            for table in self.tables:table.blockSignals(False)
    def cell(self,table,row,col):return table.item(row,col).text().strip()
    def value(self,table,row,col):
        item=table.item(row,col);text=item.text().replace(',','.');raw=item.data(Qt.UserRole)
        if isinstance(raw,(int,float)) and text==format(raw,'.12g'):return float(raw)
        result=float(text)
        if not math.isfinite(result):raise ValueError('Нужно конечное число')
        return result
    @staticmethod
    def key(text,choices):return next((key for key,label in choices.items() if text in (key,label)),text)
    def table_edited(self,table,item):
        if self.loading:return
        if table.in_batch:self.pending_batch.setdefault(table,{})[item.row()]=item.column();return
        row=item.row();col=item.column()
        try:
            if table is self.primitive_table:
                p=self.raw['primitives'][row];old=deepcopy(p);p['wall']=self.key(self.cell(table,row,1),WALLS);p['kind']=self.key(self.cell(table,row,2),KINDS)
                unit=self.cell(table,row,11)
                if unit not in UNITS:raise ValueError('Единицы mm/cm/m')
                if col==11:
                    factor=UNITS[p.get('unit','mm')]/UNITS[unit]
                    for key in ('start','end','center'):
                        if key in p:p[key]=[v*factor for v in p[key]]
                    if 'radius' in p:p['radius']*=factor
                else:
                    p['start']=[self.value(table,row,3),self.value(table,row,4)]
                    if p['kind']!='point':p['end']=[self.value(table,row,5),self.value(table,row,6)] if old['kind']!='point' else [p['start'][0]+10,p['start'][1]]
                    if p['kind']=='arc':
                        if old['kind']!='arc':p.update(self.arc_data(p['start'],p['end']));p.pop('radius',None)
                        else:
                            p['center']=[self.value(table,row,7),self.value(table,row,8)];p['radius']=self.value(table,row,9);p['clockwise']=table.item(row,10).checkState()==Qt.Checked
                            if col in (3,4,5,6,7,8):p.pop('radius',None)
                    if col in (3,4,5,6):
                        role='start' if col<5 else 'end';self.join_ends(old[role],p[role],p['wall'],p['id'],UNITS[p.get('unit','mm')])
                p['unit']=unit
            elif table is self.object_table:
                obj=self.raw['objects'][row];obj['type']=self.cell(table,row,1);obj['wall']=self.key(self.cell(table,row,2),WALLS);obj['primitive_id']=self.cell(table,row,3)
                z,r,theta=[self.value(table,row,c) for c in (4,5,6)]
                if obj['type']=='nozzle':self.update_nozzle(obj,z,r,theta)
                else:obj.update(z_mm=z,radius_mm=r,theta_deg=theta%360)
                for c,key in zip((8,9,10),('parameters','units','provenance')):obj[key]=json.loads(self.cell(table,row,c))
                self.raw=sync_groups(self.raw,self.working_nozzles)
            elif table is self.group_table:
                g=self.raw['groups'][row];g.update(members=[s.strip() for s in self.cell(table,row,1).split(';') if s.strip()],mode=self.key(self.cell(table,row,2),MODES),phase_deg=self.value(table,row,3),direction=self.value(table,row,4),row_offset_deg=self.value(table,row,5))
                if g['mode']!='uniform':g['steps_deg' if g['mode']=='steps' else 'angles_deg']=[float(s.strip().replace(',','.')) for s in self.cell(table,row,6).split(';') if s.strip()]
            else:
                s=self.raw['sections'][row];s['z_mm']=self.value(table,row,1)
                if col==2:
                    if table.item(row,2).checkState()==Qt.Checked:
                        for other in self.raw['sections']:other['is_outlet']=other is s
                        self.set_outlet(s['z_mm'])
                    else:s['is_outlet']=False
                elif s.get('is_outlet',False):self.set_outlet(s['z_mm'])
            if not self.batch_mode:self.commit('Геометрия: правка таблицы',groups=table is self.group_table)
        except (ValueError,TypeError,KeyError,StopIteration) as exc:
            self.batch_error=str(exc);self.message('Черновик не применён: '+str(exc),True)
            if not self.batch_mode:self.draw_scene()
    def batch_finished(self,table):
        pending=self.pending_batch.pop(table,{})
        if not pending:return
        self.batch_mode=True;self.batch_error=''
        try:
            for row,col in pending.items():self.table_edited(table,table.item(row,col))
        finally:self.batch_mode=False
        if not self.batch_error:self.commit('Геометрия: вставка ячеек',groups=table is self.group_table)
        else:self.draw_scene()
    def update_nozzle(self,obj,z,r,theta):
        from .v02 import effective_nozzle
        n=next(n for n in self.working_nozzles if n.id==obj['nozzle_id'])
        if z!=effective_nozzle(n,self.cfg).z_tip_mm:
            n.z_tip_mm=z
            if 'z_tip_mm' not in n.geometry_overrides:n.geometry_overrides=list(n.geometry_overrides)+['z_tip_mm']
        if r!=n.radius_mm:n.radius_mm=r
        if (theta-n.theta_deg+180)%360-180!=0:n.theta_deg=theta%360
    def join_ends(self,old,new,wall,except_id,factor=1.):
        a=np.array(old)*factor;b=np.array(new)*factor
        for p in self.raw['primitives']:
            if p['id']==except_id or p['wall']!=wall:continue
            scale=UNITS[p.get('unit','mm')]
            for role in ('start','end'):
                if role in p and np.allclose(np.array(p[role])*scale,a,atol=TOL,rtol=0):p[role]=(b/scale).tolist();p.pop('radius',None)
    def set_outlet(self,z):
        old=self.raw['outlet_z_mm'];self.raw['outlet_z_mm']=z
        for p in self.raw['primitives']:
            factor=UNITS[p.get('unit','mm')]
            for role in ('start','end'):
                if role in p and abs(p[role][0]*factor-old)<=TOL:p[role][0]=z/factor;p.pop('radius',None)
        for s in self.raw['sections']:
            if s.get('is_outlet',False):s['z_mm']=z
    def config_edited(self):
        if self.loading:return
        self.raw['enabled']=self.enabled.isChecked();old=self.raw['base_z_mm'];self.raw.update(base_z_mm=self.base.value(),internal_step_mm=self.step.value(),chord_error_mm=self.error.value());self.set_outlet(self.outlet.value())
        for p in self.raw['primitives']:
            factor=UNITS[p.get('unit','mm')]
            for role in ('start','end'):
                if role in p and abs(p[role][0]*factor-old)<=TOL:p[role][0]=self.base.value()/factor;p.pop('radius',None)
        self.commit('Параметры геометрии')
    def commit(self,label,groups=False):
        try:
            model=Chamber(self.raw)
            if groups:self.raw,self.working_nozzles=apply_groups(self.raw,self.working_nozzles)
            current=deepcopy(self.owner.nozzles);base={n.id:n for n in self.base_nozzles};working={n.id:n for n in self.working_nozzles}
            for nozzle in current:
                for key in ('theta_deg','radius_mm','z_tip_mm'):
                    if getattr(working[nozzle.id],key)!=getattr(base[nozzle.id],key):
                        setattr(nozzle,key,getattr(working[nozzle.id],key))
                        if key=='z_tip_mm':nozzle.geometry_overrides=sorted(set(nozzle.geometry_overrides)|{'z_tip_mm'})
            if model.enabled and any(o['type']=='nozzle' and o.get('wall','free')!='free' for o in model.objects):current=model.resolve_nozzle_anchors(current,self.owner.cfg)
            state=project_to_json(self.owner.cfg,current);state['settings']['geometry']=deepcopy(self.raw)
            if model.enabled:state['settings']['z_mm']=[s['z_mm'] for s in self.raw['sections']]
            # Individual/group geometry edits deliberately preserve all other
            # positions; the old global Auto-follow must not overwrite them.
            if self.raw['groups']:state['settings']['placement_mode']='manual'
            self.owner.apply_geometry_state(state,label)
            self.message(('Применено. '+WALL_BOUND_NOTICE) if model.variable and model.enabled else 'Применено. Сектора 1–36 и расходные характеристики сохранены.')
            return True
        except (ValueError,TypeError,KeyError,StopIteration) as exc:
            self.message('Черновик не применён: '+str(exc),True);self.fill_tables();self.draw_scene();return False
    def add_primitive(self,kind,points=None):
        radius=(self.cfg.r_in_mm+self.cfg.r_out_mm)/2;start,end=points if points is not None else ([self.raw['base_z_mm'],radius],[min(self.raw['outlet_z_mm'],self.raw['base_z_mm']+10),radius+5])
        p=dict(id=self.uid('p'),wall='reference',kind=kind,start=list(start),unit='mm')
        if kind!='point':p['end']=list(end)
        if kind=='arc':p.update(self.arc_data(start,end))
        self.raw['primitives'].append(p);self.selected=p['id'];self.commit('Добавить '+KINDS[kind])
    @staticmethod
    def arc_data(start,end):
        a,b=np.array(start,dtype=float),np.array(end,dtype=float);delta=b-a
        if np.linalg.norm(delta)<=TOL:raise ValueError('Разные концы дуги обязательны')
        for scale in (.5,2,10,100,1000):
            center=(a+b)/2+scale*np.array([-delta[1],delta[0]])
            t0=math.atan2(*(a-center)[::-1]);t1=math.atan2(*(b-center)[::-1]);clockwise=(t1-t0)%(2*math.pi)>math.pi
            result=dict(center=center.tolist(),clockwise=clockwise)
            try:Primitive(dict(id='arc',wall='reference',kind='arc',start=a.tolist(),end=b.tolist(),**result));return result
            except ValueError:continue
        raise ValueError('Невозможно построить однозначную дугу между точками')
    def split_primitive(self):
        row=self.primitive_table.currentRow()
        if row<0:return
        p=self.raw['primitives'][row];primitive=Primitive(p)
        if primitive.kind=='point':return
        z=(primitive.lo+primitive.hi)/2;middle=[z/UNITS[primitive.unit],float(primitive.at(z))/UNITS[primitive.unit]]
        first,second=deepcopy(p),deepcopy(p);first['end']=middle;second['start']=middle;second['id']=self.uid('p')
        self.raw['primitives'][row:row+1]=[first,second];self.commit('Разбить примитив')
    def add_object(self,kind,point=None):
        if kind=='custom':
            kind,ok=QInputDialog.getText(self,'Пользовательский тип','Название типа:')
            if not ok or not kind.strip():return
            kind=kind.strip();self.raw.setdefault('types',{}).setdefault(kind,{'label':kind})
        z,r=point if point else ((self.raw['base_z_mm']+self.raw['outlet_z_mm'])/2,(self.cfg.r_in_mm+self.cfg.r_out_mm)/2)
        obj=dict(id=self.uid(kind),type=kind,wall='free',z_mm=z,radius_mm=r,theta_deg=0.,parameters=dict(inclination_deg=None,azimuth_deg=None,flow_kg_s=None,penetration_mm=None),units=dict(length='mm',angle='deg',flow='kg/s'),provenance={})
        self.raw['objects'].append(obj);self.selected=obj['id'];self.pages.setCurrentIndex(1);self.commit('Добавить объект')
    def add_group(self):
        used={m for g in self.raw['groups'] for m in g['members']};available=[o for o in self.raw['objects'] if o['id'] not in used]
        selected={self.raw['objects'][i.row()]['id'] for i in self.object_table.selectedIndexes()}
        members=[o['id'] for o in available if o['id'] in selected] or [o['id'] for o in available if o['type']=='nozzle' and o['nozzle_id']<=70]
        if not members:self.message('Сначала добавьте/выделите объекты, не включённые в другую группу',True);return
        self.raw['groups'].append(dict(id=self.uid('g'),members=members,mode='uniform',phase_deg=0.,direction=-1,row_offset_deg=0.));self.commit('Добавить угловую группу',groups=True)
    def add_section(self):
        zs=sorted({self.raw['base_z_mm'],self.raw['outlet_z_mm'],*[s['z_mm'] for s in self.raw['sections']]});a,b=max(zip(zs,zs[1:]),key=lambda pair:pair[1]-pair[0]);self.raw['sections'].append(dict(id=self.uid('S'),z_mm=(a+b)/2,is_outlet=False));self.commit('Добавить сечение')
    def move_section(self,delta):
        row=self.section_table.currentRow();other=row+delta
        if row<0 or not 0<=other<len(self.raw['sections']):return
        self.raw['sections'][row],self.raw['sections'][other]=self.raw['sections'][other],self.raw['sections'][row];self.commit('Порядок ручных сечений')
    def delete(self,table):
        rows=sorted({i.row() for i in table.selectedIndexes()},reverse=True)
        key={self.primitive_table:'primitives',self.object_table:'objects',self.group_table:'groups',self.section_table:'sections'}[table]
        for row in rows:
            record=self.raw[key][row]
            if key=='objects' and record['type']=='nozzle':self.message('140 позиций форсунок сохраняются; отключайте их на 02',True);return
            if key=='sections' and record.get('is_outlet',False):self.message('Выходное сечение обязательно',True);return
        for row in rows:del self.raw[key][row]
        self.commit('Удалить геометрические объекты')
    def select(self,identity,object_handle=False):
        previous=self.loading;self.loading=True
        self.selected=identity;table=self.object_table if object_handle else self.primitive_table;records=self.raw['objects' if object_handle else 'primitives']
        row=next((i for i,p in enumerate(records) if p['id']==identity),None)
        if row is not None:
            table.blockSignals(True);table.setCurrentCell(row,0);table.blockSignals(False);self.pages.setCurrentIndex(1 if object_handle else 0)
        for item in self.handles:item.setSelected(item.identity==identity);item.setZValue(10 if item.identity==identity else 8)
        self.loading=previous
    def table_selected(self,table,row):
        if self.loading or row<0:return
        key={self.primitive_table:'primitives',self.object_table:'objects',self.group_table:'groups',self.section_table:'sections'}[table]
        if row>=len(self.raw[key]):return
        record=self.raw[key][row];self.selected=record['id'];members=set(record.get('members',[record['id']]))
        self.loading=True
        for handle in self.handles:handle.setSelected(handle.identity in members);handle.setZValue(10 if handle.identity in members else 8)
        self.loading=False
    def scene_selected(self):
        if self.loading:return
        items=self.sketch.scene().selectedItems()
        if items and isinstance(items[-1],Handle):
            item=items[-1]
            if self.selected!=item.identity:self.select(item.identity,item.object_handle)
    def draw_scene(self,fit=False):
        if not self.isVisible():self._pending=True;return
        old=self.loading;self.loading=True
        try:
            scene=self.sketch.scene();scene.clear();self.paths={};self.handles=[];self.section_lines={};self.section_labels={};angular=self.angular.isChecked();model=None
            try:model=Chamber(self.raw)
            except ValueError:pass
            if not angular:
                for raw in self.raw['primitives']:
                    try:p=Primitive(raw);points=p.discretize(self.raw.get('chord_error_mm',.01))
                    except ValueError:continue
                    path=QPainterPath(QPointF(points[0,0],-points[0,1]))
                    for z,r in points[1:]:path.lineTo(z,-r)
                    color={'inner':'#316cb7','outer':'#bb6740','reference':'#8a73b3'}[p.wall];item=scene.addPath(path,QPen(QColor(color),0));self.paths[p.id]=item
                    for role,point in [('start',p.start)]+([] if p.kind=='point' else [('end',p.end)])+([('center',p.center)] if p.kind=='arc' else []):
                        handle=Handle(self,p.id,role,(point[0],-point[1]),color);scene.addItem(handle);self.handles.append(handle)
                for section in self.raw['sections']:
                    try:rin,rout=model.bounds(section['z_mm']) if model else (self.cfg.r_in_mm,self.cfg.r_out_mm)
                    except ValueError:continue
                    pen=QPen(QColor('#a36bac' if section.get('is_outlet') else '#aab4c5'),0,Qt.DashLine);self.section_lines[section['id']]=scene.addLine(section['z_mm'],-rin,section['z_mm'],-rout,pen)
                    label=scene.addSimpleText(section['id']+(' · выход' if section.get('is_outlet') else ''));label.setFlag(QGraphicsItem.ItemIgnoresTransformations);label.setPos(section['z_mm'],-rout)
                    self.section_labels[section['id']]=label
            for obj in self.raw['objects']:
                try:
                    if model:z,r,theta=model.object_position(obj,self.working_nozzles,self.cfg)
                    elif obj['type']=='nozzle':
                        n=next(n for n in self.working_nozzles if n.id==obj['nozzle_id']);z,r,theta=n.z_tip_mm,n.radius_mm,n.theta_deg
                    else:z,r,theta=obj['z_mm'],obj['radius_mm'],obj.get('theta_deg',0)
                except (ValueError,StopIteration):continue
                point=(r*math.cos(math.radians(theta)),-r*math.sin(math.radians(theta))) if angular else (z,-r)
                color={'nozzle':'#61866a','hole':'#2a81b3','insert':'#b34e91'}.get(obj['type'],'#777777')
                handle=Handle(self,obj['id'],'position',point,color,True);scene.addItem(handle);self.handles.append(handle)
            rect=scene.itemsBoundingRect();pad=max(3.,max(rect.width(),rect.height())*.08);scene.setSceneRect(rect.adjusted(-pad,-pad,pad,pad))
            for handle in self.handles:handle.setSelected(handle.identity==self.selected);handle.setZValue(10 if handle.identity==self.selected else 8)
            if fit:self.fit()
        finally:self.loading=old
    def fit(self):self.sketch.fitInView(self.sketch.sceneRect(),Qt.KeepAspectRatio)
    def preview_drag(self,identity,role,position,is_object):
        # Keep Qt's active handle alive throughout the drag. Tables and paths
        # reflect valid analytic coordinates; invalid drafts get diagnosed only
        # at release and are rolled back as one action.
        try:
            self.change_position(identity,role,position,is_object)
            self.refresh_drag_values(identity,is_object)
            if not is_object:
                p=Primitive(next(p for p in self.raw['primitives'] if p['id']==identity));points=p.discretize(self.raw.get('chord_error_mm',.01));path=QPainterPath(QPointF(points[0,0],-points[0,1]))
                for z,r in points[1:]:path.lineTo(z,-r)
                if identity in self.paths:self.paths[identity].setPath(path)
        except (ValueError,KeyError):pass
    def refresh_drag_values(self,identity,is_object):
        table=self.object_table if is_object else self.primitive_table;table.blockSignals(True)
        try:
            records=self.raw['objects' if is_object else 'primitives'];row=next(i for i,p in enumerate(records) if p['id']==identity);record=records[row]
            if is_object:
                model=Chamber(self.raw);z,r,theta=model.object_position(record,self.working_nozzles,self.cfg)
                for col,value in zip((4,5,6),(z,r,theta)):self.item(table,row,col,value,col!=5 or record.get('wall','free')=='free')
            else:
                for role,columns in [('start',(3,4)),('end',(5,6)),('center',(7,8))]:
                    if role in record:
                        for col,value in zip(columns,record[role]):self.item(table,row,col,value)
                if record['kind']=='arc':self.item(table,row,9,float(np.linalg.norm(np.array(record['start'])-record['center'])))
        finally:table.blockSignals(False)
    def change_position(self,identity,role,position,is_object):
        z,r=position.x(),-position.y()
        if is_object:
            obj=next(o for o in self.raw['objects'] if o['id']==identity)
            model=Chamber(self.raw);oldz,oldr,oldtheta=model.object_position(obj,self.working_nozzles,self.cfg)
            if self.angular.isChecked():z,r,theta=oldz,oldr,math.degrees(math.atan2(-position.y(),position.x()))%360
            else:theta=oldtheta
            if obj['type']=='nozzle':self.update_nozzle(obj,z,r,theta)
            else:obj.update(z_mm=z,radius_mm=r,theta_deg=theta)
            self.raw=sync_groups(self.raw,self.working_nozzles);return
        p=next(p for p in self.raw['primitives'] if p['id']==identity);factor=UNITS[p.get('unit','mm')];primitive=Primitive(p)
        if role=='center':
            midpoint=(primitive.start+primitive.end)/2;normal=np.array([-(primitive.end-primitive.start)[1],(primitive.end-primitive.start)[0]]);normal/=np.linalg.norm(normal)
            center=midpoint+normal*np.dot(np.array([z,r])-midpoint,normal);p['center']=(center/factor).tolist();p.pop('radius',None)
        else:
            old=p[role][:]
            if abs(old[0]*factor-self.raw['base_z_mm'])<=TOL:z=self.raw['base_z_mm']
            elif abs(old[0]*factor-self.raw['outlet_z_mm'])<=TOL:z=self.raw['outlet_z_mm']
            if primitive.kind=='arc':
                delta=np.array([z,r])-primitive.center;length=np.linalg.norm(delta)
                if length<=TOL:return
                z,r=primitive.center+primitive.radius*delta/length
            # Snap to an existing same-wall endpoint within eight screen pixels.
            for other in self.raw['primitives']:
                if other['id']==identity or other['wall']!=p['wall']:continue
                for key in ('start','end'):
                    if key not in other:continue
                    candidate=np.array(other[key])*UNITS[other.get('unit','mm')]
                    a=self.sketch.mapFromScene(QPointF(z,-r));b=self.sketch.mapFromScene(QPointF(candidate[0],-candidate[1]))
                    if (a-b).manhattanLength()<=8:z,r=candidate
            p[role]=[z/factor,r/factor];p.pop('radius',None);self.join_ends(old,p[role],p['wall'],identity,factor)
    def finish_drag(self,identity,role,position,is_object,before,before_nozzles):
        try:
            self.change_position(identity,role,position,is_object);Chamber(self.raw)
            if not self.commit('Sketcher: переместить '+identity):raise ValueError(self.status.text())
        except (ValueError,KeyError) as exc:
            self.raw=before;self.working_nozzles=before_nozzles;self.rebuild();self.message('Перемещение отменено: '+str(exc),True)
    def sketch_add(self,point):
        tool=self.tool.currentIndex();position=[float(point.x()),float(-point.y())]
        if tool==0:return
        try:
            if tool==1:self.add_primitive('point',(position,position))
            elif tool in (4,5):self.add_object('hole' if tool==4 else 'insert',position)
            elif self._first is None:self._first=position;self.message('Задайте вторую точку двойным ЛКМ')
            else:
                first=self._first;self._first=None;self.add_primitive('line' if tool==2 else 'arc',(first,position))
        except ValueError as exc:self.message(str(exc),True)
