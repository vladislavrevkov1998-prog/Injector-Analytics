"""Peer Sketcher/table editor. Accepted and draft edits share the Undo stack.

The editable draft may temporarily have an open joint; it never enters a
calculation until validated. Dragging commits once, at mouse release.
"""
from copy import deepcopy
from functools import wraps
import json,math
from pathlib import Path
import numpy as np
from PyQt5.QtCore import Qt,QPointF,QRectF
from PyQt5.QtGui import QColor,QPen,QBrush,QPainterPath,QPainter,QPainterPathStroker
from PyQt5.QtWidgets import (QWidget,QVBoxLayout,QHBoxLayout,QSplitter,QTabWidget,
    QTableWidgetItem,QPushButton,QLabel,QComboBox,QCheckBox,QGraphicsView,
    QGraphicsScene,QGraphicsPathItem,QGraphicsEllipseItem,QGraphicsItem,QGraphicsSimpleTextItem,
    QStyledItemDelegate,QInputDialog,QDoubleSpinBox,QDialog,QTextBrowser,QDialogButtonBox,QFileDialog)
from .chamber import Chamber,Primitive,legacy_geometry,apply_groups,sync_groups,group_angles,UNITS,TOL,WALL_BOUND_NOTICE,number
from .core import project_to_json
from .numeric_input import DecimalSpinBox
from .numeric_delegate import NumericDelegate
from .chamber_editing import next_id,WALL_PREFIX,move_endpoint,translate,resize_arc,preview
from .chamber_import_gui import PointImportDialog
from .help_text import populate_help
from .chamber_constraints import (constraint_records,materialize_constraints,apply_constraints,
    add_constraint,remove_constraints,tangent_references)
from .chamber_preview import draft_bounds
from .chamber_constraints_gui import ConstraintDialog
from .chamber_sections import sync_section_outlet,set_section_outlet
from . import chamber_history

WALLS={'inner':'Внутренняя','outer':'Наружная','reference':'Опорная','free':'Свободно'}
KINDS={'point':'Точка','line':'Отрезок','arc':'Дуга'}
MODES={'uniform':'Равномерно','individual':'Индивидуальные углы','steps':'Массив шагов'}
PRIMITIVE_COLUMNS=['ID','Стенка','Тип','z₁','R₁','z₂','R₂','z центра','R центра','Радиус','По ЧС','Единицы','Зафиксировать\nкривую']
OBJECT_COLUMNS=['ID','Тип','Стенка','Примитив','z, мм','R, мм','θ, °','Группа','Параметры (JSON)','Единицы (JSON)','Происхождение (JSON)']
GROUP_COLUMNS=['ID','Объекты (через ;)','Режим','Фаза, °','Направление ±1','Смещение ряда, °','Углы/шаги, ° (через ;)']


def geometry_edit(method):
    """All geometry entry points share the same Air-mode mutation guard."""
    @wraps(method)
    def guarded(self,*args,**kwargs):
        if getattr(self,'is_air_mode',False):
            if self.loading:return False
            self.message('Геометрия заморожена. Для изменения переключитесь в режим «Геометрия».')
            if method.__name__ in ('table_edited','batch_finished','config_edited'):
                previous=self.loading;self.loading=True
                try:
                    self.enabled.setChecked(self.raw.get('enabled',False));self.base.setValue(self.raw['base_z_mm']);self.outlet.setValue(self.raw['outlet_z_mm']);self.step.setValue(self.raw.get('internal_step_mm',1.));self.error.setValue(self.raw.get('chord_error_mm',.01))
                    self.pending_batch.clear();self.fill_tables();self.apply_geometry_freeze()
                finally:self.loading=previous
            return False
        # PyQt trims signal arguments for a fixed-signature slot. Preserve that
        # behavior after wrapping (clicked/toggled would otherwise pass bool).
        positional=args if method.__code__.co_flags & 4 else args[:method.__code__.co_argcount-1]
        return method(self,*positional,**kwargs)
    return guarded


class Choices(QStyledItemDelegate):
    def __init__(self,values,parent):super().__init__(parent);self.values=values
    def createEditor(self,parent,option,index):
        box=QComboBox(parent)
        for key,label in self.values.items():box.addItem(label,key)
        return box
    def setEditorData(self,editor,index):editor.setCurrentIndex(max(0,editor.findText(index.data())))
    def setModelData(self,editor,model,index):model.setData(index,editor.currentText())


class DragGesture:
    def mousePressEvent(self,event):
        if event.button()!=Qt.LeftButton:event.ignore();return
        self.constraint_click=False
        if self.editor.is_air_mode:
            self.constraint_click=True;self.editor.select(self.identity,self.object_handle);event.accept();return
        if self.editor.constraint_mode:
            self.constraint_click=True
            if not self.object_handle and (self.editor.constraint_mode=='tangent' or self.role in ('start','end')):self.editor.constraint_pick(self.identity,self.role)
            else:self.editor.message('Для связи выберите круглый маркер начала или конца кривой.')
            event.accept();return
        if not self.object_handle and next(p for p in self.editor.raw['primitives'] if p['id']==self.identity).get('fixed',False):
            self.constraint_click=True;self.editor.select(self.identity);self.editor.message(self.identity+': кривая зафиксирована. Снимите флажок для перемещения.');event.accept();return
        materialize_constraints(self.editor.raw)
        self.before=deepcopy(self.editor.raw);self.before_nozzles=deepcopy(self.editor.working_nozzles)
        self.press_position=event.scenePos();self.moved=False
        self.anchor=self.pos() if self.role!='body' else event.scenePos()
        self.editor.select(self.identity,self.object_handle);event.accept()
    def mouseMoveEvent(self,event):
        if getattr(self,'constraint_click',False):event.accept();return
        self.moved=True
        self.editor.preview_drag(self.identity,self.role,self.anchor+event.scenePos()-self.press_position,
            self.object_handle,self.before,self.before_nozzles,self.anchor)
        event.accept()
    def mouseReleaseEvent(self,event):
        if getattr(self,'constraint_click',False):event.accept();return
        if getattr(self,'moved',False):
            self.editor.preview_drag(self.identity,self.role,self.anchor+event.scenePos()-self.press_position,
                self.object_handle,self.before,self.before_nozzles,self.anchor)
            self.editor.finish_drag(self.identity,self.role,self.pos(),self.object_handle,self.before,self.before_nozzles)
        event.accept()


class Handle(DragGesture,QGraphicsEllipseItem):
    def __init__(self,editor,identity,role,point,color,object_handle=False):
        super().__init__(-5,-5,10,10);self.editor=editor;self.identity=identity;self.role=role;self.object_handle=object_handle
        self.setBrush(QBrush(QColor(color)));self.setPen(QPen(QColor('#ffffff'),1));self.setZValue(8)
        self.setFlags(QGraphicsItem.ItemIsSelectable|QGraphicsItem.ItemIgnoresTransformations)
        self.setPos(QPointF(*point));self.setData(0,identity);self.setToolTip(identity+' · '+{'start':'начало: переместить точку','end':'конец: переместить точку','center':'центр: перенести дугу целиком','position':'положение объекта'}.get(role,role))
        self.setCursor(Qt.SizeAllCursor)
        self.endpoint_label=None
        if not object_handle and role in ('start','end') and next(p for p in editor.raw['primitives'] if p['id']==identity)['kind'] in ('line','arc'):
            label=QGraphicsSimpleTextItem('1' if role=='start' else '2',self)
            font=label.font();font.setPointSizeF(8.5);label.setFont(font);label.setBrush(QBrush(QColor('#737b85')))
            label.setPos(7,-18 if role=='start' else 4);label.setAcceptedMouseButtons(Qt.NoButton)
            label.setData(1,'endpoint-number');label.setToolTip(('1 — начало' if role=='start' else '2 — конец')+' · '+identity);self.endpoint_label=label
    def boundingRect(self):return QRectF(-9,-9,18,18)
    def shape(self):
        path=QPainterPath();path.addEllipse(self.boundingRect());return path
    def paint(self,painter,option,widget=None):
        color=self.brush().color();painter.setRenderHint(QPainter.Antialiasing)
        if self.isSelected():
            glow=QColor(color);glow.setAlpha(45);painter.setPen(Qt.NoPen);painter.setBrush(glow);painter.drawEllipse(QRectF(-9,-9,18,18))
        painter.setPen(QPen(color,1.7));painter.setBrush(QColor('white'))
        if self.role=='center':
            painter.drawRect(QRectF(-4,-4,8,8));painter.drawLine(QPointF(-7,0),QPointF(7,0));painter.drawLine(QPointF(0,-7),QPointF(0,7))
        else:
            if self.object_handle:painter.setBrush(color)
            painter.drawEllipse(QRectF(-4,-4,8,8))


class PrimitivePath(DragGesture,QGraphicsPathItem):
    def __init__(self,editor,identity,path):
        super().__init__(path);self.editor=editor;self.identity=identity;self.role='body';self.object_handle=False
        self.setCursor(Qt.SizeAllCursor);self.setToolTip(identity+' · перетащите линию для переноса целиком')
    def shape(self):
        stroker=QPainterPathStroker();stroker.setWidth(12/max(self.editor.sketch.transform().m11(),1e-9))
        return stroker.createStroke(self.path())
    def boundingRect(self):return self.shape().boundingRect()


class Sketch(QGraphicsView):
    def __init__(self,editor):
        super().__init__(QGraphicsScene(editor),editor);self.editor=editor;self.setRenderHint(QPainter.Antialiasing)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse);self.setMinimumSize(220,220)
        self._pan=None;self.pointer=None;self.setMouseTracking(True);self.setBackgroundBrush(QColor('#fafbfe'))
        self.setProperty('ffSketchMode','geometry')
        self.setStyleSheet('QGraphicsView { border:3px solid #28a35d; border-radius:4px; }')
    def wheelEvent(self,event):
        factor=1.18 if event.angleDelta().y()>0 else 1/1.18
        if 1e-4<self.transform().m11()*factor<1e6:self.scale(factor,factor)
        for item in self.editor.paths.values():item.prepareGeometryChange()
        event.accept()
    def grid_step(self):
        target=65/max(self.transform().m11(),1e-9);power=10**math.floor(math.log10(target))
        return next(v*power for v in (1,2,5,10) if v*power>=target)
    def drawBackground(self,painter,rect):
        super().drawBackground(painter,rect)
        if not self.editor.grid.isChecked():return
        step=self.grid_step();pen=QPen(self.palette().mid().color(),0);painter.setPen(pen)
        for axis in (0,1):
            low,high=(rect.left(),rect.right()) if axis==0 else (rect.top(),rect.bottom())
            for i in range(math.ceil(low/step),math.floor(high/step)+1):
                v=i*step
                painter.drawLine(QPointF(v,rect.top()),QPointF(v,rect.bottom())) if axis==0 else painter.drawLine(QPointF(rect.left(),v),QPointF(rect.right(),v))
    def drawForeground(self,painter,rect):
        painter.save();painter.resetTransform();painter.setPen(self.palette().text().color());width=self.viewport().width();height=self.viewport().height()
        if self.editor.grid.isChecked():
            step=self.grid_step();visible=self.mapToScene(self.viewport().rect()).boundingRect()
            for i in range(math.ceil(visible.left()/step),math.floor(visible.right()/step)+1):
                x=self.mapFromScene(QPointF(i*step,0)).x()
                if 55<x<width-35:painter.drawText(QPointF(x+3,height-28),format(i*step,'.6g'))
            for i in range(math.ceil(visible.top()/step),math.floor(visible.bottom()/step)+1):
                y=self.mapFromScene(QPointF(0,i*step)).y()
                if 45<y<height-45:painter.drawText(QPointF(6,y-3),format(-i*step,'.6g'))
        hud=self.palette().base().color();hud.setAlpha(235)
        painter.fillRect(QRectF(0,0,width,26),hud);painter.setPen(self.palette().text().color())
        painter.drawText(QPointF(9,18),'X/Y [мм] · θ против ЧС' if self.editor.angular.isChecked() else 'z → [мм]     R ↑ [мм]')
        if self.editor.selected:
            text=self.editor.selected;painter.drawText(QPointF(max(170,width-painter.fontMetrics().horizontalAdvance(text)-12),18),text)
        if self.pointer is not None:
            p=self.pointer;painter.fillRect(QRectF(0,height-23,width,23),hud)
            painter.drawText(QPointF(9,height-7),f'{"X" if self.editor.angular.isChecked() else "z"} = {p.x():.3f}    {"Y" if self.editor.angular.isChecked() else "R"} = {-p.y():.3f} мм')
        if self.editor._first is not None and self.pointer is not None:
            a=self.editor._first;painter.setPen(QPen(QColor('#8a73b3'),1,Qt.DashLine));painter.drawLine(self.mapFromScene(QPointF(a[0],-a[1])),self.mapFromScene(self.pointer))
        painter.restore()
    def mousePressEvent(self,event):
        if event.button()==Qt.MiddleButton:self._pan=event.pos();self.setCursor(Qt.ClosedHandCursor);event.accept();return
        super().mousePressEvent(event)
    def mouseMoveEvent(self,event):
        self.pointer=self.mapToScene(event.pos());self.viewport().update()
        if self._pan is not None:
            delta=event.pos()-self._pan;self._pan=event.pos();self.horizontalScrollBar().setValue(self.horizontalScrollBar().value()-delta.x());self.verticalScrollBar().setValue(self.verticalScrollBar().value()-delta.y());event.accept();return
        super().mouseMoveEvent(event)
    def leaveEvent(self,event):self.pointer=None;self.viewport().update();super().leaveEvent(event)
    def keyPressEvent(self,event):
        if event.key()==Qt.Key_Escape:self.editor.cancel_construction();self.viewport().update();self.editor.message('Построение / выбор связи отменены.');event.accept();return
        super().keyPressEvent(event)
    def mouseReleaseEvent(self,event):
        if event.button()==Qt.MiddleButton:self._pan=None;self.unsetCursor();event.accept();return
        super().mouseReleaseEvent(event)
    def mouseDoubleClickEvent(self,event):
        if self.editor.is_air_mode:event.accept();return
        if event.button()==Qt.LeftButton and not self.editor.angular.isChecked():self.editor.sketch_add(self.mapToScene(event.pos()));event.accept();return
        super().mouseDoubleClickEvent(event)


class GeometryEditor(QWidget):
    def __init__(self,owner,table_class):
        super().__init__(owner);self.owner=owner;self.loading=True;self.selected=None;self._pending=True;self.paths={};self.handles=[];self._first=None;self.batch_mode=False;self.pending_batch={};self._draft_invalid=False;self.constraint_mode=None;self._constraint_first=None;self.is_air_mode=False
        layout=QVBoxLayout(self);top=QHBoxLayout();self.enabled=QCheckBox('Профили ЖТ');self.enabled.setToolTip('Включить переменные профили стенок в расчёт');top.addWidget(self.enabled)
        self.base=self.spin(0);self.outlet=self.spin(60);self.step=self.spin(1,.0001);self.error=self.spin(.01,.000001)
        for label,widget in [('База z, мм:',self.base),('Выход, мм:',self.outlet),('Шаг, мм:',self.step),('Хорда ≤, мм:',self.error)]:top.addWidget(QLabel(label));top.addWidget(widget)
        top.addStretch()
        self.step.setToolTip('Внутренний шаг геометрических станций');self.error.setToolTip('Максимальная стрела прогиба отрезков при отображении точной дуги')
        layout.addLayout(top)
        note=QLabel('z=x [мм]; θ — от физической +X против ЧС. Изменения сразу применяются после проверки. Несогласованный черновик не участвует в расчёте.');note.setWordWrap(True);layout.addWidget(note)
        controls=QHBoxLayout();self.mode_selector=QComboBox();self.mode_selector.addItem('Геометрия','geometry');self.mode_selector.addItem('Воздушные потоки','air');controls.addWidget(QLabel('Режим:'));controls.addWidget(self.mode_selector)
        self.angular=QCheckBox('Вид по окружности');controls.addWidget(self.angular)
        self.grid=QCheckBox('Сетка');self.grid.setChecked(True);self.grid.toggled.connect(lambda:self.sketch.viewport().update());controls.addWidget(self.grid)
        fit=QPushButton('Вписать');fit.clicked.connect(self.fit);controls.addWidget(fit)
        creation=QHBoxLayout();self.tool=QComboBox();self.tool.addItems(['Выбор','Точка','Отрезок (2 точки)','Дуга (2 точки)','Отверстие','Вставка']);creation.addWidget(QLabel('Двойной ЛКМ:'));creation.addWidget(self.tool)
        self.new_wall=QComboBox()
        for key in ('reference','inner','outer'):self.new_wall.addItem(WALLS[key],key)
        creation.addWidget(QLabel('Стенка:'));creation.addWidget(self.new_wall);creation.addStretch()
        self.coincident_button=self.button(creation,'Совпадение точек',lambda:self.start_constraint('coincident'))
        self.tangent_button=self.button(creation,'Касательность',lambda:self.start_constraint('tangent'))
        for button in (self.coincident_button,self.tangent_button):button.setCheckable(True)
        self.button(creation,'Связи…',lambda:self.pages.setCurrentIndex(4));layout.addLayout(creation);controls.addStretch()
        reset=QPushButton('Сбросить черновик');reset.clicked.connect(self.reset_draft);controls.addWidget(reset)
        guide=QPushButton('Инструкция');guide.clicked.connect(self.show_guide);controls.addWidget(guide)
        undo=QPushButton('↶');redo=QPushButton('↷');undo.clicked.connect(lambda:self.undo_geometry(False));redo.clicked.connect(lambda:self.undo_geometry(True));controls.addWidget(undo);controls.addWidget(redo);layout.addLayout(controls)
        for button,tip in ((undo,'Отменить последний шаг, включая черновик или его сброс (Ctrl+Z)'),(redo,'Повторить изменение (Ctrl+Y)'),(fit,'Вписать геометрию в окно'),(reset,'Вернуть последний применённый вариант. Ctrl+Z восстановит черновик.')):button.setToolTip(tip)
        undo.setFixedWidth(30);redo.setFixedWidth(30)
        self.tool.currentIndexChanged.connect(self.cancel_construction)
        hint=QLabel('Точка — изменить конец · квадрат центра / линия — перенос целиком · колесо — масштаб · средняя кнопка — панорама · Esc — отменить построение')
        hint.setWordWrap(True);hint.setStyleSheet('color:#596780; font-size:11px;');layout.addWidget(hint)
        self.splitter=QSplitter(Qt.Horizontal);self.sketch=Sketch(self);self.splitter.addWidget(self.sketch);self.pages=QTabWidget();self.splitter.addWidget(self.pages);self.splitter.setSizes([550,650]);layout.addWidget(self.splitter,1)
        self.primitive_table=table_class(0,len(PRIMITIVE_COLUMNS));self.object_table=table_class(0,len(OBJECT_COLUMNS));self.group_table=table_class(0,len(GROUP_COLUMNS));self.section_table=table_class(0,3)
        self.tables=[self.primitive_table,self.object_table,self.group_table,self.section_table]
        self.primitive_table.horizontalHeader().moveSection(12,1)
        for tab,title,headers in zip(self.tables,['Примитивы','Объекты','Угловые группы','Расчётные сечения'],[PRIMITIVE_COLUMNS,OBJECT_COLUMNS,GROUP_COLUMNS,['ID','z=x, мм','Выход ЖТ (авто)']]):
            panel=QWidget();box=QVBoxLayout(panel);tab.setHorizontalHeaderLabels(headers);tab.setSelectionBehavior(tab.SelectItems);tab.setMinimumWidth(300);box.addWidget(tab);buttons=QHBoxLayout();box.addLayout(buttons);self.pages.addTab(panel,title)
            if tab is self.primitive_table:
                for title,callback in [('Точка',lambda:self.add_primitive('point')),('Отрезок',lambda:self.add_primitive('line')),('Дуга',lambda:self.add_primitive('arc')),('Разбить',self.split_primitive)]:self.button(buttons,title,callback)
                self.import_points_button=self.button(buttons,'Точки XLSX…',self.import_points)
                extra=QHBoxLayout();self.button(extra,'Согласовать дугу по радиусу',self.repair_arc);self.button(extra,'Исправить направление дуги',self.repair_arc_direction);extra.addStretch();box.addLayout(extra)
            elif tab is self.object_table:
                for kind,title in [('hole','Отверстие'),('insert','Вставка'),('custom','Другой тип')]:self.button(buttons,title,lambda checked=False,k=kind:self.add_object(k))
            elif tab is self.group_table:self.button(buttons,'Добавить группу',self.add_group)
            else:
                note=QLabel('Выход ЖТ = максимальное z сечений. Внутренние срезы не меняют стенки; новый выход достраивает профили по касательной, сохраняя существующую часть.');note.setWordWrap(True);box.insertWidget(0,note)
                self.button(buttons,'Добавить сечение',self.add_section);self.button(buttons,'↑',lambda:self.move_section(-1));self.button(buttons,'↓',lambda:self.move_section(1))
            self.button(buttons,'Удалить',lambda checked=False,t=tab:self.delete(t));buttons.addStretch()
            tab.itemChanged.connect(lambda item,t=tab:self.table_edited(t,item))
            tab.batchFinished.connect(lambda t=tab:self.batch_finished(t))
            tab.currentCellChanged.connect(lambda row,col,oldrow,oldcol,t=tab:self.table_selected(t,row))
        panel=QWidget();box=QVBoxLayout(panel);note=QLabel('Связи сохраняются в проекте. Совпадение переносит первую точку ко второй; касательность сохраняет отрезок и подгоняет дугу. Удаление связи оставляет текущую форму.');note.setWordWrap(True);box.addWidget(note)
        self.constraint_table=table_class(0,6);self.constraint_table.setHorizontalHeaderLabels(['ID','Связь','Кривая A','Точка A','Кривая B','Точка B']);box.addWidget(self.constraint_table)
        buttons=QHBoxLayout();self.button(buttons,'Совпадение…',lambda:self.constraint_dialog('coincident'));self.button(buttons,'Касательность…',lambda:self.constraint_dialog('tangent'));self.button(buttons,'Удалить связь',self.delete_constraints);buttons.addStretch();box.addLayout(buttons);self.pages.addTab(panel,'Связи')
        for column,values in ((1,{k:v for k,v in WALLS.items() if k!='free'}),(2,KINDS),(11,{k:k for k in UNITS})):
            self.primitive_table.setItemDelegateForColumn(column,Choices(values,self.primitive_table))
        self.object_table.setItemDelegateForColumn(2,Choices({k:v for k,v in WALLS.items() if k!='reference'},self.object_table));self.group_table.setItemDelegateForColumn(2,Choices(MODES,self.group_table))
        for column in (1,6):self.group_table.setItemDelegateForColumn(column,QStyledItemDelegate(self.group_table))
        numeric=NumericDelegate(self)
        for column in range(3,10):self.primitive_table.setItemDelegateForColumn(column,numeric)
        for column in (4,5,6):self.object_table.setItemDelegateForColumn(column,numeric)
        self.section_table.setItemDelegateForColumn(1,numeric)
        self.status=QLabel();self.status.setWordWrap(True);self.status.setFixedHeight(44);layout.addWidget(self.status)
        self.enabled.toggled.connect(self.config_edited)
        for widget in (self.base,self.outlet,self.step,self.error):widget.editingFinished.connect(self.config_edited)
        self.angular.toggled.connect(lambda:self.draw_scene(True));self.sketch.scene().selectionChanged.connect(self.scene_selected)
        self._geometry_controls=[self.enabled,self.base,self.outlet,self.step,self.error,self.tool,self.new_wall,reset,undo,redo]
        self._creation_widgets=[creation.itemAt(i).widget() for i in range(creation.count()) if creation.itemAt(i).widget() is not None]
        self._geometry_hint=hint;self._geometry_hint_text=hint.text()
        self._geometry_controls += [button for button in self.findChildren(QPushButton) if button not in (fit,guide,reset,undo,redo)]
        self._geometry_triggers={table:table.editTriggers() for table in self.tables+[self.constraint_table]}
        self._geometry_flag_cache={}
        from .air_sketcher import AirFlowPanel
        self.air_panel=AirFlowPanel(self);self.air_page_index=self.pages.addTab(self.air_panel,'Воздушные потоки')
        self.pages.setTabEnabled(self.air_page_index,False)
        self.mode_selector.currentIndexChanged.connect(self.change_edit_mode)
        self.loading=False

    def apply_mode_border(self):
        name='air' if self.is_air_mode else 'geometry'
        self.sketch.setProperty('ffSketchMode',name)
        color='#2c85dc' if self.is_air_mode else '#28a35d'
        self.sketch.setStyleSheet('QGraphicsView { border:3px solid '+color+'; border-radius:4px; }')
        self.sketch.style().unpolish(self.sketch);self.sketch.style().polish(self.sketch);self.sketch.viewport().update()

    def undo_geometry(self,redo=False):
        if self.is_air_mode:
            self.message('История геометрии заморожена в режиме воздушных потоков.');return
        (self.owner.undo_stack.redo if redo else self.owner.undo_stack.undo)()

    def change_edit_mode(self,*args):
        air=self.mode_selector.currentData()=='air'
        if air and self._draft_invalid:
            self.mode_selector.blockSignals(True);self.mode_selector.setCurrentIndex(0);self.mode_selector.blockSignals(False)
            self.message('Сначала согласуйте или сбросьте черновик геометрии: воздух использует только принятую геометрию.',True);return
        self.cancel_construction();self.is_air_mode=air
        for widget in self._geometry_controls:widget.setEnabled(not air)
        for widget in self._creation_widgets:widget.setVisible(not air)
        self._geometry_hint.setText('Выберите объект или воздушный поток · параметры обновляют векторы и пунктир · колесо — масштаб · средняя кнопка — панорама' if air else self._geometry_hint_text)
        for index in range(self.air_page_index):self.pages.setTabEnabled(index,not air)
        self.pages.setTabEnabled(self.air_page_index,air)
        self.pages.setCurrentIndex(self.air_page_index if air else 0)
        self.apply_geometry_freeze();self.apply_mode_border()
        if air:self.air_panel.set_context()
        self.draw_scene()
        self.message('Воздушные потоки: геометрия заморожена; изменения воздуха сначала видны в черновике.' if air else 'Геометрия: редактирование разблокировано.')

    def apply_geometry_freeze(self):
        for table in self.tables+[self.constraint_table]:
            table.setEditTriggers(table.NoEditTriggers if self.is_air_mode else self._geometry_triggers[table])
            blocked=table.blockSignals(True)
            try:
                for row in range(table.rowCount()):
                    for column in range(table.columnCount()):
                        item=table.item(row,column)
                        if item is None:continue
                        key=(table,row,column)
                        if self.is_air_mode:
                            self._geometry_flag_cache.setdefault(key,item.flags());item.setFlags(item.flags()&~(Qt.ItemIsEditable|Qt.ItemIsUserCheckable))
                        elif key in self._geometry_flag_cache:item.setFlags(self._geometry_flag_cache.pop(key))
            finally:table.blockSignals(blocked)

    def refresh_air_model(self):
        if not hasattr(self,'cfg'):return
        self.cfg=self.owner.cfg
        if self.is_air_mode:self.air_panel.set_context()

    @staticmethod
    def spin(value,minimum=0):
        spin=DecimalSpinBox();spin.setDecimals(8);spin.setRange(minimum,1e6);spin.setSingleStep(.001);spin.setValue(value);spin.setMaximumWidth(115);return spin
    @staticmethod
    def button(layout,title,callback):
        button=QPushButton(title);button.clicked.connect(callback);layout.addWidget(button);return button
    def uid(self,prefix):return next_id(self.raw,prefix)
    def cancel_construction(self):
        self._first=None;self.constraint_mode=None;self._constraint_first=None;self.coincident_button.setChecked(False);self.tangent_button.setChecked(False);self.sketch.viewport().update()
    @geometry_edit
    def reset_draft(self):
        before=chamber_history.snapshot(self)
        self.cancel_construction();self.load(self.owner.cfg,self.owner.nozzles,force=True)
        chamber_history.record(self,'Sketcher: сбросить черновик',before=before)
    def load(self,cfg,nozzles,force=False):
        candidate=cfg.geometry or legacy_geometry(cfg,nozzles)
        if not cfg.geometry:
            # Only the newly generated starter sketch receives new IDs.
            # Saved geometry keeps every existing identity and external link.
            for p in candidate['primitives']:p['id']=WALL_PREFIX[p['wall']]+'1'
            for index,section in enumerate(candidate['sections'],1):section['id']='S'+str(index)
        if not force and hasattr(self,'base_nozzles') and nozzles==self.base_nozzles and candidate==self.loaded_raw and cfg.nozzle_presets==self.loaded_presets:
            self.cfg=cfg
            if self.is_air_mode:self.air_panel.set_context()
            return
        incremental=False
        if not force and not self._draft_invalid and hasattr(self,'loaded_raw') and self.isVisible() and not self._pending and nozzles==self.base_nozzles and cfg.nozzle_presets==self.loaded_presets:
            old=self.loaded_raw
            incremental=({k:v for k,v in old.items() if k not in ('primitives','constraints','schema_version')}=={k:v for k,v in candidate.items() if k not in ('primitives','constraints','schema_version')} and
                [(p['id'],p['kind']) for p in old['primitives']]==[(p['id'],p['kind']) for p in candidate['primitives']])
        if incremental:
            old=self.loaded_raw;self.cfg=cfg;self.loaded_raw=deepcopy(candidate);self.raw=deepcopy(candidate);self._draft_cell_errors={}
            self.refresh_primitives(old)
            if old.get('constraints')!=candidate.get('constraints'):
                affected={ref['primitive_id'] for c in old.get('constraints',[])+candidate.get('constraints',[]) for ref in (c['a'],c['b'])}
                self.primitive_table.blockSignals(True)
                try:
                    for row,p in enumerate(self.raw['primitives']):
                        if p['id'] in affected:self.fill_primitive_row(row,p)
                finally:self.primitive_table.blockSignals(False)
            self.fill_constraints();self.sync_constraint_glyphs();chamber_history.checkpoint(self);return
        self.cfg=cfg;self.loaded_raw=deepcopy(candidate);self.loaded_presets=deepcopy(cfg.nozzle_presets);self.working_nozzles=deepcopy(nozzles);self.base_nozzles=deepcopy(nozzles);self.raw=deepcopy(candidate);self._pending=True;self._draft_invalid=False;self._draft_cell_errors={}
        self.message('Геометрия загружена. Несогласованный черновик не участвует в расчёте.')
        if self.isVisible():self.rebuild()
        if self.is_air_mode:self.air_panel.set_context()
        chamber_history.checkpoint(self)
    def showEvent(self,event):
        super().showEvent(event)
        if self._pending and hasattr(self,'raw'):self.rebuild()
    def message(self,text,error=False):self.status.setText(text);self.status.setToolTip(text);self.status.setStyleSheet('color:#a32138' if error else 'color:#52617c')
    def show_guide(self):
        self.guide_dialog=QDialog(self);self.guide_dialog.setWindowTitle('Геометрия ЖТ v0.6 — инструкция');self.guide_dialog.resize(900,750)
        layout=QVBoxLayout(self.guide_dialog);browser=QTextBrowser();populate_help(browser,'V0.6_GUIDE.md')
        layout.addWidget(browser);buttons=QDialogButtonBox(QDialogButtonBox.Close);buttons.rejected.connect(self.guide_dialog.close);layout.addWidget(buttons);self.guide_dialog.show()
    def rebuild(self):
        self.loading=True
        try:
            self.enabled.setChecked(self.raw.get('enabled',False));self.base.setValue(self.raw['base_z_mm']);self.outlet.setValue(self.raw['outlet_z_mm']);self.step.setValue(self.raw.get('internal_step_mm',1));self.error.setValue(self.raw.get('chord_error_mm',.01))
            self.fill_tables();self.draw_scene(not bool(self.paths));self._pending=False
        finally:self.loading=False
        if self.is_air_mode:self.apply_geometry_freeze()
    def item(self,table,row,column,value,editable=True,check=None):
        item=QTableWidgetItem(format(value,'.12g') if isinstance(value,(int,float)) and not isinstance(value,bool) else str(value))
        if isinstance(value,(int,float)) and not isinstance(value,bool):item.setData(Qt.UserRole,value)
        if check is not None:item.setText('');item.setFlags(Qt.ItemIsEnabled|Qt.ItemIsSelectable|Qt.ItemIsUserCheckable);item.setCheckState(Qt.Checked if check else Qt.Unchecked)
        elif not editable:item.setFlags(item.flags()&~(Qt.ItemIsEditable|Qt.ItemIsUserCheckable));item.setBackground(QColor('#eee8f6'))
        table.setItem(row,column,item)
    def fill_primitive_row(self,row,p):
        start=p['start'];end=p.get('end',['','']);center=p.get('center',['','']);kind=p['kind']
        radius=p.get('radius',float(np.linalg.norm(np.array(start)-center)) if kind=='arc' else '')
        values=[p['id'],WALLS.get(p['wall'],p['wall']),KINDS.get(kind,kind),*start,*end,*center,radius,'',p.get('unit','mm'),'']
        derived=set()
        if kind=='arc':
            for c in self.raw.get('constraints',[]):
                if c['kind']!='tangent' or not c.get('enabled',True):continue
                pair=[next(q for q in self.raw['primitives'] if q['id']==c[k]['primitive_id']) for k in ('a','b')]
                for ref in (c['a'],c['b']):
                    if all(q['kind']=='arc' for q in pair) and ref is c['b']:continue
                    if ref['primitive_id']==p['id']:derived.update((7,8,9,10,*( (3,4) if ref['endpoint']=='start' else (5,6))))
        for col,v in enumerate(values):
            editable=col!=0 and (kind!='point' or col not in (5,6)) and (kind=='arc' or col not in (7,8,9,10)) and not(p.get('fixed',False) and col in range(2,11)) and col not in derived
            check=p.get('fixed',False) if col==12 else p.get('clockwise',False) if col==10 and kind=='arc' else None
            self.item(self.primitive_table,row,col,v,editable,check=check)
            if (p.get('fixed',False) or col in derived) and col==10:self.primitive_table.item(row,col).setFlags(Qt.ItemIsEnabled|Qt.ItemIsSelectable)
            if col in derived:self.primitive_table.item(row,col).setToolTip('Определяется касательностью. Измените свободный конец дуги или удалите связь на странице «Связи».')
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
            rect=self.scene_bounds();current=self.sketch.sceneRect()
            if not current.contains(rect):self.sketch.scene().setSceneRect(current.united(rect).adjusted(-3,-3,3,3))
            self.highlight_selection()
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
                self.item(table,row,0,s['id'],False);self.item(table,row,1,s['z_mm']);self.item(table,row,2,'Да' if s.get('is_outlet',False) else '',False)
            for table in self.tables:table.resizeColumnsToContents()
            self.fill_constraints()
        finally:
            for table in self.tables:table.blockSignals(False)
            chamber_history.restore_cells(self)
    def cell(self,table,row,col):return table.item(row,col).text().strip()
    def fill_constraints(self):
        records=constraint_records(self.raw);self.constraint_table.setRowCount(len(records))
        for row,c in enumerate(records):
            values=[c['id'],('Совпадение' if c['kind']=='coincident' else 'Касание + совпадение')+(' (выкл.)' if not c.get('enabled',True) else ''),c['a']['primitive_id'],c['a']['endpoint'],c['b']['primitive_id'],c['b']['endpoint']]
            for col,value in enumerate(values):self.item(self.constraint_table,row,col,value,False)
        self.constraint_table.resizeColumnsToContents()
    @geometry_edit
    def start_constraint(self,kind):
        previous=self.constraint_mode;self.cancel_construction()
        if previous==kind:return
        self.constraint_mode=kind;self.coincident_button.setChecked(kind=='coincident');self.tangent_button.setChecked(kind=='tangent')
        self.message(('Совпадение: выберите перемещаемую точку, затем точку назначения.' if kind=='coincident' else 'Касательность: щёлкните по двум кривым. Отрезок сохраняется; для двух дуг первая подгоняется ко второй (опорной). Общая точка определяется автоматически.')+' Esc — отмена.')
    @geometry_edit
    def constraint_pick(self,identity,role):
        if self.constraint_mode=='tangent':
            if next(p for p in self.raw['primitives'] if p['id']==identity)['kind']=='point':self.message('Выберите отрезок или дугу.');return
            self.select(identity)
            if self._constraint_first is None:
                self._constraint_first=identity;self.message('Выбрана кривая '+identity+'. Выберите вторую кривую (её линию, а не общую точку).');return
            first=self._constraint_first
            if first==identity:self.message('Выберите другую кривую; можно щёлкнуть по её линии вдали от общей точки.');return
            try:a,b=tangent_references(self.raw,first,identity)
            except ValueError as exc:self.message('Касательность: '+str(exc),True);return
            if self.create_constraint('tangent',a,b):self.cancel_construction()
            return
        self.select(identity);reference=dict(primitive_id=identity,endpoint=role)
        if self._constraint_first is None:self._constraint_first=reference;self.message('Выбрана '+identity+'.'+role+'. Теперь выберите вторую точку.');return
        first=self._constraint_first;kind=self.constraint_mode;self._constraint_first=None
        if self.create_constraint(kind,first,reference):self.cancel_construction()
    @geometry_edit
    def constraint_dialog(self,kind):
        dialog=ConstraintDialog(self.raw,kind,self);self.connection_dialog=dialog
        if dialog.exec_()==QDialog.Accepted:
            try:
                a,b=dialog.references()
                self.create_constraint(kind,a,b)
            except ValueError as exc:self.message('Касательность: '+str(exc),True)
    @geometry_edit
    def create_constraint(self,kind,a,b):
        before=deepcopy(self.raw)
        try:
            self.raw=add_constraint(self.raw,kind,a,b)
            if kind=='tangent':
                for p in self.raw['primitives']:
                    if p['id'] in (a['primitive_id'],b['primitive_id']) and p['kind']=='line':p['fixed']=True
            self.commit('Sketcher: '+('совпадение точек' if kind=='coincident' else 'касательность'))
            self.fill_constraints();return True
        except (ValueError,TypeError,KeyError) as exc:
            self.raw=before;self.message('Связь не создана: '+str(exc),True);return False
    @geometry_edit
    def delete_constraints(self):
        rows={i.row() for i in self.constraint_table.selectedIndexes()};records=constraint_records(self.raw)
        if not rows:return
        self.raw=remove_constraints(self.raw,[records[row]['id'] for row in rows]);self.commit('Sketcher: удалить связи');self.fill_constraints()
    @geometry_edit
    def repair_arc(self):
        row=self.primitive_table.currentRow()
        if row<0:return
        p=self.raw['primitives'][row]
        if p['kind']!='arc':self.message('Выберите дугу в таблице.');return
        if p.get('fixed',False):self.message(p['id']+': сначала снимите фиксацию.',True);return
        try:
            materialize_constraints(self.raw);p.update(resize_arc(p,self.value(self.primitive_table,row,9)))
            for clockwise in (p.get('clockwise',False),not p.get('clockwise',False)):
                p['clockwise']=clockwise
                try:Primitive(p);break
                except ValueError:continue
            self.commit('Sketcher: согласовать дугу по радиусу')
        except ValueError as exc:self.message('Дуга не согласована: '+str(exc),True)
    @geometry_edit
    def repair_arc_direction(self):
        row=self.primitive_table.currentRow()
        if row<0:return
        p=self.raw['primitives'][row]
        if p['kind']!='arc':self.message('Выберите дугу в таблице.');return
        if p.get('fixed',False):self.message(p['id']+': сначала снимите фиксацию.',True);return
        try:Primitive(p);self.message('Направление уже задаёт однозначную дугу R(z).');return
        except ValueError:pass
        candidate=deepcopy(p);candidate['clockwise']=not candidate.get('clockwise',False)
        try:Primitive(candidate)
        except ValueError as exc:self.message('Смена направления не устраняет ошибку: '+str(exc),True);return
        p.update(candidate);self.commit('Sketcher: исправить направление дуги')
    def value(self,table,row,col):
        item=table.item(row,col);text=item.text().replace(',','.');raw=item.data(Qt.UserRole)
        if isinstance(raw,(int,float)) and text==format(raw,'.12g'):return float(raw)
        result=float(text)
        if not math.isfinite(result):raise ValueError('Нужно конечное число')
        return result
    @staticmethod
    def key(text,choices):return next((key for key,label in choices.items() if text in (key,label)),text)
    @geometry_edit
    def table_edited(self,table,item,columns=None):
        if self.loading:return
        if table.in_batch:self.pending_batch.setdefault(table,{}).setdefault(item.row(),set()).add(item.column());return
        row=item.row();col=item.column()
        columns=columns or {col}
        identity=table.item(row,0).text()
        for column in columns:self._draft_cell_errors.pop((self.tables.index(table),identity,column),None)
        try:
            if table is self.primitive_table:
                materialize_constraints(self.raw);p=self.raw['primitives'][row];old=deepcopy(p)
                if p.get('fixed',False) and columns.intersection(range(2,11)):
                    self.fill_tables();self.message(p['id']+': кривая зафиксирована; снимите флажок.',True);return
                if columns=={12}:p['fixed']=table.item(row,12).checkState()==Qt.Checked
                elif columns=={1}:p['wall']=self.key(self.cell(table,row,1),WALLS)
                else:
                    unit=self.cell(table,row,11)
                    if unit not in UNITS:raise ValueError('Единицы mm/cm/m')
                    factor=UNITS[old.get('unit','mm')]/UNITS[unit];converted=deepcopy(old)
                    for key in ('start','end','center'):
                        if key in converted:converted[key]=[v*factor for v in converted[key]]
                    if 'radius' in converted:converted['radius']*=factor
                    converted['unit']=unit
                    p.update(converted);p['unit']=unit;p['wall']=self.key(self.cell(table,row,1),WALLS);p['kind']=self.key(self.cell(table,row,2),KINDS)
                    for role,cols in [('start',(3,4)),('end',(5,6)),('center',(7,8))]:
                        if role=='end' and p['kind']=='point' or role=='center' and p['kind']!='arc':continue
                        values=list(p.get(role,[p['start'][0]+10,p['start'][1]]))
                        for index,column in enumerate(cols):
                            if column in columns:values[index]=self.value(table,row,column)
                        p[role]=values
                    if p['kind']=='arc':
                        if old['kind']!='arc':
                            if not {7,8}.issubset(columns):p.update(self.arc_data(p['start'],p['end']))
                        p['clockwise']=table.item(row,10).checkState()==Qt.Checked
                        shape=columns.intersection(range(3,10))
                        if 9 in shape:p.update(resize_arc(p,self.value(table,row,9)))
                        elif old['kind']=='arc' and shape and shape.issubset({3,4}):p.update(move_endpoint(converted,'start',p['start']))
                        elif old['kind']=='arc' and shape and shape.issubset({5,6}):p.update(move_endpoint(converted,'end',p['end']))
                        elif old['kind']=='arc' and shape and shape.issubset({7,8}):p.update(translate(converted,np.array(p['center'])-converted['center']))
                        elif shape:p.pop('radius',None)
                        if shape and 9 not in shape:p.pop('radius',None)
                    else:
                        for key in ('center','radius','clockwise'):p.pop(key,None)
                        if p['kind']=='point':p.pop('end',None)
                    p['wall']=self.key(self.cell(table,row,1),WALLS)
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
                self.raw['sections'][row]['z_mm']=self.value(table,row,1)
                self.sync_section_outlet()
            if not self.batch_mode:self.commit('Геометрия: правка таблицы',groups=table is self.group_table,sections=table is self.section_table)
        except (ValueError,TypeError,KeyError,StopIteration) as exc:
            self.batch_error=str(exc);self._draft_invalid=True;self.message('Черновик не применён: '+str(exc),True)
            for column in columns:self._draft_cell_errors[(self.tables.index(table),identity,column)]=self.cell(table,row,column)
            chamber_history.restore_cells(self)
            if not self.batch_mode:
                self.draw_scene();chamber_history.record(self,'Геометрия: правка черновика')
    @geometry_edit
    def batch_finished(self,table):
        pending=self.pending_batch.pop(table,{})
        if not pending:return
        self.batch_mode=True;self.batch_error=''
        try:
            for row,columns in pending.items():self.table_edited(table,table.item(row,max(columns)),columns)
        finally:self.batch_mode=False
        if not self.batch_error:self.commit('Геометрия: вставка ячеек',groups=table is self.group_table,sections=table is self.section_table)
        else:
            self.draw_scene();chamber_history.record(self,'Геометрия: вставка в черновик')
    @geometry_edit
    def update_nozzle(self,obj,z,r,theta):
        from .v02 import effective_nozzle
        n=next(n for n in self.working_nozzles if n.id==obj['nozzle_id'])
        if z!=effective_nozzle(n,self.cfg).z_tip_mm:
            n.z_tip_mm=z
            if 'z_tip_mm' not in n.geometry_overrides:n.geometry_overrides=list(n.geometry_overrides)+['z_tip_mm']
        if r!=n.radius_mm:n.radius_mm=r
        if (theta-n.theta_deg+180)%360-180!=0:n.theta_deg=theta%360
    def sync_section_outlet(self):self.raw=sync_section_outlet(self.raw)
    @geometry_edit
    def set_outlet(self,z):
        if abs(z-self.raw['outlet_z_mm'])>TOL:self.raw=set_section_outlet(self.raw,z)
    @geometry_edit
    def config_edited(self):
        if self.loading:return
        before=deepcopy(self.raw)
        try:
            self.raw['enabled']=self.enabled.isChecked();old=self.raw['base_z_mm'];self.raw.update(base_z_mm=self.base.value(),internal_step_mm=self.step.value(),chord_error_mm=self.error.value());self.set_outlet(self.outlet.value())
            if abs(old-self.base.value())>TOL:
                materialize_constraints(self.raw)
                for p in self.raw['primitives']:
                    factor=UNITS[p.get('unit','mm')]
                    for role in ('start','end'):
                        if role in p and abs(p[role][0]*factor-old)<=TOL:
                            moved=move_endpoint(p,role,[self.base.value()/factor,p[role][1]]);p.clear();p.update(moved)
            self.commit('Параметры геометрии')
        except ValueError as exc:self.raw=before;self.rebuild();self.message('Параметры не применены: '+str(exc),True)
    @geometry_edit
    def commit(self,label,groups=False,sections=False,record_failure=True):
        history_before=self._draft_checkpoint if hasattr(self,'_draft_checkpoint') else chamber_history.snapshot(self)
        project_before=project_to_json(self.owner.cfg,self.owner.nozzles)
        try:
            chamber_history.prune_cells(self)
            if self._draft_cell_errors:
                table_index,identity,col=next(iter(self._draft_cell_errors));header=self.tables[table_index].horizontalHeaderItem(col).text().replace('\n',' ')
                raise ValueError(identity+': исправьте ошибочное значение в столбце «'+header+'» или сбросьте черновик')
            materialize_constraints(self.raw)
            previous={p['id']:p for p in self.loaded_raw['primitives']};drivers=[]
            for p in self.raw['primitives']:
                if p['id'] not in previous:continue
                old=previous[p['id']]
                for role in ('start','end'):
                    if role in p and role in old and not np.allclose(np.array(p[role])*UNITS[p.get('unit','mm')],np.array(old[role])*UNITS[old.get('unit','mm')],rtol=0,atol=TOL):drivers.append((p['id'],role))
            self.raw=apply_constraints(self.raw,drivers=drivers,baseline=self.loaded_raw)
            if self.raw['sections'] and max(number(s['z_mm']) for s in self.raw['sections'])>number(self.raw['outlet_z_mm'])+TOL:
                self.raw=sync_section_outlet(self.raw)
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
            if model.enabled or sections:state['settings']['z_mm']=[s['z_mm'] for s in self.raw['sections']]
            # Individual/group geometry edits deliberately preserve all other
            # positions; the old global Auto-follow must not overwrite them.
            if self.raw['groups']:state['settings']['placement_mode']='manual'
            self.owner.apply_geometry_state(state,label,record_history=False)
            if self._draft_invalid:self.load(self.owner.cfg,self.owner.nozzles,force=True)
            self.message('Применено. Связи и размеры сохранены; пояснение wall-bound — в «Инструкции».' if model.variable and model.enabled else 'Применено. Сектора 1–36 и расходные характеристики сохранены.')
            chamber_history.record(self,label,before=history_before,project_before=project_before)
            return True
        except (ValueError,TypeError,KeyError,StopIteration) as exc:
            self._draft_invalid=True;self.message('Черновик не применён: '+str(exc),True);self.fill_tables();self.draw_scene()
            if record_failure:chamber_history.record(self,label,before=history_before,project_before=project_before)
            return False
    @geometry_edit
    def import_points(self):
        path,_=QFileDialog.getOpenFileName(self,'Точки геометрии','','Excel (*.xlsx)')
        if not path:return
        try:
            dialog=PointImportDialog(path,self);self.points_import_dialog=dialog
            if dialog.exec_()!=QDialog.Accepted:return
            candidate=deepcopy(self.raw);wall=dialog.wall.currentData();first=None
            for row,z,r in dialog.points:
                identity=next_id(candidate,WALL_PREFIX[wall]);first=first or identity
                candidate['primitives'].append(dict(id=identity,wall=wall,kind='point',start=[z,r],unit='mm',provenance=dict(source_kind='imported',file=Path(path).name,sheet=dialog.sheet.currentText(),excel_row=row,source_unit=dialog.unit.currentText())))
            Chamber(candidate)
            self.raw=candidate;self.selected=first
            if self.commit('Импорт точек XLSX'):self.fit()
        except (ValueError,OSError) as exc:self.message('Импорт не выполнен: '+str(exc),True)
    @geometry_edit
    def add_primitive(self,kind,points=None):
        radius=(self.cfg.r_in_mm+self.cfg.r_out_mm)/2;start,end=points if points is not None else ([self.raw['base_z_mm'],radius],[min(self.raw['outlet_z_mm'],self.raw['base_z_mm']+10),radius+5])
        wall=self.new_wall.currentData();p=dict(id=self.uid(WALL_PREFIX[wall]),wall=wall,kind=kind,start=list(start),unit='mm')
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
    @geometry_edit
    def split_primitive(self):
        row=self.primitive_table.currentRow()
        if row<0:return
        try:
            materialize_constraints(self.raw)
            p=self.raw['primitives'][row];primitive=Primitive(p)
            if primitive.kind=='point':return
            if p.get('fixed',False):self.message(p['id']+': снимите фиксацию перед разбиением.',True);return
            model=Chamber(self.raw);z=(primitive.lo+primitive.hi)/2;middle=[z/UNITS[primitive.unit],float(primitive.at(z))/UNITS[primitive.unit]]
            first,second=deepcopy(p),deepcopy(p);first['end']=middle;second['start']=middle;second['id']=self.uid(WALL_PREFIX[p['wall']])
            first_geometry=Primitive(first);second_geometry=Primitive(second)
            for obj in self.raw['objects']:
                if obj.get('primitive_id')!=p['id']:continue
                oz=model.object_position(obj,self.working_nozzles,self.cfg)[0]
                if not first_geometry.lo-TOL<=oz<=first_geometry.hi+TOL and second_geometry.lo-TOL<=oz<=second_geometry.hi+TOL:obj['primitive_id']=second['id']
            for c in self.raw['constraints']:
                for key in ('a','b'):
                    ref=c[key]
                    if ref['primitive_id']==p['id'] and ref['endpoint']=='end':ref['primitive_id']=second['id']
            self.raw['constraints'].append(dict(id=next_id({**self.raw,'groups':self.raw['groups']+self.raw['constraints']},'C'),kind='coincident',a=dict(primitive_id=first['id'],endpoint='end'),b=dict(primitive_id=second['id'],endpoint='start')))
            self.raw['primitives'][row:row+1]=[first,second];self.commit('Разбить примитив')
        except ValueError as exc:self.message('Разбиение не выполнено: '+str(exc),True)
    @geometry_edit
    def add_object(self,kind,point=None):
        if kind=='custom':
            kind,ok=QInputDialog.getText(self,'Пользовательский тип','Название типа:')
            if not ok or not kind.strip():return
            kind=kind.strip();self.raw.setdefault('types',{}).setdefault(kind,{'label':kind})
        z,r=point if point else ((self.raw['base_z_mm']+self.raw['outlet_z_mm'])/2,(self.cfg.r_in_mm+self.cfg.r_out_mm)/2)
        obj=dict(id=self.uid(kind[0].upper()),type=kind,wall='free',z_mm=z,radius_mm=r,theta_deg=0.,parameters=dict(inclination_deg=None,azimuth_deg=None,flow_kg_s=None,penetration_mm=None),units=dict(length='mm',angle='deg',flow='kg/s'),provenance={})
        self.raw['objects'].append(obj);self.selected=obj['id'];self.pages.setCurrentIndex(1);self.commit('Добавить объект')
    @geometry_edit
    def add_group(self):
        used={m for g in self.raw['groups'] for m in g['members']};available=[o for o in self.raw['objects'] if o['id'] not in used]
        selected={self.raw['objects'][i.row()]['id'] for i in self.object_table.selectedIndexes()}
        members=[o['id'] for o in available if o['id'] in selected] or [o['id'] for o in available if o['type']=='nozzle' and o['nozzle_id']<=70]
        if not members:self.message('Сначала добавьте/выделите объекты, не включённые в другую группу',True);return
        self.raw['groups'].append(dict(id=self.uid('G'),members=members,mode='uniform',phase_deg=0.,direction=-1,row_offset_deg=0.));self.commit('Добавить угловую группу',groups=True)
    @geometry_edit
    def add_section(self):
        zs=sorted({self.raw['base_z_mm'],self.raw['outlet_z_mm'],*[s['z_mm'] for s in self.raw['sections']]});a,b=max(zip(zs,zs[1:]),key=lambda pair:pair[1]-pair[0]);self.raw['sections'].append(dict(id=self.uid('S'),z_mm=(a+b)/2,is_outlet=False));self.commit('Добавить сечение',sections=True)
    @geometry_edit
    def move_section(self,delta):
        row=self.section_table.currentRow();other=row+delta
        if row<0 or not 0<=other<len(self.raw['sections']):return
        self.raw['sections'][row],self.raw['sections'][other]=self.raw['sections'][other],self.raw['sections'][row];self.commit('Порядок ручных сечений',sections=True)
    @geometry_edit
    def delete(self,table):
        rows=sorted({i.row() for i in table.selectedIndexes()},reverse=True)
        key={self.primitive_table:'primitives',self.object_table:'objects',self.group_table:'groups',self.section_table:'sections'}[table]
        for row in rows:
            record=self.raw[key][row]
            if key=='primitives' and record.get('fixed',False):self.message(record['id']+': снимите фиксацию перед удалением.',True);return
            if key=='objects' and record['type']=='nozzle':self.message('140 позиций форсунок сохраняются; отключайте их на 02',True);return
            if key=='sections' and len(self.raw['sections'])-len(rows)<1:self.message('Оставьте хотя бы одно сечение.',True);return
        before=deepcopy(self.raw);before_cells=deepcopy(self._draft_cell_errors);was_invalid=self._draft_invalid
        if key=='primitives':
            materialize_constraints(self.raw);removed={self.raw[key][row]['id'] for row in rows}
            self.raw['constraints']=[c for c in self.raw['constraints'] if not any(c[k]['primitive_id'] in removed for k in ('a','b'))]
        for row in rows:del self.raw[key][row]
        if key=='sections':
            try:self.sync_section_outlet()
            except ValueError as exc:self.raw=before;self._draft_cell_errors=before_cells;self.message('Сечения не изменены: '+str(exc),True);return
        if not self.commit('Удалить геометрические объекты',sections=key=='sections',record_failure=key!='sections') and key=='sections':
            message=self.status.text();self.raw=before;self._draft_cell_errors=before_cells;self._draft_invalid=was_invalid;self.fill_tables();self.draw_scene();self.message('Сечения не изменены. '+message,True)
    def select(self,identity,object_handle=False):
        previous=self.loading;self.loading=True
        self.selected=identity;table=self.object_table if object_handle else self.primitive_table;records=self.raw['objects' if object_handle else 'primitives']
        row=next((i for i,p in enumerate(records) if p['id']==identity),None)
        if row is not None:
            table.blockSignals(True);table.setCurrentCell(row,0);table.blockSignals(False)
            if not self.is_air_mode:self.pages.setCurrentIndex(1 if object_handle else 0)
        for item in self.handles:item.setSelected(item.identity==identity);item.setZValue(10 if item.identity==identity else 8)
        self.highlight_selection()
        self.loading=previous
        if self.is_air_mode:self.air_panel.select_geometry_source(identity)
    def highlight_selection(self):
        for identity,item in self.paths.items():
            pen=item.pen();pen.setCosmetic(True);pen.setWidthF(2.5 if identity==self.selected else 1.4);item.setPen(pen)
        self.sketch.viewport().update()
    def table_selected(self,table,row):
        if self.loading or row<0:return
        key={self.primitive_table:'primitives',self.object_table:'objects',self.group_table:'groups',self.section_table:'sections'}[table]
        if row>=len(self.raw[key]):return
        record=self.raw[key][row];self.selected=record['id'];members=set(record.get('members',[record['id']]))
        self.loading=True
        for handle in self.handles:handle.setSelected(handle.identity in members);handle.setZValue(10 if handle.identity in members else 8)
        self.highlight_selection()
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
            scene=self.sketch.scene();scene.clear();self.paths={};self.handles=[];self.constraint_glyphs=[];self._constraint_glyph_key=None;self.section_lines={};self.section_labels={};angular=self.angular.isChecked();model=None;model_error=''
            try:model=Chamber(self.raw)
            except ValueError as exc:model_error=str(exc)
            if not angular:
                for raw in self.raw['primitives']:
                    points,diagnostic=preview(raw,self.raw.get('chord_error_mm',.01))
                    diagnostic=diagnostic or model_error
                    path=QPainterPath(QPointF(points[0,0],-points[0,1]))
                    for z,r in points[1:]:path.lineTo(z,-r)
                    color='#bd354b' if diagnostic else {'inner':'#316cb7','outer':'#bb6740','reference':'#8a73b3'}[raw['wall']]
                    item=PrimitivePath(self,raw['id'],path);item.setPen(QPen(QColor(color),0,Qt.DashLine if diagnostic else Qt.SolidLine));scene.addItem(item);self.paths[raw['id']]=item
                    if diagnostic:item.setToolTip(raw['id']+' · '+diagnostic)
                    factor=UNITS[raw.get('unit','mm')]
                    for role in ['start']+([] if raw['kind']=='point' else ['end'])+(['center'] if raw['kind']=='arc' else []):
                        point=np.array(raw[role])*factor;handle=Handle(self,raw['id'],role,(point[0],-point[1]),color);scene.addItem(handle);self.handles.append(handle)
                for section in self.raw['sections']:
                    try:bounds=model.bounds(section['z_mm']) if model else draft_bounds(self.raw,section['z_mm'])
                    except ValueError:continue
                    if bounds is None:continue
                    rin,rout=bounds
                    pen=QPen(QColor('#a36bac' if section.get('is_outlet') else '#aab4c5'),0,Qt.DashLine);self.section_lines[section['id']]=scene.addLine(section['z_mm'],-rin,section['z_mm'],-rout,pen)
                    label=scene.addSimpleText(section['id']+(' · выход' if section.get('is_outlet') else ''));label.setFlag(QGraphicsItem.ItemIgnoresTransformations);label.setPos(section['z_mm'],-rout);label.setData(1,'section-label');label.setBrush(QBrush(self.sketch.palette().text().color()))
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
            rect=self.scene_bounds();pad=max(3.,max(rect.width(),rect.height())*.08);scene.setSceneRect(rect.adjusted(-pad,-pad,pad,pad))
            for handle in self.handles:handle.setSelected(handle.identity==self.selected);handle.setZValue(10 if handle.identity==self.selected else 8)
            self.sync_constraint_glyphs()
            self.highlight_selection()
            if self.is_air_mode:self.air_panel.draw_overlay()
            if fit:self.fit()
        finally:self.loading=old
    def scene_bounds(self):
        # Screen-sized endpoint numbers must not change the physical fit scale.
        rect=QRectF()
        for item in self.sketch.scene().items():
            if item.data(1)!='endpoint-number':rect=rect.united(item.sceneBoundingRect())
        return rect
    def fit(self):self.sketch.fitInView(self.sketch.sceneRect(),Qt.KeepAspectRatio)
    def sync_constraint_glyphs(self):
        scene=self.sketch.scene();records=[] if self.angular.isChecked() else [c for c in constraint_records(self.raw) if c.get('enabled',True)]
        signature=tuple((c['id'],c['kind'],c['a']['primitive_id'],c['a']['endpoint'],c['b']['primitive_id'],c['b']['endpoint']) for c in records)
        if signature!=getattr(self,'_constraint_glyph_key',None):
            for _,_,item in getattr(self,'constraint_glyphs',[]):
                if item.scene() is scene:scene.removeItem(item)
            self.constraint_glyphs=[];self._constraint_glyph_key=signature
            for c in records:
                for name in ('a','b'):
                    ref=c[name];glyph=QGraphicsEllipseItem(-8,-8,16,16);glyph.setFlag(QGraphicsItem.ItemIgnoresTransformations);glyph.setZValue(5)
                    glyph.setPen(QPen(QColor('#19846d' if c['kind']=='coincident' else '#aa6b13'),1.5));glyph.setBrush(QBrush(Qt.NoBrush));glyph.setAcceptedMouseButtons(Qt.NoButton)
                    glyph.setToolTip(c['id']+' · '+('совпадение' if c['kind']=='coincident' else 'касательность')+' · '+c['a']['primitive_id']+' ↔ '+c['b']['primitive_id']);scene.addItem(glyph);self.constraint_glyphs.append((ref['primitive_id'],ref['endpoint'],glyph))
        primitives={p['id']:p for p in self.raw['primitives']}
        for identity,role,glyph in self.constraint_glyphs:
            p=primitives.get(identity)
            if p is not None and role in p:
                point=np.array(p[role])*UNITS[p.get('unit','mm')];glyph.setPos(QPointF(point[0],-point[1]))
    @geometry_edit
    def preview_drag(self,identity,role,position,is_object,before,before_nozzles,anchor):
        # Qt never moves a selection group. Every frame is reconstructed from
        # the press snapshot, so passing through an invalid shape cannot lock it.
        previous=self.raw['primitives'];coherent_frame=self.raw.copy()
        if is_object:self.raw=deepcopy(before);self.working_nozzles=deepcopy(before_nozzles)
        else:self.raw['primitives']=deepcopy(before['primitives'])
        try:
            self.change_position(identity,role,position,is_object,anchor)
            if not is_object:
                changed=next(p for p in self.raw['primitives'] if p['id']==identity);original=next(p for p in before['primitives'] if p['id']==identity)
                drivers=[(identity,r) for r in ('start','end') if r in changed and r in original and not np.allclose(np.array(changed[r])*UNITS[changed.get('unit','mm')],np.array(original[r])*UNITS[original.get('unit','mm')],rtol=0,atol=TOL)]
                self.raw=apply_constraints(self.raw,drivers=drivers,baseline=before)
        except (ValueError,KeyError) as exc:
            # Reconstructing a degenerate arc can fail before the coincidence
            # solver runs. Keep the last coherent frame, including its peers,
            # rather than tear the joint or jump back to the press position.
            # Only the proposed primitives were copied/mutated above, so the
            # shallow frame snapshot retains their untouched previous list.
            if not is_object:self.raw=coherent_frame
            self.message('Перемещение ограничено связями: '+str(exc),True)
        if is_object:
            self.refresh_drag_values(identity,True)
            for h in self.handles:
                if h.identity==identity and h.object_handle:h.setPos(position)
        else:
            for old,raw in zip(previous,self.raw['primitives']):
                if old==raw:continue
                self.refresh_drag_values(raw['id'],False)
                points,error=preview(raw,self.raw.get('chord_error_mm',.01));path=QPainterPath(QPointF(points[0,0],-points[0,1]))
                for z,r in points[1:]:path.lineTo(z,-r)
                if raw['id'] in self.paths:
                    self.paths[raw['id']].setPath(path);self.paths[raw['id']].setPen(QPen(QColor('#bd354b' if error else {'inner':'#316cb7','outer':'#bb6740','reference':'#8a73b3'}[raw['wall']]),0))
                for h in self.handles:
                    if h.identity==raw['id'] and not h.object_handle:
                        point=np.array(raw[h.role])*UNITS[raw.get('unit','mm')];h.setPos(QPointF(point[0],-point[1]))
            self.highlight_selection()
            self.sync_constraint_glyphs()
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
    @geometry_edit
    def change_position(self,identity,role,position,is_object,anchor=None):
        z,r=position.x(),-position.y()
        if is_object:
            obj=next(o for o in self.raw['objects'] if o['id']==identity)
            model=Chamber(self.raw);oldz,oldr,oldtheta=model.object_position(obj,self.working_nozzles,self.cfg)
            if self.angular.isChecked():z,r,theta=oldz,oldr,math.degrees(math.atan2(-position.y(),position.x()))%360
            else:theta=oldtheta
            if obj['type']=='nozzle':self.update_nozzle(obj,z,r,theta)
            else:obj.update(z_mm=z,radius_mm=r,theta_deg=theta)
            self.raw=sync_groups(self.raw,self.working_nozzles);return
        p=next(p for p in self.raw['primitives'] if p['id']==identity);factor=UNITS[p.get('unit','mm')];old=deepcopy(p)
        if p.get('fixed',False):raise ValueError(identity+': кривая зафиксирована')
        if role in ('center','body'):
            origin=np.array([anchor.x(),-anchor.y()]) if role=='body' else np.array(p['center'])*factor
            p.update(translate(p,(np.array([z,r])-origin)/factor))
        else:
            # Snap to an existing same-wall endpoint within eight screen pixels.
            for other in self.raw['primitives']:
                if other['id']==identity or other['wall']!=p['wall']:continue
                for key in ('start','end'):
                    if key not in other:continue
                    candidate=np.array(other[key])*UNITS[other.get('unit','mm')]
                    if np.allclose(candidate,np.array(old[role])*factor,rtol=0,atol=TOL):continue
                    a=self.sketch.mapFromScene(QPointF(z,-r));b=self.sketch.mapFromScene(QPointF(candidate[0],-candidate[1]))
                    if (a-b).manhattanLength()<=8:z,r=candidate
            point=[z/factor,r/factor]
            try:
                moved=move_endpoint(p,role,point);p.clear();p.update(moved)
            except ValueError:p[role]=point;p.pop('radius',None);raise
    @geometry_edit
    def finish_drag(self,identity,role,position,is_object,before,before_nozzles):
        self.commit('Sketcher: переместить '+identity)
    @geometry_edit
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
