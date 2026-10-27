"""Transactional editor for the air registry. Flows are always kg/s.

No measured swirler values are manufactured: adding linked sources leaves their
method unknown until the user supplies a flow or explicitly allocates a share.
"""
from copy import deepcopy
import json
import numpy as np
from PyQt5.QtCore import Qt, pyqtSignal, QTimer
from PyQt5.QtGui import QColor, QPainter
from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QFormLayout,
    QLabel, QCheckBox, QComboBox, QTableWidget, QTableWidgetItem, QPushButton,
    QSlider, QDialogButtonBox, QMessageBox, QInputDialog, QPlainTextEdit, QWidget,
    QHeaderView, QAbstractItemView, QTextBrowser)
from .numeric_input import DecimalSpinBox

KINDS = dict(swirler='Завихрители', hole='Отверстия', insert='Вставки',
             cooling='Охлаждение', other='Прочие', unassigned='Не назначено')
COLORS = ['#3992cf', '#db963a', '#8e72c7', '#3fa58a', '#ca718e', '#9ba5af']
# Missing optional numerical cells remain null, not a fictitious measurement.
COLUMNS = [('id','ID','str'),('kind','Тип','str'),('enabled','Включён','bool'),
    ('group','Группа','str'),('nozzle_id','ID форсунки','int'),('object_id','ID геометрии','str'),
    ('method','Метод','str'),('flow_kg_s','G, кг/с','float'),('share','Доля 0…1','float'),
    ('area_mm2','F, мм²','float'),('cd','Cd','float'),('z_mm','z, мм','float'),
    ('radius_mm','R, мм','float'),('theta_deg','θ физ., °','float'),
    ('wall','Стенка','str'),('dz','dz','float'),('dr','dr','float'),('dt','dθ','float'),
    ('penetration_factor','Пробивная способность','float'),('kernel','Ядро','str'),
    ('sigma_r_mm','σR, мм','float'),('sigma_theta_deg','σθ, °','float'),
    ('spread_mm_per_mm','Расширение, мм/мм','float'),('mixed_fraction','Вовлечение 0…1','float'),
    ('mixing_length_mm','Длина вовлечения, мм','float'),('confidence','Доверие','str')]


def spin(value=0., high=1e9, decimals=12):
    box=DecimalSpinBox();box.setRange(0.,high);box.setDecimals(decimals);box.setValue(value)
    return box


class BalanceBar(QWidget):
    selected=pyqtSignal(str)
    def __init__(self,parent=None):
        super().__init__(parent);self.values={};self.regions=[];self.setMinimumHeight(45)
        self.setMouseTracking(True)
        self.setToolTip('Нажмите долю, чтобы выбрать источники этой категории в таблице.')
    def paintEvent(self,event):
        p=QPainter(self);total=sum(self.values.values());x=0.;self.regions=[]
        for i,(key,value) in enumerate(self.values.items()):
            width=self.width()*value/total if total else 0
            p.fillRect(int(x),0,max(0,int(width)),self.height(),QColor(COLORS[i%len(COLORS)]))
            if width>55:p.setPen(QColor('#101820'));p.drawText(int(x)+4,25,f'{value:.5g}')
            self.regions.append((x,x+width,key));x+=width
        p.end()
    def mousePressEvent(self,event):
        for x0,x1,key in self.regions:
            if x0<=event.x()<x1:self.selected.emit(key);break
    def mouseMoveEvent(self,event):
        for x0,x1,key in self.regions:
            if x0<=event.x()<x1:
                value=self.values[key];total=sum(self.values.values())
                self.setToolTip(f'{KINDS.get(key,key)}: {value:.10g} кг/с; {100*value/total:.6g}%' if total else 'Нет расхода')
                break


class AirBalanceDialog(QDialog):
    applied=pyqtSignal(dict)
    def __init__(self,cfg,nozzles,parent=None):
        super().__init__(parent)
        from .air_sources import default_air_model,validate_air_model
        self.cfg=deepcopy(cfg);self.nozzles=deepcopy(nozzles)
        self.raw=validate_air_model(cfg.air_model)
        self._busy=False;self.balance=None
        self.setWindowTitle('Воздух v0.8 — источники и расходный баланс');self.resize(1280,840)
        lay=QVBoxLayout(self);top=QHBoxLayout()
        self.enabled=QCheckBox('Включить воздух');self.enabled.setChecked(self.raw.get('enabled',False))
        self.total=spin(self.raw.get('total_kg_s') or 0.)
        self.l0=spin(self.raw.get('l0') or 0.,1000.)
        self.total.setToolTip('Суммарный расход воздуха через КС [кг/с].')
        self.l0.setToolTip('Явный стехиометрический коэффициент, кг воздуха/кг топлива. 0 = неизвестен: α не определяется.')
        for w in (self.enabled,QLabel('G воздуха КС, кг/с:'),self.total,QLabel('L0 (0 = неизвестен):'),self.l0):top.addWidget(w)
        lay.addLayout(top)
        algorithms=QHBoxLayout();self.distribution=QComboBox()
        self.distribution.addItem('По площади ячеек: равномерный поток','uniform')
        self.distribution.addItem('Локальные направленные ядра','directed')
        self.distribution.setCurrentIndex(max(0,self.distribution.findData(self.raw.get('distribution','directed'))))
        self.interaction=spin(self.raw.get('interaction_strength',0.),1.,4)
        self.area_confirm=QCheckBox('Принимаю гипотезу сходных условий для F / CdF')
        self.area_confirm.setChecked(self.raw.get('area_hypothesis_confirmed',False))
        self.area_fraction=spin(self.raw.get('area_allocation_fraction',1.),1.,4)
        for w in (self.distribution,QLabel('Взаимодействие 0…1:'),self.interaction,self.area_confirm,
                  QLabel('Доля остатка для F:'),self.area_fraction):algorithms.addWidget(w)
        lay.addLayout(algorithms)
        note=QLabel('Методы: fixed — заданный G; fraction — доля от G КС; area / effective_area — остаток по F / CdF; unknown — неизвестен. '
            'Пустое поле = неизвестно. dz, dr, dθ задают направление, не скорость. θ — физический +X/против ЧС. '
            'Эмпирические ширины и вовлечение задаются пользователем; воздух не меняет топливо. Изменения вступают в силу после «Применить».')
        note.setWordWrap(True);lay.addWidget(note)
        buttons=QHBoxLayout()
        for title,slot in [('Добавить источник',self.add_source),('Добавить завихрители (G неизвестен)',self.add_swirlers),
                           ('Связать с геометрией',self.add_geometry),('Удалить выбранные',self.remove_sources),
                           ('Параметры JSON…',self.edit_json),('Справка',self.show_help)]:
            b=QPushButton(title);b.clicked.connect(slot);buttons.addWidget(b)
        self.unlock_fixed=QCheckBox('Редактировать заданные G');buttons.addWidget(self.unlock_fixed)
        lay.addLayout(buttons)
        self.table=QTableWidget(0,len(COLUMNS));self.table.setHorizontalHeaderLabels([x[1] for x in COLUMNS])
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        lay.addWidget(self.table,1)
        group_row=QHBoxLayout();self.group_units=QComboBox()
        self.group_units.addItem('Доли групп, %','percent');self.group_units.addItem('Расходы групп, кг/с','flow')
        group_row.addWidget(self.group_units);group_row.addWidget(QLabel('Ползунок меняет только переменные источники группы; заданные G сохраняются.'))
        lay.addLayout(group_row)
        self.groups={}
        for key,label in KINDS.items():
            if key=='unassigned':continue
            row=QHBoxLayout();row.addWidget(QLabel(label))
            slider=QSlider(Qt.Horizontal);slider.setRange(0,10000)
            value=spin(0.,100.,6);row.addWidget(slider,1);row.addWidget(value);lay.addLayout(row)
            self.groups[key]=(slider,value)
            slider.valueChanged.connect(lambda v,k=key:self.group_changed(k,v/10000*(100 if self.group_units.currentData()=='percent' else self.total.value())))
            value.valueChanged.connect(lambda v,k=key:self.group_changed(k,v))
        self.bar=BalanceBar();self.bar.selected.connect(self.select_kind);lay.addWidget(self.bar)
        self.status=QLabel();self.status.setWordWrap(True);lay.addWidget(self.status)
        controls=QDialogButtonBox(QDialogButtonBox.Apply|QDialogButtonBox.Close)
        self.apply_button=controls.button(QDialogButtonBox.Apply);self.apply_button.clicked.connect(self.apply)
        controls.rejected.connect(self.reject);lay.addWidget(controls)
        self.timer=QTimer(self);self.timer.setSingleShot(True);self.timer.timeout.connect(self.refresh)
        self.table.itemChanged.connect(lambda:self.timer.start(50))
        self.unlock_fixed.toggled.connect(self.fill)
        for w in (self.total,self.l0,self.interaction,self.area_fraction):w.valueChanged.connect(self.refresh)
        for w in (self.enabled,self.area_confirm):w.toggled.connect(self.refresh)
        self.distribution.currentIndexChanged.connect(self.refresh);self.group_units.currentIndexChanged.connect(self.refresh)
        self.fill();self.refresh()

    def fill(self,*args):
        self._busy=True;self.table.blockSignals(True)
        self.table.setRowCount(len(self.raw.get('sources',[])))
        for row,source in enumerate(self.raw.get('sources',[])):
            direction=source.get('direction',[1.,0.,0.])
            for col,(key,label,kind) in enumerate(COLUMNS):
                value=direction[('dz','dr','dt').index(key)] if key in ('dz','dr','dt') else source.get(key)
                if key=='enabled':value=source.get(key,True)
                item=QTableWidgetItem('' if value is None else str(value))
                if kind=='bool':item.setText('');item.setCheckState(Qt.Checked if value else Qt.Unchecked)
                if source.get('method')=='fixed' and key=='flow_kg_s' and not self.unlock_fixed.isChecked():item.setFlags(item.flags() & ~Qt.ItemIsEditable)
                self.table.setItem(row,col,item)
        self.table.blockSignals(False);self._busy=False

    def read(self):
        def precise(key,control):
            value=control.value();original=self.raw.get(key)
            # Merely displaying/applying a project must not round an existing
            # physical input to the finite precision of a spin box.
            if original is not None and round(float(original),control.decimals())==value:return original
            return value
        raw=deepcopy(self.raw);raw.update(enabled=self.enabled.isChecked(),total_kg_s=precise('total_kg_s',self.total),
            l0=precise('l0',self.l0) or None,distribution=self.distribution.currentData(),interaction_strength=precise('interaction_strength',self.interaction),
            area_hypothesis_confirmed=self.area_confirm.isChecked(),area_allocation_fraction=precise('area_allocation_fraction',self.area_fraction))
        for row,source in enumerate(raw['sources']):
            direction=list(source.get('direction',[1.,0.,0.]))
            for col,(key,label,kind) in enumerate(COLUMNS):
                item=self.table.item(row,col);value=item.text().strip()
                if kind=='bool':value=item.checkState()==Qt.Checked
                elif kind in ('float','int'):
                    value=None if not value else (int(value) if kind=='int' else float(value.replace(',','.')))
                elif value=='':
                    if key in source:source[key]=None if key=='object_id' else ''
                    continue
                if key in ('dz','dr','dt'):direction[('dz','dr','dt').index(key)]=value
                else:source[key]=value
            source['direction']=direction
        return raw

    def refresh(self,*args):
        if self._busy:return
        from .air_sources import validate_air_model,resolve_sources
        from .air_balance import solve_balance
        try:
            raw=self.read();validate_air_model(raw)
            # Disabling air must remain possible after geometry/nozzle edits.
            # Keep inactive link metadata for repair without using it physically.
            self.raw=raw
            sources=resolve_sources(raw,self.cfg,self.nozzles) if raw['enabled'] else raw['sources']
            candidate=deepcopy(raw);candidate['sources']=list(sources)
            try:self.balance=solve_balance(candidate)
            except ValueError:
                if raw['enabled']:raise
                self.balance=None;self.status.setText('Воздух отключён. Черновик сохранён; перед включением проверьте расходный баланс.')
                self.apply_button.setEnabled(True);return
            b=self.balance
            self.status.setText(f'Назначено {b.assigned_kg_s:.8g} кг/с; неизвестный остаток {b.unassigned_kg_s:.8g} кг/с; '
                f'заданные G {b.fixed_kg_s:.8g} кг/с. '+ ' · '.join(map(str,b.diagnostics)))
            values={k:0. for k in KINDS}
            for source in raw['sources']:
                values[source.get('kind','other') if source.get('kind') in KINDS else 'other']+=b.flows.get(source['id'],0.)
            values['unassigned']=b.unassigned_kg_s;self.bar.values=values;self.bar.update()
            self._busy=True
            for key,(slider,number) in self.groups.items():
                variable=any(s.get('kind')==key and s.get('enabled',True) and s.get('method')!='fixed' for s in raw['sources'])
                slider.setEnabled(variable);number.setEnabled(variable)
                total=max(self.total.value(),1e-30);percent=100*values[key]/total
                number.setMaximum(100. if self.group_units.currentData()=='percent' else self.total.value())
                number.setValue(percent if self.group_units.currentData()=='percent' else values[key]);slider.setValue(round(percent*100))
            self._busy=False;self.apply_button.setEnabled(True)
        except (ValueError,TypeError,KeyError,OverflowError) as exc:
            self._busy=False;self.balance=None;self.status.setText('Баланс не применён: '+str(exc));self.apply_button.setEnabled(False)

    def group_changed(self,key,value):
        if self._busy:return
        try:
            raw=self.read();total=self.total.value()
            if total<=0:raise ValueError('Задайте положительный G воздуха КС.')
            rows=[s for s in raw['sources'] if s.get('kind')==key and s.get('enabled',True)]
            fixed=sum(float(s.get('flow_kg_s') or 0.) for s in rows if s.get('method')=='fixed')
            requested=value*total/100 if self.group_units.currentData()=='percent' else value
            if requested<fixed-1e-12:raise ValueError('Доля меньше заданных G группы; заданные расходы не изменены.')
            variable=[s for s in rows if s.get('method')!='fixed']
            if not variable:return
            weights=np.array([max(0.,s.get('share') or 0.) for s in variable]);weights=weights/weights.sum() if weights.sum() else np.full(len(variable),1/len(variable))
            for source,weight in zip(variable,weights):source.update(method='fraction',share=(requested-fixed)/total*float(weight))
            self.raw=raw;self.fill();self.refresh()
        except (ValueError,TypeError) as exc:self.status.setText(str(exc))

    def select_kind(self,key):
        self.table.clearSelection()
        if key=='unassigned':return
        for row,source in enumerate(self.raw['sources']):
            if source.get('kind')==key:
                for col in range(self.table.columnCount()):self.table.item(row,col).setSelected(True)

    def _new_source(self,kind='other',**values):
        ids={s['id'] for s in self.raw['sources']};index=1
        while f'{kind}-{index}' in ids:index+=1
        source=dict(id=f'{kind}-{index}',kind=kind,enabled=True,method='unknown',flow_kg_s=None,
            share=0.,z_mm=0.,radius_mm=(self.cfg.r_in_mm+self.cfg.r_out_mm)/2,theta_deg=0.,wall='free',
            direction=[1.,0.,0.],penetration_factor=1.,kernel='directed',sigma_r_mm=10.,sigma_theta_deg=10.,
            spread_mm_per_mm=0.,mixed_fraction=0.,confidence='unknown',provenance={'origin':'user editor; numerical kernel defaults, unmeasured'})
        source.update(values);return source

    def _capture(self):
        try:self.raw=self.read();return True
        except (ValueError,TypeError) as exc:self.status.setText(str(exc));return False

    def add_source(self):
        if not self._capture():return
        self.raw['sources'].append(self._new_source());self.fill();self.refresh()

    def add_swirlers(self):
        if not self._capture():return
        existing={s.get('nozzle_id') for s in self.raw['sources'] if s.get('kind')=='swirler'}
        for n in self.nozzles:
            if n.id not in existing:
                self.raw['sources'].append(self._new_source('swirler',nozzle_id=n.id,enabled=n.enabled,
                    z_mm=n.z_tip_mm,radius_mm=n.radius_mm,theta_deg=n.theta_deg))
        self.fill();self.refresh()

    def add_geometry(self):
        if not self._capture():return
        objects=[o for o in self.cfg.geometry.get('objects',[]) if o.get('type') in ('hole','insert','cooling','other')]
        ids=[str(o['id']) for o in objects]
        if not ids:QMessageBox.information(self,'Геометрия','В редакторе ЖТ пока нет отверстий/вставок для привязки.');return
        choice,ok=QInputDialog.getItem(self,'Источник воздуха','Геометрический объект:',ids,0,False)
        if not ok:return
        obj=next(o for o in objects if str(o['id'])==choice)
        self.raw['sources'].append(self._new_source(obj['type'],object_id=obj['id']))
        self.fill();self.refresh()

    def remove_sources(self):
        if not self._capture():return
        rows={i.row() for i in self.table.selectedIndexes()}
        self.raw['sources']=[s for i,s in enumerate(self.raw['sources']) if i not in rows]
        self.fill();self.refresh()

    def edit_json(self):
        if not self._capture():return
        dialog=QDialog(self);dialog.setWindowTitle('Все параметры воздуха (JSON, только черновик)');dialog.resize(900,650)
        layout=QVBoxLayout(dialog);editor=QPlainTextEdit(json.dumps(self.raw,ensure_ascii=False,indent=2));layout.addWidget(editor)
        controls=QDialogButtonBox(QDialogButtonBox.Ok|QDialogButtonBox.Cancel);layout.addWidget(controls)
        controls.accepted.connect(dialog.accept);controls.rejected.connect(dialog.reject)
        if dialog.exec_()!=QDialog.Accepted:return
        try:
            from .air_sources import validate_air_model
            raw=json.loads(editor.toPlainText());validate_air_model(raw)
            self._busy=True;self.raw=raw
            self.enabled.setChecked(raw.get('enabled',False));self.total.setValue(raw.get('total_kg_s') or 0.)
            self.l0.setValue(raw.get('l0') or 0.);self.area_confirm.setChecked(raw.get('area_hypothesis_confirmed',False))
            self.area_fraction.setValue(raw.get('area_allocation_fraction',1.));self.interaction.setValue(raw.get('interaction_strength',0.))
            self.distribution.setCurrentIndex(max(0,self.distribution.findData(raw.get('distribution','directed'))))
            self.fill();self._busy=False;self.refresh()
        except (ValueError,TypeError) as exc:
            self._busy=False;QMessageBox.warning(self,'Параметры воздуха',str(exc))

    def show_help(self):
        from .help_text import populate_help
        existing=getattr(self,'guide_dialog',None)
        if existing is not None:existing.close();existing.deleteLater()
        self.guide_dialog=QDialog(self);self.guide_dialog.setWindowTitle('Инструкция: воздух и α')
        self.guide_dialog.resize(950,720);layout=QVBoxLayout(self.guide_dialog)
        browser=QTextBrowser();populate_help(browser,'V0.8_GUIDE.md');layout.addWidget(browser)
        self.guide_dialog.show()

    def apply(self):
        self.refresh()
        if self.apply_button.isEnabled():self.applied.emit(deepcopy(self.raw))
