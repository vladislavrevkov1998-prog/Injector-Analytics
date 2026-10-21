"""Longitudinal geometry/section diagnostics, never an axial transport solver."""
import numpy as np
from PyQt5.QtWidgets import QDialog,QVBoxLayout,QHBoxLayout,QLabel,QComboBox,QSpinBox,QPushButton
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from .chamber import chamber,legacy_geometry,Chamber
from .analysis import cell_values,UNITS
from .visualization import color_scale


class LongitudinalDialog(QDialog):
    def __init__(self,owner):
        super().__init__(owner);self.owner=owner;self.setWindowTitle('Продольный вид ЖТ: профиль и рассчитанные сечения');self.resize(1080,800)
        from .fuel_transport import transport_enabled
        description=('Цвет: полный поток из сохраняющихся траекторий, включая отдельно учитываемую плёнку в граничных ячейках.' if transport_enabled(owner.cfg) else 'Цвет показан только в рассчитанных сечениях; прежнее ядро пересчитывает их независимо.')
        layout=QVBoxLayout(self);note=QLabel('z=x [мм], R [мм]. '+description+' Для квазистационара — накопленные значения столбцов, без единого мгновенного баланса.');note.setWordWrap(True);layout.addWidget(note)
        controls=QHBoxLayout();self.mode=QComboBox();self.mode.addItems(['Среднее по окружности','Окружная ячейка']);self.index=QSpinBox();self.index.setRange(0,owner.cfg.n_angular-1);controls.addWidget(self.mode);controls.addWidget(QLabel('Ячейка (от 0):'));controls.addWidget(self.index)
        refresh=QPushButton('Обновить из текущего проекта');refresh.clicked.connect(self.draw);controls.addWidget(refresh);controls.addStretch();layout.addLayout(controls)
        self.figure=Figure(figsize=(10,7),tight_layout=True);self.canvas=FigureCanvasQTAgg(self.figure);layout.addWidget(self.canvas,1)
        self.mode.currentIndexChanged.connect(self.draw);self.index.valueChanged.connect(self.draw);self.draw()
    def draw(self):
        self.figure.clear();owner=self.owner;model=chamber(owner.cfg) or Chamber(legacy_geometry(owner.cfg,owner.nozzles))
        ax=self.figure.add_subplot(211 if owner.results else 111)
        for wall,color in [('inner','#316cb7'),('outer','#bb6740')]:
            for p in model.profiles[wall]:
                points=p.discretize(model.error);ax.plot(points[:,0],points[:,1],color=color,lw=1.8)
        for section in model.sections:
            z=section['z_mm'];rin,rout=model.bounds(z);ax.plot([z,z],[rin,rout],ls='--',lw=.7,color='#b18dbc' if section.get('is_outlet') else '#b9bfcb');ax.annotate(section['id']+(' · выход' if section.get('is_outlet') else ''),(z,rout),fontsize=7,rotation=45)
        metric=owner.metric.currentData();points=[];values=[];section_means=[]
        for p in owner.results:
            data=cell_values(p,metric,owner.cfg.density_kg_m3,owner.volume_interval.value())
            column=min(self.index.value(),data.shape[1]-1);display=np.ma.mean(data,axis=1) if self.mode.currentIndex()==0 else data[:,column]
            for radius,value in zip(p.radial_centers_mm,display):
                if np.ma.is_masked(value) or not np.isfinite(value):continue
                points.append((p.z_mm,radius));values.append(float(value))
            section_means.append(float(np.ma.mean(display)))
        if values:
            data,norm,cmap=color_scale(np.array(values),owner.cfg);xy=np.array(points);image=ax.scatter(xy[:,0],xy[:,1],c=data,cmap=cmap,norm=norm,s=50,marker='s',zorder=3);self.figure.colorbar(image,ax=ax,label=UNITS[metric],shrink=.85)
        ax.set(xlabel='z=x, мм',ylabel='R, мм',title='Аналитические стенки и значения на центрах поясов');ax.grid(alpha=.2)
        if owner.results:
            other=self.figure.add_subplot(212);order=np.argsort([p.z_mm for p in owner.results]);zs=np.array([p.z_mm for p in owner.results]);other.plot(zs[order],np.array(section_means)[order],'.-',label='Среднее выбранных радиальных значений');other.set(xlabel='z=x, мм',ylabel=UNITS[metric]);other.grid(alpha=.2);other.legend(fontsize=8)
            if any(p.geometry_notice for p in owner.results):other.set_title('wall-bound учитывается отдельно в балансе; перенос/плёнка не моделируются',fontsize=9)
        self.index.setEnabled(self.mode.currentIndex()==1);self.canvas.draw_idle()
