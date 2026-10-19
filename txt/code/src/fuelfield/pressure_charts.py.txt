"""Pressure charts for a draft; gaps stay gaps and never become measurements."""
import numpy as np
from PyQt5.QtWidgets import QWidget,QVBoxLayout,QScrollArea,QLabel
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from .quasi import PRESSURE_FACTORS


class PressureCharts(QWidget):
    def __init__(self,parent=None):
        super().__init__(parent);layout=QVBoxLayout(self);layout.setContentsMargins(0,0,0,0)
        self.notice=QLabel('График каждого коллектора обновляется после изменения таблицы.');self.notice.setWordWrap(True);layout.addWidget(self.notice)
        self.figure=Figure(figsize=(5,5),layout='constrained');self.canvas=FigureCanvasQTAgg(self.figure)
        scroll=QScrollArea();scroll.setWidgetResizable(True);scroll.setWidget(self.canvas);layout.addWidget(scroll,1)
        self.setMinimumWidth(340);self.axes=[];self.lines=[];self.key=None;self.update_count=0

    @staticmethod
    def numbers(values,count):
        result=np.full(count,np.nan)
        for i,value in enumerate(values[:count]):
            try:result[i]=float(value)
            except (ValueError,TypeError):pass
        result[~np.isfinite(result)]=np.nan
        return result

    def show_error(self,message):
        self.notice.setText('Графики: '+str(message));self.figure.clear();self.axes=[];self.lines=[];self.key=None;self.canvas.draw_idle()

    def set_mode(self,mode):
        key=tuple((c.id,c.unit,c.kind,c.gas_unit) for c in mode.channels)
        if key!=self.key:
            self.figure.clear();self.axes=[];self.lines=[];self.key=key
            self.canvas.setMinimumHeight(max(240,180*len(key)))
            for index,channel in enumerate(mode.channels):
                ax=self.figure.add_subplot(len(key),1,index+1);self.axes.append(ax)
                lines=[ax.plot([],[],color='#2b66ad',lw=1.6,label='Δp' if channel.kind=='delta_p' else 'Топливо')[0]]
                if channel.kind!='delta_p':
                    lines.append(ax.plot([],[],color='#8b5ca6',lw=1.2,label='Газ (та же единица)')[0]);lines.append(ax.plot([],[],color='#c07016',lw=1.5,label='Δp = топливо − газ')[0]);ax.legend(fontsize=7,loc='upper right')
                self.lines.append(lines);unit='кгс/см²' if channel.unit in ('kgf/cm2','кгс/см²') else channel.unit
                ax.set_title('Коллектор '+channel.id,fontsize=10);ax.set_xlabel('t, с',fontsize=9);ax.set_ylabel(('Δp' if channel.kind=='delta_p' else 'P')+', '+unit,fontsize=9);ax.grid(alpha=.25);ax.tick_params(labelsize=8)
        gaps=0
        for channel,ax,lines in zip(mode.channels,self.axes,self.lines):
            fuel=self.numbers(channel.pressure,mode.n_steps);lines[0].set_data(mode.times,fuel);gaps+=int(np.isnan(fuel).sum())
            if len(lines)>1:
                gas=self.numbers(channel.gas_pressure,mode.n_steps)*PRESSURE_FACTORS.get(channel.gas_unit,np.nan)/PRESSURE_FACTORS.get(channel.unit,np.nan)
                lines[1].set_data(mode.times,gas);lines[2].set_data(mode.times,fuel-gas);gaps+=int(np.isnan(gas).sum())
            ax.relim();ax.autoscale_view()
        self.notice.setText(f'Коллекторов: {len(key)}. '+(f'Незаполненных/ошибочных значений: {gaps}; на графиках — разрывы.' if gaps else 'Исходные давления по постоянным шагам времени.'))
        self.update_count+=1;self.canvas.draw_idle()
