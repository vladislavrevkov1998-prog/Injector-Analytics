"""Synthetic pressure-only mode for unchanged demo_project_user_field.json."""
import sys,json,csv
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from fuelfield.quasi import QuasiSettings,PressureChannel
from fuelfield.quasi_io import save_settings
from fuelfield.analytics_io import write_workbook
root=Path(__file__).resolve().parents[1];i=np.arange(140);a=30+2*np.sin(2*np.pi*i/140)+.3*np.sin(14*np.pi*i/140);b=27+1.5*np.cos(4*np.pi*i/140+.4)
mode=QuasiSettings(140,.5,0.,0.,1,'A',(PressureChannel('A',tuple(map(float,a))),PressureChannel('B',tuple(map(float,b)))),metadata={'data_kind':'synthetic','description':'Демонстрационные давления, не измерения. Для неизменённого demo_project_user_field.json; A: ID 1–70, B: ID 71–140; Δp уже относительно газа. Δt=0.5 с, точки 0…139, последняя 357.428571°.','source_project':'demo_project_user_field.json'})
bindings={str(n):'A' if n<=70 else 'B' for n in range(1,141)};path=save_settings(root/'examples/demo_user_field_140_two_collectors.quasi.json',mode,bindings)
rows=[['step','t_s','angle_deg','A','B']]+[[int(k),float(mode.times[k]),float(mode.angles[k]),float(a[k]),float(b[k])] for k in i]
write_workbook(root/'examples/demo_user_field_140_two_collectors_pressure.xlsx',[('Synthetic pressure',rows)])
with (root/'examples/demo_user_field_140_two_collectors_pressure.csv').open('w',encoding='utf-8-sig',newline='') as f:csv.writer(f,lineterminator='\n').writerows(rows)
print(path)
