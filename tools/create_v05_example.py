"""Generate a marked synthetic sequential session; never edit the original user project."""
import json,sys,csv
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from fuelfield.core import project_from_json,project_to_json
from fuelfield.measurement import MeasurementSession,MeasurementStep,PressureSeries
from fuelfield.measurement_io import save_session
root=Path(__file__).resolve().parents[1]
cfg,ns=project_from_json(json.loads((root/'examples/demo_project_user_field.json').read_text()))
cfg.pressure_mode='calibrated';cfg.flow_pressure_kgf_cm2=10.;cfg.sector_pressure_kgf_cm2=10.;cfg.target_pressure_kgf_cm2=10.;cfg.n_sample_radial=24;cfg.n_sample_angular=288;cfg.preview_resolution=400;cfg.theta_zero_deg=90.;cfg.theta_direction=-1;cfg.display_x_axis='cells'
cfg.atomizer_types=[dict(id='synthetic-v05',name='СИНТЕТИЧЕСКАЯ Q(Δp)',kind='experimental',points=[[5,10],[10,30],[20,50],[40,80]])]
for n in ns:
 n.enabled=n.id in (1,2,71);n.atomizer_type_id='synthetic-v05';n.geometry_overrides=[];n.flow_mode='manual';n.flow_kg_h=30.;n.flow_pressure_kgf_cm2=10.;n.sector_pressure_kgf_cm2=10.
 n.sector_ml=[float(1+.25*np.sin(k*2*np.pi/36)) for k in range(36)];n.sector_source='СИНТЕТИЧЕСКИЕ секторные данные v0.5'
radii=tuple(float(v+.2) for v in cfg.radial_centers_and_edges()[0]);steps=[]
for i in range(140):
 theta=i*360/140;temperatures=[float(500+12*r+25*np.sin(np.deg2rad(theta)+.2*r)+.02*i) for r in range(6)];valid=[True]*6
 if i==17:temperatures[4]=None;valid[4]=False
 steps.append(MeasurementStep(i,2.*i,2.*i+1,tuple(temperatures),None,tuple(valid),metadata={'source_kind':'synthetic'}))
times=tuple(float(v) for v in np.arange(0,280.5,.5));a=tuple(float(12+3*np.sin(t/50)) for t in times);b=tuple(float(18+4*np.cos(t/70)) for t in times)
fuel=(PressureSeries('A',times,a,'kgf/cm2','delta_p'),PressureSeries('B',times,tuple(1e6+p*98066.5 for p in b),'Pa','p_fuel_abs'))
gas=(PressureSeries('B',(0.,280.),(1e6,1e6),'Pa','p_gas_abs'),)
s=MeasurementSession(radii,30.,tuple(steps),fuel,gas,theta_zero_deg=90.,theta_direction=-1,nominal_steps=140,nozzle_manifolds={'2':'B'},metadata={'source_kind':'synthetic','description':'Придуманные давления, T, расходы и сектора; геометрический пример на основе проекта пользователя. T создана независимо от q, не является температурным прогнозом.','clock_synchronization':'shared synthetic clock','sensor_response_s':None})
cfg._project_extra['measurement_session']=s.to_dict();cfg._project_extra['data_kind']='synthetic'
(root/'examples/demo_project_v05_synthetic.json').write_text(json.dumps(project_to_json(cfg,ns),ensure_ascii=False,indent=2,allow_nan=False)+'\n')
save_session(root/'examples/demo_session_v05_synthetic.json',s)
with (root/'examples/demo_steps_v05_synthetic.csv').open('w',encoding='utf-8',newline='') as f:
 w=csv.writer(f);w.writerow(['step_id','theta_deg','t_start_s','t_end_s',*[f'T_R{i}_C' for i in range(1,7)],*[f'valid_R{i}' for i in range(1,7)]])
 for step in steps:w.writerow([step.step_id,s.angle(step),step.t_start_s,step.t_end_s,*step.temperatures_C,*[int(v) for v in step.valid]])
with (root/'examples/demo_pressure_v05_synthetic.csv').open('w',encoding='utf-8',newline='') as f:
 w=csv.writer(f);w.writerow(['t_s','manifold_id','pressure','unit','pressure_kind','quality'])
 for p in (*fuel,*gas):
  for t,value in zip(p.t_s,p.pressure):w.writerow([t,p.manifold_id,value,p.unit,p.pressure_kind,1])
print('Synthetic v0.5 project, session, protocol and pressure CSV written')
