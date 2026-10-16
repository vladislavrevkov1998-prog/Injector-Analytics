"""Physical v0.5 acceptance: time windows, differential pressure and fixed unit kernels."""
import sys,json,unittest
from pathlib import Path
from dataclasses import replace
from copy import deepcopy
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from fuelfield.core import Settings,default_nozzles,calculate_plane,project_from_json,project_to_json
from fuelfield.measurement import PressureSeries,MeasurementStep,MeasurementSession
from fuelfield.transient import calculate_session,build_influence,flow_at_pressure,CalculationCancelled
from fuelfield.coordinates import to_global,from_global,grid_angles
ROOT=Path(__file__).resolve().parents[1]
def case():
 cfg=Settings(z_mm=[20],pressure_mode='calibrated',n_sample_radial=16,n_sample_angular=144)
 cfg.atomizer_types=[dict(id='curve',name='Синтетическая',kind='experimental',points=[[5,10],[10,30],[20,50],[40,80]])]
 ns=default_nozzles(cfg)
 for n in ns:
  n.enabled=n.id in (1,2,71);n.atomizer_type_id='curve';n.flow_kg_h=30;n.flow_pressure_kgf_cm2=10;n.geometry_overrides=[]
 radii=tuple(cfg.radial_centers_and_edges()[0]);steps=(MeasurementStep(0,0.,1.,(100,110,120,130,140,150),theta_deg=0),MeasurementStep(1,1.,2.,(101,111,121,131,None,151),theta_deg=0))
 session=MeasurementSession(radii,20.,steps,(PressureSeries('A',(0,1,2),(10,10,20)),))
 return cfg,ns,session
class TestV05Math(unittest.TestCase):
 def test_pressure_units_absolute_and_gauge(self):
  cfg,ns,s=case();fuel=PressureSeries('A',(0,2),(1961330.,1961330.),'Pa','p_fuel_abs');gas=PressureSeries('*',(0,2),(980665.,980665.),'Pa','p_gas_abs');s=replace(s,pressure_series=(fuel,),gas_series=(gas,))
  self.assertAlmostEqual(s.delta_p('A',1)[0],10)
  s=replace(s,pressure_series=(replace(fuel,pressure_kind='p_fuel_gauge',reference_id='ambient-A'),),gas_series=(replace(gas,pressure_kind='p_gas_gauge',reference_id='ambient-A'),));self.assertAlmostEqual(s.delta_p('A',1)[0],10)
  with self.assertRaises(ValueError):replace(s,gas_series=()).delta_p('A',1)
  with self.assertRaises(ValueError):replace(s,gas_series=(replace(gas,pressure_kind='p_gas_gauge'),)).delta_p('A',1)
 def test_series_quality_units_times_and_coverage(self):
  cfg,ns,s=case()
  for kwargs in [dict(t_s=(0,0,1),pressure=(1,2,3)),dict(t_s=(1,0),pressure=(1,2)),dict(t_s=(0,1),pressure=(1,2),unit='unknown')]:
   with self.assertRaises(ValueError):PressureSeries('A',**kwargs)
  with self.assertRaises(ValueError):s.pressure_series[0].value(3)
  self.assertTrue(s.pressure_series[0].value(3,True)[1])
  with self.assertRaises(ValueError):replace(s.pressure_series[0],valid=(True,False,True)).value(.5)
  with self.assertRaises(ValueError):s.pressure_series[0].value(.5,max_gap_s=.2)
 def test_clock_offset_and_interpolation(self):
  p=PressureSeries('A',(10,12),(5,15),metadata={'time_offset_s':-10})
  self.assertEqual(p.value(1)[0],10);self.assertEqual(p.t_s,(10,12));self.assertEqual(p.to_dict()['metadata']['time_offset_s'],-10)
 def test_six_channels_mask_and_sequential_windows(self):
  cfg,ns,s=case();r=calculate_session(ns,cfg,s)
  self.assertEqual(r.q_corrected.shape,(6,2));self.assertFalse(r.temperature_valid[4,1]);self.assertTrue(np.isnan(r.temperatures_C[4,1]));self.assertGreater(r.q_corrected[1,1],r.q_corrected[1,0]);self.assertIsNone(r.manifest()['composite_global_balance'])
  with self.assertRaises(ValueError):replace(s.steps[0],temperatures_C=(1,2))
  with self.assertRaises(ValueError):replace(s,steps=(s.steps[0],replace(s.steps[1],t_start_s=.5)))
 def test_constant_pressure_exact_static_and_independent_radii(self):
  cfg,ns,s=case();cfg.target_pressure_kgf_cm2=10;s=replace(s,pressure_series=(PressureSeries('A',(0,2),(10,10)),),radii_mm=tuple(v+.1 for v in s.radii_mm));r=calculate_session(ns,cfg,s);p=calculate_plane(ns,cfg,20)
  np.testing.assert_allclose(r.snapshot(.5)[0].cell_mass_kg_s,p.cell_mass_kg_s,rtol=2e-14,atol=1e-17)
  np.testing.assert_allclose(r.q_corrected,p.flux_kg_m2_s.reshape(-1)[r.cells],rtol=2e-14,atol=1e-15)
  for b in r.balances:self.assertLess(abs(b['relative_error']),1e-12)
 def test_absolute_curve_not_scaled_twice_and_zero_reference_kernel(self):
  cfg,ns,s=case();n=replace(ns[0],flow_mode='curve',flow_kg_h=0)
  self.assertAlmostEqual(flow_at_pressure(n,cfg,20).kg_s,50/3600)
  ns[0]=n;r=calculate_session(ns,cfg,s);self.assertAlmostEqual(r.mean_flows[1,0],40/3600);self.assertGreater(r.operator.mass_weights[:,0].sum(),0)
 def test_nonlinear_window_average_not_flow_of_mean_pressure(self):
  cfg,ns,s=case();s=replace(s,steps=(replace(s.steps[0],t_end_s=2),),pressure_series=(PressureSeries('A',(0,2),(5,20)),))
  r=calculate_session(ns,cfg,s);expected=((10+30)/2*5+(30+50)/2*10)/15/3600
  self.assertAlmostEqual(r.mean_flows[0,0],expected,places=12);self.assertNotAlmostEqual(r.mean_flows[0,0],flow_at_pressure(ns[0],cfg,12.5).kg_s,places=6)
 def test_multiple_manifolds_and_fixed_nozzle_ids(self):
  cfg,ns,s=case();s=replace(s,pressure_series=(PressureSeries('A',(0,2),(10,10)),PressureSeries('B',(0,2),(20,20))),nozzle_manifolds={'2':'B'})
  r=calculate_session(ns[::-1],cfg,s);self.assertEqual(r.operator.nozzle_ids,tuple(range(1,141)));self.assertAlmostEqual(r.mean_flows[0,0],30/3600);self.assertAlmostEqual(r.mean_flows[0,1],50/3600);self.assertLess(abs(r.snapshot(1)[0].rel_balance_error),1e-12)
 def test_delay_coverage_and_no_inferred_temperature(self):
  cfg,ns,s=case();delayed=replace(s,transport_delay_s=1)
  with self.assertRaises(ValueError):calculate_session(ns,cfg,delayed)
  p=PressureSeries('A',(-1,0,1,2),(5,10,10,20));s=replace(s,pressure_series=(p,),transport_delay_s=.5,point_delays_s={'1:2:1':.1});r=calculate_session(ns,cfg,s)
  self.assertNotEqual(r.point_flows[1,1,0],r.mean_flows[1,0]);self.assertIsNone(r.manifest()['temperature_model'])
  np.testing.assert_array_equal(r.temperature_valid,np.asarray([st.valid for st in s.steps]).T)
 def test_legacy_roundtrip_and_no_input_mutation(self):
  cfg,ns,s=case();before=json.dumps(project_to_json(cfg,ns),sort_keys=True);original=s.to_dict();calculate_session(ns,cfg,s)
  self.assertEqual(before,json.dumps(project_to_json(cfg,ns),sort_keys=True));self.assertEqual(s.to_dict(),original)
  data=project_to_json(cfg,ns);data['measurement_session']=s.to_dict();c,n=project_from_json(data);self.assertEqual(c._project_extra['measurement_session'],s.to_dict())
  self.assertEqual(MeasurementSession.from_dict(s.to_dict()).to_dict(),s.to_dict())
 def test_cache_geometry_validation_and_cancel(self):
  cfg,ns,s=case();op=build_influence(ns,cfg,s.z_mm);self.assertLess(op.normalization_error,1e-12);r=calculate_session(ns,cfg,s,operator=op);self.assertIs(r.operator,op)
  with self.assertRaises(ValueError):calculate_session(ns,replace(cfg,theta_zero_deg=10),s,operator=op)
  with self.assertRaises(CalculationCancelled):calculate_session(ns,cfg,s,cancel=lambda:True)
 def test_instantaneous_timestamp_matches_snapshot(self):
  cfg,ns,s=case();step=replace(s.steps[0],t_start_s=1.5,t_end_s=1.5,metadata={'sampling_method':'instantaneous'});s=replace(s,steps=(step,));r=calculate_session(ns,cfg,s)
  plane,flows=r.snapshot(1.5);np.testing.assert_allclose(r.q_corrected[:,0],plane.flux_kg_m2_s.reshape(-1)[r.cells[:,0]],rtol=1e-13)
 def test_extrapolation_permissions_are_separate_and_reported(self):
  cfg,ns,s=case();s=replace(s,steps=(replace(s.steps[0],t_start_s=2.2,t_end_s=2.8),))
  with self.assertRaises(ValueError):calculate_session(ns,cfg,s)
  r=calculate_session(ns,cfg,replace(s,allow_time_extrapolation=True));self.assertTrue(any('временная экстраполяция' in n for n in r.notices))
  cfg,ns,s=case();s=replace(s,pressure_series=(PressureSeries('A',(0,2),(50,50)),))
  with self.assertRaises(ValueError):calculate_session(ns,cfg,s)
  r=calculate_session(ns,replace(cfg,allow_extrapolation=True),s);self.assertTrue(any('характеристики' in n for n in r.notices))
 def test_sector_reference_and_explicit_capture_legacy(self):
  from fuelfield.v02 import flow_status
  cfg,ns,s=case();n=replace(ns[0],flow_mode='sectors',sector_pressure_kgf_cm2=5,capture_efficiency=.5);cfg=replace(cfg,pressure_mode='legacy')
  base=flow_status(n,cfg).kg_s;self.assertAlmostEqual(flow_at_pressure(n,cfg,10).kg_s,3*base)
  calibrated=replace(cfg,pressure_mode='calibrated',target_pressure_kgf_cm2=10);self.assertAlmostEqual(flow_at_pressure(n,calibrated,10).kg_s,flow_status(n,calibrated).kg_s)
 def test_before_tip_and_disabled_columns_have_no_injected_flow(self):
  cfg,ns,s=case();ns[0].geometry_overrides=['z_tip_mm'];ns[0].z_tip_mm=30;ns[1].enabled=False;r=calculate_session(ns,cfg,s)
  self.assertEqual(r.operator.upstream[0],0);self.assertEqual(r.operator.upstream[1],0);p,flows=r.snapshot(1);self.assertAlmostEqual(p.injected_kg_s,flows[70])
 def test_dityakin_quadrature_independent_gaussian_reference(self):
  from fuelfield.v02 import curve_for,flow_status
  cfg,ns=project_from_json(json.loads((ROOT/'examples/demo_project_user_field.json').read_text()));cfg.n_sample_radial=16;cfg.n_sample_angular=144
  for n in ns:n.enabled=n.id==1
  s=MeasurementSession(tuple(cfg.radial_centers_and_edges()[0]),20,(MeasurementStep(0,0,1,(1,2,3,4,5,6)),),(PressureSeries('A',(0,1),(20,30)),))
  r=calculate_session(ns,cfg,s);nodes,weights=np.polynomial.legendre.leggauss(32);curve=curve_for(ns[0],cfg);reference=curve.evaluate(30).flow_kg_h
  expected=sum(w*curve.evaluate(25+5*x).flow_kg_h for x,w in zip(nodes,weights))/2/reference*ns[0].flow_kg_h/3600
  self.assertAlmostEqual(r.mean_flows[0,0],expected,delta=abs(expected)*3e-6)
 def test_frame_and_clockwise_bins(self):
  cfg,ns,s=case();base=calculate_plane(ns,cfg,20);cw=replace(cfg,theta_zero_deg=90,theta_direction=-1)
  self.assertEqual(float(to_global(0,cw)),90);self.assertEqual(float(from_global(80,cw)),10)
  # 90° is an exact 35-cell shift of the 140-angle grid.
  rotated=calculate_plane(ns,cw,20);indices=[int(round(a/(360/140)))%140 for a in grid_angles(cw)]
  np.testing.assert_allclose(rotated.cell_mass_kg_s,base.cell_mass_kg_s[:,indices],rtol=2e-14,atol=1e-17)

if __name__=='__main__':unittest.main()
