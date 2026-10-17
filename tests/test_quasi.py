"""Fixed-step assembly independently compared with the v0.4.1 static solver."""
import sys,json,unittest
from pathlib import Path
from dataclasses import replace
from copy import deepcopy
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from fuelfield.core import project_from_json,project_to_json,calculate_plane,evaluate_continuous_field
from fuelfield.quasi import QuasiSettings,PressureChannel,calculate_quasi,CalculationCancelled
ROOT=Path(__file__).resolve().parents[1]

def case(count=8,constant=False):
 cfg,ns=project_from_json(json.loads((ROOT/'examples/demo_project_user_field.json').read_text()))
 cfg.n_angular=count;cfg.z_mm=[0.,30.];cfg.n_sample_radial=16;cfg.n_sample_angular=144;cfg.preview_resolution=200;cfg.pressure_mode='calibrated';cfg.flow_pressure_kgf_cm2=10.;cfg.sector_pressure_kgf_cm2=10.;cfg.target_pressure_kgf_cm2=10.;cfg.allow_extrapolation=False
 cfg.atomizer_types=[dict(id='test',name='Synthetic test',kind='experimental',points=[[5.,10.],[10.,30.],[20.,50.],[30.,100.]])]
 for n in ns:
  n.enabled=n.id in (1,2,71);n.atomizer_type_id='test';n.flow_mode='manual';n.flow_kg_h=30.;n.manifold_id='A' if n.id<=70 else 'B';n.flow_pressure_kgf_cm2=10.;n.sector_pressure_kgf_cm2=10.
 a=tuple(np.full(count,10.) if constant else np.linspace(10,30,count));b=tuple(np.full(count,10.) if constant else np.linspace(20,10,count))
 mode=QuasiSettings(count,.25,5.,0.,1,'A',(PressureChannel('A',a),PressureChannel('B',b)))
 cfg.quasistationary=True;cfg.quasi_settings=mode.to_dict();return cfg,ns,mode

class TestQuasi(unittest.TestCase):
 def test_constant_matches_v041_all_sections_cells_and_detailed_field(self):
  cfg,ns,mode=case(constant=True);out=calculate_quasi(ns,cfg)
  for i,z in enumerate(cfg.z_mm):
   p=calculate_plane(ns,cfg,z);np.testing.assert_allclose(out.composites[i].cell_mass_kg_s,p.cell_mass_kg_s,rtol=3e-14,atol=1e-15)
   f=evaluate_continuous_field(ns,cfg,z);np.testing.assert_allclose(out.smooth(i,3).flux_kg_m2_s,f.flux_kg_m2_s,rtol=2e-13,atol=1e-12)
   self.assertTrue(out.composites[i].is_composite);self.assertTrue(np.isnan(out.composites[i].rel_balance_error));self.assertLess(out.maximum_balance_error,1e-12)
 def test_columns_use_own_time_and_all_sources_from_both_collectors(self):
  cfg,ns,mode=case();out=calculate_quasi(ns,cfg,preview=False)
  for i,z in enumerate(cfg.z_mm):
   for step in range(mode.n_steps):
    mass=np.zeros((cfg.n_radial,cfg.n_angular))
    for channel in mode.channels:
     subset=[n for n in ns if n.manifold_id==channel.id];mass+=calculate_plane(subset,replace(cfg,target_pressure_kgf_cm2=channel.pressure[step]),z).cell_mass_kg_s
    np.testing.assert_allclose(out.composites[i].cell_mass_kg_s[:,step],mass[:,step],rtol=2e-13,atol=1e-14)
    np.testing.assert_allclose(out.snapshot(i,step).cell_mass_kg_s,mass,rtol=2e-13,atol=1e-14)
  self.assertEqual(out.composites[0].scan_times_s.tolist(),mode.times.tolist());self.assertIsNone(out.manifest()['composite_global_balance'])
 def test_clockwise_nonzero_start_maps_columns_without_reversing_nozzles(self):
  cfg,ns,mode=case();cfg.grid_start_deg=45.;mode=replace(mode,first_angle_deg=135.,direction=-1);cfg.quasi_settings=mode.to_dict();angles=[n.theta_deg for n in ns];out=calculate_quasi(ns,cfg,preview=False)
  self.assertEqual(mode.column_indices(cfg).tolist(),[2,1,0,7,6,5,4,3]);self.assertEqual([n.theta_deg for n in ns],angles)
  for step,col in enumerate(mode.column_indices(cfg)):self.assertEqual(out.composites[0].scan_steps[col],step)
 def test_absolute_and_gauge_and_units_use_physical_delta(self):
  cfg,ns,mode=case();a=PressureChannel('A',(1e6+10*98066.5,)*8,'Pa','p_fuel_abs',(1e6,)*8,'Pa');np.testing.assert_allclose(a.delta_p(8),10)
  b=PressureChannel('B',(15.,)*8,'bar','p_fuel_gauge',(5.,)*8,'bar','common_atm');np.testing.assert_allclose(b.delta_p(8),10e5/98066.5)
  with self.assertRaises(ValueError):replace(b,reference_id='').delta_p(8)
  with self.assertRaises(ValueError):replace(a,gas_pressure=()).delta_p(8)
  with self.assertRaises(ValueError):replace(a,gas_pressure=(2e6,)*8).delta_p(8)
 def test_no_time_averaging_or_average_pressure_substitution(self):
  cfg,ns,mode=case();out=calculate_quasi(ns,cfg,preview=False)
  # At P=10 Q=30; at P=30 Q=100. Exact per-step Q, not Q(mean P)=50.
  self.assertAlmostEqual(out.flows[0,0]*3600,30);self.assertAlmostEqual(out.flows[-1,0]*3600,100)
  np.testing.assert_allclose(np.diff(mode.times),.25);self.assertEqual(len(mode.times),8);self.assertEqual(mode.angles[-1],315)
 def test_curve_is_absolute_once_even_if_reference_zero(self):
  cfg,ns,mode=case();ns[0].flow_mode='curve';ns[0].flow_kg_h=0.;out=calculate_quasi(ns,cfg,preview=False)
  self.assertAlmostEqual(out.flows[-1,0]*3600,100);self.assertGreater(out.operators[0].weights[:,0].sum(),0)
 def test_manual_and_sector_references_and_explicit_efficiency(self):
  cfg,ns,mode=case(constant=True);n=ns[0];n.flow_mode='sectors';n.capture_efficiency=.8;n.sector_ml=[2.]*36;out=calculate_quasi(ns,cfg,preview=False)
  self.assertAlmostEqual(out.flows[0,0],sum(n.sector_ml)*1e-6*cfg.density_kg_m3/cfg.duration_s/.8)
 def test_disabled_and_before_tip_sources_not_in_snapshot_balance(self):
  cfg,ns,mode=case();ns[0].enabled=False;ns[1].geometry_overrides=['z_tip_mm'];ns[1].z_tip_mm=10.;out=calculate_quasi(ns,cfg,preview=False)
  self.assertTrue(np.all(out.flows[:,0]==0));self.assertEqual(out.operators[0].reached[1],0);self.assertEqual(out.operators[1].reached[1],1)
  self.assertLess(out.maximum_balance_error,1e-12)
 def test_invalid_fixed_protocol_and_missing_manifold_are_rejected(self):
  cfg,ns,mode=case()
  for bad in (replace(mode,dt_s=0),replace(mode,n_steps=10),replace(mode,first_angle_deg=3),replace(mode,channels=(mode.channels[0],)),replace(mode,channels=(mode.channels[0],mode.channels[0]))):
   with self.assertRaises(ValueError):bad.validate(cfg,ns)
  with self.assertRaises(ValueError):replace(mode.channels[0],pressure=(1.,)*7).delta_p(8)
  with self.assertRaises(ValueError):replace(mode.channels[0],pressure=(None,)*8).delta_p(8)
 def test_no_hidden_curve_law_or_extrapolation(self):
  cfg,ns,mode=case();ns[0].atomizer_type_id='';
  with self.assertRaises(ValueError):calculate_quasi(ns,cfg,preview=False)
  cfg,ns,mode=case();mode=replace(mode,channels=(replace(mode.channels[0],pressure=(40.,)*8),mode.channels[1]));cfg.quasi_settings=mode.to_dict()
  with self.assertRaises(ValueError):calculate_quasi(ns,cfg,preview=False)
  cfg.allow_extrapolation=True;self.assertTrue(any('экстраполяция' in n.lower() for n in calculate_quasi(ns,cfg,preview=False).notices))
 def test_nozzle_order_fixed_ids_and_cancel_and_input_immutability(self):
  cfg,ns,mode=case();before=project_to_json(cfg,ns);a=calculate_quasi(ns,cfg,preview=False);b=calculate_quasi(list(reversed(ns)),cfg,preview=False)
  np.testing.assert_array_equal(a.flows,b.flows);self.assertEqual(before,project_to_json(cfg,ns))
  with self.assertRaises(CalculationCancelled):calculate_quasi(ns,cfg,cancel=lambda:True)
  with self.assertRaises(ValueError):calculate_quasi(ns[:-1],cfg,preview=False)
 def test_legacy_json_default_stationary_and_unknowns_survive(self):
  source=json.loads((ROOT/'examples/demo_project_user_field.json').read_text());source['custom']={'unchanged':True};cfg,ns=project_from_json(source)
  self.assertFalse(cfg.quasistationary);self.assertEqual(cfg.quasi_settings,{});self.assertTrue(all(n.manifold_id=='A' for n in ns));self.assertEqual(project_to_json(cfg,ns)['custom'],source['custom'])
  mode=case()[2];self.assertEqual(QuasiSettings.from_dict(json.loads(json.dumps(mode.to_dict()))).to_dict(),mode.to_dict())
  for wrong in (source,{'format':'fuel-measurement-session','version':'0.5'}):
   with self.assertRaises(ValueError):QuasiSettings.from_dict(wrong)

if __name__=='__main__':unittest.main()
