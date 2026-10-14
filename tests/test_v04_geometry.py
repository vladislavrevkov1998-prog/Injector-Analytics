"""v0.4 geometry, arbitrary sections, graph controls and staged fuel fitting."""
import json,sys,tempfile,time,unittest
from dataclasses import replace
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from fuelfield.core import (Settings,default_nozzles,calculate_plane,evaluate_continuous_field,geometry_at,angular_profile,
 sample_kernel,polar_epure_coordinates,project_from_json,project_to_json)
from fuelfield.v02 import effective_nozzle
from fuelfield.analysis import cell_values
from fuelfield.analytics_io import write_epures_xlsx
from fuelfield.data_io import _xlsx_sheets
ROOT=Path(__file__).resolve().parents[1]
def user():return project_from_json(json.loads((ROOT/'examples/demo_project_user_field.json').read_text()))
def synthetic():
 cfg=Settings(z_mm=[10.,25.,45.,60.],geometry_model='constant',n_sample_radial=24,n_sample_angular=360)
 ns=default_nozzles(cfg)
 for n in ns:
  n.enabled=n.id in (1,2,71);n.geometry_overrides=[];n.flow_kg_h=10.;n.sector_ml=[1.+k%9*.3 for k in range(36)]
 return cfg,ns


class TestV04Physics(unittest.TestCase):
 def test_tip_inheritance_and_override(self):
  cfg,ns=user();cfg.nozzle_presets['cooled']['z_tip_mm']=12
  self.assertEqual(effective_nozzle(ns[0],cfg).z_tip_mm,12)
  ns[0].z_tip_mm=5;ns[0].geometry_overrides=['z_tip_mm'];self.assertEqual(effective_nozzle(ns[0],cfg).z_tip_mm,5)
  self.assertEqual(effective_nozzle(ns[1],cfg).z_tip_mm,12)
 def test_before_tip_both_fields_and_balance(self):
  cfg,ns=synthetic();cfg.nozzle_presets['uncooled']['z_tip_mm']=20
  before=calculate_plane(ns,cfg,19);self.assertEqual(before.injected_kg_s,0);self.assertFalse(before.cell_mass_kg_s.any())
  self.assertFalse(evaluate_continuous_field(ns,cfg,19,128).flux_kg_m2_s.any())
  self.assertEqual(len(polar_epure_coordinates(ns[0],19,cfg,8)[0]),0)
  self.assertEqual(len(sample_kernel(ns[0],19,cfg)[0]),0)
  after=calculate_plane(ns,cfg,20);self.assertGreater(after.injected_kg_s,0);self.assertLess(abs(after.rel_balance_error),1e-13)
 def test_shift_equivalence_no_double_subtraction(self):
  cfg,ns=synthetic();n=ns[0];shift=deepcopy(cfg);shift.nozzle_presets['uncooled']['z_tip_mm']=17
  shift.angular_model=cfg.angular_model='diffusion';shift.nozzle_presets['uncooled']['rotation_deg_per_mm']=cfg.nozzle_presets['uncooled']['rotation_deg_per_mm']=.7
  self.assertEqual(geometry_at(n,27,shift),geometry_at(n,10,cfg))
  np.testing.assert_array_equal(angular_profile(n,27,shift),angular_profile(n,10,cfg))
  for a,b in zip(sample_kernel(n,27,shift),sample_kernel(n,10,cfg)):np.testing.assert_array_equal(a,b)
  for a,b in zip(polar_epure_coordinates(n,27,shift,8),polar_epure_coordinates(n,10,cfg,8)):np.testing.assert_array_equal(a,b)
  np.testing.assert_array_equal(evaluate_continuous_field([n],shift,27,128).flux_kg_m2_s,evaluate_continuous_field([n],cfg,10,128).flux_kg_m2_s)
 def test_mixed_tips_reach_balance(self):
  cfg,ns=synthetic();ns[0].geometry_overrides=['z_tip_mm'];ns[0].z_tip_mm=30
  p=calculate_plane(ns,cfg,20);self.assertAlmostEqual(p.injected_kg_s,20/3600);self.assertLess(abs(p.rel_balance_error),1e-13)
 def test_sections_1_4_7_xlsx_roundtrip(self):
  for count in (1,4,7):
   cfg,ns=synthetic();cfg.z_mm=list(range(count));cfg.validate()
   c,n=project_from_json(project_to_json(cfg,ns));self.assertEqual(c.z_mm,cfg.z_mm)
   planes=[calculate_plane(n,c,z) for z in c.z_mm]
   with tempfile.TemporaryDirectory() as d:
    path=Path(d)/'epures.xlsx';write_epures_xlsx(path,planes,c);sheets=list(_xlsx_sheets(path));self.assertEqual(len(sheets),count)
    self.assertEqual(len(set(name for name,rows in sheets)),count);self.assertTrue(all(len(name)<=31 for name,rows in sheets))
 def test_invalid_sections(self):
  for values in ([],[0]*2,[0,1e-8],[-1],[np.nan],[np.inf],list(range(51))):
   cfg=Settings();cfg.z_mm=values
   with self.assertRaises(ValueError):cfg.validate()
 def test_unknown_fields_and_snapshot_restoration(self):
  cfg,ns=user();data=project_to_json(cfg,ns);data['custom']={'preserve':'yes'};data['settings']['lab']='A';data['nozzles'][0]['operator_note']='keep'
  cfg,ns=project_from_json(data);c,n=project_from_json(project_to_json(cfg,ns));self.assertEqual(project_to_json(c,n),data)

