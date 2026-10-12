import sys, unittest, tempfile, csv, os, json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from fuelfield.core import *
from fuelfield.data_io import import_sector_file
import numpy as np

class TestPhysics(unittest.TestCase):
    def setUp(self):
        self.cfg=Settings(n_sample_radial=30,n_sample_angular=180)
        self.nozzles=default_nozzles(self.cfg)
        for n in self.nozzles:
            n.flow_kg_h=10
    def test_positions_and_disable_no_shift(self):
        before=[(n.id,n.row,n.theta_deg,n.radius_mm) for n in self.nozzles]
        self.nozzles[7].enabled=False
        after=[(n.id,n.row,n.theta_deg,n.radius_mm) for n in self.nozzles]
        self.assertEqual(before,after)
        self.assertAlmostEqual(self.nozzles[70].theta_deg,360/140)
    def test_four_planes_conservative(self):
        self.cfg.z_mm=[0,15,30,45]
        for z in self.cfg.z_mm:
            p=calculate_plane(self.nozzles,self.cfg,z)
            self.assertAlmostEqual(p.injected_kg_s,140*10/3600,places=10)
            self.assertLess(abs(p.rel_balance_error),1e-12)
            self.assertTrue(np.all(p.cell_mass_kg_s >=0))
            self.assertTrue(np.all(p.flux_kg_m2_s >=0))
    def test_disabled_nozzle_changes_only_mass(self):
        p0=calculate_plane(self.nozzles,self.cfg,20)
        self.nozzles[0].enabled=False
        p1=calculate_plane(self.nozzles,self.cfg,20)
        self.assertAlmostEqual((p0.injected_kg_s-p1.injected_kg_s),10/3600,places=12)
    def test_mass_from_ml_separate_from_default(self):
        n=self.nozzles[0]
        n.sector_ml=[10]*36
        n.flow_mode='sectors'
        self.assertAlmostEqual(source_flow_kg_s(n,self.cfg),360e-6*800/180)
        n.flow_mode='manual'
        self.assertAlmostEqual(source_flow_kg_s(n,self.cfg),10/3600)
    def test_angular_modes_keep_mean(self):
        n=self.nozzles[0]
        n.sector_ml=[2]+[1]*35
        for mode in ANGULAR_MODELS:
            self.cfg.angular_model=mode
            for z in (0,20,50):
                a=angular_profile(n,z,self.cfg)
                self.assertAlmostEqual(a.mean(),1,places=10)
                self.assertTrue(np.min(a)>=-1e-12)
    def test_geometry_modes(self):
        n=self.nozzles[0]
        for model in GEOMETRY_MODELS:
            self.cfg.geometry_model=model
            for z in (0,20,100):
                r,h=geometry_at(n,z,self.cfg)
                self.assertTrue(r>=0 and h>0)
    def test_custom_radial_nodes(self):
        self.cfg.radial_centers_mm=[204,214,250,280,294,298]
        self.cfg.grid_start_deg=20
        centers,edges=self.cfg.radial_centers_and_edges()
        self.assertTrue(np.all(np.diff(edges)>0))
        self.assertAlmostEqual(edges[0],200)
        p=calculate_plane(self.nozzles[:2],self.cfg,20)
        self.assertEqual(p.cell_mass_kg_s.shape,(6,140))
    def test_serialization(self):
        self.nozzles[5].enabled=False
        data=project_to_json(self.cfg,self.nozzles)
        c,n=project_from_json(json.loads(json.dumps(data)))
        self.assertFalse(n[5].enabled)
        self.assertEqual(c.n_angular,140)
    def test_csv_parse(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=os.path.join(tmp,'input.csv')
            with open(p,'w',newline='',encoding='utf-8') as f:
                w=csv.writer(f,delimiter=';')
                w.writerow(['№ форсунки','Сумма']+list(range(1,37)))
                w.writerow([13,36]+[1]*36)
            data,sh=import_sector_file(p)
            self.assertEqual(len(data[13]),36)
            self.assertEqual(sum(data[13]),36)

if __name__=='__main__':unittest.main()

class TestPhysicalInterpretation(unittest.TestCase):
    def test_sector1_points_to_middle_line(self):
        cfg=Settings(n_sample_radial=20,n_sample_angular=360,radial_model='uniform')
        for row,radius,sign in [('outer',270,-1),('inner',230,+1)]:
            n=Nozzle(id=1,row=row,theta_deg=0,radius_mm=radius,flow_kg_h=10,
                     sector_ml=[1]+[0]*35,rc0_mm=8,h0_mm=1)
            x,y,w=sample_kernel(n,0,cfg)
            centre_x=np.average(x,weights=w)
            self.assertGreater((centre_x-radius)*sign,5)
    def test_rotated_matrix_conserves_total(self):
        cfg=Settings(n_sample_radial=20,n_sample_angular=360)
        n=Nozzle(id=1,flow_kg_h=10,radius_mm=270)
        p1=calculate_plane([n],cfg,30)
        cfg.grid_start_deg=7.7
        p2=calculate_plane([n],cfg,30)
        self.assertAlmostEqual(p1.injected_kg_s,p2.injected_kg_s,places=14)
        self.assertLess(abs(p2.rel_balance_error),1e-12)

class TestExactMeasuredSectorShares(unittest.TestCase):
    def test_piecewise_sector_1_integral(self):
        cfg=Settings(n_sample_radial=20,n_sample_angular=720,radial_model='uniform')
        n=Nozzle(id=1, row='inner',theta_deg=0,radius_mm=230,
                 flow_kg_h=10,sector_ml=[2]+[1]*35,
                 rc0_mm=7,h0_mm=2)
        x,y,w=sample_kernel(n,0,cfg)
        phi=np.rad2deg(np.arctan2(y,x-n.radius_mm))%360
        idx=np.floor((phi/10+0.5)%36).astype(int)
        sector_mass=np.bincount(idx,weights=w,minlength=36)
        self.assertAlmostEqual(sector_mass[0]/sector_mass.sum(),2/37,places=10)
        self.assertAlmostEqual(sector_mass[1]/sector_mass.sum(),1/37,places=10)

class TestDetailedVisualizations(unittest.TestCase):
    def test_continuous_preview_mass_agrees_with_control_grid(self):
        cfg=Settings(preview_resolution=550,n_sample_radial=48,n_sample_angular=360)
        ns=default_nozzles(cfg)
        for n in ns:
            n.flow_kg_h=8.0
            n.sector_ml=[2 if i<10 else 1 for i in range(36)]
        z=35
        conservative=calculate_plane(ns,cfg,z)
        smooth=evaluate_continuous_field(ns,cfg,z)
        self.assertEqual(smooth.flux_kg_m2_s.shape,(550,550))
        step=(smooth.x_mm[1]-smooth.x_mm[0])*1e-3
        approx=smooth.masked_flux_kg_m2_s.sum()*step*step
        # Separate XY visual quadrature and source-polar conservative quadrature
        # agree within spatial-resolution tolerance, but are not identically equal.
        self.assertLess(abs(approx/conservative.inside_kg_s-1),0.01)
    def test_disabled_nozzle_does_not_add_continuous_flux(self):
        cfg=Settings(preview_resolution=240)
        ns=default_nozzles(cfg)
        ns[0].flow_kg_h=10
        m1=evaluate_continuous_field(ns,cfg,10)
        ns[0].enabled=False
        m2=evaluate_continuous_field(ns,cfg,10)
        self.assertGreater(m1.flux_kg_m2_s.max(),0)
        self.assertEqual(float(m2.flux_kg_m2_s.sum()),0)
    def test_polar_rose_respects_sector_one_rotation_and_flow(self):
        cfg=Settings()
        n=Nozzle(id=1,row='inner',theta_deg=0,radius_mm=230,flow_kg_h=12,
                 sector_ml=[3]+[1]*35)
        x,y=polar_epure_coordinates(n,0,cfg,5,n_angles=144)
        # At first angular index (sector 1) the lobe is longer than at sector 2.
        self.assertGreater(x[0]-230, x[4]-230)
        self.assertAlmostEqual(y[0],0,places=11)
        self.assertAlmostEqual(x[0],x[-1],places=11)
