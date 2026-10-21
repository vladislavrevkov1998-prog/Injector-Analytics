"""Independent checks of the parameterized, one-way confined fuel operator.

Analytical tests validate mass-flow and momentum-flux units, not an actual
combustor flow. The step-refinement tests expose the finite-step contact law.
"""
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import unittest

import numpy as np
from scipy.optimize import brentq

from fuelfield.core import Settings,Nozzle
from fuelfield.transport_settings import synthetic_settings
from fuelfield.fuel_transport import confined_samples,clear_transport_cache,transport_cache_info
from fuelfield.chamber import Chamber
from test_v07_carrier import duct


def config(geometry=None,**params):
    settings=synthetic_settings()
    settings.cone_half_width_deg=0.
    settings.emission_radial_samples=1
    settings.emission_angular_samples=36
    settings.max_step_mm=2.
    for key,value in params.items():setattr(settings,key,value)
    return Settings(r_in_mm=100.,r_out_mm=160.,z_mm=[0.,40.,80.,160.],
                    fuel_transport=settings.to_dict(),geometry=geometry or duct())


def nozzle(**kwargs):
    return Nozzle(**dict(id=1,radius_mm=130.,alpha_deg=40.,flow_kg_h=3600.,**kwargs))


def analytic_time(z_mm,vz0,carrier=20.,tau=.005):
    distance=z_mm*.001
    return brentq(lambda t:carrier*t+(vz0-carrier)*tau*(1-np.exp(-t/tau))-distance,
                  0.,max(.1,2*distance/min(vz0,carrier)),xtol=1e-15)


class TestConfinedTransport(unittest.TestCase):
    def setUp(self):clear_transport_cache()

    def test_uniform_carrier_exact_relaxation_at_integration_station(self):
        cfg=config();n=nozzle();s=confined_samples(n,cfg,40.)
        vz0=40.*np.cos(np.deg2rad(20.));t=analytic_time(40.,vz0)
        self.assertTrue(np.all(s.phase==0))
        np.testing.assert_allclose(s.travel_time_s,t,rtol=0,atol=2e-13)
        carrier=np.array([20.,0.,0.])
        expected=carrier+(s.source_velocity_m_s-carrier)*np.exp(-t/.005)
        np.testing.assert_allclose(s.velocity_m_s,expected,atol=2e-9,rtol=0)
        expected_xy=np.array([130.,0.])+1000*s.source_velocity_m_s[:,1:]*.005*(1-np.exp(-t/.005))
        np.testing.assert_allclose(np.c_[s.x_mm,s.y_mm],expected_xy,atol=2e-8,rtol=0)
        np.testing.assert_allclose(s.carrier_delta_v_m_s,expected-s.source_velocity_m_s,atol=2e-9)
        np.testing.assert_allclose(s.wall_delta_v_m_s,0.,atol=1e-14)

    def test_axial_balanced_flow_has_no_drag_and_exact_travel_time(self):
        cfg=config(injection_speed_m_s=20.)
        n=replace(nozzle(),alpha_deg=0.)
        s=confined_samples(n,cfg,80.)
        np.testing.assert_allclose(s.x_mm,130.,atol=0)
        np.testing.assert_allclose(s.y_mm,0.,atol=0)
        np.testing.assert_allclose(s.travel_time_s,.004,atol=2e-16)
        np.testing.assert_allclose(s.velocity_m_s,np.tile([20.,0.,0.],(36,1)),atol=0)
        np.testing.assert_allclose(s.carrier_delta_v_m_s,0.,atol=0)
        self.assertAlmostEqual(s.weights_kg_s.sum(),1.,places=15)

    def test_long_relaxation_time_recovers_independent_ballistic_limit(self):
        cfg=config(relaxation_time_s=1e9,max_step_mm=8.)
        s=confined_samples(nozzle(),cfg,40.)
        source=s.source_velocity_m_s
        expected=np.array([130.,0.])+40.*source[:,1:]/source[:,0,None]
        np.testing.assert_allclose(np.c_[s.x_mm,s.y_mm],expected,atol=1e-10,rtol=0)
        np.testing.assert_allclose(s.travel_time_s,.04/source[:,0],atol=5e-15,rtol=0)
        np.testing.assert_allclose(s.velocity_m_s,source,atol=5e-11,rtol=0)

    def test_unmeasured_full_flow_does_not_come_from_sector_sum(self):
        cfg=config(emission_radial_samples=3,emission_angular_samples=144)
        observations=np.arange(1.,37.)**2
        n=replace(nozzle(),sector_ml=observations.tolist(),flow_kg_h=720.)
        s=confined_samples(n,cfg,0.)
        self.assertAlmostEqual(s.weights_kg_s.sum(),.2,places=15)
        source=s.source_velocity_m_s.reshape(3,144,3)
        # Derive observed directions geometrically from velocities. Sector 1
        # is centred towards the annular middle, i.e. -X for an outer nozzle.
        phi=(np.arctan2(-source[0,:,2],-source[0,:,1])%(2*np.pi))
        bins=np.floor(((np.rad2deg(phi)+5.)%360.)/10.).astype(int)
        measured=np.bincount(bins,weights=s.weights_kg_s.reshape(3,144).sum(axis=0),minlength=36)
        np.testing.assert_allclose(measured,.2*observations/observations.sum(),atol=2e-17)
        changed=replace(n,sector_ml=(observations*10000).tolist())
        np.testing.assert_allclose(confined_samples(changed,cfg,0.).weights_kg_s,s.weights_kg_s,atol=3e-18)

    def test_every_allowed_sector_subdivision_preserves_all_36_observations(self):
        observations=np.arange(1.,37.)**2
        n=replace(nozzle(),sector_ml=observations.tolist())
        for count in (36,72,108,144,180,252,324):
            cfg=config(emission_angular_samples=count)
            s=confined_samples(n,cfg,0.)
            phi=np.arctan2(-s.source_velocity_m_s[:,2],-s.source_velocity_m_s[:,1])
            bins=np.floor(((np.rad2deg(phi)+5.)%360.)/10.).astype(int)
            measured=np.bincount(bins,weights=s.weights_kg_s,minlength=36)
            with self.subTest(angular_samples=count):
                np.testing.assert_allclose(measured,observations/observations.sum(),atol=1e-15,rtol=0)

    def test_momentum_flux_matches_independent_analytic_drag_in_newtons(self):
        cfg=config();n=replace(nozzle(),flow_kg_h=1800.)
        s=confined_samples(n,cfg,40.);vz0=40.*np.cos(np.deg2rad(20.));t=analytic_time(40.,vz0)
        expected_force=.5*(20.-vz0)*(1-np.exp(-t/.005))
        force=np.sum(s.weights_kg_s[:,None]*s.carrier_delta_v_m_s,axis=0)
        self.assertAlmostEqual(force[0],expected_force,places=9)
        np.testing.assert_allclose(force[1:],0.,atol=1e-15)
        # This is kg/s times m/s, without another transit-time factor.
        self.assertGreater(abs(force[0]-expected_force*t),abs(expected_force)*.9)

    def test_radial_and_angular_output_grids_do_not_change_any_parcel(self):
        cfg=config();n=nozzle();first=confined_samples(n,cfg,80.)
        other=replace(cfg,n_radial=13,n_angular=280,grid_start_deg=17.,theta_zero_deg=91.,theta_direction=-1)
        second=confined_samples(n,other,80.)
        for name in ('x_mm','y_mm','weights_kg_s','phase','velocity_m_s','travel_time_s'):
            np.testing.assert_array_equal(getattr(first,name),getattr(second,name))

    def test_added_output_planes_do_not_change_common_result_or_ou_path(self):
        cfg=config(turbulence_rms_m_s=1.,turbulence_time_s=.002,seed=42)
        n=nozzle();first=confined_samples(n,cfg,80.)
        other=replace(cfg,z_mm=[0.,7.1,28.4,80.,115.3,160.])
        clear_transport_cache();second=confined_samples(n,other,80.)
        for name in ('x_mm','y_mm','phase','velocity_m_s','travel_time_s'):
            np.testing.assert_array_equal(getattr(first,name),getattr(second,name))

    def test_extending_output_range_of_implicit_cylinder_keeps_existing_ou_path(self):
        cfg=config(turbulence_rms_m_s=1.,turbulence_time_s=.002,seed=42)
        cfg.geometry={}
        first=confined_samples(nozzle(),cfg,80.)
        other=replace(cfg,z_mm=[0.,37.3,80.,160.3])
        clear_transport_cache();second=confined_samples(nozzle(),other,80.)
        for name in ('x_mm','y_mm','phase','velocity_m_s','travel_time_s'):
            np.testing.assert_array_equal(getattr(first,name),getattr(second,name))

    def test_global_rotation_and_flow_scaling_preserve_trajectory(self):
        cfg=config();n=nozzle();a=confined_samples(n,cfg,80.)
        b=confined_samples(replace(n,theta_deg=90.,flow_kg_h=900.),cfg,80.)
        np.testing.assert_allclose(b.x_mm,-a.y_mm,atol=2e-14)
        np.testing.assert_allclose(b.y_mm,a.x_mm,atol=2e-14)
        np.testing.assert_allclose(b.weights_kg_s,a.weights_kg_s/4.,atol=0)
        np.testing.assert_allclose(b.velocity_m_s[:,1],-a.velocity_m_s[:,2],atol=2e-15)
        np.testing.assert_allclose(b.velocity_m_s[:,2],a.velocity_m_s[:,1],atol=2e-15)
        np.testing.assert_array_equal(a.phase,b.phase)
        self.assertEqual(transport_cache_info()['entries'],1)

    def test_zero_rms_is_nozzle_permutation_invariant_nonzero_is_reproducible(self):
        cfg=config();a=nozzle();b=replace(a,id=72)
        one=confined_samples(a,cfg,80.);two=confined_samples(b,cfg,80.)
        np.testing.assert_array_equal(one.x_mm,two.x_mm)
        cfg.fuel_transport.update(turbulence_rms_m_s=2.,turbulence_time_s=.003,seed=17)
        one=confined_samples(a,cfg,80.);confined_samples(b,cfg,80.)
        clear_transport_cache();again=confined_samples(a,cfg,80.)
        np.testing.assert_array_equal(one.x_mm,again.x_mm)
        np.testing.assert_array_equal(one.velocity_m_s,again.velocity_m_s)
        cfg.fuel_transport['seed']=18
        different=confined_samples(a,cfg,80.)
        self.assertGreater(np.max(abs(one.x_mm-different.x_mm)),1e-4)

    def test_narrowing_preserves_nozzle_below_downstream_inner_radius(self):
        cfg=config(duct((100.,140.),(180.,160.)),max_step_mm=1.,relaxation_time_s=.02)
        n=replace(nozzle(),radius_mm=110.,alpha_deg=0.)
        s=confined_samples(n,cfg,160.)
        self.assertAlmostEqual(s.weights_kg_s.sum(),1.,places=15)
        self.assertTrue(np.all(s.phase==1))
        np.testing.assert_allclose(np.hypot(s.x_mm,s.y_mm),140.,atol=1e-12)
        np.testing.assert_allclose(np.linalg.norm(s.velocity_m_s,axis=1),1.,atol=1e-15)
        self.assertTrue(np.all(s.travel_time_s>.01))
        np.testing.assert_allclose(s.velocity_m_s-s.source_velocity_m_s,
                                   s.carrier_delta_v_m_s+s.wall_delta_v_m_s,atol=1e-15)

    def test_film_stays_separate_through_subsequent_widening(self):
        geometry=duct((100.,100.),(160.,160.))
        geometry['primitives'][0:1]=[
            dict(id='i1',wall='inner',kind='line',unit='mm',start=[0.,100.],end=[80.,140.]),
            dict(id='i2',wall='inner',kind='line',unit='mm',start=[80.,140.],end=[160.,100.])]
        cfg=config(geometry,relaxation_time_s=.05)
        n=replace(nozzle(),radius_mm=110.,alpha_deg=0.)
        mid=confined_samples(n,cfg,80.);out=confined_samples(n,cfg,160.)
        self.assertTrue(np.all(mid.phase==1))
        np.testing.assert_array_equal(mid.phase,out.phase)
        np.testing.assert_allclose(np.hypot(out.x_mm,out.y_mm),100.,atol=1e-12)
        np.testing.assert_allclose(out.travel_time_s-mid.travel_time_s,np.hypot(80.,40.)*.001,atol=1e-14)

    def test_source_and_nonfinite_output_errors_are_explicit(self):
        cfg=config();n=nozzle()
        for bad in (replace(n,radius_mm=90.),replace(n,z_tip_mm=-1.),replace(n,z_tip_mm=161.),
                    replace(n,alpha_deg=190.),replace(n,alpha_deg=-1.)):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):confined_samples(bad,cfg,160.)
        for z in (float('nan'),float('inf'),161.):
            with self.assertRaises(ValueError):confined_samples(n,cfg,z)
        self.assertEqual(len(confined_samples(replace(n,z_tip_mm=30.),cfg,20.).weights_kg_s),0)

    def test_output_sampling_converges_to_analytic_cylinder_solution(self):
        n=nozzle();z=37.3;vz0=40.*np.cos(np.deg2rad(20.));t=analytic_time(z,vz0)
        errors=[]
        for step in (8.,4.,2.,1.):
            s=confined_samples(n,config(max_step_mm=step),z)
            expected_xy=np.array([130.,0.])+1000*s.source_velocity_m_s[:,1:]*.005*(1-np.exp(-t/.005))
            errors.append(np.max(np.linalg.norm(np.c_[s.x_mm,s.y_mm]-expected_xy,axis=1)))
        self.assertLess(errors[-1],errors[0]/10.)
        self.assertLess(errors[-1],.001)

    def test_actual_s_profile_keeps_every_weight_inside_exact_walls(self):
        path=Path(__file__).resolve().parents[1]/'examples/demo_project_v063_s_profile.json'
        raw=json.loads(path.read_text())['settings']['geometry']
        cfg=config(raw,max_step_mm=2.,emission_angular_samples=72)
        cfg.z_mm=[0.,40.,80.,120.,160.]
        n=replace(nozzle(),radius_mm=230.,alpha_deg=20.)
        model=Chamber(raw)
        for z in cfg.z_mm:
            s=confined_samples(n,cfg,z);r=np.hypot(s.x_mm,s.y_mm);rin,rout=model.bounds(z)
            self.assertTrue(np.all(r>=rin-1e-7));self.assertTrue(np.all(r<=rout+1e-7))
            self.assertAlmostEqual(s.weights_kg_s.sum(),1.,places=15)
            self.assertEqual(len(s.weights_kg_s),72)
            self.assertTrue(np.all(np.isfinite(s.travel_time_s)))

    def test_actual_curved_profile_refinement_approaches_fine_reference(self):
        path=Path(__file__).resolve().parents[1]/'examples/demo_project_v063_s_profile.json'
        raw=json.loads(path.read_text())['settings']['geometry']
        values=[]
        # Refine the carrier/flight axial mesh together. The fixed radial mesh
        # and tolerance mean this checks numerical consistency, not truth of
        # the carrier closure or convergence to the actual turbulent chamber.
        for step in (4.,1.,.25,.125):
            cfg=config(raw,max_step_mm=step,carrier_radial_cells=32)
            cfg.z_mm=[0.,160.]
            s=confined_samples(replace(nozzle(),radius_mm=250.,alpha_deg=0.),cfg,160.)
            self.assertTrue(np.all(s.phase==0))
            values.append([s.x_mm[0],s.travel_time_s[0]])
        values=np.array(values);error=np.abs(values[:-1]-values[-1])
        self.assertTrue(np.all(error[1:]<error[:-1]))
        self.assertLess(error[-1,0],.05)
        self.assertLess(error[-1,1],1e-6)


if __name__=='__main__':unittest.main()
