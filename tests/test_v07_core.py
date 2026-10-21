"""Independent confined-flow integration, phase balance and preview checks."""
from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace
import unittest

import numpy as np

from fuelfield.chamber import legacy_geometry
from fuelfield.core import (Settings,Nozzle,calculate_plane,confined_cell_masses,
    evaluate_continuous_field,polar_epure_coordinates,sample_kernel)
from fuelfield.fuel_transport import confined_samples,clear_transport_cache
from fuelfield.transport_preview import build_polar_preview
from fuelfield.transport_settings import synthetic_settings


def duct_case(profile=False):
    parameters=synthetic_settings().to_dict()
    parameters.update(cone_half_width_deg=0.,max_step_mm=2.,
        emission_radial_samples=2,emission_angular_samples=72)
    cfg=Settings(r_in_mm=100.,r_out_mm=160.,row_inner_mm=115.,row_outer_mm=145.,
        z_mm=[0.,40.,80.],fuel_transport=parameters,n_angular=140,
        n_sample_radial=8,n_sample_angular=72,preview_resolution=200)
    if profile:
        cfg.geometry=legacy_geometry(cfg);cfg.geometry['enabled']=True
        cfg.geometry['primitives'][0]['end'][1]=145.
        cfg.geometry['primitives'][1]['end'][1]=155.
    n=Nozzle(1,row='inner',radius_mm=130.,alpha_deg=0.,flow_kg_h=36.)
    return cfg,n


class TestConfinedCore(unittest.TestCase):
    def setUp(self):clear_transport_cache()

    def test_working_source_outside_tract_is_rejected_even_before_its_tip(self):
        cfg,n=duct_case()
        for invalid in (replace(n,z_tip_mm=81.),replace(n,z_tip_mm=-1.),
                        replace(n,z_tip_mm=40.,radius_mm=90.)):
            with self.subTest(tip=invalid.z_tip_mm,radius=invalid.radius_mm):
                with self.assertRaisesRegex(ValueError,'Форсунка 1'):
                    calculate_plane([invalid],cfg,0.)
                with self.assertRaisesRegex(ValueError,'Форсунка 1'):
                    evaluate_continuous_field([invalid],cfg,0.,resolution=200)
        valid=replace(n,z_tip_mm=40.)
        self.assertEqual(calculate_plane([valid],cfg,0.).injected_kg_s,0.)
        unused=replace(n,z_tip_mm=81.,flow_kg_h=0.)
        self.assertEqual(calculate_plane([unused],cfg,0.).injected_kg_s,0.)
        self.assertEqual(evaluate_continuous_field([unused],cfg,0.,resolution=200).flux_kg_m2_s.sum(),0.)

    def assert_conserved(self,plane,expected):
        self.assertAlmostEqual(plane.injected_kg_s,expected,places=14)
        self.assertAlmostEqual(plane.inside_kg_s,expected,places=14)
        self.assertLess(abs(plane.rel_balance_error),2e-12)
        self.assertEqual(plane.inner_kg_s,0.);self.assertEqual(plane.outer_kg_s,0.)
        self.assertEqual(plane.wall_bound_kg_s,0.)
        np.testing.assert_allclose(plane.suspended_cell_mass_kg_s+
            plane.film_inner_cell_mass_kg_s+plane.film_outer_cell_mass_kg_s,
            plane.cell_mass_kg_s,rtol=2e-14,atol=1e-16)
        self.assertLess(plane.transport_momentum['relative_residual'],2e-12)

    def test_cylindrical_drag_matches_the_exact_relaxation_solution(self):
        cfg,n=duct_case();samples=confined_samples(n,cfg,80.)
        time=samples.travel_time_s;tau=cfg.fuel_transport['relaxation_time_s']
        expected_distance=20.*time+20.*tau*(-np.expm1(-time/tau))
        np.testing.assert_allclose(expected_distance,.08,rtol=2e-10,atol=2e-12)
        np.testing.assert_allclose(samples.velocity_m_s[:,0],20.+20.*np.exp(-time/tau),rtol=2e-11)
        np.testing.assert_allclose(samples.velocity_m_s[:,1:],0.,atol=1e-14)
        np.testing.assert_allclose(np.hypot(samples.x_mm,samples.y_mm),130.,atol=1e-12)
        self.assertTrue(np.all(samples.phase==0))
        plane=calculate_plane([n],cfg,80.);self.assert_conserved(plane,.01)
        np.testing.assert_allclose(plane.transport_momentum['incoming_N'],[.4,0.,0.],atol=2e-14)
        np.testing.assert_allclose(plane.transport_momentum['wall_N'],0.,atol=2e-14)
        self.assertAlmostEqual(plane.max_travel_time_s,float(time.max()),places=14)

    def test_source_inside_at_inlet_is_not_lost_when_outlet_rmin_exceeds_it(self):
        cfg,n=duct_case(profile=True);n.radius_mm=110.
        cfg.fuel_transport['relaxation_time_s']=.1
        self.assertLess(n.radius_mm,cfg.bounds_at(80.)[0])
        plane=calculate_plane([n],cfg,80.);self.assert_conserved(plane,.01)
        self.assertAlmostEqual(plane.film_inner_cell_mass_kg_s.sum(),.01,places=14)
        self.assertEqual(plane.suspended_cell_mass_kg_s.sum(),0.)
        self.assertEqual(plane.film_outer_cell_mass_kg_s.sum(),0.)
        self.assertTrue(np.all(plane.film_inner_cell_mass_kg_s[1:]==0))
        self.assertGreater(plane.max_travel_time_s,.04)
        self.assertIn('плёнк',plane.transport_notice)
        self.assertNotIn('wall-bound',plane.geometry_notice)

    def test_exact_outer_wall_film_stays_in_the_last_cell_with_explicit_impulse(self):
        cfg,n=duct_case();n.radius_mm=160.
        plane=calculate_plane([n],cfg,80.);self.assert_conserved(plane,.01)
        self.assertAlmostEqual(plane.film_outer_cell_mass_kg_s[-1].sum(),.01,places=14)
        self.assertTrue(np.all(plane.film_outer_cell_mass_kg_s[:-1]==0))
        np.testing.assert_allclose(plane.transport_momentum['outgoing_N'],[.01,0.,0.],atol=2e-14)
        np.testing.assert_allclose(plane.transport_momentum['wall_N'],[-.39,0.,0.],atol=2e-13)
        np.testing.assert_allclose(plane.transport_momentum['carrier_N'],0.,atol=2e-14)

    def test_phase_bins_accept_both_surfaces_and_reject_fuel_outside(self):
        cfg,_=duct_case();_,edges=cfg.radial_centers_and_edges(80.)
        samples=SimpleNamespace(x_mm=np.array([100.,160.,130.,130.]),
            y_mm=np.zeros(4),weights_kg_s=np.array([1.,2.,3.,4.]),phase=np.array([1,2,0,0]))
        suspended,inner,outer=confined_cell_masses(samples,cfg,edges)
        self.assertEqual(suspended.sum(),7.);self.assertEqual(inner[0].sum(),1.)
        self.assertEqual(outer[-1].sum(),2.)
        samples.x_mm[0]=99.999
        with self.assertRaisesRegex(ValueError,'вне точных стенок'):
            confined_cell_masses(samples,cfg,edges)
        samples.x_mm[0]=110.
        with self.assertRaisesRegex(ValueError,'не находится'):
            confined_cell_masses(samples,cfg,edges)

    def test_output_grid_and_measurement_frame_cannot_change_transported_mass(self):
        cfg,n=duct_case(profile=True);n.alpha_deg=40.
        original=deepcopy(cfg.fuel_transport)
        reference=confined_samples(n,cfg,80.)
        for nr,na,zero,direction in ((1,8,0.,1),(13,141,37.2,-1),(40,1440,223.,1)):
            variant=replace(cfg,n_radial=nr,n_angular=na,theta_zero_deg=zero,theta_direction=direction)
            plane=calculate_plane([n],variant,80.);self.assert_conserved(plane,.01)
            current=confined_samples(n,variant,80.)
            np.testing.assert_array_equal(current.x_mm,reference.x_mm)
            np.testing.assert_array_equal(current.y_mm,reference.y_mm)
            for identity,values in enumerate((plane.suspended_cell_mass_kg_s,
                    plane.film_inner_cell_mass_kg_s,plane.film_outer_cell_mass_kg_s)):
                self.assertAlmostEqual(float(values.sum()),float(reference.weights_kg_s[reference.phase==identity].sum()),places=14)
        self.assertEqual(cfg.fuel_transport,original)

    def test_disabled_and_before_tip_sources_are_absent_and_unit_weights_are_normalized(self):
        cfg,n=duct_case();n.z_tip_mm=40.
        self.assertEqual(len(sample_kernel(n,0.,cfg)[0]),0)
        self.assert_conserved(calculate_plane([n],cfg,0.),0.)
        self.assertAlmostEqual(sample_kernel(n,80.,cfg,unit_mass=True)[2].sum(),1.,places=13)
        n.enabled=False;self.assert_conserved(calculate_plane([n],cfg,80.),0.)

    def test_rose_keeps_source_shares_and_source_clock_without_downstream_rotation(self):
        cfg,n=duct_case();n.sector_ml=[10.]+[1.]*35;n.clock_deg=17.
        n.rotation_deg_per_mm=3.;cfg.angular_model='linear';n.decay_length_mm=1.
        near=polar_epure_coordinates(n,0.,cfg,10.)
        far=polar_epure_coordinates(n,80.,cfg,10.)
        for actual,expected in zip(far,near):np.testing.assert_array_equal(actual,expected)
        legacy=deepcopy(cfg);legacy.fuel_transport['mode']='legacy'
        legacy_far=polar_epure_coordinates(n,80.,legacy,10.)
        self.assertGreater(float(np.max(np.abs(legacy_far[0]-near[0]))),1.)


class TestTransportPreview(unittest.TestCase):
    def setUp(self):clear_transport_cache()

    def test_fine_polar_bins_and_smoothing_preserve_each_phase_integral(self):
        cfg,n=duct_case(profile=True);n.alpha_deg=40.
        plane=calculate_plane([n],cfg,80.)
        for smooth in (False,True):
            field=build_polar_preview([n],cfg,80.,radial_cells=79,angular_cells=720,smooth=smooth)
            self.assertAlmostEqual(float(np.sum(field.flux_kg_m2_s*field.cell_area_m2)),.01,places=14)
            for actual,expected in ((field.suspended_cell_mass_kg_s,plane.suspended_cell_mass_kg_s),
                    (field.film_inner_cell_mass_kg_s,plane.film_inner_cell_mass_kg_s),
                    (field.film_outer_cell_mass_kg_s,plane.film_outer_cell_mass_kg_s)):
                self.assertGreaterEqual(float(actual.min()),0.)
                self.assertAlmostEqual(float(actual.sum()),float(expected.sum()),places=14)
        unit=build_polar_preview([n],cfg,80.,unit_mass=True)
        self.assertAlmostEqual(unit.cell_mass_kg_s.sum(),1.,places=13)

    def test_narrow_channel_wall_film_remains_visible_at_both_preview_resolutions(self):
        cfg,n=duct_case(profile=True);n.radius_mm=110.
        cfg.fuel_transport['relaxation_time_s']=.1
        for dim in (200,480):
            field=evaluate_continuous_field([n],cfg,80.,resolution=dim)
            self.assertEqual(field.flux_kg_m2_s.shape,(dim,dim))
            self.assertTrue(np.isfinite(field.flux_kg_m2_s).all())
            self.assertGreater(np.count_nonzero(field.flux_kg_m2_s),2)
            self.assertGreater(float(field.flux_kg_m2_s.max()),0.)
            self.assertIn('визуальное сглаживание',field.display_notice)
            x,y=np.meshgrid(field.x_mm,field.y_mm);r=np.hypot(x,y);rin,rout=cfg.bounds_at(80.)
            self.assertTrue(np.all(field.masked_flux_kg_m2_s.mask[(r<rin)|(r>rout)]))

    def test_unit_preview_scales_linearly_and_never_changes_audited_matrix(self):
        cfg,n=duct_case();n.alpha_deg=35.;before=calculate_plane([n],cfg,80.)
        unit=evaluate_continuous_field([n],cfg,80.,unit_mass=True)
        actual=evaluate_continuous_field([n],cfg,80.)
        np.testing.assert_allclose(actual.flux_kg_m2_s,.01*unit.flux_kg_m2_s,rtol=2e-12,atol=1e-13)
        n.flow_kg_h=72.;doubled=evaluate_continuous_field([n],cfg,80.)
        np.testing.assert_allclose(doubled.flux_kg_m2_s,2.*actual.flux_kg_m2_s,rtol=2e-12,atol=1e-13)
        n.flow_kg_h=36.;after=calculate_plane([n],cfg,80.)
        np.testing.assert_array_equal(before.cell_mass_kg_s,after.cell_mass_kg_s)
        unmasked=evaluate_continuous_field([n],cfg,80.,mask_track=False)
        self.assertFalse(np.any(unmasked.masked_flux_kg_m2_s.mask))


if __name__=='__main__':unittest.main()
