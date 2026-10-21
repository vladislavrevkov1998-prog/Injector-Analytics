"""Quasi scans preserve conservative inertial-source influences and phases.

All speeds and relaxation/film parameters below are synthetic assumptions.
The independent stationary calculations evaluate the same prescribed source
state at each scanner timestamp; no pressure-propagation delay is claimed.
"""
from dataclasses import replace
import json
import time
import unittest
from unittest.mock import patch

import numpy as np

from fuelfield.core import calculate_plane,evaluate_continuous_field,project_to_json
from fuelfield.fuel_transport import TransportSamples,clear_transport_cache
from fuelfield.quasi import (CalculationCancelled,build_influence,calculate_quasi)
from fuelfield.transport_settings import synthetic_settings
from test_quasi import case
from test_v07_carrier import duct


PHASES=('suspended_cell_mass_kg_s','film_inner_cell_mass_kg_s','film_outer_cell_mass_kg_s')
MOMENTUM=('incoming_N','outgoing_N','carrier_N','wall_N','residual_N')


def confined_case(constant=False):
    cfg,nozzles,mode=case(constant=constant)
    cfg.r_in_mm=100.;cfg.r_out_mm=160.
    cfg.row_inner_mm=120.;cfg.row_outer_mm=150.
    cfg.z_mm=[0.,40.,100.,160.]
    cfg.geometry=duct((100.,110.),(160.,150.))
    cfg.geometry['sections']=[dict(id=f'S{i+1}',z_mm=z,is_outlet=z==160.)
                              for i,z in enumerate(cfg.z_mm)]
    cfg.fuel_transport=synthetic_settings().to_dict()
    cfg.fuel_transport.update(max_step_mm=4.,carrier_radial_cells=8,
        emission_radial_samples=2,emission_angular_samples=36)
    for n in nozzles:
        n.radius_mm=150. if n.row=='outer' else 120.
        n.alpha_deg=70.;n.z_tip_mm=0.
        n.geometry_overrides=list(set(n.geometry_overrides)|{'alpha_deg','z_tip_mm'})
    return cfg,nozzles,mode


def stationary_at_step(nozzles,cfg,mode,step,z):
    """Independent stationary runs for sources sharing each pressure channel."""
    results=[]
    for channel in mode.channels:
        subset=[n for n in nozzles if mode.manifold(n)==channel.id]
        state=replace(cfg,target_pressure_kgf_cm2=float(channel.delta_p(mode.n_steps)[step]))
        results.append(calculate_plane(subset,state,z))
    return results


class TestConfinedQuasi(unittest.TestCase):
    def setUp(self):clear_transport_cache()

    def test_source_beyond_outlet_is_not_silently_removed_from_every_scan(self):
        cfg,nozzles,_=confined_case(constant=True)
        working=next(n for n in nozzles if n.enabled and n.flow_kg_h>0)
        working.z_tip_mm=161.
        with self.assertRaisesRegex(ValueError,'внутри осевой области'):
            calculate_quasi(nozzles,cfg,preview=False)

    def test_constant_pressure_equals_stationary_mass_phases_momentum_and_travel(self):
        cfg,nozzles,_=confined_case(constant=True)
        result=calculate_quasi(nozzles,cfg,preview=False)
        for index,z in enumerate(cfg.z_mm):
            reference=calculate_plane(nozzles,cfg,z)
            snapshot=result.snapshot(index,3);composite=result.composites[index]
            for key in ('cell_mass_kg_s',*PHASES):
                np.testing.assert_allclose(getattr(snapshot,key),getattr(reference,key),rtol=4e-14,atol=1e-16)
                np.testing.assert_allclose(getattr(composite,key),getattr(reference,key),rtol=4e-14,atol=1e-16)
            for key in MOMENTUM:
                np.testing.assert_allclose(snapshot.transport_momentum[key],reference.transport_momentum[key],atol=2e-15)
            self.assertLess(snapshot.transport_momentum['relative_residual'],1e-13)
            self.assertEqual(snapshot.max_travel_time_s,reference.max_travel_time_s)
            self.assertAlmostEqual(snapshot.inside_kg_s,snapshot.injected_kg_s,places=15)
            self.assertEqual(snapshot.wall_bound_kg_s,0.)
            self.assertEqual(snapshot.inner_kg_s+snapshot.outer_kg_s,0.)
            self.assertIsNone(composite.transport_momentum)
            self.assertTrue(np.isnan(composite.rel_balance_error))
        self.assertGreater(result.snapshot(3,0).film_inner_cell_mass_kg_s.sum(),0.)
        self.assertGreater(result.snapshot(3,0).film_outer_cell_mass_kg_s.sum(),0.)
        self.assertLess(result.maximum_balance_error,1e-13)

    def test_two_manifolds_each_timestamp_matches_own_static_state_and_phase_column(self):
        cfg,nozzles,mode=confined_case()
        cfg.grid_start_deg=45.;cfg.theta_zero_deg=27.;cfg.theta_direction=-1
        mode=replace(mode,first_angle_deg=135.,direction=-1);cfg.quasi_settings=mode.to_dict()
        result=calculate_quasi(nozzles,cfg,preview=False)
        for index,z in enumerate(cfg.z_mm):
            for step,column in enumerate(mode.column_indices(cfg)):
                references=stationary_at_step(nozzles,cfg,mode,step,z)
                snapshot=result.snapshot(index,step)
                for key in ('cell_mass_kg_s',*PHASES):
                    expected=sum(getattr(p,key) for p in references)
                    np.testing.assert_allclose(getattr(snapshot,key),expected,rtol=5e-14,atol=1e-16)
                    np.testing.assert_allclose(getattr(result.composites[index],key)[:,column],expected[:,column],rtol=5e-14,atol=1e-16)
                for key in MOMENTUM:
                    expected=sum(np.asarray(p.transport_momentum[key]) for p in references)
                    np.testing.assert_allclose(snapshot.transport_momentum[key],expected,atol=3e-15)
                self.assertEqual(result.composites[index].scan_steps[column],step)
                self.assertEqual(result.composites[index].scan_times_s[column],mode.times[step])
                self.assertEqual(result.composites[index].scan_pressures[column]['A'],mode.channels[0].pressure[step])
        for balance in result.balances:
            self.assertAlmostEqual(balance['inside_kg_s'],balance['suspended_kg_s']+balance['film_inner_kg_s']+balance['film_outer_kg_s'],places=15)
            self.assertLess(balance['momentum_relative_residual'],1e-13)
            for axis in ('z','x','y'):
                self.assertAlmostEqual(balance[f'outgoing_{axis}_N']-balance[f'incoming_{axis}_N']
                    -balance[f'carrier_{axis}_N']-balance[f'wall_{axis}_N'],balance[f'residual_{axis}_N'],places=14)

    def test_exact_outer_film_is_binned_and_never_legacy_wall_bound(self):
        cfg,nozzles,_=confined_case()
        z=160.;_,edges=cfg.radial_centers_and_edges(z)
        weights=np.array([.2,.5,.3]);source=np.tile([40.,0.,0.],(3,1))
        outgoing=np.array([[1.,0.,0.],[1.,0.,0.],[30.,0.,0.]])
        carrier=np.array([[0.,0.,0.],[0.,0.,0.],[-10.,0.,0.]])
        samples=TransportSamples(np.array([edges[0],edges[-1],130.]),np.zeros(3),weights,
            np.array([1,2,0],dtype=np.uint8),outgoing,source,carrier,outgoing-source-carrier,np.array([.2,.3,.01]))
        with patch('fuelfield.fuel_transport.confined_samples',return_value=samples):
            operator=build_influence(nozzles[:1],cfg,z,np.array([[.04],[.08]]),preview=False)
        self.assertIsNone(operator.wall_inner);self.assertIsNone(operator.wall_outer)
        np.testing.assert_allclose(operator.weights.sum(axis=0),1.,atol=1e-15)
        snapshot=operator.plane(np.array([.08]))
        self.assertAlmostEqual(snapshot.cell_mass_kg_s.sum(),.08,places=15)
        self.assertAlmostEqual(snapshot.film_inner_cell_mass_kg_s[0].sum(),.016,places=15)
        self.assertAlmostEqual(snapshot.film_outer_cell_mass_kg_s[-1].sum(),.04,places=15)
        self.assertEqual(snapshot.film_inner_cell_mass_kg_s[1:].sum(),0.)
        self.assertEqual(snapshot.film_outer_cell_mass_kg_s[:-1].sum(),0.)
        self.assertEqual(snapshot.wall_bound_kg_s,0.)
        self.assertEqual(snapshot.inner_kg_s+snapshot.outer_kg_s,0.)
        self.assertLess(snapshot.transport_momentum['relative_residual'],1e-15)

    def test_unreached_disabled_and_zero_flow_sources_do_not_create_mass_or_transit(self):
        cfg,nozzles,mode=confined_case(constant=True)
        nozzles[0].enabled=False
        nozzles[1].z_tip_mm=60.
        result=calculate_quasi(nozzles,cfg,preview=False)
        self.assertTrue(np.all(result.flows[:,0]==0))
        for i,z in enumerate(cfg.z_mm):
            self.assertEqual(result.operators[i].reached[1],float(z>=60.))
            snapshot=result.snapshot(i,0)
            reference=calculate_plane(nozzles,cfg,z)
            np.testing.assert_allclose(snapshot.cell_mass_kg_s,reference.cell_mass_kg_s,atol=1e-16)
            self.assertEqual(snapshot.max_travel_time_s,reference.max_travel_time_s)
        empty=result.operators[-1].plane(np.zeros(len(nozzles)))
        self.assertEqual(empty.injected_kg_s+empty.inside_kg_s+empty.max_travel_time_s,0.)
        self.assertEqual(empty.transport_momentum['relative_residual'],0.)

    def test_pressure_only_scales_prescribed_influences_and_states_limits_of_scan(self):
        cfg,nozzles,mode=confined_case()
        mode=replace(mode,dt_s=1e-4);cfg.quasi_settings=mode.to_dict()
        result=calculate_quasi(nozzles,cfg,preview=False)
        manifest=result.manifest()
        self.assertGreater(manifest['max_travel_time_s'],mode.dt_s)
        self.assertEqual(manifest['travel_time_to_scan_step_ratio'],manifest['max_travel_time_s']/mode.dt_s)
        self.assertIsNone(manifest['wall_bound'])
        self.assertIsNone(manifest['composite_global_balance'])
        self.assertIn('no transport delay',manifest['pressure_response'])
        self.assertNotIn('tau=0',manifest['spatial_model'])
        self.assertTrue(any('Задержка переноса' in notice for notice in result.notices))
        self.assertTrue(any('время переноса должно быть мало' in notice for notice in result.notices))
        json.dumps(manifest,allow_nan=False)
        operator=result.operators[-1]
        a=operator.plane(result.flows[0]);b=operator.plane(3*result.flows[0])
        for key in ('cell_mass_kg_s',*PHASES):np.testing.assert_allclose(getattr(b,key),3*getattr(a,key),atol=1e-16)
        for key in MOMENTUM[:-1]:np.testing.assert_allclose(b.transport_momentum[key],3*np.asarray(a.transport_momentum[key]),atol=2e-15)
        self.assertEqual(a.max_travel_time_s,b.max_travel_time_s)

    def test_cached_preview_matches_static_and_scanner_updates_need_no_trajectory_work(self):
        cfg,nozzles,_=confined_case(constant=True)
        result=calculate_quasi(nozzles,cfg)
        for index,z in enumerate(cfg.z_mm):
            reference=evaluate_continuous_field(nozzles,cfg,z)
            np.testing.assert_allclose(result.smooth(index,3).flux_kg_m2_s,reference.flux_kg_m2_s,rtol=1e-12,atol=2e-12)
        # The interactive slider only multiplies cached influences. An engine
        # exception here detects accidental tracing on every UI time step.
        with patch('fuelfield.fuel_transport.confined_samples',side_effect=AssertionError('unexpected retrace')):
            started=time.perf_counter()
            for step in range(100):
                result.snapshot(3,step%8);result.smooth(3,step%8)
            elapsed=time.perf_counter()-started
        self.assertLess(elapsed,2.,f'100 cached preview updates took {elapsed:.3f} s')

    def test_variable_pressure_preview_uses_timestamp_not_composite_or_average(self):
        cfg,nozzles,mode=confined_case()
        result=calculate_quasi(nozzles,cfg)
        index=3;z=cfg.z_mm[index]
        previews=[]
        for step in (0,7):
            expected=0.
            for channel in mode.channels:
                subset=[n for n in nozzles if mode.manifold(n)==channel.id]
                state=replace(cfg,target_pressure_kgf_cm2=channel.pressure[step])
                expected=expected+evaluate_continuous_field(subset,state,z).flux_kg_m2_s
            preview=result.smooth(index,step).flux_kg_m2_s;previews.append(preview)
            np.testing.assert_allclose(preview,expected,rtol=2e-12,atol=2e-12)
        self.assertGreater(float(np.abs(previews[-1]-previews[0]).sum()),1.)

    def test_inputs_unchanged_reverse_order_reproducible_and_cancel(self):
        cfg,nozzles,_=confined_case()
        before=project_to_json(cfg,nozzles)
        first=calculate_quasi(nozzles,cfg,preview=False)
        second=calculate_quasi(list(reversed(nozzles)),cfg,preview=False)
        self.assertEqual(project_to_json(cfg,nozzles),before)
        for a,b in zip(first.composites,second.composites):
            for key in ('cell_mass_kg_s',*PHASES):np.testing.assert_array_equal(getattr(a,key),getattr(b,key))
        with self.assertRaises(CalculationCancelled):calculate_quasi(nozzles,cfg,cancel=lambda:True)


if __name__=='__main__':unittest.main()
