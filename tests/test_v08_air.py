"""Air ledgers, physical rays and alpha checked against independent identities."""
from copy import deepcopy
from types import SimpleNamespace
import unittest
import numpy as np
from fuelfield.core import Settings, default_nozzles
from fuelfield.coordinates import grid_angles
from fuelfield.air_sources import validate_air_model, resolve_sources
from fuelfield.air_balance import solve_balance, didenko_reference, didenko_mixed_cumulative
from fuelfield.air_jets import local_to_cartesian, cartesian_to_local, trace_jets, section_crossing
from fuelfield.air_distribution import compute_air_field, normalized_kernel
from fuelfield.mixture_field import compute_mixture


def source(identity='S', **changes):
    data = dict(id=identity, kind='hole', method='fixed', flow_kg_s=1.,
                z_mm=0., radius_mm=250., theta_deg=0., direction=[1., 0., 0.],
                sigma_r_mm=10., sigma_theta_deg=10., mixed_fraction=.25)
    data.update(changes)
    return data


def configuration(sources=None, **changes):
    cfg = Settings(n_radial=2, n_angular=8, z_mm=[0., 20., 60.])
    cfg.air_model = dict(schema_version=1, enabled=True, total_kg_s=1., l0=2.,
                         max_path_mm=120., step_mm=1., sources=sources or [source()])
    cfg.air_model.update(changes)
    return cfg


def plane(cfg, z=20., fuel=None):
    centers, edges = cfg.radial_centers_and_edges(z)
    area = np.repeat((np.pi*np.diff(edges**2)/cfg.n_angular*1e-6)[:, None], cfg.n_angular, axis=1)
    mass = np.ones(area.shape) if fuel is None else np.asarray(fuel, dtype=float)
    return SimpleNamespace(z_mm=z, cell_mass_kg_s=mass, cell_area_m2=area,
                           radial_edges_mm=edges, radial_centers_mm=centers,
                           theta_centers_deg=grid_angles(cfg), is_composite=False,
                           suspended_cell_mass_kg_s=None)


class AirBalanceTests(unittest.TestCase):
    def test_numeric_ledger_with_fixed_fraction_and_residual_area(self):
        cfg = configuration([source('F', flow_kg_s=2.), source('S', method='fraction', share=.1),
                             source('A', method='area', area_mm2=1.),
                             source('B', method='area', area_mm2=3.)],
                            total_kg_s=10., area_hypothesis_confirmed=True)
        result = solve_balance(cfg.air_model)
        self.assertEqual(result.flows, dict(F=2., S=1., A=1.75, B=5.25))
        self.assertEqual(result.unassigned_kg_s, 0.)
        self.assertAlmostEqual(result.rel_error, 0.)

    def test_partial_area_allocation_does_not_create_residual_source(self):
        cfg = configuration([source('F', flow_kg_s=2.), source('A', method='area', area_mm2=4.)],
                            total_kg_s=10., area_hypothesis_confirmed=True, area_allocation_fraction=.5)
        result = solve_balance(cfg.air_model)
        self.assertEqual(result.flows, dict(F=2., A=4.))
        self.assertEqual(result.unassigned_kg_s, 4.)
        self.assertEqual(set(result.flows), {'F', 'A'})
        self.assertAlmostEqual(sum(result.categories.values()), 10.)

    def test_effective_area_requires_consent_and_explicit_cd(self):
        raw = configuration([source(method='effective_area', area_mm2=10.)]).air_model
        with self.assertRaises(ValueError):
            solve_balance(raw)
        raw['sources'][0]['cd'] = .5
        with self.assertRaisesRegex(ValueError, 'гипотезу'):
            solve_balance(raw)
        raw['area_hypothesis_confirmed'] = True
        self.assertEqual(solve_balance(raw).flows['S'], 1.)

    def test_invalid_inputs_and_duplicate_physical_sources_fail_without_mutation(self):
        for field, value in [('flow_kg_s', -1.), ('flow_kg_s', float('nan')),
                             ('direction', [0., 0., 0.]), ('sigma_r_mm', -1.)]:
            raw = configuration([source(**{field: value})]).air_model
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                validate_air_model(raw)
        raw = configuration([source(flow_kg_s=2.)]).air_model
        before = deepcopy(raw)
        with self.assertRaisesRegex(ValueError, 'превышают'):
            solve_balance(raw)
        self.assertEqual(raw, before)
        raw = configuration([source('S1', kind='swirler', nozzle_id=1), source('S2', kind='swirler', nozzle_id=1)]).air_model
        with self.assertRaisesRegex(ValueError, 'Повторное'):
            validate_air_model(raw)

    def test_unknown_source_and_disabled_source_have_no_assigned_mass(self):
        raw = configuration([source('U', method='unknown'), source('D', enabled=False)]).air_model
        result = solve_balance(raw)
        self.assertEqual(result.flows, {'U': 0.})
        self.assertEqual(result.unassigned_kg_s, 1.)

    def test_book_oracle_is_recomputed_from_inlet_shares(self):
        reference = didenko_reference()
        self.assertAlmostEqual(sum(reference['group_shares'].values()), 1.)
        np.testing.assert_allclose(didenko_mixed_cumulative(), reference['mixed_cumulative'], rtol=0, atol=2e-15)


class AirSpatialTests(unittest.TestCase):
    def test_zero_width_limit_shares_only_adjacent_boundary_cells(self):
        cfg = configuration([source(theta_deg=22.5, sigma_r_mm=0., sigma_theta_deg=0., mixed_fraction=0.)])
        result = compute_air_field(plane(cfg), cfg, default_nozzles(cfg))
        expected = np.zeros((2, 8)); expected[:, :2] = .25
        np.testing.assert_array_equal(result.air_total_kg_s, expected)

    def test_staged_portions_have_reproducible_nonoverlapping_flow_ledger(self):
        stages = [dict(id='first', z_start_mm=0., z_end_mm=10., fraction=.2),
                  dict(id='second', z_start_mm=10., z_end_mm=20., fraction=.3),
                  dict(id='third', z_start_mm=20., z_end_mm=30., fraction=.5)]
        cfg = configuration([source(entrainment_stages=stages)])
        p = plane(cfg, 15.)
        a = compute_air_field(p, cfg, default_nozzles(cfg))
        b = compute_air_field(p, cfg, default_nozzles(cfg))
        self.assertAlmostEqual(a.air_mixed_kg_s.sum(), .35)
        portions = a.source_states[0]['portions']
        self.assertAlmostEqual(sum(item['assigned_kg_s'] for item in portions), 1.)
        self.assertAlmostEqual(sum(item['mixed_kg_s'] for item in portions), .35)
        self.assertAlmostEqual(sum(item['core_kg_s'] for item in portions), .65)
        np.testing.assert_array_equal(a.air_total_kg_s, b.air_total_kg_s)

    def test_wall_layer_follows_analytic_sloping_wall_and_normals_check_sign(self):
        cfg = configuration([source(kernel='wall', wall='outer', radius_mm=300.)])
        cfg.geometry = dict(schema_version=1, enabled=True, base_z_mm=0., outlet_z_mm=60.,
                            primitives=[dict(id='I', kind='line', wall='inner', start=[0., 200.], end=[60., 200.]),
                                        dict(id='O', kind='line', wall='outer', start=[0., 300.], end=[60., 280.])],
                            sections=[dict(id='S0', z_mm=0., is_outlet=False), dict(id='S1', z_mm=20., is_outlet=False),
                                      dict(id='OUT', z_mm=60., is_outlet=True)], objects=[], groups=[])
        result = compute_air_field(plane(cfg, 20.), cfg, default_nozzles(cfg))
        self.assertAlmostEqual(result.air_total_kg_s.sum(), 1.)
        self.assertAlmostEqual(result.source_states[0]['center'][1], 300.-20./3.)
        cfg.air_model['sources'][0].update(kernel='directed', direction=[0., 1., 0.])
        with self.assertRaisesRegex(ValueError, 'направление'):
            resolve_sources(cfg.air_model, cfg, default_nozzles(cfg))

    def test_physical_step_converges_for_crossing_jets(self):
        cfg = configuration([source('A', flow_kg_s=.5, theta_deg=359., direction=[1., .1, .05]),
                             source('B', flow_kg_s=.5, theta_deg=1., direction=[1., -.1, -.05])],
                            interaction_strength=2., max_path_mm=50.)
        nozzles = default_nozzles(cfg)
        def center(step):
            cfg.air_model['step_mm'] = step
            out = compute_air_field(plane(cfg, 20.), cfg, nozzles)
            _, r, t = out.source_states[0]['center']
            return np.array([r*np.cos(np.deg2rad(t)), r*np.sin(np.deg2rad(t))])
        coarse, fine, reference = center(2.), center(1.), center(.125)
        self.assertLess(np.linalg.norm(fine-reference), np.linalg.norm(coarse-reference))

    def test_kernel_quadrature_refines_toward_independent_continuous_integral(self):
        from scipy.integrate import quad
        cfg = configuration([source(theta_deg=22.5, sigma_r_mm=20., sigma_theta_deg=30.)])
        s = resolve_sources(cfg.air_model, cfg, default_nozzles(cfg))[0]
        coarse = plane(cfg)
        w_coarse = normalized_kernel(coarse, s, [20., 250., 22.5])
        radial = np.array([quad(lambda r: r*np.exp(-.5*((r-250.)/20.)**2), a, b)[0]
                           for a, b in zip(coarse.radial_edges_mm[:-1], coarse.radial_edges_mm[1:])])
        angular = np.array([quad(lambda t: np.exp(-.5*((((t-22.5+180.) % 360.)-180.)/30.)**2),
                                theta-22.5, theta+22.5)[0] for theta in coarse.theta_centers_deg])
        expected = np.outer(radial, angular); expected /= expected.sum()
        cfg.n_radial = 4; cfg.n_angular = 16; cfg.grid_start_deg = -11.25
        fine = plane(cfg)
        w_fine = normalized_kernel(fine, s, [20., 250., 22.5]).reshape(2, 2, 8, 2).sum(axis=(1, 3))
        self.assertLess(np.linalg.norm(w_fine-expected), np.linalg.norm(w_coarse-expected))

    def test_uniform_uses_annular_area_and_does_not_mutate_fuel(self):
        cfg = configuration(distribution='uniform')
        p = plane(cfg)
        original = p.cell_mass_kg_s.copy()
        result = compute_mixture(p, cfg, default_nozzles(cfg))
        np.testing.assert_allclose(result.air_total_kg_s, p.cell_area_m2/p.cell_area_m2.sum())
        np.testing.assert_array_equal(p.cell_mass_kg_s, original)
        self.assertAlmostEqual(result.air_core_kg_s.sum(), .75)
        self.assertAlmostEqual(result.air_mixed_kg_s.sum(), .25)

    def test_positive_local_kernels_normalize_even_when_narrow(self):
        cfg = configuration([source(sigma_r_mm=.00001, sigma_theta_deg=.00001)])
        p = plane(cfg)
        s = resolve_sources(cfg.air_model, cfg, default_nozzles(cfg))[0]
        weights = normalized_kernel(p, s, [20., 250., 22.5])
        self.assertTrue(np.isfinite(weights).all())
        self.assertGreaterEqual(weights.min(), 0.)
        self.assertAlmostEqual(weights.sum(), 1.)

    def test_local_basis_round_trip_and_rotation_equivariance(self):
        vector = np.array([.6, -.8, .1]); vector /= np.linalg.norm(vector)
        for theta in (0., 47., 310.):
            np.testing.assert_allclose(cartesian_to_local(local_to_cartesian(vector, theta), theta), vector, atol=1e-15)
        cfg = configuration([source(theta_deg=0., direction=[1., .1, .04])])
        result = compute_air_field(plane(cfg), cfg, default_nozzles(cfg))
        cfg.air_model['sources'][0]['theta_deg'] += 45.
        rotated = compute_air_field(plane(cfg), cfg, default_nozzles(cfg))
        np.testing.assert_allclose(rotated.air_total_kg_s, np.roll(result.air_total_kg_s, 1, axis=1), atol=2e-15)

    def test_output_sections_do_not_change_physical_paths_or_entrainment(self):
        cfg = configuration([source(direction=[1., .1, .02], mixing_length_mm=20.)])
        nozzles = default_nozzles(cfg)
        result = compute_air_field(plane(cfg, 20.), cfg, nozzles)
        cfg.z_mm = [10., 20., 30., 40., 50., 60.]
        other = compute_air_field(plane(cfg, 20.), cfg, nozzles)
        np.testing.assert_array_equal(result.air_total_kg_s, other.air_total_kg_s)
        self.assertEqual(result.source_states, other.source_states)

    def test_radial_and_reverse_are_not_silently_turned_downstream(self):
        cfg = configuration([source('R', z_mm=40., direction=[-1., 0., 0.])])
        result = compute_mixture(plane(cfg, 20.), cfg, default_nozzles(cfg))
        self.assertAlmostEqual(result.reverse_air_kg_s.sum(), 1.)
        self.assertEqual(result.air_total_kg_s.sum(), 0.)
        self.assertTrue(np.isnan(result.alpha_total).all())
        self.assertEqual(result.source_states[0]['status'], 'reverse')
        cfg = configuration([source(direction=[0., 1., 0.])])
        result = compute_air_field(plane(cfg, 20.), cfg, default_nozzles(cfg))
        self.assertEqual(result.air_total_kg_s.sum(), 0.)
        self.assertAlmostEqual(sum(result.source_states[0][key] for key in ('not_crossed_kg_s', 'outside_kg_s')), 1.)

    def test_no_neighbour_and_zero_penetration_keep_finite_direction(self):
        cfg = configuration([source(direction=[1., .15, .1], penetration_factor=0.)], interaction_strength=3.)
        model = validate_air_model(cfg.air_model)
        sources = resolve_sources(model, cfg, default_nozzles(cfg))
        paths = trace_jets(sources, solve_balance(model).flows, cfg, model)
        self.assertTrue(np.isfinite(paths.positions).all())
        expected = local_to_cartesian(sources[0]['direction'], sources[0]['theta_deg'])
        np.testing.assert_allclose(paths.directions[:, 0], np.broadcast_to(expected, paths.directions[:, 0].shape))

    def test_interaction_changes_directions_without_registry_order_bias(self):
        cfg = configuration([source('A', flow_kg_s=.5, theta_deg=0., direction=[1., .2, 0.]),
                             source('B', flow_kg_s=.5, theta_deg=4., direction=[1., -.2, 0.])], interaction_strength=3.)
        nozzles = default_nozzles(cfg)
        a = compute_air_field(plane(cfg), cfg, nozzles)
        cfg.air_model['sources'].reverse()
        b = compute_air_field(plane(cfg), cfg, nozzles)
        np.testing.assert_array_equal(a.air_total_kg_s, b.air_total_kg_s)
        cfg.air_model['interaction_strength'] = 0.
        baseline = compute_air_field(plane(cfg), cfg, nozzles)
        self.assertGreater(np.linalg.norm(a.air_total_kg_s-baseline.air_total_kg_s), 1e-6)
        self.assertAlmostEqual(a.air_total_kg_s.sum(), 1.)


class MixtureTests(unittest.TestCase):
    def test_precomputed_air_requires_same_physical_grid(self):
        cfg = configuration()
        p = plane(cfg)
        air = compute_air_field(p, cfg, default_nozzles(cfg))
        p.theta_centers_deg = (p.theta_centers_deg+5.) % 360.
        with self.assertRaisesRegex(ValueError, 'физическую сетку'):
            compute_mixture(p, cfg, default_nozzles(cfg), air_result=air)

    def test_alpha_masks_zero_fuel_and_uses_ratio_of_integrals(self):
        cfg = configuration(distribution='uniform')
        p = plane(cfg)
        p.cell_mass_kg_s[:] = 0.
        p.cell_mass_kg_s[0, 0] = 1.
        p.cell_mass_kg_s[1, 1] = 3.
        result = compute_mixture(p, cfg, default_nozzles(cfg))
        self.assertAlmostEqual(result.alpha_section_total, 1./8.)
        self.assertTrue(np.isnan(result.alpha_total[0, 1]))
        self.assertGreater(result.undefined_area_fraction, 0.)
        self.assertNotAlmostEqual(np.nanmean(result.alpha_total), result.alpha_section_total)

    def test_unknown_stoichiometry_does_not_block_air(self):
        cfg = configuration(l0=None)
        result = compute_mixture(plane(cfg), cfg, default_nozzles(cfg))
        self.assertAlmostEqual(result.air_total_kg_s.sum(), 1.)
        self.assertTrue(np.isnan(result.alpha_total).all())
        self.assertEqual(result.undefined_area_fraction, 1.)

    def test_suspended_only_and_composite_integral_is_undefined(self):
        cfg = configuration(distribution='uniform')
        p = plane(cfg)
        p.cell_mass_kg_s[:] = 10.
        p.suspended_cell_mass_kg_s = np.ones_like(p.cell_mass_kg_s)
        p.film_inner_cell_mass_kg_s = p.cell_mass_kg_s-p.suspended_cell_mass_kg_s
        a = compute_mixture(p, cfg, default_nozzles(cfg))
        self.assertAlmostEqual(a.alpha_section_total, 1./32.)
        self.assertIn('suspended', a.provenance['fuel_model'])
        p.is_composite = True
        b = compute_mixture(p, cfg, default_nozzles(cfg))
        self.assertTrue(np.isnan(b.alpha_section_total))
        np.testing.assert_array_equal(a.alpha_total, b.alpha_total)

    def test_disabled_model_returns_none_without_mutation(self):
        cfg = configuration(enabled=False)
        raw = deepcopy(cfg.air_model)
        self.assertIsNone(compute_mixture(plane(cfg), cfg, default_nozzles(cfg)))
        self.assertEqual(cfg.air_model, raw)
