"""Independent v0.8 acceptance oracles, authored separately from the solver.

The Didenko test uses the book's conditional ledger on explicitly synthetic
axial stations. Those stations are not measured dimensions of the book case.
"""
from copy import deepcopy
from types import SimpleNamespace
import math
import unittest

import numpy as np

from fuelfield.air_sources import default_air_model, normalize_source, resolve_sources, validate_air_model
from fuelfield.air_balance import solve_balance
from fuelfield.air_jets import trace_jets, section_crossing
from fuelfield.air_distribution import AirFieldResult, compute_air_field
from fuelfield.mixture_field import compute_mixture


def model(*sources, **changes):
    raw = default_air_model()
    raw.update(enabled=True, total_kg_s=1., l0=2., max_path_mm=30., step_mm=.5,
               sources=list(sources))
    raw.update(changes)
    return raw


def source(identity, **changes):
    raw = dict(id=identity, kind='hole', method='fixed', flow_kg_s=1.,
               z_mm=0., radius_mm=150., theta_deg=0., direction=[1., 0., 0.],
               provenance={'test': 'synthetic independent analytic oracle'})
    raw.update(changes)
    return raw


def config(raw):
    return SimpleNamespace(air_model=raw, geometry={}, r_in_mm=100., r_out_mm=200.,
                           bounds_at=lambda z: (100., 200.))


def plane(z=10., theta_offset=0.):
    edges = np.array([100., 120., 160., 200.])
    area = np.repeat((np.pi/12*np.diff((edges*.001)**2))[:, None], 12, axis=1)
    return SimpleNamespace(z_mm=z, cell_mass_kg_s=area.copy(), cell_area_m2=area,
                           radial_edges_mm=edges, theta_centers_deg=(theta_offset+np.arange(12)*30.) % 360,
                           suspended_cell_mass_kg_s=None, is_composite=False)


class TestIndependentAirLedger(unittest.TestCase):
    def test_fixed_share_and_area_reserve_distinct_parts(self):
        raw = model(source('known', flow_kg_s=2.), source('share', method='fraction', share=.1),
                    source('small', method='area', area_mm2=1.),
                    source('large', method='area', area_mm2=3.),
                    total_kg_s=10., area_hypothesis_confirmed=True)
        before = deepcopy(raw)
        out = solve_balance(raw)
        self.assertEqual(raw, before)
        self.assertEqual(out.flows, dict(known=2., share=1., small=1.75, large=5.25))
        self.assertEqual(out.unassigned_kg_s, 0.)
        self.assertAlmostEqual(sum(out.categories.values()), 10., places=13)

    def test_incomplete_registry_preserves_unknown_remainder(self):
        out = solve_balance(model(source('known', flow_kg_s=2.),
                                  source('unknown', method='unknown'), total_kg_s=10.))
        self.assertEqual(out.flows, {'known': 2., 'unknown': 0.})
        self.assertEqual(out.unassigned_kg_s, 8.)
        partial = solve_balance(model(source('a', method='area', area_mm2=20.), total_kg_s=10.,
                                      area_hypothesis_confirmed=True, area_allocation_fraction=.3))
        self.assertEqual(partial.flows['a'], 3.)
        self.assertEqual(partial.unassigned_kg_s, 7.)

    def test_budget_and_hydraulic_assumptions_are_not_silently_repaired(self):
        for raw in (model(source('too_much', flow_kg_s=1.1)),
                    model(source('area', method='area', area_mm2=1.)),
                    model(source('cd_missing', method='effective_area', area_mm2=1.), area_hypothesis_confirmed=True),
                    model(source('negative', flow_kg_s=-1.)),
                    model(source('nan', flow_kg_s=float('nan'))),
                    model(source('a'), source('a'))):
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                solve_balance(raw)

    def test_effective_area_ratios_and_normalized_finite_direction(self):
        out = solve_balance(model(source('a', method='effective_area', area_mm2=2., cd=.5),
                                  source('b', method='effective_area', area_mm2=2., cd=1.),
                                  total_kg_s=3., area_hypothesis_confirmed=True))
        self.assertEqual(out.flows, {'a': 1., 'b': 2.})
        direction = normalize_source(source('large_vector', direction=[1e200, 1e200, 0.]))['direction']
        self.assertTrue(np.isfinite(direction).all())
        self.assertAlmostEqual(np.linalg.norm(direction), 1., places=13)


class TestIndependentDirectedAir(unittest.TestCase):
    def trace(self, raw):
        cfg = config(raw)
        validated = validate_air_model(raw)
        return trace_jets(resolve_sources(raw, cfg, []), solve_balance(raw).flows, cfg, validated)

    def test_isolated_ray_is_exact_cartesian_line(self):
        raw = model(source('ray', direction=[1., 0., 1.]))
        paths = self.trace(raw)
        expected = np.array([0., 150., 0.])[None, :] + paths.path_mm[:, None]*np.array([1., 0., 1.])/math.sqrt(2.)
        np.testing.assert_allclose(paths.positions[:, 0], expected, rtol=0, atol=2e-12)
        np.testing.assert_allclose(paths.directions[:, 0], np.broadcast_to([1/math.sqrt(2.), 0., 1/math.sqrt(2.)], paths.directions[:, 0].shape), atol=1e-14)

    def test_rotation_and_registry_permutation_do_not_change_solution(self):
        raw = model(source('a', radius_mm=130., direction=[1., .5, .2]),
                    source('b', radius_mm=170., direction=[1., -.5, -.2]),
                    total_kg_s=2., interaction_strength=2.)
        original = self.trace(raw)
        reversed_raw = deepcopy(raw)
        reversed_raw['sources'].reverse()
        np.testing.assert_array_equal(self.trace(reversed_raw).positions, original.positions)
        angle = math.radians(37.)
        transform = np.array([[1., 0., 0.], [0., math.cos(angle), -math.sin(angle)],
                              [0., math.sin(angle), math.cos(angle)]])
        rotated = deepcopy(raw)
        for item in rotated['sources']:
            item['theta_deg'] += 37.
        actual = self.trace(rotated)
        np.testing.assert_allclose(actual.positions, original.positions@transform.T, rtol=0, atol=3e-11)
        np.testing.assert_allclose(actual.directions, original.directions@transform.T, rtol=0, atol=3e-13)

    def test_reverse_and_radial_are_not_falsely_counted_as_forward(self):
        for direction, expected_status in (([-1., 0., 0.], 'reverse'), ([0., 1., 0.], 'not_crossed')):
            raw = model(source('ray', z_mm=10., direction=direction))
            cfg = config(raw)
            result = compute_air_field(plane(5.), cfg, [])
            self.assertEqual(result.source_states[0]['status'], expected_status)
            self.assertEqual(result.air_total_kg_s.sum(), 0.)
            if expected_status == 'reverse':
                self.assertAlmostEqual(result.reverse_air_kg_s.sum(), 1.)
                mixture = compute_mixture(plane(5.), cfg, [], air_result=result)
                self.assertTrue(np.isnan(mixture.alpha_total).all())
            else:
                self.assertEqual(result.source_states[0]['not_crossed_kg_s'], 1.)

    def test_zero_penetration_keeps_ledger_and_isolated_path_finite(self):
        raw = model(source('ray', penetration_factor=0., direction=[1., .1, 0.]), interaction_strength=3.)
        paths = self.trace(raw)
        self.assertTrue(np.isfinite(paths.positions).all())
        result = compute_air_field(plane(), config(raw), [], paths=paths)
        self.assertAlmostEqual(result.air_total_kg_s.sum(), 1., places=13)

    def test_rays_and_sections_are_independent_of_observation_density(self):
        raw = model(source('ray', direction=[1., .2, .1], mixing_length_mm=10.))
        cfg = config(raw)
        sparse = [compute_air_field(plane(z), cfg, []) for z in (5., 20.)]
        dense = {z: compute_air_field(plane(z), cfg, []) for z in (1., 5., 7., 12., 20., 21.)}
        for z, expected in zip((5., 20.), sparse):
            np.testing.assert_array_equal(dense[z].air_total_kg_s, expected.air_total_kg_s)
            np.testing.assert_array_equal(dense[z].air_mixed_kg_s, expected.air_mixed_kg_s)

    def test_uniform_is_area_weighted_and_wall_escape_retains_flow(self):
        raw = model(source('uniform'), distribution='uniform')
        p = plane()
        result = compute_air_field(p, config(raw), [])
        np.testing.assert_allclose(result.air_total_kg_s, p.cell_area_m2/p.cell_area_m2.sum(), rtol=1e-14)
        escape = model(source('out', radius_mm=199., direction=[.2, 1., 0.]))
        out = compute_air_field(p, config(escape), [])
        self.assertEqual(out.source_states[0]['status'], 'wall_exit')
        self.assertEqual(out.source_states[0]['outside_kg_s'], 1.)
        self.assertEqual(out.air_total_kg_s.sum(), 0.)

    def test_coarse_step_cannot_teleport_through_a_profile_throat(self):
        # The exact straight ray r=150 hits the first outer line at z=10/3.
        # Its numerical step ends inside again, so endpoint-only tests fail.
        from fuelfield.chamber import Chamber
        geometry = dict(schema_version=1, enabled=True, base_z_mm=0., outlet_z_mm=10.,
                        primitives=[dict(id='inner', kind='line', wall='inner', unit='mm',
                                         start=[0., 100.], end=[10., 100.])],
                        sections=[dict(id='out', z_mm=10., is_outlet=True)], objects=[], groups=[])
        points = [[0., 200.], [4., 140.], [6., 140.], [10., 200.]]
        geometry['primitives'].extend(dict(id=f'outer{i}', kind='line', wall='outer',
                                          unit='mm', start=a, end=b)
                                      for i, (a, b) in enumerate(zip(points, points[1:])))
        raw = model(source('ray'), step_mm=10., max_path_mm=20.)
        cfg = config(raw)
        cfg.geometry = geometry
        cfg.bounds_at = Chamber(geometry).bounds
        result = compute_air_field(plane(10.), cfg, [])
        self.assertEqual(result.air_total_kg_s.sum(), 0.)
        self.assertEqual(result.source_states[0]['outside_kg_s'], 1.)
        self.assertEqual(result.source_states[0]['status'], 'wall_exit')


class TestIndependentMixtureOracles(unittest.TestCase):
    def supplied_air(self, values, mixed=None):
        total = np.array([values], dtype=float)
        part = total*.5 if mixed is None else np.array([mixed], dtype=float)
        return AirFieldResult(10., total, total-part, part, np.zeros_like(total),
                              np.zeros_like(total, dtype=bool), solve_balance(model()), [], (), {})

    def test_integral_alpha_is_ratio_of_sums_and_zero_cells_are_masked(self):
        p = SimpleNamespace(z_mm=10., cell_mass_kg_s=np.array([[1., 3., 0., 2.]]),
                            cell_area_m2=np.array([[1., 1., 5., 1.]]),
                            suspended_cell_mass_kg_s=None, is_composite=False)
        out = compute_mixture(p, config(model()), [], air_result=self.supplied_air([2., 3., 4., 0.]))
        np.testing.assert_allclose(out.alpha_total[0, [0, 1, 3]], [1., .5, 0.])
        self.assertTrue(np.isnan(out.alpha_total[0, 2]))
        self.assertEqual(out.undefined_area_fraction, 5/8)
        self.assertAlmostEqual(out.alpha_section_total, 9/12)
        p.cell_mass_kg_s = np.array([[1., 3.]])
        p.cell_area_m2 = np.ones((1, 2))
        out = compute_mixture(p, config(model()), [], air_result=self.supplied_air([2., 3.]))
        self.assertEqual(out.alpha_section_total, .625)
        self.assertNotEqual(out.alpha_section_total, float(np.mean(out.alpha_total)))

    def test_film_excluded_unknown_l0_and_composite_have_honest_masks(self):
        p = SimpleNamespace(z_mm=10., cell_mass_kg_s=np.array([[9., 9.]]),
                            suspended_cell_mass_kg_s=np.array([[1., 3.]]),
                            film_inner_cell_mass_kg_s=np.array([[8., 6.]]),
                            cell_area_m2=np.ones((1, 2)), is_composite=False)
        out = compute_mixture(p, config(model()), [], air_result=self.supplied_air([2., 3.]))
        self.assertEqual(out.alpha_section_total, .625)
        self.assertEqual(out.provenance['film_inner_kg_s'], 14.)
        unknown = compute_mixture(p, config(model(l0=None)), [], air_result=self.supplied_air([2., 3.]))
        self.assertTrue(np.isnan(unknown.alpha_total).all())
        self.assertEqual(unknown.undefined_area_fraction, 1.)
        p.is_composite = True
        composite = compute_mixture(p, config(model()), [], air_result=self.supplied_air([2., 3.]))
        self.assertTrue(np.isnan(composite.alpha_section_total))
        np.testing.assert_allclose(composite.alpha_total, [[1., .5]])

    def test_precomputed_air_cannot_be_combined_with_another_z(self):
        p = SimpleNamespace(z_mm=11., cell_mass_kg_s=np.array([[1., 3.]]),
                            cell_area_m2=np.ones((1, 2)), suspended_cell_mass_kg_s=None,
                            is_composite=False)
        with self.assertRaises(ValueError):
            compute_mixture(p, config(model()), [], air_result=self.supplied_air([2., 3.]))

    def test_didenko_conditional_ledger_survives_1d_to_2d_distribution(self):
        # Printed pp. 8–13: the independently transcribed book oracle. Physical
        # z=1..11 here is a SYNTHETIC mapping of the book's segment labels.
        targets = [.25, .29692, .35384, .45460, .53844, .63920,
                   .72304, .82380, .89228, .97, 1.]
        records = [source('front', flow_kg_s=.25, mixed_fraction=1.)]
        for identity, flow, first, portions in (
                ('active3', .1346, 2, [.2, .2, .6]),
                ('active5', .1346, 5, [.4, .6]),
                ('active7', .1346, 7, [.4, .6]),
                ('mixing9', .0962, 9, [.4, .6]),
                ('cooling2', .05, 2, [.4, .6]),
                ('cooling4', .05, 4, [.4, .6]),
                ('cooling6', .05, 6, [.4, .6]),
                ('cooling8', .05, 8, [.4, .6]),
                ('cooling10', .05, 10, [.4, .6])):
            events = [[float(first-1), 0.]]
            cumulative = 0.
            for i, part in enumerate(portions):
                cumulative += part
                events.append([float(first+i), cumulative])
            records.append(source(identity, flow_kg_s=flow, entrainment_points=events))
        raw = model(*records, distribution='directed', l0=1.)
        before = deepcopy(raw)
        cfg = config(raw)
        paths = trace_jets(resolve_sources(raw, cfg, []), solve_balance(raw).flows, cfg, validate_air_model(raw))
        for stage, expected in enumerate(targets, 1):
            p = plane(float(stage))
            # Fixed total fuel gives global alpha_KS=2.6, as in printed pp19–20.
            p.cell_mass_kg_s = p.cell_area_m2/p.cell_area_m2.sum()/2.6
            air = compute_air_field(p, cfg, [], paths=paths)
            out = compute_mixture(p, cfg, [], air_result=air)
            self.assertAlmostEqual(out.air_total_kg_s.sum(), 1., places=13)
            self.assertAlmostEqual(out.air_mixed_kg_s.sum(), expected, places=13)
            self.assertAlmostEqual(out.alpha_section_mixed, expected*2.6, places=12)
            np.testing.assert_allclose(out.air_core_kg_s+out.air_mixed_kg_s, out.air_total_kg_s, atol=1e-16)
        self.assertEqual(raw, before)
