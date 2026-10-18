"""Independent analytic geometry checks; assertions follow geometry, not UI output."""
from copy import deepcopy
import json
import math
from pathlib import Path
import subprocess
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from fuelfield.chamber import Chamber, Primitive, group_angles, legacy_geometry
from fuelfield.core import Settings, calculate_plane, default_nozzles, source_flow_kg_s
from fuelfield.quasi import PressureChannel, QuasiSettings, build_influence, calculate_quasi


def line(identity, wall, start, end, **extra):
    return dict(id=identity, wall=wall, kind='line', start=start, end=end, **extra)


def arc(identity, wall, start, end, center, clockwise, **extra):
    return dict(id=identity, wall=wall, kind='arc', start=start, end=end,
                center=center, clockwise=clockwise, **extra)


def chamber_record(primitives=None):
    return dict(schema_version=1, enabled=True, base_z_mm=0., outlet_z_mm=100.,
                chord_error_mm=.01, internal_step_mm=7.,
                primitives=primitives or [
                    line('inner', 'inner', [0., 200.], [100., 200.]),
                    line('outer', 'outer', [0., 300.], [100., 300.])],
                sections=[dict(id='inlet', z_mm=0., is_outlet=False),
                          dict(id='manual', z_mm=21.375, is_outlet=False),
                          dict(id='exit', z_mm=100., is_outlet=True)],
                objects=[], groups=[])


class TestAnalyticPrimitive(unittest.TestCase):
    def test_line_reversal_and_exact_values(self):
        p = Primitive(line('L', 'inner', [10., 200.], [70., 230.]))
        reverse = Primitive(line('L', 'inner', [70., 230.], [10., 200.]))
        z = np.linspace(10., 70., 31)
        np.testing.assert_allclose(p.at(z), 195. + .5*z, rtol=0, atol=1e-12)
        np.testing.assert_allclose(reverse.at(z), p.at(z), rtol=0, atol=1e-12)
        np.testing.assert_array_equal(p.discretize(), [[10., 200.], [70., 230.]])
        self.assertEqual(p.range(20., 40.), (205., 215.))

    def test_quarter_circle_analytic_and_reversed_direction(self):
        p = Primitive(arc('A', 'inner', [0., 250.], [50., 200.], [0., 200.], True))
        reverse = Primitive(arc('B', 'inner', [50., 200.], [0., 250.], [0., 200.], False))
        z = np.linspace(0., 50., 65)
        expected = 200. + np.sqrt(2500. - z*z)
        np.testing.assert_allclose(p.at(z), expected, rtol=0, atol=1e-11)
        np.testing.assert_allclose(reverse.at(z), expected, rtol=0, atol=1e-11)
        self.assertAlmostEqual(abs(p.sweep), math.pi/2)
        np.testing.assert_array_equal(p.discretize()[0], p.start)
        np.testing.assert_array_equal(p.discretize()[-1], p.end)

    def test_upper_and_lower_semicircle_and_extrema(self):
        top = Primitive(arc('A', 'outer', [0., 200.], [100., 200.], [50., 200.], True))
        bottom = Primitive(arc('B', 'inner', [0., 200.], [100., 200.], [50., 200.], False))
        self.assertEqual(top.range(0., 100.), (200., 250.))
        self.assertEqual(bottom.range(0., 100.), (150., 200.))
        self.assertAlmostEqual(float(top.at(50.)), 250.)
        self.assertAlmostEqual(float(bottom.at(50.)), 150.)
        self.assertAlmostEqual(float(top.at(0.)), 200.)
        self.assertAlmostEqual(float(top.at(100.)), 200.)

    def test_arc_chord_sagitta_is_bounded(self):
        for clockwise in (False, True):
            p = Primitive(arc('A', 'inner', [0., 200.], [100., 200.], [50., 200.], clockwise))
            for error in (.2, .01, 1e-4):
                with self.subTest(clockwise=clockwise, error=error):
                    points = p.discretize(error)
                    mids = (points[:-1] + points[1:])/2
                    sagitta = p.radius - np.linalg.norm(mids-p.center, axis=1)
                    self.assertGreater(len(points), 2)
                    self.assertLessEqual(float(sagitta.max()), error+1e-12)
                    np.testing.assert_allclose(np.linalg.norm(points-p.center, axis=1),
                                               p.radius, rtol=0, atol=1e-11)

    def test_normals_are_unit_gas_facing_and_tangent_orthogonal(self):
        for wall, sign in [('inner', 1), ('outer', -1)]:
            p = Primitive(line('L', wall, [0., 200.], [100., 250.]))
            np.testing.assert_allclose(p.normal(40.), sign*np.array([-.5, 1])/math.sqrt(1.25))
            for clockwise in (False, True):
                p = Primitive(arc('A', wall, [0., 200.], [100., 200.], [50., 200.], clockwise))
                for z in (0., 20., 50., 80., 100.):
                    normal = p.normal(z)
                    self.assertAlmostEqual(float(np.linalg.norm(normal)), 1.)
                    radial = np.array([z, float(p.at(z))])-p.center
                    tangent = np.array([-radial[1], radial[0]])
                    self.assertAlmostEqual(float(normal@tangent), 0., places=10)
                    self.assertGreaterEqual(sign*normal[1], -1e-14)

    def test_unit_conversion_and_translation_are_geometric(self):
        raw = arc('A', 'inner', [0., 250.], [50., 200.], [0., 200.], True, radius=50.)
        p = Primitive(raw)
        for unit, factor in [('mm', 1.), ('cm', 10.), ('m', 1000.)]:
            candidate = deepcopy(raw)
            candidate['unit'] = unit
            for key in ('start', 'end', 'center'):
                candidate[key] = (np.asarray(candidate[key])/factor).tolist()
            candidate['radius'] /= factor
            converted = Primitive(candidate)
            np.testing.assert_allclose(converted.at([0., 20., 50.]), p.at([0., 20., 50.]), atol=1e-11)
        moved = deepcopy(raw)
        for key in ('start', 'end', 'center'):
            moved[key] = (np.asarray(moved[key])+[17., 31.]).tolist()
        q = Primitive(moved)
        np.testing.assert_allclose(q.at(np.array([0., 20., 50.])+17.),
                                   p.at([0., 20., 50.])+31., atol=1e-11)

    def test_arc_uniform_scaling(self):
        raw = arc('A', 'inner', [0., 200.], [100., 200.], [50., 200.], True)
        p = Primitive(raw)
        for scale in (.1, 3., 100.):
            candidate = deepcopy(raw)
            for key in ('start', 'end', 'center'):
                candidate[key] = (np.asarray(candidate[key])*scale).tolist()
            q = Primitive(candidate)
            np.testing.assert_allclose(q.at(np.array([0., 7., 50., 100.])*scale),
                                       p.at([0., 7., 50., 100.])*scale, atol=1e-9)

    def test_invalid_arcs_and_out_of_domain_are_rejected(self):
        invalid = [
            arc('A', 'inner', [0., 250.], [50., 200.], [0., 200.], False),  # major arc
            arc('A', 'inner', [0., 250.], [50., 201.], [0., 200.], True),
            arc('A', 'inner', [0., 250.], [50., 200.], [0., 200.], True, radius=49.),
            arc('A', 'inner', [0., 20.], [100., 20.], [50., 20.], False),  # R < 0 inside
            arc('A', 'inner', [0., 250.], [0., 250.], [0., 200.], True),
        ]
        for record in invalid:
            with self.subTest(record=record), self.assertRaises(ValueError):
                Primitive(record)
        p = Primitive(line('L', 'inner', [0., 200.], [100., 200.]))
        for z in (-.01, 100.01, float('nan'), float('inf')):
            with self.subTest(z=z), self.assertRaises(ValueError):
                p.at(z)

    def test_points_and_invalid_dimensions(self):
        p = Primitive(dict(id='P', wall='reference', kind='point', start=[3., 12.]))
        np.testing.assert_array_equal(p.discretize(), [[3., 12.]])
        self.assertEqual(float(p.at(3.)), 12.)
        with self.assertRaises(ValueError):
            p.normal(3.)
        for record in [line('L', 'inner', [0., 1.], [0., 2.]),
                       line('L', 'inner', [0., 1.], [1., -2.]),
                       line('L', 'inner', [False, 1.], [1., 2.]),
                       line('L', 'inner', [0., 1.], [1., 2.], unit='inch')]:
            with self.subTest(record=record), self.assertRaises(ValueError):
                Primitive(record)


class TestChamberGeometry(unittest.TestCase):
    def test_equal_heights_positive_areas_and_annular_area(self):
        raw = chamber_record([line('inner', 'inner', [0., 200.], [100., 220.]),
                              line('outer', 'outer', [0., 300.], [100., 370.])])
        model = Chamber(raw)
        for z in (0., 13.5, 77., 100.):
            rin, rout = model.bounds(z)
            for n in (1, 6, 11, 40):
                centers, edges = model.grid(z, n)
                np.testing.assert_allclose(np.diff(edges), (rout-rin)/n, atol=1e-12)
                np.testing.assert_allclose(centers, (edges[:-1]+edges[1:])/2)
                for count in (8, 140, 359):
                    area = (math.pi/count)*np.diff((edges*1e-3)**2)
                    self.assertTrue(np.all(area > 0))
                    self.assertAlmostEqual(float(area.sum()*count), math.pi*(rout*rout-rin*rin)*1e-6, places=13)

    def test_independent_profile_partitions_and_json_order(self):
        primitives = [line('i1', 'inner', [0., 200.], [20., 210.]),
                      arc('i2', 'inner', [20., 210.], [60., 210.], [40., 210.], True),
                      line('i3', 'inner', [60., 210.], [100., 220.]),
                      line('outer', 'outer', [0., 300.], [100., 330.])]
        raw = chamber_record(primitives)
        expected = Chamber(raw)
        raw['primitives'] = list(reversed(primitives))
        shuffled = Chamber(raw)
        for z in (0., 19.5, 20., 40., 60., 75., 100.):
            np.testing.assert_allclose(expected.bounds(z), shuffled.bounds(z), atol=1e-12)
        self.assertEqual(expected.bounds(40.)[0], 230.)
        self.assertTrue(expected.variable)

    def test_manual_sections_preserved_independently_of_internal_steps(self):
        raw = chamber_record()
        model = Chamber(raw)
        self.assertEqual([s['z_mm'] for s in model.sections], [0., 21.375, 100.])
        self.assertFalse(np.any(np.isclose(model.stations(), 21.375, rtol=0, atol=1e-7)))
        self.assertEqual(model.stations()[0], 0.)
        self.assertEqual(model.stations()[-1], 100.)
        self.assertLessEqual(float(np.diff(model.stations()).max()), 7.+1e-12)
        raw['internal_step_mm'] = 2.3
        changed = Chamber(raw)
        self.assertEqual(changed.sections, model.sections)
        np.testing.assert_allclose(changed.grid(21.375, 6)[0], model.grid(21.375, 6)[0])
        with self.assertRaises(ValueError):
            changed.check_section_list([0., 20., 100.])

    def test_disconnected_overlapping_and_missing_profiles_rejected(self):
        for primitives in [
            [line('i1', 'inner', [0., 200.], [40., 200.]),
             line('i2', 'inner', [41., 200.], [100., 200.])],
            [line('i1', 'inner', [0., 200.], [60., 200.]),
             line('i2', 'inner', [50., 200.], [100., 200.])],
            [line('i1', 'inner', [0., 200.], [50., 200.]),
             line('i2', 'inner', [50., 201.], [100., 201.])],
            [line('inner', 'inner', [1., 200.], [100., 200.])],
        ]:
            with self.subTest(primitives=primitives), self.assertRaises(ValueError):
                Chamber(chamber_record(primitives+[line('outer', 'outer', [0., 300.], [100., 300.])]))

    def test_arc_crossing_between_user_sections_rejected(self):
        raw = chamber_record([
            arc('inner', 'inner', [0., 220.], [100., 220.], [50., 220.], True),
            line('outer', 'outer', [0., 260.], [100., 260.])])
        raw['sections'] = [dict(id='a', z_mm=0.), dict(id='b', z_mm=100., is_outlet=True)]
        with self.assertRaises(ValueError):
            Chamber(raw)

    def test_narrow_off_midpoint_crossing_rejected(self):
        # Peak at 37 mm: checking only endpoints/whole-domain midpoint misses it.
        raw = chamber_record([
            line('i1', 'inner', [0., 200.], [36., 200.]),
            arc('i2', 'inner', [36., 200.], [38., 200.], [37., 200.], True),
            line('i3', 'inner', [38., 200.], [100., 200.]),
            line('outer', 'outer', [0., 200.9999], [100., 200.9999])])
        with self.assertRaises(ValueError):
            Chamber(raw)

    def test_tangent_contact_rejected_but_positive_arc_gap_allowed(self):
        raw = chamber_record([
            arc('inner', 'inner', [0., 200.], [100., 200.], [50., 200.], True),
            line('outer', 'outer', [0., 250.], [100., 250.])])
        with self.assertRaises(ValueError):
            Chamber(raw)
        raw['primitives'][1]['start'][1] += .001
        raw['primitives'][1]['end'][1] += .001
        self.assertAlmostEqual(Chamber(raw).bounds(50.)[1]-250., .001)

    def test_arc_arc_interior_gap_extremum_and_tangency(self):
        # The independent analytic stationary point is z=15, not midpoint 50.
        def record(gap):
            inner_center = np.array([45., 200.])
            outer_r = 100.
            inner_r = 60.
            outer_center = np.array([65., 200.+math.sqrt(2700.)-math.sqrt(7500.)+gap])
            def upper(identity, wall, center, radius):
                start = [0., center[1]+math.sqrt(radius*radius-center[0]*center[0])]
                end = [100., center[1]+math.sqrt(radius*radius-(100.-center[0])**2)]
                return arc(identity, wall, start, end, center.tolist(), True)
            return chamber_record([upper('inner', 'inner', inner_center, inner_r),
                                   upper('outer', 'outer', outer_center, outer_r)])
        for gap in (0., -.0001):
            raw = record(gap)
            inner, outer = map(Primitive, raw['primitives'])
            self.assertGreater(float(outer.at(0.)-inner.at(0.)), 0.)
            self.assertGreater(float(outer.at(50.)-inner.at(50.)), 0.)
            self.assertGreater(float(outer.at(100.)-inner.at(100.)), 0.)
            with self.subTest(gap=gap), self.assertRaises(ValueError):
                Chamber(raw)
        model = Chamber(record(.0001))
        self.assertAlmostEqual(model.bounds(15.)[1]-model.bounds(15.)[0], .0001)

    def test_arc_arc_stationary_roots_all_branch_signs(self):
        # A dense independent circle formula supplies a safe gap margin. A
        # much finer grid than the user sections prevents accepting an arc
        # pair merely because the manual planes happen to miss a constriction.
        z = np.linspace(0., 100., 20001)
        for inner_branch in (-1, 1):
            for outer_branch in (-1, 1):
                inner_shape = inner_branch*np.sqrt(90.**2-(z-40.)**2)
                outer_shape = outer_branch*np.sqrt(80.**2-(z-60.)**2)
                raw_min = float(np.min(outer_shape-inner_shape))
                for desired_gap in (-.01, .01):
                    outer_center_r = 300.-raw_min+desired_gap
                    def circle_record(identity, wall, cx, cr, radius, branch):
                        start = [0., cr+branch*math.sqrt(radius**2-cx**2)]
                        end = [100., cr+branch*math.sqrt(radius**2-(100.-cx)**2)]
                        return arc(identity, wall, start, end, [cx, cr], branch == 1)
                    raw = chamber_record([
                        circle_record('inner', 'inner', 40., 300., 90., inner_branch),
                        circle_record('outer', 'outer', 60., outer_center_r, 80., outer_branch)])
                    with self.subTest(inner=inner_branch, outer=outer_branch, gap=desired_gap):
                        if desired_gap < 0:
                            with self.assertRaises(ValueError):
                                Chamber(raw)
                        else:
                            model = Chamber(raw)
                            self.assertTrue(all(model.bounds(float(t))[1] > model.bounds(float(t))[0]
                                                for t in z[::1000]))

    def test_close_parallel_walls_certify_without_pathological_subdivision(self):
        # Valid thin annulus with constant gap. Bound the child to catch UI-hang
        # algorithms without ever hanging the full regression suite.
        raw = chamber_record([
            line('inner', 'inner', [0., 200.], [100., 300.]),
            line('outer', 'outer', [0., 200.001], [100., 300.001])])
        source = str(Path(__file__).resolve().parents[1] / 'src')
        script = ('import sys,json;sys.path.insert(0,sys.argv[1]);'
                  'from fuelfield.chamber import Chamber;'
                  'm=Chamber(json.loads(sys.argv[2]));'
                  'assert abs(m.bounds(37)[1]-m.bounds(37)[0]-.001)<1e-9')
        try:
            result = subprocess.run([sys.executable, '-B', '-c', script, source, json.dumps(raw)],
                                    text=True, capture_output=True, timeout=5)
        except subprocess.TimeoutExpired:
            self.fail('Certifying two parallel lines exceeded 5 seconds; positive gap is constant.')
        self.assertEqual(result.returncode, 0, result.stdout+result.stderr)

    def test_outlet_and_unique_section_requirements(self):
        for mutate in [
            lambda r: r['sections'].pop(),
            lambda r: r['sections'][0].update(is_outlet=True),
            lambda r: r['sections'][1].update(z_mm=0.),
            lambda r: r['sections'][1].update(z_mm=101.),
            lambda r: r['sections'][1].update(id='inlet'),
        ]:
            raw = chamber_record()
            mutate(raw)
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                Chamber(raw)

    def test_round_trip_preserves_analytic_and_unknown_fields(self):
        raw = chamber_record([
            arc('inner', 'inner', [0., 200.], [100., 200.], [50., 200.], True),
            line('outer', 'outer', [0., 300.], [100., 300.])])
        raw['future_extension'] = dict(unit='custom', values=[1., 2.])
        raw['primitives'][0]['unknown_arc_data'] = dict(source='design')
        original = deepcopy(raw)
        model = Chamber(raw)
        restored = Chamber(json.loads(json.dumps(model.raw)))
        self.assertEqual(raw, original)
        self.assertEqual(restored.raw, original)
        self.assertEqual(restored.primitives[0].kind, 'arc')
        for z in (0., 5.7, 50., 97.1, 100.):
            np.testing.assert_allclose(model.bounds(z), restored.bounds(z), atol=1e-12)

    def test_legacy_geometry_keeps_all_nozzle_identities(self):
        cfg = Settings(z_mm=[0., 12.345, 60.])
        nozzles = default_nozzles(cfg)
        nozzles[11].enabled = False
        nozzles[11].manifold_id = 'special'
        original = deepcopy(nozzles)
        raw = legacy_geometry(cfg, nozzles)
        model = Chamber(raw)
        self.assertFalse(model.enabled)
        self.assertFalse(model.variable)
        self.assertEqual(len(model.objects), 140)
        self.assertEqual({o['nozzle_id'] for o in model.objects}, set(range(1, 141)))
        self.assertEqual(nozzles, original)
        for z in cfg.z_mm:
            np.testing.assert_allclose(model.bounds(z), [cfg.r_in_mm, cfg.r_out_mm])
            expected_centers, expected_edges = cfg.radial_centers_and_edges()
            centers, edges = model.grid(z, cfg.n_radial)
            np.testing.assert_allclose(centers, expected_centers, rtol=0, atol=1e-12)
            np.testing.assert_allclose(edges, expected_edges, rtol=0, atol=1e-12)

    def test_bound_object_outside_its_primitive_is_rejected_early(self):
        raw = chamber_record([
            line('i1', 'inner', [0., 200.], [50., 200.]),
            line('i2', 'inner', [50., 200.], [100., 200.]),
            line('outer', 'outer', [0., 300.], [100., 300.])])
        raw['objects'] = [dict(id='hole', type='hole', wall='inner', primitive_id='i1',
                               z_mm=75., radius_mm=200., theta_deg=0.,
                               parameters={}, units={}, provenance={})]
        with self.assertRaises(ValueError):
            Chamber(raw)


class TestAngularGroupGeometry(unittest.TestCase):
    def test_uniform_orientation_phase_and_row_offset(self):
        base = dict(members=['a', 'b', 'c', 'd'], mode='uniform',
                    direction=-1, phase_deg=30., row_offset_deg=15.)
        np.testing.assert_allclose(group_angles(base), [45., 315., 225., 135.])
        base['direction'] = 1
        np.testing.assert_allclose(group_angles(base), [45., 135., 225., 315.])

    def test_nonuniform_closed_steps_and_periodicity(self):
        group = dict(members=['a', 'b', 'c', 'd'], mode='steps',
                     steps_deg=[20., 60., 100., 180.], phase_deg=7., direction=1)
        expected = [7., 27., 87., 187.]
        np.testing.assert_allclose(group_angles(group), expected)
        group['phase_deg'] += 720.
        np.testing.assert_allclose(group_angles(group), expected)
        group['direction'] = -1
        np.testing.assert_allclose(group_angles(group), [7., 347., 287., 187.])

    def test_individual_angles_normalize_and_duplicates_reject(self):
        group = dict(members=['a', 'b', 'c'], mode='individual',
                     angles_deg=[-10., 30., 400.], direction=1)
        np.testing.assert_allclose(group_angles(group), [350., 30., 40.])
        group['angles_deg'] = [0., 360., 10.]
        with self.assertRaises(ValueError):
            group_angles(group)

    def test_invalid_step_counts_sums_signs_and_direction(self):
        base = dict(members=['a', 'b', 'c'], mode='steps', direction=1)
        for steps in ([180., 180.], [10., 20., 30.], [180., 180., 0.],
                      [200., 170., -10.], [120., 120., float('nan')]):
            with self.subTest(steps=steps), self.assertRaises(ValueError):
                group_angles(dict(base, steps_deg=steps))
        with self.assertRaises(ValueError):
            group_angles(dict(base, mode='uniform', direction=True))


class TestGeometryFuelAccounting(unittest.TestCase):
    def case(self, variable=True):
        cfg = Settings(z_mm=[0., 100.], n_sample_radial=12, n_sample_angular=108,
                       n_radial=6, n_angular=8)
        nozzles = default_nozzles(cfg)
        for n in nozzles:
            n.enabled = n.id in (1, 38, 107)
            n.flow_kg_h = 12.+n.id/10
        raw = legacy_geometry(cfg, nozzles)
        raw['enabled'] = True
        if variable:
            raw['primitives'][0]['end'][1] = 235.
            raw['primitives'][1]['end'][1] = 265.
        cfg.geometry = raw
        return cfg, nozzles

    def test_constant_enabled_profiles_reproduce_legacy_with_custom_centers(self):
        cfg, nozzles = self.case(variable=False)
        cfg.radial_centers_mm = [204., 214., 250., 280., 294., 298.]
        baseline = deepcopy(cfg)
        baseline.geometry = {}
        for z in cfg.z_mm:
            actual = calculate_plane(nozzles, cfg, z)
            expected = calculate_plane(nozzles, baseline, z)
            np.testing.assert_array_equal(actual.cell_mass_kg_s, expected.cell_mass_kg_s)
            np.testing.assert_array_equal(actual.cell_area_m2, expected.cell_area_m2)
            np.testing.assert_array_equal(actual.flux_kg_m2_s, expected.flux_kg_m2_s)
            self.assertEqual(actual.injected_kg_s, expected.injected_kg_s)
            self.assertEqual(actual.inner_kg_s, expected.inner_kg_s)
            self.assertEqual(actual.outer_kg_s, expected.outer_kg_s)
            self.assertEqual(actual.wall_bound_kg_s, 0.)

    def test_variable_profiles_do_not_renormalize_clipped_mass(self):
        cfg, nozzles = self.case()
        for z in cfg.z_mm:
            actual = calculate_plane(nozzles, cfg, z)
            rin, rout = cfg.bounds_at(z)
            baseline = deepcopy(cfg)
            baseline.geometry = {}
            baseline.r_in_mm, baseline.r_out_mm = rin, rout
            expected = calculate_plane(nozzles, baseline, z)
            np.testing.assert_array_equal(actual.cell_mass_kg_s, expected.cell_mass_kg_s)
            # linspace equal-height edges and legacy midpoint edges differ by
            # machine rounding; bit identity is required only in the separate
            # constant-profile compatibility test above.
            np.testing.assert_allclose(actual.cell_area_m2, expected.cell_area_m2,
                                       rtol=1e-13, atol=1e-16)
            self.assertEqual(actual.inside_kg_s, expected.inside_kg_s)
            self.assertEqual(actual.wall_inner_kg_s, expected.inner_kg_s)
            self.assertEqual(actual.wall_outer_kg_s, expected.outer_kg_s)
            self.assertEqual(actual.wall_bound_kg_s, expected.inner_kg_s+expected.outer_kg_s)
            self.assertEqual(actual.inner_kg_s, 0.)
            self.assertEqual(actual.outer_kg_s, 0.)
            self.assertLess(abs(actual.rel_balance_error), 1e-13)
            self.assertTrue(actual.geometry_notice)
        self.assertGreater(actual.wall_bound_kg_s, 0.)

    def test_unit_influences_match_stationary_accounting_and_scale_linearly(self):
        cfg, nozzles = self.case()
        flows = np.array([source_flow_kg_s(n, cfg) if n.enabled else 0. for n in nozzles])
        op = build_influence(nozzles, cfg, 100., np.stack([flows, 2*flows]), preview=False)
        actual = op.plane(flows)
        expected = calculate_plane(nozzles, cfg, 100.)
        np.testing.assert_allclose(actual.cell_mass_kg_s, expected.cell_mass_kg_s, atol=1e-17, rtol=1e-13)
        self.assertAlmostEqual(actual.wall_bound_kg_s, expected.wall_bound_kg_s, places=15)
        weights = op.weights.sum(axis=0)+op.inner+op.outer+op.wall_inner+op.wall_outer
        np.testing.assert_allclose(weights, op.reached, atol=1e-13, rtol=0)
        scaled = op.plane(2*flows)
        np.testing.assert_allclose(scaled.cell_mass_kg_s, 2*actual.cell_mass_kg_s, atol=1e-17, rtol=1e-13)
        self.assertAlmostEqual(scaled.wall_bound_kg_s, 2*actual.wall_bound_kg_s, places=15)
        self.assertLess(abs(scaled.rel_balance_error), 1e-13)

    def test_constant_pressure_composite_equals_stationary_and_snapshot_balances(self):
        cfg, nozzles = self.case()
        cfg.quasistationary = True
        cfg.quasi_settings = QuasiSettings(n_steps=8, channels=(PressureChannel('A', (30.,)*8),)).to_dict()
        result = calculate_quasi(nozzles, cfg, preview=False)
        for i, z in enumerate(cfg.z_mm):
            reference = calculate_plane(nozzles, cfg, z)
            np.testing.assert_allclose(result.composites[i].cell_mass_kg_s,
                                       reference.cell_mass_kg_s, atol=1e-17, rtol=1e-13)
            self.assertTrue(math.isnan(result.composites[i].rel_balance_error))
            for step in range(8):
                actual = result.snapshot(i, step)
                self.assertLess(abs(actual.rel_balance_error), 1e-13)
                self.assertAlmostEqual(actual.wall_bound_kg_s, reference.wall_bound_kg_s, places=15)
        for balance in result.balances:
            self.assertAlmostEqual(balance['inside_kg_s']+balance['inner_kg_s']+
                                   balance['outer_kg_s']+balance['wall_bound_kg_s'],
                                   balance['injected_kg_s'], places=14)
        self.assertIsNone(result.manifest()['composite_global_balance'])


if __name__ == '__main__':
    unittest.main()
