"""Independent geometric conservation guards for the v0.7 transport contacts."""
import math
import unittest

import numpy as np

from fuelfield.chamber import Chamber
from fuelfield.transport_contacts import (first_contacts, wall_radius,
                                         film_velocity, wall_arc_length)


def line(identity, wall, start, end, **extra):
    return dict(id=identity, wall=wall, kind='line', start=start, end=end, **extra)


def arc(identity, wall, start, end, center, clockwise, **extra):
    return dict(id=identity, wall=wall, kind='arc', start=start, end=end,
                center=center, clockwise=clockwise, **extra)


def chamber(primitives=None, base=0., outlet=100.):
    return Chamber(dict(schema_version=1, enabled=True, base_z_mm=base,
        outlet_z_mm=outlet, primitives=primitives or [
            line('inner', 'inner', [base, 200.], [outlet, 200.]),
            line('outer', 'outer', [base, 300.], [outlet, 300.])],
        sections=[dict(id='exit', z_mm=outlet, is_outlet=True)],
        objects=[], groups=[]))


class ExactContacts(unittest.TestCase):
    def test_cylinders_have_independent_exact_cartesian_intersections(self):
        model = chamber()
        start = np.array([[250., 0.], [250., 0.], [250., 0.], [250., 0.]])
        end = np.array([[350., 0.], [150., 0.], [250., 10.], [-250., 0.]])
        fractions, wall = first_contacts(model, 0., 100., start, end)
        np.testing.assert_allclose(fractions[[0, 1, 3]], [.5, .5, .1], atol=2e-14)
        self.assertTrue(np.isinf(fractions[2]))
        np.testing.assert_array_equal(wall, [2, 1, 0, 1])

    def test_cylinder_quadratic_respects_azimuth_and_nonradial_motion(self):
        model = chamber()
        # |(250, 100*s)| = 300 has its first root at sqrt(27500)/100.
        expected = math.sqrt(300.**2-250.**2)/200.
        angles = np.linspace(0., 2*math.pi, 37)
        start = 250*np.column_stack((np.cos(angles), np.sin(angles)))
        end = start+200*np.column_stack((-np.sin(angles), np.cos(angles)))
        fractions, walls = first_contacts(model, 3., 91., start, end)
        np.testing.assert_allclose(fractions, expected, atol=2e-14)
        np.testing.assert_array_equal(walls, 2)

    def test_piecewise_narrow_waist_is_detected_with_both_endpoints_inside(self):
        model = chamber([
            line('I1', 'inner', [0., 200.], [50., 270.]),
            line('I2', 'inner', [50., 270.], [100., 200.]),
            line('O', 'outer', [0., 300.], [100., 300.])])
        f, w = first_contacts(model, 0., 100., [[250., 0.]], [[250., 0.]])
        self.assertAlmostEqual(f[0], 5/14, places=13)
        self.assertEqual(w[0], 1)

    def test_circular_narrow_waists_do_not_depend_on_endpoint_classification(self):
        inner = chamber([
            arc('I', 'inner', [0., 200.], [100., 200.], [50., 200.], True),
            line('O', 'outer', [0., 300.], [100., 300.])])
        outer = chamber([
            line('I', 'inner', [0., 200.], [100., 200.]),
            arc('O', 'outer', [0., 300.], [100., 300.], [50., 300.], False)])
        for model, radius, identity in ((inner, 230., 1), (outer, 270., 2)):
            with self.subTest(identity=identity):
                f, w = first_contacts(model, 0., 100., [[radius, 0.]], [[radius, 0.]])
                np.testing.assert_allclose(f, [.1], atol=2e-12)
                np.testing.assert_array_equal(w, [identity])

    def test_squaring_does_not_accept_other_circle_branch(self):
        model = chamber([
            arc('I', 'inner', [0., 200.], [100., 200.], [50., 200.], False),
            line('O', 'outer', [0., 300.], [100., 300.])])
        f, w = first_contacts(model, 0., 100., [[230., 0.]], [[230., 0.]])
        self.assertTrue(np.isinf(f[0]))
        self.assertEqual(w[0], 0)

    def test_start_wall_enters_leaves_stays_and_tangent_curvature(self):
        model = chamber()
        start = [[200., 0.], [200., 0.], [300., 0.], [300., 0.],
                 [200., 0.], [300., 0.], [200., 0.]]
        end = [[220., 0.], [190., 0.], [280., 0.], [310., 0.],
               [200., 20.], [300., 20.], [200., 0.]]
        f, w = first_contacts(model, 0., 100., start, end)
        self.assertTrue(np.all(np.isinf(f[[0, 2, 4]])))
        np.testing.assert_array_equal(f[[1, 3, 5, 6]], 0.)
        np.testing.assert_array_equal(w, [0, 1, 0, 2, 0, 2, 1])

    def test_skipped_start_root_does_not_hide_later_contact(self):
        f, w = first_contacts(chamber(), 0., 100., [[200., 0.]], [[400., 0.]])
        np.testing.assert_allclose(f, [.5], atol=1e-14)
        np.testing.assert_array_equal(w, [2])

    def test_positive_time_tangent_is_a_contact(self):
        # A Cartesian chord touches the inner cylinder without crossing it.
        f, w = first_contacts(chamber(), 0., 100., [[200., -100.]], [[200., 100.]])
        np.testing.assert_allclose(f, [.5], atol=1e-14)
        np.testing.assert_array_equal(w, [1])
        # Analytic arc reaches this same constant-r ray at just its maximum.
        model = chamber([
            arc('I', 'inner', [0., 200.], [100., 200.], [50., 200.], True),
            line('O', 'outer', [0., 300.], [100., 300.])])
        f, w = first_contacts(model, 0., 100., [[250., 0.]], [[250., 0.]])
        np.testing.assert_allclose(f, [.5], atol=1e-12)
        np.testing.assert_array_equal(w, [1])

    def test_endpoint_contact_and_joint_are_included(self):
        model = chamber([
            line('I1', 'inner', [0., 200.], [50., 250.]),
            line('I2', 'inner', [50., 250.], [100., 200.]),
            line('O', 'outer', [0., 300.], [100., 300.])])
        f, w = first_contacts(model, 0., 50., [[250., 0.]], [[250., 0.]])
        np.testing.assert_array_equal(f, [1.])
        np.testing.assert_array_equal(w, [1])
        # From that joint, the downstream wall recedes and the ray enters fluid.
        f, w = first_contacts(model, 50., 100., [[250., 0.]], [[250., 0.]])
        self.assertTrue(np.isinf(f[0]))
        self.assertEqual(w[0], 0)

    def test_constant_z_step_is_supported(self):
        f, w = first_contacts(chamber(), 30., 30., [[250., 0.]], [[350., 0.]])
        np.testing.assert_allclose(f, [.5], atol=1e-14)
        np.testing.assert_array_equal(w, [2])

    def test_reversed_primitive_and_physical_units_preserve_contact(self):
        for unit, scale in (('mm', 1.), ('cm', 10.), ('m', 1000.)):
            model = chamber([
                arc('I', 'inner', [100./scale, 200./scale], [0., 200./scale],
                    [50./scale, 200./scale], False, unit=unit),
                line('O', 'outer', [100./scale, 300./scale], [0., 300./scale], unit=unit)])
            f, w = first_contacts(model, 0., 100., [[230., 0.]], [[230., 0.]])
            np.testing.assert_allclose(f, [.1], atol=2e-12)
            np.testing.assert_array_equal(w, [1])

    def test_arc_contact_accuracy_with_nonradial_cartesian_step(self):
        model = chamber([
            line('I', 'inner', [0., 100.], [100., 100.]),
            arc('O', 'outer', [0., 300.], [100., 300.], [50., 300.], False)])
        # Construct a known contact at z=37.125 on the actual lower branch.
        target = .37125
        radius = 300.-math.sqrt(2500.-(100.*target-50.)**2)
        x0, dx = 240., 30.
        dy = math.sqrt(radius**2-(x0+target*dx)**2)/target
        f, w = first_contacts(model, 0., 100., [[x0, 0.]], [[x0+dx, dy]])
        np.testing.assert_allclose(f, [target], atol=2e-12)
        np.testing.assert_array_equal(w, [2])

    def test_small_arc_at_large_chamber_radius_is_well_conditioned(self):
        radial_offset = 1e8
        axial_offset = 1e6
        model = chamber([
            arc('I', 'inner', [axial_offset, radial_offset],
                [axial_offset+100., radial_offset],
                [axial_offset+50., radial_offset], True),
            line('O', 'outer', [axial_offset, radial_offset+100.],
                 [axial_offset+100., radial_offset+100.])],
            base=axial_offset, outlet=axial_offset+100.)
        f, w = first_contacts(model, axial_offset, axial_offset+100.,
                              [[radial_offset+30., 0.]], [[radial_offset+30., 0.]])
        np.testing.assert_allclose(f, [.1], atol=2e-12)
        np.testing.assert_array_equal(w, [1])

    def test_zero_radial_circle_center_repeated_quartic_root(self):
        endpoint = math.sqrt(100.**2-50.**2)
        model = chamber([
            arc('I', 'inner', [0., endpoint], [100., endpoint], [50., 0.], True),
            line('O', 'outer', [0., 200.], [100., 200.])])
        f, w = first_contacts(model, 0., 100., [[100., 0.]], [[100., 0.]])
        np.testing.assert_allclose(f, [.5], atol=2e-12)
        np.testing.assert_array_equal(w, [1])

    def test_random_arc_steps_match_independent_bracket_reference(self):
        # A broad step crosses the complete two-circle waist.  The reference
        # brackets physical signed clearance, independently of quartic algebra.
        from scipy.optimize import brentq
        model = chamber([
            arc('I', 'inner', [0., 200.], [100., 200.], [50., 200.], True),
            arc('O', 'outer', [0., 400.], [100., 400.], [50., 400.], False)])
        rng = np.random.default_rng(72163)
        radius, theta = rng.uniform(200.1, 399.9, 60), rng.uniform(-math.pi, math.pi, 60)
        start = radius[:, None]*np.column_stack((np.cos(theta), np.sin(theta)))
        end = start+rng.uniform(-200., 200., (60, 2))
        f, w = first_contacts(model, 0., 100., start, end)
        sample = np.linspace(0., 1., 1001)
        inner = 200.+np.sqrt(np.maximum(0., 2500.-(100.*sample-50.)**2))
        outer = 400.-np.sqrt(np.maximum(0., 2500.-(100.*sample-50.)**2))
        for index in range(len(start)):
            radii = np.linalg.norm(start[index]+sample[:, None]*(end[index]-start[index]), axis=1)
            roots = []
            for identity, clearance in ((1, radii-inner), (2, outer-radii)):
                outside = np.flatnonzero(clearance[1:] < 0)
                if not len(outside):
                    continue
                left = outside[0]
                def signed(value):
                    r = np.linalg.norm(start[index]+value*(end[index]-start[index]))
                    circle = math.sqrt(max(0., 2500.-(100.*value-50.)**2))
                    return r-(200.+circle) if identity == 1 else 400.-circle-r
                roots.append((brentq(signed, sample[left], sample[left+1], xtol=1e-14), identity))
            if roots:
                expected, identity = min(roots)
                self.assertEqual(w[index], identity)
                self.assertAlmostEqual(f[index], expected, places=12)
            else:
                self.assertTrue(np.isinf(f[index]))
                self.assertEqual(w[index], 0)

    def test_empty_and_bad_inputs_are_explicit(self):
        model = chamber()
        f, w = first_contacts(model, 0., 100., np.empty((0, 2)), np.empty((0, 2)))
        self.assertEqual(f.shape, (0,))
        self.assertEqual(w.shape, (0,))
        for z0, z1, start, end in [
                (100., 0., [[250., 0.]], [[250., 0.]]),
                (-1., 100., [[250., 0.]], [[250., 0.]]),
                (0., 100., [[190., 0.]], [[250., 0.]]),
                (0., 100., [[310., 0.]], [[250., 0.]]),
                (0., 100., [[250., float('nan')]], [[250., 0.]]),
                (0., 100., [250., 0.], [[250., 0.]])]:
            with self.subTest(start=start), self.assertRaises(ValueError):
                first_contacts(model, z0, z1, start, end)


class ExactWallKinematics(unittest.TestCase):
    def test_wall_radius_broadcast_and_invalid_identity(self):
        model = chamber([
            line('I', 'inner', [0., 200.], [100., 250.]),
            line('O', 'outer', [0., 300.], [100., 350.])])
        np.testing.assert_allclose(wall_radius(model, [[1], [2]], [0., 50., 100.]),
                                   [[200., 225., 250.], [300., 325., 350.]])
        self.assertEqual(float(wall_radius(model, 1, 50.)), 225.)
        for identity, z in ((0, 50.), (1.5, 50.), (3, 50.), (1, 101.)):
            with self.assertRaises(ValueError):
                wall_radius(model, identity, z)

    def test_line_velocity_is_forward_independent_of_primitive_order(self):
        model = chamber([
            line('I', 'inner', [100., 250.], [0., 200.]),
            line('O', 'outer', [0., 300.], [100., 250.1])])
        velocity = film_velocity(model, [1, 1, 1], 50., [0., math.pi/2, math.pi], 7.)
        length = math.sqrt(1.25)
        expected = 7./length*np.array([[1., .5, 0.], [1., 0., .5], [1., -.5, 0.]])
        np.testing.assert_allclose(velocity, expected, atol=1e-14)
        np.testing.assert_allclose(np.linalg.norm(velocity, axis=-1), 7., atol=1e-14)

    def test_arc_vertical_tangent_speed_and_tangent_orthogonality(self):
        model = chamber([
            arc('I', 'inner', [0., 200.], [100., 200.], [50., 200.], True),
            line('O', 'outer', [0., 300.], [100., 300.])])
        z = np.array([0., 20., 50., 80., 100.])
        v = film_velocity(model, 1, z, 0., 3.)
        self.assertEqual(v[0, 0], 0.)
        self.assertEqual(v[-1, 0], 0.)
        np.testing.assert_array_equal(v[[0, -1], 1], [3., -3.])
        np.testing.assert_allclose(np.linalg.norm(v, axis=1), 3., atol=1e-14)
        normals = np.array([model.primitive_at('inner', value).normal(value) for value in z])
        np.testing.assert_allclose(np.sum(v[:, :2]*normals, axis=1), 0., atol=1e-14)
        self.assertTrue(np.all(v[:, 0] >= 0))

    def test_downstream_joint_tangent(self):
        model = chamber([
            line('I1', 'inner', [0., 200.], [50., 200.]),
            line('I2', 'inner', [50., 200.], [100., 250.]),
            line('O', 'outer', [0., 300.], [100., 300.])])
        np.testing.assert_allclose(film_velocity(model, 1, 50., 0., 1.),
                                   [1/math.sqrt(2), 1/math.sqrt(2), 0.], atol=1e-14)

    def test_geometry_tolerance_at_profile_join_is_not_an_uncovered_wall(self):
        model = chamber([
            line('I1', 'inner', [0., 200.], [50.-2e-8, 200.]),
            line('I2', 'inner', [50., 200.], [100., 200.]),
            line('O', 'outer', [0., 300.], [100., 300.])])
        np.testing.assert_allclose(wall_radius(model, 1, [50.-1e-8, 50., 50.+1e-8]), 200.)
        np.testing.assert_allclose(film_velocity(model, 1, 50.-1e-8, 0., 1.), [1., 0., 0.])

    def test_exact_piecewise_arc_length_is_additive(self):
        model = chamber([
            arc('I1', 'inner', [0., 200.], [50., 250.], [50., 200.], True),
            line('I2', 'inner', [50., 250.], [100., 250.]),
            line('O', 'outer', [0., 300.], [100., 300.])])
        expected = 50.*math.pi/2+50.
        self.assertAlmostEqual(float(wall_arc_length(model, 1, 0., 100.)), expected, places=12)
        distance = wall_arc_length(model, [1, 1, 2], [0., 25., 0.], [25., 100., 100.])
        self.assertAlmostEqual(float(distance[0]+distance[1]), expected, places=12)
        self.assertEqual(distance[2], 100.)
        np.testing.assert_array_equal(wall_arc_length(model, [1, 2], 40., 40.), 0.)

    def test_kinematic_invalid_values_fail_explicitly(self):
        model = chamber()
        for args in ((1, 50., 0., -1.), (1, 50., float('nan'), 1.),
                     (0, 50., 0., 1.), (1, 101., 0., 1.)):
            with self.assertRaises(ValueError):
                film_velocity(model, *args)
        with self.assertRaises(ValueError):
            wall_arc_length(model, 1, 70., 50.)


if __name__ == '__main__':
    unittest.main()
