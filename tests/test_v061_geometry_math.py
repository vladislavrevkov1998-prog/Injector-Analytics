"""Independent circle invariants for unrestricted v0.6.1 editing gestures."""
from copy import deepcopy
import math
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from fuelfield.chamber import Primitive, UNITS
from fuelfield.chamber_editing import (arc_circle, move_endpoint, preview,
                                       resize_arc, translate)


def circle_record(angle=.7, sweep=math.pi/2, unit='mm', center=(40., 230.), radius=30.):
    """Inputs are physical mm; serialize into independently chosen units."""
    factor = UNITS[unit]
    center = np.array(center, dtype=float)
    start = center+radius*np.array([math.cos(angle), math.sin(angle)])
    end = center+radius*np.array([math.cos(angle+sweep), math.sin(angle+sweep)])
    return dict(id='A', wall='reference', kind='arc', unit=unit,
                start=(start/factor).tolist(), end=(end/factor).tolist(),
                center=(center/factor).tolist(), radius=radius/factor,
                clockwise=sweep < 0, provenance={'synthetic': True, 'keep': [1, 2]})


def independent_sweep(raw):
    a, b, c = [np.array(raw[k], dtype=float) for k in ('start', 'end', 'center')]
    u, v = a-c, b-c
    angle = math.atan2(float(u[0]*v[1]-u[1]*v[0]), float(u@v))
    if raw['clockwise']:
        return angle if angle < 0 else angle-2*math.pi
    return angle if angle > 0 else angle+2*math.pi


class TestFreeArcEndpoints(unittest.TestCase):
    def test_endpoint_follows_cursor_opposite_fixed_and_bulge_preserved(self):
        for sweep in (-1.8, -.35, math.pi/2, 2.5, -math.pi, 1.5*math.pi):
            for angle in (-2.8, -.4, .7, 2.1):
                for unit in ('mm', 'cm', 'm'):
                    for role, opposite in [('start', 'end'), ('end', 'start')]:
                        with self.subTest(sweep=sweep, angle=angle, unit=unit, role=role):
                            original = circle_record(angle=angle, sweep=sweep, unit=unit)
                            before = deepcopy(original)
                            point = np.array(original[role])+np.array([7.25, -3.5])/UNITS[unit]
                            edited = move_endpoint(original, role, point)
                            np.testing.assert_array_equal(edited[role], point)
                            np.testing.assert_array_equal(edited[opposite], original[opposite])
                            self.assertAlmostEqual(independent_sweep(edited), sweep, places=10)
                            a, b, c = [np.array(edited[k]) for k in ('start', 'end', 'center')]
                            self.assertAlmostEqual(float(np.linalg.norm(a-c)/np.linalg.norm(b-c)), 1., places=11)
                            self.assertEqual(edited['id'], original['id'])
                            self.assertEqual(edited['unit'], unit)
                            self.assertEqual(edited['provenance'], original['provenance'])
                            self.assertNotIn('radius', edited)
                            self.assertEqual(original, before)

    def test_endpoint_motion_commutes_with_rigid_transform(self):
        raw = circle_record(sweep=-2.1)
        target = np.array(raw['end'])+[5., 7.]
        baseline = move_endpoint(raw, 'end', target)
        for angle in (.3, 1.5, 4.1):
            rotation = np.array([[math.cos(angle), -math.sin(angle)],
                                 [math.sin(angle), math.cos(angle)]])
            shift = np.array([100., -20.])
            moved = deepcopy(raw)
            for key in ('start', 'end', 'center'):
                moved[key] = (rotation@raw[key]+shift).tolist()
            result = move_endpoint(moved, 'end', rotation@target+shift)
            for key in ('start', 'end', 'center'):
                np.testing.assert_allclose(result[key], rotation@baseline[key]+shift,
                                           atol=2e-12, rtol=0)

    def test_collision_of_endpoints_is_diagnosed(self):
        raw = circle_record()
        with self.assertRaises(ValueError):
            move_endpoint(raw, 'start', raw['end'])
        with self.assertRaises(ValueError):
            move_endpoint(raw, 'end', raw['start'])

    def test_shared_endpoint_can_adapt_two_circular_neighbors_in_different_units(self):
        left = circle_record(angle=math.pi, sweep=-math.pi/2, center=(0., 200.), radius=30.)
        shared = np.array(left['end'])
        # The second arc starts at the same physical point but uses cm.
        right = circle_record(angle=math.pi, sweep=-math.pi/2, unit='cm',
                              center=shared+[20., 0.], radius=20.)
        np.testing.assert_allclose(np.array(right['start'])*10., shared, atol=1e-12)
        target = shared+[3.5, -2.25]
        updated_left = move_endpoint(left, 'end', target)
        updated_right = move_endpoint(right, 'start', target/10.)
        np.testing.assert_allclose(updated_left['end'], np.array(updated_right['start'])*10., atol=1e-12)
        self.assertAlmostEqual(independent_sweep(updated_left), -math.pi/2)
        self.assertAlmostEqual(independent_sweep(updated_right), -math.pi/2)
        np.testing.assert_array_equal(updated_left['start'], left['start'])
        np.testing.assert_array_equal(updated_right['end'], right['end'])


class TestArcTranslationAndRadius(unittest.TestCase):
    def test_translation_preserves_radius_sweep_and_all_metadata(self):
        for sweep in (-2.2, .7, math.pi, -1.5*math.pi):
            for unit in ('mm', 'cm', 'm'):
                raw = circle_record(sweep=sweep, unit=unit)
                before = deepcopy(raw)
                delta = np.array([19.5, -31.])/UNITS[unit]
                result = translate(raw, delta)
                for key in ('start', 'end', 'center'):
                    np.testing.assert_allclose(result[key], np.array(raw[key])+delta, atol=1e-12)
                self.assertAlmostEqual(independent_sweep(result), sweep, places=11)
                self.assertEqual(result['radius'], raw['radius'])
                self.assertEqual(result['provenance'], raw['provenance'])
                self.assertEqual(raw, before)

    def test_resize_preserves_both_endpoints_and_requested_radius(self):
        for sweep in (-2.2, .7, 2.5, -1.5*math.pi):
            for angle in (-2.4, .2, 2.9):
                for unit in ('mm', 'cm', 'm'):
                    with self.subTest(sweep=sweep, angle=angle, unit=unit):
                        raw = circle_record(angle=angle, sweep=sweep, unit=unit)
                        a, b, old_center = [np.array(raw[k]) for k in ('start', 'end', 'center')]
                        target_radius = 50./UNITS[unit]
                        result = resize_arc(raw, target_radius)
                        np.testing.assert_array_equal(result['start'], a)
                        np.testing.assert_array_equal(result['end'], b)
                        center = np.array(result['center'])
                        self.assertAlmostEqual(float(np.linalg.norm(a-center)), target_radius, places=11)
                        self.assertAlmostEqual(float(np.linalg.norm(b-center)), target_radius, places=11)
                        self.assertEqual(result['clockwise'], raw['clockwise'])
                        # A non-semicircle retains its center's chord side;
                        # thus a minor/major arc does not unexpectedly flip.
                        chord = b-a
                        normal = np.array([-chord[1], chord[0]])
                        self.assertGreater(float((center-(a+b)/2)@normal)*
                                           float((old_center-(a+b)/2)@normal), 0.)
                        self.assertEqual(abs(independent_sweep(result)) > math.pi,
                                         abs(sweep) > math.pi)

    def test_resize_upper_and_lower_semicircle_continues_minor_arc_side(self):
        for clockwise in (False, True):
            raw = dict(id='A', wall='reference', kind='arc', unit='mm',
                       start=[0., 200.], end=[100., 200.], center=[50., 200.],
                       clockwise=clockwise, radius=50.)
            result = resize_arc(raw, 70.)
            self.assertLess(abs(independent_sweep(result)), math.pi)
            branch = 1 if clockwise else -1
            expected_center = np.array([50., 200.-branch*math.sqrt(70.**2-50.**2)])
            np.testing.assert_allclose(result['center'], expected_center, atol=1e-12)
            # The resized physical arc remains a usable single-valued wall.
            p = Primitive(result)
            self.assertGreater(branch*(float(p.at(50.))-200.), 0.)

    def test_near_semicircle_resize_is_invariant_under_unit_change(self):
        # The center is close to the chord but physically distinct. An mm
        # tolerance must not accidentally become a much larger meter tolerance.
        for sign in (-1, 1):
            sweep = sign*(math.pi+1e-6)
            reference = None
            for unit in ('mm', 'cm', 'm'):
                raw = circle_record(sweep=sweep, unit=unit)
                result = resize_arc(raw, 50./UNITS[unit])
                center_mm = np.array(result['center'])*UNITS[unit]
                with self.subTest(sign=sign, unit=unit):
                    self.assertGreater(abs(independent_sweep(result)), math.pi)
                    if reference is None:
                        reference = center_mm
                    else:
                        np.testing.assert_allclose(center_mm, reference, rtol=0, atol=1e-9)

    def test_invalid_radius_and_degenerate_chord_reject(self):
        raw = circle_record()
        minimum = float(np.linalg.norm(np.array(raw['end'])-raw['start']))/2
        for radius in (0., -2., minimum-1e-7, float('nan'), float('inf'), True):
            with self.subTest(radius=radius), self.assertRaises(ValueError):
                resize_arc(raw, radius)
        degenerate = deepcopy(raw)
        degenerate['end'] = degenerate['start'][:]
        with self.assertRaises(ValueError):
            resize_arc(degenerate, 100.)


class TestInvalidDraftVisibility(unittest.TestCase):
    def test_nonmonotone_arc_has_exact_visible_preview_and_diagnostic(self):
        raw = circle_record(angle=.2, sweep=1.5*math.pi)
        with self.assertRaises(ValueError):
            Primitive(raw)
        points, error = preview(raw, .01)
        self.assertTrue(error)
        self.assertGreater(len(points), 10)
        np.testing.assert_array_equal(points[0], raw['start'])
        np.testing.assert_array_equal(points[-1], raw['end'])
        np.testing.assert_allclose(np.linalg.norm(points-np.array(raw['center']), axis=1),
                                   raw['radius'], atol=1e-11, rtol=0)

    def test_invalid_arc_preview_and_sagitta_use_physical_units(self):
        for unit in ('mm', 'cm', 'm'):
            raw = circle_record(angle=.2, sweep=-1.5*math.pi, unit=unit)
            points, error = preview(raw, .005)
            self.assertTrue(error)
            center = np.array(raw['center'])*UNITS[unit]
            np.testing.assert_allclose(points[0], np.array(raw['start'])*UNITS[unit], atol=1e-12)
            np.testing.assert_allclose(points[-1], np.array(raw['end'])*UNITS[unit], atol=1e-12)
            mids = (points[:-1]+points[1:])/2
            sagitta = 30.-np.linalg.norm(mids-center, axis=1)
            self.assertLessEqual(float(sagitta.max()), .005+1e-11)

    def test_inconsistent_circle_remains_control_polygon(self):
        raw = circle_record()
        raw['end'][0] += 9.
        points, error = preview(raw, .01)
        self.assertTrue(error)
        self.assertGreaterEqual(len(points), 2)
        self.assertTrue(np.isfinite(points).all())
        np.testing.assert_array_equal(points[0], raw['start'])
        np.testing.assert_array_equal(points[-1], raw['end'])


if __name__ == '__main__':
    unittest.main()
