"""Independent geometric invariants for explicit v0.6.2 endpoint constraints."""
from copy import deepcopy
import math
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from fuelfield.chamber import Primitive, UNITS
from fuelfield.chamber_constraints import (add_constraint, apply_constraints,
    materialize_constraints, remove_constraints, tangent_arc, validate_constraints)


def ref(identity, endpoint):
    return dict(primitive_id=identity, endpoint=endpoint)


def line(identity, a, b, unit='mm', fixed=False):
    scale = UNITS[unit]
    return dict(id=identity, kind='line', wall='reference', unit=unit,
                start=(np.array(a, dtype=float)/scale).tolist(),
                end=(np.array(b, dtype=float)/scale).tolist(), fixed=fixed)


def original_arc(unit='mm', reverse=False):
    factor = UNITS[unit]
    raw = dict(id='arc', kind='arc', wall='reference', unit=unit,
               start=(np.array([22., 202.])/factor).tolist(),
               end=(np.array([80., 240.])/factor).tolist(),
               center=(np.array([22.5, 264.5])/factor).tolist(), clockwise=False,
               fixed=False, provenance=dict(origin='synthetic audit', unknown=[1, 2]))
    if reverse:
        raw['start'], raw['end'] = raw['end'], raw['start']
        raw['clockwise'] = not raw['clockwise']
    return raw


def raw_geometry(primitives, constraints=()):
    return dict(schema_version=2, enabled=False, base_z_mm=0., outlet_z_mm=100.,
                primitives=deepcopy(primitives), objects=[], groups=[],
                sections=[dict(id='outlet', z_mm=100., is_outlet=True)],
                constraints=deepcopy(list(constraints)), unknown={'keep': True})


def lookup(raw, identity):
    return next(p for p in raw['primitives'] if p['id'] == identity)


def point_mm(raw, role):
    return np.array(raw[role])*UNITS[raw.get('unit', 'mm')]


class TestExactArcLineTangency(unittest.TestCase):
    def check_tangent(self, arc, endpoint, target, line_endpoint):
        p = point_mm(arc, endpoint)
        contact = point_mm(target, line_endpoint)
        c = point_mm(arc, 'center')
        q = point_mm(arc, 'end' if endpoint == 'start' else 'start')
        direction = point_mm(target, 'end')-point_mm(target, 'start')
        direction /= np.linalg.norm(direction)
        np.testing.assert_allclose(p, contact, rtol=0, atol=1e-10)
        radius = np.linalg.norm(p-c)
        self.assertGreater(radius, 0)
        self.assertAlmostEqual(float(np.linalg.norm(q-c)), float(radius), places=9)
        self.assertLess(abs(float((p-c)@direction)), 1e-9)

    def test_unique_circle_preserves_other_end_and_never_changes_line(self):
        for unit in ('mm', 'cm', 'm'):
            for reverse_arc in (False, True):
                for reverse_line in (False, True):
                    with self.subTest(unit=unit, reverse_arc=reverse_arc, reverse_line=reverse_line):
                        arc = original_arc(unit, reverse_arc)
                        target = line('line', [0., 200.], [20., 200.], 'cm', True)
                        if reverse_line:
                            target['start'], target['end'] = target['end'], target['start']
                        role = 'end' if reverse_arc else 'start'
                        opposite = 'start' if reverse_arc else 'end'
                        target_role = 'start' if reverse_line else 'end'
                        before_arc, before_line = deepcopy(arc), deepcopy(target)
                        result = tangent_arc(arc, role, target, target_role)
                        self.check_tangent(result, role, target, target_role)
                        np.testing.assert_array_equal(result[opposite], arc[opposite])
                        np.testing.assert_allclose(point_mm(result, 'center'), [20., 265.], atol=1e-9)
                        self.assertEqual(result['id'], arc['id'])
                        self.assertEqual(result['unit'], unit)
                        self.assertEqual(result['provenance'], arc['provenance'])
                        self.assertEqual(arc, before_arc)
                        self.assertEqual(target, before_line)
                        Primitive(result)  # A usable R(z) branch exists here.

    def test_tangent_solution_commutes_with_rotation_and_translation(self):
        baseline_arc = original_arc()
        baseline_line = line('line', [0., 200.], [20., 200.])
        center = np.array([20., 265.])
        for angle in (0., .2, -.2, math.pi, math.pi+.2):
            rotation = np.array([[math.cos(angle), -math.sin(angle)],
                                 [math.sin(angle), math.cos(angle)]])
            shift = np.array([1000., 1000.])
            arc, target = deepcopy(baseline_arc), deepcopy(baseline_line)
            for primitive in (arc, target):
                for key in ('start', 'end', 'center'):
                    if key in primitive:
                        primitive[key] = (rotation@primitive[key]+shift).tolist()
            with self.subTest(angle=angle):
                result = tangent_arc(arc, 'start', target, 'end')
                self.check_tangent(result, 'start', target, 'end')
                np.testing.assert_allclose(point_mm(result, 'center'), rotation@center+shift,
                                           rtol=0, atol=2e-9)

    def test_admissible_arc_orientation_is_chosen_when_old_direction_is_long_arc(self):
        arc = original_arc()
        arc['clockwise'] = True
        target = line('line', [0., 200.], [20., 200.])
        result = tangent_arc(arc, 'start', target, 'end')
        self.check_tangent(result, 'start', target, 'end')
        primitive = Primitive(result)
        self.assertLess(abs(primitive.sweep), math.pi)

    def test_no_finite_circle_if_other_end_lies_on_tangent_line(self):
        target = line('line', [0., 200.], [20., 200.])
        arc = original_arc()
        arc['end'] = [80., 200.]
        with self.assertRaises(ValueError):
            tangent_arc(arc, 'start', target, 'end')

    def test_coincident_arc_ends_and_zero_line_are_diagnosed(self):
        arc = original_arc()
        target = line('line', [0., 200.], [20., 200.])
        arc['end'] = target['end'][:]
        with self.assertRaises(ValueError):
            tangent_arc(arc, 'start', target, 'end')
        target['start'] = target['end'][:]
        with self.assertRaises(ValueError):
            tangent_arc(original_arc(), 'start', target, 'end')


class TestExplicitCoincidentComponents(unittest.TestCase):
    def test_legacy_links_materialize_once_but_explicit_empty_never_rejoins(self):
        raw = raw_geometry([line('a', [0., 200.], [20., 210.]),
                            line('b', [20., 210.], [40., 220.]),
                            line('different-wall', [20., 210.], [70., 240.])])
        lookup(raw, 'different-wall')['wall'] = 'outer'
        raw['schema_version'] = 1
        del raw['constraints']
        materialize_constraints(raw)
        self.assertEqual(raw['schema_version'], 2)
        pairs = [{(c['a']['primitive_id'], c['a']['endpoint']),
                  (c['b']['primitive_id'], c['b']['endpoint'])}
                 for c in raw['constraints'] if c['kind'] == 'coincident']
        self.assertEqual(pairs, [{('a', 'end'), ('b', 'start')}])
        raw['constraints'] = []
        materialize_constraints(raw)
        self.assertEqual(raw['constraints'], [])
        lookup(raw, 'a')['end'] = [25., 215.]
        result = apply_constraints(raw, drivers=[('a', 'end')])
        np.testing.assert_array_equal(lookup(result, 'b')['start'], [20., 210.])

    def test_deleted_coincidence_does_not_reappear_from_equal_coordinates(self):
        raw = raw_geometry([line('a', [0., 200.], [20., 210.]),
                            line('b', [20., 210.], [40., 220.])])
        linked = add_constraint(raw, 'coincident', ref('a', 'end'), ref('b', 'start'))
        ids = [c['id'] for c in linked['constraints']]
        self.assertTrue(ids)
        freed = remove_constraints(linked, ids)
        freed['primitives'][0]['end'] = [25., 215.]
        actual = apply_constraints(freed, drivers=[('a', 'end')])
        np.testing.assert_array_equal(lookup(actual, 'a')['end'], [25., 215.])
        np.testing.assert_array_equal(lookup(actual, 'b')['start'], [20., 210.])
        self.assertFalse(actual['constraints'])

    def test_transitive_component_propagates_only_linked_endpoints(self):
        raw = raw_geometry([line('a', [0., 200.], [20., 210.]),
                            line('b', [20., 210.], [40., 220.], 'cm'),
                            line('c', [20., 210.], [60., 230.]),
                            line('free', [20., 210.], [70., 240.])])
        raw = add_constraint(raw, 'coincident', ref('a', 'end'), ref('b', 'start'))
        raw = add_constraint(raw, 'coincident', ref('b', 'start'), ref('c', 'start'))
        lookup(raw, 'a')['end'] = [23., 214.]
        result = apply_constraints(raw, drivers=[('a', 'end')])
        for identity, role in [('a', 'end'), ('b', 'start'), ('c', 'start')]:
            np.testing.assert_allclose(point_mm(lookup(result, identity), role), [23., 214.], atol=1e-10)
        np.testing.assert_array_equal(lookup(result, 'free')['start'], [20., 210.])
        np.testing.assert_array_equal(lookup(result, 'b')['end'], [4., 22.])
        self.assertEqual(result['unknown'], raw['unknown'])

    def test_large_reverse_order_component_has_no_python_recursion_limit(self):
        # This ordering deliberately makes an unbalanced parent chain. A
        # collection of valid coincident endpoints must not crash Python/UI.
        count = 1100
        primitives = [line(f'p{i}', [20., 210.], [40.+i, 220.]) for i in range(count)]
        constraints = [dict(id=f'C{i}', kind='coincident',
                            a=ref(f'p{i}', 'start'), b=ref(f'p{i-1}', 'start'))
                       for i in range(1, count)]
        raw = raw_geometry(primitives, constraints)
        result = apply_constraints(raw, drivers=[(f'p{count-1}', 'start')])
        self.assertEqual(len(result['constraints']), count-1)
        self.assertEqual(len(result['primitives']), count)
        validate_constraints(result)

    def test_coincident_arc_adapts_circle_while_preserving_opposite_end(self):
        arc = original_arc('cm')
        raw = raw_geometry([arc, line('driver', [0., 200.], [22., 202.])])
        raw = add_constraint(raw, 'coincident', ref('driver', 'end'), ref('arc', 'start'))
        opposite = lookup(raw, 'arc')['end'][:]
        lookup(raw, 'driver')['end'] = [25., 205.]
        result = apply_constraints(raw, drivers=[('driver', 'end')])
        updated = lookup(result, 'arc')
        np.testing.assert_allclose(point_mm(updated, 'start'), [25., 205.], atol=1e-10)
        np.testing.assert_array_equal(updated['end'], opposite)
        a, b, c = [point_mm(updated, key) for key in ('start', 'end', 'center')]
        self.assertAlmostEqual(float(np.linalg.norm(a-c)), float(np.linalg.norm(b-c)), places=10)

    def test_disabled_constraint_does_not_propagate_or_require_satisfaction(self):
        constraint = dict(id='C1', kind='coincident', a=ref('a', 'end'),
                          b=ref('b', 'start'), enabled=False)
        raw = raw_geometry([line('a', [0., 200.], [20., 210.]),
                            line('b', [30., 220.], [40., 230.])], [constraint])
        actual = apply_constraints(raw, drivers=[('a', 'end')])
        self.assertEqual(actual['primitives'], raw['primitives'])
        validate_constraints(actual)

    def test_conflicting_drivers_do_not_silently_choose_last(self):
        raw = raw_geometry([line('a', [0., 200.], [20., 210.]),
                            line('b', [20., 210.], [40., 220.])])
        raw = add_constraint(raw, 'coincident', ref('a', 'end'), ref('b', 'start'))
        lookup(raw, 'a')['end'] = [23., 214.]
        lookup(raw, 'b')['start'] = [24., 215.]
        before = deepcopy(raw)
        with self.assertRaises(ValueError):
            apply_constraints(raw, drivers=[('a', 'end'), ('b', 'start')])
        self.assertEqual(raw, before)

    def test_fixed_anchor_never_moves_to_satisfy_free_driver(self):
        raw = raw_geometry([line('a', [0., 200.], [20., 210.], fixed=True),
                            line('b', [20., 210.], [40., 220.])])
        raw = add_constraint(raw, 'coincident', ref('a', 'end'), ref('b', 'start'))
        baseline = deepcopy(raw)
        lookup(raw, 'b')['start'] = [25., 215.]
        try:
            result = apply_constraints(raw, drivers=[('b', 'start')], baseline=baseline)
        except ValueError:
            pass  # Rejecting a drag against the lock is also valid.
        else:
            np.testing.assert_array_equal(lookup(result, 'a'), lookup(baseline, 'a'))
            np.testing.assert_allclose(point_mm(lookup(result, 'b'), 'start'), [20., 210.])
        self.assertEqual(lookup(raw, 'a'), lookup(baseline, 'a'))

    def test_fixed_geometry_edit_is_rejected_against_baseline(self):
        raw = raw_geometry([line('fixed', [0., 200.], [20., 210.], fixed=True)])
        candidate = deepcopy(raw)
        lookup(candidate, 'fixed')['end'] = [21., 211.]
        with self.assertRaises(ValueError):
            apply_constraints(candidate, baseline=raw)

    def test_fixed_primitive_unit_conversion_preserves_physical_geometry(self):
        raw = raw_geometry([line('fixed', [0., 200.], [20., 210.], fixed=True)])
        candidate = deepcopy(raw)
        primitive = lookup(candidate, 'fixed')
        primitive['unit'] = 'cm'
        for key in ('start', 'end'):
            primitive[key] = (np.array(primitive[key])/10.).tolist()
        actual = apply_constraints(candidate, baseline=raw)
        for key in ('start', 'end'):
            np.testing.assert_allclose(point_mm(lookup(actual, 'fixed'), key),
                                       point_mm(lookup(raw, 'fixed'), key), atol=1e-10)

    def test_fixed_arc_clockwise_change_is_a_geometry_change(self):
        arc = original_arc()
        arc['fixed'] = True
        raw = raw_geometry([arc])
        candidate = deepcopy(raw)
        lookup(candidate, 'arc')['clockwise'] = not arc['clockwise']
        with self.assertRaises(ValueError):
            apply_constraints(candidate, baseline=raw)

    def test_fixed_arc_cannot_be_reshaped_by_tangency(self):
        arc = original_arc()
        arc['fixed'] = True
        raw = raw_geometry([arc, line('line', [0., 200.], [20., 200.], fixed=True)])
        with self.assertRaises(ValueError):
            linked = add_constraint(raw, 'tangent', ref('arc', 'start'), ref('line', 'end'))
            apply_constraints(linked)

    def test_two_different_fixed_anchors_are_diagnosed(self):
        raw = raw_geometry([line('a', [0., 200.], [20., 210.], fixed=True),
                            line('b', [21., 211.], [40., 220.], fixed=True)])
        with self.assertRaises(ValueError):
            linked = add_constraint(raw, 'coincident', ref('a', 'end'), ref('b', 'start'))
            apply_constraints(linked)


class TestCombinedTangentConstraints(unittest.TestCase):
    def case(self, second_tangent=False, compatible=True):
        arc = original_arc()
        primitives = [arc, line('line1', [0., 200.], [20., 200.], fixed=True)]
        if second_tangent:
            endpoint = [90., 264.] if compatible else [90., 240.]
            primitives.append(line('line2', [80., 240.], endpoint, fixed=True))
        raw = raw_geometry(primitives)
        raw = add_constraint(raw, 'tangent', ref('arc', 'start'), ref('line1', 'end'))
        if second_tangent:
            raw = add_constraint(raw, 'tangent', ref('line2', 'start'), ref('arc', 'end'))
        return raw

    def test_tangent_binds_contact_and_only_arc_changes(self):
        raw = self.case()
        before = deepcopy(raw)
        result = apply_constraints(raw)
        TestExactArcLineTangency().check_tangent(lookup(result, 'arc'), 'start',
                                               lookup(result, 'line1'), 'end')
        self.assertEqual(lookup(result, 'line1'), lookup(before, 'line1'))
        np.testing.assert_array_equal(lookup(result, 'arc')['end'], lookup(before, 'arc')['end'])
        validate_constraints(result)
        self.assertEqual(raw, before)

    def test_two_compatible_tangencies_share_one_circle(self):
        raw = self.case(second_tangent=True)
        result = apply_constraints(raw)
        arc = lookup(result, 'arc')
        checker = TestExactArcLineTangency()
        checker.check_tangent(arc, 'start', lookup(result, 'line1'), 'end')
        checker.check_tangent(arc, 'end', lookup(result, 'line2'), 'start')
        np.testing.assert_allclose(point_mm(arc, 'center'), [20., 265.], atol=1e-10)
        # Applying again or reversing equation order must not change geometry.
        again = apply_constraints(result)
        np.testing.assert_allclose(point_mm(lookup(again, 'arc'), 'center'), [20., 265.], atol=1e-10)
        reversed_raw = deepcopy(raw)
        reversed_raw['constraints'].reverse()
        alternate = apply_constraints(reversed_raw)
        np.testing.assert_allclose(point_mm(lookup(alternate, 'arc'), 'center'), [20., 265.], atol=1e-10)

    def test_two_incompatible_tangencies_are_not_last_equation_wins(self):
        # add_constraint is allowed to validate eagerly, so either adding the
        # impossible second equation or applying the batch must diagnose it.
        with self.assertRaises(ValueError):
            raw = self.case(second_tangent=True, compatible=False)
            apply_constraints(raw)

    def test_same_contact_parallel_lines_are_compatible_in_either_orientation(self):
        raw = self.case()
        raw['primitives'].append(line('line2', [20., 200.], [40., 200.], fixed=True))
        raw = add_constraint(raw, 'tangent', ref('arc', 'start'), ref('line2', 'start'))
        actual = apply_constraints(raw)
        center = point_mm(lookup(actual, 'arc'), 'center')
        np.testing.assert_allclose(center, [20., 265.], atol=1e-10)
        validate_constraints(actual)

    def test_same_contact_nonparallel_lines_cannot_both_be_tangent(self):
        with self.assertRaises(ValueError):
            raw = self.case()
            raw['primitives'].append(line('line2', [20., 200.], [40., 210.], fixed=True))
            raw = add_constraint(raw, 'tangent', ref('arc', 'start'), ref('line2', 'start'))
            apply_constraints(raw)

    def test_removing_tangent_releases_its_implicit_contact_binding(self):
        result = apply_constraints(self.case())
        released = remove_constraints(result, [c['id'] for c in result['constraints']])
        baseline = deepcopy(released)
        lookup(released, 'arc')['start'] = [25., 205.]
        # A free invalid draft is not silently reattached merely because an
        # earlier tangent record used to connect these endpoints.
        actual = apply_constraints(released, drivers=[('arc', 'start')], baseline=baseline)
        np.testing.assert_array_equal(lookup(actual, 'arc')['start'], [25., 205.])
        np.testing.assert_array_equal(lookup(actual, 'line1')['end'], [20., 200.])

    def test_removing_explicit_coincidence_removes_dependent_same_pair_tangent(self):
        raw = self.case()
        raw = add_constraint(raw, 'coincident', ref('arc', 'start'), ref('line1', 'end'))
        coincidences = [c['id'] for c in raw['constraints'] if c['kind'] == 'coincident']
        self.assertTrue(coincidences)
        released = remove_constraints(raw, coincidences)
        self.assertFalse(any(c['kind'] == 'tangent' for c in released['constraints']))


class TestConstraintRecordValidation(unittest.TestCase):
    def test_bad_endpoint_missing_object_and_duplicate_ids_are_diagnosed(self):
        base = raw_geometry([line('a', [0., 200.], [20., 210.]),
                             line('b', [20., 210.], [40., 220.])])
        valid = dict(id='C1', kind='coincident', a=ref('a', 'end'), b=ref('b', 'start'))
        invalid = []
        for bad_ref in (ref('missing', 'start'), ref('a', 'center'), ref('a', 'unknown')):
            item = deepcopy(valid)
            item['a'] = bad_ref
            invalid.append([item])
        invalid.append([deepcopy(valid), deepcopy(valid)])
        invalid.append([dict(valid, kind='unknown')])
        invalid.append([dict(valid, kind='tangent')])  # two lines are not arc+line
        for constraints in invalid:
            candidate = deepcopy(base)
            candidate['constraints'] = constraints
            with self.subTest(constraints=constraints), self.assertRaises(ValueError):
                validate_constraints(candidate)


if __name__ == '__main__':
    unittest.main()
