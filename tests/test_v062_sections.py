"""Independent outlet/section invariants, with no Qt or computed-flow oracle."""
from copy import deepcopy
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from fuelfield.chamber import Chamber, Primitive, UNITS
from fuelfield.chamber_constraints import validate_constraints
from fuelfield.chamber_sections import sync_section_outlet, set_section_outlet


def line(identity, wall, a, b, unit='mm', fixed=False):
    scale = UNITS[unit]
    return dict(id=identity, wall=wall, kind='line', unit=unit,
                start=(np.array(a)/scale).tolist(),
                end=(np.array(b)/scale).tolist(), fixed=fixed,
                provenance={'source_kind': 'synthetic', 'keep': {'custom': identity}})


def straight(outlet=40., fixed=False):
    return dict(schema_version=2, enabled=True, base_z_mm=0., outlet_z_mm=outlet,
                chord_error_mm=.01, internal_step_mm=7.125,
                primitives=[line('old-inner', 'inner', [0., 200.], [outlet, 200.+outlet/4], fixed=fixed),
                            line('old-outer', 'outer', [0., 300.], [outlet, 300.+outlet/2], fixed=fixed),
                            dict(id='annotation', wall='reference', kind='point',
                                 start=[3., 250.], future={'keep': 'annotation'})],
                sections=[dict(id='S-user-zero', z_mm=0., is_outlet=False),
                          dict(id='S-user-mid', z_mm=10.25, is_outlet=False,
                               future={'keep': [1, None]}),
                          dict(id='S-user-outlet', z_mm=outlet, is_outlet=True,
                               future={'keep': 'last'})],
                objects=[], groups=[], constraints=[],
                future={'keep': {'unknown': [False, 360]}})


def curved(unit='mm', reversed_endpoints=False, fixed=False):
    raw = straight(30.)
    factor = UNITS[unit]
    arc = dict(id='user-arc', wall='inner', kind='arc', unit=unit,
               start=(np.array([0., 250.])/factor).tolist(),
               end=(np.array([30., 240.])/factor).tolist(),
               center=(np.array([0., 200.])/factor).tolist(),
               radius=50./factor, clockwise=True, fixed=fixed,
               provenance={'source_kind': 'synthetic', 'unknown': [1, 'circle']})
    outer = line('user-outer', 'outer', [0., 350.], [30., 350.], 'm', fixed)
    if reversed_endpoints:
        arc['start'], arc['end'] = arc['end'], arc['start']
        arc['clockwise'] = not arc['clockwise']
        outer['start'], outer['end'] = outer['end'], outer['start']
    raw['primitives'] = [arc, outer]
    return raw


def record(raw, identity):
    return next(p for p in raw['primitives'] if p['id'] == identity)


class TestSectionOutlet(unittest.TestCase):
    def assert_old_interval(self, before, after, stop):
        old_model, new_model = Chamber(before), Chamber(after)
        z = np.linspace(before['base_z_mm'], stop, 101)
        for wall in ('inner', 'outer'):
            actual = [new_model.bounds(float(x))[0 if wall == 'inner' else 1] for x in z]
            expected = [old_model.bounds(float(x))[0 if wall == 'inner' else 1] for x in z]
            np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-11)

    def test_internal_section_only_keeps_profiles_metadata_units_and_order(self):
        raw = straight()
        raw['sections'][1]['z_mm'] = 17.375
        raw['sections'] = [raw['sections'][2], raw['sections'][0], raw['sections'][1]]
        before = deepcopy(raw)
        result = sync_section_outlet(raw)
        self.assertEqual(raw, before)
        self.assertEqual(result, before)
        result['sections'][2]['future']['keep'].append('changed output')
        self.assertEqual(raw, before)
        self.assertEqual(result['outlet_z_mm'], 40.)

    def test_extension_adds_tangent_lines_and_preserves_fixed_originals_exactly(self):
        raw = straight(fixed=True)
        raw['sections'].append(dict(id='new-measurement', z_mm=60., is_outlet=False,
                                    provenance={'keep': 'manual station'}))
        original = deepcopy(raw)
        before = straight(fixed=True)
        result = sync_section_outlet(raw)
        self.assertEqual(raw, original)
        self.assertEqual(result['primitives'][:3], original['primitives'])
        self.assertEqual(result['outlet_z_mm'], 60.)
        self.assertEqual([s['id'] for s in result['sections']], [s['id'] for s in original['sections']])
        self.assertEqual([s['id'] for s in result['sections'] if s['is_outlet']], ['new-measurement'])
        self.assertEqual(result['internal_step_mm'], original['internal_step_mm'])
        self.assertEqual(result['future'], original['future'])
        added = result['primitives'][3:]
        self.assertEqual(len(added), 2)
        for wall, start, end in [('inner', [40., 210.], [60., 215.]),
                                 ('outer', [40., 320.], [60., 330.])]:
            new = next(p for p in added if p['wall'] == wall)
            self.assertEqual(new['kind'], 'line')
            self.assertEqual(new['unit'], 'mm')
            np.testing.assert_array_equal(new['start'], start)
            np.testing.assert_array_equal(new['end'], end)
            self.assertEqual(new['provenance']['operation'], 'extend_outlet_tangent')
            self.assertEqual(new['provenance']['from_primitive_id'], 'old-'+wall)
            self.assertEqual(new['provenance']['previous_outlet_mm'], 40.)
        validate_constraints(result)
        self.assert_old_interval(before, result, 40.)

    def test_arc_extension_uses_physical_tangent_for_mixed_units_and_reversed_ends(self):
        for unit in ('mm', 'cm', 'm'):
            for reverse in (False, True):
                with self.subTest(unit=unit, reversed=reverse):
                    raw = curved(unit, reverse, fixed=True)
                    before = deepcopy(raw)
                    result = set_section_outlet(raw, 40.)
                    self.assertEqual(raw, before)
                    self.assertEqual(record(result, 'user-arc'), record(before, 'user-arc'))
                    self.assertEqual(record(result, 'user-outer'), record(before, 'user-outer'))
                    added = [p for p in result['primitives'] if p['id'] not in {'user-arc', 'user-outer'}]
                    inner = next(p for p in added if p['wall'] == 'inner')
                    np.testing.assert_allclose(inner['start'], [30., 240.], rtol=0, atol=1e-12)
                    np.testing.assert_allclose(inner['end'], [40., 232.5], rtol=0, atol=1e-11)
                    self.assertEqual(inner['provenance']['from_primitive_id'], 'user-arc')
                    self.assertEqual(result['sections'][-1]['future'], {'keep': 'last'})
                    self.assert_old_interval(before, result, 30.)

    def test_line_shrink_trims_at_same_analytic_profile_without_input_mutation(self):
        raw = straight(60.)
        before = deepcopy(raw)
        result = set_section_outlet(raw, 40.)
        self.assertEqual(raw, before)
        self.assertEqual(result['outlet_z_mm'], 40.)
        self.assertEqual([s['id'] for s in result['sections']], [s['id'] for s in before['sections']])
        self.assertEqual(result['sections'][-1]['future'], before['sections'][-1]['future'])
        np.testing.assert_array_equal(record(result, 'old-inner')['end'], [40., 210.])
        np.testing.assert_array_equal(record(result, 'old-outer')['end'], [40., 320.])
        self.assertEqual(record(result, 'old-inner')['provenance'], record(before, 'old-inner')['provenance'])
        self.assert_old_interval(before, result, 40.)

    def test_arc_shrink_keeps_center_circle_direction_and_unknown_metadata(self):
        for unit in ('mm', 'cm'):
            for reverse in (False, True):
                with self.subTest(unit=unit, reversed=reverse):
                    raw = curved(unit, reverse)
                    before = deepcopy(raw)
                    result = set_section_outlet(raw, 20.)
                    a, b = record(before, 'user-arc'), record(result, 'user-arc')
                    self.assertEqual(a['center'], b['center'])
                    self.assertEqual(a['clockwise'], b['clockwise'])
                    self.assertEqual(a['provenance'], b['provenance'])
                    self.assertEqual(a['unit'], b['unit'])
                    cut = np.array(b['start' if reverse else 'end'])*UNITS[unit]
                    np.testing.assert_allclose(cut, [20., 200.+np.sqrt(2100.)], rtol=0, atol=1e-11)
                    self.assertAlmostEqual(Primitive(a).radius, Primitive(b).radius, places=11)
                    self.assertEqual(raw, before)
                    self.assert_old_interval(before, result, 20.)

    def test_explicit_outlet_moves_only_previous_maximum_and_rejects_overlap(self):
        raw = straight()
        raw['sections'] = [raw['sections'][2], raw['sections'][0], raw['sections'][1]]
        original = deepcopy(raw)
        result = set_section_outlet(raw, 50.)
        self.assertEqual(result['sections'][0]['id'], 'S-user-outlet')
        self.assertEqual(result['sections'][0]['z_mm'], 50.)
        self.assertEqual(result['sections'][1:], original['sections'][1:])
        self.assertEqual(raw, original)
        for z in (0., 10.25, 9., -1.):
            with self.subTest(z=z):
                with self.assertRaises(ValueError):
                    set_section_outlet(raw, z)
                self.assertEqual(raw, original)

    def test_fixed_trim_or_removal_is_rejected_without_changing_source(self):
        for raw in (straight(fixed=True), curved(fixed=True)):
            before = deepcopy(raw)
            with self.assertRaisesRegex(ValueError, 'зафиксирован'):
                set_section_outlet(raw, 20.)
            self.assertEqual(raw, before)

    def test_shrink_at_line_arc_boundary_removes_only_downstream_constraints(self):
        raw = straight(60.)
        raw['primitives'] = []
        for wall, radius in (('inner', 200.), ('outer', 300.)):
            raw['primitives'].extend([
                line(wall+'-A', wall, [0., radius], [20., radius]),
                line(wall+'-B', wall, [20., radius], [40., radius]),
                line(wall+'-C', wall, [40., radius], [60., radius]),
            ])
        middle = record(raw, 'outer-B')
        middle.update(kind='arc', center=[30., 310.], clockwise=False, fixed=True,
                      future_arc={'keep': [1, None]})
        for index, wall in enumerate(('inner', 'outer')):
            for offset, left, right in ((0, 'A', 'B'), (1, 'B', 'C')):
                raw['constraints'].append(dict(
                    id=f'C{index*2+offset+1}', kind='coincident',
                    a=dict(primitive_id=wall+'-'+left, endpoint='end'),
                    b=dict(primitive_id=wall+'-'+right, endpoint='start'),
                    unknown={'keep': wall+left+right}))
        before = deepcopy(raw)
        result = set_section_outlet(raw, 40.)
        self.assertEqual(raw, before)
        self.assertEqual([p['id'] for p in result['primitives']],
                         ['inner-A', 'inner-B', 'outer-A', 'outer-B'])
        self.assertEqual(record(result, 'outer-B'), record(before, 'outer-B'))
        self.assertEqual([c['id'] for c in result['constraints']], ['C1', 'C3'])
        self.assertEqual(result['constraints'], [before['constraints'][0], before['constraints'][2]])
        self.assert_old_interval(before, result, 40.)

    def test_vertical_arc_tangent_and_missing_terminal_fail_atomically(self):
        raw = curved()
        raw['outlet_z_mm'] = 50.
        raw['sections'][-1]['z_mm'] = 50.
        arc = record(raw, 'user-arc')
        arc['end'] = [50., 200.]
        record(raw, 'user-outer')['end'][0] = .05
        original = deepcopy(raw)
        with self.assertRaisesRegex(ValueError, 'вертикальн'):
            set_section_outlet(raw, 60.)
        self.assertEqual(raw, original)
        raw = straight()
        record(raw, 'old-outer')['end'][0] = 39.
        original = deepcopy(raw)
        with self.assertRaisesRegex(ValueError, 'конец профиля'):
            set_section_outlet(raw, 60.)
        self.assertEqual(raw, original)

    def test_extend_then_shrink_to_original_restores_old_profile_records_and_constraint_set(self):
        raw = curved()
        original = deepcopy(raw)
        extended = set_section_outlet(raw, 40.)
        restored = set_section_outlet(extended, 30.)
        self.assertEqual(raw, original)
        self.assertEqual(restored['primitives'], original['primitives'])
        self.assertEqual(restored['constraints'], original['constraints'])
        self.assertEqual(restored['sections'], original['sections'])
        self.assertEqual(restored['future'], original['future'])
        self.assert_old_interval(original, restored, 30.)


if __name__ == '__main__':
    unittest.main()
