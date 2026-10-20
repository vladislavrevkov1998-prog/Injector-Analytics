"""Synthetic exact geometry regressions for object tangency and endpoint tolerance."""
from copy import deepcopy
import math
import unittest

import numpy as np

from fuelfield.chamber import Primitive, TOL, UNITS
from fuelfield.chamber_constraints import (add_constraint, add_object_tangent,
    apply_constraints, remove_constraints, tangent_arc, tangent_references,
    validate_constraints)
from test_v062_constraints_math import (line, lookup, original_arc, point_mm,
    raw_geometry, ref)


def reference_arc(unit='mm', internal=False, fixed=False):
    factor=UNITS[unit]
    center=np.array([20.,300. if internal else 220.])
    radius=center[1]-200.
    start=np.array([0.,center[1]-math.sqrt(radius*radius-400.)])
    return dict(id='reference',kind='arc',wall='reference',unit=unit,
                start=(start/factor).tolist(),end=(np.array([20.,200.])/factor).tolist(),
                center=(center/factor).tolist(),clockwise=False,fixed=fixed,
                metadata={'source':'synthetic exact circle'})


class TestObjectContacts(unittest.TestCase):
    def test_nearest_pair_is_resolved_without_point_hit_testing(self):
        raw=raw_geometry([original_arc(),line('line',[0.,200.],[20.,200.])])
        a,b=tangent_references(raw,'line','arc')
        self.assertEqual((a,b),(ref('arc','start'),ref('line','end')))
        actual=add_object_tangent(raw,'line','arc')
        validate_constraints(actual)
        np.testing.assert_allclose(point_mm(lookup(actual,'arc'),'center'),[20.,265.])

    def test_existing_transitive_coincidence_has_priority_over_distance(self):
        arc=original_arc()
        target=line('line',[0.,200.],[20.,200.])
        bridge=line('bridge',[20.,200.],[100.,300.])
        raw=raw_geometry([arc,target,bridge],[
            dict(id='C1',kind='coincident',a=ref('arc','end'),b=ref('bridge','start')),
            dict(id='C2',kind='coincident',a=ref('bridge','start'),b=ref('line','start'))])
        # This raw draft is deliberately not solved: endpoint coincidence
        # metadata should select the intended contact even before validation.
        self.assertEqual(tangent_references(raw,'arc','line'),
                         (ref('arc','end'),ref('line','start')))

    def test_coincidence_then_object_tangency_preserves_both_relations(self):
        raw=raw_geometry([original_arc(),line('line',[0.,200.],[20.,200.])])
        joined=add_constraint(raw,'coincident',ref('arc','start'),ref('line','end'))
        result=add_object_tangent(joined,'arc','line')
        self.assertEqual([c['kind'] for c in result['constraints']],['coincident','tangent'])
        validate_constraints(result)
        freed=remove_constraints(result,[result['constraints'][-1]['id']])
        self.assertEqual([c['kind'] for c in freed['constraints']],['coincident'])

    def test_ambiguous_contact_and_invalid_object_pair_are_diagnosed(self):
        raw=raw_geometry([original_arc(),line('line',[22.,180.],[22.,224.])])
        with self.assertRaisesRegex(ValueError,'неоднозначен'):
            tangent_references(raw,'arc','line')
        for pair in [('arc','arc'),('arc','missing')]:
            with self.subTest(pair=pair),self.assertRaises(ValueError):
                tangent_references(raw,*pair)
        raw['primitives'].append(line('line2',[0.,250.],[10.,250.]))
        with self.assertRaises(ValueError):tangent_references(raw,'line','line2')


class TestExactArcArcTangency(unittest.TestCase):
    def test_internal_and_external_circle_contacts_preserve_reference_and_other_end(self):
        for unit in ('mm','cm','m'):
            for internal in (False,True):
                for reverse in (False,True):
                    with self.subTest(unit=unit,internal=internal,reverse=reverse):
                        arc=original_arc(unit,reverse)
                        target=reference_arc('cm',internal,fixed=True)
                        before=deepcopy(target)
                        role='end' if reverse else 'start'
                        other='start' if reverse else 'end'
                        fitted=tangent_arc(arc,role,target,'end')
                        self.assertEqual(target,before)
                        self.assertEqual(fitted[other],arc[other])
                        self.assertEqual(fitted['provenance'],arc['provenance'])
                        contact=point_mm(fitted,role)
                        np.testing.assert_allclose(contact,[20.,200.],atol=1e-10)
                        center=point_mm(fitted,'center')
                        np.testing.assert_allclose(center,[20.,265.],atol=1e-10)
                        radii=[contact-center,contact-point_mm(target,'center')]
                        self.assertLess(abs(float(np.linalg.det(np.array(radii)))),1e-9)
                        Primitive(fitted)

    def test_object_arc_arc_tangency_uses_second_as_reference_and_roundtrips(self):
        raw=raw_geometry([original_arc(),reference_arc()])
        before=deepcopy(raw)
        result=add_object_tangent(raw,'arc','reference')
        self.assertEqual(raw,before)
        self.assertEqual(lookup(result,'reference'),lookup(raw,'reference'))
        self.assertEqual(result['constraints'][0]['a'],ref('arc','start'))
        self.assertEqual(result['constraints'][0]['b'],ref('reference','end'))
        validate_constraints(result)
        again=apply_constraints(result)
        np.testing.assert_allclose(point_mm(lookup(again,'arc'),'center'),[20.,265.])

    def test_fixed_first_arc_becomes_reference_and_both_fixed_tangent_are_accepted(self):
        target=reference_arc(fixed=True)
        raw=raw_geometry([original_arc(),target])
        self.assertEqual(tangent_references(raw,'reference','arc'),
                         (ref('arc','start'),ref('reference','end')))
        result=add_object_tangent(raw,'reference','arc')
        lookup(result,'arc')['fixed']=True
        again=apply_constraints(result,baseline=result)
        self.assertEqual(again,result)

    def test_two_fixed_non_tangent_arcs_do_not_move(self):
        first=original_arc();first['fixed']=True
        raw=raw_geometry([first,reference_arc(fixed=True)])
        before=deepcopy(raw)
        with self.assertRaises(ValueError):add_object_tangent(raw,'arc','reference')
        self.assertEqual(raw,before)

    def test_reference_tangency_is_solved_before_dependent_arc_independent_of_record_order(self):
        # Reference circle is first fitted to a line; the dependent arc then
        # uses its freshly calculated normal at its opposite contact.
        target=reference_arc()
        # A valid initial circle through start/end, with a different tangent.
        target['start']=[0.,200.]
        target['end']=[20.,200.]
        target['center']=[10.,220.]
        target['clockwise']=False
        dependent=deepcopy(original_arc())
        dependent['start']=[22.,202.]
        target_line=line('line',[-20.,220.],[0.,220.],fixed=True)
        # Fit the reference start to [0,220], preserving [20,200], giving
        # center [0,200]. Its end then has a vertical tangent. Use a dependent
        # opposite endpoint that admits that exact contact geometry.
        dependent['end']=[50.,170.]
        dependent['center']=[52.,202.]
        dependent['clockwise']=False
        records=[dict(id='C2',kind='tangent',a=ref('arc','start'),b=ref('reference','end')),
                 dict(id='C1',kind='tangent',a=ref('reference','start'),b=ref('line','end'))]
        raw=raw_geometry([dependent,target,target_line],records)
        result=apply_constraints(raw)
        validate_constraints(result)
        np.testing.assert_allclose(point_mm(lookup(result,'reference'),'center'),[0.,200.])
        np.testing.assert_allclose(point_mm(lookup(result,'arc'),'center'),[50.,200.])
        raw['constraints'].reverse()
        alternate=apply_constraints(raw)
        for identity in ('arc','reference'):
            np.testing.assert_allclose(point_mm(lookup(alternate,identity),'center'),
                                       point_mm(lookup(result,identity),'center'))

    def test_unsatisfied_arc_dependency_cycle_has_actionable_error(self):
        arc=original_arc();arc['start']=[20.,200.];arc['center']=[40.,235.]
        raw=raw_geometry([arc,reference_arc()],[
            dict(id='C1',kind='tangent',a=ref('arc','start'),b=ref('reference','end')),
            dict(id='C2',kind='tangent',a=ref('reference','end'),b=ref('arc','start'))])
        with self.assertRaisesRegex(ValueError,'цикл'):apply_constraints(raw)


class TestPhysicalArcReversalTolerance(unittest.TestCase):
    def arc(self,radius,overshoot_mm,clockwise=True):
        center=np.array([100.,radius+200.])
        angle=overshoot_mm/radius
        start=center+radius*np.array([math.cos(angle),math.sin(angle)])
        end=center+radius*np.array([math.cos(-math.pi/4),math.sin(-math.pi/4)])
        return dict(id='synthetic',kind='arc',wall='outer',unit='mm',
                    start=start.tolist(),end=end.tolist(),center=center.tolist(),clockwise=clockwise)

    def test_roundoff_at_vertical_tangent_is_accepted_in_physical_units(self):
        for radius in (.01,1.,1000.):
            for unit in ('mm','cm','m'):
                raw=self.arc(radius,TOL/4)
                factor=UNITS[unit];raw['unit']=unit
                for key in ('start','end','center'):raw[key]=(np.array(raw[key])/factor).tolist()
                with self.subTest(radius=radius,unit=unit):
                    primitive=Primitive(raw)
                    self.assertLessEqual(abs(float(primitive.at(primitive.start[0]))-primitive.start[1]),TOL*1.01)

    def test_real_reversal_is_rejected_even_if_z_excursion_is_below_tolerance(self):
        raw=self.arc(1000.,.001)
        self.assertLess((raw['center'][0]+1000.)-raw['start'][0],TOL)
        with self.assertRaisesRegex(ValueError,'По ЧС'):Primitive(raw)

    def test_wrong_clockwise_major_arc_is_never_silently_corrected(self):
        with self.assertRaisesRegex(ValueError,'поворачивает по z'):
            Primitive(self.arc(30.,0.,clockwise=False))


if __name__=='__main__':unittest.main()
