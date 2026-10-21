"""Arc pairs that are not already tangent, including their outer constraints."""
from copy import deepcopy
import unittest
import numpy as np
from fuelfield.chamber import Primitive,UNITS
from fuelfield.chamber_constraints import add_constraint,add_object_tangent,apply_constraints,validate_constraints
from test_v062_constraints_math import raw_geometry,ref,lookup,point_mm


def pair(unit0='mm',unit1='mm',reverse=False):
    records=[dict(id='L1',kind='line',wall='reference',start=[-20.,300.],end=[0.,300.],unit='mm',fixed=True),
        dict(id='A1',kind='arc',wall='reference',start=[0.,300.],end=[20.,280.],center=[0.,280.],clockwise=True,unit='mm'),
        dict(id='A2',kind='arc',wall='reference',start=[20.,280.],end=[50.,260.],center=[50.,292.5],clockwise=False,unit='mm'),
        dict(id='L2',kind='line',wall='reference',start=[50.,260.],end=[70.,260.],unit='mm',fixed=True)]
    for p,unit in ((records[1],unit0),(records[2],unit1)):
        for key in ('start','end','center'):p[key]=(np.array(p[key])/UNITS[unit]).tolist()
        p['unit']=unit;p['future']={'preserve':[1,2,3]}
        if reverse:p['start'],p['end']=p['end'],p['start'];p['clockwise']=not p['clockwise']
    raw=raw_geometry(records)
    raw=add_constraint(raw,'tangent',ref('A1','end' if reverse else 'start'),ref('L1','end'))
    raw=add_constraint(raw,'tangent',ref('A2','start' if reverse else 'end'),ref('L2','start'))
    return add_constraint(raw,'coincident',ref('A1','start' if reverse else 'end'),ref('A2','end' if reverse else 'start'))


class TestBiarcConstrainedPair(unittest.TestCase):
    def test_both_outer_line_tangencies_survive_adding_arc_arc(self):
        for unit0,unit1 in (('mm','mm'),('cm','m')):
            for reverse in (False,True):
                with self.subTest(unit0=unit0,unit1=unit1,reverse=reverse):
                    raw=pair(unit0,unit1,reverse);before=deepcopy(raw)
                    result=add_object_tangent(raw,'A1','A2');validate_constraints(result)
                    self.assertEqual(raw,before);self.assertEqual(len(result['constraints']),4)
                    for identity in ('L1','L2'):self.assertEqual(lookup(result,identity),lookup(raw,identity))
                    out0='end' if reverse else 'start';out1='start' if reverse else 'end';joint0='start' if reverse else 'end';joint1='end' if reverse else 'start'
                    np.testing.assert_allclose(point_mm(lookup(result,'A1'),out0),[0.,300.],atol=1e-10)
                    np.testing.assert_allclose(point_mm(lookup(result,'A2'),out1),[50.,260.],atol=1e-10)
                    joint=point_mm(lookup(result,'A1'),joint0)
                    self.assertGreater(np.linalg.norm(joint-[20.,280.]),1.)
                    np.testing.assert_allclose(joint,point_mm(lookup(result,'A2'),joint1),atol=1e-10)
                    for identity in ('A1','A2'):
                        Primitive(lookup(result,identity));self.assertEqual(lookup(result,identity)['future'],{'preserve':[1,2,3]})
                    again=apply_constraints(result);validate_constraints(again)
                    for identity in ('A1','A2'):
                        np.testing.assert_allclose(point_mm(lookup(again,identity),'center'),point_mm(lookup(result,identity),'center'),atol=1e-9)

    def test_fixed_join_and_fixed_arc_are_not_moved_to_force_tangency(self):
        raw=pair();raw['primitives'].append(dict(id='P',kind='point',wall='reference',start=[20.,280.],unit='mm',fixed=True))
        raw=add_constraint(raw,'coincident',ref('A1','end'),ref('P','start'));before=deepcopy(raw)
        with self.assertRaisesRegex(ValueError,'касатель'):add_object_tangent(raw,'A1','A2')
        self.assertEqual(raw,before)
        raw=pair();lookup(raw,'A1')['fixed']=True;before=deepcopy(raw)
        with self.assertRaisesRegex(ValueError,'касатель'):add_object_tangent(raw,'A1','A2')
        self.assertEqual(raw,before)

    def test_extra_coincident_curve_follows_the_refitted_join(self):
        raw=pair();raw['primitives'].append(dict(id='L3',kind='line',wall='reference',start=[20.,280.],end=[25.,281.],unit='mm'))
        raw=add_constraint(raw,'coincident',ref('L3','start'),ref('A1','end'))
        result=add_object_tangent(raw,'A1','A2');validate_constraints(result)
        np.testing.assert_allclose(point_mm(lookup(result,'L3'),'start'),point_mm(lookup(result,'A1'),'end'),atol=1e-10)
        self.assertEqual(lookup(result,'L3')['end'],[25.,281.])

    def test_pair_solution_is_invariant_under_small_rotation_scale_and_translation(self):
        import math
        original=pair();solved=add_object_tangent(original,'A1','A2')
        for angle,scale in ((.1,1.),(-.1,10.)):
            transform=np.array([[math.cos(angle),-math.sin(angle)],[math.sin(angle),math.cos(angle)]])*scale
            shift=np.array([500.,500.]);raw=deepcopy(original)
            for p in raw['primitives']:
                for role in ('start','end','center'):
                    if role in p:p[role]=(transform@np.array(p[role])+shift).tolist()
            actual=add_object_tangent(raw,'A1','A2');validate_constraints(actual)
            for identity in ('A1','A2'):
                for role in ('start','end','center'):
                    expected=transform@point_mm(lookup(solved,identity),role)+shift
                    np.testing.assert_allclose(point_mm(lookup(actual,identity),role),expected,rtol=0,atol=2e-5)
