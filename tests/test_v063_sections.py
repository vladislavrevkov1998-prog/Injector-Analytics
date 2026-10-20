"""A section must recognize geometry already drawn past the old outlet."""
from copy import deepcopy
import unittest

import numpy as np

from fuelfield.chamber import Chamber
from fuelfield.chamber_sections import sync_section_outlet
from test_v062_sections import straight, line, curved, record


class TestDrawnProfileOutlet(unittest.TestCase):
    def test_completed_profiles_are_not_extended_or_changed(self):
        raw=straight(160.,fixed=True)
        raw['outlet_z_mm']=60.
        raw['sections'][-1]['z_mm']=60.
        raw['sections'].append(dict(id='S5',z_mm=160.,is_outlet=False,
                                   future={'keep':'manually placed'}))
        original=deepcopy(raw)
        result=sync_section_outlet(raw)
        self.assertEqual(raw,original)
        self.assertEqual(result['primitives'],original['primitives'])
        self.assertEqual(result['constraints'],original['constraints'])
        self.assertEqual(result['outlet_z_mm'],160.)
        self.assertEqual([s['id'] for s in result['sections'] if s['is_outlet']],['S5'])
        self.assertEqual(result['sections'][-1]['future'],{'keep':'manually placed'})
        Chamber(result)

    def test_completed_arc_end_with_vertical_tangent_needs_no_extension(self):
        raw=curved(fixed=True)
        arc=record(raw,'user-arc');arc['end']=[50.,200.]
        record(raw,'user-outer')['end'][0]=.05
        raw['outlet_z_mm']=20.
        raw['sections'][-1]['z_mm']=20.
        raw['sections'].append(dict(id='S5',z_mm=50.,is_outlet=False))
        before=deepcopy(raw)
        result=sync_section_outlet(raw)
        self.assertEqual(result['primitives'],before['primitives'])
        self.assertEqual(raw,before)
        self.assertEqual(result['outlet_z_mm'],50.)
        Chamber(result)

    def test_only_shorter_wall_is_extended_from_its_actual_terminal(self):
        for unit in ('mm','cm','m'):
            for reversed_ends in (False,True):
                with self.subTest(unit=unit,reversed=reversed_ends):
                    raw=straight(40.,fixed=True)
                    continuation=line('manual-inner','inner',[40.,210.],[60.,212.],unit,fixed=True)
                    if reversed_ends:continuation['start'],continuation['end']=continuation['end'],continuation['start']
                    raw['primitives'].append(continuation)
                    record(raw,'old-outer')['end']=[80.,340.]
                    raw['sections'].append(dict(id='S5',z_mm=80.,is_outlet=False))
                    before=deepcopy(raw)
                    result=sync_section_outlet(raw)
                    self.assertEqual(raw,before)
                    self.assertEqual(result['primitives'][:4],before['primitives'])
                    self.assertEqual(len(result['primitives']),5)
                    new=result['primitives'][-1]
                    self.assertEqual(new['wall'],'inner')
                    self.assertEqual(new['provenance']['from_primitive_id'],'manual-inner')
                    self.assertEqual(new['provenance']['previous_outlet_mm'],40.)
                    np.testing.assert_allclose(new['start'],[60.,212.],atol=1e-12,rtol=0)
                    np.testing.assert_allclose(new['end'],[80.,214.],atol=1e-12,rtol=0)
                    Chamber(result)

    def test_real_gap_overlap_and_discontinuity_still_rejected_atomically(self):
        for a,b in (([41.,210.],[60.,210.]),([39.,210.],[60.,210.]),
                    ([40.,211.],[60.,211.])):
            with self.subTest(start=a):
                raw=straight()
                raw['primitives'].append(line('manual-inner','inner',a,b))
                raw['sections'].append(dict(id='S5',z_mm=60.,is_outlet=False))
                before=deepcopy(raw)
                with self.assertRaisesRegex(ValueError,'разрыв/перекрытие'):
                    sync_section_outlet(raw)
                self.assertEqual(raw,before)

    def test_does_not_silently_trim_hand_drawn_geometry_beyond_new_section(self):
        raw=straight(80.)
        raw['outlet_z_mm']=40.;raw['sections'][-1]['z_mm']=40.
        raw['sections'].append(dict(id='S5',z_mm=60.,is_outlet=False))
        before=deepcopy(raw)
        with self.assertRaisesRegex(ValueError,'дальше последнего сечения'):
            sync_section_outlet(raw)
        self.assertEqual(raw,before)

    def test_numeric_section_coordinates_are_normalized_when_marking_outlet(self):
        raw=straight(80.)
        raw['outlet_z_mm']=40.;raw['sections'][-1]['z_mm']='40'
        raw['sections'].append(dict(id='S5',z_mm='80',is_outlet=False))
        before=deepcopy(raw)
        result=sync_section_outlet(raw)
        self.assertEqual(raw,before)
        self.assertEqual(result['outlet_z_mm'],80.)
        self.assertEqual(result['sections'][-1]['z_mm'],'80')
        self.assertEqual([s['id'] for s in result['sections'] if s['is_outlet']],['S5'])


if __name__=='__main__':unittest.main()
