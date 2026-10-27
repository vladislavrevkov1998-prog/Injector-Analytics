"""Tiny physical inputs must not create Inf alpha or lose representable ratios."""
from types import SimpleNamespace
import unittest
import numpy as np
from fuelfield.core import Settings
from fuelfield.mixture_field import compute_mixture,flow_ratio


class TestMixtureNumericLimits(unittest.TestCase):
    def test_representable_ratio_survives_underflow_of_l0_times_fuel(self):
        ratio=flow_ratio(np.array([1e-300,0.]),np.array([1e-300,1e-300]),1e-300)
        self.assertTrue(np.isfinite(ratio).all())
        self.assertAlmostEqual(ratio[0]/1e300,1.,places=12)
        self.assertEqual(ratio[1],0.)

    def test_representable_ratio_survives_overflow_of_product(self):
        self.assertAlmostEqual(float(flow_ratio(1e308,1e308,1e308))/1e-308,1.,places=12)

    def test_unrepresentable_alpha_has_explicit_mask_and_area_diagnostic(self):
        cfg=Settings(air_model=dict(enabled=True,total_kg_s=1.,l0=15.,sources=[]))
        plane=SimpleNamespace(z_mm=0.,cell_mass_kg_s=np.array([[1e-320,1.]]),
                              cell_area_m2=np.array([[1.,3.]]),is_composite=False)
        air=SimpleNamespace(z_mm=0.,air_total_kg_s=np.array([[1.,0.]]),
            air_mixed_kg_s=np.zeros((1,2)),air_core_kg_s=np.array([[1.,0.]]),
            unsupported_mask=np.zeros((1,2),bool),reverse_air_kg_s=np.zeros((1,2)),
            diagnostics=(),provenance={},balance=None,source_states=[])
        result=compute_mixture(plane,cfg,[],air_result=air)
        self.assertTrue(np.isnan(result.alpha_total[0,0]))
        self.assertEqual(result.alpha_total[0,1],0.)
        self.assertEqual(result.undefined_area_fraction,.25)
        self.assertTrue(any('диапазон' in message for message in result.diagnostics))
