"""The shipped synthetic examples exercise real project and fuel interfaces."""
import json
from pathlib import Path
import unittest
import numpy as np
from fuelfield.core import project_from_json,calculate_plane
from fuelfield.air_sources import resolve_sources
from fuelfield.air_balance import solve_balance,didenko_reference
from fuelfield.mixture_field import compute_mixture

ROOT=Path(__file__).resolve().parents[1]


class TestAirExamples(unittest.TestCase):
    def test_didenko_project_reproduces_book_ledger_and_integral_alpha(self):
        cfg,ns=project_from_json(json.loads((ROOT/'examples/demo_project_v08_didenko_reference.json').read_text()))
        expected=didenko_reference()['mixed_cumulative']
        for z,fraction in zip(cfg.z_mm,expected):
            plane=calculate_plane(ns,cfg,z);result=compute_mixture(plane,cfg,ns)
            self.assertAlmostEqual(plane.inside_kg_s,1/(2.6*15.),places=14)
            self.assertAlmostEqual(result.air_mixed_kg_s.sum(),fraction,places=12)
            self.assertAlmostEqual(result.alpha_section_mixed,2.6*fraction,places=12)
        self.assertTrue(all('SYNTHETIC' in n.sector_source for n in ns))

    def test_synthetic_registry_has_140_distinct_individual_inputs_and_residual(self):
        cfg,ns=project_from_json(json.loads((ROOT/'examples/demo_project_v08_air_synthetic.json').read_text()))
        sources=resolve_sources(cfg.air_model,cfg,ns);balance=solve_balance(cfg.air_model)
        swirlers=[s for s in sources if s['kind']=='swirler']
        self.assertEqual({s['nozzle_id'] for s in swirlers},set(range(1,141)))
        self.assertGreater(len({s['flow_kg_s'] for s in swirlers}),20)
        self.assertAlmostEqual(balance.unassigned_kg_s,.24,places=12)
        self.assertTrue(np.isfinite(list(balance.flows.values())).all())
        self.assertTrue(all('SYNTHETIC' in n.sector_source for n in ns))
