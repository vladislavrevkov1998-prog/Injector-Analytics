"""Acceptance tests for the independent experimental v0.2 foundation."""
import json
from pathlib import Path
import sys
import unittest
import warnings
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from fuelfield.characteristics import (ExperimentalCharacteristic, PressureOutOfRange,
    ExtrapolationWarning, flow_to_kg_h, flow_from_kg_h)


class TestExperimentalCharacteristic(unittest.TestCase):
    def test_sorting_keeps_exact_measurements_and_does_not_mutate_input(self):
        original = [[30, 24.123456789123], [2.5, 2.123456789123], [10, 6]]
        curve = ExperimentalCharacteristic(original)
        self.assertEqual(original[0][0], 30)
        self.assertEqual(curve.points, ((2.5, 2.123456789123), (10, 6.), (30, 24.123456789123)))
        for pressure, flow in original:
            result = curve.evaluate(pressure)
            self.assertEqual(result.flow_kg_h, flow)
            self.assertFalse(result.extrapolated)
        original[0][1] = 999
        self.assertEqual(curve.evaluate(30).flow_kg_h, 24.123456789123)

    def test_piecewise_interpolation_allows_nonmonotonic_measured_curve(self):
        curve = ExperimentalCharacteristic([(1, 10), (3, 6), (5, 20)])
        self.assertEqual(curve.evaluate(2).flow_kg_h, 8)
        self.assertEqual(curve.evaluate(4).flow_kg_h, 13)
        self.assertEqual(curve.points, ((1, 10), (3, 6), (5, 20)))

    def test_known_pressure_conversion_ratio_2_5_to_30(self):
        curve = ExperimentalCharacteristic([(2.5, 2), (30, 24)])
        self.assertEqual(curve.evaluate(30).flow_kg_h / curve.evaluate(2.5).flow_kg_h, 12)

    def test_invalid_points_are_rejected(self):
        cases = [[], [(0, 1)], [(-1, 1)], [(1, -1)], [(1, 2), (1, 3)],
                 [(float('nan'), 1)], [(1, float('nan'))], [(float('inf'), 1)],
                 [(1, float('inf'))]]
        for points in cases:
            with self.subTest(points=points), self.assertRaises(ValueError):
                ExperimentalCharacteristic(points)

    def test_invalid_requested_pressure_is_rejected(self):
        curve = ExperimentalCharacteristic([(1, 0), (2, 1)])
        for pressure in (0, -1, float('nan'), float('inf')):
            with self.subTest(pressure=pressure), self.assertRaises(ValueError):
                curve.evaluate(pressure)

    def test_extrapolation_requires_opt_in_and_warns_with_status(self):
        curve = ExperimentalCharacteristic([(2, 2), (4, 4)])
        for pressure in (1, 5):
            with self.assertRaises(PressureOutOfRange):curve.evaluate(pressure)
            with self.assertWarns(ExtrapolationWarning):
                result = curve.evaluate(pressure, allow_extrapolation=True)
            self.assertTrue(result.extrapolated)
            self.assertEqual(result.flow_kg_h, pressure)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            result = curve.evaluate(3)
        self.assertFalse(result.extrapolated)
        self.assertEqual(caught, [])

    def test_single_point_is_valid_but_does_not_invent_a_slope(self):
        curve = ExperimentalCharacteristic([(2.5, 5)])
        self.assertEqual(curve.evaluate(2.5).flow_kg_h, 5)
        with self.assertRaises(PressureOutOfRange):curve.evaluate(30)
        with self.assertRaises(ValueError):curve.evaluate(30, allow_extrapolation=True)

    def test_negative_extrapolation_is_not_clamped(self):
        curve = ExperimentalCharacteristic([(2, 2), (4, 1)])
        with self.assertRaises(ValueError):curve.evaluate(8, allow_extrapolation=True)
        self.assertEqual(curve.points, ((2, 2), (4, 1)))

    def test_json_roundtrip_and_wrong_units_rejected(self):
        curve = ExperimentalCharacteristic([(30, 24), (2.5, 2)])
        data = json.loads(json.dumps(curve.to_dict()))
        self.assertEqual(ExperimentalCharacteristic.from_dict(data), curve)
        for key, value in (('kind', 'theoretical'), ('flow_unit', 'l/min'), ('pressure_unit', 'Pa')):
            broken = dict(data);broken[key] = value
            with self.assertRaises(ValueError):ExperimentalCharacteristic.from_dict(broken)

    def test_arbitrary_point_count(self):
        curve = ExperimentalCharacteristic([(p, p*2) for p in range(1, 101)])
        self.assertEqual(len(curve.points), 100)
        self.assertEqual(curve.evaluate(50.5).flow_kg_h, 101)


class TestFlowUnits(unittest.TestCase):
    def test_known_conversion_and_roundtrip(self):
        self.assertEqual(flow_to_kg_h(1, 'l/min', 800), 48)
        self.assertEqual(flow_from_kg_h(48, 'l/min', 800), 1)
        for density in (600, 800, 1000):
            for value in (0, 0.123456789, 200):
                self.assertAlmostEqual(flow_from_kg_h(flow_to_kg_h(value,'l/min',density),
                                                     'l/min',density),value)
                self.assertEqual(flow_to_kg_h(value, 'kg/h', density), value)
                self.assertEqual(flow_from_kg_h(value, 'kg/h', density), value)

    def test_invalid_conversion_inputs_rejected(self):
        for convert in (flow_to_kg_h, flow_from_kg_h):
            for value, unit, density in ((-1, 'kg/h', 800), (float('nan'), 'kg/h', 800),
                (float('inf'), 'kg/h', 800), (1, 'unknown', 800), (1, 'l/min', 0),
                (1, 'l/min', -800), (1, 'l/min', float('nan')), (1, 'l/min', float('inf'))):
                with self.subTest(convert=convert.__name__, value=value, density=density), \
                     self.assertRaises(ValueError):convert(value, unit, density)


if __name__ == '__main__':unittest.main()
