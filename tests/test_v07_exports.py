"""Round-trip phase, settings and momentum reports without hiding film as spray."""
import csv
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from fuelfield.core import Settings,default_nozzles,calculate_plane
from fuelfield.transport_settings import synthetic_settings
from fuelfield.analytics_io import write_epures_xlsx,transport_tables
from fuelfield.data_io import _xlsx_sheets,write_result_csv
from fuelfield.regression import (RegressionSnapshot,RegressionTarget,ParameterSpec,
    active_parameters,fit,predict_snapshot)


def fixture():
    cfg=Settings(r_in_mm=100.,r_out_mm=160.,row_inner_mm=120.,row_outer_mm=150.,
                 z_mm=[80.],n_angular=8,n_radial=3)
    cfg.fuel_transport=synthetic_settings().to_dict()
    cfg.fuel_transport.update(emission_radial_samples=1,emission_angular_samples=36,max_step_mm=4.)
    ns=default_nozzles(cfg)
    for n in ns:
        n.enabled=n.id==1;n.flow_kg_h=3.6;n.alpha_deg=80.
    return cfg,ns


class TestTransportExports(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg,cls.ns=fixture();cls.plane=calculate_plane(cls.ns,cls.cfg,80.)

    def test_phase_csv_and_settings_companion_round_trip(self):
        p=self.plane
        self.assertGreater(p.film_outer_cell_mass_kg_s.sum(),0.)
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'matrix.csv';write_result_csv(path,[p],cfg=self.cfg)
            with path.open(encoding='utf-8-sig') as file:rows=list(csv.DictReader(file,delimiter=';'))
            for row in rows:
                self.assertAlmostEqual(float(row['mass_flow_kg_s']),sum(float(row[k]) for k in
                    ('suspended_kg_s','film_inner_kg_s','film_outer_kg_s')),delta=1e-14)
            self.assertAlmostEqual(sum(float(row['film_outer_kg_s']) for row in rows),p.film_outer_cell_mass_kg_s.sum(),places=15)
            metadata=json.loads(path.with_name('matrix_transport.json').read_text())
            self.assertEqual(metadata['settings']['fuel_transport'],self.cfg.fuel_transport)
            self.assertIn('not suspended volumetric density',metadata['film_flux'])
            self.assertIn('component order z,x,y',metadata['momentum_units'])
            with path.with_name('matrix_Fuel_momentum_N.csv').open(encoding='utf-8-sig') as file:momentum=list(csv.DictReader(file))
            self.assertEqual([r['component'] for r in momentum],['z','x','y'])
            for row in momentum:
                residual=float(row['outgoing_N'])-float(row['incoming_N'])-float(row['carrier_N'])-float(row['wall_and_film_driving_N'])
                self.assertAlmostEqual(residual,float(row['residual_N']),delta=1e-15)

    def test_xlsx_total_label_phase_cells_and_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'epures.xlsx';write_epures_xlsx(path,[self.plane],self.cfg)
            sheets=dict(_xlsx_sheets(path))
            self.assertTrue({'Fuel_phases','Transport_balance','Fuel_momentum_N','Metadata'}.issubset(sheets))
            self.assertIn('total incl. film',sheets['z1 80 мм'][0][1])
            rows=sheets['Fuel_phases'];headers=rows[0]
            total=sum(float(r[headers.index('total_including_film_kg_s')]) for r in rows[1:])
            self.assertAlmostEqual(total,self.plane.injected_kg_s,places=15)
            settings=json.loads(''.join(row[2] for row in sheets['Metadata'][1:] if row[0]=='settings'))
            self.assertEqual(settings['fuel_transport'],self.cfg.fuel_transport)

    def test_composite_has_phase_cells_but_no_global_momentum_claim(self):
        p=replace(self.plane,is_composite=True,injected_kg_s=float('nan'))
        tables=dict(transport_tables([p]));balance=tables['Transport_balance'][1]
        self.assertIn('different_times',balance[1]);self.assertIsNone(balance[2]);self.assertIsNone(balance[7])
        momentum=tables['Fuel_momentum_N'][1]
        self.assertEqual(momentum[2],'not_defined');self.assertTrue(all(v is None for v in momentum[3:]))

    def test_legacy_csv_structure_remains_unchanged(self):
        cfg=deepcopy(self.cfg);cfg.fuel_transport={'mode':'legacy'}
        p=calculate_plane(self.ns,cfg,80.)
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'legacy.csv';write_result_csv(path,[p],cfg=cfg)
            with path.open(encoding='utf-8-sig') as file:header=next(csv.reader(file,delimiter=';'))
            self.assertEqual(header,['z_mm','radial_level','theta_node','radius_center_mm','theta_deg','cell_area_m2','mass_flow_kg_s','flux_kg_m2_s'])
            self.assertFalse(path.with_name('legacy_transport.json').exists())

    def test_legacy_identification_rejected_before_optimizer_or_target_access(self):
        target=RegressionTarget('flux',(),(),(),(),'кг/(м²·с)',metadata={'source_kind':'synthetic'})
        snapshot=RegressionSnapshot.capture(self.cfg,self.ns)
        self.assertEqual(active_parameters(self.cfg,self.ns,target,'uncooled'),[])
        for operation in (lambda:fit(snapshot,target,[ParameterSpec('h0_mm','uncooled',4.,1.,8.)]),
                          lambda:predict_snapshot(snapshot,target)):
            with self.assertRaisesRegex(ValueError,'Идентификация прежних параметров факела недоступна'):
                operation()

    def test_quasi_reports_preserve_per_step_phase_and_force_ledgers(self):
        from test_quasi import case
        from fuelfield.quasi import calculate_quasi
        from fuelfield.quasi_io import write_report
        cfg,ns,mode=case(constant=True)
        cfg.fuel_transport=synthetic_settings().to_dict()
        cfg.fuel_transport.update(emission_radial_samples=1,emission_angular_samples=36,max_step_mm=4.)
        for n in ns:n.alpha_deg=80.
        result=calculate_quasi(ns,cfg,preview=False)
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'quasi.xlsx';write_report(path,result)
            sheets=dict(_xlsx_sheets(path));headers=sheets['Instant_balances'][0]
            self.assertIn('film_outer_kg_s',headers);self.assertIn('incoming_z_N',headers)
            for row in sheets['Instant_balances'][1:]:
                get=lambda key:float(row[headers.index(key)])
                self.assertAlmostEqual(get('injected_kg_s'),get('suspended_kg_s')+get('film_inner_kg_s')+get('film_outer_kg_s'),delta=1e-14)
                self.assertAlmostEqual(get('outgoing_z_N')-get('incoming_z_N')-get('carrier_z_N')-get('wall_z_N'),get('residual_z_N'),delta=1e-14)
            path=Path(directory)/'quasi.csv';paths=write_report(path,result)
            self.assertEqual(len(paths),7);self.assertTrue(all(p.is_file() for p in paths))
            manifest=json.loads(path.with_name('quasi_transport.json').read_text())
            self.assertIsNone(manifest['composite_global_balance'])
            self.assertEqual(manifest['settings']['fuel_transport'],cfg.fuel_transport)


if __name__=='__main__':unittest.main()
