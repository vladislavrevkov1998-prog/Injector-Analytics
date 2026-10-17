import sys,tempfile,json,unittest,csv
from pathlib import Path
from dataclasses import replace
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from fuelfield.quasi import PressureChannel,calculate_quasi
from fuelfield.quasi_io import save_settings,load_settings,mode_path,read_pressure_tables,import_pressure_table,write_report
from fuelfield.analytics_io import write_workbook
from fuelfield.data_io import _xlsx_sheets
from test_quasi import case,ROOT

class TestQuasiIO(unittest.TestCase):
 def test_mode_tag_distinct_format_bindings_and_roundtrip(self):
  cfg,ns,mode=case()
  with tempfile.TemporaryDirectory() as folder:
   path=save_settings(Path(folder)/'settings.json',mode,{'1':'A','71':'B'});self.assertEqual(path.name,'settings.quasi.json');a,bindings=load_settings(path);self.assertEqual(a.to_dict(),mode.to_dict());self.assertEqual(bindings,{'1':'A','71':'B'})
   bad=Path(folder)/'project.json';bad.write_text((ROOT/'examples/demo_project_user_field.json').read_text())
   with self.assertRaises(ValueError):load_settings(bad)
 def test_real_xlsx_mapping_constant_time_and_provenance(self):
  cfg,ns,mode=case();draft=replace(mode,channels=(replace(mode.channels[0],pressure=(None,)*8),mode.channels[1]))
  with tempfile.TemporaryDirectory() as folder:
   p=Path(folder)/'pressure.xlsx';rows=[['milliseconds','fuel A','fuel B']]+[[i*500,10+i,20-i] for i in range(8)];write_workbook(p,[('measured',rows)]);tables=read_pressure_tables(p)
   imported=import_pressure_table(draft,tables[0],{'A':{'pressure':'fuel A'},'B':{'pressure':'fuel B'}},'milliseconds',.001)
   self.assertEqual(imported.dt_s,.5);self.assertEqual(imported.channels[0].pressure,tuple(range(10,18)));self.assertEqual(imported.metadata['imports'][0]['source']['raw_rows'][1][1],'10')
   self.assertEqual(len(imported.metadata['imports'][0]['source']['sha256']),64);self.assertEqual(draft.channels[0].pressure,(None,)*8)
 def test_nonuniform_missing_duplicate_time_and_wrong_count_rejected(self):
  cfg,ns,mode=case();table={'headers':['t','p'],'rows':[[i,10] for i in range(8)],'source':{}}
  for times in ([0,1,2,3,4,5,6,8],[0,1,2,3,4,5,6,6]):
   table['rows']=[[t,10] for t in times]
   with self.assertRaises(ValueError):import_pressure_table(mode,table,{'A':{'pressure':'p'}},'t')
  table['rows']=[[i,10] for i in range(7)]
  with self.assertRaises(ValueError):import_pressure_table(mode,table,{'A':{'pressure':'p'}})
 def test_gas_column_units_import_and_time_grid_not_silently_changed(self):
  cfg,ns,mode=case();mode=replace(mode,channels=(PressureChannel('A',(None,)*8,'Pa','p_fuel_abs',(None,)*8,'Pa'),))
  table={'headers':['t','fuel','gas'],'rows':[[i,1980665,1e6] for i in range(8)],'source':{}}
  a=import_pressure_table(mode,table,{'A':dict(pressure='fuel',gas_pressure='gas')},'t');np.testing.assert_allclose(a.channels[0].delta_p(8),10)
  with self.assertRaises(ValueError):import_pressure_table(mode,table,{'A':dict(pressure='fuel',gas_pressure='gas')},'t',adopt_time=False)
 def test_xlsx_and_csv_matrix_time_and_audit_exports(self):
  cfg,ns,mode=case();out=calculate_quasi(ns,cfg,preview=False)
  with tempfile.TemporaryDirectory() as folder:
   p=Path(folder)/'out.xlsx';write_report(p,out);sheets=dict(_xlsx_sheets(p));self.assertEqual(len(sheets['z1 0 мм']),9);self.assertEqual(len(sheets['Pressure_flows'][0]),145);self.assertEqual(len(sheets['Instant_balances']),17)
   p=Path(folder)/'out.csv';paths=write_report(p,out);self.assertEqual(len(paths),4)
   with p.open(encoding='utf-8-sig') as f:rows=list(csv.reader(f,delimiter=';'))
   self.assertIn('time_s',rows[0]);self.assertEqual(rows[1][-4],'quasistationary');self.assertEqual(json.loads(rows[1][-1])['A'],10)
   manifest=json.loads(paths[-1].read_text());self.assertIsNone(manifest['manifest']['composite_global_balance']);self.assertEqual(len(manifest['project']['nozzles']),140)

if __name__=='__main__':unittest.main()
