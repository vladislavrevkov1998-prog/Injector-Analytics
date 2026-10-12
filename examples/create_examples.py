"""Regenerate the example files from the user-supplied measurements (local only)."""
from pathlib import Path
import json, sys, csv
project=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(project/'src'))
from fuelfield.core import Settings, default_nozzles, project_to_json
from fuelfield.data_io import import_sector_file
source=Path(sys.argv[1])
measurements,sheet=import_sector_file(str(source))
with (project/'examples'/'measurements_35.csv').open('w',encoding='utf-8-sig',newline='') as f:
    writer=csv.writer(f,delimiter=';')
    writer.writerow(['ID']+list(range(1,37)))
    for i,parts in sorted(measurements.items()):writer.writerow([i]+parts)
settings=Settings()
nozzles=default_nozzles(settings)
for n in nozzles:
    n.flow_kg_h=10.0   # EXPLICIT DEMONSTRATION ONLY
    if n.id in measurements:
        n.sector_ml=measurements[n.id]
        n.sector_source='пользовательский пример: 35 эпюр'
    else:
        n.sector_source='ДЕМО: равномерная эпюра (нет измерений)'
path=project/'examples'/'demo_project_140.json'
path.write_text(json.dumps(project_to_json(settings,nozzles),indent=2,ensure_ascii=False),encoding='utf-8')
print('Imported',len(measurements),'measurements from',sheet,'and saved',path)
