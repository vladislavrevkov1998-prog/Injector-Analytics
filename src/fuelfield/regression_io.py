"""Traceable fuel-target inputs; never infer temperature as a fuel target."""
import csv,json,hashlib
from pathlib import Path
import numpy as np
from .core import project_from_json
from .regression import RegressionTarget,QUANTITIES
from .data_io import _xlsx_sheets
from .analytics_io import write_workbook

def load_target(path,quantity='flux',cancel=None):
    path=Path(path);raw=path.read_bytes();source={'source_file':path.name,'source_sha256':hashlib.sha256(raw).hexdigest()}
    if path.suffix.lower()=='.json':
        data=json.loads(raw.decode('utf-8-sig'))
        if data.get('format')=='fuel-field-pilot-v1':
            cfg,ns=project_from_json(data);target=RegressionTarget.from_project(cfg,ns,quantity,cancel=cancel)
            return RegressionTarget(**{**vars(target),'metadata':{**target.metadata,**source}})
        if data.get('format')!='fuel-regression-target-v1':raise ValueError('JSON должен быть топливным проектом или fuel-regression-target-v1')
        rows=data['observations'];metadata={**data.get('metadata',{}),**source}
    else:
        if path.suffix.lower()=='.xlsx':
            tables=list(_xlsx_sheets(path));tables=[(name,rows) for name,rows in tables if rows and 'quantity' in rows[0] and 'value' in rows[0]]
            if not tables:raise ValueError('Excel: нужен топливный лист с quantity, unit, source_kind, z_mm, radius_mm, theta_deg, value')
            rows=[dict(zip(cells[0],r)) for name,cells in tables for r in cells[1:] if any(v is not None for v in r)]
        elif path.suffix.lower()=='.csv':
            text=raw.decode('utf-8-sig')
            try:dialect=csv.Sniffer().sniff(text[:8192],delimiters=',;\t')
            except csv.Error:dialect=csv.excel
            rows=list(csv.DictReader(text.splitlines(),dialect=dialect))
        else:raise ValueError('Топливный эталон: JSON/CSV/XLSX')
        metadata=source
    if not rows:raise ValueError('Пустой эталон')
    quantities={r.get('quantity') for r in rows};units={r.get('unit') for r in rows};kinds={r.get('source_kind',metadata.get('source_kind')) for r in rows}
    if quantities!={quantity} or units!={QUANTITIES.get(quantity)} or not kinds.issubset({'fuel','synthetic','reference_model'}):raise ValueError('Нужны явные топливные quantity/unit/source_kind; подгонка к температуре запрещена')
    def num(row,key,default=None):
        value=row.get(key,default)
        if value in ('',None):value=default
        if value is None:raise ValueError('Отсутствует '+key)
        return float(str(value).replace(',','.'))
    def valid(row):return str(row.get('valid',True)).lower() not in ('0','false','no')
    return RegressionTarget(quantity,tuple(num(r,'z_mm') for r in rows),tuple(num(r,'radius_mm') for r in rows),
        tuple(num(r,'theta_deg') for r in rows),tuple(num(r,'value',np.nan) for r in rows),QUANTITIES[quantity],
        tuple(valid(r) for r in rows),tuple(num(r,'weight',1.) for r in rows),tuple(r.get('split') or 'train' for r in rows),
        tuple(num(r,'area_m2') for r in rows) if quantity=='cell_mass' else (),{**metadata,'source_kind':next(iter(kinds))})


def write_target(path,target):
    rows=[];mask,weights,split=target.arrays()
    for i,y in enumerate(target.y):
        row=dict(quantity=target.quantity,unit=target.unit,source_kind=target.metadata['source_kind'],z_mm=target.z_mm[i],radius_mm=target.radius_mm[i],theta_deg=target.theta_deg[i],
                 value=float(y) if np.isfinite(y) else None,valid=bool(mask[i]),weight=float(weights[i]),split=str(split[i]))
        if target.area_m2:row['area_m2']=target.area_m2[i]
        rows.append(row)
    path=Path(path)
    if path.suffix=='.json':path.write_text(json.dumps(dict(format='fuel-regression-target-v1',metadata=target.metadata,observations=rows),ensure_ascii=False,indent=2,allow_nan=False))
    else:
        with path.open('w',newline='',encoding='utf-8-sig') as file:
            writer=csv.DictWriter(file,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def write_fit_xlsx(path,result):
    parameters=[['scope','key','unit','initial','min','max','fitted','delta','fixed','bound_active']]
    for p,value in zip(result.parameters,result.fitted):parameters.append([p.scope,p.key,p.unit,p.initial,p.minimum,p.maximum,value,value-p.initial,p.fixed,p.key in result.diagnostics.get('active_bounds',[])])
    metrics=[['partition','n','RMSE','MAE','R2']]
    for name,m in result.metrics.items():metrics.append([name,m['n'],m['RMSE'],m['MAE'],'N/A' if m['R2'] is None else m['R2']])
    metrics += [['cost_initial',result.initial_cost],['cost_final',result.cost],['status',result.status],['evaluations',result.evaluations],['rank',result.diagnostics['rank']],['condition',result.diagnostics['condition'] or 'N/A']]
    residuals=[['quantity','unit','source_kind','z_mm','radius_mm','theta_deg','value','initial','fitted','residual','valid','weight','split']]
    mask,weights,split=result.target.arrays()
    for i,y in enumerate(result.target.y):
        val=lambda v:float(v) if np.isfinite(v) else None
        residuals.append([result.target.quantity,result.target.unit,result.target.metadata['source_kind'],result.target.z_mm[i],result.target.radius_mm[i],result.target.theta_deg[i],val(y),val(result.initial_prediction[i]),val(result.prediction[i]),val(result.prediction[i]-y),bool(mask[i]),float(weights[i]),str(split[i])])
    metadata=[['key','value'],['snapshot_sha256',result.snapshot.sha256],['settings',json.dumps(result.settings,ensure_ascii=False)],['source',json.dumps(result.target.metadata,ensure_ascii=False)]]
    for label,payload in [('initial_project',result.snapshot.payload),('fitted_project',result.candidate_payload)]:
        metadata += [[f'{label}_{i//16000:04d}',payload[i:i+16000]] for i in range(0,len(payload),16000)]
    validation=[['status','conclusion'],['scope','fuel → fuel; температура не используется'],['diagnostics',json.dumps(result.diagnostics,ensure_ascii=False)]]+[['warning',note] for note in result.warnings]
    write_workbook(path,[('Parameters',parameters),('Metrics',metrics),('Residuals',residuals),('Metadata',metadata),('Validation',validation)])
