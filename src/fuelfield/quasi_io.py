"""Distinct portable mode files, explicit wide-table mapping, and scan reports."""
import csv,json,hashlib
from pathlib import Path
from copy import deepcopy
from dataclasses import replace
import numpy as np
from .quasi import QuasiSettings,PressureChannel,finite
from .data_io import _xlsx_sheets,write_result_csv
from .analytics_io import write_workbook,transport_tables
from .analysis import cell_values,UNITS

def mode_path(path):
    path=Path(path)
    return path if path.name.lower().endswith('.quasi.json') else path.with_name((path.stem if path.suffix.lower()=='.json' else path.name)+'.quasi.json')

def load_settings(path):
    raw=json.loads(Path(path).read_text(encoding='utf-8-sig'));mode=QuasiSettings.from_dict(raw);mode.validate();bindings={}
    for key,value in raw.get('nozzle_manifolds',{}).items():
        if not str(key).isdigit() or not 1<=int(key)<=140 or not isinstance(value,str) or not value.strip():raise ValueError('В JSON неверная привязка форсунки к коллектору')
        bindings[str(int(key))]=str(value)
    return mode,bindings

def save_settings(path,mode,bindings=None):
    mode.validate();path=mode_path(path);path.write_text(json.dumps(mode.to_dict(bindings),ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8');return path

def read_pressure_tables(path):
    path=Path(path);blob=path.read_bytes()
    if path.suffix.lower()=='.xlsx':sheets=list(_xlsx_sheets(path))
    elif path.suffix.lower()=='.csv':
        text=blob.decode('utf-8-sig')
        try:dialect=csv.Sniffer().sniff(text[:8192],delimiters=',;\t')
        except csv.Error:dialect=csv.excel
        sheets=[(path.stem,list(csv.reader(text.splitlines(),dialect)))]
    else:raise ValueError('Нужна характеристика в XLSX или CSV')
    tables=[]
    for name,rows in sheets:
        if not rows or len(rows)<2:continue
        headers=[str(v or '').strip() for v in rows[0]]
        if len(set(headers))!=len(headers):raise ValueError('Заголовки столбцов должны быть уникальны')
        tables.append(dict(name=name,headers=headers,rows=[r for r in rows[1:] if any(v is not None and str(v).strip() for v in r)],source=dict(file_name=path.name,sha256=hashlib.sha256(blob).hexdigest(),raw_rows=deepcopy(rows))))
    if not tables:raise ValueError('В файле нет листа с заголовками и данными')
    return tables

def import_pressure_table(mode,table,mappings,time_column=None,time_scale_s=1.,adopt_time=True):
    rows=table['rows'];headers=table['headers']
    if len(rows)!=mode.n_steps:raise ValueError(f'В таблице {len(rows)} шагов, требуется {mode.n_steps}; строки не дополняются и не обрезаются')
    def values(column,label):
        if column not in headers:raise ValueError('Не выбран столбец '+label)
        index=headers.index(column);output=[]
        for step,row in enumerate(rows):
            value=row[index] if index<len(row) else None
            output.append(finite(str(value).replace(',','.'),f'{label}, шаг {step}'))
        return tuple(output)
    t0,dt=mode.t_start_s,mode.dt_s
    if time_column is not None:
        times=np.asarray(values(time_column,'Время'))*finite(time_scale_s,'Единица времени');delta=np.diff(times)
        if delta[0]<=0 or not np.allclose(delta,delta[0],rtol=1e-8,atol=1e-8):raise ValueError('Временной шаг должен быть постоянным и положительным; повторы/неравномерные времена не интерполируются')
        if adopt_time:t0,dt=float(times[0]),float(delta[0])
        elif not np.allclose(times,mode.times,rtol=0,atol=1e-7):raise ValueError('Времена файла не совпадают с началом/постоянным шагом режима')
    channels=[];used=False
    for channel in mode.channels:
        mapping=mappings.get(channel.id,{})
        if mapping.get('pressure') is None:channels.append(channel);continue
        used=True;p=values(mapping['pressure'],channel.id+' давление');gas=() if channel.kind=='delta_p' else values(mapping.get('gas_pressure'),channel.id+' газ')
        channels.append(replace(channel,pressure=p,gas_pressure=gas))
    if not used:raise ValueError('Выберите хотя бы один столбец давления коллектора')
    metadata=deepcopy(mode.metadata);metadata.setdefault('imports',[]).append(dict(source=deepcopy(table['source']),mapping=deepcopy(mappings),time_column=time_column,time_scale_s=time_scale_s,adopt_time=adopt_time))
    result=replace(mode,t_start_s=t0,dt_s=dt,channels=tuple(channels),metadata=metadata)
    # Unmapped collectors may still be empty in a draft. Each imported physical channel must be valid.
    for channel in result.channels:
        if mappings.get(channel.id,{}).get('pressure') is not None:channel.delta_p(mode.n_steps)
    return result

def report_tables(result,metric='flux',interval=180.):
    tables=[];cfg=result.cfg
    confined=(cfg.fuel_transport or {}).get('mode','legacy')=='confined_inertial'
    for index,p in enumerate(result.composites):
        header=['angle_deg',*[f'R={r:.12g} мм [{UNITS[metric]}]'+(' total incl. film' if confined else '') for r in p.radial_centers_mm],'scan_step','time_s',*[f'delta_p_{key}_kgf_cm2' for key in result.pressures]];rows=[header];values=cell_values(p,metric,cfg.density_kg_m3,interval)
        for col,angle in enumerate(p.theta_centers_deg):rows.append([float(angle),*[None if np.ma.is_masked(v) else float(v) for v in values[:,col]],int(p.scan_steps[col]),float(p.scan_times_s[col]),*[p.scan_pressures[col][key] for key in result.pressures]])
        tables.append((f'z{index+1} {p.z_mm:g} мм'[:31],rows))
    flows=[['step','time_s','scan_angle_deg',*[f'delta_p_{key}_kgf_cm2' for key in result.pressures],*[f'G_{n.id}_kg_s' for n in result.nozzles]]]
    for step in range(result.settings.n_steps):flows.append([step,float(result.settings.times[step]),float(result.settings.angles[step]),*[float(v[step]) for v in result.pressures.values()],*[float(v) for v in result.flows[step]]])
    balances=[list(result.balances[0])]+[list(b.values()) for b in result.balances]
    metadata=[['record','part','json']]
    for name,obj in [('manifest',result.manifest()),('project',json.loads(result.project_payload))]:
        raw=json.dumps(obj,ensure_ascii=False,allow_nan=False,separators=(',',':'))
        metadata.extend([name,i//16000,raw[i:i+16000]] for i in range(0,len(raw),16000))
    return [*tables,*transport_tables(result.composites),('Pressure_flows',flows),('Instant_balances',balances),('Metadata',metadata)]

def write_report(path,result,metric='flux',interval=180.):
    path=Path(path);tables=report_tables(result,metric,interval)
    if path.suffix.lower()=='.xlsx':write_workbook(path,tables);return [path]
    if path.suffix.lower()!='.csv':raise ValueError('Отчёт: CSV или XLSX')
    write_result_csv(path,result.composites,cfg=result.cfg);paths=[path]
    if any(getattr(p,'suspended_cell_mass_kg_s',None) is not None for p in result.composites):
        paths.extend(path.with_name(path.stem+suffix) for suffix in ('_Transport_balance.csv','_Fuel_momentum_N.csv','_transport.json'))
    for name,rows in tables[-3:-1]:
        target=path.with_name(path.stem+'_'+name+'.csv')
        with target.open('w',encoding='utf-8-sig',newline='') as f:csv.writer(f).writerows(rows)
        paths.append(target)
    manifest=path.with_name(path.stem+'_quasi_manifest.json');manifest.write_text(json.dumps(dict(manifest=result.manifest(),project=json.loads(result.project_payload)),ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8');paths.append(manifest)
    return paths
