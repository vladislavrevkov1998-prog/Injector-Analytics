"""Lossless raw-table capture, explicit import mappings, and traceable session reports."""
import csv,hashlib,json
from pathlib import Path
from copy import deepcopy
from .data_io import _xlsx_sheets
from .analytics_io import write_workbook
from .measurement import MeasurementSession,MeasurementStep,PressureSeries,PRESSURE_FACTORS,finite


def read_tables(path):
    path=Path(path);blob=path.read_bytes()
    if path.suffix.lower()=='.xlsx':tables=list(_xlsx_sheets(path))
    elif path.suffix.lower()=='.csv':
        text=blob.decode('utf-8-sig')
        try:dialect=csv.Sniffer().sniff(text[:8192],delimiters=',;\t')
        except csv.Error:dialect=csv.excel
        tables=[(path.stem,list(csv.reader(text.splitlines(),dialect)))]
    else:raise ValueError('Ожидается CSV или XLSX')
    output=[dict(name=name,headers=[str(c or '').strip() for c in rows[0]],rows=[list(r) for r in rows[1:]],source=dict(file_name=path.name,sha256=hashlib.sha256(blob).hexdigest(),raw_rows=deepcopy(rows))) for name,rows in tables if rows and len(rows)>1]
    if not output:raise ValueError('В таблице нет заголовков и строк данных')
    return output

def cell(row,headers,key,mapping,default=None):
    column=mapping.get(key,key)
    if column is None:return default
    try:index=headers.index(column) if isinstance(column,str) else int(column)
    except ValueError:return default
    value=row[index] if index<len(row) else None
    return default if value is None or str(value).strip()=='' else value

def number(value,label):return finite(str(value).strip().replace(',','.'),label)

def boolean(value,default=True):
    if value is None:return default
    if isinstance(value,bool):return value
    value=str(value).strip().lower()
    if value in ('1','true','yes','да','valid','ok'):return True
    if value in ('0','false','no','нет','invalid','bad'):return False
    raise ValueError(f'Неизвестный флаг качества: {value}')

def import_steps(table,mapping=None,duration_s=None,time_scale_s=1.):
    mapping=mapping or {};steps=[];headers=table['headers']
    for row in table['rows']:
        if not any(v is not None and str(v).strip() for v in row):continue
        ident=number(cell(row,headers,'step_id',mapping,len(steps)),'Индекс остановки');start=number(cell(row,headers,'t_start_s',mapping),'Время начала')*number(time_scale_s,'Масштаб времени')
        end=cell(row,headers,'t_end_s',mapping);metadata={}
        if end is None:
            if duration_s is None or duration_s==0:end=start;metadata['sampling_method']='instantaneous'
            else:end=start+number(duration_s,'Длительность окна');metadata['sampling_method']='explicit_duration'
        else:end=number(end,'Время окончания')*number(time_scale_s,'Масштаб времени')
        angle=cell(row,headers,'theta_deg',mapping);temperatures=[];valid=[]
        for channel in range(1,7):
            value=cell(row,headers,f'T_R{channel}_C',mapping)
            temperatures.append(None if value is None else number(value,f'T_R{channel}_C'))
            valid.append(boolean(cell(row,headers,f'valid_R{channel}',mapping),value is not None))
        steps.append(MeasurementStep(int(ident) if ident.is_integer() else ident,start,end,tuple(temperatures),None if angle is None else number(angle,'Угол'),tuple(valid),str(cell(row,headers,'note',mapping,'')),metadata))
    if not steps:raise ValueError('Нет строк остановок')
    return tuple(steps)

def import_pressures(table,mapping=None,unit=None,pressure_kind=None,manifold_id='A',reference_id='',time_offset_s=0.,time_scale_s=1.):
    mapping=mapping or {};headers=table['headers'];groups={}
    for row in table['rows']:
        if not any(v is not None and str(v).strip() for v in row):continue
        group=str(cell(row,headers,'manifold_id',mapping,manifold_id));kind=cell(row,headers,'pressure_kind',mapping,pressure_kind);row_unit=cell(row,headers,'unit',mapping,unit)
        if kind is None or row_unit not in PRESSURE_FACTORS:raise ValueError('Для давления нужны явные единицы и физический вид (Δp / абсолютное / избыточное)')
        reference=str(cell(row,headers,'reference_id',mapping,reference_id));key=(group,kind,reference)
        values=groups.setdefault(key,dict(times=[],pressures=[],valid=[],units=[]))
        timestamp=number(cell(row,headers,'t_s',mapping),'Время давления')*number(time_scale_s,'Масштаб времени');p=cell(row,headers,'pressure',mapping)
        valid=boolean(cell(row,headers,'quality',mapping),p is not None)
        values['times'].append(timestamp);values['pressures'].append(None if p is None else number(p,'Давление')*PRESSURE_FACTORS[row_unit]);values['valid'].append(valid);values['units'].append(row_unit)
    output=[]
    for (group,kind,reference),v in groups.items():
        output.append(PressureSeries(group,tuple(v['times']),tuple(v['pressures']),'kgf/cm2',kind,tuple(v['valid']),reference,dict(time_offset_s=number(time_offset_s,'Сдвиг часов'),source_units=v['units'],source=deepcopy(table['source']))))
    if not output:raise ValueError('Нет строк давления')
    return tuple(output)

def load_session(path):
    data=json.loads(Path(path).read_text(encoding='utf-8-sig'))
    if data.get('format')=='fuel-field-pilot-v1':data=data.get('measurement_session')
    if not isinstance(data,dict):raise ValueError('JSON не содержит измерительную сессию')
    return MeasurementSession.from_dict(data)

def save_session(path,session):Path(path).write_text(json.dumps(session.to_dict(),ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')

def chunks(label,data):
    text=json.dumps(data,ensure_ascii=False,separators=(',',':'),allow_nan=False)
    return [[label,i//16000,text[i:i+16000]] for i in range(0,len(text),16000)]

def report_tables(result):
    s=result.session;op=result.operator;measurements=[['step_id','channel','radius_measured_mm','theta_session_deg','theta_global_deg','t_start_s','t_end_s','sampling_method','T_measured_C','valid_T','q_corrected_kg_m2_s','control_r_min_mm','control_r_max_mm','control_theta_center_global_deg','control_area_m2','flow_row_id']]
    for i,step in enumerate(s.steps):
        for r,radius in enumerate(s.radii_mm):
            cell_index=int(result.cells[r,i]);ri,ai=divmod(cell_index,op.cfg.n_angular)
            measurements.append([step.step_id,r+1,radius,s.angle(step),float(result.global_angles[i]),step.t_start_s,step.t_end_s,step.metadata.get('sampling_method','window'),step.temperatures_C[r],bool(step.valid[r]),float(result.q_corrected[r,i]),float(op.radial_edges[ri]),float(op.radial_edges[ri+1]),float(op.angles[ai]),float(op.area_m2[ri,ai]),i])
    flows=[['flow_row_id','step_id','t_start_s','t_end_s',*[f'G_{j}_kg_s' for j in op.nozzle_ids]]]
    for i,step in enumerate(s.steps):flows.append([i,step.step_id,step.t_start_s,step.t_end_s,*[float(v) for v in result.mean_flows[i]]])
    pressure=[['manifold_id','t_start_s','t_end_s','min_delta_p_kgf_cm2','max_delta_p_kgf_cm2','mean_delta_p_kgf_cm2']]
    for record in result.pressure_windows:pressure.append([record[k] for k in ('manifold_id','t_start_s','t_end_s','min_delta_p','max_delta_p','mean_delta_p')])
    balances=[['t_s','injected_kg_s','inside_kg_s','inner_kg_s','outer_kg_s','relative_error']]
    for record in result.balances:balances.append([record[k] for k in balances[0]])
    raw_pressure=[['manifold_id','pressure_kind','t_raw_s','t_aligned_s','pressure','unit','valid','reference_id']]
    for series in (*s.pressure_series,*s.gas_series):
        for t,aligned,p,valid in zip(series.t_s,series._times,series.pressure,series.valid):raw_pressure.append([series.manifold_id,series.pressure_kind,t,float(aligned),p,series.unit,bool(valid),series.reference_id])
    metadata=[['record','part','json'],*chunks('manifest',result.manifest()),*chunks('project',json.loads(result.project_payload)),*chunks('raw_session',s.to_dict())]
    tables=[('Measurements',measurements),('Flows',flows),('Pressure_windows',pressure),('Instant_balances',balances),('Raw_pressure',raw_pressure),('Metadata',metadata)]
    if s.transport_delay_s or s.nozzle_delays_s or s.point_delays_s:
        points=[['step_id','channel',*[f'G_{j}_kg_s' for j in op.nozzle_ids],*[f'delay_{j}_s' for j in op.nozzle_ids]]]
        for i,step in enumerate(s.steps):
            for r in range(6):points.append([step.step_id,r+1,*[float(v) for v in result.point_flows[r,i]],*[s.delay(step,r,n) for n in result.nozzles]])
        tables.append(('Delayed_flows',points))
    return tables

def write_report(path,result):
    path=Path(path);tables=report_tables(result)
    if path.suffix.lower()=='.xlsx':write_workbook(path,tables);return [path]
    if path.suffix.lower()!='.csv':raise ValueError('Отчёт: CSV или XLSX')
    output=[]
    for index,(name,rows) in enumerate(tables):
        target=path if index==0 else path.with_name(path.stem+'_'+name+'.csv')
        with target.open('w',encoding='utf-8-sig',newline='') as stream:csv.writer(stream).writerows(rows)
        output.append(target)
    manifest=path.with_name(path.stem+'_manifest.json');manifest.write_text(json.dumps(result.manifest(),ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8');output.append(manifest)
    source=path.with_name(path.stem+'_session.json');save_session(source,result.session);output.append(source)
    return output
