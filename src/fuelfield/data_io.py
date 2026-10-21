"""Read common Excel/CSV sector tables without an Excel runtime dependency.

XLSX files are OpenXML zip archives. We need numeric cached values only and
never modify the user's workbook. Supports shared strings and inline strings.
"""
from __future__ import annotations
from pathlib import Path
import csv, io, re, zipfile
from xml.etree import ElementTree as ET

SS = '{http://schemas.openxmlformats.org/spreadsheetml/2006/main}'
REL = '{http://schemas.openxmlformats.org/officeDocument/2006/relationships}'
PR = '{http://schemas.openxmlformats.org/package/2006/relationships}'

def _col_idx(cell_address: str):
    letters = re.match(r'[A-Z]+', cell_address)
    value = 0
    for c in letters.group():
        value = 26*value + ord(c)-64
    return value-1

def _num(value):
    if value is None or str(value).strip() == '':
        return None
    try:
        return float(str(value).strip().replace(' ', '').replace(',', '.'))
    except ValueError:
        return None

def _extract_rows(cells: list[list]):
    """Look for a row with 36 consecutive numbered sector headers, fallback
    to sector fields in ID,sector1..sector36 CSV files.
    """
    for i, row in enumerate(cells[:40]):
        for col in range(max(0,len(row)-35)):
            if all(_num(row[col+k]) == k+1 for k in range(36)):
                # ID lives in first col for original spreadsheet; if header
                # first 36 cols, data starts after ID and col shift can be zero.
                id_col = 0 if col > 0 else None
                out = {}
                for values in cells[i+1:]:
                    if not values: continue
                    v = values[id_col] if id_col is not None and id_col < len(values) else None
                    if v is None: continue
                    num = _num(v)
                    if num is None or int(num) != num or int(num)<1: continue
                    if len(values) < col+36: continue
                    parts = [_num(x) for x in values[col:col+36]]
                    if any(x is None for x in parts) or min(parts)<0: continue
                    out[int(num)] = parts
                if out: return out
    # Common CSV: ID, v1,...,v36; supports a leading text header.
    out = {}
    for row in cells:
        if len(row) < 37: continue
        num = _num(row[0])
        if num is None or num<1 or num != int(num): continue
        parts = [_num(x) for x in row[1:37]]
        if any(x is None for x in parts) or min(parts)<0: continue
        out[int(num)] = parts
    if out: return out
    raise ValueError('Не нашёл данные. Ожидается ID форсунки и 36 числовых столбцов; для Excel допускается строка заголовков 1..36.')


def _xlsx_sheets(path: Path, with_row_numbers=False):
    with zipfile.ZipFile(path,'r') as z:
        wbroot = ET.fromstring(z.read('xl/workbook.xml'))
        relroot = ET.fromstring(z.read('xl/_rels/workbook.xml.rels'))
        targets = {r.attrib['Id']:r.attrib['Target'] for r in relroot.findall(PR+'Relationship')}
        shared = []
        if 'xl/sharedStrings.xml' in z.namelist():
            stringsroot = ET.fromstring(z.read('xl/sharedStrings.xml'))
            for si in stringsroot.findall(SS+'si'):
                shared.append(''.join((s.text or '') for s in si.iter(SS+'t')))
        for sh in wbroot.find(SS+'sheets'):
            name = sh.attrib['name']
            target = targets.get(sh.attrib.get(REL+'id',''))
            if not target: continue
            if target.startswith('/'):
                filename = target.lstrip('/')
            else:
                filename = 'xl/'+target.replace('\\','/')
            # normalize x/../y paths if needed
            import posixpath
            filename = posixpath.normpath(filename)
            if filename not in z.namelist(): continue
            root = ET.fromstring(z.read(filename))
            result=[]
            for row in root.iter(SS+'row'):
                values = {}
                for c in row.findall(SS+'c'):
                    idx = _col_idx(c.attrib['r'])
                    kind = c.attrib.get('t')
                    v = c.find(SS+'v')
                    if kind == 'inlineStr':
                        inline = c.find(SS+'is')
                        val = ''.join((t.text or '') for t in inline.iter(SS+'t')) if inline is not None else ''
                    elif v is None:
                        val = 'Формула без вычисленного результата' if with_row_numbers and c.find(SS+'f') is not None else None
                    elif kind == 's':
                        val = shared[int(v.text)]
                    elif kind == 'b' and with_row_numbers:
                        val = v.text == '1'
                    else:
                        val = v.text
                    values[idx] = val
                if values:
                    line=[None]*(max(values)+1)
                    for idx,val in values.items():line[idx]=val
                    result.append((int(row.attrib['r']),line) if with_row_numbers else line)
            yield name,result


def import_sector_file(path: str):
    path = Path(path)
    if path.suffix.lower() == '.xlsx':
        sheets = list(_xlsx_sheets(path))
        # prefer original measurements rather than normalized processing sheet
        sheets.sort(key=lambda x: (0 if 'эксперимент' in x[0].lower() else 1))
        failures=[]
        for name, rows in sheets:
            try:
                result = _extract_rows(rows)
                return result, name
            except ValueError as exc:
                failures.append(str(exc))
        raise ValueError('Не удалось найти таблицу измерений в XLSX. '+' '.join(failures[:1]))
    if path.suffix.lower() == '.csv':
        raw = path.read_text(encoding='utf-8-sig')
        try:
            dialect = csv.Sniffer().sniff(raw[:4096],delimiters=',;\t')
            sep = dialect.delimiter
        except csv.Error:
            sep = ';' if raw.count(';')>raw.count(',') else ','
        return _extract_rows(list(csv.reader(io.StringIO(raw),delimiter=sep))), 'CSV'
    raise ValueError('Поддерживаются .xlsx и .csv')


def write_sector_csv(path: str, nozzles):
    with open(path,'w',encoding='utf-8-sig',newline='') as f:
        out=csv.writer(f,delimiter=';')
        out.writerow(['ID']+[str(i) for i in range(1,37)])
        for n in nozzles:out.writerow([n.id]+[f'{x:.10g}' for x in n.sector_ml])


def write_result_csv(path: str, planes, cfg=None):
    import json
    planes=list(planes)
    composite=any(getattr(p,'is_composite',False) for p in planes)
    confined=any(getattr(p,'suspended_cell_mass_kg_s',None) is not None for p in planes)
    phase_columns=['total_including_film_kg_s','suspended_kg_s','film_inner_kg_s','film_outer_kg_s',
                   'suspended_flux_kg_m2_s','film_inner_cell_equivalent_kg_m2_s','film_outer_cell_equivalent_kg_m2_s'] if confined else []
    with open(path,'w',encoding='utf-8-sig',newline='') as f:
        out=csv.writer(f,delimiter=';')
        out.writerow(['z_mm','radial_level','theta_node','radius_center_mm',
                      'theta_deg','cell_area_m2','mass_flow_kg_s','flux_kg_m2_s']+phase_columns+(['calculation_mode','scan_step','time_s','delta_p_kgf_cm2_json'] if composite else []))
        for p in planes:
            for i,r in enumerate(p.radial_centers_mm):
                for j,th in enumerate(p.theta_centers_deg):
                    phases=[]
                    if confined:
                        arrays=[getattr(p,name,None) for name in ('suspended_cell_mass_kg_s','film_inner_cell_mass_kg_s','film_outer_cell_mass_kg_s')]
                        if any(a is None for a in arrays):raise ValueError('Нельзя смешать сечения с разными моделями переноса в одном CSV')
                        values=[float(a[i,j]) for a in arrays];area=float(p.cell_area_m2[i,j])
                        phases=[float(p.cell_mass_kg_s[i,j]),*values,*[value/area for value in values]]
                    out.writerow([f'{p.z_mm:.8g}',i+1,j+1,f'{r:.8g}',f'{th:.8g}',
                                  f'{p.cell_area_m2[i,j]:.12g}',
                                  f'{p.cell_mass_kg_s[i,j]:.12g}',
                                  f'{p.flux_kg_m2_s[i,j]:.12g}']+phases+((['quasistationary',int(p.scan_steps[j]),float(p.scan_times_s[j]),json.dumps(p.scan_pressures[j],ensure_ascii=False)] if p.is_composite else ['stationary','','','']) if composite else []))
    if not composite and any(p.geometry_notice for p in planes) and not confined:
        from .analysis import balance_rows
        target=Path(path).with_name(Path(path).stem+'_balances.csv')
        with target.open('w',encoding='utf-8-sig',newline='') as file:csv.writer(file).writerows(balance_rows(planes))
    if confined:
        from .analytics_io import transport_tables
        from .core import project_to_json
        tables=transport_tables(planes)
        for name,rows in tables[1:]:
            target=Path(path).with_name(Path(path).stem+'_'+name+'.csv')
            with target.open('w',encoding='utf-8-sig',newline='') as file:csv.writer(file).writerows(rows)
        metadata=dict(format='fuel-confined-transport-result',version=1,
            quantity='mass flux; total includes suspended fuel and surface film integrated into boundary cells',
            phase_model='100% capture at first contact; film is separate from suspended fuel',
            film_flux='cell-equivalent surface flow divided by cell area; not suspended volumetric density',
            momentum_units='N = kg/s * m/s; component order z,x,y; carrier and wall/film driving act on fuel',
            composite_global_balance=None if composite else 'see Transport_balance.csv',
            settings=project_to_json(cfg,[])['settings'] if cfg is not None else None,
            notices=sorted({getattr(p,'transport_notice','') for p in planes}))
        target=Path(path).with_name(Path(path).stem+'_transport.json')
        target.write_text(json.dumps(metadata,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def import_characteristic_file(path, unit='kg/h', density_kg_m3=800., sheet=None):
    """Read two-column measured pressure/flow pairs atomically (CSV or XLSX).

    Pressure is kgf/cm². Flow unit is chosen explicitly by the caller; invalid
    data rows are rejected with their row number, never silently skipped.
    """
    from .characteristics import ExperimentalCharacteristic, flow_to_kg_h
    path=Path(path)
    if path.suffix.lower()=='.xlsx':
        try:sheets=list(_xlsx_sheets(path))
        except (zipfile.BadZipFile,ET.ParseError,KeyError,IndexError,TypeError) as exc:
            raise ValueError('Не удалось прочитать таблицы XLSX: '+str(exc)) from exc
        if not sheets:raise ValueError('XLSX не содержит таблиц.')
        if sheet is None:
            if len(sheets)!=1:raise ValueError('Выберите лист XLSX явно.')
            name,rows=sheets[0]
        else:
            selected=next((x for x in sheets if x[0]==sheet),None)
            if selected is None:raise ValueError('Лист XLSX не найден.')
            name,rows=selected
    elif path.suffix.lower()=='.csv':
        raw=path.read_text(encoding='utf-8-sig')
        try:sep=csv.Sniffer().sniff(raw[:4096],delimiters=',;\t').delimiter
        except csv.Error:sep=';' if ';' in raw else ','
        rows=list(csv.reader(io.StringIO(raw),delimiter=sep));name='CSV'
    else:raise ValueError('Поддерживаются CSV и XLSX.')
    pairs=[]
    header_seen=False
    for index,row in enumerate(rows,1):
        if not row or all(v is None or not str(v).strip() for v in row):continue
        if len(row)<2:raise ValueError(f'Строка {index}: нужны давление и расход.')
        pressure=_num(row[0]);flow=_num(row[1])
        if not pairs and not header_seen and pressure is None and flow is None:
            p_label=str(row[0]).strip().lower();q_label=str(row[1]).strip().lower()
            if p_label.startswith(('Δp'.lower(),'dp','pressure','давление','перепад')) and q_label.startswith(('q','flow','расход')):
                if re.search(r'(^|[(_ \[,])(kpa|mpa|pa|кпа|мпа|па)([)\] ,]|$)',p_label):
                    raise ValueError('Импорт ожидает давление в кгс/см²; предварительно переведите Па/кПа/МПа.')
                declared='kg/h' if any(x in q_label for x in ('kg/h','kg_h','кг/ч')) else 'l/min' if any(x in q_label for x in ('l/min','l_min','л/мин')) else None
                if declared is not None and declared!=unit:
                    raise ValueError('Единицы расхода в заголовке не совпадают с выбранными единицами ввода.')
                header_seen=True;continue
        if pressure is None or flow is None:raise ValueError(f'Строка {index}: неверное давление или расход.')
        try:pairs.append((pressure,flow_to_kg_h(flow,unit,density_kg_m3)))
        except ValueError as exc:raise ValueError(f'Строка {index}: {exc}') from exc
    return ExperimentalCharacteristic(pairs),name


def write_characteristic_csv(path,curve):
    with open(path,'w',encoding='utf-8-sig',newline='') as f:
        writer=csv.writer(f,delimiter=';')
        writer.writerow(['pressure_kgf_cm2','flow_kg_h'])
        writer.writerows(curve.points)
