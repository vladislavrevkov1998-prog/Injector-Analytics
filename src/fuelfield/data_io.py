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


def _xlsx_sheets(path: Path):
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
                        val = None
                    elif kind == 's':
                        val = shared[int(v.text)]
                    else:
                        val = v.text
                    values[idx] = val
                if values:
                    line=[None]*(max(values)+1)
                    for idx,val in values.items():line[idx]=val
                    result.append(line)
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


def write_result_csv(path: str, planes):
    with open(path,'w',encoding='utf-8-sig',newline='') as f:
        out=csv.writer(f,delimiter=';')
        out.writerow(['z_mm','radial_level','theta_node','radius_center_mm',
                      'theta_deg','cell_area_m2','mass_flow_kg_s','flux_kg_m2_s'])
        for p in planes:
            for i,r in enumerate(p.radial_centers_mm):
                for j,th in enumerate(p.theta_centers_deg):
                    out.writerow([f'{p.z_mm:.8g}',i+1,j+1,f'{r:.8g}',f'{th:.8g}',
                                  f'{p.cell_area_m2[i,j]:.12g}',
                                  f'{p.cell_mass_kg_s[i,j]:.12g}',
                                  f'{p.flux_kg_m2_s[i,j]:.12g}'])
