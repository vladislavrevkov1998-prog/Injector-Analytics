"""External chart tables and portable four-section OpenXML export."""
import csv
import json
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
from xml.etree import ElementTree as ET
import numpy as np
from .analysis import cell_values, UNITS
from .data_io import _xlsx_sheets
from .core import project_to_json

NS='http://schemas.openxmlformats.org/spreadsheetml/2006/main'
REL='http://schemas.openxmlformats.org/officeDocument/2006/relationships'
PKG='http://schemas.openxmlformats.org/package/2006/relationships'

def read_chart_tables(path):
    path=Path(path)
    if path.suffix.lower()=='.xlsx':tables=list(_xlsx_sheets(path))
    elif path.suffix.lower()=='.csv':
        text=path.read_text(encoding='utf-8-sig')
        try:dialect=csv.Sniffer().sniff(text[:8192],delimiters=',;\t')
        except csv.Error:dialect=csv.excel
        tables=[(path.stem,list(csv.reader(text.splitlines(),dialect)))]
    else:raise ValueError('Ожидается CSV или XLSX')
    out=[]
    for name,rows in tables:
        rows=[r for r in rows if any(v is not None and str(v).strip() for v in r)]
        if len(rows)<2:continue
        headers=[str(v or '').strip() or f'Столбец {i+1}' for i,v in enumerate(rows[0])]
        columns={}
        for i,header in enumerate(headers):
            try:
                values=np.array([float(str(r[i]).strip().replace(',','.')) for r in rows[1:]])
                if np.all(np.isfinite(values)):columns[i]=values
            except (ValueError,IndexError):continue
        if len(columns)>=2:out.append((name,headers,columns))
    if not out:raise ValueError('Нужны заголовки и хотя бы два полностью заполненных числовых столбца (без NaN/Inf).')
    return out


def _col(i):
    result='';i+=1
    while i:i,k=divmod(i-1,26);result=chr(65+k)+result
    return result


def write_epures_xlsx(path, planes, cfg, metric='flux', interval=180.):
    if not 1<=len(planes)<=50:raise ValueError('Экспорт требует 1–50 рассчитанных сечений')
    metadata=json.dumps(project_to_json(cfg,[])['settings'],ensure_ascii=False,separators=(',',':'))
    sheets=[]
    for p in planes:
        values=cell_values(p,metric,cfg.density_kg_m3,interval)
        headers=['angle_deg']
        for r in p.radial_centers_mm:headers.extend([f'R={r:.12g} мм [{UNITS[metric]}]',f'x R={r:.12g} [мм]',f'y R={r:.12g} [мм]'])
        headers+=['z_mm','units','pressure_mode','target_pressure_kgf_cm2','density_kg_m3','interval_s','settings_json']
        rows=[headers]
        for j,angle in enumerate(p.theta_centers_deg):
            row=[float(angle)]
            for k,r in enumerate(p.radial_centers_mm):
                v=values[k,j]
                row.extend([None if np.ma.is_masked(v) else float(v),float(r*np.cos(np.deg2rad(angle))),float(r*np.sin(np.deg2rad(angle)))])
            row += [p.z_mm,UNITS[metric],cfg.pressure_mode,cfg.target_pressure_kgf_cm2,cfg.density_kg_m3,interval,metadata]
            rows.append(row)
        root=ET.Element('worksheet',xmlns=NS);data=ET.SubElement(root,'sheetData')
        for i,row in enumerate(rows,1):
            node=ET.SubElement(data,'row',r=str(i))
            for j,value in enumerate(row):
                if value is None:continue
                c=ET.SubElement(node,'c',r=f'{_col(j)}{i}')
                if isinstance(value,(int,float,np.number)):ET.SubElement(c,'v').text=str(value)
                else:
                    c.set('t','inlineStr');ET.SubElement(ET.SubElement(c,'is'),'t').text=str(value)
        sheets.append(ET.tostring(root,encoding='utf-8',xml_declaration=True))
    types=ET.Element('Types',xmlns='http://schemas.openxmlformats.org/package/2006/content-types')
    ET.SubElement(types,'Default',Extension='rels',ContentType='application/vnd.openxmlformats-package.relationships+xml')
    ET.SubElement(types,'Default',Extension='xml',ContentType='application/xml')
    ET.SubElement(types,'Override',PartName='/xl/workbook.xml',ContentType='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml')
    wb=ET.Element('workbook',xmlns=NS);ws=ET.SubElement(wb,'sheets')
    rels=ET.Element('Relationships',xmlns=PKG)
    for i in range(1,len(planes)+1):
        ET.SubElement(ws,'sheet',name=f'z{i} {planes[i-1].z_mm:.9g} мм'[:31],sheetId=str(i),attrib={f'{{{REL}}}id':f'rId{i}'})
        ET.SubElement(rels,'Relationship',Id=f'rId{i}',Type=REL+'/worksheet',Target=f'worksheets/sheet{i}.xml')
        ET.SubElement(types,'Override',PartName=f'/xl/worksheets/sheet{i}.xml',ContentType='application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml')
    rootrels=ET.Element('Relationships',xmlns=PKG)
    ET.SubElement(rootrels,'Relationship',Id='rId1',Type=REL+'/officeDocument',Target='xl/workbook.xml')
    with ZipFile(path,'w',ZIP_DEFLATED) as z:
        for name,root in [('[Content_Types].xml',types),('_rels/.rels',rootrels),('xl/workbook.xml',wb),('xl/_rels/workbook.xml.rels',rels)]:z.writestr(name,ET.tostring(root,encoding='utf-8',xml_declaration=True))
        for i,sheet in enumerate(sheets,1):z.writestr(f'xl/worksheets/sheet{i}.xml',sheet)


def write_workbook(path,tables):
    """Generic OpenXML report with explicit sheet names and cached values."""
    types=ET.Element('Types',xmlns='http://schemas.openxmlformats.org/package/2006/content-types')
    for extension,content in [('rels','application/vnd.openxmlformats-package.relationships+xml'),('xml','application/xml')]:ET.SubElement(types,'Default',Extension=extension,ContentType=content)
    ET.SubElement(types,'Override',PartName='/xl/workbook.xml',ContentType='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml')
    wb=ET.Element('workbook',xmlns=NS);sheets=ET.SubElement(wb,'sheets');rels=ET.Element('Relationships',xmlns=PKG)
    rootrels=ET.Element('Relationships',xmlns=PKG);ET.SubElement(rootrels,'Relationship',Id='rId1',Type=REL+'/officeDocument',Target='xl/workbook.xml')
    with ZipFile(path,'w',ZIP_DEFLATED) as z:
        for index,(name,rows) in enumerate(tables,1):
            if not name or len(name)>31:raise ValueError('Имя Excel-листа: 1–31 символ')
            ET.SubElement(sheets,'sheet',name=name,sheetId=str(index),attrib={f'{{{REL}}}id':f'rId{index}'})
            ET.SubElement(rels,'Relationship',Id=f'rId{index}',Type=REL+'/worksheet',Target=f'worksheets/sheet{index}.xml')
            ET.SubElement(types,'Override',PartName=f'/xl/worksheets/sheet{index}.xml',ContentType='application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml')
            root=ET.Element('worksheet',xmlns=NS);data=ET.SubElement(root,'sheetData')
            for i,row in enumerate(rows,1):
                node=ET.SubElement(data,'row',r=str(i))
                for j,value in enumerate(row):
                    if value is None:continue
                    cell=ET.SubElement(node,'c',r=f'{_col(j)}{i}')
                    if isinstance(value,(int,float,np.number)) and np.isfinite(value):ET.SubElement(cell,'v').text=str(int(value) if isinstance(value,(bool,np.bool_)) else value)
                    else:cell.set('t','inlineStr');ET.SubElement(ET.SubElement(cell,'is'),'t').text=str(value)
            z.writestr(f'xl/worksheets/sheet{index}.xml',ET.tostring(root,encoding='utf-8',xml_declaration=True))
        for name,root in [('[Content_Types].xml',types),('_rels/.rels',rootrels),('xl/workbook.xml',wb),('xl/_rels/workbook.xml.rels',rels)]:z.writestr(name,ET.tostring(root,encoding='utf-8',xml_declaration=True))
