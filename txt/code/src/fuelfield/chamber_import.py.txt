"""XLSX point coordinates: explicit columns/units, exact row diagnostics."""
from pathlib import Path
from zipfile import BadZipFile
from xml.etree.ElementTree import ParseError
from .data_io import _xlsx_sheets
from .chamber import UNITS, number


def read_point_tables(path):
    try:
        tables = list(_xlsx_sheets(Path(path), with_row_numbers=True))
    except (OSError, BadZipFile, ParseError, KeyError, IndexError, ValueError, TypeError) as exc:
        raise ValueError('Не удалось прочитать XLSX: '+str(exc)) from exc
    tables = [(name, rows) for name, rows in tables if rows]
    if not tables: raise ValueError('В книге нет листов с данными')
    return tables


def parse_points(rows, z_column, r_column, first_row=2, unit='mm'):
    if z_column == r_column: raise ValueError('Для z и R выберите разные столбцы')
    if unit not in UNITS: raise ValueError('Выберите единицы mm, cm или m')
    points=[];seen=set()
    for row_number, values in rows:
        if row_number < first_row or all(v is None or v == '' for v in values): continue
        try:
            pair=[]
            for column, name in ((z_column,'z'),(r_column,'R')):
                value=values[column] if column < len(values) else None
                if isinstance(value,str):value=value.strip().replace(',','.')
                pair.append(number(value,name)*UNITS[unit])
            if pair[1]<=0:raise ValueError('R должен быть положительным')
            if tuple(pair) in seen:raise ValueError('Повторная точка z/R')
        except ValueError as exc:raise ValueError(f'Строка Excel {row_number}: {exc}') from exc
        seen.add(tuple(pair));points.append((row_number,*pair))
    if not points:raise ValueError('В выбранных столбцах и строках нет точек')
    return points
