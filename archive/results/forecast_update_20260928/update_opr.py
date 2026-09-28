"""Update only the primary OPR forecast sheet, preserving unsupported Excel parts.

Write formula caches alongside formulas so the delivered values are readable
without an Excel calculation engine. Keep the immutable pre-update backup.
"""
from pathlib import Path
import shutil
import zipfile
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
from openpyxl.utils.datetime import from_excel

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
NS = {'s': 'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}


def main():
    target = ROOT / 'assets/06_2026_02_Прогноз.xlsx'
    backup = OUT / '06_2026_02_Прогноз.before.xlsx'
    if not backup.exists():
        shutil.copy2(target, backup)
    actual = pd.read_csv(ROOT / 'data/inflation_data.csv', sep=';', decimal=',')
    actual.index = pd.to_datetime(actual.Date, dayfirst=True).dt.to_period('M').dt.to_timestamp()
    future = pd.read_csv(OUT / 'forecast_2026_2027.csv', parse_dates=['date']).set_index('date')
    mom = pd.concat([actual.mom, future.central_mom_index])
    yoy = (mom / 100).rolling(12).apply(np.prod, raw=True) * 100
    with zipfile.ZipFile(target) as archive:
        parts = {name: archive.read(name) for name in archive.namelist()}
    workbook = ET.fromstring(parts['xl/workbook.xml'])
    sheets = workbook.find('s:sheets', NS)
    sheet = next(s for s in sheets if s.attrib['name'] == 'Прогноз')
    rid = sheet.attrib['{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id']
    rels = ET.fromstring(parts['xl/_rels/workbook.xml.rels'])
    path = next(x.attrib['Target'] for x in rels if x.attrib['Id'] == rid)
    path = path.lstrip('/') if path.startswith('/') else 'xl/' + path
    xml = ET.fromstring(parts[path])
    edits = 0
    for row in xml.findall('s:sheetData/s:row', NS):
        cells = {c.attrib['r']: c for c in row}
        number = int(row.attrib['r'])
        a = cells.get(f'A{number}')
        if a is None or a.find('s:v', NS) is None:
            continue
        try:
            date = pd.Timestamp(from_excel(float(a.find('s:v', NS).text))).to_period('M').to_timestamp()
        except (ValueError, TypeError, OverflowError):
            continue
        if date not in mom.index or date < pd.Timestamp('2022-01-01'):
            continue
        for col, value in [('E', mom.loc[date]), ('F', yoy.loc[date])]:
            cell = cells.get(f'{col}{number}')
            if cell is None:
                cell = ET.SubElement(row, f'{{{NS["s"]}}}c', {'r': f'{col}{number}'})
            cell.attrib.pop('t', None)
            v = cell.find('s:v', NS)
            if v is None:
                v = ET.SubElement(cell, f'{{{NS["s"]}}}v')
            if col == 'E':
                f = cell.find('s:f', NS)
                if f is not None:
                    cell.remove(f)
            # Existing 12-month formulas preserved; cache must respect ROUND(...,2).
            v.text = format(round(float(value), 2), '.12g')
        for col, value in [('C', mom.loc[date]-100), ('D', round(yoy.loc[date], 2)-100)]:
            cell = cells.get(f'{col}{number}')
            if cell is not None and cell.find('s:f', NS) is not None:
                v = cell.find('s:v', NS)
                if v is None:
                    v = ET.SubElement(cell, f'{{{NS["s"]}}}v')
                v.text = format(float(value), '.12g')
        edits += 1
    assert edits == 72, edits  # January 2022 through December 2027.
    parts[path] = ET.tostring(xml, encoding='utf-8', xml_declaration=True)
    # Only the sheet XML changes: preserve charts, drawings, styles and other sheets byte-for-byte.
    temporary = target.with_suffix('.update.xlsx')
    with zipfile.ZipFile(temporary, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        for name, content in parts.items():
            archive.writestr(name, content)
    temporary.replace(target)
    print(f'Updated {edits} monthly rows; E facts/forecasts and F cached YoY.')


if __name__ == '__main__':
    main()
