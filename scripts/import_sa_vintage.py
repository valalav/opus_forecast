#!/usr/bin/env python3
"""Import one complete KBR SA workbook vintage, verifying its cached values against db.

Usage: python3 scripts/import_sa_vintage.py source.xlsx --data-dir data
Only mom_sa (SA MoM index=100) is accepted. Raw data are never an SA fallback.
"""
import argparse
import csv
import datetime as dt
import hashlib
import io
import json
from pathlib import Path

import openpyxl


def extract_vintage(source):
    source = Path(source)
    workbook = openpyxl.load_workbook(source, read_only=True, data_only=True)
    try:
        rows = list(workbook['mom_sa'].values)
        if rows[0][6] != 'Кабардино-Балкарская Республика' or rows[0][10] != 7:
            raise ValueError('Workbook must explicitly select KBR, region 7')
        if 'momsa' not in str(rows[2][1]).lower():
            raise ValueError('Expected momsa measure (SA MoM index)')
        dates = [d for d in rows[3][3:] if isinstance(d, dt.datetime)]
        if not dates or len(dates) != len(set(dates)) or dates != sorted(dates):
            raise ValueError('Missing, duplicate or unsorted observation months')
        series = {int(r[1]): (r[2], list(r[3:3+len(dates)]))
                  for r in rows[4:] if isinstance(r[1], (int, float))}
        for code in (1, 2, 3, 4):
            if code not in series or not all(isinstance(v, (int, float)) for v in series[code][1]):
                raise ValueError(f'Main aggregate {code} is incomplete')
        db = {(r[0], int(r[1])): r[4] for r in workbook['db'].iter_rows(min_row=2, values_only=True)
              if r[2] == 7 and r[1] in series}
        checked = 0
        for code, (_, values) in series.items():
            for date, value in zip(dates, values):
                if isinstance(value, (int, float)):
                    other = db.get((date, code))
                    if not isinstance(other, (int, float)) or abs(value-other) > 1e-9:
                        raise ValueError(f'mom_sa/db mismatch at {date:%Y-%m}, code {code}')
                    checked += 1
        return dates, series, checked
    finally:
        workbook.close()


def csv_bytes(rows):
    stream = io.StringIO(newline='')
    writer = csv.writer(stream, delimiter=';', lineterminator='\n')
    writer.writerows(rows)
    return stream.getvalue().encode('utf-8-sig')


def import_vintage(source, data_dir):
    source, data_dir = Path(source), Path(data_dir)
    dates, series, checked = extract_vintage(source)
    number = lambda v: str(v).replace('.', ',') if isinstance(v, (int, float)) else ''
    outputs = {
        'sa_fl.csv': csv_bytes([['Код', 'Товар', 'Дата', 'Значение']] + [
            [code, series[code][0], date.strftime('%d.%m.%Y'), number(series[code][1][i])]
            for i, date in enumerate(dates) for code in (1, 2, 3, 4)]),
        'mom_sa_kbr.csv': csv_bytes([['Код', 'Товар'] + [d.strftime('%Y-%m') for d in dates]] + [
            [code, name] + [number(v) for v in values] for code, (name, values) in series.items()]),
        'sa_hor.csv': csv_bytes([[f'{code}_{name}' for code, (name, _) in series.items()]] + [
            [number(values[i]) for _, values in series.values()] for i in range(len(dates))]),
    }
    manifest = {'source': str(source.resolve()), 'sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
                'region': 'Кабардино-Балкарская Республика', 'region_code': 7,
                'sheet': 'mom_sa', 'units': 'SA MoM index, previous month=100',
                'first': dates[0].strftime('%Y-%m'), 'last': dates[-1].strftime('%Y-%m'),
                'db_cross_check_count': checked, 'db_cross_check_mismatches': [],
                'observation_months': [d.strftime('%Y-%m') for d in dates],
                'outputs': {name: {'sha256': hashlib.sha256(content).hexdigest()}
                            for name, content in outputs.items()}}
    data_dir.mkdir(parents=True, exist_ok=True)
    # Preserve complete old files before replacing the vintage.
    backup = data_dir / 'vintage_backups' / dt.datetime.now().strftime('%Y%m%dT%H%M%S%f')
    backup.mkdir(parents=True)
    for name in [*outputs, 'sa_source_manifest.json']:
        if (data_dir / name).exists():
            (backup / name).write_bytes((data_dir / name).read_bytes())
    for name, content in outputs.items():
        (data_dir / name).with_suffix('.csv.tmp').write_bytes(content)
    for name in outputs:
        (data_dir / name).with_suffix('.csv.tmp').replace(data_dir / name)
    (data_dir / 'sa_source_manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+'\n', encoding='utf-8', newline='\n')
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('--data-dir', type=Path, default=Path('data'))
    args = parser.parse_args()
    print(json.dumps(import_vintage(args.source, args.data_dir), ensure_ascii=False, indent=2))
