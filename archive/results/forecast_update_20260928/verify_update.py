"""Independent arithmetic, provenance and OPR checks for the dated forecast."""
from pathlib import Path
import hashlib
import json
import zipfile
import warnings

import numpy as np
import pandas as pd
import openpyxl

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent


def main():
    f = pd.read_csv(OUT / 'forecast_2026_2027.csv', parse_dates=['date'])
    assert f.date.tolist() == pd.date_range('2026-09-01','2027-12-01',freq='MS').tolist()
    assert np.isfinite(f.select_dtypes('number')).all().all()
    np.testing.assert_allclose(f.central_mom_pp, np.round(
        f.baseline_mom_pp + f.utility_adjustment_pp + f.incremental_cost_pp - f.easing_pp, 2))
    assert np.all(f.low_mom_pp <= f.central_mom_pp) and np.all(f.high_mom_pp >= f.central_mom_pp)
    raw = pd.read_csv(ROOT / 'data/inflation_data.csv', sep=';', decimal=',')
    dates = pd.to_datetime(raw.Date, dayfirst=True).dt.to_period('M').dt.to_timestamp()
    facts = dict(zip(dates, raw.mom))
    for scenario in ['central','low','high']:
        values = dict(facts)
        values.update(dict(zip(f.date, 100 + f[f'{scenario}_mom_pp'])))
        for row in f.itertuples():
            months = pd.date_range(row.date - pd.DateOffset(months=11), row.date, freq='MS')
            result = 1.
            for month in months:
                result *= values[month] / 100
            assert abs((result-1)*100 - getattr(row, f'{scenario}_yoy_pct')) < 1e-7
    policy = json.loads((ROOT / 'data/send_ready_policy_trajectory.json').read_text())
    assert policy['latest_official_month'] == '2026-08-01'
    assert policy['forecast_dates'] == f.date.dt.strftime('%Y-%m-%d').tolist()
    np.testing.assert_allclose(policy['mom_pp'], f.central_mom_pp)
    manifest = json.loads((OUT / 'calculation_manifest.json').read_text())
    for name, expected in manifest['sources'].items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == expected, name
    assert abs(manifest['utility_weight'] - .05958) < 1e-12
    assert f.loc[f.date.eq('2026-11-01'),'utility_adjustment_pp'].iloc[0] == 0
    comparison = pd.read_csv(OUT / 'august_model_comparison.csv')
    np.testing.assert_allclose(comparison.precompute_absolute_error_pp,
        abs(comparison.precompute_forecast_mom_pp - .12))
    assert comparison.loc[comparison.precompute_absolute_error_pp.idxmin(),'model'] == 'Micro'
    warnings.filterwarnings('ignore', category=UserWarning)
    wb = openpyxl.load_workbook(ROOT / 'assets/06_2026_02_Прогноз.xlsx',data_only=True)
    sheet = wb['Прогноз']
    values = dict(facts)
    values.update(dict(zip(f.date, f.central_mom_index)))
    n = 0
    for row in sheet.iter_rows(min_row=3,max_row=74,max_col=6):
        date = pd.Timestamp(row[0].value)
        assert abs(row[4].value-values[date]) < 1e-8
        product = np.prod([values[d]/100 for d in pd.date_range(date-pd.DateOffset(months=11),date,freq='MS')])*100
        assert abs(row[5].value-round(product,2)) < 1e-8
        n += 1
    with zipfile.ZipFile(OUT/'06_2026_02_Прогноз.before.xlsx') as old, zipfile.ZipFile(ROOT/'assets/06_2026_02_Прогноз.xlsx') as new:
        assert old.namelist() == new.namelist()
        changed = [name for name in old.namelist() if old.read(name) != new.read(name)]
        assert changed == ['xl/worksheets/sheet1.xml'], changed
    result = {'status':'PASS','forecast_months':len(f),'opr_rows_checked':n,
        'independent_yoy_checks':len(f)*3,'historical_model_forecasts':len(comparison),
        'opr_changed_zip_parts':changed,'source_hashes_checked':len(manifest['sources']),
        'max_central_mom':float(f.central_mom_pp.max()),
        'largest_monthly_change_pp':float(f.central_mom_pp.diff().abs().max()),
        'negative_months':f.loc[f.central_mom_pp.lt(0),'date'].dt.strftime('%Y-%m').tolist()}
    (OUT/'verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
    print(json.dumps(result,ensure_ascii=False))


if __name__ == '__main__':
    main()
