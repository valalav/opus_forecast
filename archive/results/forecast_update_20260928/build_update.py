"""Rebuild the dated expert scenario from verified August inputs and model cache.

Run from repository root. Expert adjustments are assumptions, not fitted coefficients.
The production ensemble weights and model specifications are unchanged.
"""
from pathlib import Path
import hashlib
import json
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
OUT = Path(__file__).resolve().parent


def main():
    from sirena.data_loader import load_model_data
    raw = load_model_data('raw', include_macro=True)
    assert raw.index.max() == pd.Timestamp('2026-08-01')
    cache = json.loads((ROOT / 'data/precomputed_forecasts.json').read_text())
    dates = pd.date_range('2026-09-01', '2027-12-01', freq='MS')
    assert cache['forecast_dates'] == dates.strftime('%Y-%m-%d').tolist()
    baseline = np.asarray(cache['forecasts']['Ensemble'], dtype=float)
    weights = pd.read_csv(ROOT / 'data/access_weights.csv')
    utility_weight = float(weights.loc[
        weights.Region_code.eq(7) & weights.Item_code.eq(279)
        & weights.Day.eq('01/01/26 00:00:00'), 'Weight_vertical'].iloc[0])
    micro = pd.read_csv(ROOT / 'data/kbr_indices.csv', parse_dates=['Date'])
    utility = micro[micro.Item_code.eq(279)].set_index('Date').MoM - 100
    history = utility[(utility.index.year >= 2016) & (utility.index.year <= 2025)]
    # Approximation to the recurring tariff effect embedded in calendar models.
    # Its removal is uncertain; disclose sensitivity instead of claiming identification.
    embedded_july_rate = float(history[history.index.month == 7].median())
    embedded_october_rate = float(history[history.index.month == 10].median())
    direct = np.zeros(16)
    direct[1] = utility_weight * (10.0 - embedded_october_rate)
    direct[10] = utility_weight * (11.0 - embedded_july_rate)
    # Incremental cost pass-through ABOVE the model baseline; no repetition of
    # the June/July gasoline price level shock. Not econometrically estimated.
    cost = np.array([.08, .15, .10, .06, .04, .03, .02, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    # Conditional easing: reduced persistence of the summer shock during 2027.
    # This is explicitly scenario judgment, not a promised return to a target.
    easing = np.r_[np.zeros(4), np.linspace(.10, .22, 12)]
    central = np.round(baseline + direct + cost - easing, 2)
    low = np.round(central - np.r_[[.20, .25, .25, .25], np.full(12, .15)], 2)
    high = np.round(central + np.r_[[.20, .35, .30, .25], np.full(12, .18)], 2)
    frame = pd.DataFrame({'date': dates, 'baseline_mom_pp': baseline,
        'utility_adjustment_pp': direct, 'incremental_cost_pp': cost,
        'easing_pp': easing, 'central_mom_pp': central,
        'low_mom_pp': low, 'high_mom_pp': high})
    observed = raw['Все товары и услуги'] - 100
    for scenario in ['central', 'low', 'high']:
        all_mom = pd.concat([observed, pd.Series(frame[f'{scenario}_mom_pp'].values, index=dates)])
        yoy = (1 + all_mom / 100).rolling(12).apply(np.prod, raw=True).sub(1).mul(100)
        frame[f'{scenario}_yoy_pct'] = yoy.reindex(dates).values
    frame['central_mom_index'] = 100 + central
    frame.to_csv(OUT / 'forecast_2026_2027.csv', index=False, float_format='%.8f')
    actual_2026 = observed[observed.index.year == 2026]
    export = pd.concat([
        pd.DataFrame({'date': actual_2026.index, 'mom_pp': actual_2026.values, 'status': 'fact'}),
        pd.DataFrame({'date': dates, 'mom_pp': central, 'status': 'forecast'})], ignore_index=True)
    all_path = pd.concat([observed, pd.Series(central, index=dates)])
    export['mom_index'] = export.mom_pp + 100
    export['yoy_pct'] = export.date.map((1 + all_path / 100).rolling(12).apply(np.prod, raw=True).sub(1).mul(100))
    export.to_csv(OUT / 'full_path_2026_2027.csv', index=False, float_format='%.8f')
    july_sensitivity = pd.DataFrame({'embedded_july_rate': [0, embedded_july_rate, 4.97, 11.78]})
    july_sensitivity['july_2027_mom_pp'] = baseline[10] + utility_weight * (11-july_sensitivity.embedded_july_rate) - easing[10]
    july_sensitivity.to_csv(OUT / 'tariff_sensitivity.csv', index=False)
    with pd.ExcelWriter(OUT / 'forecast_calculations.xlsx', engine='openpyxl') as writer:
        export.to_excel(writer, sheet_name='Факт и прогноз', index=False)
        frame.to_excel(writer, sheet_name='Сценарии и вклады', index=False)
        july_sensitivity.to_excel(writer, sheet_name='Тарифная чувствительность', index=False)
        for name in ['august_model_comparison', 'diagnostics']:
            p = OUT / f'{name}.csv'
            if p.exists():
                pd.read_csv(p).to_excel(writer, sheet_name=name[:31], index=False)
        for sheet in writer.book:
            sheet.freeze_panes = 'B2'
            sheet.auto_filter.ref = sheet.dimensions
            for col in sheet.columns:
                sheet.column_dimensions[col[0].column_letter].width = min(42, max(18, len(str(col[0].value)) + 2))
    meta = {'as_of': '2026-09-28', 'cutoff': '2026-08', 'utility_weight': utility_weight,
        'tariff_2026_pct': 10, 'tariff_2027_pct': 11,
        'tariff_2027_status': 'scenario; national forecast reported 2026-09-24, KBR order unverified',
        'embedded_july_rate': embedded_july_rate, 'embedded_october_rate': embedded_october_rate,
        'model_status': cache['model_status'],
        'scenario_intervals': 'conditional scenarios, not statistical confidence intervals',
        'sources': {f: hashlib.sha256((ROOT / f).read_bytes()).hexdigest() for f in [
            'data/inflation_data.csv', 'data/sa_fl.csv', 'data/mom_sa_kbr.csv',
            'data/access_weights.csv', 'data/kbr_indices.csv', 'data/precomputed_forecasts.json']}}
    (OUT / 'calculation_manifest.json').write_text(json.dumps(meta, ensure_ascii=False, indent=2))
    policy = {'generated_at': cache['generated_at'], 'latest_official_month': '2026-08-01',
        'forecast_dates': dates.strftime('%Y-%m-%d').tolist(), 'mom_pp': central.tolist(),
        'scenario': 'august_actual_october_tariffs_cost_pass_through_20260928',
        'source_artifact': 'archive/results/forecast_update_20260928/REPORT.md',
        'description': 'August actual; 10% utility tariff in October 2026; incremental cost pass-through; conditional easing and July 2027 tariff scenario. See report for limits.'}
    (ROOT / 'data/send_ready_policy_trajectory.json').write_text(json.dumps(policy, ensure_ascii=False, indent=2) + '\n')
    make_chart(frame)
    print(frame.loc[frame.date.isin(pd.to_datetime(['2026-10-01','2026-11-01','2026-12-01','2027-12-01'])),
                    ['date','central_mom_pp','central_yoy_pct','low_yoy_pct','high_yoy_pct']].round(3).to_string(index=False))


def make_chart(frame):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import plotly.graph_objects as go
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    for ax, suffix, label in [(axes[0], 'mom_pp', '% к предыдущему месяцу'), (axes[1], 'yoy_pct', '% к тому же месяцу прошлого года')]:
        ax.fill_between(frame.date, frame[f'low_{suffix}'], frame[f'high_{suffix}'], color='#d6e7f5', label='Сценарный диапазон')
        ax.plot(frame.date, frame[f'central_{suffix}'], color='#175b91', marker='o', label='Центральный сценарий')
        ax.set_ylabel(label)
        ax.grid(alpha=.2)
    axes[0].set_title('КБР: прогноз после факта августа 2026 · расчёт 28.09.2026')
    axes[0].legend(loc='upper right')
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(OUT / 'forecast.png', dpi=170)
    plt.close(fig)
    fig = go.Figure()
    for scenario, label in [('low','Снижение давления'),('central','Центральный'),('high','Сохранение давления')]:
        fig.add_scatter(x=frame.date, y=frame[f'{scenario}_mom_pp'], mode='lines+markers', name=label)
    fig.update_layout(title='ИПЦ КБР: сценарии до декабря 2027', yaxis_title='% к предыдущему месяцу', template='plotly_white')
    fig.write_html(OUT / 'forecast.html', include_plotlyjs=True)


if __name__ == '__main__':
    main()
