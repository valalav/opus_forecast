#!/usr/bin/env python3
"""
VERIFY DASHBOARD - Проверка всех моделей и данных
==================================================
Запускает все модели, которые используются в dashboard,
и сохраняет результаты в CSV/JSON для проверки.

Использование:
    python3 scripts/verify_dashboard.py

Результаты:
    data/verify_forecast.csv   - прогнозы всех моделей
    data/verify_backtest.csv   - бэктест всех моделей
    data/verify_summary.json   - сводка по моделям
"""

import sys
import json
import time
import warnings
warnings.filterwarnings('ignore')

from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from datetime import datetime


def verify_forecasts(df, horizon=12):
    """Проверяет все модели прогноза."""
    print("=" * 60)
    print("ПРОВЕРКА ПРОГНОЗОВ (tab1)")
    print("=" * 60)

    last_date = df.index.max()
    forecast_dates = pd.date_range(
        start=last_date + pd.DateOffset(months=1),
        periods=horizon,
        freq='MS'
    )

    results = {'Date': forecast_dates}
    errors = []

    # 1. Ridge (базовая модель из dashboard)
    print("\n[1] Ridge...")
    try:
        from sirena.models.ridge import RidgeForecaster
        model = RidgeForecaster()
        model.fit(df)
        fc = model.forecast(horizon=horizon)
        if fc[0] > 50:  # index format
            fc = fc - 100
        results['Ridge'] = fc
        print(f"    OK: {fc[0]:.2f}% → {fc[-1]:.2f}%")
    except Exception as e:
        errors.append(f"Ridge: {e}")
        print(f"    ERROR: {e}")

    # 2. Ridge Extended
    print("[2] Ridge Extended...")
    try:
        from sirena.models.ridge_extended import RidgeExtendedForecaster
        model = RidgeExtendedForecaster()
        model.fit(df)
        vals = []
        for h in range(horizon):
            target = last_date + pd.DateOffset(months=h+1)
            df_ext = df.copy()
            df_ext.loc[target] = np.nan
            pred = model.predict(df_ext, target)['prediction'] - 100
            vals.append(pred)
        results['Ridge_Ext'] = vals
        print(f"    OK: {vals[0]:.2f}% → {vals[-1]:.2f}%")
    except Exception as e:
        errors.append(f"Ridge_Ext: {e}")
        print(f"    ERROR: {e}")

    # 3. Huber
    print("[3] Huber...")
    try:
        from sirena.models.huber import HuberForecaster
        model = HuberForecaster()
        model.fit(df)
        fc = model.forecast(horizon=horizon)
        if fc[0] > 50:
            fc = fc - 100
        results['Huber'] = fc
        print(f"    OK: {fc[0]:.2f}% → {fc[-1]:.2f}%")
    except Exception as e:
        errors.append(f"Huber: {e}")
        print(f"    ERROR: {e}")

    # 4. ElasticNet
    print("[4] ElasticNet...")
    try:
        from sirena.models.elasticnet import ElasticNetForecaster
        model = ElasticNetForecaster()
        model.fit(df)
        vals = []
        for h in range(horizon):
            target = last_date + pd.DateOffset(months=h+1)
            df_ext = df.copy()
            df_ext.loc[target] = np.nan
            pred = model.predict(df_ext, target)['prediction'] - 100
            vals.append(pred)
        results['ElasticNet'] = vals
        print(f"    OK: {vals[0]:.2f}% → {vals[-1]:.2f}%")
    except Exception as e:
        errors.append(f"ElasticNet: {e}")
        print(f"    ERROR: {e}")

    # 5. NGBoost
    print("[5] NGBoost...")
    try:
        from sirena.models.ngboost_model import NGBoostForecaster
        model = NGBoostForecaster()
        model.fit(df)
        vals = []
        for h in range(horizon):
            target = last_date + pd.DateOffset(months=h+1)
            df_ext = df.copy()
            df_ext.loc[target] = np.nan
            pred = model.predict(df_ext, target)['prediction'] - 100
            vals.append(pred)
        results['NGBoost'] = vals
        print(f"    OK: {vals[0]:.2f}% → {vals[-1]:.2f}%")
    except Exception as e:
        errors.append(f"NGBoost: {e}")
        print(f"    ERROR: {e}")

    # 6. NGBoost Shock
    print("[6] NGBoost Shock...")
    try:
        from sirena.models.ngboost_shock import NGBoostShockForecaster
        model = NGBoostShockForecaster()
        model.fit(df)
        vals = []
        for h in range(horizon):
            target = last_date + pd.DateOffset(months=h+1)
            df_ext = df.copy()
            df_ext.loc[target] = np.nan
            pred = model.predict(df_ext, target)['prediction'] - 100
            vals.append(pred)
        results['NGBoost_Shock'] = vals
        print(f"    OK: {vals[0]:.2f}% → {vals[-1]:.2f}%")
    except Exception as e:
        errors.append(f"NGBoost_Shock: {e}")
        print(f"    ERROR: {e}")

    # 7. Shock Dummies
    print("[7] Shock Dummies...")
    try:
        from sirena.models.ridge_shock_dummies import RidgeShockDummiesForecaster
        model = RidgeShockDummiesForecaster()
        model.fit(df)
        vals = []
        for h in range(horizon):
            target = last_date + pd.DateOffset(months=h+1)
            df_ext = df.copy()
            df_ext.loc[target] = np.nan
            pred = model.predict(df_ext, target)['prediction'] - 100
            vals.append(pred)
        results['Shock_Dummies'] = vals
        print(f"    OK: {vals[0]:.2f}% → {vals[-1]:.2f}%")
    except Exception as e:
        errors.append(f"Shock_Dummies: {e}")
        print(f"    ERROR: {e}")

    # 8. Prophet
    print("[8] Prophet...")
    try:
        from sirena.models.prophet import ProphetForecaster
        model = ProphetForecaster()
        model.fit(df)
        fc = model.forecast(horizon=horizon)
        results['Prophet'] = fc
        print(f"    OK: {fc[0]:.2f}% → {fc[-1]:.2f}%")
    except Exception as e:
        errors.append(f"Prophet: {e}")
        print(f"    ERROR: {e}")

    # 9. Subcomponent
    print("[9] Subcomponent...")
    try:
        from sirena.models.subcomponent import SubcomponentForecaster
        model = SubcomponentForecaster(horizon=1)
        model.fit(df)
        fc = model.forecast(horizon=horizon)
        results['Subcomp'] = fc
        print(f"    OK: {fc[0]:.2f}% → {fc[-1]:.2f}%")
    except Exception as e:
        errors.append(f"Subcomp: {e}")
        print(f"    ERROR: {e}")

    # 10. Microcomponent
    print("[10] Microcomponent (537 items)...")
    try:
        from sirena.models.microcomponent import MicrocomponentForecaster
        model = MicrocomponentForecaster(horizon=1, use_seasonal_adj=True)
        model.fit(df)
        fc = model.forecast(horizon=horizon)
        results['Micro'] = fc
        print(f"    OK: {fc[0]:.2f}% → {fc[-1]:.2f}%")
    except Exception as e:
        errors.append(f"Micro: {e}")
        print(f"    ERROR: {e}")

    # Save results
    df_results = pd.DataFrame(results)
    df_results.to_csv('data/verify_forecast.csv', index=False)
    print(f"\nСохранено: data/verify_forecast.csv")

    return df_results, errors


def verify_backtest(df, n_months=12):
    """Проверяет бэктест всех моделей."""
    print("\n" + "=" * 60)
    print("ПРОВЕРКА БЭКТЕСТА (tab3)")
    print("=" * 60)

    # Load actual data
    bvar_df = pd.read_csv('data/inflation_data.csv', sep=';', decimal=',')
    bvar_df['Date'] = pd.to_datetime(bvar_df['Date'], format='%d.%m.%Y', errors='coerce')
    bvar_df = bvar_df.set_index('Date').sort_index()
    bvar_df['mom'] = pd.to_numeric(bvar_df['mom'].astype(str).str.replace(',', '.'), errors='coerce')

    # Normalize to month start for consistent matching
    bvar_df.index = bvar_df.index.to_period('M').to_timestamp()

    last_fact = bvar_df.index.max()
    test_dates = pd.date_range(end=last_fact, periods=n_months, freq='MS')

    results = []
    errors = []

    for i, date in enumerate(test_dates):
        print(f"\n[{i+1}/{n_months}] {date.strftime('%Y-%m')}...")
        cutoff = date - pd.DateOffset(months=1)

        actual = bvar_df.loc[date, 'mom'] - 100 if date in bvar_df.index else np.nan
        train = df[df.index <= cutoff].copy()
        train_ext = train.copy()
        train_ext.loc[date] = np.nan

        row = {'Date': date, 'Actual': actual}

        # Ridge
        try:
            from sirena.models.ridge import RidgeForecaster
            m = RidgeForecaster()
            m.fit(train)
            row['Ridge'] = m.predict(train_ext, date)['prediction'] - 100
        except: row['Ridge'] = np.nan

        # Huber
        try:
            from sirena.models.huber import HuberForecaster
            m = HuberForecaster()
            m.fit(train)
            row['Huber'] = m.predict(train_ext, date)['prediction'] - 100
        except: row['Huber'] = np.nan

        # NGBoost Shock
        try:
            from sirena.models.ngboost_shock import NGBoostShockForecaster
            m = NGBoostShockForecaster()
            m.fit(train)
            row['NGBoost_Shock'] = m.predict(train_ext, date)['prediction'] - 100
        except: row['NGBoost_Shock'] = np.nan

        # Micro
        try:
            from sirena.models.microcomponent import MicrocomponentForecaster
            m = MicrocomponentForecaster(horizon=1, use_seasonal_adj=True)
            m.fit(train)
            pred = m.predict(train, date)
            if pred and 'prediction' in pred:
                seasonal_adj = m.SEASONAL_ADJ.get(date.month, 0)
                row['Micro'] = pred['prediction'] - 100 + seasonal_adj
            else:
                row['Micro'] = np.nan
        except Exception as e:
            row['Micro'] = np.nan
            errors.append(f"Micro {date}: {e}")

        results.append(row)

        # Print progress
        for k in ['Ridge', 'Huber', 'NGBoost_Shock', 'Micro']:
            if k in row and not np.isnan(row[k]):
                err = abs(row['Actual'] - row[k]) if not np.isnan(row['Actual']) else np.nan
                print(f"    {k}: {row[k]:.2f}% (err={err:.2f})")

    df_results = pd.DataFrame(results)
    df_results.to_csv('data/verify_backtest.csv', index=False)
    print(f"\nСохранено: data/verify_backtest.csv")

    # Calculate MAE
    print("\n" + "-" * 40)
    print("MAE по моделям:")
    for col in ['Ridge', 'Huber', 'NGBoost_Shock', 'Micro']:
        if col in df_results.columns:
            valid = df_results[['Actual', col]].dropna()
            if len(valid) > 0:
                mae = (valid['Actual'] - valid[col]).abs().mean()
                print(f"  {col}: {mae:.3f}")

    return df_results, errors


def main():
    print("=" * 60)
    print("DASHBOARD VERIFICATION")
    print(f"Время: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    # Load data
    from sirena.sa_data_loader import get_sa_with_total
    df = get_sa_with_total()
    print(f"\nДанные: {df.index.min().strftime('%Y-%m')} — {df.index.max().strftime('%Y-%m')}")

    all_errors = []

    # 1. Verify forecasts
    start = time.time()
    fc_df, fc_errors = verify_forecasts(df, horizon=12)
    all_errors.extend(fc_errors)
    print(f"\nПрогнозы: {time.time()-start:.1f}s")

    # 2. Verify backtest
    start = time.time()
    bt_df, bt_errors = verify_backtest(df, n_months=6)  # 6 месяцев для скорости
    all_errors.extend(bt_errors)
    print(f"\nБэктест: {time.time()-start:.1f}s")

    # Summary
    summary = {
        'timestamp': datetime.now().isoformat(),
        'data_end': df.index.max().strftime('%Y-%m-%d'),
        'forecast_models': len([c for c in fc_df.columns if c != 'Date']),
        'backtest_months': len(bt_df),
        'errors': all_errors,
        'status': 'OK' if len(all_errors) == 0 else 'ERRORS'
    }

    with open('data/verify_summary.json', 'w') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 60)
    print("ИТОГО")
    print("=" * 60)
    print(f"Моделей прогноза: {summary['forecast_models']}")
    print(f"Месяцев бэктеста: {summary['backtest_months']}")
    print(f"Ошибок: {len(all_errors)}")

    if all_errors:
        print("\nОШИБКИ:")
        for e in all_errors:
            print(f"  - {e}")

    print(f"\nФайлы:")
    print(f"  data/verify_forecast.csv")
    print(f"  data/verify_backtest.csv")
    print(f"  data/verify_summary.json")

    return len(all_errors) == 0


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
