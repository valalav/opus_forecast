#!/usr/bin/env python3
"""
ГЕНЕРАТОР ПРОГНОЗОВ ВСЕХ МОДЕЛЕЙ (v2)
=====================================

Использует итеративный прогноз через predict() для всех моделей.
Прогнозирует экзогенные переменные через ExogForecaster.

Запуск:
    python3 scripts/generate_forecasts.py

Результат: archive/results/forecasts_current.csv

Автор: Claude Code
"""

import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime

# Добавляем корень проекта
sys.path.insert(0, str(Path(__file__).parent.parent))

# Импорты моделей
from sirena.models.ridge import RidgeForecaster
from sirena.models.ridge_extended import RidgeExtendedForecaster
from sirena.models.ridge_macro import RidgeMacroForecaster
from sirena.models.ridge_shock_dummies import RidgeShockDummiesForecaster
from sirena.models.bayesian_ridge import BayesianRidgeForecaster
from sirena.models.elasticnet import ElasticNetForecaster
from sirena.models.huber import HuberForecaster
from sirena.models.ngboost_model import NGBoostForecaster
from sirena.models.ngboost_shock import NGBoostShockForecaster
from sirena.models.bvar import BVARForecaster
from sirena.models.arima import SARIMAForecaster
from sirena.models.lightgbm import LightGBMForecaster
from sirena.models.prophet import ProphetForecaster
from sirena.models.ets import ETSForecaster
from sirena.models.ebm import EBMForecaster
from sirena.models.catboost_model import CatBoostForecaster
from sirena.models.subcomponent import SubcomponentForecaster
from sirena.models.subcomponent_multi import SubcomponentMultiForecaster
from sirena.models.microcomponent import MicrocomponentForecaster
from sirena.exog_forecaster import ExogForecaster

PROJECT_ROOT = Path(__file__).parent.parent
RESULTS_DIR = PROJECT_ROOT / 'archive' / 'results'


def load_data():
    """Загрузить все данные."""
    # Ridge data (infl_kbr.csv)
    df = pd.read_csv(PROJECT_ROOT / 'data' / 'infl_kbr.csv', sep=';')
    df['MoM'] = df['MoM'].astype(str).str.replace(',', '.').astype(float)
    df['Day'] = pd.to_datetime(df['Day'], format='%d.%m.%Y')
    df['Day'] = df['Day'].dt.to_period('M').dt.to_timestamp()
    df_pivot = df.pivot_table(index='Day', columns='Товар', values='MoM', aggfunc='first')
    df_pivot = df_pivot.sort_index()
    for col in df_pivot.columns:
        df_pivot[col] = pd.to_numeric(df_pivot[col], errors='coerce')

    # Macro data (inflation_data.csv)
    macro_df = pd.read_csv(PROJECT_ROOT / 'data' / 'inflation_data.csv', sep=';', decimal=',')
    macro_df['Date'] = pd.to_datetime(macro_df['Date'], format='%d.%m.%Y', errors='coerce')
    macro_df['Date'] = macro_df['Date'].dt.to_period('M').dt.to_timestamp()
    macro_df = macro_df.set_index('Date').sort_index()

    # Add Brent prices
    brent_path = PROJECT_ROOT / 'data' / 'brent_prices.csv'
    if brent_path.exists():
        brent_df = pd.read_csv(brent_path)
        brent_df['Date'] = pd.to_datetime(brent_df['Date'])
        brent_df['Date'] = brent_df['Date'].dt.to_period('M').dt.to_timestamp()
        brent_df = brent_df.set_index('Date')
        if 'brent' in brent_df.columns:
            macro_df = macro_df.join(brent_df['brent'], how='left')

    # BVAR data
    bvar_data = pd.DataFrame(index=macro_df.index)
    bvar_data['CPI'] = macro_df['mom'] - 100
    bvar_data['Food'] = macro_df['Prod'] - 100
    bvar_data['NonFood'] = macro_df['Nonprod'] - 100
    bvar_data['Services'] = macro_df['Serv'] - 100
    if 'usd_nom_i' in macro_df.columns:
        bvar_data['USD'] = macro_df['usd_nom_i'] - 100
    if 'Ruonia' in macro_df.columns:
        bvar_data['RUONIA'] = macro_df['Ruonia']
    bvar_data = bvar_data.dropna()

    return df_pivot, macro_df, bvar_data


def iterative_forecast(model, df, horizon: int, target_col: str = 'Все товары и услуги') -> np.ndarray:
    """
    Итеративный прогноз через predict() для моделей которые его поддерживают.

    Args:
        model: Обученная модель с методом predict()
        df: DataFrame с историческими данными
        horizon: Горизонт прогноза
        target_col: Целевая колонка

    Returns:
        numpy array с прогнозами (MoM%)
    """
    last_date = df.index.max()
    predictions = []

    # Копия данных для итеративного прогноза
    df_work = df.copy()

    for h in range(horizon):
        target_date = last_date + pd.DateOffset(months=h+1)

        # ИСПРАВЛЕНИЕ: копируем предыдущую строку, а не ставим все NaN!
        # Это сохраняет компоненты (food, nonfood, services) для признаков
        df_ext = df_work.copy()
        prev_date = df_ext.index[-1]
        df_ext.loc[target_date] = df_ext.loc[prev_date].copy()
        df_ext.loc[target_date, target_col] = np.nan  # Только target = NaN
        df_ext = df_ext.sort_index()

        try:
            pred_result = model.predict(df_ext, target_date)
            pred = pred_result['prediction']

            # Конвертируем в MoM% если нужно
            if abs(pred) > 50:  # Это индекс 100.xx
                pred = pred - 100
        except Exception as e:
            # Fallback: используем seasonal_norm если есть, иначе среднее
            if hasattr(model, 'seasonal_norm') and model.seasonal_norm:
                month = target_date.month
                pred = model.seasonal_norm.get(month, 100.0) - 100
            else:
                pred = (df_work[target_col].tail(12).mean() - 100) if target_col in df_work.columns else 0.5

        predictions.append(pred)

        # Добавляем прогноз в данные для следующего шага
        # Обновляем ВСЕ колонки (используем предыдущее значение как прокси)
        if target_date not in df_work.index:
            df_work.loc[target_date] = df_work.loc[prev_date].copy()
        df_work.loc[target_date, target_col] = pred + 100

    return np.array(predictions)


def generate_forecasts(horizon: int = 12):
    """Сгенерировать прогнозы всех моделей."""
    print("Загрузка данных...")
    df_ridge, macro_df, bvar_data = load_data()

    last_date = df_ridge.index.max()
    print(f"Последняя дата: {last_date.strftime('%Y-%m')}")
    print(f"Горизонт: {horizon} месяцев\n")

    # Прогноз экзогенных переменных
    print("Прогноз экзогенных переменных...")
    exog = ExogForecaster()
    exog.fit(macro_df)
    exog_forecast = exog.forecast(horizon)
    print(f"  Экзогенные: {list(exog_forecast.columns)}\n")

    # Расширенные данные с прогнозом экзогенных
    macro_extended = pd.concat([macro_df, exog_forecast])
    macro_extended = macro_extended.sort_index()
    # Заполняем mom NaN для будущих дат
    macro_extended.loc[exog_forecast.index, 'mom'] = np.nan

    # Даты прогноза
    forecast_dates = pd.date_range(
        start=last_date + pd.DateOffset(months=1),
        periods=horizon,
        freq='MS'
    )

    results = {'Date': forecast_dates}

    # ===== Модели с итеративным прогнозом через predict() =====
    print("Модели с итеративным прогнозом:")

    # Ridge
    print(f"  Ridge...", end=' ', flush=True)
    try:
        model = RidgeForecaster()
        model.fit(df_ridge, 'Все товары и услуги')
        fc = iterative_forecast(model, df_ridge, horizon, 'Все товары и услуги')
        results['Ridge'] = fc
        print("OK")
    except Exception as e:
        print(f"FAIL: {e}")
        results['Ridge'] = [np.nan] * horizon

    # Ridge Extended
    print(f"  Ridge_Ext...", end=' ', flush=True)
    try:
        model = RidgeExtendedForecaster()
        model.fit(df_ridge, 'Все товары и услуги')
        fc = iterative_forecast(model, df_ridge, horizon, 'Все товары и услуги')
        results['Ridge_Ext'] = fc
        print("OK")
    except Exception as e:
        print(f"FAIL: {e}")
        results['Ridge_Ext'] = [np.nan] * horizon

    # Ridge Shock
    print(f"  Ridge_Shock...", end=' ', flush=True)
    try:
        model = RidgeShockDummiesForecaster()
        model.fit(df_ridge, 'Все товары и услуги')
        fc = iterative_forecast(model, df_ridge, horizon, 'Все товары и услуги')
        results['Ridge_Shock'] = fc
        print("OK")
    except Exception as e:
        print(f"FAIL: {e}")
        results['Ridge_Shock'] = [np.nan] * horizon

    # Bayesian Ridge
    print(f"  Bayes_Ridge...", end=' ', flush=True)
    try:
        model = BayesianRidgeForecaster()
        model.fit(df_ridge, 'Все товары и услуги')
        fc = iterative_forecast(model, df_ridge, horizon, 'Все товары и услуги')
        results['Bayes_Ridge'] = fc
        print("OK")
    except Exception as e:
        print(f"FAIL: {e}")
        results['Bayes_Ridge'] = [np.nan] * horizon

    # ElasticNet
    print(f"  ElasticNet...", end=' ', flush=True)
    try:
        model = ElasticNetForecaster()
        model.fit(df_ridge, 'Все товары и услуги')
        fc = iterative_forecast(model, df_ridge, horizon, 'Все товары и услуги')
        results['ElasticNet'] = fc
        print("OK")
    except Exception as e:
        print(f"FAIL: {e}")
        results['ElasticNet'] = [np.nan] * horizon

    # Huber
    print(f"  Huber...", end=' ', flush=True)
    try:
        model = HuberForecaster()
        model.fit(df_ridge, 'Все товары и услуги')
        fc = iterative_forecast(model, df_ridge, horizon, 'Все товары и услуги')
        results['Huber'] = fc
        print("OK")
    except Exception as e:
        print(f"FAIL: {e}")
        results['Huber'] = [np.nan] * horizon

    # NGBoost
    print(f"  NGBoost...", end=' ', flush=True)
    try:
        model = NGBoostForecaster()
        model.fit(df_ridge, 'Все товары и услуги')
        fc = iterative_forecast(model, df_ridge, horizon, 'Все товары и услуги')
        results['NGBoost'] = fc
        print("OK")
    except Exception as e:
        print(f"FAIL: {e}")
        results['NGBoost'] = [np.nan] * horizon

    # NGBoost Shock
    print(f"  NGBoost_Shock...", end=' ', flush=True)
    try:
        model = NGBoostShockForecaster()
        model.fit(df_ridge, 'Все товары и услуги')
        fc = iterative_forecast(model, df_ridge, horizon, 'Все товары и услуги')
        results['NGBoost_Shock'] = fc
        print("OK")
    except Exception as e:
        print(f"FAIL: {e}")
        results['NGBoost_Shock'] = [np.nan] * horizon

    # LightGBM
    print(f"  LightGBM...", end=' ', flush=True)
    try:
        model = LightGBMForecaster()
        model.fit(df_ridge, 'Все товары и услуги')
        fc = iterative_forecast(model, df_ridge, horizon, 'Все товары и услуги')
        results['LightGBM'] = fc
        print("OK")
    except Exception as e:
        print(f"FAIL: {e}")
        results['LightGBM'] = [np.nan] * horizon

    # EBM
    print(f"  EBM...", end=' ', flush=True)
    try:
        model = EBMForecaster()
        model.fit(df_ridge, 'Все товары и услуги')
        fc = iterative_forecast(model, df_ridge, horizon, 'Все товары и услуги')
        results['EBM'] = fc
        print("OK")
    except Exception as e:
        print(f"FAIL: {e}")
        results['EBM'] = [np.nan] * horizon

    # CatBoost
    print(f"  CatBoost...", end=' ', flush=True)
    try:
        model = CatBoostForecaster()
        model.fit(df_ridge, 'Все товары и услуги')
        fc = iterative_forecast(model, df_ridge, horizon, 'Все товары и услуги')
        results['CatBoost'] = fc
        print("OK")
    except Exception as e:
        print(f"FAIL: {e}")
        results['CatBoost'] = [np.nan] * horizon

    # ===== Модели с собственным forecast() =====
    print("\nМодели с собственным forecast():")

    # Prophet
    print(f"  Prophet...", end=' ', flush=True)
    try:
        model = ProphetForecaster()
        model.fit(df_ridge, 'Все товары и услуги')
        fc = model.forecast(horizon)
        fc = np.array(fc)
        if np.nanmean(np.abs(fc)) > 50:
            fc = fc - 100
        results['Prophet'] = fc
        print("OK")
    except Exception as e:
        print(f"FAIL: {e}")
        results['Prophet'] = [np.nan] * horizon

    # ETS
    print(f"  ETS...", end=' ', flush=True)
    try:
        model = ETSForecaster()
        model.fit(df_ridge, 'Все товары и услуги')
        fc = model.forecast(horizon)
        fc = np.array(fc)
        if np.nanmean(np.abs(fc)) > 50:
            fc = fc - 100
        results['ETS'] = fc
        print("OK")
    except Exception as e:
        print(f"FAIL: {e}")
        results['ETS'] = [np.nan] * horizon

    # Subcomponent models
    print(f"  Subcomp...", end=' ', flush=True)
    try:
        model = SubcomponentForecaster()
        model.fit(df_ridge, 'Все товары и услуги')
        fc = model.forecast(horizon)
        results['Subcomp'] = fc
        print("OK")
    except Exception as e:
        print(f"FAIL: {e}")
        results['Subcomp'] = [np.nan] * horizon

    print(f"  Subcomp_Multi...", end=' ', flush=True)
    try:
        model = SubcomponentMultiForecaster()
        model.fit(df_ridge, 'Все товары и услуги')
        fc = model.forecast(horizon)
        results['Subcomp_Multi'] = fc
        print("OK")
    except Exception as e:
        print(f"FAIL: {e}")
        results['Subcomp_Multi'] = [np.nan] * horizon

    print(f"  Micro...", end=' ', flush=True)
    try:
        model = MicrocomponentForecaster()
        model.fit(df_ridge, 'Все товары и услуги')
        fc = model.forecast(horizon)
        results['Micro'] = fc
        print("OK")
    except Exception as e:
        print(f"FAIL: {e}")
        results['Micro'] = [np.nan] * horizon

    # ===== Модели с макро-данными =====
    print("\nМодели с макро-данными:")

    # Ridge_Macro (без brent для стабильности)
    print(f"  Ridge_Macro...", end=' ', flush=True)
    try:
        macro_no_brent = macro_df.drop(columns=['brent'], errors='ignore')
        model = RidgeMacroForecaster()
        model.fit(macro_no_brent, 'mom')
        fc = model.forecast(horizon)
        results['Ridge_Macro'] = fc
        print("OK")
    except Exception as e:
        print(f"FAIL: {e}")
        results['Ridge_Macro'] = [np.nan] * horizon

    # BVAR
    print(f"  BVAR...", end=' ', flush=True)
    try:
        var_names = ['CPI', 'Food', 'NonFood', 'Services']
        if 'USD' in bvar_data.columns:
            var_names.append('USD')
        if 'RUONIA' in bvar_data.columns:
            var_names.append('RUONIA')
        bvar = BVARForecaster(lags=1, lambda1=0.2, var_names=var_names)
        bvar.fit(bvar_data, 'CPI')
        fc = bvar.forecast(horizon)
        results['BVAR'] = fc
        print("OK")
    except Exception as e:
        print(f"FAIL: {e}")
        results['BVAR'] = [np.nan] * horizon

    # SARIMA
    print(f"  SARIMA...", end=' ', flush=True)
    try:
        sarima = SARIMAForecaster()
        sarima.fit(df_ridge, 'Все товары и услуги')
        fc = sarima.forecast(horizon)
        results['SARIMA'] = fc
        print("OK")
    except Exception as e:
        print(f"FAIL: {e}")
        results['SARIMA'] = [np.nan] * horizon

    # Создаём DataFrame
    df_fc = pd.DataFrame(results)

    # Сохраняем
    output_path = RESULTS_DIR / 'forecasts_current.csv'
    df_fc.to_csv(output_path, index=False)
    print(f"\nПрогнозы сохранены: {output_path}")

    # Сохраняем метаданные
    meta = {
        'generated_at': datetime.now().isoformat(),
        'last_data_date': last_date.isoformat(),
        'horizon': horizon,
        'models': list(df_fc.columns[1:]),
        'exog_variables': list(exog_forecast.columns)
    }
    meta_path = RESULTS_DIR / 'forecasts_meta.json'
    import json
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    return df_fc


if __name__ == '__main__':
    os.chdir(PROJECT_ROOT)
    generate_forecasts()
