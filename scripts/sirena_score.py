#!/usr/bin/env python3
"""
SIRENA SCORE — Комплексная метрика качества моделей
====================================================

Формула:
  SIRENA_Score = Weighted_MAE × KPI_Penalty

  где:
    Weighted_MAE = 0.50 × MAE_h1 + 0.30 × MAE_h2 + 0.20 × MAE_h12
    KPI_Penalty = 2 - KPI_rate  (от 1.0 до 2.0)

Веса:
  - h=1 (50%) — главный КПЭ
  - h=2 (30%) — среднесрочный
  - h=12 (20%) — долгосрочный

Меньше Score = лучше

Запуск:
  python3 scripts/sirena_score.py

Результат:
  - archive/results/sirena_score_history.csv — динамика по датам
  - archive/results/sirena_score_summary.csv — итоговый рейтинг
  - assets/charts/sirena_score_dynamics.html — график динамики

Автор: Claude Code
Дата: 2025-12-30
"""

import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np
import warnings
from datetime import datetime
from typing import Dict, List, Tuple, Optional

warnings.filterwarnings('ignore')

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))
# Imports
from sirena.models.bvar import BayesianVAR
from sirena.models.ebm import EBMForecaster
from sirena.models.ets import ETSForecaster
from sirena.models.lightgbm import LightGBMForecaster
from sirena.models.prophet import ProphetForecaster
from sirena.models.ridge import RidgeForecaster
from sirena.models.ridge_extended import RidgeExtendedForecaster
from sirena.models.bayesian_ridge import BayesianRidgeForecaster
from sirena.models.elasticnet import ElasticNetForecaster
from sirena.models.huber import HuberForecaster
from sirena.models.ridge_shock_dummies import RidgeShockDummiesForecaster

# Optional imports
try:
    from sirena.models.ngboost_model import NGBoostForecaster
    NGBOOST_AVAILABLE = True
except ImportError:
    NGBOOST_AVAILABLE = False

try:
    from sirena.models.catboost_model import CatBoostForecaster
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False

try:
    from sirena.models.subcomponent import SubcomponentForecaster
    from sirena.models.subcomponent_multi import SubcomponentMultiForecaster
    SUBCOMP_AVAILABLE = True
except ImportError:
    SUBCOMP_AVAILABLE = False

try:
    from sirena.models.micro_arima import MicroARIMAForecaster
    MICRO_AVAILABLE = True
except ImportError:
    MICRO_AVAILABLE = False

from sirena.models.arima import SARIMAForecaster


# ============================================================================
# SIRENA SCORE CONFIGURATION
# ============================================================================

WEIGHTS = {
    'h1': 0.50,   # 50% weight for 1-month ahead
    'h2': 0.30,   # 30% weight for 2-months ahead
    'h12': 0.20,  # 20% weight for 12-months ahead
}

KPI_THRESHOLD = 0.5  # |error| <= 0.5 = hit

# Rolling window for calculating metrics (months)
ROLLING_WINDOW = 12


# ============================================================================
# DATA LOADING
# ============================================================================

def load_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load inflation data for backtest"""

    # BVAR data (source of truth)
    bvar_df = pd.read_csv(PROJECT_ROOT / 'data' / 'inflation_data.csv',
                          sep=';', decimal=',')

    # Fix columns
    for col in ['mom', 'Prod', 'Nonprod', 'Serv', 'usd_nom_i', 'Ruonia']:
        if col in bvar_df.columns:
            if bvar_df[col].dtype == object:
                bvar_df[col] = bvar_df[col].astype(str).str.replace(',', '.')
            bvar_df[col] = pd.to_numeric(bvar_df[col], errors='coerce')

    bvar_df['Date'] = pd.to_datetime(bvar_df['Date'], format='%d.%m.%Y', errors='coerce')
    bvar_df['Date'] = bvar_df['Date'].dt.to_period('M').dt.to_timestamp()
    bvar_df = bvar_df.set_index('Date').sort_index()

    # Ridge data
    try:
        ridge_df = pd.read_csv(PROJECT_ROOT / 'data' / 'infl_kbr.csv',
                               sep=';', decimal=',')
        if 'Day' in ridge_df.columns:
            ridge_df['Date'] = pd.to_datetime(ridge_df['Day'], format='%d.%m.%Y')
        ridge_df['MoM'] = pd.to_numeric(
            ridge_df['MoM'].astype(str).str.replace(',', '.'),
            errors='coerce'
        )
        ridge_df = ridge_df.pivot_table(
            index='Date', columns='Товар', values='MoM', aggfunc='first'
        )
        ridge_df = ridge_df.sort_index()
        ridge_df.index = ridge_df.index.to_period('M').to_timestamp()
    except:
        ridge_df = pd.DataFrame({
            'Все товары и услуги': bvar_df['mom'],
            'Продовольственные товары': bvar_df['Prod'],
            'Непродовольственные товары': bvar_df['Nonprod'],
            'Услуги': bvar_df['Serv']
        })

    # Add macro columns
    for col in ['usd_nom_i', 'Ki', 'Ruonia', 'Ki_i']:
        if col in bvar_df.columns:
            ridge_df[col] = bvar_df[col]

    return bvar_df, ridge_df


# ============================================================================
# MODEL FORECASTERS
# ============================================================================

def forecast_model(model_name: str, train_df: pd.DataFrame,
                   target_date: pd.Timestamp, horizon: int) -> float:
    """Get forecast from a model for specific horizon"""

    try:
        train_ext = train_df.copy()
        train_ext.loc[target_date] = np.nan

        if model_name == 'Ridge':
            model: RidgeForecaster = RidgeForecaster(use_macro=False)
            model.fit(train_df, 'Все товары и услуги')
            result = model.predict(train_ext, target_date)
            return result['prediction'] - 100

        elif model_name == 'Ridge_Ext':
            model = RidgeExtendedForecaster()
            model.fit(train_df, 'Все товары и услуги')
            result = model.predict(train_ext, target_date)
            return result['prediction'] - 100

        elif model_name == 'Bayes_Ridge':
            model = BayesianRidgeForecaster()
            model.fit(train_df, 'Все товары и услуги')
            result = model.predict(train_ext, target_date)
            return result['prediction'] - 100

        elif model_name == 'ElasticNet':
            model = ElasticNetForecaster()
            model.fit(train_df, 'Все товары и услуги')
            result = model.predict(train_ext, target_date)
            return result['prediction'] - 100

        elif model_name == 'Huber':
            model = HuberForecaster()
            model.fit(train_df, 'Все товары и услуги')
            result = model.predict(train_ext, target_date)
            return result['prediction'] - 100

        elif model_name == 'Ridge_Shock':
            model = RidgeShockDummiesForecaster(use_2022_dummy=False)
            model.fit(train_df, 'Все товары и услуги')
            result = model.predict(train_ext, target_date)
            return result['prediction'] - 100

        elif model_name == 'NGBoost' and NGBOOST_AVAILABLE:
            model = NGBoostForecaster()
            model.fit(train_df, 'Все товары и услуги')
            result = model.predict(train_ext, target_date)
            return result['prediction'] - 100

        elif model_name == 'EBM':
            model = EBMForecaster()
            model.fit(train_df, 'Все товары и услуги')
            fc = model.forecast(horizon=horizon)
            return fc[horizon - 1] - 100

        elif model_name == 'LightGBM':
            model = LightGBMForecaster()
            model.fit(train_df, 'Все товары и услуги')
            fc = model.forecast(horizon=horizon)
            return fc[horizon - 1]

        elif model_name == 'Prophet':
            model = ProphetForecaster()
            model.fit(train_df, 'Все товары и услуги')
            fc = model.forecast(horizon=horizon)
            return fc[horizon - 1]

        elif model_name == 'ETS':
            model = ETSForecaster()
            model.fit(train_df, 'Все товары и услуги')
            fc = model.forecast(horizon=horizon)
            return fc[horizon - 1]

        elif model_name == 'SARIMA':
            sarima_df = pd.DataFrame({
                'Все товары и услуги': train_df['Все товары и услуги'].dropna()
            })
            model: SARIMAForecaster = SARIMAForecaster()
            model.fit(sarima_df, 'Все товары и услуги')
            fc = model.forecast_with_intervals(horizon=horizon)
            return fc['mean'][horizon - 1]

        elif model_name == 'CatBoost' and CATBOOST_AVAILABLE:
            model = CatBoostForecaster()
            model.fit(train_df, 'Все товары и услуги')
            fc = model.forecast(horizon=horizon)
            return fc[horizon - 1] - 100

        elif model_name == 'Subcomp' and SUBCOMP_AVAILABLE:
            model = SubcomponentForecaster(horizon=horizon)
            model.fit(train_df, 'Все товары и услуги')
            result = model.predict(train_df, target_date)
            return result['prediction'] - 100

        elif model_name == 'Subcomp_Multi' and SUBCOMP_AVAILABLE:
            model = SubcomponentMultiForecaster(horizon=horizon)
            model.fit(train_df, 'Все товары и услуги')
            result = model.predict(train_df, target_date)
            return result['prediction'] - 100

        elif model_name == 'Micro' and MICRO_AVAILABLE:
            model = MicroARIMAForecaster(horizon=horizon)
            model.fit()
            result = model.predict(train_df, target_date)
            if result and 'prediction' in result and not np.isnan(result['prediction']):
                return result['prediction'] - 100
            return np.nan

    except Exception as e:
        pass

    return np.nan


# ============================================================================
# SIRENA SCORE CALCULATION
# ============================================================================

def calculate_sirena_score(mae_h1: float, mae_h2: float, mae_h12: float,
                           kpi_rate: float) -> float:
    """
    Calculate SIRENA Score

    Args:
        mae_h1: MAE for h=1
        mae_h2: MAE for h=2
        mae_h12: MAE for h=12
        kpi_rate: KPI hit rate (0 to 1)

    Returns:
        SIRENA Score (lower is better)
    """
    # Weighted MAE
    weighted_mae = (WEIGHTS['h1'] * mae_h1 +
                    WEIGHTS['h2'] * mae_h2 +
                    WEIGHTS['h12'] * mae_h12)

    # KPI penalty (1.0 to 2.0)
    kpi_penalty = 2.0 - min(max(kpi_rate, 0), 1)

    return weighted_mae * kpi_penalty


def run_extended_backtest(start_date: str = '2020-01-01') -> pd.DataFrame:
    """
    Run extended backtest from start_date to present

    Returns DataFrame with predictions for all models, all horizons, all dates
    """
    print(f"\n{'='*70}")
    print(f"SIRENA SCORE — Extended Backtest from {start_date}")
    print(f"{'='*70}\n")

    # Load data
    bvar_df, ridge_df = load_data()

    # Prepare BVAR data
    bvar_data = pd.DataFrame()
    bvar_data['CPI'] = bvar_df['mom'] - 100
    bvar_data['Food'] = bvar_df['Prod'] - 100
    bvar_data['NonFood'] = bvar_df['Nonprod'] - 100
    bvar_data['Services'] = bvar_df['Serv'] - 100
    bvar_data['USD'] = bvar_df['usd_nom_i'] - 100
    bvar_data['RUONIA'] = bvar_df['Ruonia']
    bvar_data = bvar_data.dropna()

    # Define models to test
    models = ['Ridge', 'Ridge_Ext', 'Bayes_Ridge', 'ElasticNet', 'Huber',
              'Ridge_Shock', 'EBM', 'LightGBM', 'Prophet', 'ETS', 'SARIMA']

    if NGBOOST_AVAILABLE:
        models.append('NGBoost')
    if CATBOOST_AVAILABLE:
        models.append('CatBoost')
    if SUBCOMP_AVAILABLE:
        models.extend(['Subcomp', 'Subcomp_Multi'])
    if MICRO_AVAILABLE:
        models.append('Micro')

    print(f"Models: {len(models)}")
    print(f"  {', '.join(models)}")

    # Get test dates
    start = pd.Timestamp(start_date)
    last_date = bvar_data.index.max()

    # For h=12, we need cutoff at least 12 months before last_date
    test_dates = pd.date_range(start=start, end=last_date, freq='MS')
    print(f"\nTest period: {start.strftime('%Y-%m')} to {last_date.strftime('%Y-%m')}")
    print(f"Test dates: {len(test_dates)}")

    # Results storage
    results = []

    # Progress
    total = len(test_dates) * len(models) * 3  # 3 horizons
    current = 0

    for target_date in test_dates:
        if target_date not in bvar_data.index:
            continue

        actual = bvar_data.loc[target_date, 'CPI']

        for horizon in [1, 2, 12]:
            # Cutoff = target_date - horizon months
            cutoff = target_date - pd.DateOffset(months=horizon)

            if cutoff < ridge_df.index.min():
                continue

            train_df = ridge_df[ridge_df.index <= cutoff].copy()

            if len(train_df) < 36:  # Min training size
                continue

            for model_name in models:
                current += 1

                # Get prediction
                pred = forecast_model(model_name, train_df, target_date, horizon)

                results.append({
                    'Date': target_date,
                    'Horizon': horizon,
                    'Model': model_name,
                    'Actual': actual,
                    'Prediction': pred,
                    'Error': actual - pred if not np.isnan(pred) else np.nan,
                    'AbsError': abs(actual - pred) if not np.isnan(pred) else np.nan,
                    'KPI_Hit': abs(actual - pred) <= KPI_THRESHOLD if not np.isnan(pred) else np.nan
                })

        # Progress every 6 months
        if target_date.month in [1, 7]:
            pct = current / total * 100
            print(f"  {target_date.strftime('%Y-%m')}: {pct:.0f}% complete")

    return pd.DataFrame(results)


def calculate_rolling_scores(results_df: pd.DataFrame,
                             window: int = ROLLING_WINDOW) -> pd.DataFrame:
    """
    Calculate rolling SIRENA Score for each model

    Uses rolling window of `window` months to calculate metrics
    """
    print(f"\nCalculating rolling SIRENA Scores (window={window} months)...")

    models = results_df['Model'].unique()
    dates = sorted(results_df['Date'].unique())

    scores = []

    for date in dates:
        # Rolling window: last `window` months up to `date`
        window_start = date - pd.DateOffset(months=window-1)
        window_data = results_df[
            (results_df['Date'] >= window_start) &
            (results_df['Date'] <= date)
        ]

        if len(window_data) == 0:
            continue

        for model in models:
            model_data = window_data[window_data['Model'] == model]

            # Calculate MAE for each horizon
            mae_h1 = model_data[model_data['Horizon'] == 1]['AbsError'].mean()
            mae_h2 = model_data[model_data['Horizon'] == 2]['AbsError'].mean()
            mae_h12 = model_data[model_data['Horizon'] == 12]['AbsError'].mean()

            # Calculate KPI rate (based on h=1 primarily)
            h1_data = model_data[model_data['Horizon'] == 1]
            kpi_hits = h1_data['KPI_Hit'].sum() if len(h1_data) > 0 else 0
            kpi_total = len(h1_data.dropna(subset=['KPI_Hit']))
            kpi_rate = kpi_hits / kpi_total if kpi_total > 0 else 0

            # Handle NaN MAEs
            if np.isnan(mae_h1):
                mae_h1 = 1.0  # Penalty for missing data
            if np.isnan(mae_h2):
                mae_h2 = 1.0
            if np.isnan(mae_h12):
                mae_h12 = 1.0

            # Calculate SIRENA Score
            score = calculate_sirena_score(mae_h1, mae_h2, mae_h12, kpi_rate)

            scores.append({
                'Date': date,
                'Model': model,
                'MAE_h1': mae_h1,
                'MAE_h2': mae_h2,
                'MAE_h12': mae_h12,
                'KPI_Rate': kpi_rate,
                'SIRENA_Score': score,
                'Window_Size': kpi_total
            })

    return pd.DataFrame(scores)


def generate_score_chart(scores_df: pd.DataFrame) -> str:
    """Generate interactive HTML chart of score dynamics"""

    try:
        import plotly.graph_objects as go
    except ImportError:
        print("Plotly not available, skipping chart")
        return None

    # Get unique models and dates
    models = scores_df['Model'].unique()

    # Color palette
    colors = [
        '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
        '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
        '#aec7e8', '#ffbb78', '#98df8a', '#ff9896', '#c5b0d5'
    ]

    fig = go.Figure()

    # Find best models for default visibility
    latest = scores_df[scores_df['Date'] == scores_df['Date'].max()]
    best_models = latest.nsmallest(5, 'SIRENA_Score')['Model'].tolist()

    for i, model in enumerate(models):
        model_data = scores_df[scores_df['Model'] == model].sort_values('Date')

        visible = True if model in best_models else 'legendonly'

        fig.add_trace(go.Scatter(
            x=model_data['Date'],
            y=model_data['SIRENA_Score'],
            name=model,
            line=dict(color=colors[i % len(colors)], width=2),
            mode='lines',
            visible=visible,
            hovertemplate=(
                f'<b>{model}</b><br>' +
                'Date: %{x}<br>' +
                'SIRENA Score: %{y:.3f}<br>' +
                '<extra></extra>'
            )
        ))

    fig.update_layout(
        title='SIRENA Score Dynamics (Lower = Better)',
        xaxis_title='Date',
        yaxis_title='SIRENA Score',
        hovermode='x unified',
        height=600,
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        updatemenus=[dict(
            type='buttons', direction='right', x=1, y=-0.15, xanchor='right',
            showactive=True, bgcolor='#f0f0f0', borderwidth=0,
            buttons=[
                dict(label='Top 5', method='restyle',
                     args=[{'visible': [m in best_models for m in models]}]),
                dict(label='All', method='restyle', args=[{'visible': True}]),
            ]
        )]
    )

    # Add navigation
    nav_html = """
<div style="background:#f8f9fa;padding:8px 15px;border-bottom:1px solid #ddd;font-family:Arial,sans-serif;font-size:13px;position:fixed;top:0;left:0;right:0;z-index:1000;display:flex;gap:15px;align-items:center;">
  <b style="color:#333;">СИРЕНА-КБР</b>
  <span style="color:#999;">|</span>
  <a href="sirena_score_dynamics.html" style="color:#e67e22;text-decoration:none;font-weight:bold;">SIRENA Score</a>
  <span style="color:#999;">|</span>
  <a href="backtest_h1_predictions.html" style="color:#1f77b4;text-decoration:none;">h=1</a>
  <a href="backtest_h2_predictions.html" style="color:#1f77b4;text-decoration:none;">h=2</a>
  <a href="backtest_h12_predictions.html" style="color:#1f77b4;text-decoration:none;">h=12</a>
  <span style="color:#999;">|</span>
  <a href="index.html" style="color:#999;text-decoration:none;margin-left:auto;">Главная</a>
</div>
<div style="height:45px;"></div>
"""

    html = fig.to_html(full_html=True, include_plotlyjs=True)
    html = html.replace('<body>', f'<body>{nav_html}')

    output_path = PROJECT_ROOT / 'assets' / 'charts' / 'sirena_score_dynamics.html'
    output_path.write_text(html)

    return str(output_path)


def main():
    """Main entry point"""

    # Run extended backtest
    results_df = run_extended_backtest(start_date='2020-01-01')

    # Save raw results
    results_path = PROJECT_ROOT / 'archive' / 'results' / 'sirena_score_raw.csv'
    results_df.to_csv(results_path, index=False)
    print(f"\nRaw results saved: {results_path}")

    # Calculate rolling scores
    scores_df = calculate_rolling_scores(results_df)

    # Save score history
    history_path = PROJECT_ROOT / 'archive' / 'results' / 'sirena_score_history.csv'
    scores_df.to_csv(history_path, index=False)
    print(f"Score history saved: {history_path}")

    # Calculate final summary (latest scores)
    latest = scores_df[scores_df['Date'] == scores_df['Date'].max()].copy()
    latest = latest.sort_values('SIRENA_Score')
    latest['Rank'] = range(1, len(latest) + 1)

    summary_path = PROJECT_ROOT / 'archive' / 'results' / 'sirena_score_summary.csv'
    latest.to_csv(summary_path, index=False)
    print(f"Summary saved: {summary_path}")

    # Print top 10
    print(f"\n{'='*70}")
    print("TOP 10 MODELS BY SIRENA SCORE")
    print(f"{'='*70}")
    print(f"\n{'Rank':<5} {'Model':<18} {'Score':>8} {'MAE h=1':>8} {'MAE h=2':>8} {'MAE h=12':>9} {'KPI%':>6}")
    print("-" * 70)

    for _, row in latest.head(10).iterrows():
        print(f"{int(row['Rank']):<5} {row['Model']:<18} {row['SIRENA_Score']:>8.3f} "
              f"{row['MAE_h1']:>8.3f} {row['MAE_h2']:>8.3f} {row['MAE_h12']:>9.3f} "
              f"{row['KPI_Rate']*100:>5.1f}%")

    # Generate chart
    chart_path = generate_score_chart(scores_df)
    if chart_path:
        print(f"\nChart saved: {chart_path}")

    print(f"\n{'='*70}")
    print("SIRENA SCORE CALCULATION COMPLETE")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    main()
