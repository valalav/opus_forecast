#!/usr/bin/env python3
"""
PRECOMPUTE FORECASTS
====================
Generates forecasts from all models and saves to JSON for fast dashboard loading.

Run this script whenever you want to update the forecasts:
    python3 scripts/precompute_forecasts.py

Output: data/precomputed_forecasts.json
"""

import json
import time
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, cast

# Add project root to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from sirena.sa_data_loader import get_sa_with_total


def _to_float_array(forecast_values: Any) -> np.ndarray:
    raw_values = forecast_values.values if hasattr(forecast_values, 'values') else forecast_values
    return np.asarray(raw_values, dtype=float)


def _store_forecast(results: Dict[str, Any], model_key: str, forecast_values: Any) -> np.ndarray:
    """Normalize model forecast output and store it under the target key."""
    fc = _to_float_array(forecast_values)
    if fc[0] > 50:
        fc = fc - 100
    results['forecasts'][model_key] = fc.tolist()
    return fc


def _load_forecast_input_data() -> pd.DataFrame:
    """Load model input data and append fresh raw facts when SA data lags.

    The legacy forecast path is trained on ``sa_fl.csv``. Monthly facts arrive first in
    ``inflation_data.csv``; until official/derived SA rows are refreshed, append only the
    missing latest months from the source-of-truth file so forecast dates move forward.
    """
    sa_df = get_sa_with_total()

    raw = pd.read_csv('data/inflation_data.csv', sep=';', decimal=',', encoding='utf-8-sig')
    raw['Date'] = pd.to_datetime(raw['Date'], format='%d.%m.%Y', errors='coerce')
    raw['Date'] = raw['Date'].dt.to_period('M').dt.to_timestamp()

    source_cols = {
        'mom': 'Все товары и услуги',
        'Prod': 'Продовольственные товары',
        'Nonprod': 'Непродовольственные товары',
        'Serv': 'Услуги',
    }
    raw_indexed = raw.set_index('Date')
    raw_input = cast(pd.DataFrame, raw_indexed[list(source_cols)].copy())
    raw_input.columns = [source_cols[str(col)] for col in raw_input.columns]
    for col in raw_input.columns:
        raw_input[col] = pd.to_numeric(raw_input[col], errors='coerce')
    raw_input = raw_input.dropna(how='all').sort_index()

    sa_last = cast(pd.Timestamp, pd.to_datetime(pd.Index(sa_df.index)).max())
    raw_last = cast(pd.Timestamp, pd.to_datetime(pd.Index(raw_input.index)).max())

    if raw_last > sa_last:
        missing = raw_input[raw_input.index > sa_last]
        print(
            "  WARNING: sa_fl.csv lags inflation_data.csv; "
            f"appending raw source-of-truth rows for {len(missing)} month(s): "
            + ", ".join(
                cast(pd.Timestamp, pd.Timestamp(d)).strftime('%Y-%m')
                for d in missing.index
            )
        )
        sa_df = pd.concat([sa_df, missing], axis=0).sort_index()

    return cast(pd.DataFrame, sa_df)


def compute_all_forecasts(horizon: int = 12) -> Dict[str, Any]:
    """Compute forecasts from all models."""
    print("=" * 60)
    print("PRECOMPUTING FORECASTS")
    print("=" * 60)

    # Load data
    print("\n[1/6] Loading data...")
    df = _load_forecast_input_data()
    datetime_index = cast(pd.DatetimeIndex, pd.to_datetime(pd.Index(df.index)))
    last_date = cast(pd.Timestamp, datetime_index.max())
    print(f"  Last data point: {last_date.strftime('%Y-%m')}")

    results: Dict[str, Any] = {
        'generated_at': datetime.now().isoformat(),
        'last_data_date': last_date.strftime('%Y-%m-%d'),
        'horizon': horizon,
        'forecasts': {}
    }

    # Generate forecast dates
    forecast_dates = pd.date_range(
        start=last_date + pd.DateOffset(months=1),
        periods=horizon,
        freq='MS'
    )
    results['forecast_dates'] = [d.strftime('%Y-%m-%d') for d in forecast_dates]

    # =========================================================================
    # Model 1: Ridge (baseline)
    # =========================================================================
    print("\n[2/10] Computing Ridge forecast...")
    try:
        from sirena.models.ridge import RidgeForecaster
        start = time.time()
        ridge = RidgeForecaster()
        ridge.fit(df)
        ridge_fc = _store_forecast(results, 'Ridge', ridge.forecast(horizon=horizon))
        print(f"  Done in {time.time()-start:.1f}s")
        print(f"  Trajectory: {ridge_fc[0]:.2f}% → {ridge_fc[-1]:.2f}%")
    except Exception as e:
        print(f"  ERROR: {e}")
        results['forecasts']['Ridge'] = None

    # =========================================================================
    # Model 2: Huber
    # =========================================================================
    print("\n[3/10] Computing Huber forecast...")
    try:
        from sirena.models.huber import HuberForecaster
        start = time.time()
        model = HuberForecaster()
        model.fit(df)
        fc = _store_forecast(results, 'Huber', model.forecast(horizon=horizon))
        print(f"  Done in {time.time()-start:.1f}s")
        print(f"  Trajectory: {fc[0]:.2f}% → {fc[-1]:.2f}%")
    except Exception as e:
        print(f"  ERROR: {e}")
        results['forecasts']['Huber'] = None

    # =========================================================================
    # Model 3: RidgeShockDummies
    # =========================================================================
    print("\n[4/10] Computing RidgeShockDummies forecast...")
    try:
        from sirena.models.ridge_shock_dummies import RidgeShockDummiesForecaster
        start = time.time()
        model = RidgeShockDummiesForecaster()
        model.fit(df)
        fc = _store_forecast(results, 'RidgeShockDummies', model.forecast(horizon=horizon))
        print(f"  Done in {time.time()-start:.1f}s")
        print(f"  Trajectory: {fc[0]:.2f}% → {fc[-1]:.2f}%")
    except Exception as e:
        print(f"  ERROR: {e}")
        results['forecasts']['RidgeShockDummies'] = None

    # =========================================================================
    # Model 3b: Ridge Shock Rolling (First-wave derivative)
    # =========================================================================
    print("\n[4/13] Computing Ridge_Shock_Roll24 forecast...")
    try:
        from sirena.models.ridge_shock_rolling import RidgeShockRollingForecaster

        start = time.time()
        model = RidgeShockRollingForecaster(seasonality_window=24, use_2022_dummy=False)
        model.fit(df)
        fc = _store_forecast(results, 'Ridge_Shock_Roll24', model.forecast(horizon=horizon))
        print(f"  Done in {time.time()-start:.1f}s")
        print(f"  Trajectory: {fc[0]:.2f}% → {fc[-1]:.2f}%")
    except Exception as e:
        print(f"  ERROR: {e}")
        results['forecasts']['Ridge_Shock_Roll24'] = None

    # =========================================================================
    # Model 3c: Ridge Production Proxy (First-wave derivative)
    # =========================================================================
    print("\n[5/13] Computing Ridge_ProdProxy forecast...")
    try:
        from sirena.models.ridge_production_proxy import RidgeProductionProxyForecaster

        start = time.time()
        model = RidgeProductionProxyForecaster(use_2022_dummy=False)
        model.fit(df)
        fc = _store_forecast(results, 'Ridge_ProdProxy', model.forecast(horizon=horizon))
        print(f"  Done in {time.time()-start:.1f}s")
        print(f"  Trajectory: {fc[0]:.2f}% → {fc[-1]:.2f}%")
    except Exception as e:
        print(f"  ERROR: {e}")
        results['forecasts']['Ridge_ProdProxy'] = None

    # =========================================================================
    # Model 3d: Ridge Asymmetric ERPT Proxy (First-wave derivative)
    # =========================================================================
    print("\n[6/13] Computing Ridge_AsymERPT forecast...")
    try:
        from sirena.models.ridge_asymmetric_erpt_proxy import (
            RidgeAsymmetricERPTProxyForecaster,
        )

        start = time.time()
        model = RidgeAsymmetricERPTProxyForecaster(use_2022_dummy=False)
        model.fit(df)
        fc = _store_forecast(results, 'Ridge_AsymERPT', model.forecast(horizon=horizon))
        print(f"  Done in {time.time()-start:.1f}s")
        print(f"  Trajectory: {fc[0]:.2f}% → {fc[-1]:.2f}%")
    except Exception as e:
        print(f"  ERROR: {e}")
        results['forecasts']['Ridge_AsymERPT'] = None

    # =========================================================================
    # Model 4: ElasticNet
    # =========================================================================
    print("\n[7/13] Computing ElasticNet forecast...")
    try:
        from sirena.models.elasticnet import ElasticNetForecaster
        start = time.time()
        model = ElasticNetForecaster()
        model.fit(df)
        fc = _store_forecast(results, 'ElasticNet', model.forecast(horizon=horizon))
        print(f"  Done in {time.time()-start:.1f}s")
        print(f"  Trajectory: {fc[0]:.2f}% → {fc[-1]:.2f}%")
    except Exception as e:
        print(f"  ERROR: {e}")
        results['forecasts']['ElasticNet'] = None

    # =========================================================================
    # Model 5: NGBoostShock
    # =========================================================================
    print("\n[8/13] Computing NGBoostShock forecast...")
    try:
        from sirena.models.ngboost_shock import NGBoostShockForecaster
        start = time.time()
        model = NGBoostShockForecaster()
        model.fit(df)
        fc = _store_forecast(results, 'NGBoostShock', model.forecast(horizon=horizon))
        print(f"  Done in {time.time()-start:.1f}s")
        print(f"  Trajectory: {fc[0]:.2f}% → {fc[-1]:.2f}%")
    except Exception as e:
        print(f"  ERROR: {e}")
        results['forecasts']['NGBoostShock'] = None

    # =========================================================================
    # Model 6: NGBoost
    # =========================================================================
    print("\n[9/13] Computing NGBoost forecast...")
    try:
        from sirena.models.ngboost_model import NGBoostForecaster
        start = time.time()
        model = NGBoostForecaster()
        model.fit(df)
        fc = _store_forecast(results, 'NGBoost', model.forecast(horizon=horizon))
        print(f"  Done in {time.time()-start:.1f}s")
        print(f"  Trajectory: {fc[0]:.2f}% → {fc[-1]:.2f}%")
    except Exception as e:
        print(f"  ERROR: {e}")
        results['forecasts']['NGBoost'] = None
    
    # =========================================================================
    # Model 7: RidgeExtended
    # =========================================================================
    print("\n[10/13] Computing RidgeExtended forecast...")
    try:
        from sirena.models.ridge_extended import RidgeExtendedForecaster
        start = time.time()
        model = RidgeExtendedForecaster()
        model.fit(df)
        fc = _store_forecast(results, 'RidgeExtended', model.forecast(horizon=horizon))
        print(f"  Done in {time.time()-start:.1f}s")
        print(f"  Trajectory: {fc[0]:.2f}% → {fc[-1]:.2f}%")
    except Exception as e:
        print(f"  ERROR: {e}")
        results['forecasts']['RidgeExtended'] = None

    # =========================================================================
    # Model 8: EBM
    # =========================================================================
    print("\n[11/13] Computing EBM forecast...")
    try:
        from sirena.models.ebm import EBMForecaster
        start = time.time()
        model = EBMForecaster()
        model.fit(df)
        fc = _store_forecast(results, 'EBM', model.forecast(horizon=horizon))
        print(f"  Done in {time.time()-start:.1f}s")
        print(f"  Trajectory: {fc[0]:.2f}% → {fc[-1]:.2f}%")
    except Exception as e:
        print(f"  ERROR: {e}")
        results['forecasts']['EBM'] = None

    # =========================================================================
    # Model 9: Prophet
    # =========================================================================
    print("\n[12/13] Computing Prophet forecast...")
    try:
        from sirena.models.prophet import ProphetForecaster
        start = time.time()
        model = ProphetForecaster()
        model.fit(df)
        pred = _store_forecast(results, 'Prophet', model.forecast(horizon=horizon))
        print(f"  Done in {time.time() - start:.1f}s")
        print(f"  Trajectory: {pred[0]:.2f}% → {pred[-1]:.2f}%")
    except Exception as e:
        print(f"  ERROR: {e}")
        results['forecasts']['Prophet'] = None

    # =========================================================================
    # Model 10: Mandatory VAR-family policy
    # =========================================================================
    print("\n[13/14] Computing VARPolicy forecast...")
    try:
        from sirena.models.var_policy import VARPolicyForecaster
        start = time.time()
        model = VARPolicyForecaster()
        model.fit(df)
        fc = _store_forecast(results, 'VARPolicy', model.forecast(horizon=horizon))
        diagnostics = model.diagnostics()
        results.setdefault('diagnostics', {})['VARPolicy'] = diagnostics
        print(f"  Done in {time.time()-start:.1f}s")
        print(f"  Regime/policy: {diagnostics.get('last_regime')}")
        print(f"  Trajectory: {fc[0]:.2f}% → {fc[-1]:.2f}%")
    except Exception as e:
        print(f"  ERROR: {e}")
        results['forecasts']['VARPolicy'] = None

    # =========================================================================
    # Model 11: Mandatory factor-family policy
    # =========================================================================
    print("\n[14/15] Computing FactorPolicy forecast...")
    try:
        from sirena.models.factor_policy import FactorPolicyForecaster
        start = time.time()
        model = FactorPolicyForecaster()
        model.fit(df)
        fc = _store_forecast(results, 'FactorPolicy', model.forecast(horizon=horizon))
        diagnostics = model.diagnostics()
        results.setdefault('diagnostics', {})['FactorPolicy'] = diagnostics
        print(f"  Done in {time.time()-start:.1f}s")
        print(f"  Factors/explained variance: {diagnostics.get('n_factors')} / {diagnostics.get('pca_explained_variance_sum'):.2%}")
        print(f"  Trajectory: {fc[0]:.2f}% → {fc[-1]:.2f}%")
    except Exception as e:
        print(f"  ERROR: {e}")
        results['forecasts']['FactorPolicy'] = None

    # =========================================================================
    # 12. Nowcast (Auxiliary) + Weekly Bridge Diagnostics
    # =========================================================================
    print("\n[15/15] Computing Nowcast (Weekly Bridge)...")
    try:
        from sirena.data.weekly_bridge import (
            compute_weekly_bridge_for_months,
            weekly_model_blend_weights,
        )

        bridge_diagnostics = compute_weekly_bridge_for_months(results['forecast_dates'])
        results.setdefault('diagnostics', {})['weekly_bridge'] = bridge_diagnostics
        print(
            "  Weekly bridge source: "
            f"{bridge_diagnostics.get('source_file', 'unknown')} "
            f"({bridge_diagnostics.get('deduped_rows', 0)} deduped rows)"
        )

        nowcast_list: List[Optional[float]] = [None] * horizon
        by_month = bridge_diagnostics.get('by_month', {})

        for h_idx, forecast_date_str in enumerate(results['forecast_dates']):
            forecast_date = cast(pd.Timestamp, pd.Timestamp(forecast_date_str))
            month_key = forecast_date.strftime('%Y-%m')
            month_bridge = by_month.get(month_key, {})
            chain = month_bridge.get('chain') if isinstance(month_bridge, dict) else None
            if not chain:
                continue

            weekly_signal = chain.get('extrapolated_mom')
            weeks_count = int(chain.get('weeks_count') or 0)
            if weekly_signal is None or not np.isfinite(float(weekly_signal)):
                continue

            model_values = [
                v[h_idx]
                for k, v in results['forecasts'].items()
                if v is not None
                and len(v) > h_idx
                and v[h_idx] is not None
                and np.isfinite(v[h_idx])
                and k not in ['Nowcast', 'Micro', 'Ensemble']
            ]
            if model_values:
                model_proxy = float(np.mean(model_values))
                w_weekly, w_model = weekly_model_blend_weights(weeks_count)
            else:
                model_proxy = 0.0
                w_weekly, w_model = 1.0, 0.0

            nowcast_val = w_weekly * float(weekly_signal) + w_model * model_proxy
            nowcast_list[h_idx] = float(nowcast_val)
            month_bridge['nowcast_blend'] = {
                'model_proxy': float(model_proxy),
                'weekly_weight': float(w_weekly),
                'model_weight': float(w_model),
                'nowcast_mom': float(nowcast_val),
            }

            print(f"  Target {month_key} (h={h_idx + 1})")
            print(f"    Weekly bridge signal: {float(weekly_signal):+.3f}%")
            print(f"    Model proxy: {model_proxy:+.3f}%")
            print(f"    Weights: {int(w_weekly*100)}% weekly / {int(w_model*100)}% model")
            print(f"    NOWCAST: {nowcast_val:+.3f}%")

        if any(v is not None for v in nowcast_list):
            results['forecasts']['Nowcast'] = nowcast_list
        else:
            print("  No weekly bridge data found for forecast months.")

    except Exception as e:
        print(f"  ERROR calculating Weekly Bridge Nowcast: {e}")
        import traceback
        traceback.print_exc()

    # =========================================================================
    # 12. Micro (Auxiliary) # Modified by edit
    # =========================================================================
    print("\nComputing Micro forecast (Auxiliary)...")
    try:
        from sirena.models.microcomponent import MicrocomponentForecaster
        start = time.time()
        model = MicrocomponentForecaster(horizon=1)
        model.fit(df)
        fc = model.forecast(horizon=horizon)
        results['forecasts']['Micro'] = fc.tolist()
        print(f"  Done in {time.time()-start:.1f}s")
        print(f"  Trajectory: {fc[0]:.2f}% → {fc[-1]:.2f}%")
    except Exception as e:
        print(f"  ERROR: {e}")
        results['forecasts']['Micro'] = None

    # =========================================================================
    # 13. Micro_SM (external Linux statsmodels forecast)
    # =========================================================================
    print("\nComputing Micro_SM forecast (External statsmodels)...")
    try:
        from sirena.models.micro_statsmodels_external import MicroStatsmodelsExternalForecaster
        start = time.time()
        model = MicroStatsmodelsExternalForecaster(horizon=1)
        model.fit(df)
        fc = _store_forecast(results, 'Micro_SM', model.forecast(horizon=horizon))
        print(f"  Done in {time.time()-start:.1f}s")
        print(f"  Trajectory: {fc[0]:.2f}% → {fc[-1]:.2f}%")
    except Exception as e:
        print(f"  ERROR: {e}")
        results['forecasts']['Micro_SM'] = None

    # =========================================================================
    # Ensemble (weighted average v5.2)
    # =========================================================================
    print("\nComputing weighted Ensemble (v5.2)...")
    weights = {
        'Huber': 0.18,
        'RidgeShockDummies': 0.17,
        'ElasticNet': 0.17,
        'NGBoostShock': 0.16,
        'NGBoost': 0.12,
        'Ridge': 0.08,
        'RidgeExtended': 0.05,
        'Prophet': 0.04,
        'EBM': 0.03,
        # Mandatory VAR-family control model. Kept modest because ML models remain
        # stronger on h=1, while VARPolicy adds interpretable macro/seasonal signal.
        'VARPolicy': 0.03,
        # Mandatory factor-family control model selected from the factor research grid.
        'FactorPolicy': 0.03,
    }

    ensemble_fc = []
    for h in range(horizon):
        total_w = 0
        weighted_sum = 0
        for model, w in weights.items():
            fc = results['forecasts'].get(model)
            # Ensure we have valid data for this model
            if fc is not None and len(fc) > h and fc[h] is not None:
                weighted_sum += w * fc[h]
                total_w += w
        if total_w > 0:
            ensemble_fc.append(weighted_sum / total_w)
        else:
            ensemble_fc.append(0)

    results['forecasts']['Ensemble'] = ensemble_fc
    print(f"  Trajectory: {ensemble_fc[0]:.2f}% → {ensemble_fc[-1]:.2f}%")

    return results


def save_results(results, output_path):
    """Save results to JSON."""
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nSaved to: {output_path}")


def main():
    output_path = Path(__file__).parent.parent / 'data' / 'precomputed_forecasts.json'

    results = compute_all_forecasts(horizon=12)
    save_results(results, output_path)

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Generated: {results['generated_at']}")
    print(f"Data up to: {results['last_data_date']}")
    print(f"Horizon: {results['horizon']} months")
    print(f"Models: {len(results['forecasts'])}")

    print("\nTrajectories (h=1 → h=12):")
    for model, fc in results['forecasts'].items():
        if fc and fc[0] is not None:
            if fc[-1] is not None:
                print(f"  {model:12s}: {fc[0]:+.2f}% → {fc[-1]:+.2f}%")
            else:
                print(f"  {model:12s}: {fc[0]:+.2f}% (h=1 only)")

    print("\n" + "=" * 60)
    print("Done! Dashboard will use precomputed_forecasts.json")
    print("=" * 60)


if __name__ == '__main__':
    main()
