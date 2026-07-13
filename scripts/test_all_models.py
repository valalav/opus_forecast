#!/usr/bin/env python3
"""
SIRENA-KBR v4.8 - Тест всех моделей ансамбля
=============================================

Запуск: python3 scripts/test_all_models.py
"""

import sys
import os
import traceback
from pathlib import Path
from datetime import datetime
import json

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np

# Suppress warnings for cleaner output
import warnings
warnings.filterwarnings('ignore')


def test_model(model_class, model_name: str, df: pd.DataFrame, target_date: pd.Timestamp) -> dict:
    """Test a single model and return results."""
    result = {
        'model': model_name,
        'status': 'UNKNOWN',
        'fit_time': None,
        'predict_time': None,
        'prediction': None,
        'error': None
    }

    try:
        import time

        # Instantiate
        model = model_class()

        # Fit
        start = time.time()
        model.fit(df)
        result['fit_time'] = round(time.time() - start, 3)

        # Predict
        df_ext = df.copy()
        df_ext.loc[target_date] = np.nan

        start = time.time()
        pred = model.predict(df_ext, target_date)
        result['predict_time'] = round(time.time() - start, 3)

        if isinstance(pred, dict):
            result['prediction'] = round(pred.get('prediction', 0) - 100, 4)
        else:
            result['prediction'] = round(float(pred) - 100, 4)

        result['status'] = 'OK'

    except Exception as e:
        result['status'] = 'FAIL'
        result['error'] = str(e)
        traceback.print_exc()

    return result


def main():
    print("=" * 70)
    print("SIRENA-KBR v4.8 - Тест всех моделей ансамбля")
    print("=" * 70)
    print()

    # Load data
    from sirena import DataLoader
    loader = DataLoader()
    df = loader.load_monthly_kbr()

    if df is None:
        print("ERROR: Не удалось загрузить данные!")
        return 1

    print(f"Данные загружены: {len(df)} месяцев ({df.index.min()} - {df.index.max()})")

    # Target date = last month + 1
    last_date = df.dropna(subset=['Все товары и услуги']).index.max()
    target_date = last_date + pd.DateOffset(months=1)
    print(f"Прогноз на: {target_date.strftime('%Y-%m')}")
    print()

    # Models to test
    models_to_test = []

    # Production models (9)
    from sirena.models import (
        RidgeForecaster,
        RidgeExtendedForecaster,
        RidgeShockDummiesForecaster,
        HuberForecaster,
        ElasticNetForecaster,
        ProphetForecaster,
        EBMForecaster,
    )

    models_to_test.extend([
        (RidgeForecaster, 'Ridge'),
        (RidgeExtendedForecaster, 'RidgeExtended'),
        (RidgeShockDummiesForecaster, 'RidgeShock'),
        (HuberForecaster, 'Huber'),
        (ElasticNetForecaster, 'ElasticNet'),
        (ProphetForecaster, 'Prophet'),
        (EBMForecaster, 'EBM'),
    ])

    # Optional: NGBoost
    try:
        from sirena.models import NGBoostForecaster, NGBOOST_AVAILABLE
        if NGBOOST_AVAILABLE:
            models_to_test.append((NGBoostForecaster, 'NGBoost'))
    except ImportError:
        print("NGBoost not available")

    try:
        from sirena.models import NGBoostShockForecaster
        if NGBoostShockForecaster:
            models_to_test.append((NGBoostShockForecaster, 'NGBoostShock'))
    except ImportError:
        print("NGBoostShock not available")

    # Auxiliary models
    from sirena.models import (
        BVARForecaster,
        ETSForecaster,
        SARIMAForecaster,
        LightGBMForecaster,
    )

    models_to_test.extend([
        (BVARForecaster, 'BVAR'),
        (ETSForecaster, 'ETS'),
        (SARIMAForecaster, 'SARIMA'),
        (LightGBMForecaster, 'LightGBM'),
    ])

    # Run tests
    results = []
    print("-" * 70)
    print(f"{'Model':<20} {'Status':<8} {'Fit(s)':<10} {'Pred(s)':<10} {'MoM%':<10}")
    print("-" * 70)

    for model_class, model_name in models_to_test:
        result = test_model(model_class, model_name, df.copy(), target_date)
        results.append(result)

        status_icon = "✓" if result['status'] == 'OK' else "✗"
        fit_time = f"{result['fit_time']:.3f}" if result['fit_time'] else "-"
        pred_time = f"{result['predict_time']:.3f}" if result['predict_time'] else "-"
        prediction = f"{result['prediction']:.3f}" if result['prediction'] is not None else "-"

        print(f"{status_icon} {model_name:<18} {result['status']:<8} {fit_time:<10} {pred_time:<10} {prediction:<10}")

        if result['error']:
            print(f"   ERROR: {result['error'][:60]}...")

    print("-" * 70)

    # Summary
    ok_count = sum(1 for r in results if r['status'] == 'OK')
    fail_count = sum(1 for r in results if r['status'] == 'FAIL')

    print()
    print(f"ИТОГО: {ok_count} OK, {fail_count} FAIL из {len(results)} моделей")

    # Save results
    results_file = Path(__file__).parent.parent / 'logs' / 'model_test_results.json'
    results_file.parent.mkdir(exist_ok=True)

    with open(results_file, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'data_range': f"{df.index.min()} - {df.index.max()}",
            'target_date': target_date.strftime('%Y-%m-%d'),
            'results': results
        }, f, indent=2, ensure_ascii=False)

    print(f"\nРезультаты сохранены: {results_file}")

    return 0 if fail_count == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
