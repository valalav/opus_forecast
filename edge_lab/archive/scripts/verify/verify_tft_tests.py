#!/usr/bin/env python3
"""
Verification script for Temporal Fusion Transformer (TFT) Model
============================================================

This script verifies that:
1. TFT is registered in ModelRegistry
2. TFTForecaster can be imported
3. Model can be fitted
4. Model can predict
5. Model can forecast
6. Model can backtest
7. Weights can be extracted (acceptance criterion for task 22)
8. Feature importance works
9. Attention weights work
10. Model info retrieval works
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def generate_test_data(with_exog=True):
    """Generate test data for TFT."""
    dates = pd.date_range("2015-01-01", periods=120, freq="MS")
    np.random.seed(42)

    if with_exog:
        data = pd.DataFrame(
            {
                "Все товары и услуги": 100.5 + np.random.randn(120) * 0.3,
                "brent": 75 + np.random.randn(120) * 5,
                "usd_nom_i": 80 + np.random.randn(120) * 3,
                "Ki": 16 + np.random.randn(120) * 0.5,
                "Продовольственные товары": 100.6 + np.random.randn(120) * 0.4,
                "Непродовольственные товары": 100.3 + np.random.randn(120) * 0.2,
                "Услуги": 100.4 + np.random.randn(120) * 0.3,
            },
            index=dates,
        )
    else:
        data = pd.DataFrame(
            {
                "Все товары и услуги": 100.5 + np.random.randn(120) * 0.3,
                "Продовольственные товары": 100.6 + np.random.randn(120) * 0.4,
                "Непродовольственные товары": 100.3 + np.random.randn(120) * 0.2,
                "Услуги": 100.4 + np.random.randn(120) * 0.3,
            },
            index=dates,
        )

    return data


def main():
    """Run all verification tests."""
    print("=" * 80)
    print("Temporal Fusion Transformer (TFT) Verification Script")
    print("=" * 80)
    print()

    all_passed = True
    tests_passed = 0
    tests_total = 10

    # Test 1: Model Registry Registration
    print("Test 1: TFT registered in ModelRegistry...")
    try:
        from sirena.models import ModelRegistry

        models = ModelRegistry.list_models()
        if "tft" in models:
            print("  ✅ TFT is registered in ModelRegistry")
            tests_passed += 1
        else:
            print("  ❌ TFT is NOT registered in ModelRegistry")
            all_passed = False
    except Exception as e:
        print(f"  ❌ Error checking registry: {e}")
        all_passed = False
    print()

    # Test 2: Model Importable
    print("Test 2: TemporalFusionForecaster can be imported...")
    try:
        from sirena.models import TemporalFusionForecaster

        model = TemporalFusionForecaster()
        if model.name == "tft":
            print("  ✅ TemporalFusionForecaster imported successfully")
            tests_passed += 1
        else:
            print(f"  ❌ Model name is '{model.name}', expected 'tft'")
            all_passed = False
    except Exception as e:
        print(f"  ❌ Error importing model: {e}")
        all_passed = False
    print()

    # Test 3: Model can be fitted
    print("Test 3: Model can be fitted...")
    try:
        from sirena.models import TemporalFusionForecaster

        data = generate_test_data(with_exog=True)
        model = TemporalFusionForecaster(
            hidden_layers=1,
            hidden_size=32,
            max_iter=50,
        )
        model.fit(data, "Все товары и услуги")

        if model.is_fitted:
            print("  ✅ Model fitted successfully")
            tests_passed += 1
        else:
            print("  ❌ Model not fitted after fit()")
            all_passed = False
    except Exception as e:
        print(f"  ❌ Error fitting model: {e}")
        all_passed = False
    print()

    # Test 4: Model can predict
    print("Test 4: Model can predict...")
    try:
        from sirena.models import TemporalFusionForecaster

        data = generate_test_data(with_exog=True)
        model = TemporalFusionForecaster(
            hidden_layers=1,
            hidden_size=32,
            max_iter=50,
        )
        model.fit(data, "Все товары и услуги")

        target_date = data.index[-10]
        result = model.predict(data, target_date)

        if "prediction" in result and not np.isnan(result["prediction"]):
            print("  ✅ Model predicts successfully")
            tests_passed += 1
        else:
            print("  ❌ Prediction result invalid")
            all_passed = False
    except Exception as e:
        print(f"  ❌ Error predicting: {e}")
        all_passed = False
    print()

    # Test 5: Model can forecast
    print("Test 5: Model can forecast (multi-horizon)...")
    try:
        from sirena.models import TemporalFusionForecaster

        data = generate_test_data(with_exog=True)
        model = TemporalFusionForecaster(
            hidden_layers=1,
            hidden_size=32,
            max_iter=50,
        )
        model.fit(data, "Все товары и услуги")

        horizon = 6
        forecast = model.forecast(horizon=horizon)

        if len(forecast) == horizon and not np.any(np.isnan(forecast)):
            print("  ✅ Model forecasts successfully")
            tests_passed += 1
        else:
            print(f"  ❌ Forecast invalid: len={len(forecast)}, expected={horizon}")
            all_passed = False
    except Exception as e:
        print(f"  ❌ Error forecasting: {e}")
        all_passed = False
    print()

    # Test 6: Model can backtest
    print("Test 6: Model can backtest...")
    try:
        from sirena.models import TemporalFusionForecaster

        data = generate_test_data(with_exog=True)
        model = TemporalFusionForecaster(
            hidden_layers=1,
            hidden_size=32,
            max_iter=50,
        )

        results = model.backtest(data, start_date="2018-01-01")

        if len(results) > 0 and "error" in results.columns:
            mae = (results["error"].abs()).mean()
            print(f"  ✅ Backtest successful (MAE: {mae:.4f}, n={len(results)})")
            tests_passed += 1
        else:
            print("  ❌ Backtest results invalid")
            all_passed = False
    except Exception as e:
        print(f"  ❌ Error backtesting: {e}")
        all_passed = False
    print()

    # Test 7: Weights can be extracted (ACCEPTANCE CRITERION for Task 22)
    print("Test 7: Weights can be extracted (Task 22 acceptance criterion)...")
    try:
        from sirena.models import TemporalFusionForecaster

        data = generate_test_data(with_exog=True)
        model = TemporalFusionForecaster(
            hidden_layers=1,
            hidden_size=32,
            max_iter=50,
        )
        model.fit(data, "Все товары и услуги")

        weights = model.get_weights()

        if (
            "attention_weights" in weights
            and "network_weights" in weights
            and isinstance(weights["attention_weights"], dict)
            and len(weights["attention_weights"]) > 0
        ):
            print(f"  ✅ Weights extracted successfully")
            print(
                f"      - Attention weights: {len(weights['attention_weights'])} features"
            )
            print(
                f"      - Network weights: {len(weights['network_weights']['layer_weights'])} layers"
            )
            tests_passed += 1
        else:
            print("  ❌ Weights extraction failed")
            all_passed = False
    except Exception as e:
        print(f"  ❌ Error extracting weights: {e}")
        all_passed = False
    print()

    # Test 8: Feature importance works
    print("Test 8: Feature importance extraction works...")
    try:
        from sirena.models import TemporalFusionForecaster

        data = generate_test_data(with_exog=True)
        model = TemporalFusionForecaster(
            hidden_layers=1,
            hidden_size=32,
            max_iter=50,
        )
        model.fit(data, "Все товары и услуги")

        importance = model.get_feature_importance()

        if (
            len(importance) > 0
            and "feature" in importance.columns
            and "importance" in importance.columns
            and "type" in importance.columns
        ):
            top_feature = importance.iloc[0]
            print(
                f"  ✅ Feature importance works (top: {top_feature['feature']}, imp: {top_feature['importance']:.4f})"
            )
            tests_passed += 1
        else:
            print("  ❌ Feature importance invalid")
            all_passed = False
    except Exception as e:
        print(f"  ❌ Error getting feature importance: {e}")
        all_passed = False
    print()

    # Test 9: Attention weights work
    print("Test 9: Attention weights extraction works...")
    try:
        from sirena.models import TemporalFusionForecaster

        data = generate_test_data(with_exog=True)
        model = TemporalFusionForecaster(
            hidden_layers=1,
            hidden_size=32,
            max_iter=50,
        )
        model.fit(data, "Все товары и услуги")

        attention = model.get_attention_weights()

        if attention is not None and isinstance(attention, dict) and len(attention) > 0:
            total = sum(attention.values())
            if total > 0.9 and total <= 1.1:
                print(
                    f"  ✅ Attention weights work (sum: {total:.4f}, n={len(attention)})"
                )
                tests_passed += 1
            else:
                print(f"  ❌ Attention weights don't sum to ~1.0: {total:.4f}")
                all_passed = False
        else:
            print("  ❌ Attention weights invalid")
            all_passed = False
    except Exception as e:
        print(f"  ❌ Error getting attention weights: {e}")
        all_passed = False
    print()

    # Test 10: Model info retrieval works
    print("Test 10: Model info retrieval works...")
    try:
        from sirena.models import TemporalFusionForecaster

        model = TemporalFusionForecaster(
            hidden_layers=2,
            hidden_size=64,
            activation="tanh",
        )

        # Get info before fitting
        info_before = model.get_model_info()

        # Fit and get info after fitting
        data = generate_test_data(with_exog=True)
        model.fit(data, "Все товары и услуги")

        info_after = model.get_model_info()

        if (
            "name" in info_before
            and "hidden_layers" in info_before
            and "hidden_size" in info_before
            and info_after["is_fitted"]
            and "n_features" in info_after
        ):
            print(f"  ✅ Model info works (n_features: {info_after['n_features']})")
            tests_passed += 1
        else:
            print("  ❌ Model info invalid")
            all_passed = False
    except Exception as e:
        print(f"  ❌ Error getting model info: {e}")
        all_passed = False
    print()

    # Summary
    print("=" * 80)
    print(f"VERIFICATION SUMMARY: {tests_passed}/{tests_total} tests passed")
    print("=" * 80)

    if all_passed:
        print()
        print("✅ ALL VERIFICATION TESTS PASSED!")
        print()
        print("Task 22 acceptance criteria met:")
        print("  - Weights can be extracted ✅")
        print("  - All model functionality works ✅")
        print()
        return 0
    else:
        print()
        print("❌ SOME TESTS FAILED")
        print()
        return 1


if __name__ == "__main__":
    sys.exit(main())
