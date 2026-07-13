#!/usr/bin/env python3
"""
Verification script for MIDASForecaster.
Tests that the model is properly registered and functional.
"""

import sys
from pathlib import Path

# Add parent directory to path (opus_forecast)
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pandas as pd
import numpy as np
from datetime import datetime


def test_model_registration():
    """Test that MIDAS is registered in ModelRegistry."""
    from sirena.models import ModelRegistry

    models = ModelRegistry.list_models()
    assert "midas" in models, "MIDAS should be registered in ModelRegistry"
    print("✓ MIDAS is registered in ModelRegistry")


def test_model_import():
    """Test that MIDASForecaster can be imported."""
    from sirena.models import MIDASForecaster

    assert MIDASForecaster is not None, "MIDASForecaster should be importable"
    assert MIDASForecaster.name == "midas", "Model name should be 'midas'"
    print("✓ MIDASForecaster can be imported")


def test_basic_fit_predict():
    """Test basic fit and predict functionality."""
    from sirena.models import MIDASForecaster

    # Generate sample data
    dates = pd.date_range("2015-01-01", periods=120, freq="MS")
    np.random.seed(42)
    data = pd.DataFrame(
        {
            "Все товары и услуги": 100.5 + np.random.randn(120) * 0.3,
            "brent": 75 + np.random.randn(120) * 5,
            "usd_nom_i": 80 + np.random.randn(120) * 3,
            "Ki": 16 + np.random.randn(120) * 0.5,
        },
        index=dates,
    )

    # Fit model
    model = MIDASForecaster(weight_type="almon", poly_order=2)
    model.fit(data)
    assert model._is_fitted, "Model should be fitted"
    print("✓ Model can be fitted")

    # Predict
    target_date = pd.Timestamp("2024-01-01")
    result = model.predict(data, target_date)
    assert "prediction" in result, "Prediction should be returned"
    assert isinstance(result["prediction"], (int, float)), (
        "Prediction should be numeric"
    )
    print("✓ Model can predict")

    # Forecast
    forecast = model.forecast(6)
    assert len(forecast) == 6, "Forecast should return 6 predictions"
    print("✓ Model can forecast")


def test_weight_types():
    """Test all weight types work."""
    from sirena.models import MIDASForecaster

    dates = pd.date_range("2015-01-01", periods=120, freq="MS")
    np.random.seed(42)
    data = pd.DataFrame(
        {
            "Все товары и услуги": 100.5 + np.random.randn(120) * 0.3,
            "brent": 75 + np.random.randn(120) * 5,
        },
        index=dates,
    )

    for weight_type in ["almon", "exp", "beta", "normalized_exp"]:
        model = MIDASForecaster(weight_type=weight_type)
        model.fit(data)
        assert model._is_fitted, f"Model with {weight_type} should fit"

    print("✓ All weight types (almon, exp, beta, normalized_exp) work")


def test_midas_weights():
    """Test MIDAS weight functions produce valid outputs."""
    from sirena.models import MIDASForecaster

    model = MIDASForecaster(weight_type="almon", poly_order=2)

    # Test Almon weights
    weights, theta = model._get_midas_weights(8)
    assert len(weights) == 8, "Should have 8 weights"
    assert len(theta) == 3, "Almon should have 3 parameters (polynomial order 2)"
    print("✓ Almon weights work")

    # Test exponential weights
    model_exp = MIDASForecaster(weight_type="exp")
    weights_exp, theta_exp = model_exp._get_midas_weights(10)
    assert len(weights_exp) == 10, "Should have 10 weights"
    assert np.all(weights_exp > 0), "Exponential weights should be positive"
    print("✓ Exponential weights work")

    # Test normalized exponential weights
    model_norm = MIDASForecaster(weight_type="normalized_exp")
    weights_norm, _ = model_norm._get_midas_weights(10)
    np.testing.assert_allclose(weights_norm.sum(), 1.0, rtol=1e-10)
    print("✓ Normalized exponential weights sum to 1")


def test_feature_importance():
    """Test feature importance extraction."""
    from sirena.models import MIDASForecaster

    dates = pd.date_range("2015-01-01", periods=120, freq="MS")
    np.random.seed(42)
    data = pd.DataFrame(
        {
            "Все товары и услуги": 100.5 + np.random.randn(120) * 0.3,
            "brent": 75 + np.random.randn(120) * 5,
            "usd_nom_i": 80 + np.random.randn(120) * 3,
            "Ki": 16 + np.random.randn(120) * 0.5,
        },
        index=dates,
    )

    model = MIDASForecaster(weight_type="almon")
    model.fit(data)

    importance = model.get_feature_importance()
    assert isinstance(importance, pd.DataFrame), "Should return DataFrame"
    assert "feature" in importance.columns, "Should have feature column"
    assert "coefficient" in importance.columns, "Should have coefficient column"
    assert "abs_coef" in importance.columns, "Should have abs_coef column"
    print("✓ Feature importance works")


def test_backtest():
    """Test backtest functionality."""
    from sirena.models import MIDASForecaster

    dates = pd.date_range("2015-01-01", periods=120, freq="MS")
    np.random.seed(42)
    data = pd.DataFrame(
        {
            "Все товары и услуги": 100.5 + np.random.randn(120) * 0.3,
            "brent": 75 + np.random.randn(120) * 5,
        },
        index=dates,
    )

    model = MIDASForecaster(weight_type="almon")
    results = model.backtest(data, start_date="2023-01-01")

    assert isinstance(results, pd.DataFrame), "Should return DataFrame"
    if len(results) > 0:
        assert "date" in results.columns, "Should have date column"
        assert "actual" in results.columns, "Should have actual column"
        assert "prediction" in results.columns, "Should have prediction column"
        assert "error" in results.columns, "Should have error column"
        mae = (results["error"].abs()).mean()
        assert mae >= 0, "MAE should be non-negative"

    print("✓ Backtest works")


def test_model_info():
    """Test model information retrieval."""
    from sirena.models import MIDASForecaster

    model = MIDASForecaster(weight_type="almon", poly_order=3)
    info = model.get_model_info()

    assert info["name"] == "midas", "Name should be 'midas'"
    assert info["weight_type"] == "almon", "Weight type should match"
    assert info["poly_order"] == 3, "Poly order should match"
    assert info["is_fitted"] == False, "Should not be fitted yet"
    print("✓ Model info retrieval works")


def main():
    """Run all verification tests."""
    print("=" * 60)
    print("MIDASForecaster Verification")
    print("=" * 60)
    print()

    tests = [
        test_model_registration,
        test_model_import,
        test_basic_fit_predict,
        test_weight_types,
        test_midas_weights,
        test_feature_importance,
        test_backtest,
        test_model_info,
    ]

    failed = []
    for test in tests:
        try:
            test()
        except Exception as e:
            print(f"✗ {test.__name__} FAILED: {e}")
            failed.append((test.__name__, e))
            import traceback

            traceback.print_exc()

    print()
    print("=" * 60)
    if not failed:
        print("✅ ALL VERIFICATION TESTS PASSED!")
        print("=" * 60)
        return 0
    else:
        print(f"❌ {len(failed)} TEST(S) FAILED:")
        for name, error in failed:
            print(f"  - {name}: {error}")
        print("=" * 60)
        return 1


if __name__ == "__main__":
    sys.exit(main())
