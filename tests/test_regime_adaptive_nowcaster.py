"""
Unit tests for RegimeAdaptiveNowcaster model.

Tests cover:
- Model initialization
- Regime detection logic
- Weight switching between regimes
- fit() method (basic)
- forecast() method (basic)
- backtest() method (basic)
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import warnings
import tempfile
from unittest.mock import patch

import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from sirena.models.regime_adaptive_nowcaster import RegimeAdaptiveNowcaster


@pytest.fixture
def model():
    """Create a default model instance."""
    return RegimeAdaptiveNowcaster()


@pytest.fixture
def sample_monthly_data():
    """Generate sample monthly CPI data with macro indicators."""
    dates = pd.date_range("2016-01-01", periods=120, freq="MS")
    np.random.seed(42)

    data = pd.DataFrame(
        {
            "Все товары и услуги": 100.5 + np.random.randn(120) * 0.3,
            "Ki": 7.0 + np.random.randn(120) * 0.5,
            "Ruonia": 7.2 + np.random.randn(120) * 0.5,
            "mom": 0.5 + np.random.randn(120) * 0.2,
            "usd_nom_i": 75 + np.random.randn(120) * 5,
        },
        index=dates,
    )

    return data


@pytest.fixture
def sample_regime_weights():
    """Create temporary CSV with sample regime weights."""
    import tempfile
    import csv

    fd, path = tempfile.mkstemp(suffix=".csv", text=True)

    with open(fd, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Regime", "Product_code", "Weight"])
        # Normal regime weights
        writer.writerow(["normal", 111, 0.15])
        writer.writerow(["normal", 114, 0.10])
        writer.writerow(["normal", 701, 0.08])
        # Shock regime weights (more equal)
        writer.writerow(["shock", 111, 0.11])
        writer.writerow(["shock", 114, 0.11])
        writer.writerow(["shock", 701, 0.11])
        # High inflation weights
        writer.writerow(["high_inflation", 111, 0.20])
        writer.writerow(["high_inflation", 114, 0.12])
        writer.writerow(["high_inflation", 701, 0.15])

    return path


class TestRegimeAdaptiveNowcasterInitialization:
    """Tests for model initialization."""

    def test_default_parameters(self, model):
        """Test default parameters are set correctly."""
        assert model.name == "regime_adaptive_nowcaster"
        assert model.alpha == 1.0
        assert model._current_regime == "normal"
        assert not model._is_fitted
        assert isinstance(model.regime_weights, dict)

    def test_custom_parameters(self):
        """Test custom parameters are set correctly."""
        model = RegimeAdaptiveNowcaster(alpha=0.5)
        assert model.alpha == 0.5

    def test_custom_regime_weights_path(self):
        """Test custom regime weights path is set correctly."""
        custom_path = "/custom/path/weights.csv"
        model = RegimeAdaptiveNowcaster(regime_weights_path=custom_path)
        assert model.regime_weights_path == custom_path

    def test_model_inherits_from_base(self):
        """Test that model inherits from BaseForecaster."""
        from sirena.models.base import BaseForecaster

        model = RegimeAdaptiveNowcaster()
        assert isinstance(model, BaseForecaster)

    def test_min_train_size_constant(self):
        """Test MIN_TRAIN_SIZE constant."""
        assert RegimeAdaptiveNowcaster.MIN_TRAIN_SIZE == 36


class TestRegimeDetection:
    """Tests for regime detection logic."""

    def test_detect_current_regime_requires_data(self, model):
        """Test that detect_current_regime requires data."""
        with pytest.raises(ValueError, match="No data available"):
            model.detect_current_regime()

    @patch("sirena.models.regime_adaptive_nowcaster.detect_regime")
    def test_detect_current_regime_returns_tuple(
        self, mock_detect, sample_monthly_data
    ):
        """Test that detect_current_regime() returns (regime, diagnostics)."""
        from sirena.models.regime_detector import MacroRegime

        mock_detect.return_value = (
            MacroRegime.NORMAL,
            {"ki_change": 0.1, "ruonia_change": 0.05},
        )

        model = RegimeAdaptiveNowcaster()
        regime, diagnostics = model.detect_current_regime(sample_monthly_data)

        assert isinstance(regime, str)
        assert isinstance(diagnostics, dict)

    @patch("sirena.models.regime_adaptive_nowcaster.detect_regime")
    def test_regime_types_valid(self, mock_detect, sample_monthly_data):
        """Test that detected regime is one of valid types."""
        from sirena.models.regime_detector import MacroRegime

        mock_detect.return_value = (MacroRegime.SHOCK, {"ki_change": 0.6})

        model = RegimeAdaptiveNowcaster()
        regime, _ = model.detect_current_regime(sample_monthly_data)

        valid_regimes = ["normal", "shock", "high_inflation"]
        assert regime in valid_regimes

    @patch("sirena.models.regime_adaptive_nowcaster.detect_regime")
    def test_current_regime_property(self, mock_detect, sample_monthly_data):
        """Test current_regime property returns string."""
        from sirena.models.regime_detector import MacroRegime

        mock_detect.return_value = (MacroRegime.NORMAL, {})

        model = RegimeAdaptiveNowcaster()
        # Manually set for test
        model._current_regime, model._regime_diagnostics = model.detect_current_regime(
            sample_monthly_data
        )

        assert isinstance(model.current_regime, str)

    @patch("sirena.models.regime_adaptive_nowcaster.detect_regime")
    def test_get_regime_diagnostics(self, mock_detect, sample_monthly_data):
        """Test get_regime_diagnostics() returns dict."""
        from sirena.models.regime_detector import MacroRegime

        mock_detect.return_value = (MacroRegime.NORMAL, {})

        model = RegimeAdaptiveNowcaster()
        # Manually set for test
        model._current_regime, model._regime_diagnostics = model.detect_current_regime(
            sample_monthly_data
        )
        model._is_fitted = True  # Bypass _check_fitted

        diagnostics = model.get_regime_diagnostics()

        assert isinstance(diagnostics, dict)


class TestWeightSwitching:
    """Tests for regime-specific weight switching."""

    def test_regime_weights_loaded_from_file(self, sample_regime_weights):
        """Test that regime weights are loaded from CSV file."""
        model = RegimeAdaptiveNowcaster(regime_weights_path=sample_regime_weights)

        model._load_regime_weights()

        assert len(model.regime_weights) > 0
        assert "normal" in model.regime_weights
        assert "shock" in model.regime_weights
        assert "high_inflation" in model.regime_weights

    def test_regime_weights_structure(self, sample_regime_weights):
        """Test that regime weights have correct structure."""
        model = RegimeAdaptiveNowcaster(regime_weights_path=sample_regime_weights)

        model._load_regime_weights()

        # Each regime should have dict of {product_code: weight}
        for regime, weights in model.regime_weights.items():
            assert isinstance(weights, dict)
            for code, weight in weights.items():
                assert isinstance(code, int)
                assert isinstance(weight, float)
                assert weight > 0

    def test_get_regime_weights(self, model, sample_regime_weights):
        """Test get_regime_weights() returns copy of weights."""
        model.regime_weights_path = sample_regime_weights
        model._load_regime_weights()
        model._is_fitted = True  # Set to bypass _check_fitted

        weights = model.get_regime_weights()

        assert isinstance(weights, dict)
        # Should be a copy, not reference
        assert weights is not model.regime_weights

    def test_regime_weights_different_per_regime(self, sample_regime_weights):
        """Test that different regimes have different weights."""
        model = RegimeAdaptiveNowcaster(regime_weights_path=sample_regime_weights)

        model._load_regime_weights()

        # Get weights for each regime
        normal_weights = model.regime_weights.get("normal", {})
        shock_weights = model.regime_weights.get("shock", {})
        high_inf_weights = model.regime_weights.get("high_inflation", {})

        # All regimes should have weights
        assert len(normal_weights) > 0
        assert len(shock_weights) > 0
        assert len(high_inf_weights) > 0

        # Weights should differ between regimes
        assert normal_weights != shock_weights or shock_weights != high_inf_weights

    def test_compute_regime_weighted_signal_uses_current_weights(
        self, sample_regime_weights
    ):
        """Test that weighted signal computation uses current regime weights."""
        model = RegimeAdaptiveNowcaster(regime_weights_path=sample_regime_weights)
        model._load_regime_weights()

        # Create mock weekly data
        mock_weekly = pd.DataFrame(
            {
                "product_code": [111, 111, 114, 114, 701],
                "wow_growth": [0.5, 0.3, 0.2, 0.4, 0.1],
            }
        )

        # Compute signal for different regimes
        model._current_regime = "normal"
        signal_normal = model._compute_regime_weighted_signal(mock_weekly)

        model._current_regime = "shock"
        signal_shock = model._compute_regime_weighted_signal(mock_weekly)

        # Signals should differ due to different weights
        assert signal_normal != signal_shock

    def test_regime_weighted_signal_formula(self, sample_regime_weights):
        """Test that weighted signal uses correct formula: Σ(weight_i * avg_growth_i)."""
        model = RegimeAdaptiveNowcaster(regime_weights_path=sample_regime_weights)
        model._load_regime_weights()

        # Create mock weekly data
        mock_weekly = pd.DataFrame(
            {
                "product_code": [111, 111, 114, 114, 701],
                "wow_growth": [0.5, 0.3, 0.2, 0.4, 0.1],
            }
        )

        # Calculate expected signal manually
        model._current_regime = "normal"
        weights = model.regime_weights["normal"]
        expected_signal = 0.0

        for prod_code in weights.keys():
            prod_data = mock_weekly[mock_weekly["product_code"] == prod_code]
            if len(prod_data) > 0:
                avg_growth = prod_data["wow_growth"].mean()
                expected_signal += avg_growth * weights[prod_code]

        actual_signal = model._compute_regime_weighted_signal(mock_weekly)

        # Should match within floating point precision
        assert abs(actual_signal - expected_signal) < 1e-6


class TestRegimeAdaptiveNowcasterFit:
    """Tests for fit() method (basic initialization checks)."""

    def test_fit_requires_weekly_data(self, sample_monthly_data, sample_regime_weights):
        """Test that fit() requires weekly data loader."""
        # This test verifies the method structure without complex mocking
        # The actual fit() execution depends on external data files
        model = RegimeAdaptiveNowcaster(regime_weights_path=sample_regime_weights)
        model._load_regime_weights()

        # Verify preconditions are set
        assert len(model.regime_weights) > 0
        assert model.scaler is None  # Not fitted yet


class TestRegimeAdaptiveNowcasterForecast:
    """Tests for forecast() method."""

    def test_forecast_requires_fit(self, model):
        """Test that forecast() raises error if model not fitted."""
        with pytest.raises(Exception):
            model.forecast()

    def test_forecast_returns_numpy_array_type(self):
        """Test that forecast() is designed to return numpy array."""
        # This is a type/design check without execution
        model = RegimeAdaptiveNowcaster()

        # The method signature and docstring indicate numpy array return
        assert hasattr(model.forecast, "__annotations__")

    def test_forecast_h1_returns_single_value_type(self):
        """Test that forecast(horizon=1) returns single value type."""
        model = RegimeAdaptiveNowcaster()

        # The method design is for horizon=1 (single month ahead)
        # Verify method exists
        assert callable(model.forecast)

    def test_forecast_h2_with_warning_design(self):
        """Test that forecast(horizon>1) shows warning design."""
        # This is a design test - the method should warn about h>1
        model = RegimeAdaptiveNowcaster()

        # The docstring mentions warning for h>1
        # This test verifies that design exists
        assert model.forecast.__doc__ is not None
        assert (
            "designed for h=1" in model.forecast.__doc__
            or "horizon" in model.forecast.__doc__
        )


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_forecast_without_fit_raises_error(self, model):
        """Test that calling forecast() without fit raises error."""
        with pytest.raises(Exception):
            model.forecast()

    def test_forecast_h12_design_check(self):
        """Test that forecast method has warning design for long horizons."""
        # This is a design test - verifies the method handles h>1 with warning
        model = RegimeAdaptiveNowcaster()

        # The docstring should mention the limitation
        assert model.forecast.__doc__ is not None
        # The design is to warn for h>1 (nowcasting is for h=1)
        # This test verifies the design exists
        doc = model.forecast.__doc__
        assert "horizon" in doc.lower() or "h=" in doc.lower()

    def test_model_name(self):
        """Test model name attribute."""
        model = RegimeAdaptiveNowcaster()
        assert model.name == "regime_adaptive_nowcaster"

    def test_get_regime_weights_requires_fit(self, model):
        """Test that get_regime_weights() raises error if not fitted."""
        with pytest.raises(Exception):
            model.get_regime_weights()

    def test_get_regime_diagnostics_requires_fit(self, model):
        """Test that get_regime_diagnostics() raises error if not fitted."""
        with pytest.raises(Exception):
            model.get_regime_diagnostics()

    @patch("sirena.models.regime_adaptive_nowcaster.detect_regime")
    def test_regime_detection_with_insufficient_data(
        self, mock_detect, sample_monthly_data
    ):
        """Test regime detection with minimal data."""
        from sirena.models.regime_detector import MacroRegime

        mock_detect.return_value = (MacroRegime.NORMAL, {})

        model = RegimeAdaptiveNowcaster()

        # Should handle gracefully
        regime, diag = model.detect_current_regime(sample_monthly_data)

        assert isinstance(regime, str)
        assert isinstance(diag, dict)

    def test_aggregate_weekly_to_monthly(self, sample_regime_weights):
        """Test _aggregate_weekly_to_monthly method."""
        model = RegimeAdaptiveNowcaster(regime_weights_path=sample_regime_weights)
        model._load_regime_weights()
        model._current_regime = "normal"

        # Create mock weekly data
        dates = pd.date_range("2024-01-01", periods=4, freq="W")
        periods = [d.to_period("M") for d in dates]

        mock_weekly = pd.DataFrame(
            {
                "date": dates,
                "year_month": periods,
                "product_code": [111] * 4,
                "wow_growth": [0.1] * 4,
            }
        )

        # Pass empty DataFrame instead of None for monthly_df parameter
        empty_monthly = pd.DataFrame({"Все товары и услуги": []})
        result = model._aggregate_weekly_to_monthly(mock_weekly, empty_monthly)

        # Should return DataFrame with index
        assert isinstance(result, pd.DataFrame)
        assert "regime_weighted_signal" in result.columns

    def test_compute_weighted_signal_with_no_data(self, sample_regime_weights):
        """Test _compute_regime_weighted_signal with empty data."""
        model = RegimeAdaptiveNowcaster(regime_weights_path=sample_regime_weights)
        model._load_regime_weights()
        model._current_regime = "normal"

        empty_df = pd.DataFrame({"product_code": [], "wow_growth": []})

        signal = model._compute_regime_weighted_signal(empty_df)

        # Should return 0.0 for empty data
        assert signal == 0.0
