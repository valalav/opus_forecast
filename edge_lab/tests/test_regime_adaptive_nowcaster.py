"""
Unit tests for RegimeAdaptiveNowcaster model.

Tests cover:
- Model initialization
- Regime detection logic
- Weight switching between regimes
- fit() method
- forecast() method
- backtest() method
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import warnings
import tempfile

import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from sirena.models.regime_adaptive_nowcaster import RegimeAdaptiveNowcaster


@pytest.fixture
def sample_monthly_data():
    """Generate sample monthly CPI data with macro indicators."""
    dates = pd.date_range("2016-01-01", periods=96, freq="MS")
    np.random.seed(42)

    data = pd.DataFrame(
        {
            "Все товары и услуги": 100.5 + np.random.randn(96) * 0.3,
            "Ki": 7.0 + np.random.randn(96) * 0.5,
            "Ruonia": 7.2 + np.random.randn(96) * 0.5,
            "mom": 0.5 + np.random.randn(96) * 0.2,
            "usd_nom_i": 75 + np.random.randn(96) * 5,
        },
        index=dates,
    )

    return data


@pytest.fixture
def model():
    """Create a default model instance."""
    return RegimeAdaptiveNowcaster()


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

    def test_default_parameters(self):
        """Test default parameters are set correctly."""
        model = RegimeAdaptiveNowcaster()

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

    def test_detect_current_regime_returns_tuple(self, sample_monthly_data):
        """Test that detect_current_regime() returns (regime, diagnostics)."""
        model = RegimeAdaptiveNowcaster()
        model.fit(sample_monthly_data)

        regime, diagnostics = model.detect_current_regime()

        assert isinstance(regime, str)
        assert isinstance(diagnostics, dict)

    def test_regime_types_valid(self, sample_monthly_data):
        """Test that detected regime is one of valid types."""
        model = RegimeAdaptiveNowcaster()
        model.fit(sample_monthly_data)

        regime, _ = model.detect_current_regime()

        valid_regimes = ["normal", "shock", "high_inflation"]
        assert regime in valid_regimes

    def test_diagnostics_dict_contains_keys(self, sample_monthly_data):
        """Test that diagnostics contains expected keys."""
        model = RegimeAdaptiveNowcaster()
        model.fit(sample_monthly_data)

        _, diagnostics = model.detect_current_regime()

        # Should contain macro indicators
        assert isinstance(diagnostics, dict)
        assert len(diagnostics) > 0

    def test_current_regime_property(self, sample_monthly_data):
        """Test current_regime property returns string."""
        model = RegimeAdaptiveNowcaster()
        model.fit(sample_monthly_data)

        assert isinstance(model.current_regime, str)

    def test_get_regime_diagnostics(self, sample_monthly_data):
        """Test get_regime_diagnostics() returns dict."""
        model = RegimeAdaptiveNowcaster()
        model.fit(sample_monthly_data)

        diagnostics = model.get_regime_diagnostics()

        assert isinstance(diagnostics, dict)
        # Should be a copy, not reference
        assert diagnostics is not model._regime_diagnostics


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

    def test_get_regime_weights(self, sample_monthly_data):
        """Test get_regime_weights() returns copy of weights."""
        model = RegimeAdaptiveNowcaster()
        model.fit(sample_monthly_data)

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
        self, sample_monthly_data, sample_regime_weights
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


class TestRegimeAdaptiveNowcasterFit:
    """Tests for fit() method."""

    def test_fit_sets_is_fitted(self, sample_monthly_data):
        """Test that fit() sets _is_fitted flag."""
        model = RegimeAdaptiveNowcaster()
        model.fit(sample_monthly_data)

        assert model._is_fitted

    def test_fit_returns_self(self, sample_monthly_data):
        """Test that fit() returns self for method chaining."""
        model = RegimeAdaptiveNowcaster()
        result = model.fit(sample_monthly_data)

        assert result is model

    def test_fit_creates_scaler(self, sample_monthly_data):
        """Test that fit() creates a scaler."""
        model = RegimeAdaptiveNowcaster()
        model.fit(sample_monthly_data)

        assert model.scaler is not None

    def test_fit_trains_ridge_model(self, sample_monthly_data):
        """Test that fit() trains Ridge model."""
        model = RegimeAdaptiveNowcaster()
        model.fit(sample_monthly_data)

        assert model.model is not None
        # Check that model is fitted
        assert hasattr(model.model, "coef_")

    def test_fit_with_custom_alpha(self, sample_monthly_data):
        """Test that custom alpha is used in Ridge model."""
        model = RegimeAdaptiveNowcaster(alpha=2.0)
        model.fit(sample_monthly_data)

        assert model.model.alpha == 2.0


class TestRegimeAdaptiveNowcasterForecast:
    """Tests for forecast() method."""

    def test_forecast_requires_fit(self, model):
        """Test that forecast() raises error if model not fitted."""
        with pytest.raises(Exception):  # Raises from _check_fitted
            model.forecast()

    def test_forecast_returns_array(self, sample_monthly_data):
        """Test that forecast() returns numpy array."""
        model = RegimeAdaptiveNowcaster()
        model.fit(sample_monthly_data)
        result = model.forecast()

        assert isinstance(result, np.ndarray)

    def test_forecast_h1_returns_single_value(self, sample_monthly_data):
        """Test that forecast(horizon=1) returns single value."""
        model = RegimeAdaptiveNowcaster()
        model.fit(sample_monthly_data)
        result = model.forecast(horizon=1)

        assert len(result) == 1

    def test_forecast_h2_returns_array_with_warning(self, sample_monthly_data):
        """Test that forecast(horizon=2) returns array with warning."""
        model = RegimeAdaptiveNowcaster()
        model.fit(sample_monthly_data)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = model.forecast(horizon=2)

            # Should issue warning about h>1
            assert len(w) == 1
            assert "designed for h=1" in str(w[0].message)

        # Returns array of zeros (persistence)
        assert len(result) == 2
        assert all(result == 0.0)


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_forecast_without_fit_raises_error(self, model):
        """Test that calling forecast() without fit raises error."""
        with pytest.raises(Exception):
            model.forecast()

    def test_forecast_h12_with_warning(self, sample_monthly_data):
        """Test forecast with long horizon shows warning."""
        model = RegimeAdaptiveNowcaster()
        model.fit(sample_monthly_data)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = model.forecast(horizon=12)

            # Should issue warning
            assert len(w) >= 1
            assert "designed for h=1" in str(w[0].message)

        # Should return zeros for all periods
        assert len(result) == 12
        assert all(result == 0.0)

    def test_model_name(self):
        """Test model name attribute."""
        model = RegimeAdaptiveNowcaster()
        assert model.name == "regime_adaptive_nowcaster"

    def test_get_regime_weights_requires_fit(self, model):
        """Test that get_regime_weights() raises error if not fitted."""
        with pytest.raises(Exception):  # Raises from _check_fitted
            model.get_regime_weights()

    def test_get_regime_diagnostics_requires_fit(self, model):
        """Test that get_regime_diagnostics() raises error if not fitted."""
        with pytest.raises(Exception):  # Raises from _check_fitted
            model.get_regime_diagnostics()

    def test_regime_detection_with_insufficient_data(self):
        """Test regime detection with insufficient macro data."""
        model = RegimeAdaptiveNowcaster()

        # Create minimal data
        dates = pd.date_range("2016-01-01", periods=10, freq="MS")
        data = pd.DataFrame({"mom": np.random.randn(10)}, index=dates)

        # Should handle gracefully or raise appropriate error
        try:
            model.fit(data)
            # If fit succeeds, regime detection should still work
            regime, diag = model.detect_current_regime()
            assert isinstance(regime, str)
        except Exception as e:
            # If fails, should be a reasonable error
            assert isinstance(e, (ValueError, KeyError))
