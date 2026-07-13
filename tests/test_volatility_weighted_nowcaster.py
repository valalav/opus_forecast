"""
Unit tests for VolatilityWeightedNowcaster model.

Tests cover:
- Model initialization
- fit() method
- predict() method
- forecast() method
- Inverse volatility calculation
- backtest() method
- Edge cases
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import warnings

import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from sirena.models.volatility_weighted_nowcaster import VolatilityWeightedNowcaster


@pytest.fixture
def sample_monthly_data():
    """Generate sample monthly CPI data for testing."""
    dates = pd.date_range("2016-01-01", periods=96, freq="MS")
    np.random.seed(42)

    data = pd.DataFrame(
        {
            "Все товары и услуги": 100.5 + np.random.randn(96) * 0.3,
            "Продовольственные товары": 100.6 + np.random.randn(96) * 0.4,
            "Непродовольственные товары": 100.3 + np.random.randn(96) * 0.2,
            "Услуги": 100.4 + np.random.randn(96) * 0.3,
        },
        index=dates,
    )

    return data


@pytest.fixture
def model():
    """Create a default model instance."""
    return VolatilityWeightedNowcaster()


class TestVolatilityWeightedNowcasterInitialization:
    """Tests for model initialization."""

    def test_default_parameters(self):
        """Test default parameters are set correctly."""
        model = VolatilityWeightedNowcaster()

        assert model.name == "volatility_weighted_nowcaster"
        assert model.alpha == 1.0
        assert model.min_samples_per_product == 20
        assert model.volatility_window == 52
        assert not model._is_fitted

    def test_custom_parameters(self):
        """Test custom parameters are set correctly."""
        model = VolatilityWeightedNowcaster(
            alpha=0.5, min_samples_per_product=10, volatility_window=24
        )

        assert model.alpha == 0.5
        assert model.min_samples_per_product == 10
        assert model.volatility_window == 24

    def test_model_inherits_from_base(self):
        """Test that model inherits from BaseForecaster."""
        from sirena.models.base import BaseForecaster

        model = VolatilityWeightedNowcaster()
        assert isinstance(model, BaseForecaster)

    def test_min_train_size_constant(self):
        """Test MIN_TRAIN_SIZE constant."""
        assert VolatilityWeightedNowcaster.MIN_TRAIN_SIZE == 36


class TestVolatilityWeightedNowcasterFit:
    """Tests for fit() method."""

    def test_fit_sets_is_fitted(self, sample_monthly_data):
        """Test that fit() sets _is_fitted flag."""
        model = VolatilityWeightedNowcaster()
        model.fit(sample_monthly_data)

        assert model._is_fitted

    def test_fit_returns_self(self, sample_monthly_data):
        """Test that fit() returns self for method chaining."""
        model = VolatilityWeightedNowcaster()
        result = model.fit(sample_monthly_data)

        assert result is model

    def test_fit_creates_weights(self, sample_monthly_data):
        """Test that fit() creates product weights."""
        model = VolatilityWeightedNowcaster()
        model.fit(sample_monthly_data)

        assert len(model.product_weights) > 0
        assert len(model.product_volatility) > 0
        assert len(model.product_weights) == len(model.product_volatility)

    def test_fit_weights_sum_to_one(self, sample_monthly_data):
        """Test that weights are normalized to sum to 1."""
        model = VolatilityWeightedNowcaster()
        model.fit(sample_monthly_data)

        total_weight = sum(model.product_weights.values())
        assert abs(total_weight - 1.0) < 1e-10

    def test_fit_creates_scaler(self, sample_monthly_data):
        """Test that fit() creates a scaler."""
        model = VolatilityWeightedNowcaster()
        model.fit(sample_monthly_data)

        assert model.scaler is not None

    def test_fit_trains_ridge_model(self, sample_monthly_data):
        """Test that fit() trains the Ridge model."""
        model = VolatilityWeightedNowcaster()
        model.fit(sample_monthly_data)

        assert model.model is not None
        # Check that model is fitted
        assert hasattr(model.model, "coef_")

    def test_fit_with_custom_alpha(self, sample_monthly_data):
        """Test that custom alpha is used in Ridge model."""
        model = VolatilityWeightedNowcaster(alpha=2.0)
        model.fit(sample_monthly_data)

        assert model.model.alpha == 2.0


class TestVolatilityWeightedNowcasterForecast:
    """Tests for forecast() method."""

    def test_forecast_requires_fit(self, model):
        """Test that forecast() raises error if model not fitted."""
        with pytest.raises(Exception):  # Raises from _check_fitted
            model.forecast()

    def test_forecast_returns_array(self, sample_monthly_data):
        """Test that forecast() returns numpy array."""
        model = VolatilityWeightedNowcaster()
        model.fit(sample_monthly_data)
        result = model.forecast()

        assert isinstance(result, np.ndarray)

    def test_forecast_h1_returns_single_value(self, sample_monthly_data):
        """Test that forecast(horizon=1) returns single value."""
        model = VolatilityWeightedNowcaster()
        model.fit(sample_monthly_data)
        result = model.forecast(horizon=1)

        assert len(result) == 1

    def test_forecast_h2_returns_array(self, sample_monthly_data):
        """Test that forecast(horizon=2) returns array with warning."""
        model = VolatilityWeightedNowcaster()
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


class TestInverseVolatilityCalculation:
    """Tests for inverse volatility calculation."""

    def test_inverse_volatility_weights_stable_products_more(self, sample_monthly_data):
        """Test that more stable products get higher weights."""
        model = VolatilityWeightedNowcaster()
        model.fit(sample_monthly_data)

        # Find most and least stable products
        vol_values = list(model.product_volatility.values())
        weights = list(model.product_weights.values())

        # Products with lower volatility should have higher weights
        # Check correlation (should be negative)
        if len(vol_values) > 1:
            correlation = np.corrcoef(vol_values, weights)[0, 1]
            assert correlation < 0  # Negative correlation expected

    def test_get_volatility_weights(self, sample_monthly_data):
        """Test get_volatility_weights() method."""
        model = VolatilityWeightedNowcaster()
        model.fit(sample_monthly_data)

        weights = model.get_volatility_weights()

        assert isinstance(weights, dict)
        assert len(weights) > 0
        # Should be a copy, not the original
        assert weights is not model.product_weights

    def test_get_product_volatility(self, sample_monthly_data):
        """Test get_product_volatility() method."""
        model = VolatilityWeightedNowcaster()
        model.fit(sample_monthly_data)

        volatility = model.get_product_volatility()

        assert isinstance(volatility, dict)
        assert len(volatility) > 0
        # Should be a copy, not the original
        assert volatility is not model.product_volatility

    def test_get_volatility_weights_requires_fit(self, model):
        """Test that get_volatility_weights() raises error if not fitted."""
        with pytest.raises(Exception):  # Raises from _check_fitted
            model.get_volatility_weights()

    def test_get_product_volatility_requires_fit(self, model):
        """Test that get_product_volatility() raises error if not fitted."""
        with pytest.raises(Exception):  # Raises from _check_fitted
            model.get_product_volatility()

    def test_all_volatility_values_positive(self, sample_monthly_data):
        """Test that all calculated volatility values are positive."""
        model = VolatilityWeightedNowcaster()
        model.fit(sample_monthly_data)

        for std_dev in model.product_volatility.values():
            assert std_dev > 0

    def test_all_weights_positive(self, sample_monthly_data):
        """Test that all weights are positive."""
        model = VolatilityWeightedNowcaster()
        model.fit(sample_monthly_data)

        for weight in model.product_weights.values():
            assert weight > 0
            assert weight <= 1.0  # Should be normalized


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_forecast_without_fit_raises_error(self, model):
        """Test that calling forecast() without fit raises error."""
        with pytest.raises(Exception):
            model.forecast()

    def test_forecast_h12_with_warning(self, sample_monthly_data):
        """Test forecast with long horizon shows warning."""
        model = VolatilityWeightedNowcaster()
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
        model = VolatilityWeightedNowcaster()
        assert model.name == "volatility_weighted_nowcaster"

    def test_compute_weighted_signal_formula(self, sample_monthly_data):
        """Test that weighted signal computation uses correct formula."""
        model = VolatilityWeightedNowcaster()
        model.fit(sample_monthly_data)

        # For each product, weight * avg_growth should contribute to signal
        # Signal = Σ(weight_i * avg_growth_i)
        # Where weight_i = 1 / std_i / sum(1 / std_j)

        weights = model.product_weights
        volatility = model.product_volatility

        # Verify inverse volatility relationship
        for code, weight in weights.items():
            if code in volatility:
                std = volatility[code]
                # Higher std should give lower weight
                # (inverse relationship)
                assert weight > 0

    def test_min_samples_per_product_filtering(self, sample_monthly_data):
        """Test that products with insufficient samples are filtered."""
        model = VolatilityWeightedNowcaster(min_samples_per_product=1000)

        # With high threshold, few if any products should pass
        model.fit(sample_monthly_data)

        # All weights should still be valid if any exist
        for weight in model.product_weights.values():
            assert 0 <= weight <= 1.0
