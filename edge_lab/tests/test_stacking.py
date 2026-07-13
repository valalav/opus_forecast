"""
Unit tests for StackingRegressorForecaster

Tests cover:
- Model initialization and parameters
- fit() with sample data
- predict() method
- forecast() method
- get_meta_weights() to verify meta-learner weights
- Edge cases: insufficient data, unfitted model
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import sys

# Add parent directory to path to import sirena module
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent.parent))

from sirena.models.stacking_regressor import StackingRegressorForecaster


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
def stacking_model():
    """Create default StackingRegressorForecaster instance."""
    return StackingRegressorForecaster()


class TestStackingRegressorInitialization:
    """Tests for StackingRegressorForecaster initialization."""

    def test_default_parameters(self, stacking_model):
        """Test default parameters are set correctly."""
        assert stacking_model.name == "stacking_regressor"
        assert stacking_model.ridge_alpha == 0.3
        assert stacking_model.ebm_max_bins == 256
        assert stacking_model.OUTLIER_YEARS == [2022, 2010]
        assert stacking_model.MIN_TRAIN_SIZE == 36

    def test_custom_parameters(self):
        """Test custom parameters are set correctly."""
        model = StackingRegressorForecaster(ridge_alpha=0.5, ebm_max_bins=128)
        assert model.ridge_alpha == 0.5
        assert model.ebm_max_bins == 128

    def test_base_features_defined(self, stacking_model):
        """Test that BASE_FEATURES are defined."""
        assert len(stacking_model.BASE_FEATURES) == 9
        assert "y_lag1" in stacking_model.BASE_FEATURES
        assert "month_sin" in stacking_model.BASE_FEATURES
        assert "food_lag1" in stacking_model.BASE_FEATURES


class TestStackingRegressorFit:
    """Tests for StackingRegressorForecaster fit method."""

    def test_fit_creates_stacking_model(self, stacking_model, sample_monthly_data):
        """Test that fit creates stacking model."""
        stacking_model.fit(sample_monthly_data)

        assert stacking_model.stacking is not None
        assert stacking_model.scaler is not None
        assert stacking_model.seasonal_norm is not None

    def test_fit_sets_fitted_flag(self, stacking_model, sample_monthly_data):
        """Test that fit sets _is_fitted flag."""
        assert not stacking_model._is_fitted

        stacking_model.fit(sample_monthly_data)

        assert stacking_model._is_fitted

    def test_fit_saves_train_date(self, stacking_model, sample_monthly_data):
        """Test that fit saves last train date."""
        stacking_model.fit(sample_monthly_data)

        assert stacking_model._last_train_date == sample_monthly_data.index.max()
        assert stacking_model._target_col == "Все товары и услуги"

    def test_fit_with_sufficient_data(self, stacking_model, sample_monthly_data):
        """Test fit with sufficient data."""
        # Should not raise exception
        stacking_model.fit(sample_monthly_data)
        assert stacking_model._is_fitted

    def test_fit_with_insufficient_data(self, stacking_model):
        """Test fit raises error with insufficient data."""
        dates = pd.date_range("2020-01-01", periods=24, freq="MS")
        data = pd.DataFrame(
            {"Все товары и услуги": 100.0 + np.random.randn(24) * 0.3},
            index=dates,
        )

        with pytest.raises(ValueError, match="Недостаточно данных"):
            stacking_model.fit(data)


class TestStackingRegressorPredict:
    """Tests for StackingRegressorForecaster predict method."""

    def test_predict_returns_dict(self, stacking_model, sample_monthly_data):
        """Test that predict returns dict with expected keys."""
        stacking_model.fit(sample_monthly_data)

        target_date = sample_monthly_data.index[-1]
        result = stacking_model.predict(sample_monthly_data, target_date)

        assert isinstance(result, dict)
        assert "prediction" in result
        assert "pred_stacking" in result
        assert "pred_ets" in result
        assert "date" in result
        assert "model" in result

    def test_predict_without_fit_raises_error(
        self, stacking_model, sample_monthly_data
    ):
        """Test that predict without fit raises error."""
        with pytest.raises(ValueError):
            stacking_model.predict(sample_monthly_data, sample_monthly_data.index[0])

    def test_predict_returns_numeric_value(self, stacking_model, sample_monthly_data):
        """Test that predict returns numeric prediction."""
        stacking_model.fit(sample_monthly_data)

        target_date = sample_monthly_data.index[-1]
        result = stacking_model.predict(sample_monthly_data, target_date)

        assert isinstance(result["prediction"], (int, float, np.number))
        assert not np.isnan(result["prediction"])

    def test_predict_date_matches_input(self, stacking_model, sample_monthly_data):
        """Test that predict date matches input date."""
        stacking_model.fit(sample_monthly_data)

        target_date = sample_monthly_data.index[-1]
        result = stacking_model.predict(sample_monthly_data, target_date)

        assert result["date"] == target_date


class TestStackingRegressorForecast:
    """Tests for StackingRegressorForecaster forecast method."""

    def test_forecast_returns_array(self, stacking_model, sample_monthly_data):
        """Test that forecast returns numpy array."""
        stacking_model.fit(sample_monthly_data)

        horizon = 12
        predictions = stacking_model.forecast(horizon)

        assert isinstance(predictions, np.ndarray)
        assert len(predictions) == horizon

    def test_forecast_without_fit_raises_error(self, stacking_model):
        """Test that forecast without fit raises error."""
        with pytest.raises(ValueError):
            stacking_model.forecast(12)

    def test_forecast_default_horizon(self, stacking_model, sample_monthly_data):
        """Test forecast with default horizon."""
        stacking_model.fit(sample_monthly_data)

        predictions = stacking_model.forecast()

        assert len(predictions) == 12

    def test_forecast_custom_horizon(self, stacking_model, sample_monthly_data):
        """Test forecast with custom horizon."""
        stacking_model.fit(sample_monthly_data)

        for h in [6, 12, 24]:
            predictions = stacking_model.forecast(h)
            assert len(predictions) == h

    def test_forecast_contains_numeric_values(
        self, stacking_model, sample_monthly_data
    ):
        """Test that forecast contains valid numeric values."""
        stacking_model.fit(sample_monthly_data)

        predictions = stacking_model.forecast(12)

        assert all(isinstance(p, (int, float, np.number)) for p in predictions)
        assert not any(np.isnan(predictions))


class TestStackingRegressorMetaWeights:
    """Tests for StackingRegressorForecaster get_meta_weights method."""

    def test_get_meta_weights_returns_dict(self, stacking_model, sample_monthly_data):
        """Test that get_meta_weights returns dict."""
        stacking_model.fit(sample_monthly_data)

        weights = stacking_model.get_meta_weights()

        assert isinstance(weights, dict)
        assert len(weights) > 0

    def test_get_meta_weights_without_fit_raises_error(self, stacking_model):
        """Test that get_meta_weights without fit raises error."""
        with pytest.raises(ValueError):
            stacking_model.get_meta_weights()

    def test_get_meta_weights_contains_base_models(
        self, stacking_model, sample_monthly_data
    ):
        """Test that meta weights contain base model coefficients."""
        stacking_model.fit(sample_monthly_data)

        weights = stacking_model.get_meta_weights()

        # Should have at least ridge and ebm/gbm weights
        assert "ridge" in weights or "gbm" in weights
        assert "bias" in weights

    def test_get_meta_weights_coefficients_are_numeric(
        self, stacking_model, sample_monthly_data
    ):
        """Test that meta weights contain numeric coefficients."""
        stacking_model.fit(sample_monthly_data)

        weights = stacking_model.get_meta_weights()

        for key, value in weights.items():
            assert isinstance(value, (int, float, np.number))
            assert not np.isnan(value)


class TestStackingRegressorFeaturePreparation:
    """Tests for StackingRegressorForecaster feature preparation."""

    def test_prepare_features_creates_lags(self, stacking_model, sample_monthly_data):
        """Test that _prepare_features creates lag features."""
        df_prep = stacking_model._prepare_features(sample_monthly_data)

        assert "y_lag1" in df_prep.columns
        assert "y_lag2" in df_prep.columns
        assert "y_lag12" in df_prep.columns

    def test_prepare_features_creates_seasonal_features(
        self, stacking_model, sample_monthly_data
    ):
        """Test that _prepare_features creates seasonal features."""
        df_prep = stacking_model._prepare_features(sample_monthly_data)

        assert "month_sin" in df_prep.columns
        assert "month_cos" in df_prep.columns

    def test_prepare_features_creates_component_lags(
        self, stacking_model, sample_monthly_data
    ):
        """Test that _prepare_features creates component lags."""
        df_prep = stacking_model._prepare_features(sample_monthly_data)

        assert "food_lag1" in df_prep.columns
        assert "nonfood_lag1" in df_prep.columns
        assert "services_lag1" in df_prep.columns


class TestStackingRegressorBacktest:
    """Tests for StackingRegressorForecaster backtest method."""

    def test_backtest_returns_dataframe(self, stacking_model, sample_monthly_data):
        """Test that backtest returns DataFrame."""
        result = stacking_model.backtest(sample_monthly_data, start_date="2020-01-01")

        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0

    def test_backtest_has_expected_columns(self, stacking_model, sample_monthly_data):
        """Test that backtest has expected columns."""
        result = stacking_model.backtest(sample_monthly_data, start_date="2020-01-01")

        expected_cols = ["date", "actual", "prediction", "error", "pred_stacking"]
        for col in expected_cols:
            assert col in result.columns

    def test_backtest_error_calculation(self, stacking_model, sample_monthly_data):
        """Test that backtest calculates error correctly."""
        result = stacking_model.backtest(sample_monthly_data, start_date="2020-01-01")

        # error = actual - prediction
        for _, row in result.iterrows():
            expected_error = row["actual"] - row["prediction"]
            assert abs(row["error"] - expected_error) < 1e-10


class TestStackingRegressorEdgeCases:
    """Tests for StackingRegressorForecaster edge cases."""

    def test_seasonal_norm_computation(self, stacking_model, sample_monthly_data):
        """Test seasonal norm is computed correctly."""
        stacking_model.fit(sample_monthly_data)

        # Seasonal norm should have 12 months
        assert len(stacking_model.seasonal_norm) == 12
        assert all(1 <= m <= 12 for m in stacking_model.seasonal_norm.index)

    def test_outlier_years_excluded(self, stacking_model):
        """Test that outlier years are handled."""
        dates = pd.date_range("2010-01-01", periods=180, freq="MS")
        data = pd.DataFrame(
            {"Все товары и услуги": 100.0 + np.random.randn(180) * 0.3},
            index=dates,
        )

        # Add large outlier in 2010
        data.loc[pd.Timestamp("2010-06-01"), "Все товары и услуги"] = 105.0
        data.loc[pd.Timestamp("2022-03-01"), "Все товары и услуги"] = 104.0

        stacking_model.fit(data)

        # Model should still train (outliers excluded)
        assert stacking_model._is_fitted

    def test_model_registration(self):
        """Test that model is registered in ModelRegistry."""
        from sirena.models.registry import ModelRegistry

        model = ModelRegistry.get("stacking_regressor")

        assert model is not None
        assert model == StackingRegressorForecaster
