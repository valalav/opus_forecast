"""
Unit tests for StackingRegressorForecaster model.

Tests cover:
- Model initialization
- fit() method
- predict() method
- forecast() method
- get_meta_weights() method (KEY REQUIREMENT)
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

from sirena.models.stacking_regressor import StackingRegressorForecaster


@pytest.fixture
def sample_monthly_data():
    """Generate sample monthly CPI data for testing."""
    dates = pd.date_range("2016-01-01", periods=120, freq="MS")
    np.random.seed(42)

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


@pytest.fixture
def model():
    """Create a default model instance."""
    return StackingRegressorForecaster()


class TestStackRegressorForecasterInitialization:
    """Tests for model initialization."""

    def test_default_parameters(self):
        """Test default parameters are set correctly."""
        model = StackingRegressorForecaster()

        assert model.name == "stacking_regressor"
        assert model.ridge_alpha == 0.3
        assert model.ebm_max_bins == 256
        assert not model._is_fitted
        assert model.stacking is None
        assert model.scaler is None
        assert model.seasonal_norm is None

    def test_custom_parameters(self):
        """Test custom parameters are set correctly."""
        model = StackingRegressorForecaster(ridge_alpha=1.0, ebm_max_bins=128)

        assert model.ridge_alpha == 1.0
        assert model.ebm_max_bins == 128

    def test_model_inherits_from_base(self):
        """Test that model inherits from BaseForecaster."""
        from sirena.models.base import BaseForecaster

        model = StackingRegressorForecaster()
        assert isinstance(model, BaseForecaster)

    def test_min_train_size_constant(self):
        """Test MIN_TRAIN_SIZE constant."""
        assert StackingRegressorForecaster.MIN_TRAIN_SIZE == 36

    def test_outlier_years_constant(self):
        """Test OUTLIER_YEARS constant."""
        assert StackingRegressorForecaster.OUTLIER_YEARS == [2022, 2010]

    def test_base_features_constant(self):
        """Test BASE_FEATURES constant has expected features."""
        expected_features = [
            "y_lag1",
            "y_lag2",
            "y_lag12",
            "y_ma3",
            "month_sin",
            "month_cos",
            "food_lag1",
            "nonfood_lag1",
            "services_lag1",
        ]
        assert StackingRegressorForecaster.BASE_FEATURES == expected_features


class TestStackRegressorForecasterFit:
    """Tests for fit() method."""

    def test_fit_sets_is_fitted(self, sample_monthly_data):
        """Test that fit() sets _is_fitted flag."""
        model = StackingRegressorForecaster()
        model.fit(sample_monthly_data)

        assert model._is_fitted

    def test_fit_returns_self(self, sample_monthly_data):
        """Test that fit() returns self for method chaining."""
        model = StackingRegressorForecaster()
        result = model.fit(sample_monthly_data)

        assert result is model

    def test_fit_creates_stacking_model(self, sample_monthly_data):
        """Test that fit() creates stacking model."""
        model = StackingRegressorForecaster()
        model.fit(sample_monthly_data)

        assert model.stacking is not None
        assert hasattr(model.stacking, "estimators_")

    def test_fit_creates_scaler(self, sample_monthly_data):
        """Test that fit() creates a scaler."""
        model = StackingRegressorForecaster()
        model.fit(sample_monthly_data)

        assert model.scaler is not None

    def test_fit_creates_seasonal_norm(self, sample_monthly_data):
        """Test that fit() creates seasonal_norm."""
        model = StackingRegressorForecaster()
        model.fit(sample_monthly_data)

        assert model.seasonal_norm is not None
        assert isinstance(model.seasonal_norm, pd.Series)
        assert len(model.seasonal_norm) == 12  # One for each month

    def test_fit_sets_train_df(self, sample_monthly_data):
        """Test that fit() saves _train_df."""
        model = StackingRegressorForecaster()
        model.fit(sample_monthly_data)

        assert model._train_df is not None
        assert isinstance(model._train_df, pd.DataFrame)

    def test_fit_sets_target_col(self, sample_monthly_data):
        """Test that fit() sets _target_col."""
        model = StackingRegressorForecaster()
        model.fit(sample_monthly_data)

        assert model._target_col == "Все товары и услуги"

    def test_fit_with_custom_target_col(self, sample_monthly_data):
        """Test that fit() uses custom target column."""
        model = StackingRegressorForecaster()
        model.fit(sample_monthly_data, target_col="Продовольственные товары")

        assert model._target_col == "Продовольственные товары"

    def test_fit_with_custom_ridge_alpha(self, sample_monthly_data):
        """Test that custom ridge_alpha is used."""
        model = StackingRegressorForecaster(ridge_alpha=2.0)
        model.fit(sample_monthly_data)

        assert model.ridge_alpha == 2.0

    def test_fit_with_custom_ebm_bins(self, sample_monthly_data):
        """Test that custom ebm_max_bins is used."""
        model = StackingRegressorForecaster(ebm_max_bins=64)
        model.fit(sample_monthly_data)

        assert model.ebm_max_bins == 64


class TestMetaLearnerWeights:
    """Tests for get_meta_weights() method - KEY REQUIREMENT."""

    def test_get_meta_weights_requires_fit(self, model):
        """Test that get_meta_weights() raises error if not fitted."""
        with pytest.raises(Exception):
            model.get_meta_weights()

    def test_get_meta_weights_returns_dict(self, sample_monthly_data):
        """Test that get_meta_weights() returns dictionary."""
        model = StackingRegressorForecaster()
        model.fit(sample_monthly_data)

        weights = model.get_meta_weights()

        assert isinstance(weights, dict)
        assert len(weights) > 0

    def test_meta_weights_contains_base_model_names(self, sample_monthly_data):
        """Test that weights contain 'ridge' and 'ebm' (or 'gbm' fallback)."""
        model = StackingRegressorForecaster()
        model.fit(sample_monthly_data)

        weights = model.get_meta_weights()

        # Should have at least one base model
        has_ridge = "ridge" in weights
        has_ebm = "ebm" in weights
        has_gbm = "gbm" in weights

        assert has_ridge, "Should have ridge weight"
        assert has_ebm or has_gbm, "Should have ebm or gbm fallback weight"

    def test_meta_weights_contains_bias(self, sample_monthly_data):
        """Test that weights contain 'bias' key."""
        model = StackingRegressorForecaster()
        model.fit(sample_monthly_data)

        weights = model.get_meta_weights()

        assert "bias" in weights

    def test_meta_weights_are_numeric(self, sample_monthly_data):
        """Test that all weights are numeric (float)."""
        model = StackingRegressorForecaster()
        model.fit(sample_monthly_data)

        weights = model.get_meta_weights()

        for key, value in weights.items():
            assert isinstance(value, (int, float)), f"{key}: {value} is not numeric"

    def test_meta_weights_bias_is_intercept(self, sample_monthly_data):
        """Test that bias matches linear regression intercept."""
        model = StackingRegressorForecaster()
        model.fit(sample_monthly_data)

        weights = model.get_meta_weights()

        if model.stacking.final_estimator_ is not None:
            expected_bias = float(model.stacking.final_estimator_.intercept_)
            assert abs(weights["bias"] - expected_bias) < 1e-6

    def test_meta_weights_sum_to_forecast_formula(self, sample_monthly_data):
        """Test that weights correspond to linear regression coefficients."""
        model = StackingRegressorForecaster()
        model.fit(sample_monthly_data)

        weights = model.get_meta_weights()

        if model.stacking.final_estimator_ is not None:
            coefs = model.stacking.final_estimator_.coef_
            estimators = [name for name, _ in model.stacking.estimators]

            # Check that weights match coefficients
            for i, name in enumerate(estimators):
                if name in weights:
                    assert abs(weights[name] - float(coefs[i])) < 1e-6


class TestStackRegressorForecasterForecast:
    """Tests for forecast() method."""

    def test_forecast_requires_fit(self, model):
        """Test that forecast() raises error if model not fitted."""
        with pytest.raises(Exception):
            model.forecast()

    def test_forecast_returns_array(self, sample_monthly_data):
        """Test that forecast() returns numpy array."""
        model = StackingRegressorForecaster()
        model.fit(sample_monthly_data)
        result = model.forecast()

        assert isinstance(result, np.ndarray)

    def test_forecast_default_horizon(self, sample_monthly_data):
        """Test that forecast() with default horizon returns 12 values."""
        model = StackingRegressorForecaster()
        model.fit(sample_monthly_data)
        result = model.forecast()

        assert len(result) == 12

    def test_forecast_custom_horizon(self, sample_monthly_data):
        """Test that forecast(horizon=6) returns 6 values."""
        model = StackingRegressorForecaster()
        model.fit(sample_monthly_data)
        result = model.forecast(horizon=6)

        assert len(result) == 6

    def test_forecast_h1_returns_single_value(self, sample_monthly_data):
        """Test that forecast(horizon=1) returns single value."""
        model = StackingRegressorForecaster()
        model.fit(sample_monthly_data)
        result = model.forecast(horizon=1)

        assert len(result) == 1


class TestStackRegressorForecasterPredict:
    """Tests for predict() method."""

    def test_predict_requires_fit(self, model, sample_monthly_data):
        """Test that predict() raises error if model not fitted."""
        target_date = sample_monthly_data.index[-1]

        with pytest.raises(Exception):
            model.predict(sample_monthly_data, target_date)

    def test_predict_returns_dict(self, sample_monthly_data):
        """Test that predict() returns dictionary."""
        model = StackingRegressorForecaster()
        model.fit(sample_monthly_data)

        target_date = sample_monthly_data.index[-1]
        result = model.predict(sample_monthly_data, target_date)

        assert isinstance(result, dict)

    def test_predict_contains_required_keys(self, sample_monthly_data):
        """Test that predict() returns dict with all required keys."""
        model = StackingRegressorForecaster()
        model.fit(sample_monthly_data)

        target_date = sample_monthly_data.index[-1]
        result = model.predict(sample_monthly_data, target_date)

        required_keys = ["date", "prediction", "pred_stacking", "pred_ets", "model"]
        for key in required_keys:
            assert key in result, f"Missing key: {key}"

    def test_predict_prediction_is_numeric(self, sample_monthly_data):
        """Test that predict() returns numeric prediction."""
        model = StackingRegressorForecaster()
        model.fit(sample_monthly_data)

        target_date = sample_monthly_data.index[-1]
        result = model.predict(sample_monthly_data, target_date)

        assert isinstance(result["prediction"], (int, float, np.number))

    def test_predict_date_matches_target(self, sample_monthly_data):
        """Test that predict() returns correct date."""
        model = StackingRegressorForecaster()
        model.fit(sample_monthly_data)

        target_date = sample_monthly_data.index[-1]
        result = model.predict(sample_monthly_data, target_date)

        assert result["date"] == target_date

    def test_predict_pred_stacking_is_numeric(self, sample_monthly_data):
        """Test that pred_stacking is numeric."""
        model = StackingRegressorForecaster()
        model.fit(sample_monthly_data)

        target_date = sample_monthly_data.index[-1]
        result = model.predict(sample_monthly_data, target_date)

        assert isinstance(result["pred_stacking"], (int, float, np.number))

    def test_predict_pred_ets_is_numeric(self, sample_monthly_data):
        """Test that pred_ets is numeric."""
        model = StackingRegressorForecaster()
        model.fit(sample_monthly_data)

        target_date = sample_monthly_data.index[-1]
        result = model.predict(sample_monthly_data, target_date)

        assert isinstance(result["pred_ets"], (int, float, np.number))

    def test_predict_model_name_correct(self, sample_monthly_data):
        """Test that predict() returns correct model name."""
        model = StackingRegressorForecaster()
        model.fit(sample_monthly_data)

        target_date = sample_monthly_data.index[-1]
        result = model.predict(sample_monthly_data, target_date)

        assert result["model"] == "stacking_regressor"


class TestStackRegressorForecasterBacktest:
    """Tests for backtest() method."""

    def test_backtest_returns_dataframe(self, sample_monthly_data):
        """Test that backtest() returns DataFrame."""
        model = StackingRegressorForecaster()
        result = model.backtest(sample_monthly_data, start_date="2019-01-01")

        assert isinstance(result, pd.DataFrame)

    def test_backtest_has_required_columns(self, sample_monthly_data):
        """Test that backtest() returns DataFrame with required columns."""
        model = StackingRegressorForecaster()
        result = model.backtest(sample_monthly_data, start_date="2019-01-01")

        required_columns = ["date", "actual", "prediction", "error", "pred_stacking"]
        for col in required_columns:
            assert col in result.columns, f"Missing column: {col}"

    def test_backtest_results_not_empty(self, sample_monthly_data):
        """Test that backtest() produces results."""
        model = StackingRegressorForecaster()
        result = model.backtest(sample_monthly_data, start_date="2019-01-01")

        assert len(result) > 0

    def test_backtest_date_index_correct(self, sample_monthly_data):
        """Test that backtest() dates match requested period."""
        model = StackingRegressorForecaster()
        result = model.backtest(sample_monthly_data, start_date="2019-01-01")

        if len(result) > 0:
            first_date = result["date"].min()
            start = pd.Timestamp("2019-01-01")
            assert first_date >= start

    def test_backtest_actual_values_numeric(self, sample_monthly_data):
        """Test that actual values are numeric."""
        model = StackingRegressorForecaster()
        result = model.backtest(sample_monthly_data, start_date="2019-01-01")

        assert all(
            pd.api.types.is_numeric_dtype(result[col])
            for col in ["actual", "prediction", "error", "pred_stacking"]
        )


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_forecast_without_fit_raises_error(self, model):
        """Test that calling forecast() without fit raises error."""
        with pytest.raises(Exception):
            model.forecast()

    def test_predict_without_fit_raises_error(self, model, sample_monthly_data):
        """Test that calling predict() without fit raises error."""
        target_date = sample_monthly_data.index[-1]

        with pytest.raises(Exception):
            model.predict(sample_monthly_data, target_date)

    def test_get_meta_weights_without_fit_raises_error(self, model):
        """Test that calling get_meta_weights() without fit raises error."""
        with pytest.raises(Exception):
            model.get_meta_weights()

    def test_fit_with_insufficient_data_raises_error(self, sample_monthly_data):
        """Test that fit() with too little data raises ValueError."""
        # Create tiny dataset (< MIN_TRAIN_SIZE)
        tiny_data = sample_monthly_data.iloc[:20].copy()

        model = StackingRegressorForecaster()

        with pytest.raises(ValueError, match="Недостаточно данных"):
            model.fit(tiny_data)

    def test_model_name(self):
        """Test model name attribute."""
        model = StackingRegressorForecaster()
        assert model.name == "stacking_regressor"

    def test_feature_preparation_creates_lags(self, sample_monthly_data):
        """Test that _prepare_features creates lag features."""
        model = StackingRegressorForecaster()
        df_prep = model._prepare_features(sample_monthly_data)

        assert "y_lag1" in df_prep.columns
        assert "y_lag2" in df_prep.columns
        assert "y_lag12" in df_prep.columns

    def test_feature_preparation_creates_seasonal(self, sample_monthly_data):
        """Test that _prepare_features creates seasonal features."""
        model = StackingRegressorForecaster()
        df_prep = model._prepare_features(sample_monthly_data)

        assert "month_sin" in df_prep.columns
        assert "month_cos" in df_prep.columns

    def test_feature_preparation_creates_component_lags(self, sample_monthly_data):
        """Test that _prepare_features creates component lag features."""
        model = StackingRegressorForecaster()
        df_prep = model._prepare_features(sample_monthly_data)

        assert "food_lag1" in df_prep.columns
        assert "nonfood_lag1" in df_prep.columns
        assert "services_lag1" in df_prep.columns
