"""
Unit tests for TemporalFusionForecaster (TFT)
Tests are created in edge_lab but import from parent sirena package
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import sys

# Add parent directory to path to import sirena module
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Import model for tests that create new instances
from sirena.models import TemporalFusionForecaster


@pytest.fixture
def sample_data():
    """Generate sample inflation data for testing."""
    dates = pd.date_range("2015-01-01", periods=120, freq="MS")
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
def sample_data_with_exog():
    """Generate sample data with exogenous features."""
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

    return data


@pytest.fixture
def tft_default():
    """Create default TFT model."""
    from sirena.models import TemporalFusionForecaster

    return TemporalFusionForecaster()


@pytest.fixture
def tft_small():
    """Create TFT model with smaller architecture for faster testing."""
    from sirena.models import TemporalFusionForecaster

    return TemporalFusionForecaster(
        hidden_layers=1,
        hidden_size=32,
        max_iter=100,
    )


@pytest.fixture
def tft_tanh():
    """Create TFT model with tanh activation."""
    from sirena.models import TemporalFusionForecaster

    return TemporalFusionForecaster(activation="tanh")


@pytest.fixture
def tft_lbfgs():
    """Create TFT model with lbfgs solver."""
    from sirena.models import TemporalFusionForecaster

    return TemporalFusionForecaster(solver="lbfgs", max_iter=50)


# =============================================================================
# Test 1: Static Covariates
# =============================================================================


class TestStaticCovariates:
    """Test static covariate preparation."""

    def test_static_covariates_shape(self, sample_data, tft_default):
        """Test static covariates produce correct shape."""
        df_prep = tft_default._prepare_static_covariates(sample_data)

        assert df_prep.shape[0] == sample_data.shape[0], "Row count mismatch"
        assert df_prep.shape[1] >= sample_data.shape[1], "Should add features"

    def test_month_features(self, sample_data, tft_default):
        """Test month-based features."""
        df_prep = tft_default._prepare_static_covariates(sample_data)

        assert "month" in df_prep.columns, "Month column missing"
        assert "month_sin" in df_prep.columns, "Month sin missing"
        assert "month_cos" in df_prep.columns, "Month cos missing"

        # Check cyclical encoding
        jan_idx = df_prep.index.month == 1
        # sin(2*pi*1/12) = sin(pi/6) ≈ 0.5
        # cos(2*pi*1/12) = cos(pi/6) ≈ 0.866
        assert df_prep.loc[jan_idx, "month_sin"].values[0] > 0.4, (
            "Jan sin should be ~0.5"
        )
        assert df_prep.loc[jan_idx, "month_cos"].values[0] > 0.8, (
            "Jan cos should be ~0.866"
        )

    def test_quarter_features(self, sample_data, tft_default):
        """Test quarter-based features."""
        df_prep = tft_default._prepare_static_covariates(sample_data)

        assert "quarter" in df_prep.columns, "Quarter column missing"
        assert "quarter_sin" in df_prep.columns, "Quarter sin missing"
        assert "quarter_cos" in df_prep.columns, "Quarter cos missing"

        # Check quarter values
        q1_idx = df_prep.index.quarter == 1
        assert (df_prep.loc[q1_idx, "quarter"] == 1).all(), "Q1 quarters incorrect"

    def test_calendar_flags(self, sample_data, tft_default):
        """Test calendar flag features."""
        df_prep = tft_default._prepare_static_covariates(sample_data)

        assert "is_jan" in df_prep.columns, "is_jan missing"
        assert "is_jul" in df_prep.columns, "is_jul missing"
        assert "is_dec" in df_prep.columns, "is_dec missing"
        assert "is_q1" in df_prep.columns, "is_q1 missing"

        # Check flag values
        jan_count = (df_prep["is_jan"] == 1).sum()
        jan_dates = (df_prep.index.month == 1).sum()
        assert jan_count == jan_dates, f"Jan count mismatch: {jan_count} vs {jan_dates}"


# =============================================================================
# Test 2: Dynamic Features
# =============================================================================


class TestDynamicFeatures:
    """Test dynamic feature preparation."""

    def test_dynamic_features_shape(self, sample_data, tft_default):
        """Test dynamic features produce correct shape."""
        df_prep = tft_default._prepare_dynamic_features(sample_data)

        assert df_prep.shape[0] == sample_data.shape[0], "Row count mismatch"
        assert df_prep.shape[1] > sample_data.shape[1], "Should add lag features"

    def test_lagged_features(self, sample_data, tft_default):
        """Test lagged target features."""
        df_prep = tft_default._prepare_dynamic_features(sample_data)

        for lag in [1, 2, 3, 6, 12]:
            assert f"y_lag{lag}" in df_prep.columns, f"y_lag{lag} missing"

        # Check lag values
        y = sample_data["Все товары и услуги"].values
        assert np.allclose(df_prep["y_lag1"].iloc[1:].values, y[:-1], rtol=1e-5), (
            "Lag1 values incorrect"
        )

    def test_momentum_features(self, sample_data, tft_default):
        """Test momentum (difference) features."""
        df_prep = tft_default._prepare_dynamic_features(sample_data)

        assert "y_diff1" in df_prep.columns, "y_diff1 missing"
        assert "y_diff3" in df_prep.columns, "y_diff3 missing"
        assert "y_diff6" in df_prep.columns, "y_diff6 missing"

    def test_rolling_features(self, sample_data, tft_default):
        """Test rolling statistics features."""
        df_prep = tft_default._prepare_dynamic_features(sample_data)

        assert "y_ma3" in df_prep.columns, "y_ma3 missing"
        assert "y_ma6" in df_prep.columns, "y_ma6 missing"
        assert "y_std3" in df_prep.columns, "y_std3 missing"

    def test_exogenous_features(self, sample_data_with_exog, tft_default):
        """Test exogenous feature extraction."""
        df_prep = tft_default._prepare_dynamic_features(sample_data_with_exog)

        # USD features
        assert "usd_lag1" in df_prep.columns, "usd_lag1 missing"
        assert "usd_lag2" in df_prep.columns, "usd_lag2 missing"
        assert "usd_diff1" in df_prep.columns, "usd_diff1 missing"

        # Brent features
        assert "brent_lag1" in df_prep.columns, "brent_lag1 missing"
        assert "brent_lag3" in df_prep.columns, "brent_lag3 missing"
        assert "brent_diff1" in df_prep.columns, "brent_diff1 missing"

        # Ki features
        assert "ki_lag1" in df_prep.columns, "ki_lag1 missing"
        assert "ki_lag3" in df_prep.columns, "ki_lag3 missing"
        assert "ki_diff1" in df_prep.columns, "ki_diff1 missing"

    def test_component_features(self, sample_data, tft_default):
        """Test component feature extraction."""
        df_prep = tft_default._prepare_dynamic_features(sample_data)

        assert "food_lag1" in df_prep.columns, "food_lag1 missing"
        assert "nonfood_lag1" in df_prep.columns, "nonfood_lag1 missing"
        assert "services_lag1" in df_prep.columns, "services_lag1 missing"


# =============================================================================
# Test 3: Feature Selection
# =============================================================================


class TestFeatureSelection:
    """Test feature selection logic."""

    def test_select_features_basic(self, sample_data, tft_default):
        """Test basic feature selection."""
        df_prep = tft_default._prepare_static_covariates(sample_data)
        df_prep = tft_default._prepare_dynamic_features(df_prep)

        features = tft_default._select_features(df_prep, "Все товары и услуги")

        assert len(features) > 0, "Should select features"
        assert "y_lag1" in features, "Should include y_lag1"
        assert "month_sin" in features, "Should include month_sin"
        assert "month_cos" in features, "Should include month_cos"

    def test_select_features_with_exog(self, sample_data_with_exog, tft_default):
        """Test feature selection with exogenous variables."""
        df_prep = tft_default._prepare_static_covariates(sample_data_with_exog)
        df_prep = tft_default._prepare_dynamic_features(df_prep)

        features = tft_default._select_features(df_prep, "Все товары и услуги")

        # Should include exogenous features
        assert "usd_lag1" in features, "Should include usd_lag1"
        assert "brent_lag1" in features, "Should include brent_lag1"
        assert "ki_lag1" in features, "Should include ki_lag1"

    def test_static_dynamic_separation(self, sample_data, tft_small):
        """Test separation of static and dynamic features."""
        # First fit model to set static/dynamic features
        tft_small.fit(sample_data, "Все товары и услуги")

        # After fit, should have separated features
        assert tft_small._static_features is not None, "Static features not set"
        assert tft_small._dynamic_features is not None, "Dynamic features not set"

        # Check separation
        static_set = set(tft_small._static_features)
        dynamic_set = set(tft_small._dynamic_features)
        assert len(static_set.intersection(dynamic_set)) == 0, (
            "Features should not overlap"
        )


# =============================================================================
# Test 4: Model Fitting
# =============================================================================


class TestModelFitting:
    """Test model fitting."""

    def test_fit_basic(self, sample_data, tft_small):
        """Test basic model fitting."""
        model = tft_small.fit(sample_data, "Все товары и услуги")

        assert model.is_fitted, "Model should be fitted"
        assert model.model is not None, "Model instance not created"
        assert model.scaler is not None, "Scaler not created"

    def test_fit_with_exog(self, sample_data_with_exog, tft_small):
        """Test fitting with exogenous features."""
        model = tft_small.fit(sample_data_with_exog, "Все товары и услуги")

        assert model.is_fitted, "Model should be fitted"
        assert len(model._dynamic_features) > 10, "Should have dynamic features"

    def test_insufficient_data(self, sample_data, tft_default):
        """Test error on insufficient data."""
        small_data = sample_data.iloc[:20]

        with pytest.raises(ValueError, match="Недостаточно данных|Insufficient data"):
            tft_default.fit(small_data, "Все товары и услуги")

    def test_fit_different_activations(self, sample_data_with_exog):
        """Test fitting with different activations."""
        for activation in ["relu", "tanh", "logistic"]:
            model = TemporalFusionForecaster(
                activation=activation,
                max_iter=50,
            )
            model.fit(sample_data_with_exog, "Все товары и услуги")

            assert model.is_fitted, f"Model with {activation} not fitted"

    def test_fit_different_solvers(self, sample_data_with_exog):
        """Test fitting with different solvers."""
        for solver in ["adam", "lbfgs"]:
            model = TemporalFusionForecaster(
                solver=solver,
                max_iter=50 if solver == "adam" else 25,
            )
            model.fit(sample_data_with_exog, "Все товары и услуги")

            assert model.is_fitted, f"Model with {solver} not fitted"

    def test_fit_saves_features(self, sample_data, tft_small):
        """Test that fit saves feature lists."""
        tft_small.fit(sample_data, "Все товары и услуги")

        assert tft_small._final_features is not None, "Final features not saved"
        assert len(tft_small._final_features) > 0, "No features saved"
        assert tft_small._static_features is not None, "Static features not saved"
        assert tft_small._dynamic_features is not None, "Dynamic features not saved"


# =============================================================================
# Test 5: Prediction
# =============================================================================


class TestPrediction:
    """Test prediction functionality."""

    def test_predict_single_date(self, sample_data, tft_small):
        """Test single date prediction."""
        tft_small.fit(sample_data, "Все товары и услуги")

        target_date = sample_data.index[-10]
        result = tft_small.predict(sample_data, target_date)

        assert "prediction" in result, "Prediction key missing"
        assert "date" in result, "Date key missing"
        assert "features_used" in result, "Features used key missing"
        assert result["date"] == target_date, "Date mismatch"
        assert isinstance(result["prediction"], (int, float, np.number)), (
            "Prediction not numeric"
        )

    def test_predict_with_exog(self, sample_data_with_exog, tft_small):
        """Test prediction with exogenous features."""
        tft_small.fit(sample_data_with_exog, "Все товары и услуги")

        target_date = sample_data_with_exog.index[-10]
        result = tft_small.predict(sample_data_with_exog, target_date)

        assert result is not None, "Prediction result is None"
        assert not np.isnan(result["prediction"]), "Prediction is NaN"

    def test_predict_future_date(self, sample_data, tft_small):
        """Test prediction for future date (not in data)."""
        tft_small.fit(sample_data, "Все товары и услуги")

        last_date = sample_data.index.max()
        future_date = last_date + pd.DateOffset(months=1)

        result = tft_small.predict(sample_data, future_date)

        assert result is not None, "Future prediction is None"
        assert not np.isnan(result["prediction"]), "Future prediction is NaN"

    def test_predict_before_fit_error(self, sample_data, tft_default):
        """Test error when predicting before fit."""
        with pytest.raises(ValueError, match="не обучена"):
            tft_default.predict(sample_data, sample_data.index[-1])

    def test_predict_returns_attention(self, sample_data, tft_small):
        """Test that prediction returns attention weights."""
        tft_small.fit(sample_data, "Все товары и услуги")

        target_date = sample_data.index[-10]
        result = tft_small.predict(sample_data, target_date)

        assert "attention_weights" in result, "Attention weights missing"
        assert isinstance(result["attention_weights"], dict), "Attention not a dict"


# =============================================================================
# Test 6: Forecast
# =============================================================================


class TestForecast:
    """Test multi-horizon forecasting."""

    def test_forecast_basic(self, sample_data, tft_small):
        """Test basic forecasting."""
        tft_small.fit(sample_data, "Все товары и услуги")

        horizon = 6
        forecast = tft_small.forecast(horizon=horizon)

        assert forecast is not None, "Forecast is None"
        assert len(forecast) == horizon, (
            f"Forecast length mismatch: {len(forecast)} vs {horizon}"
        )
        assert not np.any(np.isnan(forecast)), "Forecast contains NaN values"

    def test_forecast_different_horizons(self, sample_data, tft_small):
        """Test forecasting with different horizons."""
        tft_small.fit(sample_data, "Все товары и услуги")

        for horizon in [1, 3, 6, 12]:
            forecast = tft_small.forecast(horizon=horizon)

            assert len(forecast) == horizon, f"Horizon {horizon} failed"

    def test_forecast_before_fit_error(self, sample_data, tft_default):
        """Test error when forecasting before fit."""
        with pytest.raises(ValueError, match="не обучена"):
            tft_default.forecast(horizon=12)


# =============================================================================
# Test 7: Backtest
# =============================================================================


class TestBacktest:
    """Test backtesting functionality."""

    def test_backtest_basic(self, sample_data, tft_small):
        """Test basic backtest."""
        results = tft_small.backtest(sample_data, start_date="2018-01-01")

        assert results is not None, "Backtest results is None"
        assert len(results) > 0, "No backtest results"
        assert "date" in results.columns, "Date column missing"
        assert "actual" in results.columns, "Actual column missing"
        assert "prediction" in results.columns, "Prediction column missing"
        assert "error" in results.columns, "Error column missing"

    def test_backtest_mae_calculation(self, sample_data, tft_small):
        """Test backtest MAE calculation."""
        results = tft_small.backtest(sample_data, start_date="2018-01-01")

        mae = (results["error"].abs()).mean()

        assert mae >= 0, "MAE should be non-negative"
        # Note: MAE can be high for small test data and short training
        # Just verify it's a finite number
        assert np.isfinite(mae), "MAE should be finite"

    def test_backtest_with_exog(self, sample_data_with_exog, tft_small):
        """Test backtest with exogenous features."""
        results = tft_small.backtest(sample_data_with_exog, start_date="2018-01-01")

        assert len(results) > 0, "No backtest results with exog"

    def test_backtest_structure(self, sample_data, tft_small):
        """Test backtest result structure."""
        results = tft_small.backtest(sample_data, start_date="2018-01-01")

        # Check for TFT-specific columns
        assert "n_static_features" in results.columns, "n_static_features missing"
        assert "n_dynamic_features" in results.columns, "n_dynamic_features missing"
        assert "top_attention" in results.columns, "top_attention missing"


# =============================================================================
# Test 8: Feature Importance
# =============================================================================


class TestFeatureImportance:
    """Test feature importance extraction."""

    def test_get_feature_importance(self, sample_data, tft_small):
        """Test getting feature importance."""
        tft_small.fit(sample_data, "Все товары и услуги")

        importance = tft_small.get_feature_importance()

        assert importance is not None, "Feature importance is None"
        assert "feature" in importance.columns, "Feature column missing"
        assert "importance" in importance.columns, "Importance column missing"
        assert "type" in importance.columns, "Type column missing"
        assert len(importance) > 0, "No importance values"

    def test_feature_importance_sorting(self, sample_data, tft_small):
        """Test feature importance is sorted."""
        tft_small.fit(sample_data, "Все товары и услуги")

        importance = tft_small.get_feature_importance()

        # Check sorted by importance (descending)
        imp_values = importance["importance"].values
        assert all(
            imp_values[i] >= imp_values[i + 1] for i in range(len(imp_values) - 1)
        ), "Importance not sorted"

    def test_feature_importance_types(self, sample_data, tft_small):
        """Test feature type classification."""
        tft_small.fit(sample_data, "Все товары и услуги")

        importance = tft_small.get_feature_importance()

        static_count = (importance["type"] == "static").sum()
        dynamic_count = (importance["type"] == "dynamic").sum()

        assert static_count > 0, "No static features in importance"
        assert dynamic_count > 0, "No dynamic features in importance"


# =============================================================================
# Test 9: Attention Weights
# =============================================================================


class TestAttentionWeights:
    """Test attention weight extraction."""

    def test_get_attention_weights(self, sample_data, tft_small):
        """Test getting attention weights."""
        tft_small.fit(sample_data, "Все товары и услуги")

        attention = tft_small.get_attention_weights()

        assert attention is not None, "Attention weights is None"
        assert isinstance(attention, dict), "Attention not a dict"
        assert len(attention) > 0, "No attention weights"

    def test_attention_weights_sum(self, sample_data, tft_small):
        """Test attention weights sum to approximately 1."""
        tft_small.fit(sample_data, "Все товары и услуги")

        attention = tft_small.get_attention_weights()
        total = sum(attention.values())

        assert total > 0.9 and total <= 1.1, f"Attention sum: {total}, expected ~1.0"

    def test_attention_in_prediction(self, sample_data, tft_small):
        """Test attention weights are in prediction result."""
        tft_small.fit(sample_data, "Все товары и услуги")

        target_date = sample_data.index[-10]
        result = tft_small.predict(sample_data, target_date)

        assert "attention_weights" in result, "Attention not in prediction"
        assert result["attention_weights"] is not None, "Attention is None"


# =============================================================================
# Test 10: Model Info
# =============================================================================


class TestModelInfo:
    """Test model information retrieval."""

    def test_get_model_info(self, tft_default):
        """Test getting model info."""
        info = tft_default.get_model_info()

        assert info is not None, "Model info is None"
        assert "name" in info, "Name key missing"
        assert "hidden_layers" in info, "Hidden layers missing"
        assert "hidden_size" in info, "Hidden size missing"
        assert "activation" in info, "Activation missing"
        assert "solver" in info, "Solver missing"

    def test_model_info_after_fit(self, sample_data, tft_small):
        """Test model info after fitting."""
        tft_small.fit(sample_data, "Все товары и услуги")

        info = tft_small.get_model_info()

        assert info["is_fitted"], "is_fitted should be True"
        assert "n_features" in info, "n_features missing"
        assert "n_static_features" in info, "n_static_features missing"
        assert "n_dynamic_features" in info, "n_dynamic_features missing"

    def test_model_info_matches_params(self, tft_default):
        """Test model info matches constructor params."""
        info = tft_default.get_model_info()

        assert info["hidden_layers"] == tft_default.hidden_layers, (
            "Hidden layers mismatch"
        )
        assert info["hidden_size"] == tft_default.hidden_size, "Hidden size mismatch"
        assert info["activation"] == tft_default.activation, "Activation mismatch"
        assert info["solver"] == tft_default.solver, "Solver mismatch"


# =============================================================================
# Test 11: Weights Extraction (Task 22 Acceptance Criterion)
# =============================================================================


class TestWeightsExtraction:
    """Test weights extraction (acceptance criterion for task 22)."""

    def test_get_weights(self, sample_data, tft_small):
        """Test getting model weights."""
        tft_small.fit(sample_data, "Все товары и услуги")

        weights = tft_small.get_weights()

        assert weights is not None, "Weights is None"
        assert "attention_weights" in weights, "Attention weights missing"
        assert "network_weights" in weights, "Network weights missing"

    def test_attention_weights_structure(self, sample_data, tft_small):
        """Test attention weights structure."""
        tft_small.fit(sample_data, "Все товары и услуги")

        weights = tft_small.get_weights()

        attention = weights["attention_weights"]
        assert isinstance(attention, dict), "Attention not a dict"
        assert len(attention) > 0, "No attention weights"

        # Check all features have weights
        for feature in tft_small._final_features:
            assert feature in attention, f"Weight for {feature} missing"

    def test_network_weights_structure(self, sample_data, tft_small):
        """Test network weights structure."""
        tft_small.fit(sample_data, "Все товары и услуги")

        weights = tft_small.get_weights()

        net_weights = weights["network_weights"]
        assert "layer_weights" in net_weights, "layer_weights missing"
        assert "layer_biases" in net_weights, "layer_biases missing"

        # Should have one more layer than hidden_layers (input + hidden + output)
        n_layers = len(net_weights["layer_weights"])
        expected_layers = tft_small.hidden_layers + 1
        assert n_layers == expected_layers, (
            f"Expected {expected_layers} layers, got {n_layers}"
        )

    def test_weights_before_fit_error(self, tft_default):
        """Test error when getting weights before fit."""
        with pytest.raises(ValueError, match="не обучена"):
            tft_default.get_weights()


# =============================================================================
# Test 12: Integration
# =============================================================================


class TestIntegration:
    """Integration tests."""

    def test_full_workflow(self, sample_data, tft_small):
        """Test full workflow: fit → predict → forecast → backtest."""
        # Fit
        model = tft_small.fit(sample_data, "Все товары и услуги")
        assert model.is_fitted

        # Predict
        target_date = sample_data.index[-5]
        pred_result = model.predict(sample_data, target_date)
        assert "prediction" in pred_result

        # Forecast
        forecast = model.forecast(horizon=3)
        assert len(forecast) == 3

        # Backtest
        bt_results = model.backtest(sample_data, start_date="2017-01-01")
        assert len(bt_results) > 0

    def test_model_registry_registration(self):
        """Test model is registered in registry."""
        from sirena.models import ModelRegistry

        models = ModelRegistry.list_models()
        assert "tft" in models, "TFT not registered"

    def test_model_importable(self):
        """Test model can be imported."""
        from sirena.models import TemporalFusionForecaster

        model = TemporalFusionForecaster()
        assert model.name == "tft"

    def test_integration_with_components(self, sample_data, tft_small):
        """Test with component features."""
        model = tft_small.fit(sample_data, "Все товары и услуги")

        # Check component features are in model
        has_food = any("food" in f for f in model._final_features)
        has_nonfood = any("nonfood" in f for f in model._final_features)
        has_services = any("services" in f for f in model._final_features)

        assert has_food or has_nonfood or has_services, (
            "Should use at least one component feature"
        )

    def test_different_architectures(self, sample_data_with_exog):
        """Test different network architectures."""
        configs = [
            {"hidden_layers": 1, "hidden_size": 16},
            {"hidden_layers": 2, "hidden_size": 32},
            {"hidden_layers": 3, "hidden_size": 64},
        ]

        for config in configs:
            model = TemporalFusionForecaster(**config, max_iter=50)
            model.fit(sample_data_with_exog, "Все товары и услуги")

            assert model.is_fitted, f"Config {config} failed"

            info = model.get_model_info()
            assert info["hidden_layers"] == config["hidden_layers"]
            assert info["hidden_size"] == config["hidden_size"]
