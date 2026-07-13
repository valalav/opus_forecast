"""
Unit tests for MIDASForecaster
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
def sample_data_with_hf():
    """Generate sample data with high-frequency features."""
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
def midas_almon():
    """Create MIDAS model with Almon weights."""
    from sirena.models import MIDASForecaster

    return MIDASForecaster(weight_type="almon", poly_order=2)


@pytest.fixture
def midas_exp():
    """Create MIDAS model with exponential weights."""
    from sirena.models import MIDASForecaster

    return MIDASForecaster(weight_type="exp")


@pytest.fixture
def midas_beta():
    """Create MIDAS model with Beta weights."""
    from sirena.models import MIDASForecaster

    return MIDASForecaster(weight_type="beta")


@pytest.fixture
def midas_norm_exp():
    """Create MIDAS model with normalized exponential weights."""
    from sirena.models import MIDASForecaster

    return MIDASForecaster(weight_type="normalized_exp")


# =============================================================================
# Test 1: MIDAS Weight Functions
# =============================================================================


class TestMidasWeights:
    """Test MIDAS weighting functions."""

    def test_almon_weights_shape(self, midas_almon):
        """Test Almon weights produce correct shape."""
        weights, theta = midas_almon._get_midas_weights(8)
        assert weights.shape == (8,), f"Expected shape (8,), got {weights.shape}"
        assert theta.shape == (3,), f"Expected shape (3,), got {theta.shape}"

    def test_almon_weights_polynomial(self, midas_almon):
        """Test Almon weights follow polynomial pattern."""
        weights, theta = midas_almon._get_midas_weights(10)
        k = np.arange(10)

        # Reconstruct from polynomial
        reconstructed = np.zeros(10)
        for p, theta_p in enumerate(theta):
            reconstructed += theta_p * (k**p)

        np.testing.assert_allclose(
            weights,
            reconstructed,
            rtol=1e-10,
            err_msg="Almon weights don't match polynomial formula",
        )

    def test_exp_weights_decay(self, midas_exp):
        """Test exponential weights decay over time."""
        weights, theta = midas_exp._get_midas_weights(10)

        # Weights should decrease (or stay constant)
        for i in range(1, len(weights)):
            assert weights[i] <= weights[i] + 1e-10, (
                f"Exp weights should decay: {weights[i - 1]} -> {weights[i]}"
            )

    def test_exp_weights_formula(self, midas_exp):
        """Test exponential weights match formula."""
        weights, theta = midas_exp._get_midas_weights(8)
        k = np.arange(8)
        expected = np.exp(-theta[0] * k)

        np.testing.assert_allclose(
            weights, expected, rtol=1e-10, err_msg="Exp weights don't match formula"
        )

    def test_beta_weights_shape(self, midas_beta):
        """Test Beta weights produce correct shape."""
        weights, theta = midas_beta._get_midas_weights(12)
        assert weights.shape == (12,), f"Expected shape (12,), got {weights.shape}"
        assert theta.shape == (2,), f"Expected shape (2,), got {theta.shape}"

    def test_beta_weights_positive(self, midas_beta):
        """Test Beta weights are non-negative."""
        weights, _ = midas_beta._get_midas_weights(10)
        assert np.all(weights >= 0), "Beta weights should be non-negative"

    def test_normalized_exp_sum_to_one(self, midas_norm_exp):
        """Test normalized exponential weights sum to 1."""
        weights, _ = midas_norm_exp._get_midas_weights(15)

        # Should sum to approximately 1
        np.testing.assert_allclose(
            weights.sum(),
            1.0,
            rtol=1e-10,
            err_msg="Normalized exp weights should sum to 1",
        )

    def test_normalized_exp_positive(self, midas_norm_exp):
        """Test normalized exponential weights are positive."""
        weights, _ = midas_norm_exp._get_midas_weights(10)
        assert np.all(weights > 0), "Normalized exp weights should be positive"


# =============================================================================
# Test 2: Aggregation Functions
# =============================================================================


class TestMidasAggregation:
    """Test high-frequency aggregation functions."""

    def test_aggregate_hf_to_mf_shape(self, midas_almon, sample_data):
        """Test aggregation produces correct shape."""
        # Simulate weekly data
        weekly_dates = pd.date_range("2019-01-01", periods=200, freq="W")
        hf_series = pd.Series(75 + np.random.randn(200) * 5, index=weekly_dates)

        agg_df = midas_almon._aggregate_hf_to_mf(hf_series, sample_data.index, 4)

        assert agg_df.shape == (120, 4), f"Expected shape (120, 4), got {agg_df.shape}"
        assert list(agg_df.columns) == [
            "hf_agg_L1",
            "hf_agg_L2",
            "hf_agg_L3",
            "hf_agg_L4",
        ]

    def test_aggregate_hf_to_mf_values(self, midas_almon):
        """Test aggregation produces valid values."""
        weekly_dates = pd.date_range("2019-01-01", periods=100, freq="W")
        hf_series = pd.Series(100 + np.arange(100), index=weekly_dates)

        monthly_dates = pd.date_range("2020-01-01", periods=12, freq="MS")
        agg_df = midas_almon._aggregate_hf_to_mf(hf_series, monthly_dates, 2)

        # Should have values (not NaN)
        assert agg_df["hf_agg_L1"].notna().sum() > 0, "Should have aggregated values"

    def test_apply_midas_weights_shape(self, midas_almon, sample_data):
        """Test MIDAS weight application produces correct shape."""
        agg_df = pd.DataFrame(
            {
                "hf_agg_L1": np.random.randn(120),
                "hf_agg_L2": np.random.randn(120),
                "hf_agg_L3": np.random.randn(120),
                "hf_agg_L4": np.random.randn(120),
            },
            index=sample_data.index,
        )

        weights, _ = midas_almon._get_midas_weights(4)
        weighted = midas_almon._apply_midas_weights(agg_df, weights)

        assert isinstance(weighted, pd.Series), "Should return Series"
        assert len(weighted) == 120, f"Expected length 120, got {len(weighted)}"

    def test_apply_midas_weights_formula(self, midas_almon):
        """Test MIDAS weight application matches formula."""
        agg_df = pd.DataFrame(
            {
                "hf_agg_L1": np.ones(10),
                "hf_agg_L2": np.ones(10),
                "hf_agg_L3": np.ones(10),
            },
            index=pd.date_range("2020-01-01", periods=10, freq="MS"),
        )

        weights, _ = midas_almon._get_midas_weights(3)
        weighted = midas_almon._apply_midas_weights(agg_df, weights)

        expected = weights.sum()
        np.testing.assert_allclose(
            weighted.values,
            expected,
            rtol=1e-10,
            err_msg="Weighted sum doesn't match formula",
        )


# =============================================================================
# Test 3: Model Fitting
# =============================================================================


class TestMidasFit:
    """Test model fitting functionality."""

    def test_fit_basic(self, midas_almon, sample_data):
        """Test basic model fitting."""
        model = midas_almon.fit(sample_data)

        assert model._is_fitted, "Model should be fitted"
        assert model.model is not None, "Model should have underlying Ridge"
        assert model.scaler is not None, "Model should have scaler"
        assert model._last_train_date == sample_data.index[-1]

    def test_fit_with_hf_features(self, midas_almon, sample_data_with_hf):
        """Test fitting with high-frequency features."""
        model = midas_almon.fit(sample_data_with_hf)

        assert model._is_fitted, "Model should be fitted"
        assert len(model._available_hf) > 0, "Should detect HF features"

    def test_fit_insufficient_data(self, midas_almon):
        """Test fitting with insufficient data raises error."""
        short_data = pd.DataFrame(
            {"Все товары и услуги": [100.1, 100.2, 100.3]},
            index=pd.date_range("2020-01-01", periods=3, freq="MS"),
        )

        with pytest.raises(ValueError, match="Недостаточно данных"):
            midas_almon.fit(short_data)

    def test_fit_different_weight_types(self, sample_data_with_hf):
        """Test fitting with different weight types."""
        from sirena.models import MIDASForecaster

        for weight_type in ["almon", "exp", "beta", "normalized_exp"]:
            model = MIDASForecaster(weight_type=weight_type)
            model.fit(sample_data_with_hf)
            assert model._is_fitted, f"Model with {weight_type} should fit"

    def test_fit_saves_midas_transformers(self, midas_almon, sample_data_with_hf):
        """Test fitting saves MIDAS transformers."""
        model = midas_almon.fit(sample_data_with_hf)

        for hf_name in model._available_hf:
            assert hf_name in model._midas_transformers, (
                f"Should save transformer for {hf_name}"
            )
            assert "weights" in model._midas_transformers[hf_name], (
                f"Should have weights for {hf_name}"
            )


# =============================================================================
# Test 4: Prediction
# =============================================================================


class TestMidasPredict:
    """Test prediction functionality."""

    def test_predict_single_date(self, midas_almon, sample_data_with_hf):
        """Test prediction for single date."""
        model = midas_almon.fit(sample_data_with_hf)
        target_date = pd.Timestamp("2024-01-01")

        result = model.predict(sample_data_with_hf, target_date)

        assert "prediction" in result, "Should return prediction"
        assert "date" in result, "Should return date"
        assert result["date"] == target_date, "Date should match"
        assert isinstance(result["prediction"], (int, float)), (
            "Prediction should be numeric"
        )

    def test_predict_not_fitted(self, midas_almon, sample_data):
        """Test prediction raises error when not fitted."""
        with pytest.raises(ValueError, match="не обучена"):
            midas_almon.predict(sample_data, pd.Timestamp("2024-01-01"))

    def test_predict_returns_hf_contribution(self, midas_almon, sample_data_with_hf):
        """Test prediction returns HF feature contributions."""
        model = midas_almon.fit(sample_data_with_hf)
        target_date = pd.Timestamp("2024-01-01")

        result = model.predict(sample_data_with_hf, target_date)

        assert "hf_contribution" in result, "Should return HF contributions"
        assert "hf_features" in result, "Should return HF features list"

    def test_forecast_horizon(self, midas_almon, sample_data_with_hf):
        """Test forecast for horizon."""
        model = midas_almon.fit(sample_data_with_hf)
        horizon = 6

        forecast = model.forecast(horizon)

        assert len(forecast) == horizon, (
            f"Expected {horizon} predictions, got {len(forecast)}"
        )
        assert isinstance(forecast, np.ndarray), "Should return numpy array"


# =============================================================================
# Test 5: Backtest
# =============================================================================


class TestMidasBacktest:
    """Test backtest functionality."""

    def test_backtest_shape(self, midas_almon, sample_data_with_hf):
        """Test backtest returns correct shape."""
        results = midas_almon.backtest(sample_data_with_hf, start_date="2022-01-01")

        assert isinstance(results, pd.DataFrame), "Should return DataFrame"
        assert "date" in results.columns, "Should have date column"
        assert "actual" in results.columns, "Should have actual column"
        assert "prediction" in results.columns, "Should have prediction column"
        assert "error" in results.columns, "Should have error column"

    def test_backtest_no_results_when_insufficient_data(self, midas_almon):
        """Test backtest with insufficient training data."""
        short_data = pd.DataFrame(
            {
                "Все товары и услуги": 100 + np.random.randn(30) * 0.5,
                "brent": 75 + np.random.randn(30) * 5,
            },
            index=pd.date_range("2020-01-01", periods=30, freq="MS"),
        )

        results = midas_almon.backtest(short_data, start_date="2022-01-01")

        assert len(results) == 0, "Should have no results with insufficient data"

    def test_backtest_mae_calculation(self, midas_almon, sample_data_with_hf):
        """Test backtest calculates MAE correctly."""
        results = midas_almon.backtest(sample_data_with_hf, start_date="2023-01-01")

        if len(results) > 0:
            mae = (results["error"].abs()).mean()
            assert isinstance(mae, (int, float)), "MAE should be numeric"
            assert mae >= 0, "MAE should be non-negative"

    def test_backtest_with_different_weight_types(self, sample_data_with_hf):
        """Test backtest with different weight types."""
        from sirena.models import MIDASForecaster

        for weight_type in ["almon", "exp", "beta"]:
            model = MIDASForecaster(weight_type=weight_type)
            results = model.backtest(sample_data_with_hf, start_date="2023-01-01")

            assert isinstance(results, pd.DataFrame), (
                f"Backtest should work for {weight_type}"
            )


# =============================================================================
# Test 6: Feature Importance
# =============================================================================


class TestMidasFeatureImportance:
    """Test feature importance functionality."""

    def test_get_feature_importance(self, midas_almon, sample_data_with_hf):
        """Test getting feature importance."""
        model = midas_almon.fit(sample_data_with_hf)
        importance = model.get_feature_importance()

        assert isinstance(importance, pd.DataFrame), "Should return DataFrame"
        assert "feature" in importance.columns, "Should have feature column"
        assert "coefficient" in importance.columns, "Should have coefficient column"
        assert "abs_coef" in importance.columns, "Should have abs_coef column"

    def test_get_feature_importance_sorted(self, midas_almon, sample_data_with_hf):
        """Test feature importance is sorted by absolute coefficient."""
        model = midas_almon.fit(sample_data_with_hf)
        importance = model.get_feature_importance()

        abs_coefs = importance["abs_coef"].values
        assert all(
            abs_coefs[i] >= abs_coefs[i + 1] for i in range(len(abs_coefs) - 1)
        ), "Should be sorted by absolute coefficient"

    def test_get_feature_importance_marks_hf_features(
        self, midas_almon, sample_data_with_hf
    ):
        """Test HF features are marked in importance."""
        model = midas_almon.fit(sample_data_with_hf)
        importance = model.get_feature_importance()

        if len(model._available_hf) > 0:
            assert "is_hf" in importance.columns, "Should have is_hf column"
            assert importance["is_hf"].any(), "Should mark HF features"


# =============================================================================
# Test 7: Model Info
# =============================================================================


class TestMidasModelInfo:
    """Test model information retrieval."""

    def test_get_model_info_basic(self, midas_almon):
        """Test getting basic model info."""
        info = midas_almon.get_model_info()

        assert info["name"] == "midas", "Name should be midas"
        assert info["weight_type"] == "almon", "Weight type should match"
        assert info["poly_order"] == 2, "Poly order should match"
        assert info["is_fitted"] == False, "Should not be fitted yet"

    def test_get_model_info_after_fit(self, midas_almon, sample_data_with_hf):
        """Test getting model info after fitting."""
        model = midas_almon.fit(sample_data_with_hf)
        info = model.get_model_info()

        assert info["is_fitted"] == True, "Should be fitted"
        assert info["n_features"] > 0, "Should have features"
        assert isinstance(info["hf_features"], list), "HF features should be list"

    def test_get_midas_weights(self, midas_almon, sample_data_with_hf):
        """Test getting MIDAS weights."""
        model = midas_almon.fit(sample_data_with_hf)

        for hf_name in model._available_hf:
            weights, theta = model.get_midas_weights(hf_name)

            assert isinstance(weights, np.ndarray), "Weights should be numpy array"
            assert isinstance(theta, np.ndarray), "Theta should be numpy array"
            assert len(weights) > 0, "Should have weights"

    def test_get_midas_weights_invalid_feature(self, midas_almon, sample_data_with_hf):
        """Test getting MIDAS weights for invalid feature raises error."""
        model = midas_almon.fit(sample_data_with_hf)

        with pytest.raises(ValueError, match="not available"):
            model.get_midas_weights("invalid_feature")


# =============================================================================
# Test 8: Integration
# =============================================================================


class TestMidasIntegration:
    """Integration tests for MIDAS model."""

    def test_full_workflow(self, midas_almon, sample_data_with_hf):
        """Test complete workflow: fit -> predict -> backtest."""
        # Fit
        model = midas_almon.fit(sample_data_with_hf)
        assert model._is_fitted

        # Predict
        target_date = pd.Timestamp("2024-01-01")
        pred_result = model.predict(sample_data_with_hf, target_date)
        assert "prediction" in pred_result

        # Forecast
        forecast = model.forecast(6)
        assert len(forecast) == 6

        # Backtest
        bt_results = model.backtest(sample_data_with_hf, start_date="2023-01-01")
        assert isinstance(bt_results, pd.DataFrame)

    def test_model_registry(self):
        """Test MIDAS is registered in model registry."""
        from sirena.models import ModelRegistry

        # Check if model is registered
        models = ModelRegistry.list_models()
        assert "midas" in models, "MIDAS should be registered"

    def test_import_from_models_package(self):
        """Test MIDAS can be imported from models package."""
        from sirena.models import MIDASForecaster

        assert MIDASForecaster is not None, "MIDASForecaster should be importable"
        assert MIDASForecaster.name == "midas", "Model name should be midas"

    def test_compare_weight_types_performance(self, sample_data_with_hf):
        """Compare performance of different weight types."""
        from sirena.models import MIDASForecaster

        results = {}

        for weight_type in ["almon", "exp", "beta"]:
            model = MIDASForecaster(weight_type=weight_type)
            bt_results = model.backtest(sample_data_with_hf, start_date="2023-01-01")

            if len(bt_results) > 0:
                mae = (bt_results["error"].abs()).mean()
                results[weight_type] = mae

        # All weight types should produce some results
        assert len(results) > 0, "At least one weight type should work"

        # All MAEs should be reasonable
        for weight_type, mae in results.items():
            assert mae < 10.0, (
                f"MAE for {weight_type} should be reasonable (<10), got {mae}"
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
