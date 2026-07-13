"""
Unit tests for EBMForecaster
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
    dates = pd.date_range("2020-01-01", periods=60, freq="MS")
    np.random.seed(42)

    data = pd.DataFrame(
        {
            "Все товары и услуги": 100.5 + np.random.randn(60) * 0.3,
            "Продовольственные товары": 100.6 + np.random.randn(60) * 0.4,
            "Непродовольственные товары": 100.3 + np.random.randn(60) * 0.2,
            "Услуги": 100.4 + np.random.randn(60) * 0.3,
        },
        index=dates,
    )

    return data


@pytest.fixture
def sample_data_with_macro():
    """Generate sample inflation data with macro features."""
    dates = pd.date_range("2020-01-01", periods=60, freq="MS")
    np.random.seed(42)

    data = pd.DataFrame(
        {
            "Все товары и услуги": 100.5 + np.random.randn(60) * 0.3,
            "Продовольственные товары": 100.6 + np.random.randn(60) * 0.4,
            "Непродовольственные товары": 100.3 + np.random.randn(60) * 0.2,
            "Услуги": 100.4 + np.random.randn(60) * 0.3,
            "Ki": 16 + np.random.randn(60) * 0.5,
            "Ruonia": 15 + np.random.randn(60) * 0.5,
        },
        index=dates,
    )

    return data


class TestEBMForecaster:
    """Test suite for EBMForecaster."""

    def test_import_model(self):
        """Test model import."""
        from sirena.models.ebm import EBMForecaster

        model = EBMForecaster()
        assert model is not None
        assert model.name == "ebm"

    def test_model_parameters(self):
        """Test model default parameters."""
        from sirena.models.ebm import EBMForecaster

        model = EBMForecaster()

        assert model.max_bins == 256
        assert model.max_interaction_bins == 32
        assert model.interactions == 0
        assert model.outer_bags == 8
        assert model.inner_bags == 0
        assert model.learning_rate == 0.01
        assert model.min_samples_leaf == 2
        assert model.max_leaves == 3
        assert model.MIN_TRAIN_SIZE == 24
        assert model.OUTLIER_YEARS == [2022, 2010]

    def test_base_features_list(self):
        """Test base features list."""
        from sirena.models.ebm import EBMForecaster

        model = EBMForecaster()

        required_features = [
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

        for feature in required_features:
            assert feature in model.BASE_FEATURES

    def test_fit_basic(self, sample_data):
        """Test basic fit functionality."""
        from sirena.models.ebm import EBMForecaster

        model = EBMForecaster()
        model.fit(sample_data)

        assert model.is_fitted
        assert model.model is not None
        assert model.scaler is not None
        assert model.seasonal_mean is not None
        assert len(model.seasonal_mean) == 12

    def test_fit_custom_parameters(self, sample_data):
        """Test fit with custom parameters."""
        from sirena.models.ebm import EBMForecaster

        model = EBMForecaster(
            max_bins=128,
            max_interaction_bins=16,
            interactions=5,
            outer_bags=10,
            inner_bags=2,
            learning_rate=0.02,
            min_samples_leaf=3,
            max_leaves=4,
        )
        model.fit(sample_data)

        assert model.is_fitted
        assert model.max_bins == 128
        assert model.interactions == 5

    def test_fit_insufficient_data(self, sample_data):
        """Test fit with insufficient data."""
        from sirena.models.ebm import EBMForecaster

        small_data = sample_data.iloc[:20]

        model = EBMForecaster()

        with pytest.raises(ValueError, match="Недостаточно данных"):
            model.fit(small_data)

    def test_fit_empty_dataframe(self):
        """Test fit with empty DataFrame."""
        from sirena.models.ebm import EBMForecaster

        empty_df = pd.DataFrame()

        model = EBMForecaster()

        with pytest.raises(ValueError, match="DataFrame пустой"):
            model.fit(empty_df)

    def test_fit_missing_target_column(self, sample_data):
        """Test fit with missing target column."""
        from sirena.models.ebm import EBMForecaster

        df_no_target = sample_data.drop(columns=["Все товары и услуги"])

        model = EBMForecaster()

        with pytest.raises(ValueError, match="Колонка.*не найдена"):
            model.fit(df_no_target)

    def test_predict_basic(self, sample_data):
        """Test basic predict functionality."""
        from sirena.models.ebm import EBMForecaster

        model = EBMForecaster()
        model.fit(sample_data)

        target_date = sample_data.index[-1]
        result = model.predict(sample_data, target_date)

        assert "prediction" in result
        assert "date" in result
        assert "model" in result
        assert "explanation" in result
        assert result["model"] == "ebm"
        assert result["date"] == target_date

    def test_predict_range(self, sample_data):
        """Test predict returns reasonable values."""
        from sirena.models.ebm import EBMForecaster

        model = EBMForecaster()
        model.fit(sample_data)

        target_date = sample_data.index[-1]
        result = model.predict(sample_data, target_date)

        assert 98 < result["prediction"] < 102

    def test_predict_not_fitted_error(self, sample_data):
        """Test predict raises error when model not fitted."""
        from sirena.models.ebm import EBMForecaster

        model = EBMForecaster()

        target_date = sample_data.index[-1]

        with pytest.raises(ValueError, match="не обучена"):
            model.predict(sample_data, target_date)

    def test_predict_explanation_structure(self, sample_data):
        """Test explanation structure in predict."""
        from sirena.models.ebm import EBMForecaster

        model = EBMForecaster()
        model.fit(sample_data)

        target_date = sample_data.index[-1]
        result = model.predict(sample_data, target_date)

        explanation = result["explanation"]

        if explanation is not None:
            assert "intercept" in explanation
            assert "contributions" in explanation
            assert "total" in explanation

    def test_forecast_basic(self, sample_data):
        """Test forecast functionality."""
        from sirena.models.ebm import EBMForecaster

        model = EBMForecaster()
        model.fit(sample_data)

        horizon = 12
        forecast = model.forecast(horizon)

        assert len(forecast) == horizon
        assert all(isinstance(v, (int, float)) for v in forecast)

    def test_forecast_different_horizons(self, sample_data):
        """Test forecast with different horizons."""
        from sirena.models.ebm import EBMForecaster

        model = EBMForecaster()
        model.fit(sample_data)

        for h in [1, 6, 12, 24]:
            forecast = model.forecast(h)
            assert len(forecast) == h

    def test_forecast_not_fitted_error(self, sample_data):
        """Test forecast raises error when model not fitted."""
        from sirena.models.ebm import EBMForecaster

        model = EBMForecaster()

        with pytest.raises(ValueError, match="не обучена"):
            model.forecast(12)

    def test_backtest_basic(self, sample_data):
        """Test backtest functionality."""
        from sirena.models.ebm import EBMForecaster

        model = EBMForecaster()
        results = model.backtest(sample_data, start_date="2023-01-01")

        assert isinstance(results, pd.DataFrame)

        if not results.empty:
            assert "date" in results.columns
            assert "actual" in results.columns
            assert "prediction" in results.columns
            assert "error" in results.columns

    def test_backtest_custom_start_date(self, sample_data):
        """Test backtest with custom start date."""
        from sirena.models.ebm import EBMForecaster

        model = EBMForecaster()
        results = model.backtest(sample_data, start_date="2022-06-01")

        assert isinstance(results, pd.DataFrame)

        if not results.empty:
            min_date = results["date"].min()
            assert min_date >= pd.Timestamp("2022-06-01")

    def test_get_feature_importance(self, sample_data):
        """Test get feature importance."""
        from sirena.models.ebm import EBMForecaster

        model = EBMForecaster()
        model.fit(sample_data)

        importance = model.get_feature_importance()

        assert isinstance(importance, pd.DataFrame)
        assert "feature" in importance.columns
        assert "importance" in importance.columns

    def test_get_feature_importance_sorted(self, sample_data):
        """Test feature importance is sorted."""
        from sirena.models.ebm import EBMForecaster

        model = EBMForecaster()
        model.fit(sample_data)

        importance = model.get_feature_importance()

        importance_values = importance["importance"].values
        if len(importance_values) > 1:
            assert all(
                abs(importance_values[i]) >= abs(importance_values[i + 1])
                for i in range(len(importance_values) - 1)
            )

    def test_get_feature_importance_not_fitted_error(self, sample_data):
        """Test get_feature_importance raises error when not fitted."""
        from sirena.models.ebm import EBMForecaster

        model = EBMForecaster()

        with pytest.raises(ValueError, match="не обучена"):
            model.get_feature_importance()

    def test_explain(self, sample_data):
        """Test explain method for global interpretability."""
        from sirena.models.ebm import EBMForecaster

        model = EBMForecaster()
        model.fit(sample_data)

        explanation = model.explain()

        assert isinstance(explanation, dict)
        assert "type" in explanation
        assert "feature_importance" in explanation
        assert explanation["type"] in ["ebm", "fallback_ridge"]

    def test_explain_ebm_type(self, sample_data):
        """Test explain returns EBM type explanation when available."""
        from sirena.models.ebm import EBMForecaster

        model = EBMForecaster()
        model.fit(sample_data)

        explanation = model.explain()

        if not model._use_fallback:
            assert explanation["type"] == "ebm"
            assert "feature_shapes" in explanation
            assert "intercept" in explanation

    def test_explain_fallback_ridge_type(self, sample_data):
        """Test explain returns fallback type when InterpretML unavailable."""
        from sirena.models.ebm import EBMForecaster

        model = EBMForecaster()
        model._use_fallback = True
        model.fit(sample_data)

        explanation = model.explain()

        assert explanation["type"] == "fallback_ridge"
        assert "feature_importance" in explanation

    def test_prepare_features(self, sample_data):
        """Test feature preparation."""
        from sirena.models.ebm import EBMForecaster

        model = EBMForecaster()

        df_prep = model._prepare_features(sample_data)

        assert "month" in df_prep.columns
        assert "year" in df_prep.columns
        assert "y_lag1" in df_prep.columns
        assert "y_lag2" in df_prep.columns
        assert "y_lag12" in df_prep.columns
        assert "y_ma3" in df_prep.columns
        assert "month_sin" in df_prep.columns
        assert "month_cos" in df_prep.columns

    def test_prepare_features_components(self, sample_data):
        """Test feature preparation with component lags."""
        from sirena.models.ebm import EBMForecaster

        model = EBMForecaster()

        df_prep = model._prepare_features(sample_data)

        assert "food_lag1" in df_prep.columns
        assert "nonfood_lag1" in df_prep.columns
        assert "services_lag1" in df_prep.columns

    def test_prepare_features_missing_components(self, sample_data):
        """Test feature preparation when component columns are missing."""
        from sirena.models.ebm import EBMForecaster

        model = EBMForecaster()

        df_no_components = sample_data.drop(
            columns=[
                "Продовольственные товары",
                "Непродовольственные товары",
                "Услуги",
            ]
        )

        df_prep = model._prepare_features(df_no_components)

        # Fallback to y_lag1 when components are missing
        assert "food_lag1" in df_prep.columns
        assert "nonfood_lag1" in df_prep.columns
        assert "services_lag1" in df_prep.columns

    def test_check_fitted(self):
        """Test _check_fitted method."""
        from sirena.models.ebm import EBMForecaster

        model = EBMForecaster()

        with pytest.raises(ValueError, match="не обучена"):
            model._check_fitted()

    def test_custom_parameters(self):
        """Test custom EBM parameters."""
        from sirena.models.ebm import EBMForecaster

        model = EBMForecaster(
            max_bins=64,
            interactions=3,
            outer_bags=5,
            learning_rate=0.05,
        )

        assert model.max_bins == 64
        assert model.interactions == 3
        assert model.outer_bags == 5
        assert model.learning_rate == 0.05

    def test_repr(self):
        """Test model string representation."""
        from sirena.models.ebm import EBMForecaster

        model = EBMForecaster()
        repr_str = repr(model)

        assert "EBMForecaster" in repr_str

    def test_outlier_years_exclusion(self, sample_data):
        """Test that outlier years are excluded from training."""
        from sirena.models.ebm import EBMForecaster

        dates_with_outliers = pd.date_range("2010-01-01", periods=180, freq="MS")
        np.random.seed(42)

        data_with_outliers = pd.DataFrame(
            {
                "Все товары и услуги": 100.5 + np.random.randn(180) * 0.3,
                "Продовольственные товары": 100.6 + np.random.randn(180) * 0.4,
                "Непродовольственные товары": 100.3 + np.random.randn(180) * 0.2,
                "Услуги": 100.4 + np.random.randn(180) * 0.3,
            },
            index=dates_with_outliers,
        )

        model = EBMForecaster()
        model.fit(data_with_outliers)

        assert model._last_train_date is not None

    def test_get_metrics(self, sample_data):
        """Test get_metrics method."""
        from sirena.models.ebm import EBMForecaster

        model = EBMForecaster()
        results = model.backtest(sample_data, start_date="2023-01-01")

        metrics = model.get_metrics(results)

        assert isinstance(metrics, dict)
        assert "MAE" in metrics
        assert "RMSE" in metrics
        assert "KPI" in metrics

    def test_get_metrics_empty_dataframe(self, sample_data):
        """Test get_metrics with empty DataFrame."""
        from sirena.models.ebm import EBMForecaster

        model = EBMForecaster()
        empty_results = pd.DataFrame()

        metrics = model.get_metrics(empty_results)

        assert metrics["MAE"] == 0
        assert metrics["RMSE"] == 0
        assert metrics["KPI"] == 0

    def test_local_explanation_structure(self, sample_data):
        """Test local explanation structure."""
        from sirena.models.ebm import EBMForecaster

        model = EBMForecaster()
        model.fit(sample_data)

        target_date = sample_data.index[-1]
        result = model.predict(sample_data, target_date)

        explanation = result["explanation"]

        if explanation is not None:
            assert "intercept" in explanation
            assert "contributions" in explanation
            assert "total" in explanation

            contributions = explanation["contributions"]
            for feature in model.BASE_FEATURES:
                if feature in contributions:
                    assert "value" in contributions[feature]
                    assert "contribution" in contributions[feature]

    def test_seasonal_mean_computation(self, sample_data):
        """Test seasonal mean computation during fit."""
        from sirena.models.ebm import EBMForecaster

        model = EBMForecaster()
        model.fit(sample_data)

        assert model.seasonal_mean is not None
        assert len(model.seasonal_mean) == 12
        assert all(month in model.seasonal_mean.index for month in range(1, 13))

    def test_fallback_model_initialization(self):
        """Test fallback model is initialized when InterpretML unavailable."""
        from sirena.models.ebm import EBMForecaster, INTERPRET_AVAILABLE

        model = EBMForecaster()

        if not INTERPRET_AVAILABLE:
            assert model._use_fallback is True
            assert model.fallback_model is not None

    def test_interactions_parameter(self):
        """Test interactions parameter controls feature interactions."""
        from sirena.models.ebm import EBMForecaster

        model_no_interactions = EBMForecaster(interactions=0)
        model_with_interactions = EBMForecaster(interactions=5)

        assert model_no_interactions.interactions == 0
        assert model_with_interactions.interactions == 5

    def test_bags_parameters(self):
        """Test outer_bags and inner_bags parameters."""
        from sirena.models.ebm import EBMForecaster

        model = EBMForecaster(outer_bags=10, inner_bags=5)

        assert model.outer_bags == 10
        assert model.inner_bags == 5
