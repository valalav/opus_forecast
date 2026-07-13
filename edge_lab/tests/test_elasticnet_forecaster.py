"""
Unit tests for ElasticNetForecaster
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


class TestElasticNetForecaster:
    """Test suite for ElasticNetForecaster."""

    def test_import_model(self):
        """Test model import."""
        from sirena.models.elasticnet import ElasticNetForecaster

        model = ElasticNetForecaster()
        assert model is not None
        assert model.name == "elasticnet"

    def test_model_parameters(self):
        """Test model default parameters."""
        from sirena.models.elasticnet import ElasticNetForecaster

        model = ElasticNetForecaster()

        assert model.MIN_TRAIN_SIZE == 36
        assert model.OUTLIER_YEARS == [2010]
        assert model.SAMPLE_WEIGHT_2022 == 0.25
        assert model.use_macro == True
        assert model._has_macro == False
        assert model.l1_ratios == [0.1, 0.3, 0.5, 0.7, 0.9]
        assert model.alphas == [0.001, 0.01, 0.1, 0.3, 1.0]
        assert model.cv == 5

    def test_base_features_list(self):
        """Test base features list."""
        from sirena.models.elasticnet import ElasticNetForecaster

        model = ElasticNetForecaster()

        required_features = [
            "y_lag1",
            "y_lag2",
            "y_lag12",
            "y_lag3",
            "y_lag6",
            "y_ma3",
            "y_ma6",
            "d_y_lag1",
            "d_y_lag3",
            "y_vol3",
            "y_vol6",
            "month_sin",
            "month_cos",
            "quarter_sin",
            "quarter_cos",
            "is_jan",
            "is_dec",
            "is_tariff_month",
            "is_q1",
            "is_summer",
            "food_lag1",
            "nonfood_lag1",
            "services_lag1",
            "seasonal_norm",
            "deviation_lag1",
        ]

        for feature in required_features:
            assert feature in model.BASE_FEATURES

    def test_macro_features_list(self):
        """Test macro features list."""
        from sirena.models.elasticnet import ElasticNetForecaster

        model = ElasticNetForecaster()

        expected_macro = ["ruonia_diff_lag1", "spread_lag4", "ki_diff_lag6", "ki_vol"]
        assert model.MACRO_FEATURES == expected_macro

    def test_ets_weights(self):
        """Test ETS weights dictionary."""
        from sirena.models.elasticnet import ElasticNetForecaster

        model = ElasticNetForecaster()

        assert len(model.ETS_WEIGHTS) == 12
        assert 1 in model.ETS_WEIGHTS
        assert model.ETS_WEIGHTS[1] == 0.9

    def test_custom_parameters(self):
        """Test custom parameters initialization."""
        from sirena.models.elasticnet import ElasticNetForecaster

        custom_l1 = [0.2, 0.4, 0.6]
        custom_alpha = [0.05, 0.1, 0.5]
        custom_cv = 3

        model = ElasticNetForecaster(
            l1_ratios=custom_l1, alphas=custom_alpha, cv=custom_cv
        )

        assert model.l1_ratios == custom_l1
        assert model.alphas == custom_alpha
        assert model.cv == custom_cv

    def test_fit_basic(self, sample_data):
        """Test basic fit functionality."""
        from sirena.models.elasticnet import ElasticNetForecaster

        model = ElasticNetForecaster(use_macro=False)
        model.fit(sample_data)

        assert model.is_fitted
        assert model.model is not None
        assert model.scaler is not None
        assert model.seasonal_norm is not None
        assert len(model.seasonal_norm) == 12
        assert model._best_alpha is not None
        assert model._best_l1_ratio is not None

    def test_fit_with_macro(self, sample_data_with_macro):
        """Test fit with macro features."""
        from sirena.models.elasticnet import ElasticNetForecaster

        model = ElasticNetForecaster(use_macro=True)
        model.fit(sample_data_with_macro)

        assert model.is_fitted
        assert model._has_macro
        assert "ruonia_diff_lag1" in model._features

    def test_fit_without_macro(self, sample_data_with_macro):
        """Test fit without macro features."""
        from sirena.models.elasticnet import ElasticNetForecaster

        model = ElasticNetForecaster(use_macro=False)
        model.fit(sample_data_with_macro)

        assert model.is_fitted
        assert not model._has_macro
        assert "ruonia_diff_lag1" not in model._features

    def test_fit_insufficient_data(self, sample_data):
        """Test fit with insufficient data."""
        from sirena.models.elasticnet import ElasticNetForecaster

        small_data = sample_data.iloc[:30]

        model = ElasticNetForecaster()

        with pytest.raises(ValueError, match="Недостаточно данных"):
            model.fit(small_data)

    def test_fit_empty_dataframe(self):
        """Test fit with empty DataFrame."""
        from sirena.models.elasticnet import ElasticNetForecaster

        empty_df = pd.DataFrame()

        model = ElasticNetForecaster()

        with pytest.raises(ValueError, match="DataFrame пустой"):
            model.fit(empty_df)

    def test_fit_missing_target_column(self, sample_data):
        """Test fit with missing target column."""
        from sirena.models.elasticnet import ElasticNetForecaster

        df_no_target = sample_data.drop(columns=["Все товары и услуги"])

        model = ElasticNetForecaster()

        with pytest.raises(ValueError, match="Колонка.*не найдена"):
            model.fit(df_no_target)

    def test_predict_basic(self, sample_data):
        """Test basic predict functionality."""
        from sirena.models.elasticnet import ElasticNetForecaster

        model = ElasticNetForecaster(use_macro=False)
        model.fit(sample_data)

        target_date = sample_data.index[-1]
        result = model.predict(sample_data, target_date)

        assert "prediction" in result
        assert "pred_elasticnet" in result
        assert "pred_ets" in result
        assert "ets_weight" in result
        assert "model" in result
        assert "has_macro" in result
        assert "best_alpha" in result
        assert "best_l1_ratio" in result
        assert result["model"] == "elasticnet"
        assert result["best_alpha"] is not None
        assert result["best_l1_ratio"] is not None

    def test_predict_with_macro(self, sample_data_with_macro):
        """Test predict with macro features."""
        from sirena.models.elasticnet import ElasticNetForecaster

        model = ElasticNetForecaster(use_macro=True)
        model.fit(sample_data_with_macro)

        target_date = sample_data_with_macro.index[-1]
        result = model.predict(sample_data_with_macro, target_date)

        assert result["prediction"] is not None
        assert result["has_macro"] == True

    def test_predict_range(self, sample_data):
        """Test predict returns reasonable values."""
        from sirena.models.elasticnet import ElasticNetForecaster

        model = ElasticNetForecaster(use_macro=False)
        model.fit(sample_data)

        target_date = sample_data.index[-1]
        result = model.predict(sample_data, target_date)

        assert 98 < result["prediction"] < 102
        assert 98 < result["pred_elasticnet"] < 102
        assert 98 < result["pred_ets"] < 102
        assert 0 <= result["ets_weight"] <= 1
        assert result["best_alpha"] > 0
        assert 0 <= result["best_l1_ratio"] <= 1

    def test_predict_not_fitted_error(self, sample_data):
        """Test predict raises error when model not fitted."""
        from sirena.models.elasticnet import ElasticNetForecaster

        model = ElasticNetForecaster()

        target_date = sample_data.index[-1]

        with pytest.raises(ValueError, match="не обучена"):
            model.predict(sample_data, target_date)

    def test_forecast_basic(self, sample_data):
        """Test forecast functionality."""
        from sirena.models.elasticnet import ElasticNetForecaster

        model = ElasticNetForecaster(use_macro=False)
        model.fit(sample_data)

        horizon = 12
        forecast = model.forecast(horizon)

        assert len(forecast) == horizon
        assert all(isinstance(v, (int, float)) for v in forecast)

    def test_forecast_different_horizons(self, sample_data):
        """Test forecast with different horizons."""
        from sirena.models.elasticnet import ElasticNetForecaster

        model = ElasticNetForecaster(use_macro=False)
        model.fit(sample_data)

        for h in [1, 6, 12, 24]:
            forecast = model.forecast(h)
            assert len(forecast) == h

    def test_forecast_not_fitted_error(self, sample_data):
        """Test forecast raises error when model not fitted."""
        from sirena.models.elasticnet import ElasticNetForecaster

        model = ElasticNetForecaster()

        with pytest.raises(ValueError, match="не обучена"):
            model.forecast(12)

    def test_backtest_basic(self, sample_data):
        """Test backtest functionality."""
        from sirena.models.elasticnet import ElasticNetForecaster

        model = ElasticNetForecaster(use_macro=False)
        results = model.backtest(sample_data, start_date="2023-01-01")

        assert isinstance(results, pd.DataFrame)

        if not results.empty:
            assert "date" in results.columns
            assert "actual" in results.columns
            assert "prediction" in results.columns
            assert "error" in results.columns
            assert "pred_elasticnet" in results.columns
            assert "best_alpha" in results.columns
            assert "best_l1_ratio" in results.columns
            assert "has_macro" in results.columns

    def test_backtest_custom_start_date(self, sample_data):
        """Test backtest with custom start date."""
        from sirena.models.elasticnet import ElasticNetForecaster

        model = ElasticNetForecaster(use_macro=False)
        results = model.backtest(sample_data, start_date="2022-06-01")

        assert isinstance(results, pd.DataFrame)

        if not results.empty:
            min_date = results["date"].min()
            assert min_date >= pd.Timestamp("2022-06-01")

    def test_backtest_with_macro(self, sample_data_with_macro):
        """Test backtest with macro features."""
        from sirena.models.elasticnet import ElasticNetForecaster

        model = ElasticNetForecaster(use_macro=True)
        results = model.backtest(sample_data_with_macro, start_date="2023-01-01")

        assert isinstance(results, pd.DataFrame)

        if not results.empty:
            assert "has_macro" in results.columns
            assert results["has_macro"].iloc[0] == True

    def test_get_feature_importance(self, sample_data):
        """Test get feature importance."""
        from sirena.models.elasticnet import ElasticNetForecaster

        model = ElasticNetForecaster(use_macro=False)
        model.fit(sample_data)

        importance = model.get_feature_importance()

        assert isinstance(importance, pd.DataFrame)
        assert "feature" in importance.columns
        assert "coefficient" in importance.columns
        assert "abs_coef" in importance.columns
        assert "is_selected" in importance.columns
        assert "is_macro" in importance.columns

    def test_get_feature_importance_sorted(self, sample_data):
        """Test feature importance is sorted by absolute coefficient."""
        from sirena.models.elasticnet import ElasticNetForecaster

        model = ElasticNetForecaster(use_macro=False)
        model.fit(sample_data)

        importance = model.get_feature_importance()

        abs_coefs = importance["abs_coef"].values
        assert all(abs_coefs[i] >= abs_coefs[i + 1] for i in range(len(abs_coefs) - 1))

    def test_get_feature_importance_not_fitted_error(self, sample_data):
        """Test get_feature_importance raises error when not fitted."""
        from sirena.models.elasticnet import ElasticNetForecaster

        model = ElasticNetForecaster()

        with pytest.raises(ValueError, match="не обучена"):
            model.get_feature_importance()

    def test_get_selected_features(self, sample_data):
        """Test get selected features."""
        from sirena.models.elasticnet import ElasticNetForecaster

        model = ElasticNetForecaster(use_macro=False)
        model.fit(sample_data)

        selected = model.get_selected_features()

        assert isinstance(selected, list)
        assert all(isinstance(f, str) for f in selected)
        assert len(selected) <= len(model._features)

    def test_get_selected_features_not_fitted_error(self, sample_data):
        """Test get_selected_features raises error when not fitted."""
        from sirena.models.elasticnet import ElasticNetForecaster

        model = ElasticNetForecaster()

        with pytest.raises(ValueError, match="не обучена"):
            model.get_selected_features()

    def test_get_model_info(self, sample_data):
        """Test get model info."""
        from sirena.models.elasticnet import ElasticNetForecaster

        model = ElasticNetForecaster(use_macro=False)
        model.fit(sample_data)

        info = model.get_model_info()

        assert isinstance(info, dict)
        assert "name" in info
        assert "best_alpha" in info
        assert "best_l1_ratio" in info
        assert "features_count" in info
        assert "selected_features_count" in info
        assert "has_macro" in info
        assert "is_fitted" in info
        assert info["name"] == "elasticnet"
        assert info["best_alpha"] == model._best_alpha
        assert info["best_l1_ratio"] == model._best_l1_ratio
        assert info["is_fitted"] == True
        assert info["has_macro"] == False
        assert info["features_count"] == len(model._features)
        assert info["selected_features_count"] <= info["features_count"]

    def test_get_model_info_with_macro(self, sample_data_with_macro):
        """Test get model info with macro features."""
        from sirena.models.elasticnet import ElasticNetForecaster

        model = ElasticNetForecaster(use_macro=True)
        model.fit(sample_data_with_macro)

        info = model.get_model_info()

        assert info["has_macro"] == True
        assert info["features_count"] > len(model.BASE_FEATURES)

    def test_get_model_info_not_fitted(self):
        """Test get model info when not fitted."""
        from sirena.models.elasticnet import ElasticNetForecaster

        model = ElasticNetForecaster()

        info = model.get_model_info()

        assert isinstance(info, dict)
        assert info["is_fitted"] == False
        assert info["features_count"] == 0
        assert info["selected_features_count"] == 0

    def test_prepare_features(self, sample_data):
        """Test feature preparation."""
        from sirena.models.elasticnet import ElasticNetForecaster

        model = ElasticNetForecaster()

        df_prep = model._prepare_features(sample_data)

        assert "month" in df_prep.columns
        assert "year" in df_prep.columns
        assert "quarter" in df_prep.columns
        assert "y_lag1" in df_prep.columns
        assert "y_lag3" in df_prep.columns
        assert "y_lag6" in df_prep.columns
        assert "d_y_lag1" in df_prep.columns
        assert "d_y_lag3" in df_prep.columns
        assert "y_vol3" in df_prep.columns
        assert "y_vol6" in df_prep.columns
        assert "is_jan" in df_prep.columns
        assert "is_tariff_month" in df_prep.columns

    def test_prepare_features_components(self, sample_data):
        """Test feature preparation with component lags."""
        from sirena.models.elasticnet import ElasticNetForecaster

        model = ElasticNetForecaster()

        df_prep = model._prepare_features(sample_data)

        assert "food_lag1" in df_prep.columns
        assert "nonfood_lag1" in df_prep.columns
        assert "services_lag1" in df_prep.columns

    def test_add_macro_features(self, sample_data_with_macro):
        """Test macro features addition."""
        from sirena.models.elasticnet import ElasticNetForecaster

        model = ElasticNetForecaster()

        df_prep = model._add_macro_features(sample_data_with_macro)

        assert "ruonia_diff" in df_prep.columns
        assert "ruonia_diff_lag1" in df_prep.columns
        assert "spread" in df_prep.columns
        assert "spread_lag4" in df_prep.columns
        assert "ki_diff" in df_prep.columns
        assert "ki_diff_lag6" in df_prep.columns
        assert "ki_vol" in df_prep.columns

    def test_add_macro_features_no_macro_data(self, sample_data):
        """Test macro features with no macro data."""
        from sirena.models.elasticnet import ElasticNetForecaster

        model = ElasticNetForecaster()

        df_prep = model._add_macro_features(sample_data)

        assert "ruonia_diff_lag1" not in df_prep.columns

    def test_compute_seasonal_norm(self, sample_data):
        """Test seasonal norm computation."""
        from sirena.models.elasticnet import ElasticNetForecaster

        model = ElasticNetForecaster()

        df_prep = model._prepare_features(sample_data)
        seasonal_norm = model._compute_seasonal_norm(df_prep)

        assert len(seasonal_norm) == 12
        assert all(month in seasonal_norm.index for month in range(1, 13))

    def test_check_fitted(self):
        """Test _check_fitted method."""
        from sirena.models.elasticnet import ElasticNetForecaster

        model = ElasticNetForecaster()

        with pytest.raises(ValueError, match="не обучена"):
            model._check_fitted()

    def test_use_macro_parameter(self):
        """Test use_macro parameter."""
        from sirena.models.elasticnet import ElasticNetForecaster

        model = ElasticNetForecaster(use_macro=False)

        assert model.use_macro == False

    def test_repr(self):
        """Test model string representation."""
        from sirena.models.elasticnet import ElasticNetForecaster

        model = ElasticNetForecaster()
        repr_str = repr(model)

        assert "ElasticNetForecaster" in repr_str

    def test_outlier_years_exclusion(self, sample_data):
        """Test that outlier years are excluded from training."""
        from sirena.models.elasticnet import ElasticNetForecaster

        dates_with_2010 = pd.date_range("2010-01-01", periods=180, freq="MS")
        np.random.seed(42)

        data_with_2010 = pd.DataFrame(
            {
                "Все товары и услуги": 100.5 + np.random.randn(180) * 0.3,
                "Продовольственные товары": 100.6 + np.random.randn(180) * 0.4,
                "Непродовольственные товары": 100.3 + np.random.randn(180) * 0.2,
                "Услуги": 100.4 + np.random.randn(180) * 0.3,
            },
            index=dates_with_2010,
        )

        model = ElasticNetForecaster(use_macro=False)
        model.fit(data_with_2010)

        assert model._last_train_date is not None

    def test_ets_weight_application(self, sample_data):
        """Test ETS weights are applied correctly."""
        from sirena.models.elasticnet import ElasticNetForecaster

        model = ElasticNetForecaster(use_macro=False)
        model.fit(sample_data)

        target_date = sample_data.index[-1]
        result = model.predict(sample_data, target_date)

        weight = model.ETS_WEIGHTS.get(target_date.month, 0.3)

        expected_pred = (1 - weight) * result["pred_elasticnet"] + weight * result[
            "pred_ets"
        ]

        assert abs(result["prediction"] - expected_pred) < 0.0001

    def test_sample_weighting_2022(self, sample_data_with_macro):
        """Test that 2022 gets lower sample weight."""
        from sirena.models.elasticnet import ElasticNetForecaster

        dates_with_2022 = pd.date_range("2016-01-01", periods=120, freq="MS")
        np.random.seed(42)

        data_with_2022 = pd.DataFrame(
            {
                "Все товары и услуги": 100.5 + np.random.randn(120) * 0.3,
                "Продовольственные товары": 100.6 + np.random.randn(120) * 0.4,
                "Непродовольственные товары": 100.3 + np.random.randn(120) * 0.2,
                "Услуги": 100.4 + np.random.randn(120) * 0.3,
                "Ki": 16 + np.random.randn(120) * 0.5,
                "Ruonia": 15 + np.random.randn(120) * 0.5,
            },
            index=dates_with_2022,
        )

        model = ElasticNetForecaster(use_macro=True)
        model.fit(data_with_2022)

        assert model.is_fitted
        assert model.SAMPLE_WEIGHT_2022 == 0.25

    def test_elasticnet_feature_selection(self, sample_data):
        """Test that ElasticNet performs feature selection."""
        from sirena.models.elasticnet import ElasticNetForecaster

        model = ElasticNetForecaster(use_macro=False)
        model.fit(sample_data)

        importance = model.get_feature_importance()
        selected_count = len(importance[importance["is_selected"]])

        assert selected_count <= len(model._features)

    def test_backtest_custom_target_column(self, sample_data):
        """Test backtest with custom target column."""
        from sirena.models.elasticnet import ElasticNetForecaster

        model = ElasticNetForecaster(use_macro=False)
        results = model.backtest(
            sample_data, start_date="2023-01-01", target_col="Продовольственные товары"
        )

        assert isinstance(results, pd.DataFrame)

        if not results.empty:
            assert "actual" in results.columns
            assert "prediction" in results.columns
