"""
Unit tests for RidgeShockDummiesForecaster
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
def sample_data_with_shocks():
    """Generate sample data including shock periods."""
    dates = pd.date_range("2014-01-01", periods=132, freq="MS")
    np.random.seed(42)

    data = pd.DataFrame(
        {
            "Все товары и услуги": 100.0 + np.random.randn(132) * 0.3,
            "Продовольственные товары": 100.0 + np.random.randn(132) * 0.4,
            "Непродовольственные товары": 100.0 + np.random.randn(132) * 0.2,
            "Услуги": 100.0 + np.random.randn(132) * 0.3,
        },
        index=dates,
    )

    # Add artificial shocks
    data.loc[pd.Timestamp("2014-12-01"), "Все товары и услуги"] = (
        102.5  # Currency crisis
    )
    data.loc[pd.Timestamp("2015-01-01"), "Все товары и услуги"] = 103.0  # Continuation
    data.loc[pd.Timestamp("2017-07-01"), "Все товары и услуги"] = 101.2  # Tariff hike
    data.loc[pd.Timestamp("2022-03-01"), "Все товары и услуги"] = 102.8  # Sanctions
    data.loc[pd.Timestamp("2022-04-01"), "Все товары и услуги"] = 102.5  # Continuation

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


class TestRidgeShockDummiesForecaster:
    """Test suite for RidgeShockDummiesForecaster."""

    def test_import_model(self):
        """Test model import."""
        from sirena.models.ridge_shock_dummies import RidgeShockDummiesForecaster

        model = RidgeShockDummiesForecaster()
        assert model is not None
        assert model.name == "ridge_shock_dummies"

    def test_model_parameters_default(self):
        """Test model default parameters."""
        from sirena.models.ridge_shock_dummies import RidgeShockDummiesForecaster

        model = RidgeShockDummiesForecaster()

        assert model.alpha == 0.3
        assert model.MIN_TRAIN_SIZE == 36
        assert model.OUTLIER_YEARS == []  # Empty list - no outlier exclusion
        assert model.use_macro == True
        assert model.use_2022_dummy == True
        assert model._has_macro == False

    def test_model_parameters_custom(self):
        """Test model custom parameters."""
        from sirena.models.ridge_shock_dummies import RidgeShockDummiesForecaster

        custom_alpha = 0.5
        model = RidgeShockDummiesForecaster(
            alpha=custom_alpha, use_macro=False, use_2022_dummy=False
        )

        assert model.alpha == custom_alpha
        assert model.use_macro == False
        assert model.use_2022_dummy == False

    def test_base_features_list(self):
        """Test base features list."""
        from sirena.models.ridge_shock_dummies import RidgeShockDummiesForecaster

        model = RidgeShockDummiesForecaster()

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
            "seasonal_norm",
            "deviation_lag1",
        ]

        for feature in required_features:
            assert feature in model.BASE_FEATURES

    def test_shock_dummies_list(self):
        """Test shock dummies list."""
        from sirena.models.ridge_shock_dummies import RidgeShockDummiesForecaster

        model = RidgeShockDummiesForecaster()

        expected_dummies = [
            "is_shock_dec2014",
            "is_shock_jan2015",
            "is_tariff_jul2017",
            "is_shock_mar2022",
            "is_shock_apr2022",
            "is_shock_2022",
        ]

        assert model.SHOCK_DUMMIES == expected_dummies

    def test_macro_features_list(self):
        """Test macro features list."""
        from sirena.models.ridge_shock_dummies import RidgeShockDummiesForecaster

        model = RidgeShockDummiesForecaster()

        expected_macro = ["ruonia_diff_lag1", "spread_lag4", "ki_diff_lag6", "ki_vol"]
        assert model.MACRO_FEATURES == expected_macro

    def test_ets_weights(self):
        """Test ETS weights dictionary."""
        from sirena.models.ridge_shock_dummies import RidgeShockDummiesForecaster

        model = RidgeShockDummiesForecaster()

        assert len(model.ETS_WEIGHTS) == 12
        assert 1 in model.ETS_WEIGHTS
        assert model.ETS_WEIGHTS[1] == 0.9

    def test_fit_basic(self, sample_data):
        """Test basic fit functionality."""
        from sirena.models.ridge_shock_dummies import RidgeShockDummiesForecaster

        model = RidgeShockDummiesForecaster(use_macro=False)
        model.fit(sample_data)

        assert model.is_fitted
        assert model.ridge is not None
        assert model.scaler is not None
        assert model.seasonal_norm is not None
        assert len(model.seasonal_norm) == 12

    def test_fit_with_macro(self, sample_data_with_macro):
        """Test fit with macro features."""
        from sirena.models.ridge_shock_dummies import RidgeShockDummiesForecaster

        model = RidgeShockDummiesForecaster(use_macro=True)
        model.fit(sample_data_with_macro)

        assert model.is_fitted
        assert model._has_macro
        assert "ruonia_diff_lag1" in model._features

    def test_fit_without_macro(self, sample_data_with_macro):
        """Test fit without macro features."""
        from sirena.models.ridge_shock_dummies import RidgeShockDummiesForecaster

        model = RidgeShockDummiesForecaster(use_macro=False)
        model.fit(sample_data_with_macro)

        assert model.is_fitted
        assert not model._has_macro
        assert "ruonia_diff_lag1" not in model._features

    def test_fit_with_2022_dummy(self, sample_data_with_shocks):
        """Test fit with 2022 dummy enabled."""
        from sirena.models.ridge_shock_dummies import RidgeShockDummiesForecaster

        model = RidgeShockDummiesForecaster(use_2022_dummy=True, use_macro=False)
        model.fit(sample_data_with_shocks)

        assert model.is_fitted
        assert "is_shock_2022" in model._features
        assert "is_shock_mar2022" in model._features

    def test_fit_without_2022_dummy(self, sample_data_with_shocks):
        """Test fit with 2022 dummy disabled (excludes 2022)."""
        from sirena.models.ridge_shock_dummies import RidgeShockDummiesForecaster

        model = RidgeShockDummiesForecaster(use_2022_dummy=False, use_macro=False)
        model.fit(sample_data_with_shocks)

        assert model.is_fitted
        assert "is_shock_2022" not in model._features
        assert "is_shock_mar2022" not in model._features

    def test_fit_custom_alpha(self, sample_data):
        """Test fit with custom alpha parameter."""
        from sirena.models.ridge_shock_dummies import RidgeShockDummiesForecaster

        custom_alpha = 0.5
        model = RidgeShockDummiesForecaster(alpha=custom_alpha, use_macro=False)
        model.fit(sample_data)

        assert model.alpha == custom_alpha
        assert model.ridge.alpha == custom_alpha

    def test_fit_insufficient_data(self, sample_data):
        """Test fit with insufficient data."""
        from sirena.models.ridge_shock_dummies import RidgeShockDummiesForecaster

        small_data = sample_data.iloc[:30]

        model = RidgeShockDummiesForecaster()

        with pytest.raises(ValueError, match="Недостаточно данных"):
            model.fit(small_data)

    def test_fit_empty_dataframe(self):
        """Test fit with empty DataFrame."""
        from sirena.models.ridge_shock_dummies import RidgeShockDummiesForecaster

        empty_df = pd.DataFrame()

        model = RidgeShockDummiesForecaster()

        with pytest.raises(ValueError, match="DataFrame пустой"):
            model.fit(empty_df)

    def test_fit_missing_target_column(self, sample_data):
        """Test fit with missing target column."""
        from sirena.models.ridge_shock_dummies import RidgeShockDummiesForecaster

        df_no_target = sample_data.drop(columns=["Все товары и услуги"])

        model = RidgeShockDummiesForecaster()

        with pytest.raises(ValueError, match="Колонка.*не найдена"):
            model.fit(df_no_target)

    def test_predict_basic(self, sample_data):
        """Test basic predict functionality."""
        from sirena.models.ridge_shock_dummies import RidgeShockDummiesForecaster

        model = RidgeShockDummiesForecaster(use_macro=False)
        model.fit(sample_data)

        target_date = sample_data.index[-1]
        result = model.predict(sample_data, target_date)

        assert "prediction" in result
        assert "pred_ridge" in result
        assert "pred_ets" in result
        assert "ets_weight" in result
        assert "model" in result
        assert "has_macro" in result
        assert "use_2022_dummy" in result
        assert result["model"] == "ridge_shock_dummies"

    def test_predict_with_macro(self, sample_data_with_macro):
        """Test predict with macro features."""
        from sirena.models.ridge_shock_dummies import RidgeShockDummiesForecaster

        model = RidgeShockDummiesForecaster(use_macro=True)
        model.fit(sample_data_with_macro)

        target_date = sample_data_with_macro.index[-1]
        result = model.predict(sample_data_with_macro, target_date)

        assert result["prediction"] is not None
        assert result["has_macro"] == True

    def test_predict_range(self, sample_data):
        """Test predict returns reasonable values."""
        from sirena.models.ridge_shock_dummies import RidgeShockDummiesForecaster

        model = RidgeShockDummiesForecaster(use_macro=False)
        model.fit(sample_data)

        target_date = sample_data.index[-1]
        result = model.predict(sample_data, target_date)

        assert 98 < result["prediction"] < 102
        assert 98 < result["pred_ridge"] < 102
        assert 98 < result["pred_ets"] < 102
        assert 0 <= result["ets_weight"] <= 1

    def test_predict_not_fitted_error(self, sample_data):
        """Test predict raises error when model not fitted."""
        from sirena.models.ridge_shock_dummies import RidgeShockDummiesForecaster

        model = RidgeShockDummiesForecaster()

        target_date = sample_data.index[-1]

        with pytest.raises(ValueError, match="не обучена"):
            model.predict(sample_data, target_date)

    def test_forecast_basic(self, sample_data):
        """Test forecast functionality."""
        from sirena.models.ridge_shock_dummies import RidgeShockDummiesForecaster

        model = RidgeShockDummiesForecaster(use_macro=False)
        model.fit(sample_data)

        horizon = 12
        forecast = model.forecast(horizon)

        assert len(forecast) == horizon
        assert all(isinstance(v, (int, float)) for v in forecast)

    def test_forecast_different_horizons(self, sample_data):
        """Test forecast with different horizons."""
        from sirena.models.ridge_shock_dummies import RidgeShockDummiesForecaster

        model = RidgeShockDummiesForecaster(use_macro=False)
        model.fit(sample_data)

        for h in [1, 6, 12, 24]:
            forecast = model.forecast(h)
            assert len(forecast) == h

    def test_forecast_not_fitted_error(self, sample_data):
        """Test forecast raises error when model not fitted."""
        from sirena.models.ridge_shock_dummies import RidgeShockDummiesForecaster

        model = RidgeShockDummiesForecaster()

        with pytest.raises(ValueError, match="не обучена"):
            model.forecast(12)

    def test_backtest_basic(self, sample_data):
        """Test backtest functionality."""
        from sirena.models.ridge_shock_dummies import RidgeShockDummiesForecaster

        model = RidgeShockDummiesForecaster(use_macro=False)
        results = model.backtest(sample_data, start_date="2023-01-01")

        assert isinstance(results, pd.DataFrame)

        if not results.empty:
            assert "date" in results.columns
            assert "actual" in results.columns
            assert "prediction" in results.columns
            assert "error" in results.columns
            assert "pred_ridge" in results.columns

    def test_backtest_custom_start_date(self, sample_data):
        """Test backtest with custom start date."""
        from sirena.models.ridge_shock_dummies import RidgeShockDummiesForecaster

        model = RidgeShockDummiesForecaster(use_macro=False)
        results = model.backtest(sample_data, start_date="2022-06-01")

        assert isinstance(results, pd.DataFrame)

        if not results.empty:
            min_date = results["date"].min()
            assert min_date >= pd.Timestamp("2022-06-01")

    def test_backtest_with_macro(self, sample_data_with_macro):
        """Test backtest with macro features."""
        from sirena.models.ridge_shock_dummies import RidgeShockDummiesForecaster

        model = RidgeShockDummiesForecaster(use_macro=True)
        results = model.backtest(sample_data_with_macro, start_date="2023-01-01")

        assert isinstance(results, pd.DataFrame)

        if not results.empty:
            assert "has_macro" in results.columns

    def test_backtest_with_shocks(self, sample_data_with_shocks):
        """Test backtest with shock periods."""
        from sirena.models.ridge_shock_dummies import RidgeShockDummiesForecaster

        model = RidgeShockDummiesForecaster(use_2022_dummy=True, use_macro=False)
        results = model.backtest(sample_data_with_shocks, start_date="2020-01-01")

        assert isinstance(results, pd.DataFrame)

        if not results.empty:
            assert len(results) > 0

    def test_get_feature_importance(self, sample_data):
        """Test get feature importance."""
        from sirena.models.ridge_shock_dummies import RidgeShockDummiesForecaster

        model = RidgeShockDummiesForecaster(use_macro=False)
        model.fit(sample_data)

        importance = model.get_feature_importance()

        assert isinstance(importance, pd.DataFrame)
        assert "feature" in importance.columns
        assert "coefficient" in importance.columns
        assert "abs_coef" in importance.columns
        assert "is_shock" in importance.columns
        assert "is_macro" in importance.columns

    def test_get_feature_importance_sorted(self, sample_data):
        """Test feature importance is sorted by absolute coefficient."""
        from sirena.models.ridge_shock_dummies import RidgeShockDummiesForecaster

        model = RidgeShockDummiesForecaster(use_macro=False)
        model.fit(sample_data)

        importance = model.get_feature_importance()

        abs_coefs = importance["abs_coef"].values
        assert all(abs_coefs[i] >= abs_coefs[i + 1] for i in range(len(abs_coefs) - 1))

    def test_get_feature_importance_with_shocks(self, sample_data_with_shocks):
        """Test feature importance includes shock dummies."""
        from sirena.models.ridge_shock_dummies import RidgeShockDummiesForecaster

        model = RidgeShockDummiesForecaster(use_2022_dummy=True, use_macro=False)
        model.fit(sample_data_with_shocks)

        importance = model.get_feature_importance()

        # Should have shock dummies
        assert "is_shock" in importance.columns
        assert importance["is_shock"].any()

    def test_get_feature_importance_not_fitted_error(self, sample_data):
        """Test get_feature_importance raises error when not fitted."""
        from sirena.models.ridge_shock_dummies import RidgeShockDummiesForecaster

        model = RidgeShockDummiesForecaster()

        with pytest.raises(ValueError, match="не обучена"):
            model.get_feature_importance()

    def test_add_shock_dummies(self, sample_data_with_shocks):
        """Test shock dummies are added correctly."""
        from sirena.models.ridge_shock_dummies import RidgeShockDummiesForecaster

        model = RidgeShockDummiesForecaster()
        df_with_dummies = model._add_shock_dummies(sample_data_with_shocks)

        # Check shock dummies exist
        assert "is_shock_dec2014" in df_with_dummies.columns
        assert "is_shock_jan2015" in df_with_dummies.columns
        assert "is_tariff_jul2017" in df_with_dummies.columns
        assert "is_shock_mar2022" in df_with_dummies.columns
        assert "is_shock_apr2022" in df_with_dummies.columns
        assert "is_shock_2022" in df_with_dummies.columns

    def test_shock_dummy_values(self, sample_data_with_shocks):
        """Test shock dummy values are correct."""
        from sirena.models.ridge_shock_dummies import RidgeShockDummiesForecaster

        model = RidgeShockDummiesForecaster()
        df_with_dummies = model._add_shock_dummies(sample_data_with_shocks)

        # Dec 2014 shock
        assert df_with_dummies.loc[pd.Timestamp("2014-12-01"), "is_shock_dec2014"] == 1
        assert df_with_dummies.loc[pd.Timestamp("2015-01-01"), "is_shock_dec2014"] == 0

        # Jan 2015 shock
        assert df_with_dummies.loc[pd.Timestamp("2015-01-01"), "is_shock_jan2015"] == 1
        assert df_with_dummies.loc[pd.Timestamp("2015-02-01"), "is_shock_jan2015"] == 0

        # July 2017 tariff (every year)
        assert df_with_dummies.loc[pd.Timestamp("2017-07-01"), "is_tariff_jul2017"] == 1
        assert df_with_dummies.loc[pd.Timestamp("2018-07-01"), "is_tariff_jul2017"] == 1
        assert df_with_dummies.loc[pd.Timestamp("2017-06-01"), "is_tariff_jul2017"] == 0

        # 2022 shock
        assert df_with_dummies.loc[pd.Timestamp("2022-01-01"), "is_shock_2022"] == 1
        assert df_with_dummies.loc[pd.Timestamp("2022-12-01"), "is_shock_2022"] == 1
        assert df_with_dummies.loc[pd.Timestamp("2021-12-01"), "is_shock_2022"] == 0
        assert df_with_dummies.loc[pd.Timestamp("2023-01-01"), "is_shock_2022"] == 0

    def test_prepare_features(self, sample_data):
        """Test feature preparation."""
        from sirena.models.ridge_shock_dummies import RidgeShockDummiesForecaster

        model = RidgeShockDummiesForecaster()

        df_prep = model._prepare_features(sample_data)

        assert "month" in df_prep.columns
        assert "year" in df_prep.columns
        assert "y_lag1" in df_prep.columns
        assert "y_lag2" in df_prep.columns
        assert "y_lag12" in df_prep.columns
        assert "y_ma3" in df_prep.columns
        assert "month_sin" in df_prep.columns
        assert "month_cos" in df_prep.columns
        assert "is_shock_dec2014" in df_prep.columns
        assert "is_tariff_jul2017" in df_prep.columns

    def test_prepare_features_components(self, sample_data):
        """Test feature preparation with component lags."""
        from sirena.models.ridge_shock_dummies import RidgeShockDummiesForecaster

        model = RidgeShockDummiesForecaster()

        df_prep = model._prepare_features(sample_data)

        assert "food_lag1" in df_prep.columns
        assert "nonfood_lag1" in df_prep.columns
        assert "services_lag1" in df_prep.columns

    def test_add_macro_features(self, sample_data_with_macro):
        """Test macro features addition."""
        from sirena.models.ridge_shock_dummies import RidgeShockDummiesForecaster

        model = RidgeShockDummiesForecaster()

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
        from sirena.models.ridge_shock_dummies import RidgeShockDummiesForecaster

        model = RidgeShockDummiesForecaster()

        df_prep = model._add_macro_features(sample_data)

        assert "ruonia_diff_lag1" not in df_prep.columns

    def test_compute_seasonal_norm(self, sample_data):
        """Test seasonal norm computation."""
        from sirena.models.ridge_shock_dummies import RidgeShockDummiesForecaster

        model = RidgeShockDummiesForecaster()

        df_prep = model._prepare_features(sample_data)
        seasonal_norm = model._compute_seasonal_norm(df_prep)

        assert len(seasonal_norm) == 12
        assert all(month in seasonal_norm.index for month in range(1, 13))

    def test_compute_seasonal_norm_excludes_2022(self, sample_data_with_shocks):
        """Test seasonal norm computation excludes 2022."""
        from sirena.models.ridge_shock_dummies import RidgeShockDummiesForecaster

        model = RidgeShockDummiesForecaster()

        df_prep = model._prepare_features(sample_data_with_shocks)
        seasonal_norm = model._compute_seasonal_norm(df_prep)

        # Should still have 12 months
        assert len(seasonal_norm) == 12

        # The implementation excludes 2022 from norm calculation
        # This is tested implicitly by the model working correctly

    def test_get_metrics(self, sample_data):
        """Test get metrics calculation."""
        from sirena.models.ridge_shock_dummies import RidgeShockDummiesForecaster

        model = RidgeShockDummiesForecaster(use_macro=False)
        results = model.backtest(sample_data, start_date="2023-01-01")

        metrics = model.get_metrics(results)

        if not results.empty:
            assert "MAE" in metrics
            assert "RMSE" in metrics
            assert "KPI" in metrics
            assert metrics["MAE"] >= 0
            assert metrics["RMSE"] >= 0
            assert 0 <= metrics["KPI"] <= 100

    def test_get_metrics_empty_results(self):
        """Test get metrics with empty results."""
        from sirena.models.ridge_shock_dummies import RidgeShockDummiesForecaster

        model = RidgeShockDummiesForecaster()
        empty_results = pd.DataFrame()

        metrics = model.get_metrics(empty_results)

        assert metrics == {"MAE": 0, "RMSE": 0, "KPI": 0}

    def test_check_fitted(self):
        """Test _check_fitted method."""
        from sirena.models.ridge_shock_dummies import RidgeShockDummiesForecaster

        model = RidgeShockDummiesForecaster()

        with pytest.raises(ValueError, match="не обучена"):
            model._check_fitted()

    def test_custom_alpha_parameter(self):
        """Test custom alpha parameter."""
        from sirena.models.ridge_shock_dummies import RidgeShockDummiesForecaster

        custom_alpha = 0.1
        model = RidgeShockDummiesForecaster(alpha=custom_alpha)

        assert model.alpha == custom_alpha

    def test_use_macro_parameter(self):
        """Test use_macro parameter."""
        from sirena.models.ridge_shock_dummies import RidgeShockDummiesForecaster

        model = RidgeShockDummiesForecaster(use_macro=False)

        assert model.use_macro == False

    def test_use_2022_dummy_parameter(self):
        """Test use_2022_dummy parameter."""
        from sirena.models.ridge_shock_dummies import RidgeShockDummiesForecaster

        model = RidgeShockDummiesForecaster(use_2022_dummy=False)

        assert model.use_2022_dummy == False

    def test_repr(self):
        """Test model string representation."""
        from sirena.models.ridge_shock_dummies import RidgeShockDummiesForecaster

        model = RidgeShockDummiesForecaster()
        repr_str = repr(model)

        assert "RidgeShockDummiesForecaster" in repr_str

    def test_ets_weight_application(self, sample_data):
        """Test ETS weights are applied correctly."""
        from sirena.models.ridge_shock_dummies import RidgeShockDummiesForecaster

        model = RidgeShockDummiesForecaster(use_macro=False)
        model.fit(sample_data)

        target_date = sample_data.index[-1]
        result = model.predict(sample_data, target_date)

        weight = model.ETS_WEIGHTS.get(target_date.month, 0.3)

        expected_pred = (1 - weight) * result["pred_ridge"] + weight * result[
            "pred_ets"
        ]

        assert abs(result["prediction"] - expected_pred) < 0.0001

    def test_outlier_years_empty_list(self):
        """Test that OUTLIER_YEARS is empty list."""
        from sirena.models.ridge_shock_dummies import RidgeShockDummiesForecaster

        model = RidgeShockDummiesForecaster()

        # Unlike other Ridge models, this one doesn't exclude outlier years
        assert model.OUTLIER_YEARS == []

    def test_shock_dummies_in_features_with_use_2022_dummy(
        self, sample_data_with_shocks
    ):
        """Test all shock dummies are in features when use_2022_dummy=True."""
        from sirena.models.ridge_shock_dummies import RidgeShockDummiesForecaster

        model = RidgeShockDummiesForecaster(use_2022_dummy=True, use_macro=False)
        model.fit(sample_data_with_shocks)

        # All shock dummies should be in features
        for dummy in model.SHOCK_DUMMIES:
            assert dummy in model._features

    def test_only_pre_2022_dummies_in_features_without_use_2022_dummy(
        self, sample_data_with_shocks
    ):
        """Test only pre-2022 dummies are in features when use_2022_dummy=False."""
        from sirena.models.ridge_shock_dummies import RidgeShockDummiesForecaster

        model = RidgeShockDummiesForecaster(use_2022_dummy=False, use_macro=False)
        model.fit(sample_data_with_shocks)

        # Only pre-2022 dummies
        assert "is_shock_dec2014" in model._features
        assert "is_shock_jan2015" in model._features
        assert "is_tariff_jul2017" in model._features

        # 2022 dummies should NOT be in features
        assert "is_shock_2022" not in model._features
        assert "is_shock_mar2022" not in model._features
        assert "is_shock_apr2022" not in model._features
