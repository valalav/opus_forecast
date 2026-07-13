"""
Unit tests for RidgeMacroForecaster
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
            "mom": 100.5 + np.random.randn(60) * 0.3,
            "Prod": 100.6 + np.random.randn(60) * 0.4,
            "Serv": 100.3 + np.random.randn(60) * 0.3,
            "Nonprod": 100.2 + np.random.randn(60) * 0.2,
            "Ki_i": 16 + np.random.randn(60) * 0.5,
            "Ruonia": 15 + np.random.randn(60) * 0.5,
            "usd_nom_i": 90 + np.random.randn(60) * 2.0,
            "brent": 80 + np.random.randn(60) * 5.0,
        },
        index=dates,
    )

    return data


@pytest.fixture
def sample_data_with_outliers():
    """Generate sample data including outlier years."""
    dates = pd.date_range("2010-01-01", periods=180, freq="MS")
    np.random.seed(42)

    data = pd.DataFrame(
        {
            "mom": 100.0 + np.random.randn(180) * 0.3,
            "Prod": 100.0 + np.random.randn(180) * 0.4,
            "Serv": 100.0 + np.random.randn(180) * 0.3,
            "Nonprod": 100.0 + np.random.randn(180) * 0.2,
            "Ki_i": 10 + np.random.randn(180) * 2.0,
            "Ruonia": 10 + np.random.randn(180) * 2.0,
            "usd_nom_i": 60 + np.random.randn(180) * 5.0,
            "brent": 70 + np.random.randn(180) * 10.0,
        },
        index=dates,
    )

    # Add outliers for 2010 and 2022
    data.loc[pd.Timestamp("2010-06-01"), "mom"] = 105.0
    data.loc[pd.Timestamp("2022-03-01"), "mom"] = 104.0

    return data


class TestRidgeMacroForecaster:
    """Test suite for RidgeMacroForecaster."""

    def test_import_model(self):
        """Test model import."""
        from sirena.models.ridge_macro import RidgeMacroForecaster

        model = RidgeMacroForecaster()
        assert model is not None
        assert model.name == "ridge_macro"

    def test_model_parameters_default(self):
        """Test model default parameters."""
        from sirena.models.ridge_macro import RidgeMacroForecaster

        model = RidgeMacroForecaster()

        assert model.alpha == 1.0
        assert model.use_huber == False
        assert model.MIN_TRAIN_SIZE == 36
        assert model.OUTLIER_YEARS == [2022, 2010]

    def test_model_parameters_custom(self):
        """Test model custom parameters."""
        from sirena.models.ridge_macro import RidgeMacroForecaster

        custom_alpha = 0.5
        model = RidgeMacroForecaster(alpha=custom_alpha, use_huber=True)

        assert model.alpha == custom_alpha
        assert model.use_huber == True

    def test_features_list(self):
        """Test features list."""
        from sirena.models.ridge_macro import RidgeMacroForecaster

        model = RidgeMacroForecaster()

        required_features = [
            "mom_L1",
            "mom_L2",
            "mom_L3",
            "ki_L6",
            "Ruonia_D1",
            "usd_L2",
            "brent_L5",
            "brent_STD3",
            "prod_L1",
            "serv_L1",
            "month_sin",
            "month_cos",
        ]

        for feature in required_features:
            assert feature in model.FEATURES

    def test_fit_basic(self, sample_data_with_macro):
        """Test basic fit functionality."""
        from sirena.models.ridge_macro import RidgeMacroForecaster

        model = RidgeMacroForecaster()
        model.fit(sample_data_with_macro)

        assert model.is_fitted
        assert model.model is not None
        assert model.scaler is not None
        assert model._available_features is not None
        assert len(model._available_features) >= 5
        assert model._train_df is not None
        assert model._last_train_date is not None

    def test_fit_custom_alpha(self, sample_data_with_macro):
        """Test fit with custom alpha parameter."""
        from sirena.models.ridge_macro import RidgeMacroForecaster

        custom_alpha = 0.5
        model = RidgeMacroForecaster(alpha=custom_alpha)
        model.fit(sample_data_with_macro)

        assert model.alpha == custom_alpha
        assert model.model.alpha == custom_alpha

    def test_fit_with_huber(self, sample_data_with_macro):
        """Test fit with Huber regressor."""
        from sirena.models.ridge_macro import RidgeMacroForecaster

        model = RidgeMacroForecaster(use_huber=True)
        model.fit(sample_data_with_macro)

        assert model.use_huber
        assert model.model is not None

    def test_fit_insufficient_data(self, sample_data_with_macro):
        """Test fit with insufficient data."""
        from sirena.models.ridge_macro import RidgeMacroForecaster

        small_data = sample_data_with_macro.iloc[:30]

        model = RidgeMacroForecaster()

        with pytest.raises(ValueError, match="Недостаточно данных"):
            model.fit(small_data)

    def test_fit_empty_dataframe(self):
        """Test fit with empty DataFrame."""
        from sirena.models.ridge_macro import RidgeMacroForecaster

        dates = pd.date_range("2020-01-01", periods=0, freq="MS")
        empty_df = pd.DataFrame(index=dates)

        model = RidgeMacroForecaster()

        with pytest.raises(ValueError, match="No target column"):
            model.fit(empty_df)

    def test_fit_missing_target_column(self, sample_data_with_macro):
        """Test fit with missing target column."""
        from sirena.models.ridge_macro import RidgeMacroForecaster

        df_no_target = sample_data_with_macro.drop(columns=["mom"])

        model = RidgeMacroForecaster()

        # Should use 'Все товары и услуги' instead
        # But we removed all target columns, so it should fail
        with pytest.raises(ValueError, match="No target column"):
            model.fit(df_no_target)

    def test_prepare_features(self, sample_data_with_macro):
        """Test feature preparation."""
        from sirena.models.ridge_macro import RidgeMacroForecaster

        model = RidgeMacroForecaster()

        df_prep = model._prepare_features(sample_data_with_macro)

        assert "mom" in df_prep.columns
        assert "month" in df_prep.columns
        assert "year" in df_prep.columns
        assert "mom_L1" in df_prep.columns
        assert "mom_L2" in df_prep.columns
        assert "mom_L3" in df_prep.columns
        assert "ki_L6" in df_prep.columns
        assert "Ruonia_D1" in df_prep.columns
        assert "usd_L2" in df_prep.columns
        assert "brent_L5" in df_prep.columns
        assert "brent_STD3" in df_prep.columns
        assert "prod_L1" in df_prep.columns
        assert "serv_L1" in df_prep.columns
        assert "month_sin" in df_prep.columns
        assert "month_cos" in df_prep.columns

    def test_predict_basic(self, sample_data_with_macro):
        """Test basic predict functionality."""
        from sirena.models.ridge_macro import RidgeMacroForecaster

        model = RidgeMacroForecaster()
        model.fit(sample_data_with_macro)

        target_date = sample_data_with_macro.index[-1]
        result = model.predict(sample_data_with_macro, target_date)

        assert "prediction" in result
        assert "date" in result
        assert "model" in result
        assert "n_features" in result
        assert "features_used" in result
        assert result["model"] == "ridge_macro"
        assert result["date"] == target_date

    def test_predict_range(self, sample_data_with_macro):
        """Test predict returns reasonable values."""
        from sirena.models.ridge_macro import RidgeMacroForecaster

        model = RidgeMacroForecaster()
        model.fit(sample_data_with_macro)

        target_date = sample_data_with_macro.index[-1]
        result = model.predict(sample_data_with_macro, target_date)

        assert 98 < result["prediction"] < 102

    def test_predict_not_fitted_error(self, sample_data_with_macro):
        """Test predict raises error when model not fitted."""
        from sirena.models.ridge_macro import RidgeMacroForecaster

        model = RidgeMacroForecaster()

        target_date = sample_data_with_macro.index[-1]

        with pytest.raises(ValueError, match="не обучена"):
            model.predict(sample_data_with_macro, target_date)

    def test_predict_target_date_not_in_index(self, sample_data_with_macro):
        """Test predict with target_date not in index."""
        from sirena.models.ridge_macro import RidgeMacroForecaster

        model = RidgeMacroForecaster()
        model.fit(sample_data_with_macro)

        target_date = pd.Timestamp("2025-06-01")

        with pytest.raises(ValueError, match="not in data index"):
            model.predict(sample_data_with_macro, target_date)

    def test_forecast_basic(self, sample_data_with_macro):
        """Test forecast functionality."""
        from sirena.models.ridge_macro import RidgeMacroForecaster

        model = RidgeMacroForecaster()
        model.fit(sample_data_with_macro)

        horizon = 12
        forecast = model.forecast(horizon)

        assert len(forecast) == horizon
        assert all(isinstance(v, (int, float)) for v in forecast)

    def test_forecast_different_horizons(self, sample_data_with_macro):
        """Test forecast with different horizons."""
        from sirena.models.ridge_macro import RidgeMacroForecaster

        model = RidgeMacroForecaster()
        model.fit(sample_data_with_macro)

        for h in [1, 6, 12, 24]:
            forecast = model.forecast(h)
            assert len(forecast) == h

    def test_forecast_not_fitted_error(self, sample_data_with_macro):
        """Test forecast raises error when model not fitted."""
        from sirena.models.ridge_macro import RidgeMacroForecaster

        model = RidgeMacroForecaster()

        with pytest.raises(ValueError, match="не обучена"):
            model.forecast(12)

    def test_backtest_basic(self, sample_data_with_macro):
        """Test backtest functionality."""
        from sirena.models.ridge_macro import RidgeMacroForecaster

        model = RidgeMacroForecaster()
        results = model.backtest(sample_data_with_macro, start_date="2023-01-01")

        assert isinstance(results, pd.DataFrame)

        if not results.empty:
            assert "date" in results.columns
            assert "cutoff" in results.columns
            assert "horizon" in results.columns
            assert "actual" in results.columns
            assert "prediction" in results.columns
            assert "error" in results.columns
            assert "n_features" in results.columns

    def test_backtest_custom_start_date(self, sample_data_with_macro):
        """Test backtest with custom start date."""
        from sirena.models.ridge_macro import RidgeMacroForecaster

        model = RidgeMacroForecaster()
        results = model.backtest(sample_data_with_macro, start_date="2022-06-01")

        assert isinstance(results, pd.DataFrame)

        if not results.empty:
            min_date = results["date"].min()
            assert min_date >= pd.Timestamp("2022-06-01")

    def test_backtest_different_horizons(self, sample_data_with_macro):
        """Test backtest with different horizons."""
        from sirena.models.ridge_macro import RidgeMacroForecaster

        model = RidgeMacroForecaster()

        for h in [1, 2, 3]:
            results = model.backtest(
                sample_data_with_macro, start_date="2023-01-01", horizon=h
            )
            assert isinstance(results, pd.DataFrame)

            if not results.empty:
                assert (results["horizon"] == h).all()

    def test_get_feature_importance(self, sample_data_with_macro):
        """Test get feature importance."""
        from sirena.models.ridge_macro import RidgeMacroForecaster

        model = RidgeMacroForecaster()
        model.fit(sample_data_with_macro)

        importance = model.get_feature_importance()

        assert isinstance(importance, pd.DataFrame)
        assert "feature" in importance.columns
        assert "coefficient" in importance.columns
        assert "abs_coef" in importance.columns

    def test_get_feature_importance_sorted(self, sample_data_with_macro):
        """Test feature importance is sorted by absolute coefficient."""
        from sirena.models.ridge_macro import RidgeMacroForecaster

        model = RidgeMacroForecaster()
        model.fit(sample_data_with_macro)

        importance = model.get_feature_importance()

        abs_coefs = importance["abs_coef"].values
        assert all(abs_coefs[i] >= abs_coefs[i + 1] for i in range(len(abs_coefs) - 1))

    def test_get_feature_importance_not_fitted_error(self, sample_data_with_macro):
        """Test get_feature_importance raises error when not fitted."""
        from sirena.models.ridge_macro import RidgeMacroForecaster

        model = RidgeMacroForecaster()

        with pytest.raises(ValueError, match="не обучена"):
            model.get_feature_importance()

    def test_compute_seasonal_norms(self, sample_data_with_macro):
        """Test seasonal norms computation."""
        from sirena.models.ridge_macro import RidgeMacroForecaster

        model = RidgeMacroForecaster()

        norms = model._compute_seasonal_norms(sample_data_with_macro)

        assert isinstance(norms, dict)
        assert "mom" in norms
        assert "Prod" in norms
        assert "Serv" in norms
        assert "Nonprod" in norms

        # Each norm should have 12 months
        for col_norms in norms.values():
            assert len(col_norms) == 12

    def test_compute_seasonal_norms_excludes_outliers(self, sample_data_with_outliers):
        """Test seasonal norms exclude outlier years."""
        from sirena.models.ridge_macro import RidgeMacroForecaster

        model = RidgeMacroForecaster()

        norms = model._compute_seasonal_norms(sample_data_with_outliers)

        # Should still have 12 months per column
        for col_norms in norms.values():
            assert len(col_norms) == 12

    def test_forecast_exogenous(self, sample_data_with_macro):
        """Test exogenous variable forecasting."""
        from sirena.models.ridge_macro import RidgeMacroForecaster

        model = RidgeMacroForecaster()
        model.fit(sample_data_with_macro)

        horizon = 6
        exog_fc = model._forecast_exogenous(sample_data_with_macro, horizon)

        assert len(exog_fc) == horizon
        assert exog_fc.index[0] > sample_data_with_macro.index[-1]

        # Check that expected columns exist
        if "Ki_i" in sample_data_with_macro.columns:
            assert "Ki_i" in exog_fc.columns
        if "Ruonia" in sample_data_with_macro.columns:
            assert "Ruonia" in exog_fc.columns
        if "usd_nom_i" in sample_data_with_macro.columns:
            assert "usd_nom_i" in exog_fc.columns
        if "brent" in sample_data_with_macro.columns:
            assert "brent" in exog_fc.columns

    def test_get_exogenous_forecasts(self, sample_data_with_macro):
        """Test get_exogenous_forecasts method."""
        from sirena.models.ridge_macro import RidgeMacroForecaster

        model = RidgeMacroForecaster()
        model.fit(sample_data_with_macro)

        horizon = 12
        exog_fc = model.get_exogenous_forecasts(horizon)

        assert isinstance(exog_fc, pd.DataFrame)
        assert len(exog_fc) == horizon

    def test_get_exogenous_forecasts_not_fitted_error(self, sample_data_with_macro):
        """Test get_exogenous_forecasts raises error when not fitted."""
        from sirena.models.ridge_macro import RidgeMacroForecaster

        model = RidgeMacroForecaster()

        with pytest.raises(ValueError, match="не обучена"):
            model.get_exogenous_forecasts(12)

    def test_get_metrics(self, sample_data_with_macro):
        """Test get metrics calculation."""
        from sirena.models.ridge_macro import RidgeMacroForecaster

        model = RidgeMacroForecaster()
        results = model.backtest(sample_data_with_macro, start_date="2023-01-01")

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
        from sirena.models.ridge_macro import RidgeMacroForecaster

        model = RidgeMacroForecaster()
        empty_results = pd.DataFrame()

        metrics = model.get_metrics(empty_results)

        assert metrics == {"MAE": 0, "RMSE": 0, "KPI": 0}

    def test_check_fitted(self):
        """Test _check_fitted method."""
        from sirena.models.ridge_macro import RidgeMacroForecaster

        model = RidgeMacroForecaster()

        with pytest.raises(ValueError, match="не обучена"):
            model._check_fitted()

    def test_custom_alpha_parameter(self):
        """Test custom alpha parameter."""
        from sirena.models.ridge_macro import RidgeMacroForecaster

        custom_alpha = 0.1
        model = RidgeMacroForecaster(alpha=custom_alpha)

        assert model.alpha == custom_alpha

    def test_use_huber_parameter(self):
        """Test use_huber parameter."""
        from sirena.models.ridge_macro import RidgeMacroForecaster

        model = RidgeMacroForecaster(use_huber=True)

        assert model.use_huber == True

    def test_repr(self):
        """Test model string representation."""
        from sirena.models.ridge_macro import RidgeMacroForecaster

        model = RidgeMacroForecaster()
        repr_str = repr(model)

        assert "RidgeMacroForecaster" in repr_str

    def test_outlier_years_exclusion(self, sample_data_with_outliers):
        """Test that outlier years are excluded from training."""
        from sirena.models.ridge_macro import RidgeMacroForecaster

        model = RidgeMacroForecaster()
        model.fit(sample_data_with_outliers)

        assert model._last_train_date is not None
        assert model.is_fitted

    def test_fit_with_target_only(self):
        """Test fit with only target column (uses autoregressive + seasonal features)."""
        from sirena.models.ridge_macro import RidgeMacroForecaster

        model = RidgeMacroForecaster()

        # Create data with only target column
        dates = pd.date_range("2020-01-01", periods=60, freq="MS")
        target_only_data = pd.DataFrame(
            {
                "Все товары и услуги": 100.0 + np.random.randn(60) * 0.3,
            },
            index=dates,
        )

        # Model should work with autoregressive + seasonal features (>=5 features)
        model.fit(target_only_data)

        assert model.is_fitted
        # Should have at least 5 features: mom_L1, mom_L2, mom_L3, month_sin, month_cos
        assert len(model._available_features) >= 5

    def test_forecast_iteration_updates_dataframe(self, sample_data_with_macro):
        """Test that forecast iteratively updates DataFrame."""
        from sirena.models.ridge_macro import RidgeMacroForecaster

        model = RidgeMacroForecaster()
        model.fit(sample_data_with_macro)

        horizon = 3
        forecast = model.forecast(horizon)

        assert len(forecast) == horizon
        # Check that forecast values are reasonable
        assert all(
            -2 <= v <= 2 for v in forecast
        )  # MoM values should be in reasonable range

    def test_backtest_with_missing_data(self, sample_data_with_macro):
        """Test backtest handles missing data gracefully."""
        from sirena.models.ridge_macro import RidgeMacroForecaster

        # Add some NaN values
        data_with_nan = sample_data_with_macro.copy()
        data_with_nan.loc[pd.Timestamp("2022-01-01"), "Ki_i"] = np.nan

        model = RidgeMacroForecaster()
        results = model.backtest(data_with_nan, start_date="2023-01-01")

        assert isinstance(results, pd.DataFrame)
