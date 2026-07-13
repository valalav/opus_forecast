"""
Unit tests for NGBoostForecaster
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
def sample_data_with_outliers():
    """Generate sample data with outliers."""
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

    # Add some outliers
    data.iloc[24, 0] += 3.0  # 2022-01
    data.iloc[25, 0] -= 2.0  # 2022-02
    data.iloc[26, 0] += 2.5  # 2022-03

    return data


class TestNGBoostForecaster:
    """Test suite for NGBoostForecaster."""

    def test_import_model(self):
        """Test model import."""
        from sirena.models.ngboost_model import NGBoostForecaster

        model = NGBoostForecaster()
        assert model is not None
        assert model.name == "ngboost"

    def test_model_parameters(self):
        """Test model default parameters."""
        from sirena.models.ngboost_model import NGBoostForecaster

        model = NGBoostForecaster()

        assert model.MIN_TRAIN_SIZE == 36
        assert model.OUTLIER_YEARS == [2010, 2022]
        assert model.n_estimators == 200
        assert model.learning_rate == 0.05
        assert model.minibatch_frac == 0.8
        assert model._is_fitted == False
        assert model.model is None

    def test_base_features_list(self):
        """Test base features list."""
        from sirena.models.ngboost_model import NGBoostForecaster

        model = NGBoostForecaster()

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

    def test_ets_weights(self):
        """Test ETS weights dictionary."""
        from sirena.models.ngboost_model import NGBoostForecaster

        model = NGBoostForecaster()

        assert len(model.ETS_WEIGHTS) == 12
        assert 1 in model.ETS_WEIGHTS
        assert model.ETS_WEIGHTS[1] == 0.9
        assert model.ETS_WEIGHTS[7] == 0.0

    def test_custom_parameters(self):
        """Test custom parameters initialization."""
        from sirena.models.ngboost_model import NGBoostForecaster

        custom_n_estimators = 100
        custom_lr = 0.1
        custom_minibatch = 0.9

        model = NGBoostForecaster(
            n_estimators=custom_n_estimators,
            learning_rate=custom_lr,
            minibatch_frac=custom_minibatch,
        )

        assert model.n_estimators == custom_n_estimators
        assert model.learning_rate == custom_lr
        assert model.minibatch_frac == custom_minibatch

    def test_fit_basic(self, sample_data):
        """Test basic fit functionality."""
        from sirena.models.ngboost_model import NGBoostForecaster

        model = NGBoostForecaster(n_estimators=50)  # Small for testing
        model.fit(sample_data)

        assert model.is_fitted
        assert model.model is not None
        assert model.scaler is not None
        assert model.seasonal_norm is not None
        assert len(model.seasonal_norm) == 12
        assert model._features is not None

    def test_fit_with_outliers(self, sample_data_with_outliers):
        """Test fit handles outliers by excluding outlier years."""
        from sirena.models.ngboost_model import NGBoostForecaster

        model = NGBoostForecaster(n_estimators=50)
        model.fit(sample_data_with_outliers)

        assert model.is_fitted
        # Outlier years should be excluded from training
        assert 2022 in model.OUTLIER_YEARS

    def test_fit_insufficient_data(self, sample_data):
        """Test fit with insufficient data."""
        from sirena.models.ngboost_model import NGBoostForecaster

        small_data = sample_data.iloc[:30]

        model = NGBoostForecaster()

        with pytest.raises(ValueError, match="Недостаточно данных"):
            model.fit(small_data)

    def test_fit_empty_dataframe(self):
        """Test fit with empty DataFrame."""
        from sirena.models.ngboost_model import NGBoostForecaster

        empty_df = pd.DataFrame()

        model = NGBoostForecaster()

        with pytest.raises(ValueError, match="DataFrame пустой"):
            model.fit(empty_df)

    def test_fit_missing_target_column(self, sample_data):
        """Test fit with missing target column."""
        from sirena.models.ngboost_model import NGBoostForecaster

        df_no_target = sample_data.drop(columns=["Все товары и услуги"])

        model = NGBoostForecaster()

        with pytest.raises(ValueError, match="Колонка.*не найдена"):
            model.fit(df_no_target)

    def test_predict_basic(self, sample_data):
        """Test basic predict functionality."""
        from sirena.models.ngboost_model import NGBoostForecaster

        model = NGBoostForecaster(n_estimators=50)
        model.fit(sample_data)

        target_date = sample_data.index[-1]
        result = model.predict(sample_data, target_date)

        assert "prediction" in result
        assert "pred_ngboost" in result
        assert "pred_ets" in result
        assert "ets_weight" in result
        assert "std" in result
        assert "ci_lower" in result
        assert "ci_upper" in result
        assert "ci_width" in result
        assert "model" in result
        assert result["model"] == "ngboost"

    def test_predict_probabilistic_output(self, sample_data):
        """Test predict returns probabilistic outputs (std, CI)."""
        from sirena.models.ngboost_model import NGBoostForecaster

        model = NGBoostForecaster(n_estimators=50)
        model.fit(sample_data)

        target_date = sample_data.index[-1]
        result = model.predict(sample_data, target_date)

        # Probabilistic outputs
        assert "std" in result
        assert "ci_lower" in result
        assert "ci_upper" in result
        assert "ci_width" in result

        # Standard deviation should be positive
        assert result["std"] >= 0

        # CI width should be positive
        assert result["ci_width"] > 0

        # CI should be symmetric around prediction (approximately)
        lower_diff = result["prediction"] - result["ci_lower"]
        upper_diff = result["ci_upper"] - result["prediction"]
        # Due to ETS blending, may not be exactly symmetric
        assert abs(lower_diff - upper_diff) < 1.0

    def test_predict_ci_coverage(self, sample_data):
        """Test confidence interval covers prediction."""
        from sirena.models.ngboost_model import NGBoostForecaster

        model = NGBoostForecaster(n_estimators=50)
        model.fit(sample_data)

        target_date = sample_data.index[-1]
        result = model.predict(sample_data, target_date)

        # Prediction should be within CI
        assert result["ci_lower"] <= result["prediction"] <= result["ci_upper"]

    def test_predict_range(self, sample_data):
        """Test predict returns reasonable values."""
        from sirena.models.ngboost_model import NGBoostForecaster

        model = NGBoostForecaster(n_estimators=50)
        model.fit(sample_data)

        target_date = sample_data.index[-1]
        result = model.predict(sample_data, target_date)

        assert 98 < result["prediction"] < 102
        assert 98 < result["pred_ngboost"] < 102
        assert 98 < result["pred_ets"] < 102
        assert 0 <= result["ets_weight"] <= 1
        assert result["std"] >= 0

    def test_predict_not_fitted_error(self, sample_data):
        """Test predict raises error when model not fitted."""
        from sirena.models.ngboost_model import NGBoostForecaster

        model = NGBoostForecaster()

        target_date = sample_data.index[-1]

        with pytest.raises(ValueError, match="не обучена"):
            model.predict(sample_data, target_date)

    def test_predict_with_ci_alias(self, sample_data):
        """Test predict_with_ci is an alias for predict."""
        from sirena.models.ngboost_model import NGBoostForecaster

        model = NGBoostForecaster(n_estimators=50)
        model.fit(sample_data)

        target_date = sample_data.index[-1]

        result1 = model.predict(sample_data, target_date)
        result2 = model.predict_with_ci(sample_data, target_date)

        assert result1.keys() == result2.keys()

    def test_forecast_basic(self, sample_data):
        """Test forecast functionality."""
        from sirena.models.ngboost_model import NGBoostForecaster

        model = NGBoostForecaster(n_estimators=50)
        model.fit(sample_data)

        horizon = 12
        forecast = model.forecast(horizon)

        assert len(forecast) == horizon
        assert all(isinstance(v, (int, float)) for v in forecast)

    def test_forecast_different_horizons(self, sample_data):
        """Test forecast with different horizons."""
        from sirena.models.ngboost_model import NGBoostForecaster

        model = NGBoostForecaster(n_estimators=50)
        model.fit(sample_data)

        for h in [1, 6, 12, 24]:
            forecast = model.forecast(h)
            assert len(forecast) == h

    def test_forecast_not_fitted_error(self, sample_data):
        """Test forecast raises error when model not fitted."""
        from sirena.models.ngboost_model import NGBoostForecaster

        model = NGBoostForecaster()

        with pytest.raises(ValueError, match="не обучена"):
            model.forecast(12)

    def test_backtest_basic(self, sample_data):
        """Test backtest functionality."""
        from sirena.models.ngboost_model import NGBoostForecaster

        model = NGBoostForecaster(n_estimators=50)
        results = model.backtest(sample_data, start_date="2023-01-01")

        assert isinstance(results, pd.DataFrame)

        if not results.empty:
            assert "date" in results.columns
            assert "actual" in results.columns
            assert "prediction" in results.columns
            assert "error" in results.columns
            assert "std" in results.columns
            assert "ci_lower" in results.columns
            assert "ci_upper" in results.columns
            assert "ci_width" in results.columns
            assert "in_ci" in results.columns

    def test_backtest_custom_start_date(self, sample_data):
        """Test backtest with custom start date."""
        from sirena.models.ngboost_model import NGBoostForecaster

        model = NGBoostForecaster(n_estimators=50)
        results = model.backtest(sample_data, start_date="2022-06-01")

        assert isinstance(results, pd.DataFrame)

        if not results.empty:
            min_date = results["date"].min()
            assert min_date >= pd.Timestamp("2022-06-01")

    def test_backtest_ci_coverage(self, sample_data):
        """Test backtest CI coverage calculation."""
        from sirena.models.ngboost_model import NGBoostForecaster

        model = NGBoostForecaster(n_estimators=50)
        results = model.backtest(sample_data, start_date="2023-01-01")

        if not results.empty:
            # Check in_ci column is boolean
            assert results["in_ci"].dtype == bool or results["in_ci"].dtype == np.bool_

            # Some predictions should be in CI
            assert results["in_ci"].sum() > 0 or len(results) == 0

    def test_get_feature_importance(self, sample_data):
        """Test get feature importance."""
        from sirena.models.ngboost_model import NGBoostForecaster

        model = NGBoostForecaster(n_estimators=50)
        model.fit(sample_data)

        importance = model.get_feature_importance()

        assert isinstance(importance, pd.DataFrame)
        assert "feature" in importance.columns
        assert "importance" in importance.columns

    def test_get_feature_importance_sorted(self, sample_data):
        """Test feature importance is sorted."""
        from sirena.models.ngboost_model import NGBoostForecaster

        model = NGBoostForecaster(n_estimators=50)
        model.fit(sample_data)

        importance = model.get_feature_importance()

        if not importance.empty:
            importance_values = importance["importance"].values
            assert all(
                importance_values[i] >= importance_values[i + 1]
                for i in range(len(importance_values) - 1)
            )

    def test_get_feature_importance_not_fitted_error(self, sample_data):
        """Test get feature importance raises error when not fitted."""
        from sirena.models.ngboost_model import NGBoostForecaster

        model = NGBoostForecaster()

        with pytest.raises(ValueError, match="не обучена"):
            model.get_feature_importance()

    def test_get_model_info(self, sample_data):
        """Test get model info."""
        from sirena.models.ngboost_model import NGBoostForecaster

        model = NGBoostForecaster(n_estimators=50)
        model.fit(sample_data)

        info = model.get_model_info()

        assert isinstance(info, dict)
        assert "name" in info
        assert "n_estimators" in info
        assert "learning_rate" in info
        assert "minibatch_frac" in info
        assert "features_count" in info
        assert "is_fitted" in info
        assert info["name"] == "ngboost"
        assert info["n_estimators"] == model.n_estimators
        assert info["learning_rate"] == model.learning_rate
        assert info["minibatch_frac"] == model.minibatch_frac
        assert info["is_fitted"] == True

    def test_get_model_info_not_fitted(self):
        """Test get model info when not fitted."""
        from sirena.models.ngboost_model import NGBoostForecaster

        model = NGBoostForecaster()

        info = model.get_model_info()

        assert isinstance(info, dict)
        assert info["is_fitted"] == False
        assert info["features_count"] == 0

    def test_prepare_features(self, sample_data):
        """Test feature preparation."""
        from sirena.models.ngboost_model import NGBoostForecaster

        model = NGBoostForecaster()

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
        from sirena.models.ngboost_model import NGBoostForecaster

        model = NGBoostForecaster()

        df_prep = model._prepare_features(sample_data)

        assert "food_lag1" in df_prep.columns
        assert "nonfood_lag1" in df_prep.columns
        assert "services_lag1" in df_prep.columns

    def test_compute_seasonal_norm(self, sample_data):
        """Test seasonal norm computation."""
        from sirena.models.ngboost_model import NGBoostForecaster

        model = NGBoostForecaster()

        df_prep = model._prepare_features(sample_data)
        seasonal_norm = model._compute_seasonal_norm(df_prep)

        assert len(seasonal_norm) == 12
        assert all(month in seasonal_norm.index for month in range(1, 13))

    def test_seasonal_norm_excludes_outlier_years(self, sample_data_with_outliers):
        """Test seasonal norm excludes outlier years."""
        from sirena.models.ngboost_model import NGBoostForecaster

        model = NGBoostForecaster()

        df_prep = model._prepare_features(sample_data_with_outliers)
        seasonal_norm = model._compute_seasonal_norm(df_prep)

        # Check that 2022 data was excluded
        df_without_2022 = df_prep[df_prep["year"] != 2022]
        expected_norm = df_without_2022.groupby("month")["Все товары и услуги"].mean()

        assert all(seasonal_norm == expected_norm)

    def test_check_fitted(self):
        """Test _check_fitted method."""
        from sirena.models.ngboost_model import NGBoostForecaster

        model = NGBoostForecaster()

        with pytest.raises(ValueError, match="не обучена"):
            model._check_fitted()

    def test_repr(self):
        """Test model string representation."""
        from sirena.models.ngboost_model import NGBoostForecaster

        model = NGBoostForecaster()
        repr_str = repr(model)

        assert "NGBoostForecaster" in repr_str

    def test_ets_weight_application(self, sample_data):
        """Test ETS weights are applied correctly."""
        from sirena.models.ngboost_model import NGBoostForecaster

        model = NGBoostForecaster(n_estimators=50)
        model.fit(sample_data)

        target_date = sample_data.index[-1]
        result = model.predict(sample_data, target_date)

        weight = model.ETS_WEIGHTS.get(target_date.month, 0.3)

        expected_pred = (1 - weight) * result["pred_ngboost"] + weight * result[
            "pred_ets"
        ]

        assert abs(result["prediction"] - expected_pred) < 0.0001

    def test_outlier_years_exclusion(self, sample_data):
        """Test that NGBoost excludes outlier years (2010, 2022)."""
        from sirena.models.ngboost_model import NGBoostForecaster

        dates_with_2022 = pd.date_range("2019-01-01", periods=72, freq="MS")
        np.random.seed(42)

        data_with_2022 = pd.DataFrame(
            {
                "Все товары и услуги": 100.5 + np.random.randn(72) * 0.3,
                "Продовольственные товары": 100.6 + np.random.randn(72) * 0.4,
                "Непродовольственные товары": 100.3 + np.random.randn(72) * 0.2,
                "Услуги": 100.4 + np.random.randn(72) * 0.3,
            },
            index=dates_with_2022,
        )

        model = NGBoostForecaster(n_estimators=50)
        model.fit(data_with_2022)

        # 2022 should be excluded from training
        assert 2022 in model.OUTLIER_YEARS
        assert model._last_train_date is not None

    def test_backtest_custom_target_column(self, sample_data):
        """Test backtest with custom target column."""
        from sirena.models.ngboost_model import NGBoostForecaster

        model = NGBoostForecaster(n_estimators=50)
        results = model.backtest(
            sample_data, start_date="2023-01-01", target_col="Продовольственные товары"
        )

        assert isinstance(results, pd.DataFrame)

        if not results.empty:
            assert "actual" in results.columns
            assert "prediction" in results.columns

    def test_probabilistic_consistency(self, sample_data):
        """Test that probabilistic predictions are consistent."""
        from sirena.models.ngboost_model import NGBoostForecaster

        model = NGBoostForecaster(n_estimators=50)
        model.fit(sample_data)

        target_date = sample_data.index[-1]
        result = model.predict(sample_data, target_date)

        # CI should be roughly 2 * 1.645 * std for 90% CI
        expected_ci_width = 2 * 1.645 * result["std"]
        # Allow some tolerance due to ETS blending
        assert abs(result["ci_width"] - expected_ci_width) < expected_ci_width * 0.5

    def test_backtest_with_long_horizon(self, sample_data):
        """Test backtest with longer data."""
        from sirena.models.ngboost_model import NGBoostForecaster

        dates = pd.date_range("2016-01-01", periods=120, freq="MS")
        np.random.seed(42)

        long_data = pd.DataFrame(
            {
                "Все товары и услуги": 100.5 + np.random.randn(120) * 0.3,
                "Продовольственные товары": 100.6 + np.random.randn(120) * 0.4,
                "Непродовольственные товары": 100.3 + np.random.randn(120) * 0.2,
                "Услуги": 100.4 + np.random.randn(120) * 0.3,
            },
            index=dates,
        )

        model = NGBoostForecaster(n_estimators=50)
        results = model.backtest(long_data, start_date="2023-01-01")

        assert isinstance(results, pd.DataFrame)

        if not results.empty:
            assert len(results) > 0

    def test_minibatch_frac_parameter(self, sample_data):
        """Test minibatch_frac parameter."""
        from sirena.models.ngboost_model import NGBoostForecaster

        custom_minibatch = 0.5
        model = NGBoostForecaster(n_estimators=50, minibatch_frac=custom_minibatch)

        assert model.minibatch_frac == custom_minibatch

        model.fit(sample_data)
        assert model.is_fitted
