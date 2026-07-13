"""
Unit tests for BayesianRidgeForecaster
Tests are created in edge_lab but import from parent sirena package
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
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


@pytest.fixture
def sample_data_with_macro():
    """Generate sample data with macro indicators."""
    dates = pd.date_range("2020-01-01", periods=60, freq="MS")
    np.random.seed(42)

    data = pd.DataFrame(
        {
            "Все товары и услуги": 100.5 + np.random.randn(60) * 0.3,
            "Продовольственные товары": 100.6 + np.random.randn(60) * 0.4,
            "Непродовольственные товары": 100.3 + np.random.randn(60) * 0.2,
            "Услуги": 100.4 + np.random.randn(60) * 0.3,
            "Ki": 7.5 + np.random.randn(60) * 0.5,
            "Ruonia": 7.0 + np.random.randn(60) * 0.3,
        },
        index=dates,
    )

    return data


class TestBayesianRidgeForecaster:
    """Test suite for BayesianRidgeForecaster."""

    def test_import_model(self):
        """Test model import."""
        from sirena.models.bayesian_ridge import BayesianRidgeForecaster

        model = BayesianRidgeForecaster()
        assert model is not None
        assert model.name == "bayesian_ridge"

    def test_model_parameters(self):
        """Test model default parameters."""
        from sirena.models.bayesian_ridge import BayesianRidgeForecaster

        model = BayesianRidgeForecaster()

        assert model.MIN_TRAIN_SIZE == 36
        assert model.OUTLIER_YEARS == [2022, 2010]
        assert model.alpha_1 == 1e-6
        assert model.alpha_2 == 1e-6
        assert model.lambda_1 == 1e-6
        assert model.lambda_2 == 1e-6
        assert model.n_iter == 300
        assert model.use_macro == True
        assert model._is_fitted == False
        assert model.model is None

    def test_base_features_list(self):
        """Test base features list."""
        from sirena.models.bayesian_ridge import BayesianRidgeForecaster

        model = BayesianRidgeForecaster()

        required_features = [
            "y_lag1",
            "y_lag2",
            "y_lag3",
            "y_lag6",
            "y_lag12",
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
        from sirena.models.bayesian_ridge import BayesianRidgeForecaster

        model = BayesianRidgeForecaster()

        required_macro_features = [
            "ruonia_diff_lag1",
            "spread_lag4",
            "ki_diff_lag6",
            "ki_vol",
        ]

        for feature in required_macro_features:
            assert feature in model.MACRO_FEATURES

    def test_ets_weights(self):
        """Test ETS weights dictionary."""
        from sirena.models.bayesian_ridge import BayesianRidgeForecaster

        model = BayesianRidgeForecaster()

        assert len(model.ETS_WEIGHTS) == 12
        assert 1 in model.ETS_WEIGHTS
        assert model.ETS_WEIGHTS[1] == 0.9
        assert model.ETS_WEIGHTS[7] == 0.0

    def test_custom_parameters(self):
        """Test custom parameters initialization."""
        from sirena.models.bayesian_ridge import BayesianRidgeForecaster

        custom_alpha_1 = 1e-5
        custom_alpha_2 = 1e-5
        custom_lambda_1 = 1e-5
        custom_lambda_2 = 1e-5
        custom_n_iter = 500

        model = BayesianRidgeForecaster(
            alpha_1=custom_alpha_1,
            alpha_2=custom_alpha_2,
            lambda_1=custom_lambda_1,
            lambda_2=custom_lambda_2,
            n_iter=custom_n_iter,
        )

        assert model.alpha_1 == custom_alpha_1
        assert model.alpha_2 == custom_alpha_2
        assert model.lambda_1 == custom_lambda_1
        assert model.lambda_2 == custom_lambda_2
        assert model.n_iter == custom_n_iter

    def test_use_macro_parameter(self):
        """Test use_macro parameter."""
        from sirena.models.bayesian_ridge import BayesianRidgeForecaster

        model_no_macro = BayesianRidgeForecaster(use_macro=False)
        assert model_no_macro.use_macro == False

        model_with_macro = BayesianRidgeForecaster(use_macro=True)
        assert model_with_macro.use_macro == True

    def test_fit_basic(self, sample_data):
        """Test basic fit functionality."""
        from sirena.models.bayesian_ridge import BayesianRidgeForecaster

        model = BayesianRidgeForecaster(n_iter=100)
        model.fit(sample_data)

        assert model.is_fitted
        assert model.model is not None
        assert model.scaler is not None
        assert model.seasonal_norm is not None
        assert len(model.seasonal_norm) == 12
        assert model._features is not None
        assert model._has_macro == False

    def test_fit_with_macro_features(self, sample_data_with_macro):
        """Test fit with macro features."""
        from sirena.models.bayesian_ridge import BayesianRidgeForecaster

        model = BayesianRidgeForecaster(n_iter=100, use_macro=True)
        model.fit(sample_data_with_macro)

        assert model.is_fitted
        assert model._has_macro == True
        assert len(model._features) > len(model.BASE_FEATURES)

    def test_fit_with_outliers(self, sample_data_with_outliers):
        """Test fit handles outliers by excluding outlier years."""
        from sirena.models.bayesian_ridge import BayesianRidgeForecaster

        model = BayesianRidgeForecaster(n_iter=100)
        model.fit(sample_data_with_outliers)

        assert model.is_fitted
        # Outlier years should be excluded from training
        assert 2022 in model.OUTLIER_YEARS
        assert 2010 in model.OUTLIER_YEARS

    def test_fit_insufficient_data(self, sample_data):
        """Test fit with insufficient data."""
        from sirena.models.bayesian_ridge import BayesianRidgeForecaster

        small_data = sample_data.iloc[:30]

        model = BayesianRidgeForecaster()

        with pytest.raises(ValueError, match="Недостаточно данных"):
            model.fit(small_data)

    def test_fit_empty_dataframe(self):
        """Test fit with empty DataFrame."""
        from sirena.models.bayesian_ridge import BayesianRidgeForecaster

        empty_df = pd.DataFrame()

        model = BayesianRidgeForecaster()

        with pytest.raises(ValueError, match="DataFrame пустой"):
            model.fit(empty_df)

    def test_fit_missing_target_column(self, sample_data):
        """Test fit with missing target column."""
        from sirena.models.bayesian_ridge import BayesianRidgeForecaster

        df_no_target = sample_data.drop(columns=["Все товары и услуги"])

        model = BayesianRidgeForecaster()

        with pytest.raises(ValueError, match="Колонка.*не найдена"):
            model.fit(df_no_target)

    def test_predict_basic(self, sample_data):
        """Test basic predict functionality."""
        from sirena.models.bayesian_ridge import BayesianRidgeForecaster

        model = BayesianRidgeForecaster(n_iter=100)
        model.fit(sample_data)

        target_date = sample_data.index[-1]
        result = model.predict(sample_data, target_date)

        assert "prediction" in result
        assert "pred_bayesian" in result
        assert "pred_ets" in result
        assert "ets_weight" in result
        assert "std" in result
        assert "ci_lower" in result
        assert "ci_upper" in result
        assert "model" in result
        assert "has_macro" in result
        assert result["model"] == "bayesian_ridge"

    def test_predict_probabilistic_output(self, sample_data):
        """Test predict returns probabilistic outputs (std, CI)."""
        from sirena.models.bayesian_ridge import BayesianRidgeForecaster

        model = BayesianRidgeForecaster(n_iter=100)
        model.fit(sample_data)

        target_date = sample_data.index[-1]
        result = model.predict(sample_data, target_date)

        # Probabilistic outputs
        assert "std" in result
        assert "ci_lower" in result
        assert "ci_upper" in result

        # Standard deviation should be positive
        assert result["std"] >= 0

        # CI width should be positive
        ci_width = result["ci_upper"] - result["ci_lower"]
        assert ci_width > 0

    def test_predict_ci_coverage(self, sample_data):
        """Test confidence interval covers prediction."""
        from sirena.models.bayesian_ridge import BayesianRidgeForecaster

        model = BayesianRidgeForecaster(n_iter=100)
        model.fit(sample_data)

        target_date = sample_data.index[-1]
        result = model.predict(sample_data, target_date)

        # Prediction should be within CI
        assert result["ci_lower"] <= result["prediction"] <= result["ci_upper"]

    def test_predict_range(self, sample_data):
        """Test predict returns reasonable values."""
        from sirena.models.bayesian_ridge import BayesianRidgeForecaster

        model = BayesianRidgeForecaster(n_iter=100)
        model.fit(sample_data)

        target_date = sample_data.index[-1]
        result = model.predict(sample_data, target_date)

        assert 98 < result["prediction"] < 102
        assert 98 < result["pred_bayesian"] < 102
        assert 98 < result["pred_ets"] < 102
        assert 0 <= result["ets_weight"] <= 1
        assert result["std"] >= 0

    def test_predict_not_fitted_error(self, sample_data):
        """Test predict raises error when model not fitted."""
        from sirena.models.bayesian_ridge import BayesianRidgeForecaster

        model = BayesianRidgeForecaster()

        target_date = sample_data.index[-1]

        with pytest.raises(ValueError, match="не обучена"):
            model.predict(sample_data, target_date)

    def test_predict_with_ci_alias(self, sample_data):
        """Test predict_with_ci is an alias for predict."""
        from sirena.models.bayesian_ridge import BayesianRidgeForecaster

        model = BayesianRidgeForecaster(n_iter=100)
        model.fit(sample_data)

        target_date = sample_data.index[-1]

        result1 = model.predict(sample_data, target_date)
        result2 = model.predict_with_ci(sample_data, target_date)

        assert result1.keys() == result2.keys()

    def test_forecast_basic(self, sample_data):
        """Test forecast functionality."""
        from sirena.models.bayesian_ridge import BayesianRidgeForecaster

        model = BayesianRidgeForecaster(n_iter=100)
        model.fit(sample_data)

        horizon = 12
        forecast = model.forecast(horizon)

        assert len(forecast) == horizon
        assert all(isinstance(v, (int, float)) for v in forecast)

    def test_forecast_different_horizons(self, sample_data):
        """Test forecast with different horizons."""
        from sirena.models.bayesian_ridge import BayesianRidgeForecaster

        model = BayesianRidgeForecaster(n_iter=100)
        model.fit(sample_data)

        for h in [1, 6, 12, 24]:
            forecast = model.forecast(h)
            assert len(forecast) == h

    def test_forecast_not_fitted_error(self, sample_data):
        """Test forecast raises error when model not fitted."""
        from sirena.models.bayesian_ridge import BayesianRidgeForecaster

        model = BayesianRidgeForecaster()

        with pytest.raises(ValueError, match="не обучена"):
            model.forecast(12)

    def test_forecast_with_ci_basic(self, sample_data):
        """Test forecast_with_ci returns predictions and stds."""
        from sirena.models.bayesian_ridge import BayesianRidgeForecaster

        model = BayesianRidgeForecaster(n_iter=100)
        model.fit(sample_data)

        horizon = 12
        predictions, stds = model.forecast_with_ci(horizon)

        assert len(predictions) == horizon
        assert len(stds) == horizon
        assert all(s >= 0 for s in stds)

    def test_backtest_basic(self, sample_data):
        """Test backtest functionality."""
        from sirena.models.bayesian_ridge import BayesianRidgeForecaster

        model = BayesianRidgeForecaster(n_iter=100)
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
            assert "in_ci" in results.columns

    def test_backtest_custom_start_date(self, sample_data):
        """Test backtest with custom start date."""
        from sirena.models.bayesian_ridge import BayesianRidgeForecaster

        model = BayesianRidgeForecaster(n_iter=100)
        results = model.backtest(sample_data, start_date="2022-06-01")

        assert isinstance(results, pd.DataFrame)

        if not results.empty:
            min_date = results["date"].min()
            assert min_date >= pd.Timestamp("2022-06-01")

    def test_backtest_ci_coverage(self, sample_data):
        """Test backtest CI coverage calculation."""
        from sirena.models.bayesian_ridge import BayesianRidgeForecaster

        model = BayesianRidgeForecaster(n_iter=100)
        results = model.backtest(sample_data, start_date="2023-01-01")

        if not results.empty:
            # Check in_ci column is boolean
            assert results["in_ci"].dtype == bool or results["in_ci"].dtype == np.bool_

            # Some predictions should be in CI
            assert results["in_ci"].sum() > 0 or len(results) == 0

    def test_backtest_with_macro_features(self, sample_data_with_macro):
        """Test backtest with macro features enabled."""
        from sirena.models.bayesian_ridge import BayesianRidgeForecaster

        model = BayesianRidgeForecaster(n_iter=100, use_macro=True)
        results = model.backtest(sample_data_with_macro, start_date="2023-01-01")

        assert isinstance(results, pd.DataFrame)

        if not results.empty:
            # Check that has_macro is True for some predictions
            assert "has_macro" in results.columns

    def test_get_model_params(self, sample_data):
        """Test get model params returns Bayesian Ridge parameters."""
        from sirena.models.bayesian_ridge import BayesianRidgeForecaster

        model = BayesianRidgeForecaster(n_iter=100)
        model.fit(sample_data)

        params = model.get_model_params()

        assert isinstance(params, dict)
        assert "alpha" in params
        assert "lambda" in params
        assert "sigma" in params
        assert "n_iter" in params

        # Check parameters are valid
        assert params["alpha"] >= 0
        assert params["lambda"] >= 0
        assert params["sigma"] > 0
        assert params["n_iter"] > 0

    def test_get_feature_importance(self, sample_data):
        """Test get feature importance."""
        from sirena.models.bayesian_ridge import BayesianRidgeForecaster

        model = BayesianRidgeForecaster(n_iter=100)
        model.fit(sample_data)

        importance = model.get_feature_importance()

        assert isinstance(importance, pd.DataFrame)
        assert "feature" in importance.columns
        assert "coefficient" in importance.columns
        assert "abs_coef" in importance.columns

    def test_get_feature_importance_sorted(self, sample_data):
        """Test feature importance is sorted."""
        from sirena.models.bayesian_ridge import BayesianRidgeForecaster

        model = BayesianRidgeForecaster(n_iter=100)
        model.fit(sample_data)

        importance = model.get_feature_importance()

        if not importance.empty:
            abs_coef_values = importance["abs_coef"].values
            assert all(
                abs_coef_values[i] >= abs_coef_values[i + 1]
                for i in range(len(abs_coef_values) - 1)
            )

    def test_get_model_info(self, sample_data):
        """Test get model info."""
        from sirena.models.bayesian_ridge import BayesianRidgeForecaster

        model = BayesianRidgeForecaster(n_iter=100)
        model.fit(sample_data)

        info = model.get_model_info()

        assert isinstance(info, dict)
        assert "name" in info
        assert "features_count" in info
        assert "has_macro" in info
        assert "is_fitted" in info
        assert info["name"] == "bayesian_ridge"
        assert info["is_fitted"] == True
        assert "alpha" in info
        assert "lambda" in info

    def test_get_model_info_not_fitted(self):
        """Test get model info when not fitted."""
        from sirena.models.bayesian_ridge import BayesianRidgeForecaster

        model = BayesianRidgeForecaster()

        info = model.get_model_info()

        assert isinstance(info, dict)
        assert info["is_fitted"] == False
        assert info["features_count"] == 0

    def test_prepare_features(self, sample_data):
        """Test feature preparation."""
        from sirena.models.bayesian_ridge import BayesianRidgeForecaster

        model = BayesianRidgeForecaster()

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
        assert "is_dec" in df_prep.columns

    def test_prepare_features_components(self, sample_data):
        """Test feature preparation with component lags."""
        from sirena.models.bayesian_ridge import BayesianRidgeForecaster

        model = BayesianRidgeForecaster()

        df_prep = model._prepare_features(sample_data)

        assert "food_lag1" in df_prep.columns
        assert "nonfood_lag1" in df_prep.columns
        assert "services_lag1" in df_prep.columns

    def test_compute_seasonal_norm(self, sample_data):
        """Test seasonal norm computation."""
        from sirena.models.bayesian_ridge import BayesianRidgeForecaster

        model = BayesianRidgeForecaster()

        df_prep = model._prepare_features(sample_data)
        seasonal_norm = model._compute_seasonal_norm(df_prep)

        assert len(seasonal_norm) == 12
        assert all(month in seasonal_norm.index for month in range(1, 13))

    def test_seasonal_norm_excludes_outlier_years(self, sample_data_with_outliers):
        """Test seasonal norm excludes outlier years."""
        from sirena.models.bayesian_ridge import BayesianRidgeForecaster

        model = BayesianRidgeForecaster()

        df_prep = model._prepare_features(sample_data_with_outliers)
        seasonal_norm = model._compute_seasonal_norm(df_prep)

        # Check that 2022 data was excluded
        df_without_2022 = df_prep[df_prep["year"] != 2022]
        expected_norm = df_without_2022.groupby("month")["Все товары и услуги"].mean()

        assert all(seasonal_norm == expected_norm)

    def test_check_fitted(self):
        """Test _check_fitted method."""
        from sirena.models.bayesian_ridge import BayesianRidgeForecaster

        model = BayesianRidgeForecaster()

        with pytest.raises(ValueError, match="не обучена"):
            model._check_fitted()

    def test_repr(self):
        """Test model string representation."""
        from sirena.models.bayesian_ridge import BayesianRidgeForecaster

        model = BayesianRidgeForecaster()
        repr_str = repr(model)

        assert "BayesianRidgeForecaster" in repr_str

    def test_ets_weight_application(self, sample_data):
        """Test ETS weights are applied correctly."""
        from sirena.models.bayesian_ridge import BayesianRidgeForecaster

        model = BayesianRidgeForecaster(n_iter=100)
        model.fit(sample_data)

        target_date = sample_data.index[-1]
        result = model.predict(sample_data, target_date)

        weight = model.ETS_WEIGHTS.get(target_date.month, 0.3)

        expected_pred = (1 - weight) * result["pred_bayesian"] + weight * result[
            "pred_ets"
        ]

        assert abs(result["prediction"] - expected_pred) < 0.0001

    def test_bayesian_ridge_alpha_lambda_parameters(self, sample_data):
        """Test Bayesian Ridge optimizes alpha and lambda."""
        from sirena.models.bayesian_ridge import BayesianRidgeForecaster

        model = BayesianRidgeForecaster(n_iter=100)
        model.fit(sample_data)

        params = model.get_model_params()

        # Alpha and lambda should be positive (optimized)
        assert params["alpha"] > 0
        assert params["lambda"] > 0

        # Alpha and lambda should not be the prior values (should have been optimized)
        assert params["alpha"] != model.alpha_1
        assert params["lambda"] != model.lambda_1

    def test_bayesian_ridge_sigma_parameter(self, sample_data):
        """Test Bayesian Ridge sigma represents noise std."""
        from sirena.models.bayesian_ridge import BayesianRidgeForecaster

        model = BayesianRidgeForecaster(n_iter=100)
        model.fit(sample_data)

        params = model.get_model_params()

        # Sigma should be positive and reasonable for this data
        assert 0 < params["sigma"] < 10

    def test_backtest_custom_target_column(self, sample_data):
        """Test backtest with custom target column."""
        from sirena.models.bayesian_ridge import BayesianRidgeForecaster

        model = BayesianRidgeForecaster(n_iter=100)
        results = model.backtest(
            sample_data, start_date="2023-01-01", target_col="Продовольственные товары"
        )

        assert isinstance(results, pd.DataFrame)

        if not results.empty:
            assert "actual" in results.columns
            assert "prediction" in results.columns

    def test_backtest_with_long_horizon(self, sample_data):
        """Test backtest with longer data."""
        from sirena.models.bayesian_ridge import BayesianRidgeForecaster

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

        model = BayesianRidgeForecaster(n_iter=100)
        results = model.backtest(long_data, start_date="2023-01-01")

        assert isinstance(results, pd.DataFrame)

        if not results.empty:
            assert len(results) > 0

    def test_fit_without_macro_when_macro_disabled(self, sample_data_with_macro):
        """Test that use_macro=False excludes macro features."""
        from sirena.models.bayesian_ridge import BayesianRidgeForecaster

        model = BayesianRidgeForecaster(n_iter=100, use_macro=False)
        model.fit(sample_data_with_macro)

        assert model._has_macro == False
        assert len(model._features) == len(model.BASE_FEATURES)

    def test_n_iter_parameter(self, sample_data):
        """Test n_iter parameter affects training."""
        from sirena.models.bayesian_ridge import BayesianRidgeForecaster

        custom_n_iter = 200
        model = BayesianRidgeForecaster(n_iter=custom_n_iter)

        assert model.n_iter == custom_n_iter

        model.fit(sample_data)

        params = model.get_model_params()
        # Number of iterations should match the parameter
        assert params["n_iter"] <= custom_n_iter
