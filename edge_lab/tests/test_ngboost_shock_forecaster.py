"""
Unit tests for NGBoostShockForecaster
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
    """Generate sample data with historical shock periods."""
    dates = pd.date_range("2014-01-01", periods=120, freq="MS")
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

    # Add shocks to simulate real data
    # Dec 2014 shock
    data.loc["2014-12-01", "Все товары и услуги"] += 2.5
    # Jan 2015 shock
    data.loc["2015-01-01", "Все товары и услуги"] += 3.0
    # 2022 shocks
    data.loc["2022-03-01", "Все товары и услуги"] += 2.0
    data.loc["2022-04-01", "Все товары и услуги"] += 2.5

    return data


@pytest.fixture
def sample_data_with_2022():
    """Generate sample data with 2022 outlier year."""
    dates = pd.date_range("2019-01-01", periods=72, freq="MS")
    np.random.seed(42)

    data = pd.DataFrame(
        {
            "Все товары и услуги": 100.5 + np.random.randn(72) * 0.3,
            "Продовольственные товары": 100.6 + np.random.randn(72) * 0.4,
            "Непродовольственные товары": 100.3 + np.random.randn(72) * 0.2,
            "Услуги": 100.4 + np.random.randn(72) * 0.3,
        },
        index=dates,
    )

    # Add 2022 outliers
    data.iloc[36:48, 0] += np.array(
        [2.5, 3.0, 2.0, 1.5, 1.0, 0.8, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
    )

    return data


class TestNGBoostShockForecaster:
    """Test suite for NGBoostShockForecaster."""

    def test_import_model(self):
        """Test model import."""
        from sirena.models.ngboost_shock import NGBoostShockForecaster

        model = NGBoostShockForecaster()
        assert model is not None
        assert model.name == "ngboost_shock"

    def test_model_parameters(self):
        """Test model default parameters."""
        from sirena.models.ngboost_shock import NGBoostShockForecaster

        model = NGBoostShockForecaster()

        assert model.MIN_TRAIN_SIZE == 36
        assert model.OUTLIER_YEARS == [2010]  # 2022 NOT excluded!
        assert model.n_estimators == 200
        assert model.learning_rate == 0.05
        assert model.MINIBATCH_FRAC == 0.8
        assert model._is_fitted == False
        assert model.model is None

    def test_shock_dummies_list(self):
        """Test shock dummies list."""
        from sirena.models.ngboost_shock import NGBoostShockForecaster

        model = NGBoostShockForecaster()

        required_dummies = [
            "is_shock_dec2014",
            "is_shock_jan2015",
            "is_shock_mar2022",
            "is_shock_apr2022",
            "is_shock_2022",
        ]

        for dummy in required_dummies:
            assert dummy in model.SHOCK_DUMMIES

        assert len(model.SHOCK_DUMMIES) == 5

    def test_base_features_list(self):
        """Test base features list."""
        from sirena.models.ngboost_shock import NGBoostShockForecaster

        model = NGBoostShockForecaster()

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
        from sirena.models.ngboost_shock import NGBoostShockForecaster

        model = NGBoostShockForecaster()

        assert len(model.ETS_WEIGHTS) == 12
        assert 1 in model.ETS_WEIGHTS
        assert model.ETS_WEIGHTS[1] == 0.9
        assert model.ETS_WEIGHTS[7] == 0.0

    def test_custom_parameters(self):
        """Test custom parameters initialization."""
        from sirena.models.ngboost_shock import NGBoostShockForecaster

        custom_n_estimators = 100
        custom_lr = 0.1

        model = NGBoostShockForecaster(
            n_estimators=custom_n_estimators,
            learning_rate=custom_lr,
        )

        assert model.n_estimators == custom_n_estimators
        assert model.learning_rate == custom_lr

    def test_add_shock_dummies_dec2014(self, sample_data):
        """Test shock dummy for Dec 2014."""
        from sirena.models.ngboost_shock import NGBoostShockForecaster

        model = NGBoostShockForecaster()
        df_with_dummies = model._add_shock_dummies(sample_data)

        assert "is_shock_dec2014" in df_with_dummies.columns

        # Sample data is 2020-2025, so no Dec 2014
        assert df_with_dummies["is_shock_dec2014"].sum() == 0

    def test_add_shock_dummies_jan2015(self, sample_data):
        """Test shock dummy for Jan 2015."""
        from sirena.models.ngboost_shock import NGBoostShockForecaster

        model = NGBoostShockForecaster()
        df_with_dummies = model._add_shock_dummies(sample_data)

        assert "is_shock_jan2015" in df_with_dummies.columns

        # Sample data is 2020-2025, so no Jan 2015
        assert df_with_dummies["is_shock_jan2015"].sum() == 0

    def test_add_shock_dummies_mar2022(self, sample_data_with_2022):
        """Test shock dummy for Mar 2022."""
        from sirena.models.ngboost_shock import NGBoostShockForecaster

        model = NGBoostShockForecaster()
        df_with_dummies = model._add_shock_dummies(sample_data_with_2022)

        assert "is_shock_mar2022" in df_with_dummies.columns

        # Mar 2022 should have shock = 1
        assert df_with_dummies.loc["2022-03-01", "is_shock_mar2022"] == 1

    def test_add_shock_dummies_apr2022(self, sample_data_with_2022):
        """Test shock dummy for Apr 2022."""
        from sirena.models.ngboost_shock import NGBoostShockForecaster

        model = NGBoostShockForecaster()
        df_with_dummies = model._add_shock_dummies(sample_data_with_2022)

        assert "is_shock_apr2022" in df_with_dummies.columns

        # Apr 2022 should have shock = 1
        assert df_with_dummies.loc["2022-04-01", "is_shock_apr2022"] == 1

    def test_add_shock_dummies_2022_year(self, sample_data_with_2022):
        """Test shock dummy for entire 2022 year."""
        from sirena.models.ngboost_shock import NGBoostShockForecaster

        model = NGBoostShockForecaster()
        df_with_dummies = model._add_shock_dummies(sample_data_with_2022)

        assert "is_shock_2022" in df_with_dummies.columns

        # All 2022 months should have is_shock_2022 = 1
        months_2022 = df_with_dummies[df_with_dummies.index.year == 2022]
        assert all(months_2022["is_shock_2022"] == 1)

    def test_add_shock_dummies_all_periods(self, sample_data_with_shocks):
        """Test all shock dummies on historical data."""
        from sirena.models.ngboost_shock import NGBoostShockForecaster

        model = NGBoostShockForecaster()
        df_with_dummies = model._add_shock_dummies(sample_data_with_shocks)

        # All shock dummies should be present
        for dummy in model.SHOCK_DUMMIES:
            assert dummy in df_with_dummies.columns

        # Check specific shock periods
        assert df_with_dummies.loc["2014-12-01", "is_shock_dec2014"] == 1
        assert df_with_dummies.loc["2015-01-01", "is_shock_jan2015"] == 1
        assert df_with_dummies.loc["2022-03-01", "is_shock_mar2022"] == 1
        assert df_with_dummies.loc["2022-04-01", "is_shock_apr2022"] == 1

        # All 2022 months should have is_shock_2022 = 1
        months_2022 = df_with_dummies[df_with_dummies.index.year == 2022]
        assert all(months_2022["is_shock_2022"] == 1)

    def test_fit_basic(self, sample_data):
        """Test basic fit functionality."""
        from sirena.models.ngboost_shock import NGBoostShockForecaster

        model = NGBoostShockForecaster(n_estimators=50)
        model.fit(sample_data)

        assert model.is_fitted
        assert model.model is not None
        assert model.scaler is not None
        assert model.seasonal_norm is not None
        assert len(model.seasonal_norm) == 12
        assert model._features is not None

    def test_fit_includes_2022(self, sample_data_with_2022):
        """Test fit includes 2022 data (unlike NGBoostForecaster)."""
        from sirena.models.ngboost_shock import NGBoostShockForecaster

        model = NGBoostShockForecaster(n_estimators=50)
        model.fit(sample_data_with_2022)

        assert model.is_fitted
        # 2022 should NOT be in OUTLIER_YEARS
        assert 2022 not in model.OUTLIER_YEARS
        # Only 2010 is excluded
        assert model.OUTLIER_YEARS == [2010]

    def test_fit_with_shocks(self, sample_data_with_shocks):
        """Test fit handles shock periods with dummies."""
        from sirena.models.ngboost_shock import NGBoostShockForecaster

        model = NGBoostShockForecaster(n_estimators=50)
        model.fit(sample_data_with_shocks)

        assert model.is_fitted
        # Shock dummies should be in features
        for dummy in model.SHOCK_DUMMIES:
            assert dummy in model._features

    def test_fit_insufficient_data(self, sample_data):
        """Test fit with insufficient data."""
        from sirena.models.ngboost_shock import NGBoostShockForecaster

        small_data = sample_data.iloc[:30]

        model = NGBoostShockForecaster()

        with pytest.raises(ValueError, match="Недостаточно данных"):
            model.fit(small_data)

    def test_fit_empty_dataframe(self):
        """Test fit with empty DataFrame."""
        from sirena.models.ngboost_shock import NGBoostShockForecaster

        empty_df = pd.DataFrame()

        model = NGBoostShockForecaster()

        with pytest.raises(ValueError, match="DataFrame пустой"):
            model.fit(empty_df)

    def test_fit_missing_target_column(self, sample_data):
        """Test fit with missing target column."""
        from sirena.models.ngboost_shock import NGBoostShockForecaster

        df_no_target = sample_data.drop(columns=["Все товары и услуги"])

        model = NGBoostShockForecaster()

        with pytest.raises(ValueError, match="Колонка.*не найдена"):
            model.fit(df_no_target)

    def test_predict_basic(self, sample_data):
        """Test basic predict functionality."""
        from sirena.models.ngboost_shock import NGBoostShockForecaster

        model = NGBoostShockForecaster(n_estimators=50)
        model.fit(sample_data)

        target_date = sample_data.index[-1]
        result = model.predict(sample_data, target_date)

        assert "prediction" in result
        assert "pred_ngboost" in result
        assert "pred_ets" in result
        assert "std" in result
        assert "ci_lower" in result
        assert "ci_upper" in result
        assert "model" in result
        assert result["model"] == "ngboost_shock"

    def test_predict_probabilistic_output(self, sample_data):
        """Test predict returns probabilistic outputs (std, CI)."""
        from sirena.models.ngboost_shock import NGBoostShockForecaster

        model = NGBoostShockForecaster(n_estimators=50)
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
        from sirena.models.ngboost_shock import NGBoostShockForecaster

        model = NGBoostShockForecaster(n_estimators=50)
        model.fit(sample_data)

        target_date = sample_data.index[-1]
        result = model.predict(sample_data, target_date)

        # Prediction should be within CI
        assert result["ci_lower"] <= result["prediction"] <= result["ci_upper"]

    def test_predict_range(self, sample_data):
        """Test predict returns reasonable values."""
        from sirena.models.ngboost_shock import NGBoostShockForecaster

        model = NGBoostShockForecaster(n_estimators=50)
        model.fit(sample_data)

        target_date = sample_data.index[-1]
        result = model.predict(sample_data, target_date)

        assert 98 < result["prediction"] < 102
        assert 98 < result["pred_ngboost"] < 102
        assert 98 < result["pred_ets"] < 102
        assert result["std"] >= 0

    def test_predict_not_fitted_error(self, sample_data):
        """Test predict raises error when model not fitted."""
        from sirena.models.ngboost_shock import NGBoostShockForecaster

        model = NGBoostShockForecaster()

        target_date = sample_data.index[-1]

        with pytest.raises(ValueError, match="не обучена"):
            model.predict(sample_data, target_date)

    def test_forecast_basic(self, sample_data):
        """Test forecast functionality."""
        from sirena.models.ngboost_shock import NGBoostShockForecaster

        model = NGBoostShockForecaster(n_estimators=50)
        model.fit(sample_data)

        horizon = 12
        forecast = model.forecast(horizon)

        assert len(forecast) == horizon
        assert all(isinstance(v, (int, float)) for v in forecast)

    def test_forecast_different_horizons(self, sample_data):
        """Test forecast with different horizons."""
        from sirena.models.ngboost_shock import NGBoostShockForecaster

        model = NGBoostShockForecaster(n_estimators=50)
        model.fit(sample_data)

        for h in [1, 6, 12, 24]:
            forecast = model.forecast(h)
            assert len(forecast) == h

    def test_forecast_not_fitted_error(self, sample_data):
        """Test forecast raises error when model not fitted."""
        from sirena.models.ngboost_shock import NGBoostShockForecaster

        model = NGBoostShockForecaster()

        with pytest.raises(ValueError, match="не обучена"):
            model.forecast(12)

    def test_backtest_basic(self, sample_data):
        """Test backtest functionality."""
        from sirena.models.ngboost_shock import NGBoostShockForecaster

        model = NGBoostShockForecaster(n_estimators=50)
        results = model.backtest(sample_data, start_date="2023-01-01")

        assert isinstance(results, pd.DataFrame)

        if not results.empty:
            assert "date" in results.columns
            assert "actual" in results.columns
            assert "prediction" in results.columns
            assert "error" in results.columns

    def test_backtest_custom_start_date(self, sample_data):
        """Test backtest with custom start date."""
        from sirena.models.ngboost_shock import NGBoostShockForecaster

        model = NGBoostShockForecaster(n_estimators=50)
        results = model.backtest(sample_data, start_date="2022-06-01")

        assert isinstance(results, pd.DataFrame)

        if not results.empty:
            min_date = results["date"].min()
            assert min_date >= pd.Timestamp("2022-06-01")

    def test_backtest_with_shocks(self, sample_data_with_shocks):
        """Test backtest with historical shock periods."""
        from sirena.models.ngboost_shock import NGBoostShockForecaster

        model = NGBoostShockForecaster(n_estimators=50)
        results = model.backtest(sample_data_with_shocks, start_date="2023-01-01")

        assert isinstance(results, pd.DataFrame)

        if not results.empty:
            assert "date" in results.columns
            assert "actual" in results.columns
            assert "prediction" in results.columns
            assert "error" in results.columns

    def test_get_feature_importance(self, sample_data):
        """Test get feature importance."""
        from sirena.models.ngboost_shock import NGBoostShockForecaster

        model = NGBoostShockForecaster(n_estimators=50)
        model.fit(sample_data)

        importance = model.get_feature_importance()

        assert isinstance(importance, pd.DataFrame)
        assert "feature" in importance.columns
        assert "importance" in importance.columns
        assert "is_shock" in importance.columns

    def test_get_feature_importance_shock_dummies(self, sample_data_with_shocks):
        """Test shock dummies are marked in feature importance."""
        from sirena.models.ngboost_shock import NGBoostShockForecaster

        model = NGBoostShockForecaster(n_estimators=50)
        model.fit(sample_data_with_shocks)

        importance = model.get_feature_importance()

        # Check is_shock column exists
        assert "is_shock" in importance.columns

        # All shock dummies should be present
        shock_rows = importance[importance["feature"].isin(model.SHOCK_DUMMIES)]
        assert len(shock_rows) == len(model.SHOCK_DUMMIES)

        # All shock dummies should have is_shock = True
        assert all(shock_rows["is_shock"])

    def test_get_feature_importance_sorted(self, sample_data):
        """Test feature importance is sorted."""
        from sirena.models.ngboost_shock import NGBoostShockForecaster

        model = NGBoostShockForecaster(n_estimators=50)
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
        from sirena.models.ngboost_shock import NGBoostShockForecaster

        model = NGBoostShockForecaster()

        with pytest.raises(ValueError, match="не обучена"):
            model.get_feature_importance()

    def test_prepare_features(self, sample_data):
        """Test feature preparation includes shock dummies."""
        from sirena.models.ngboost_shock import NGBoostShockForecaster

        model = NGBoostShockForecaster()

        df_prep = model._prepare_features(sample_data)

        # Base features should be present
        assert "month" in df_prep.columns
        assert "year" in df_prep.columns
        assert "y_lag1" in df_prep.columns
        assert "month_sin" in df_prep.columns

        # Shock dummies should be present
        for dummy in model.SHOCK_DUMMIES:
            assert dummy in df_prep.columns

    def test_compute_seasonal_norm_excludes_2022(self, sample_data_with_2022):
        """Test seasonal norm computation excludes 2022."""
        from sirena.models.ngboost_shock import NGBoostShockForecaster

        model = NGBoostShockForecaster()

        df_prep = model._prepare_features(sample_data_with_2022)
        seasonal_norm = model._compute_seasonal_norm(df_prep)

        # Check that 2022 data was excluded
        df_without_2022 = df_prep[df_prep["year"] != 2022]
        expected_norm = df_without_2022.groupby("month")["Все товары и услуги"].mean()

        assert all(seasonal_norm == expected_norm)

    def test_check_fitted(self):
        """Test _check_fitted method."""
        from sirena.models.ngboost_shock import NGBoostShockForecaster

        model = NGBoostShockForecaster()

        with pytest.raises(ValueError, match="не обучена"):
            model._check_fitted()

    def test_repr(self):
        """Test model string representation."""
        from sirena.models.ngboost_shock import NGBoostShockForecaster

        model = NGBoostShockForecaster()
        repr_str = repr(model)

        assert "NGBoostShockForecaster" in repr_str

    def test_2022_included_in_training(self, sample_data_with_2022):
        """Test that 2022 is included in training data."""
        from sirena.models.ngboost_shock import NGBoostShockForecaster

        model = NGBoostShockForecaster(n_estimators=50)
        model.fit(sample_data_with_2022)

        # 2022 should NOT be excluded
        assert 2022 not in model.OUTLIER_YEARS

        # Check that training data includes 2022
        assert model._last_train_date is not None

    def test_backtest_custom_target_column(self, sample_data):
        """Test backtest with custom target column."""
        from sirena.models.ngboost_shock import NGBoostShockForecaster

        model = NGBoostShockForecaster(n_estimators=50)
        results = model.backtest(
            sample_data, start_date="2023-01-01", target_col="Продовольственные товары"
        )

        assert isinstance(results, pd.DataFrame)

        if not results.empty:
            assert "actual" in results.columns
            assert "prediction" in results.columns

    def test_features_include_shock_dummies(self, sample_data):
        """Test that features include shock dummies after fit."""
        from sirena.models.ngboost_shock import NGBoostShockForecaster

        model = NGBoostShockForecaster(n_estimators=50)
        model.fit(sample_data)

        # Check all shock dummies are in features
        for dummy in model.SHOCK_DUMMIES:
            assert dummy in model._features

        # Total features = BASE_FEATURES + SHOCK_DUMMIES
        expected_count = len(model.BASE_FEATURES) + len(model.SHOCK_DUMMIES)
        assert len(model._features) == expected_count

    def test_backtest_with_long_horizon(self, sample_data):
        """Test backtest with longer data."""
        from sirena.models.ngboost_shock import NGBoostShockForecaster

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

        model = NGBoostShockForecaster(n_estimators=50)
        results = model.backtest(long_data, start_date="2023-01-01")

        assert isinstance(results, pd.DataFrame)

        if not results.empty:
            assert len(results) > 0
