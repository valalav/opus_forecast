"""
Robustness Tests: NaN Handling
============================

Test suite to verify models handle missing data (NaNs) correctly.
Models should either:
1. Handle NaNs gracefully (dropna, fillna, imputation)
2. Raise clean, informative errors
"""

import pytest
import numpy as np
import pandas as pd
import re
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from sirena.models import (
    RidgeForecaster,
    HuberForecaster,
    ElasticNetForecaster,
    NGBoostForecaster,
    EBMForecaster,
    SubcomponentForecaster,
    NaiveSeasonalForecaster,
    HoltWintersForecaster,
)


class TestNaNHandling:
    """Test suite for NaN robustness."""

    @pytest.fixture
    def sample_data(self):
        """Generate clean sample data."""
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
    def data_with_random_nans(self, sample_data):
        """Data with random NaNs (5% of values)."""
        df = sample_data.copy()
        np.random.seed(123)

        mask = np.random.random(df.shape) < 0.05
        df[mask] = np.nan

        return df

    @pytest.fixture
    def data_with_target_nans(self, sample_data):
        """Data with NaNs only in target column."""
        df = sample_data.copy()
        df.loc[df.index[10:15], "Все товары и услуги"] = np.nan
        return df

    @pytest.fixture
    def data_with_trailing_nans(self, sample_data):
        """Data with NaNs at end (common in time series)."""
        df = sample_data.copy()
        df.loc[df.index[-5:], "Все товары и услуги"] = np.nan
        return df

    @pytest.fixture
    def data_with_all_nan_column(self, sample_data):
        """Data with one completely NaN column."""
        df = sample_data.copy()
        df["Продовольственные товары"] = np.nan
        return df

    @pytest.fixture
    def data_with_large_nan_block(self, sample_data):
        """Data with large block of NaNs."""
        df = sample_data.copy()
        df.loc[df.index[15:25], "Все товары и услуги"] = np.nan
        return df

    def test_ridge_with_random_nans(self, data_with_random_nans):
        """Ridge should handle random NaNs via dropna."""
        model = RidgeForecaster()

        model.fit(data_with_random_nans)
        assert model._is_fitted

        fc = model.forecast(horizon=6)
        assert fc is not None
        assert len(fc) == 6
        assert not np.any(np.isnan(fc))

    def test_ridge_with_target_nans(self, data_with_target_nans):
        """Ridge should drop NaN target rows."""
        model = RidgeForecaster()

        model.fit(data_with_target_nans)
        assert model._is_fitted

        fc = model.forecast(horizon=3)
        assert fc is not None

    def test_ridge_with_trailing_nans(self, data_with_trailing_nans):
        """Ridge should handle trailing NaNs (common lag effect)."""
        model = RidgeForecaster()

        model.fit(data_with_trailing_nans)
        assert model._is_fitted

        fc = model.forecast(horizon=3)
        assert fc is not None
        assert len(fc) == 3

    def test_ridge_with_large_nan_block(self, data_with_large_nan_block):
        """Ridge should handle large NaN blocks."""
        model = RidgeForecaster()

        model.fit(data_with_large_nan_block)
        assert model._is_fitted

        fc = model.forecast(horizon=3)
        assert fc is not None

    def test_huber_with_random_nans(self, data_with_random_nans):
        """Huber (robust model) should handle NaNs."""
        model = HuberForecaster()

        model.fit(data_with_random_nans)
        assert model._is_fitted

        fc = model.forecast(horizon=6)
        assert fc is not None
        assert not np.any(np.isnan(fc))

    def test_huber_with_target_nans(self, data_with_target_nans):
        """Huber should handle target NaNs."""
        model = HuberForecaster()

        model.fit(data_with_target_nans)
        assert model._is_fitted

        fc = model.forecast(horizon=3)
        assert fc is not None

    def test_elasticnet_with_random_nans(self, data_with_random_nans):
        """ElasticNet should handle NaNs."""
        model = ElasticNetForecaster()

        model.fit(data_with_random_nans)
        assert model._is_fitted

        fc = model.forecast(horizon=6)
        assert fc is not None

    def test_elasticnet_with_trailing_nans(self, data_with_trailing_nans):
        """ElasticNet should handle trailing NaNs."""
        model = ElasticNetForecaster()

        model.fit(data_with_trailing_nans)
        assert model._is_fitted

        fc = model.forecast(horizon=3)
        assert fc is not None

    def test_ebm_with_random_nans(self, data_with_random_nans):
        """EBM should handle NaNs via dropna."""
        model = EBMForecaster()

        model.fit(data_with_random_nans)
        assert model._is_fitted

        fc = model.forecast(horizon=6)
        assert fc is not None

    def test_ebm_with_target_nans(self, data_with_target_nans):
        """EBM should handle target NaNs."""
        model = EBMForecaster()

        model.fit(data_with_target_nans)
        assert model._is_fitted

        fc = model.forecast(horizon=3)
        assert fc is not None

    def test_ngboost_with_random_nans(self, data_with_random_nans):
        """NGBoost should handle NaNs."""
        model = NGBoostForecaster()

        model.fit(data_with_random_nans)
        assert model._is_fitted

        fc = model.forecast(horizon=6)
        assert fc is not None

    def test_ngboost_with_trailing_nans(self, data_with_trailing_nans):
        """NGBoost should handle trailing NaNs."""
        model = NGBoostForecaster()

        model.fit(data_with_trailing_nans)
        assert model._is_fitted

        fc = model.forecast(horizon=3)
        assert fc is not None

    def test_naive_seasonal_with_nans(self, data_with_random_nans):
        """NaiveSeasonal should handle NaNs via dropna."""
        model = NaiveSeasonalForecaster()

        model.fit(data_with_random_nans)
        assert model._is_fitted

        fc = model.forecast(horizon=12)
        assert fc is not None
        assert len(fc) == 12

    def test_holt_winters_with_nans(self, data_with_random_nans):
        """HoltWinters should handle NaNs via dropna."""
        model = HoltWintersForecaster()

        model.fit(data_with_random_nans)
        assert model._is_fitted

        fc = model.forecast(horizon=6)
        assert fc is not None

    def test_predict_with_nans_in_features(self, sample_data):
        """Test predict with NaN in features - should raise clean error."""
        model = RidgeForecaster()
        model.fit(sample_data)

        target_date = sample_data.index[-1]
        df_test = sample_data.copy()
        df_test.loc[df_test.index[-2], "Продовольственные товары"] = np.nan

        with pytest.raises(ValueError) as exc_info:
            model.predict(df_test, target_date)

        error_msg = str(exc_info.value).lower()
        assert "nan" in error_msg, f"Error should mention NaN, got: {error_msg}"

    def test_subcomponent_with_nans(self, data_with_random_nans):
        """Subcomponent should handle NaNs."""
        model = SubcomponentForecaster(horizon=1)

        model.fit(data_with_random_nans)
        assert model._is_fitted

        fc = model.forecast()
        assert fc is not None

    def test_model_clean_error_on_all_nan_data(self, sample_data):
        """Models should raise clean error if ALL target data is NaN."""
        df_all_nan = sample_data.copy()
        df_all_nan["Все товары и услуги"] = np.nan

        models_to_test = [
            RidgeForecaster,
            HuberForecaster,
            ElasticNetForecaster,
        ]

        for ModelClass in models_to_test:
            model = ModelClass()

            with pytest.raises(ValueError) as exc_info:
                model.fit(df_all_nan)

            error_msg = str(exc_info.value).lower()
            assert any(
                keyword in error_msg
                for keyword in ["недостаточно", "insufficient", "empty", "nan"]
            ), f"{ModelClass.__name__} should raise clear error for all-NaN data"

    def test_prediction_no_nan_output(self, data_with_random_nans):
        """Forecasts should never contain NaN in output."""
        models_to_test = [
            RidgeForecaster(),
            HuberForecaster(),
            ElasticNetForecaster(),
            EBMForecaster(),
        ]

        for model in models_to_test:
            model.fit(data_with_random_nans)
            fc = model.forecast(horizon=6)

            assert not np.any(np.isnan(fc)), (
                f"{model.name} forecast contains NaN values"
            )

    def test_multiple_nan_patterns(self, sample_data):
        """Test multiple NaN injection patterns with different ratios."""
        patterns = [
            (0.02, "2%"),
            (0.05, "5%"),
        ]

        model = RidgeForecaster()

        for nan_ratio, pattern_name in patterns:
            df = sample_data.copy()
            np.random.seed(42)

            mask = np.random.random(df.shape) < nan_ratio
            df[mask] = np.nan

            model.fit(df)
            assert model._is_fitted, f"Failed to fit with {pattern_name}"

            fc = model.forecast(horizon=3)
            assert fc is not None, f"Failed to forecast with {pattern_name}"
            assert not np.any(np.isnan(fc)), f"Forecast has NaNs with {pattern_name}"

    def test_min_train_size_with_nans(self, sample_data):
        """MIN_TRAIN_SIZE should account for NaN removal."""
        df = sample_data.copy()

        model = RidgeForecaster()

        model.fit(df)
        assert model._is_fitted

        effective_size = len(df.dropna(subset=["Все товары и услуги"]))
        assert effective_size >= model.MIN_TRAIN_SIZE, (
            "Effective training size after NaN dropna should meet MIN_TRAIN_SIZE"
        )
