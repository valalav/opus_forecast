#!/usr/bin/env python3
"""
Test ExogProphet Forecaster
===========================
Unit tests for ExogProphetForecaster with Brent regressor
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
    dates = pd.date_range("2020-01-31", periods=60, freq="MS")
    np.random.seed(42)

    data = pd.DataFrame(
        {
            "mom": 100.5 + np.random.randn(60) * 0.3,
            "Prod": 100.6 + np.random.randn(60) * 0.4,
            "Serv": 100.3 + np.random.randn(60) * 0.3,
            "Nonprod": 100.2 + np.random.randn(60) * 0.2,
        },
        index=dates,
    )

    return data


@pytest.fixture
def sample_data_with_brent():
    """Generate sample data with Brent prices aligned."""
    dates = pd.date_range("2020-01-31", periods=60, freq="MS")
    np.random.seed(42)

    data = pd.DataFrame(
        {
            "mom": 100.5 + np.random.randn(60) * 0.3,
            "Prod": 100.6 + np.random.randn(60) * 0.4,
            "Serv": 100.3 + np.random.randn(60) * 0.3,
            "Nonprod": 100.2 + np.random.randn(60) * 0.2,
            "usd_nom_i": 90 + np.random.randn(60) * 2.0,
            "Ki": 16 + np.random.randn(60) * 0.5,
            "brent": 80 + np.random.randn(60) * 5.0,
        },
        index=dates,
    )

    return data


@pytest.fixture
def sample_brent_data():
    """Generate sample Brent data aligned with macro dates."""
    dates = pd.date_range("2020-01-31", periods=60, freq="MS")
    np.random.seed(42)

    data = pd.DataFrame(
        {
            "brent": 80 + np.random.randn(60) * 5.0,
            "brent_pct": np.random.randn(60) * 2.0,
        },
        index=dates,
    )

    return data


class TestExogProphetForecaster:
    """Test suite for ExogProphetForecaster."""

    def test_import_model(self):
        """Test model import."""
        from sirena.models.exog_prophet import ExogProphetForecaster, PROPHET_AVAILABLE

        model = ExogProphetForecaster()
        assert model is not None
        assert model.name == "exog_prophet"

    def test_model_parameters_default(self):
        """Test model default parameters."""
        from sirena.models.exog_prophet import ExogProphetForecaster

        model = ExogProphetForecaster()

        assert model.use_usd is True
        assert model.use_brent is True
        assert model.use_ki is True
        assert model.yearly_seasonality is True
        assert model.seasonality_mode == "additive"
        assert model.changepoint_prior_scale == 0.05
        assert model.seasonality_prior_scale == 10.0
        assert model.outlier_years == [2022]

    def test_model_parameters_custom(self):
        """Test model with custom parameters."""
        from sirena.models.exog_prophet import ExogProphetForecaster

        model = ExogProphetForecaster(
            use_usd=False,
            use_brent=False,
            use_ki=False,
            yearly_seasonality=False,
            seasonality_mode="multiplicative",
            changepoint_prior_scale=0.1,
            seasonality_prior_scale=5.0,
            outlier_years=[2020, 2022],
        )

        assert model.use_usd is False
        assert model.use_brent is False
        assert model.use_ki is False
        assert model.yearly_seasonality is False
        assert model.seasonality_mode == "multiplicative"
        assert model.changepoint_prior_scale == 0.1
        assert model.seasonality_prior_scale == 5.0
        assert model.outlier_years == [2020, 2022]

    def test_lag_constants(self):
        """Test lag constants."""
        from sirena.models.exog_prophet import ExogProphetForecaster

        model = ExogProphetForecaster()

        assert model.USD_LAG == 2
        assert model.BRENT_LAG == 5
        assert model.KI_LAG == 6

    def test_load_macro_data(self):
        """Test macro data loading."""
        from sirena.models.exog_prophet import ExogProphetForecaster

        model = ExogProphetForecaster()
        macro_df = model._load_macro_data()

        assert macro_df is not None
        assert isinstance(macro_df, pd.DataFrame)
        assert len(macro_df) > 0
        assert "mom" in macro_df.columns
        assert isinstance(macro_df.index, pd.DatetimeIndex)

    def test_load_brent_data(self):
        """Test Brent data loading."""
        from sirena.models.exog_prophet import ExogProphetForecaster

        model = ExogProphetForecaster()
        brent_df = model._load_brent_data()

        # Brent data may or may not exist
        if brent_df is not None:
            assert isinstance(brent_df, pd.DataFrame)
            assert len(brent_df) > 0
            assert "brent" in brent_df.columns
            assert isinstance(brent_df.index, pd.DatetimeIndex)

    def test_prepare_features(self):
        """Test feature preparation with lags."""
        from sirena.models.exog_prophet import ExogProphetForecaster

        model = ExogProphetForecaster(use_usd=True, use_brent=True, use_ki=True)

        # Create sample data with required columns
        df = pd.DataFrame(
            {
                "mom": np.arange(100, 160, dtype=float),
                "usd_nom_i": np.arange(90, 150, dtype=float),
                "Ki": np.arange(15, 75, dtype=float),
            },
            index=pd.date_range("2020-01-31", periods=60, freq="MS"),
        )

        # Mock brent_df with aligned dates
        model.brent_df = pd.DataFrame(
            {
                "brent": np.arange(75, 135, dtype=float),
            },
            index=df.index,
        )

        prepared = model._prepare_features(df)

        assert prepared is not None
        assert "usd_lag2" in prepared.columns
        assert "ki_lag6" in prepared.columns
        assert "brent_lag5" in prepared.columns

        # Check that lag creates NaN values at start
        assert prepared["usd_lag2"].isna().sum() >= 2
        assert prepared["ki_lag6"].isna().sum() >= 6
        assert prepared["brent_lag5"].isna().sum() >= 5

    def test_brent_regressor_enabled(self):
        """Test that Brent regressor is enabled by default."""
        from sirena.models.exog_prophet import ExogProphetForecaster

        model = ExogProphetForecaster()

        assert model.use_brent is True

    def test_brent_regressor_disabled(self):
        """Test that Brent regressor can be disabled."""
        from sirena.models.exog_prophet import ExogProphetForecaster

        model = ExogProphetForecaster(use_brent=False)

        assert model.use_brent is False

    def test_regressors_initialization(self):
        """Test regressors list initialization."""
        from sirena.models.exog_prophet import ExogProphetForecaster

        model = ExogProphetForecaster()

        assert model.regressors == []
        assert model.model is None
        assert model.last_date is None
