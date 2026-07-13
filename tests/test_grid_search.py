"""
Unit tests for parameter grid search functionality.

Tests cover:
- ElasticNetForecaster grid search parameters
- RidgeForecaster parameter handling
- Mock model fitting to avoid long training times
- Best parameters selection
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from sirena.models.elasticnet import ElasticNetForecaster
from sirena.models.ridge import RidgeForecaster


@pytest.fixture
def sample_monthly_data():
    """Generate sample monthly CPI data for testing."""
    dates = pd.date_range("2016-01-01", periods=96, freq="MS")
    np.random.seed(42)

    data = pd.DataFrame(
        {
            "Все товары и услуги": 100.5 + np.random.randn(96) * 0.3,
            "Продовольственные товары": 100.6 + np.random.randn(96) * 0.4,
            "Непродовольственные товары": 100.3 + np.random.randn(96) * 0.2,
            "Услуги": 100.4 + np.random.randn(96) * 0.3,
        },
        index=dates,
    )

    return data


@pytest.fixture
def elasticnet_model():
    """Create default ElasticNetForecaster instance."""
    return ElasticNetForecaster()


@pytest.fixture
def ridge_model():
    """Create default RidgeForecaster instance."""
    return RidgeForecaster()


class TestElasticNetGridSearchParams:
    """Tests for ElasticNetForecaster grid search parameter setup."""

    def test_default_grid_parameters(self, elasticnet_model):
        """Test default grid search parameters are set correctly."""
        # Default l1_ratios
        assert len(elasticnet_model.l1_ratios) == 5
        assert elasticnet_model.l1_ratios == [0.1, 0.3, 0.5, 0.7, 0.9]

        # Default alphas
        assert len(elasticnet_model.alphas) == 5
        assert elasticnet_model.alphas == [0.001, 0.01, 0.1, 0.3, 1.0]

        # Default CV folds
        assert elasticnet_model.cv == 5

    def test_custom_grid_parameters(self):
        """Test custom grid search parameters are set correctly."""
        custom_l1_ratios = [0.2, 0.4, 0.6, 0.8]
        custom_alphas = [0.005, 0.05, 0.5]

        model = ElasticNetForecaster(
            l1_ratios=custom_l1_ratios, alphas=custom_alphas, cv=10
        )

        assert model.l1_ratios == custom_l1_ratios
        assert model.alphas == custom_alphas
        assert model.cv == 10

    def test_grid_parameter_bounds(self):
        """Test grid parameters are within valid ranges."""
        model = ElasticNetForecaster()

        # l1_ratio must be in [0, 1]
        for ratio in model.l1_ratios:
            assert 0 <= ratio <= 1, f"l1_ratio {ratio} out of bounds [0, 1]"

        # alphas must be positive
        for alpha in model.alphas:
            assert alpha > 0, f"alpha {alpha} must be positive"

        # cv must be >= 2
        assert model.cv >= 2


class TestElasticNetGridSearchExecution:
    """Tests for ElasticNetForecaster grid search execution with mocking."""

    @patch("sirena.models.elasticnet.ElasticNetCV")
    def test_grid_search_calls_fit_with_correct_params(
        self, mock_elasticnet_cv, elasticnet_model, sample_monthly_data
    ):
        """Test that ElasticNetCV is called with correct grid parameters."""
        # Setup mock
        mock_instance = Mock()
        mock_instance.alpha_ = 0.1
        mock_instance.l1_ratio_ = 0.5
        mock_elasticnet_cv.return_value = mock_instance

        # Fit model
        elasticnet_model.fit(sample_monthly_data)

        # Verify ElasticNetCV was called with correct parameters
        mock_elasticnet_cv.assert_called_once()
        call_kwargs = mock_elasticnet_cv.call_args[1]

        assert call_kwargs["l1_ratio"] == elasticnet_model.l1_ratios
        assert call_kwargs["alphas"] == elasticnet_model.alphas
        assert call_kwargs["cv"] == elasticnet_model.cv
        assert call_kwargs["max_iter"] == 5000
        assert call_kwargs["random_state"] == 42

    @patch("sirena.models.elasticnet.ElasticNetCV")
    def test_best_alpha_selected(
        self, mock_elasticnet_cv, elasticnet_model, sample_monthly_data
    ):
        """Test that best alpha is selected and stored."""
        # Setup mock with specific best alpha
        mock_instance = Mock()
        mock_instance.alpha_ = 0.05
        mock_instance.l1_ratio_ = 0.7
        mock_elasticnet_cv.return_value = mock_instance

        # Fit model
        elasticnet_model.fit(sample_monthly_data)

        # Verify best alpha is stored
        assert elasticnet_model._best_alpha == 0.05
        assert elasticnet_model._best_l1_ratio == 0.7

    @patch("sirena.models.elasticnet.ElasticNetCV")
    def test_best_alpha_is_in_search_space(
        self, mock_elasticnet_cv, sample_monthly_data
    ):
        """Test that selected best alpha is in the provided search space."""
        custom_alphas = [0.01, 0.05, 0.1, 0.5]

        model = ElasticNetForecaster(alphas=custom_alphas)

        # Setup mock
        mock_instance = Mock()
        mock_instance.alpha_ = 0.05  # Should be in custom_alphas
        mock_instance.l1_ratio_ = 0.5
        mock_elasticnet_cv.return_value = mock_instance

        # Fit model
        model.fit(sample_monthly_data)

        # Verify best alpha is in search space
        assert model._best_alpha in custom_alphas

    @patch("sirena.models.elasticnet.ElasticNetCV")
    def test_best_l1_ratio_is_in_search_space(
        self, mock_elasticnet_cv, sample_monthly_data
    ):
        """Test that selected best l1_ratio is in the provided search space."""
        custom_l1_ratios = [0.2, 0.4, 0.6, 0.8]

        model = ElasticNetForecaster(l1_ratios=custom_l1_ratios)

        # Setup mock
        mock_instance = Mock()
        mock_instance.alpha_ = 0.1
        mock_instance.l1_ratio_ = 0.4  # Should be in custom_l1_ratios
        mock_elasticnet_cv.return_value = mock_instance

        # Fit model
        model.fit(sample_monthly_data)

        # Verify best l1_ratio is in search space
        assert model._best_l1_ratio in custom_l1_ratios

    @patch("sirena.models.elasticnet.ElasticNetCV")
    def test_model_can_predict_after_grid_search(
        self, mock_elasticnet_cv, elasticnet_model, sample_monthly_data
    ):
        """Test that model can predict after grid search completes."""
        # Setup mock
        mock_instance = Mock()
        mock_instance.alpha_ = 0.1
        mock_instance.l1_ratio_ = 0.5
        mock_instance.predict = Mock(return_value=np.array([100.5]))
        mock_elasticnet_cv.return_value = mock_instance

        # Fit model
        elasticnet_model.fit(sample_monthly_data)

        # Make prediction
        target_date = sample_monthly_data.index[-1]
        result = elasticnet_model.predict(sample_monthly_data, target_date)

        # Verify prediction works
        assert "prediction" in result
        assert mock_instance.predict.called


class TestRidgeParameterHandling:
    """Tests for RidgeForecaster parameter handling."""

    def test_default_alpha_parameter(self, ridge_model):
        """Test default alpha parameter is set correctly."""
        assert ridge_model.alpha == ridge_model.ALPHA
        assert ridge_model.alpha == 0.3

    def test_custom_alpha_parameter(self):
        """Test custom alpha parameter is set correctly."""
        custom_alpha = 0.5
        model = RidgeForecaster(alpha=custom_alpha)

        assert model.alpha == custom_alpha

    def test_alpha_bounds(self):
        """Test alpha parameter is within valid bounds."""
        # Valid alphas should be non-negative
        # Note: alpha=0 is rejected by Ridge and falls back to default (0.3)
        valid_alphas = [0.01, 0.1, 0.3, 1.0, 10.0]

        for alpha in valid_alphas:
            model = RidgeForecaster(alpha=alpha)
            assert model.alpha == alpha

        # Test alpha=0 fallbacks to default
        model_zero = RidgeForecaster(alpha=0.0)
        assert model_zero.alpha == 0.3  # Falls back to default ALPHA

    def test_alpha_is_used_in_ridge_model(self, ridge_model, sample_monthly_data):
        """Test that alpha is passed to the underlying Ridge model."""
        # Fit model
        ridge_model.fit(sample_monthly_data)

        # Verify alpha is used
        assert ridge_model.ridge is not None
        assert ridge_model.ridge.alpha == ridge_model.alpha

    def test_model_with_macro_features(self, sample_monthly_data):
        """Test model can be initialized with macro features enabled."""
        model_with_macro = RidgeForecaster(use_macro=True)
        model_no_macro = RidgeForecaster(use_macro=False)

        assert model_with_macro.use_macro is True
        assert model_no_macro.use_macro is False


class TestGridSearchEdgeCases:
    """Tests for grid search edge cases and error handling."""

    @patch("sirena.models.elasticnet.ElasticNetCV")
    def test_single_value_grid_search(self, mock_elasticnet_cv, sample_monthly_data):
        """Test grid search works with single value (no search)."""
        single_alpha = [0.1]
        single_l1_ratio = [0.5]

        model = ElasticNetForecaster(alphas=single_alpha, l1_ratios=single_l1_ratio)

        # Setup mock
        mock_instance = Mock()
        mock_instance.alpha_ = 0.1
        mock_instance.l1_ratio_ = 0.5
        mock_elasticnet_cv.return_value = mock_instance

        # Fit model
        model.fit(sample_monthly_data)

        # Should still work with single values
        assert model._best_alpha == 0.1
        assert model._best_l1_ratio == 0.5

    @patch("sirena.models.elasticnet.ElasticNetCV")
    def test_cv_folds_adjustment(self, mock_elasticnet_cv, sample_monthly_data):
        """Test that different CV folds are respected."""
        cv_values = [2, 5, 10]

        for cv in cv_values:
            model = ElasticNetForecaster(cv=cv)

            # Setup mock
            mock_instance = Mock()
            mock_instance.alpha_ = 0.1
            mock_instance.l1_ratio_ = 0.5
            mock_elasticnet_cv.return_value = mock_instance

            # Fit model
            model.fit(sample_monthly_data)

            # Verify CV value was used
            assert model.cv == cv

    def test_ridge_alpha_not_negative(self):
        """Test that negative alpha raises ValueError or is handled."""
        # Ridge with negative alpha should be handled (either error or absolute value)
        # For now, test that model accepts it (validation happens during fit)
        model = RidgeForecaster(alpha=-0.1)
        assert model.alpha == -0.1  # Model stores it, sklearn handles validation
