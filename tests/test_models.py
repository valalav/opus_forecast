"""
Тесты моделей прогнозирования
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestRidgeForecaster:
    """Тесты модели RidgeForecaster."""

    @pytest.fixture
    def sample_data(self):
        """Генерация тестовых данных."""
        dates = pd.date_range('2020-01-01', periods=60, freq='MS')
        np.random.seed(42)

        data = pd.DataFrame({
            'Все товары и услуги': 100.5 + np.random.randn(60) * 0.3,
            'Продовольственные товары': 100.6 + np.random.randn(60) * 0.4,
            'Непродовольственные товары': 100.3 + np.random.randn(60) * 0.2,
            'Услуги': 100.4 + np.random.randn(60) * 0.3
        }, index=dates)

        return data

    def test_model_import(self):
        """Проверка импорта модели."""
        from sirena.models.ridge import RidgeForecaster

        model = RidgeForecaster(use_macro=False)
        assert model is not None

    def test_model_parameters(self):
        """Проверка параметров модели."""
        from sirena.models.ridge import RidgeForecaster

        model = RidgeForecaster(use_macro=False)

        assert model.ALPHA == 0.3
        assert 2022 in model.OUTLIER_YEARS
        assert len(model.ETS_WEIGHTS) == 12
        assert len(model.FEATURES) == 11

    def test_fit(self, sample_data):
        """Тест обучения модели."""
        from sirena.models.ridge import RidgeForecaster

        model = RidgeForecaster(use_macro=False)
        model.fit(sample_data, 'Все товары и услуги')

        assert model.ridge is not None
        assert model.seasonal_norm is not None
        assert len(model.seasonal_norm) == 12

    def test_predict(self, sample_data):
        """Тест прогнозирования."""
        from sirena.models.ridge import RidgeForecaster

        model = RidgeForecaster(use_macro=False)
        model.fit(sample_data, 'Все товары и услуги')

        # Прогноз на последнюю дату
        target_date = sample_data.index[-1]
        result = model.predict(sample_data, target_date)

        assert 'prediction' in result
        assert 'pred_ridge' in result
        assert 'pred_ets' in result
        assert 99 < result['prediction'] < 102  # Разумный диапазон

    def test_backtest(self, sample_data):
        """Тест бэктестирования."""
        from sirena.models.ridge import RidgeForecaster

        model = RidgeForecaster(use_macro=False)
        results = model.backtest(sample_data, start_date='2023-01-01')

        assert isinstance(results, pd.DataFrame)
        if not results.empty:
            assert 'date' in results.columns
            assert 'actual' in results.columns
            assert 'prediction' in results.columns
            assert 'error' in results.columns

    def test_metrics(self, sample_data):
        """Тест расчёта метрик."""
        from sirena.models.ridge import RidgeForecaster

        model = RidgeForecaster(use_macro=False)
        results = model.backtest(sample_data, start_date='2023-01-01')

        if not results.empty:
            metrics = model.get_metrics(results)

            assert 'MAE' in metrics
            assert 'RMSE' in metrics
            assert 'KPI' in metrics
            assert metrics['MAE'] >= 0
            assert metrics['RMSE'] >= 0


class TestBayesianVAR:
    """Тесты BVAR модели."""

    @pytest.fixture
    def sample_bvar_data(self):
        """Генерация тестовых данных для BVAR."""
        dates = pd.date_range('2020-01-01', periods=60, freq='MS')
        np.random.seed(42)

        data = pd.DataFrame({
            'CPI': np.random.randn(60) * 0.3,
            'Food': np.random.randn(60) * 0.4,
            'USD': np.random.randn(60) * 1.5,
            'RUONIA': 15 + np.random.randn(60) * 0.5
        }, index=dates)

        return data

    def test_bvar_import(self):
        """Проверка импорта BVAR."""
        from sirena.models.bvar import BVARForecaster

        assert BVARForecaster is not None

    def test_bvar_fit(self, sample_bvar_data):
        """Тест обучения BVAR."""
        from sirena.models.bvar import BVARForecaster

        model: BVARForecaster = BVARForecaster(
            lags=4,
            lambda1=1.0,
            var_names=['CPI', 'Food', 'USD', 'RUONIA']
        )
        model.fit(sample_bvar_data, target_col='CPI')

        assert model.B_post is not None

    def test_bvar_forecast(self, sample_bvar_data):
        """Тест прогнозирования BVAR."""
        from sirena.models.bvar import BVARForecaster

        model: BVARForecaster = BVARForecaster(
            lags=4,
            lambda1=1.0,
            var_names=['CPI', 'Food', 'USD', 'RUONIA']
        )
        model.fit(sample_bvar_data, target_col='CPI')
        fc = model.forecast_full(horizon=12)

        assert 'median' in fc
        assert fc['median'].shape[0] == 12
        assert fc['median'].shape[1] == 4  # 4 переменные

    def test_bvar_forecast_is_seed_reproducible(self, sample_bvar_data):
        """A fixed seed yields an identical posterior point trajectory."""
        from sirena.models.bvar import BVARForecaster

        kwargs = {
            'lags': 4,
            'lambda1': 1.0,
            'n_draws': 50,
            'random_state': 42,
            'var_names': ['CPI', 'Food', 'USD', 'RUONIA'],
        }
        first = BVARForecaster(**kwargs).fit(sample_bvar_data, target_col='CPI')
        second = BVARForecaster(**kwargs).fit(sample_bvar_data, target_col='CPI')

        np.testing.assert_array_equal(
            first.forecast(horizon=3),
            second.forecast(horizon=3),
        )

    def test_bvar_backtest_forwards_random_state(
        self, sample_bvar_data, monkeypatch
    ):
        """Each rolling-cutoff model receives the caller's random seed."""
        import sirena.models.bvar as bvar_module

        captured_seeds = []

        class SpyBVAR:
            def __init__(self, **kwargs):
                captured_seeds.append(kwargs['random_state'])

            def fit(self, df, target_col):
                return self

            def forecast(self, horizon):
                return np.zeros(horizon)

        model = bvar_module.BVARForecaster(
            lags=4,
            random_state=314159,
            var_names=['CPI', 'Food', 'USD', 'RUONIA'],
        )
        monkeypatch.setattr(bvar_module, 'BVARForecaster', SpyBVAR)

        model.backtest(sample_bvar_data, start_date='2022-01-01', target_col='CPI')

        assert captured_seeds
        assert set(captured_seeds) == {314159}


class TestSirenaARIMA:
    """Тесты ARIMA модели."""

    @pytest.fixture
    def sample_ts(self):
        """Генерация временного ряда."""
        dates = pd.date_range('2020-01-01', periods=48, freq='MS')
        np.random.seed(42)

        ts = pd.Series(
            0.5 + np.random.randn(48) * 0.3,
            index=dates
        )
        return ts

    def test_arima_import(self):
        """Проверка импорта ARIMA."""
        from sirena.models.arima import AR1Forecaster, SARIMAForecaster

        assert SARIMAForecaster is not None
        assert AR1Forecaster is not None

    def test_ar1_fit(self, sample_ts):
        """Тест AR(1)."""
        from sirena.models.arima import AR1Forecaster

        train_df = pd.DataFrame({'Все товары и услуги': sample_ts})
        model: AR1Forecaster = AR1Forecaster()
        model.fit(train_df, 'Все товары и услуги')

        assert model.fit_result is not None

    def test_sarima_fit(self, sample_ts):
        """Тест SARIMA."""
        from sirena.models.arima import SARIMAForecaster

        train_df = pd.DataFrame({'Все товары и услуги': sample_ts})
        model: SARIMAForecaster = SARIMAForecaster()
        model.fit(train_df, 'Все товары и услуги')

        assert model.fit_result is not None

    def test_forecast(self, sample_ts):
        """Тест прогноза ARIMA."""
        from sirena.models.arima import SARIMAForecaster

        train_df = pd.DataFrame({'Все товары и услуги': sample_ts})
        model: SARIMAForecaster = SARIMAForecaster()
        model.fit(train_df, 'Все товары и услуги')
        fc = model.forecast_with_intervals(horizon=12)

        assert 'mean' in fc
        assert len(fc['mean']) == 12
