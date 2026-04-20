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


class TestSirenaKBR:
    """Тесты модели Ridge (SirenaKBR_v24)."""

    @pytest.fixture
    def sample_data(self):
        """Генерация тестовых данных."""
        dates = pd.date_range('2020-01-01', periods=48, freq='MS')
        np.random.seed(42)

        data = pd.DataFrame({
            'Все товары и услуги': 100.5 + np.random.randn(48) * 0.3,
            'Продовольственные товары': 100.6 + np.random.randn(48) * 0.4,
            'Непродовольственные товары': 100.3 + np.random.randn(48) * 0.2,
            'Услуги': 100.4 + np.random.randn(48) * 0.3
        }, index=dates)

        return data

    def test_model_import(self):
        """Проверка импорта модели."""
        from sirena_kbr_v2_4_auto import SirenaKBR_v24

        model = SirenaKBR_v24()
        assert model is not None

    def test_model_parameters(self):
        """Проверка параметров модели."""
        from sirena_kbr_v2_4_auto import SirenaKBR_v24

        model = SirenaKBR_v24()

        assert model.ALPHA == 0.3
        assert 2022 in model.OUTLIER_YEARS
        assert len(model.ETS_WEIGHTS) == 12
        assert len(model.feature_cols) == 11

    def test_fit(self, sample_data):
        """Тест обучения модели."""
        from sirena_kbr_v2_4_auto import SirenaKBR_v24

        model = SirenaKBR_v24()
        model.fit(sample_data)

        assert model.ridge is not None
        assert model.seasonal_norm is not None
        assert len(model.seasonal_norm) == 12

    def test_predict(self, sample_data):
        """Тест прогнозирования."""
        from sirena_kbr_v2_4_auto import SirenaKBR_v24

        model = SirenaKBR_v24()
        model.fit(sample_data)

        # Прогноз на последнюю дату
        target_date = sample_data.index[-1]
        result = model.predict(sample_data, target_date)

        assert 'prediction' in result
        assert 'pred_ridge' in result
        assert 'pred_ets' in result
        assert 99 < result['prediction'] < 102  # Разумный диапазон

    def test_backtest(self, sample_data):
        """Тест бэктестирования."""
        from sirena_kbr_v2_4_auto import SirenaKBR_v24

        model = SirenaKBR_v24()
        results = model.backtest(sample_data, start_date='2023-01-01')

        assert isinstance(results, pd.DataFrame)
        if not results.empty:
            assert 'date' in results.columns
            assert 'actual' in results.columns
            assert 'prediction' in results.columns
            assert 'error' in results.columns

    def test_metrics(self, sample_data):
        """Тест расчёта метрик."""
        from sirena_kbr_v2_4_auto import SirenaKBR_v24

        model = SirenaKBR_v24()
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
        from sirena_arima import SirenaARIMA

        model = SirenaARIMA()
        assert model is not None

    def test_ar1_fit(self, sample_ts):
        """Тест AR(1)."""
        from sirena_arima import SirenaARIMA

        model = SirenaARIMA()
        model.fit_ar1(sample_ts)

        assert model.fit_res is not None

    def test_sarima_fit(self, sample_ts):
        """Тест SARIMA."""
        from sirena_arima import SirenaARIMA

        model = SirenaARIMA()
        model.fit_sarima(sample_ts)

        assert model.fit_res is not None

    def test_forecast(self, sample_ts):
        """Тест прогноза ARIMA."""
        from sirena_arima import SirenaARIMA

        model = SirenaARIMA()
        model.fit_sarima(sample_ts)
        fc = model.forecast(steps=12)

        assert 'mean' in fc
        assert len(fc['mean']) == 12
