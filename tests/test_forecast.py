"""
Тесты модуля прогнозирования
"""

import importlib.util
import json
import pytest
import numpy as np
import pandas as pd
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from sirena.forecast import (
    EnsembleForecaster,
    calculate_cumulative_inflation,
    calculate_metrics
)


class TestCalculateCumulativeInflation:
    """Тесты расчёта накопленной инфляции."""

    def test_no_inflation(self):
        """Нулевая инфляция (все месяцы = 100)."""
        mom_values = np.array([100.0] * 12)
        result = calculate_cumulative_inflation(mom_values)

        assert abs(result) < 1e-10

    def test_constant_inflation(self):
        """Постоянная инфляция 1% в месяц."""
        mom_values = np.array([101.0] * 12)
        result = calculate_cumulative_inflation(mom_values)

        # (1.01^12 - 1) * 100 ≈ 12.68%
        expected = (1.01 ** 12 - 1) * 100
        assert abs(result - expected) < 0.01

    def test_single_month(self):
        """Один месяц."""
        mom_values = np.array([102.0])
        result = calculate_cumulative_inflation(mom_values)

        assert abs(result - 2.0) < 1e-10

    def test_deflation(self):
        """Дефляция."""
        mom_values = np.array([99.0] * 6)
        result = calculate_cumulative_inflation(mom_values)

        # Должна быть отрицательной
        assert result < 0


class TestCalculateMetrics:
    """Тесты расчёта метрик качества."""

    def test_perfect_forecast(self):
        """Идеальный прогноз (ошибка = 0)."""
        actual = np.array([100.5, 100.3, 100.8])
        predicted = np.array([100.5, 100.3, 100.8])

        metrics = calculate_metrics(actual, predicted)

        assert metrics['MAE'] == 0.0
        assert metrics['RMSE'] == 0.0
        assert metrics['KPI_pct'] == 100.0

    def test_constant_error(self):
        """Постоянная ошибка 0.3."""
        actual = np.array([100.5, 100.3, 100.8])
        predicted = np.array([100.2, 100.0, 100.5])

        metrics = calculate_metrics(actual, predicted)

        assert abs(metrics['MAE'] - 0.3) < 1e-10
        assert metrics['KPI_count'] == 3  # Все ошибки < 0.5

    def test_kpi_threshold(self):
        """Проверка порога KPI (±0.5)."""
        actual = np.array([100.0, 100.0, 100.0, 100.0])
        predicted = np.array([100.4, 100.6, 99.5, 100.0])  # Ошибки: 0.4, 0.6, 0.5, 0

        metrics = calculate_metrics(actual, predicted)

        # Ошибки <= 0.5: 0.4, 0.5, 0 = 3 из 4
        assert metrics['KPI_count'] == 3
        assert metrics['KPI_pct'] == 75.0


class TestEnsembleForecaster:
    """Тесты ансамблевого прогнозирования."""

    def test_default_weights(self):
        """Проверка весов по умолчанию."""
        forecaster = EnsembleForecaster()

        assert forecaster.ridge_weight == 0.6
        assert forecaster.bvar_weight == 0.3
        assert forecaster.sarima_weight == 0.1

    def test_custom_weights(self):
        """Пользовательские веса."""
        forecaster = EnsembleForecaster(
            ridge_weight=0.5,
            bvar_weight=0.3,
            sarima_weight=0.2
        )

        assert forecaster.ridge_weight == 0.5
        assert forecaster.sarima_weight == 0.2

    def test_combine_forecasts(self):
        """Объединение прогнозов."""
        forecaster = EnsembleForecaster()

        ridge = pd.DataFrame({
            'Date': pd.date_range('2025-01-01', periods=3, freq='MS'),
            'MoM': [0.5, 0.6, 0.7]
        })
        bvar = pd.DataFrame({
            'Date': pd.date_range('2025-01-01', periods=3, freq='MS'),
            'BVAR': [0.4, 0.5, 0.6]
        })
        sarima = pd.DataFrame({
            'Date': pd.date_range('2025-01-01', periods=3, freq='MS'),
            'SARIMA': [0.6, 0.7, 0.8]
        })

        ensemble = forecaster.combine_forecasts(ridge, bvar, sarima)

        assert ensemble is not None
        assert len(ensemble) == 3

        # Проверка первого значения: 0.6*0.5 + 0.3*0.4 + 0.1*0.6 = 0.48
        expected_first = 0.6 * 0.5 + 0.3 * 0.4 + 0.1 * 0.6
        assert abs(ensemble[0] - expected_first) < 1e-10

    def test_combine_with_missing_bvar(self):
        """Объединение при отсутствии BVAR."""
        forecaster = EnsembleForecaster()

        ridge = pd.DataFrame({
            'Date': pd.date_range('2025-01-01', periods=3, freq='MS'),
            'MoM': [0.5, 0.6, 0.7]
        })

        ensemble = forecaster.combine_forecasts(ridge, None, None)

        assert ensemble is None


class TestForecastH12ProductionCache:
    """Contracts for the dashboard's production h=12 trajectory."""

    def test_h12_table_uses_cached_production_ensemble(self, monkeypatch):
        """The table exposes the canonical Ensemble, not a live auxiliary mean."""
        page_path = Path(__file__).parent.parent / "pages" / "1_Forecast.py"
        spec = importlib.util.spec_from_file_location("forecast_page", page_path)
        assert spec is not None and spec.loader is not None
        forecast_page = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(forecast_page)

        class FakeColumn:
            def metric(self, *args, **kwargs):
                pass

        class FakeStreamlit:
            def __init__(self):
                self.errors = []
                self.table = None

            def subheader(self, *args, **kwargs):
                pass

            def error(self, message):
                self.errors.append(message)

            def warning(self, *args, **kwargs):
                pass

            def markdown(self, *args, **kwargs):
                pass

            def columns(self, count):
                return [FakeColumn() for _ in range(count)]

            def caption(self, *args, **kwargs):
                pass

            def dataframe(self, styled, **kwargs):
                self.table = styled.data.copy()

            def download_button(self, *args, **kwargs):
                pass

            def plotly_chart(self, *args, **kwargs):
                pass

        fake_st = FakeStreamlit()
        monkeypatch.setattr(forecast_page, "st", fake_st)

        raw = pd.read_csv(
            Path(__file__).parent.parent / "data" / "inflation_data.csv",
            sep=";",
            decimal=",",
            encoding="utf-8-sig",
        )
        raw["Date"] = (
            pd.to_datetime(raw["Date"], format="%d.%m.%Y")
            .dt.to_period("M")
            .dt.to_timestamp()
        )
        df = raw.set_index("Date")[
            ["mom", "Prod", "Nonprod", "Serv"]
        ].rename(
            columns={
                "mom": "Все товары и услуги",
                "Prod": "Продовольственные товары",
                "Nonprod": "Непродовольственные товары",
                "Serv": "Услуги",
            }
        )

        forecast_page.render_forecast_h12_tab(
            df,
            df.index.max(),
            {},
            lambda horizon: None,
        )

        cache = json.loads(
            (
                Path(__file__).parent.parent / "data" / "precomputed_forecasts.json"
            ).read_text(encoding="utf-8")
        )
        assert not fake_st.errors
        assert fake_st.table is not None
        assert "BVAR" not in fake_st.table.columns
        assert "VARPolicy" in fake_st.table.columns
        assert fake_st.table.loc[0, "Ensemble"] == pytest.approx(
            cache["forecasts"]["Ensemble"][0]
        )
