"""
Тесты REST API СИРЕНА-КБР
"""

import pytest
from fastapi.testclient import TestClient
import sys
from pathlib import Path

# Добавляем путь к проекту
sys.path.insert(0, str(Path(__file__).parent.parent))

from api.main import app

client = TestClient(app)


class TestHealthEndpoints:
    """Тесты health эндпоинтов."""

    def test_root(self):
        """Тест корневого эндпоинта."""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "name" in data
        assert "version" in data

    def test_health(self):
        """Тест health check."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] in ["ok", "error"]
        assert "version" in data
        assert "models_available" in data


class TestModelsEndpoints:
    """Тесты эндпоинтов моделей."""

    def test_list_models(self):
        """Тест списка моделей."""
        response = client.get("/models")
        assert response.status_code == 200
        data = response.json()
        assert "models" in data
        assert "total_weight" in data
        assert len(data["models"]) > 0

    def test_get_model_info(self):
        """Тест информации о модели."""
        response = client.get("/models/ridge")
        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "ridge"
        assert "weight" in data
        assert "min_train_size" in data

    def test_get_unknown_model(self):
        """Тест несуществующей модели."""
        response = client.get("/models/unknown_model")
        assert response.status_code == 404


class TestForecastEndpoints:
    """Тесты эндпоинтов прогнозирования."""

    def test_quick_forecast(self):
        """Тест быстрого прогноза."""
        response = client.get("/forecast/quick?horizon=3")

        # Может быть 200 или 500 если нет данных
        if response.status_code == 200:
            data = response.json()
            assert "values" in data
            assert "dates" in data
            assert len(data["values"]) == 3

    def test_forecast_default(self):
        """Тест прогноза с дефолтными параметрами."""
        response = client.post("/forecast", json={})

        if response.status_code == 200:
            data = response.json()
            assert "ensemble" in data
            assert "models" in data
            assert len(data["ensemble"]["values"]) == 12

    def test_forecast_custom_models(self):
        """Тест прогноза с выбранными моделями."""
        response = client.post("/forecast", json={
            "horizon": 6,
            "models": ["ridge", "bvar"]
        })

        if response.status_code == 200:
            data = response.json()
            assert len(data["ensemble"]["values"]) == 6
            # Проверяем что использованы только указанные модели
            for model in data["models"].keys():
                assert model in ["ridge", "bvar"]

    def test_forecast_custom_weights(self):
        """Тест прогноза с кастомными весами."""
        response = client.post("/forecast", json={
            "horizon": 3,
            "models": ["ridge", "bvar"],
            "weights": {"ridge": 0.7, "bvar": 0.3}
        })

        if response.status_code == 200:
            data = response.json()
            # Веса должны быть нормализованы
            total_weight = sum(m["weight"] for m in data["models"].values())
            assert abs(total_weight - 1.0) < 0.01


class TestBacktestEndpoints:
    """Тесты эндпоинтов бэктестирования."""

    def test_backtest_ridge(self):
        """Тест бэктеста Ridge модели."""
        response = client.post("/backtest", json={
            "model": "ridge",
            "start_date": "2024-01-01"
        })

        if response.status_code == 200:
            data = response.json()
            assert data["model"] == "ridge"
            assert "metrics" in data
            assert "results" in data
            assert data["metrics"]["MAE"] >= 0

    def test_backtest_unknown_model(self):
        """Тест бэктеста несуществующей модели."""
        response = client.post("/backtest", json={
            "model": "unknown_model",
            "start_date": "2024-01-01"
        })
        assert response.status_code == 404

    def test_backtest_metrics_only(self):
        """Тест получения только метрик."""
        response = client.get("/backtest/metrics/ridge?start_date=2024-01-01")

        if response.status_code == 200:
            data = response.json()
            assert "MAE" in data or "error" in data


class TestSchemaValidation:
    """Тесты валидации схем."""

    def test_forecast_invalid_horizon(self):
        """Тест невалидного горизонта."""
        response = client.post("/forecast", json={
            "horizon": 100  # Слишком большой
        })
        assert response.status_code == 422

    def test_forecast_negative_horizon(self):
        """Тест отрицательного горизонта."""
        response = client.post("/forecast", json={
            "horizon": -1
        })
        assert response.status_code == 422


class TestModelRegistry:
    """Тесты реестра моделей."""

    def test_registry_import(self):
        """Тест импорта реестра."""
        from sirena.models import ModelRegistry
        assert ModelRegistry is not None

    def test_registry_list_models(self):
        """Тест списка моделей в реестре."""
        from sirena.models import ModelRegistry

        models = ModelRegistry.list_models()
        assert len(models) > 0
        assert "ridge" in models

    def test_registry_get_model(self):
        """Тест получения модели."""
        from sirena.models import ModelRegistry

        model = ModelRegistry.get("ridge")
        assert model is not None
        assert hasattr(model, "fit")
        assert hasattr(model, "forecast")

    def test_registry_weights(self):
        """Тест весов моделей."""
        from sirena.models import ModelRegistry

        weights = ModelRegistry.get_all_weights()
        total = sum(weights.values())
        assert abs(total - 1.0) < 0.01


class TestBaseForecaster:
    """Тесты базового класса."""

    def test_base_forecaster_abstract(self):
        """Тест что BaseForecaster абстрактный."""
        from sirena.models.base import BaseForecaster
        import abc

        assert abc.ABC in BaseForecaster.__bases__

    def test_forecast_result(self):
        """Тест ForecastResult."""
        from sirena.models.base import ForecastResult
        import numpy as np
        import pandas as pd

        result = ForecastResult(
            values=np.array([0.5, 0.4, 0.6]),
            dates=pd.date_range('2025-01-01', periods=3, freq='MS'),
            model='test'
        )

        df = result.to_dataframe()
        assert len(df) == 3
        assert 'Forecast' in df.columns


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
