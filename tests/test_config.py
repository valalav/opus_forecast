"""
Тесты модуля конфигурации
"""

import pytest
import json
import tempfile
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from sirena.config import Config, ModelConfig, EnsembleConfig


class TestModelConfig:
    """Тесты ModelConfig."""

    def test_default_values(self):
        """Проверка значений по умолчанию."""
        config = ModelConfig()

        assert config.ridge_alpha == 0.3
        assert config.horizon_months == 12
        assert 2022 in config.outlier_years
        assert 2010 in config.outlier_years

    def test_ets_weights(self):
        """Проверка весов ETS для всех месяцев."""
        config = ModelConfig()

        assert len(config.ets_weights) == 12
        for month in range(1, 13):
            assert month in config.ets_weights
            assert 0 <= config.ets_weights[month] <= 1

    def test_volatile_months_high_ets(self):
        """Волатильные месяцы должны иметь высокий вес ETS."""
        config = ModelConfig()

        # Январь, Май, Сентябрь, Октябрь - волатильные
        assert config.ets_weights[1] >= 0.9  # Январь
        assert config.ets_weights[5] >= 0.9  # Май
        assert config.ets_weights[9] >= 0.9  # Сентябрь
        assert config.ets_weights[10] >= 0.9  # Октябрь


class TestEnsembleConfig:
    """Тесты EnsembleConfig."""

    def test_default_weights(self):
        """Проверка весов ансамбля по умолчанию."""
        config = EnsembleConfig()

        assert config.ridge_weight == 0.6
        assert config.bvar_weight == 0.3
        assert config.sarima_weight == 0.1

    def test_weights_sum_to_one(self):
        """Сумма весов должна равняться 1."""
        config = EnsembleConfig()

        assert config.validate() is True

    def test_invalid_weights(self):
        """Некорректные веса не проходят валидацию."""
        config = EnsembleConfig(
            ridge_weight=0.5,
            bvar_weight=0.3,
            sarima_weight=0.1
        )

        assert config.validate() is False


class TestConfig:
    """Тесты главного класса Config."""

    def test_default_initialization(self):
        """Инициализация без файла конфигурации."""
        config = Config()

        assert config.model is not None
        assert config.ensemble is not None
        assert config.data is not None

    def test_save_and_load(self):
        """Сохранение и загрузка конфигурации."""
        config = Config()
        config.model.ridge_alpha = 0.5
        config.model.horizon_months = 24

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            config.save(f.name)

            # Загрузка
            loaded = Config(f.name)

            assert loaded.model.ridge_alpha == 0.5
            assert loaded.model.horizon_months == 24

    def test_data_paths(self):
        """Проверка путей к данным."""
        config = Config()

        assert config.data.inflation_data.name == "inflation_data.csv"
        assert config.data.kbr_data.name == "infl_kbr.csv"
        assert config.data.weekly_prices.name == "weekly_prices.csv"
