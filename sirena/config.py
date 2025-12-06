"""
Конфигурация проекта СИРЕНА-КБР
===============================
"""

import json
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class ModelConfig:
    """Параметры модели Ridge."""
    outlier_years: List[int] = field(default_factory=lambda: [2010, 2022])
    ridge_alpha: float = 0.3
    horizon_months: int = 12

    # Веса ETS по месяцам (0 = только Ridge, 1 = только ETS)
    ets_weights: Dict[int, float] = field(default_factory=lambda: {
        1: 0.9,   # Январь
        2: 0.0,   # Февраль
        3: 0.5,   # Март
        4: 0.3,   # Апрель
        5: 0.9,   # Май
        6: 0.5,   # Июнь
        7: 0.0,   # Июль
        8: 0.5,   # Август
        9: 0.9,   # Сентябрь
        10: 0.9,  # Октябрь
        11: 0.0,  # Ноябрь
        12: 0.0   # Декабрь
    })


@dataclass
class EnsembleConfig:
    """Веса ансамбля моделей."""
    ridge_weight: float = 0.6
    bvar_weight: float = 0.3
    sarima_weight: float = 0.1

    def validate(self) -> bool:
        """Проверка суммы весов = 1."""
        total = self.ridge_weight + self.bvar_weight + self.sarima_weight
        return abs(total - 1.0) < 1e-6


@dataclass
class DataConfig:
    """Пути к данным."""
    data_dir: Path = field(default_factory=lambda: Path("data"))

    @property
    def inflation_data(self) -> Path:
        return self.data_dir / "inflation_data.csv"

    @property
    def kbr_data(self) -> Path:
        return self.data_dir / "infl_kbr.csv"

    @property
    def weekly_prices(self) -> Path:
        return self.data_dir / "weekly_prices.csv"

    @property
    def regional_data(self) -> Path:
        return self.data_dir / "all_regions_indices.csv"


class Config:
    """Главный класс конфигурации."""

    def __init__(self, config_path: Optional[str] = None):
        self.model = ModelConfig()
        self.ensemble = EnsembleConfig()
        self.data = DataConfig()

        if config_path:
            self.load(config_path)

    def load(self, path: str) -> None:
        """Загрузка конфигурации из JSON."""
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        if 'model_params' in data:
            mp = data['model_params']
            if 'outlier_years' in mp:
                self.model.outlier_years = mp['outlier_years']
            if 'ridge_alpha' in mp:
                self.model.ridge_alpha = mp['ridge_alpha']
            if 'horizon_months' in mp:
                self.model.horizon_months = mp['horizon_months']

        if 'ets_weights' in data:
            self.model.ets_weights = {int(k): v for k, v in data['ets_weights'].items()}

        if 'ensemble' in data:
            ens = data['ensemble']
            if 'ridge_weight' in ens:
                self.ensemble.ridge_weight = ens['ridge_weight']
            if 'bvar_weight' in ens:
                self.ensemble.bvar_weight = ens['bvar_weight']
            if 'sarima_weight' in ens:
                self.ensemble.sarima_weight = ens['sarima_weight']

    def save(self, path: str) -> None:
        """Сохранение конфигурации в JSON."""
        data = {
            'model_params': {
                'outlier_years': self.model.outlier_years,
                'ridge_alpha': self.model.ridge_alpha,
                'horizon_months': self.model.horizon_months
            },
            'ets_weights': {str(k): v for k, v in self.model.ets_weights.items()},
            'ensemble': {
                'ridge_weight': self.ensemble.ridge_weight,
                'bvar_weight': self.ensemble.bvar_weight,
                'sarima_weight': self.ensemble.sarima_weight
            }
        }
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4, ensure_ascii=False)


# Глобальный экземпляр конфигурации
_config: Optional[Config] = None


def get_config() -> Config:
    """Получить глобальную конфигурацию."""
    global _config
    if _config is None:
        config_path = Path("config.json")
        if config_path.exists():
            _config = Config(str(config_path))
        else:
            _config = Config()
    return _config
