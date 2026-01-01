"""
СИРЕНА-КБР: Система Интеллектуального Регионального Анализа
==========================================================

Платформа прогнозирования инфляции в Кабардино-Балкарской Республике.

v5.0 Unified Architecture:
    - SIRENA Unified API — единый интерфейс для всех функций
    - Ki Trajectory Forecaster — эндогенная ставка по правилу Тейлора
    - Regime Detector — определение режима экономики
    - Production Proxy Features — индикаторы спроса (Torg, pp)

v4.8 Ансамбль (9 моделей, веса на основе h=1 backtest MAE):
    - Huber: 18% (лучшая h=1, MAE 0.289)
    - RidgeShockDummies: 17% (MAE 0.299)
    - ElasticNet: 17% (MAE 0.301)
    - NGBoostShock: 16% (лучшая h=2)
    - NGBoost: 12%
    - Ridge: 8% (baseline)
    - RidgeExtended: 5%
    - Prophet: 4% (лучшая h=12)
    - EBM: 3%

Быстрый старт (v5.0):
    from sirena import SIRENA

    # Создание и прогноз
    sirena = SIRENA()
    sirena.load_data().fit()

    # Текущий режим
    print(sirena.regime)  # 🔴 Шок

    # Прогноз с авто-траекторией Ki
    fc = sirena.forecast(12, scenario='auto')
    print(fc.summary())

    # Все сценарии
    df = sirena.quick_forecast(12)

Классический API:
    from sirena.models import ModelRegistry

    model = ModelRegistry.get("huber")
    model.fit(df)
    forecast = model.predict(df, target_date)
"""

__version__ = "5.0.0"
__author__ = "СИРЕНА Team"

# Конфигурация и загрузка данных
from .config import Config
from .data_loader import DataLoader

# Модели
from .models import (
    BaseForecaster,
    ForecastResult,
    ModelRegistry,
    RidgeForecaster,
    RidgeExtendedForecaster,
    RidgeShockDummiesForecaster,
    HuberForecaster,
    ElasticNetForecaster,
    ProphetForecaster,
    EBMForecaster,
    create_ebm_model,
    # Auxiliary (not in ensemble)
    BVARForecaster,
    ETSForecaster,
    SARIMAForecaster,
    LightGBMForecaster,
)

# Ансамбль
from .forecast import EnsembleForecaster

# Производительность
from .async_runner import AsyncModelRunner, run_ensemble_async, run_ensemble_parallel
from .cache import ForecastCache, get_cache, clear_cache

# Макро-признаки (v4.0.1)
from .macro_features import (
    add_macro_features,
    get_best_macro_features,
    get_minimal_macro_features,
    MACRO_FEATURES_BEST,
    MACRO_FEATURES_MINIMAL,
    # v5.0: Production Proxy Features
    load_production_proxies,
    add_production_features,
    PRODUCTION_FEATURES,
)

# v5.0: Unified API
from .unified_api import SIRENA, create_sirena, ForecastResult as UnifiedForecastResult, RegimeInfo

# v5.0: Phase 4 Models
from .models import (
    # Ki Trajectory
    KiTrajectoryForecaster,
    TaylorRuleParams,
    # Unified Subcomponent
    UnifiedSubcomponentForecaster,
    # Regime Detector
    MacroRegime,
    detect_regime,
    get_regime_lags,
    get_regime_history,
)

__all__ = [
    # Версия
    '__version__',
    '__author__',

    # Конфигурация
    'Config',
    'DataLoader',

    # Базовые классы
    'BaseForecaster',
    'ForecastResult',
    'ModelRegistry',

    # Production Models (v4.8)
    'RidgeForecaster',
    'RidgeExtendedForecaster',
    'RidgeShockDummiesForecaster',
    'HuberForecaster',
    'ElasticNetForecaster',
    'ProphetForecaster',
    'EBMForecaster',
    'create_ebm_model',

    # Auxiliary Models
    'BVARForecaster',
    'ETSForecaster',
    'SARIMAForecaster',
    'LightGBMForecaster',

    # Ансамбль
    'EnsembleForecaster',

    # Производительность
    'AsyncModelRunner',
    'run_ensemble_async',
    'run_ensemble_parallel',
    'ForecastCache',
    'get_cache',
    'clear_cache',

    # Макро-признаки
    'add_macro_features',
    'get_best_macro_features',
    'get_minimal_macro_features',
    'MACRO_FEATURES_BEST',
    'MACRO_FEATURES_MINIMAL',

    # v5.0: Production Proxy Features
    'load_production_proxies',
    'add_production_features',
    'PRODUCTION_FEATURES',

    # v5.0: Unified API
    'SIRENA',
    'create_sirena',
    'UnifiedForecastResult',
    'RegimeInfo',

    # v5.0: Phase 4 Models
    'KiTrajectoryForecaster',
    'TaylorRuleParams',
    'UnifiedSubcomponentForecaster',
    'MacroRegime',
    'detect_regime',
    'get_regime_lags',
    'get_regime_history',
]
