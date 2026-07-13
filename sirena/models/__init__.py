"""
Модели прогнозирования СИРЕНА-КБР v4.8
======================================

Production Models (Default Ensemble, based on h=1 backtest MAE):
- Huber: 18% (лучшая h=1, MAE 0.289)
- RidgeShockDummies: 17% (MAE 0.299)
- ElasticNet: 17% (MAE 0.301)
- NGBoostShock: 16% (лучшая h=2, MAE 0.291)
- NGBoost: 12% (MAE 0.312)
- Ridge: 8% (baseline, MAE 0.310)
- RidgeExtended: 5% (MAE 0.318)
- Prophet: 4% (лучшая h=12, MAE 0.277)
- EBM: 3% (MAE 0.340)

Auxiliary Models (not in ensemble, used for analysis):
- SARIMA, ETS, BVAR, BayesianRidge, LightGBM

Removed from ensemble (v4.8):
- BVAR (MAE 0.430 h=1, 5.714 h=12 - catastrophic)
- LMMR (MAE 1.8-3.0 - completely broken)
- ETS (MAE 0.385 - 24% worse than baseline)
- SARIMA (MAE 0.351 - 13% worse than baseline)
- CatBoost (MAE 0.368 - 16.5% worse than baseline)

Использование:
    from sirena.models import ModelRegistry, BaseForecaster

    # Получить модель
    model = ModelRegistry.get("ridge")
    model = ModelRegistry.get("huber")
    model = ModelRegistry.get("ridge_shock")

    # Список моделей
    models = ModelRegistry.list_models()
"""

from .base import BaseForecaster, ForecastResult
from .registry import ModelRegistry, registry

# =============================================================================
# Production Models (9 models in v4.8 ensemble)
# =============================================================================

# Ridge family (core)
from .ridge import RidgeForecaster
from .ridge_extended import RidgeExtendedForecaster
from .ridge_extended_production_proxy import RidgeExtendedProductionProxyForecaster
from .ridge_extended_rolling import RidgeExtendedRollingForecaster
from .ridge_shock_dummies import RidgeShockDummiesForecaster
from .ridge_shock_rolling import RidgeShockRollingForecaster
from .ridge_production_proxy import RidgeProductionProxyForecaster
from .ridge_production_proxy_rolling import RidgeProductionProxyRollingForecaster
from .ridge_asymmetric_erpt_proxy import RidgeAsymmetricERPTProxyForecaster
from .ridge_macro import (
    RidgeMacroForecaster,
)  # Optimal macro features (USD lag 2, Ki lag 6, Brent lag 5)
from .budget_ridge import BudgetRidgeForecaster  # Budget-enhanced Ridge (budget lag 3)

# Regularized models
from .elasticnet import ElasticNetForecaster
from .huber import HuberForecaster
from .huber_rolling import HuberRollingForecaster
from .huber_production_proxy import HuberProductionProxyForecaster

# Probabilistic models
from .prophet import ProphetForecaster
from .exog_prophet import (
    ExogProphetForecaster,
)  # Prophet with exogenous (USD, Brent, Ki)
from .ebm import EBMForecaster, create_ebm_model

# NGBoost (optional, requires ngboost)
try:
    from .ngboost_model import NGBoostForecaster, NGBOOST_AVAILABLE
except ImportError:
    NGBoostForecaster = None
    NGBOOST_AVAILABLE = False

# NGBoost Shock (uses NGBoost with shock dummies)
try:
    from .ngboost_shock import NGBoostShockForecaster
except ImportError:
    NGBoostShockForecaster = None

# =============================================================================
# Auxiliary Models (not in ensemble, for analysis/comparison)
# =============================================================================

from .arima import SARIMAForecaster, AR1Forecaster
from .ets import ETSForecaster
from .holt_winters import HoltWintersForecaster
from .naive_seasonal import NaiveSeasonalForecaster
from .bvar import BVARForecaster
from .bvar_rate import BVARRateForecaster
from .var_policy import VARPolicyForecaster
from .factor_policy import FactorPolicyForecaster
from .factor_bridge import FactorBridgeForecaster
from .stationary_block_favar import StationaryBlockFAVARForecaster
from .bayesian_ridge import BayesianRidgeForecaster
from .lightgbm import LightGBMForecaster

# CatBoost (optional)
try:
    from .catboost_model import CatBoostForecaster, CATBOOST_AVAILABLE
except ImportError:
    CatBoostForecaster = None
    CATBOOST_AVAILABLE = False

# XGBoost (optional)
try:
    from .xgboost_model import XGBoostForecaster, XGBOOST_AVAILABLE
except ImportError:
    XGBoostForecaster = None
    XGBOOST_AVAILABLE = False

# =============================================================================
# Experimental Models
# =============================================================================

# Subcomponent models (bottom-up with 45 subcomponents)
from .subcomponent import SubcomponentForecaster  # Optimal features (USD, Brent, etc.)
from .subcomponent_multi import (
    SubcomponentMultiForecaster,
)  # Optimal models (Ridge, Prophet, NGBoost)

# Microcomponent model (bottom-up with 537 microcomponents)
from .microcomponent import (
    MicrocomponentForecaster,
)  # Individual Ridge/Voting per micro
from .micro_statsmodels_external import MicroStatsmodelsExternalForecaster

# Hierarchical Microcomponent (full hierarchy: micro → subcomp → comp → total)
from .hierarchical_micro import HierarchicalMicroForecaster  # Hierarchical bottom-up

# Optimized Microcomponent (category-specific models)
from .micro_optimized import (
    MicroOptimizedForecaster,
)  # Huber for stable, Ridge for volatile

# Horizon-adaptive Ensemble (Huber + Micro with adaptive weights)
from .horizon_ensemble import HorizonEnsembleForecaster  # Best of both worlds

# Scenario Models (Rate transmission with calibration)
from .scenario_rate import ScenarioRateModel, TransmissionParams, AsymmetricParams
from .subcomponent_scenario import SubcomponentScenarioForecaster

# Ki Trajectory Model (endogenous rate forecasting)
from .ki_trajectory import KiTrajectoryForecaster, TaylorRuleParams

# Unified Subcomponent Model (integrated baseline + scenario)
from .unified_subcomp import UnifiedSubcomponentForecaster

# MIDAS Model (Mixed Data Sampling)
from .midas import MIDASForecaster
from .stacking_regressor import StackingRegressorForecaster

# Temporal Fusion Transformer (TFT)
from .tft import TemporalFusionForecaster

# Conformal Prediction (calibrated confidence intervals)
from .conformal import ConformalForecaster

# Regime Detector (adaptive lags based on macro regime)
from .regime_detector import (
    MacroRegime,
    RegimeConfig,
    RegimeThresholds,
    REGIME_CONFIGS,
    detect_regime,
    get_regime_lags,
    get_regime_history,
    detect_shock_periods,
    KNOWN_SHOCK_PERIODS,
)

# Weekly Nowcasting models (experimental)
from .volatility_weighted_nowcaster import VolatilityWeightedNowcaster
from .regime_adaptive_nowcaster import RegimeAdaptiveNowcaster

# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Base
    "BaseForecaster",
    "ForecastResult",
    "ModelRegistry",
    "registry",
    # Production Models (v4.8 ensemble)
    "RidgeForecaster",
    "RidgeExtendedForecaster",
    "RidgeExtendedProductionProxyForecaster",
    "RidgeExtendedRollingForecaster",
    "RidgeShockDummiesForecaster",
    "RidgeShockRollingForecaster",
    "RidgeProductionProxyForecaster",
    "RidgeProductionProxyRollingForecaster",
    "RidgeAsymmetricERPTProxyForecaster",
    "RidgeMacroForecaster",
    "BudgetRidgeForecaster",
    "ElasticNetForecaster",
    "HuberForecaster",
    "HuberRollingForecaster",
    "HuberProductionProxyForecaster",
    "ProphetForecaster",
    "ExogProphetForecaster",
    "EBMForecaster",
    "create_ebm_model",
    "NGBoostForecaster",
    "NGBOOST_AVAILABLE",
    "NGBoostShockForecaster",
    # Auxiliary Models
    "SARIMAForecaster",
    "AR1Forecaster",
    "ETSForecaster",
    "NaiveSeasonalForecaster",
    "BVARForecaster",
    "BVARRateForecaster",
    "VARPolicyForecaster",
    "FactorPolicyForecaster",
    "FactorBridgeForecaster",
    "StationaryBlockFAVARForecaster",
    "BayesianRidgeForecaster",
    "LightGBMForecaster",
    "CatBoostForecaster",
    "CATBOOST_AVAILABLE",
    "XGBoostForecaster",
    "XGBOOST_AVAILABLE",
    # Experimental (Subcomponent models)
    "SubcomponentForecaster",
    "SubcomponentMultiForecaster",
    # Experimental (Microcomponent model)
    "MicrocomponentForecaster",
    "MicroStatsmodelsExternalForecaster",
    # Experimental (Hierarchical Microcomponent)
    "HierarchicalMicroForecaster",
    # Experimental (Optimized Microcomponent)
    "MicroOptimizedForecaster",
    # Experimental (Horizon-adaptive Ensemble)
    "HorizonEnsembleForecaster",
    # Scenario Models (Rate transmission)
    "ScenarioRateModel",
    "TransmissionParams",
    "AsymmetricParams",
    "SubcomponentScenarioForecaster",
    # Ki Trajectory (v5.0)
    "KiTrajectoryForecaster",
    "TaylorRuleParams",
    # Unified Subcomponent (v5.0)
    "UnifiedSubcomponentForecaster",
    # MIDAS (Mixed Data Sampling)
    "MIDASForecaster",
    # Temporal Fusion Transformer (TFT)
    "TemporalFusionForecaster",
    # Conformal Prediction (calibrated CI)
    "ConformalForecaster",
    # Weekly Nowcasting models (experimental)
    "VolatilityWeightedNowcaster",
    "RegimeAdaptiveNowcaster",
    # StackingRegressor (meta-model)
    "StackingRegressorForecaster",
]
