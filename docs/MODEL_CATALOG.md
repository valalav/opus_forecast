# Каталог моделей СИРЕНА-КБР v5.3

## Production Ensemble (Ансамбль v4.8)

Веса оптимизированы по MAE на горизонте h=1:

| Модель | Вес | MAE h=1 | Файл | Описание |
|--------|-----|---------|------|----------|
| **Huber** | 18% | 0.289 | `sirena/models/huber.py` | Робастная регрессия (Huber loss) |
| **RidgeShockDummies** | 17% | 0.299 | `sirena/models/ridge_shock_dummies.py` | Ridge с шоковыми дамми |
| **ElasticNet** | 17% | 0.301 | `sirena/models/elasticnet.py` | L1+L2 регуляризация |
| **NGBoostShock** | 16% | 0.291 | `sirena/models/ngboost_shock.py` | NGBoost с шоковыми дамми |
| **NGBoost** | 12% | 0.312 | `sirena/models/ngboost_model.py` | Probabilistic gradient boosting |
| **Ridge** | 8% | 0.310 | `sirena/models/ridge.py` | Baseline (Ridge + ETS сезонность) |
| **RidgeExtended** | 5% | 0.318 | `sirena/models/ridge_extended.py` | Ridge с momentum и volatility |
| **Prophet** | 4% | 0.277 | `sirena/models/prophet.py` | Facebook Prophet (лучшая на h=12) |
| **EBM** | 3% | 0.340 | `sirena/models/ebm.py` | Explainable Boosting Machine |

*Примечание: Prophet имеет лучший MAE 0.277 на h=12, но слабее на коротких горизонтах.*

---

## 🏆 Лучшие модели по SIRENA Score

| Ранг | Модель | Score | MAE h=1 | MAE h=2 | MAE h=12 | Файл |
|------|--------|-------|---------|---------|----------|------|
| 🥇 1 | **SubcomponentMulti** | **0.515** | 0.265 | 0.278 | 0.264 | `sirena/models/subcomponent_multi.py` |
| 🥈 2 | EBM | 0.592 | 0.309 | 0.309 | 0.309 | `sirena/models/ebm.py` |
| 🥉 3 | Subcomponent | 0.607 | 0.285 | 0.379 | 0.302 | `sirena/models/subcomponent.py` |

**SIRENA Score формула:** `(0.50×MAE_h1 + 0.30×MAE_h2 + 0.20×MAE_h12) × (2 - KPI_rate)`

---

## Experimental Models (v5.x)

### Subcomponent Models (Bottom-up)

| Модель | Файл | Описание | MAE h=1 |
|--------|------|----------|---------|
| **SubcomponentMulti** | `subcomponent_multi.py` | Multi-horizon bottom-up с Ridge/Prophet/NGBoost | 0.265 |
| **Subcomponent** | `subcomponent.py` | Bottom-up по 3 компонентам (Food, NonFood, Services) | 0.285 |
| **SubcomponentScenario** | `subcomponent_scenario.py` | Интеграция baseline + scenario | — |
| **UnifiedSubcomponent** | `unified_subcomp.py` | Единый интерфейс для subcomponent | — |

### Microcomponent Models (537 микрокомпонентов)

| Модель | Файл | Описание |
|--------|------|----------|
| **Microcomponent** | `microcomponent.py` | Bottom-up по 537 микрокомпонентам (индивидуальные Ridge/Voting) |
| **MicroOptimized** | `micro_optimized.py` | Оптимизированный: Huber для stable, Ridge для volatile |
| **HierarchicalMicro** | `hierarchical_micro.py` | Полная иерархия: micro → subcomp → comp → total |
| **MicroPlodovoshchi** | `micro_plodovoshchi.py` | Специализированная для плодоовощей |
| **MicroARIMA** | `micro_arima.py` | ARIMA для микрокомпонентов (MAE 0.415, SIRENA Score 0.701) |

### Deep Learning & Advanced Models

| Модель | Файл | Описание | Статус |
|--------|------|----------|--------|
| **TFT** | `tft.py` | Temporal Fusion Transformer (attention-based) | Experimental |
| **MIDAS** | `midas.py` | Mixed Data Sampling (высокочастотные данные) | Experimental |
| **LSTM** | `archive/models/lstm_model.py` | Удалена из production (заменена EBM) | Archived |

### Probabilistic & Uncertainty Models

| Модель | Файл | Описание |
|--------|------|----------|
| **NGBoost** | `ngboost_model.py` | Probabilistic gradient boosting |
| **NGBoostShock** | `ngboost_shock.py` | NGBoost с шоковыми дамми |
| **Conformal** | `conformal.py` | Калиброванные доверительные интервалы |
| **BayesianRidge** | `bayesian_ridge.py` | Байесовская Ridge с доверительными интервалами |

### Ridge Family (Extended)

| Модель | Файл | Описание |
|--------|------|----------|
| **Ridge** | `ridge.py` | Baseline: Ridge + ETS сезонность |
| **RidgeExtended** | `ridge_extended.py` | + momentum (d_y_lag1, d_y_lag3), volatility (y_vol3, y_vol6) |
| **RidgeShockDummies** | `ridge_shock_dummies.py` | + шоковые дамми (2022, COVID, etc.) |
| **RidgeMacro** | `ridge_macro.py` | + оптимальные макро-признаки (USD lag 2, Ki lag 6, Brent lag 5) |
| **BudgetRidge** | `budget_ridge.py` | + бюджетные данные (budget lag 3) |

### Scenario & Policy Models

| Модель | Файл | Описание |
|--------|------|----------|
| **KiTrajectory** | `ki_trajectory.py` | Эндогенный прогноз ключевой ставки (Taylor Rule) |
| **ScenarioRate** | `scenario_rate.py` | Модель трансмиссии ставки с калибровкой |
| **RegimeDetector** | `regime_detector.py` | Адаптивные лаги на основе макрорежима (shock/normal/high_inflation) |
| **VARPolicy** | `var_policy.py` | Обязательная VAR-family policy: RegimeMacroVARX для h=1, SeasonalVAR для траектории h=12 |
| **FactorPolicy** | `factor_policy.py` | Обязательная factor-family policy: Robust seasonal FAVAR, 2 PCA-фактора, h=1 MAE 0.371 |
| **FactorBridge** | `factor_bridge.py` | Agent-reviewed block-factor bridge challenger: direct Huber/Ridge equations, best compact h=1 MAE 0.370, not promoted on h=2/h=12 |
| **StationaryBlockFAVAR** | `stationary_block_favar.py` | Diagnostics-aware factor report model: component and monetary PCA blocks on stationary inputs, selected on ADF/KPSS/BG/LB/ARCH gates, h=1 MAE 0.378 |

### Nowcasting Models

| Модель | Файл | Описание |
|--------|------|----------|
| **VolatilityWeightedNowcaster** | `volatility_weighted_nowcaster.py` | Взвешенный по волатильности |
| **RegimeAdaptiveNowcaster** | `regime_adaptive_nowcaster.py` | Режим-адаптивный nowcaster |
| **WeeklyForecaster** | `archive/models/weekly.py` | Архивная версия (в `archive/`) |

### Ensemble & Meta Models

| Модель | Файл | Описание |
|--------|------|----------|
| **HorizonEnsemble** | `horizon_ensemble.py` | Адаптивный Huber + Micro |
| **StackingRegressor** | `stacking_regressor.py` | Meta-learning регрессор |

### Auxiliary Models (не в ансамбле)

| Модель | Файл | Описание |
|--------|------|----------|
| **SARIMA** | `arima.py` | Сезонная ARIMA |
| **AR1** | `arima.py` | Авторегрессия первого порядка |
| **ETS** | `ets.py` | Exponential Smoothing |
| **HoltWinters** | `holt_winters.py` | Holt-Winters сезонность |
| **BVAR** | `bvar.py` | Байесовская VAR (удалена из ансамбля — катастрофический h=12) |
| **BVARRate** | `bvar_rate.py` | BVAR с ключевой ставкой |
| **LightGBM** | `lightgbm.py` | Gradient boosting |
| **CatBoost** | `catboost_model.py` | CatBoost для малых выборок |
| **XGBoost** | `xgboost_model.py` | XGBoost |
| **NaiveSeasonal** | `naive_seasonal.py` | Наивная сезонная модель |

---

## Базовый интерфейс

Все модели наследуются от `BaseForecaster`:

```python
from sirena.models.base import BaseForecaster

class MyModel(BaseForecaster):
    def fit(self, df, target='Все товары и услуги'):
        pass
    
    def predict(self, df, target_date) -> dict:
        return {'prediction': value}
    
    def forecast(self, horizon=12) -> np.ndarray:
        pass
```

---

## Примеры использования

### Huber (лучшая на h=1)

```python
from sirena.models import HuberForecaster

model = HuberForecaster(epsilon=1.35)
model.fit(df)
pred = model.predict(df_ext, target_date)
print(f"Prediction: {pred['prediction']:.2f}%")
```

### SubcomponentMulti (лучшая по SIRENA Score)

```python
from sirena.models import SubcomponentMultiForecaster

model = SubcomponentMultiForecaster()
model.fit(df)
fc = model.forecast(horizon=12)
# Автоматически выбирает оптимальные модели по компонентам
```

### RidgeShockDummies

```python
from sirena.models import RidgeShockDummiesForecaster

model = RidgeShockDummiesForecaster()
model.fit(df)
fc = model.forecast(horizon=12)
# Автоматически учитывает шоковые периоды (2022, COVID, etc.)
```

### NGBoostShock (лучшая на h=2)

```python
from sirena.models import NGBoostShockForecaster

model = NGBoostShockForecaster(n_estimators=200)
model.fit(df)
fc = model.forecast(horizon=12)
# Возвращает mean, можно получить CI
```

### Prophet (лучшая на h=12)

```python
from sirena.models import ProphetForecaster

model = ProphetForecaster()
model.fit(df)
fc = model.forecast(horizon=12)
```

### Conformal Prediction (доверительные интервалы)

```python
from sirena.models import ConformalForecaster

# Обертка над любой моделью
base_model = HuberForecaster()
model = ConformalForecaster(base_model, coverage=0.9)
model.fit(df)
pred = model.predict_with_ci(df, target_date)
# Возвращает prediction, ci_lower, ci_upper
```

---

## Регистр моделей (ModelRegistry)

Все модели регистрируются через декоратор:

```python
from sirena.models import ModelRegistry, BaseForecaster

@ModelRegistry.register("my_model")
class MyModel(BaseForecaster):
    name = "my_model"
    # ...
```

Получить модель:
```python
from sirena.models import ModelRegistry

model = ModelRegistry.get("huber")
model = ModelRegistry.get("subcomponent_multi")

# Список всех моделей
models = ModelRegistry.list_models()
```

---

## Экспортируемые модели

Полный список в `sirena/models/__init__.py`:

```python
from sirena.models import (
    # Production (9 моделей)
    RidgeForecaster,
    RidgeExtendedForecaster,
    RidgeShockDummiesForecaster,
    RidgeMacroForecaster,
    BudgetRidgeForecaster,
    ElasticNetForecaster,
    HuberForecaster,
    ProphetForecaster,
    ExogProphetForecaster,
    EBMForecaster,
    NGBoostForecaster,
    NGBoostShockForecaster,
    
    # Experimental (лучшие)
    SubcomponentMultiForecaster,  # Лучшая по SIRENA Score
    SubcomponentForecaster,
    MicrocomponentForecaster,
    HierarchicalMicroForecaster,
    MicroOptimizedForecaster,
    
    # Advanced
    MIDASForecaster,
    TemporalFusionForecaster,
    ConformalForecaster,
    KiTrajectoryForecaster,
    ScenarioRateModel,
    RegimeDetector,
    
    # Auxiliary
    SARIMAForecaster,
    ETSForecaster,
    BVARForecaster,
    BayesianRidgeForecaster,
    LightGBMForecaster,
    CatBoostForecaster,
    XGBoostForecaster,
)
```

---

## Метрики качества (Backtest 2020-2025)

### Production Ensemble

| Модель | MAE h=1 | MAE h=2 | MAE h=12 | SIRENA Score |
|--------|---------|---------|----------|--------------|
| **Huber** | 0.289 | — | — | — |
| **RidgeShockDummies** | 0.299 | — | — | — |
| **ElasticNet** | 0.301 | — | — | — |
| **NGBoostShock** | 0.291 | 0.291 | — | — |
| **Ridge** | 0.310 | — | 0.338 | — |
| **Prophet** | — | — | **0.277** | — |

### Experimental Models

| Модель | MAE h=1 | MAE h=2 | MAE h=12 | SIRENA Score |
|--------|---------|---------|----------|--------------|
| **SubcomponentMulti** | **0.265** | **0.278** | **0.264** | **0.515** |
| **Subcomponent** | 0.285 | 0.379 | 0.302 | 0.607 |
| **EBM** | 0.309 | 0.309 | 0.309 | 0.592 |

### Удалённые из ансамбля (плохие результаты)

| Модель | MAE h=1 | MAE h=12 | Причина |
|--------|---------|----------|---------|
| BVAR | 0.430 | **5.714** | Катастрофический h=12 |
| LMMR | 1.8-3.0 | — | Полностью сломан |
| ETS | 0.385 | — | +24% vs Ridge |
| SARIMA | 0.351 | — | +13% vs Ridge |
| CatBoost | 0.368 | — | +16.5% vs Ridge |

---

*Последнее обновление: 2 февраля 2026*
