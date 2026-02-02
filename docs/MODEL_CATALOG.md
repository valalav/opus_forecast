# Каталог моделей СИРЕНА-КБР v5.2

## Production Models (Ансамбль)

Веса на основе h=1 backtest MAE:

| Модель | Вес | MAE h=1 | Файл |
|--------|-----|---------|------|
| Huber | 18% | 0.288 | `models/huber.py` |
| RidgeShockDummies | 17% | 0.299 | `models/ridge_shock_dummies.py` |
| ElasticNet | 17% | 0.301 | `models/elasticnet.py` |
| NGBoostShock | 16% | 0.291 | `models/ngboost_shock.py` |
| NGBoost | 12% | 0.312 | `models/ngboost_model.py` |
| Ridge | 8% | 0.290 | `models/ridge.py` |
| RidgeExtended | 5% | 0.318 | `models/ridge_extended.py` |
| Prophet | 4% | 0.277* | `models/prophet.py` |
| EBM | 3% | 0.340 | `models/ebm.py` |

*Prophet лучшая на h=12

## Auxiliary Models

| Модель | Назначение | Файл |
|--------|------------|------|
| Micro | 497 микрокомпонентов | `models/microcomponent.py` |
| HorizonEnsemble | Адаптивный Huber+Micro | `models/horizon_ensemble.py` |
| Subcomponent | Bottom-up 3 компонента | `models/subcomponent.py` |
| SubcomponentMulti | Multi-horizon | `models/subcomponent_multi.py` |
| Nowcast | Недельные данные | `scripts/precompute_forecasts.py` |

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

## Примеры использования

### Ridge (baseline)

```python
from sirena.models import RidgeForecaster

model = RidgeForecaster()
model.fit(df)
fc = model.forecast(horizon=12)
```

### Huber (робастная)

```python
from sirena.models import HuberForecaster

model = HuberForecaster(epsilon=1.35)
model.fit(df)
pred = model.predict(df_ext, target_date)
```

### NGBoost (вероятностная)

```python
from sirena.models import NGBoostForecaster

model = NGBoostForecaster(n_estimators=200)
model.fit(df)
fc = model.forecast(horizon=12)
# Возвращает mean, можно получить CI
```

### Prophet (сезонность)

```python
from sirena.models import ProphetForecaster

model = ProphetForecaster()
model.fit(df)
fc = model.forecast(horizon=12)  # Лучшая на h=12
```

## Регистр моделей

Все экспортируемые модели в `sirena/models/__init__.py`:

```python
from .ridge import RidgeForecaster
from .huber import HuberForecaster
from .elasticnet import ElasticNetForecaster
from .ngboost_model import NGBoostForecaster
from .ngboost_shock import NGBoostShockForecaster
from .prophet import ProphetForecaster
from .ebm import EBMForecaster
# ... и т.д.
```
