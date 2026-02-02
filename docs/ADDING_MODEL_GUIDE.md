# Добавление новой модели в СИРЕНА-КБР

Полный чеклист для добавления новой модели прогнозирования.

## 11 обязательных мест

| # | Файл | Что добавить |
|---|------|--------------|
| 1 | `sirena/models/{name}.py` | Файл модели (наследуется от BaseForecaster) |
| 2 | `sirena/models/__init__.py` | Импорт + `__all__` |
| 3 | `dashboard.py` ALL_MODELS | `'ModelName'` |
| 4 | `dashboard.py` MODEL_COLORS | `'ModelName': '#color'` |
| 5 | `scripts/backtest_framework.py` | Импорт модели |
| 6 | `scripts/backtest_framework.py` | `def _forecast_{name}()` |
| 7 | `scripts/backtest_framework.py` | Вызов в `_run_rolling` |
| 8 | `scripts/backtest_framework.py` | Вызов в `_run_h12` |
| 9 | `archive/results/backtest_h1_predictions.csv` | Колонка (после бэктеста) |
| 10 | `archive/results/backtest_h2_predictions.csv` | Колонка (после бэктеста) |
| 11 | `archive/results/backtest_h12_predictions.csv` | Колонка (после бэктеста) |

## Шаг 1: Создать файл модели

```python
# sirena/models/new_model.py
from .base import BaseForecaster
import pandas as pd
import numpy as np

class NewModelForecaster(BaseForecaster):
    def __init__(self, **kwargs):
        super().__init__()
        # Инициализация
    
    def fit(self, df: pd.DataFrame, target: str = 'Все товары и услуги'):
        # Обучение модели
        self.is_fitted = True
        return self
    
    def predict(self, df: pd.DataFrame, target_date: pd.Timestamp) -> dict:
        # Прогноз на одну дату
        return {'prediction': value, 'model': 'NewModel'}
    
    def forecast(self, horizon: int = 12) -> np.ndarray:
        # Прогноз на горизонт
        return predictions
```

## Шаг 2: Добавить в __init__.py

```python
# sirena/models/__init__.py
from .new_model import NewModelForecaster

__all__ = [
    # ... existing
    'NewModelForecaster',
]
```

## Шаг 3: Dashboard - Прогноз (tab1)

Найти блок `with st.spinner("Расчет ансамбля моделей..."):`

```python
new_model_df = None
try:
    from sirena.models.new_model import NewModelForecaster
    model = NewModelForecaster()
    model.fit(df)
    vals = []
    for h in range(horizon):
        target_date = last_date + pd.DateOffset(months=h+1)
        df_ext = df.copy()
        df_ext.loc[target_date] = np.nan
        pred = model.predict(df_ext, target_date)['prediction'] - 100
        vals.append(pred)
    new_model_df = pd.DataFrame({
        'Date': pd.date_range(start=last_date + pd.DateOffset(months=1), periods=horizon, freq='MS'),
        'NewModel': vals
    })
except:
    pass
```

## Шаг 4: Dashboard - model_weights

```python
model_weights = {
    'NewModel': (new_model_df['NewModel'].values if new_model_df is not None else None, 0.10),
    # ...
}
```

## Шаг 5: Dashboard - График

```python
if new_model_df is not None:
    fig_fc.add_trace(go.Scatter(
        x=new_model_df['Date'], y=new_model_df['NewModel'],
        name='NewModel', line=dict(color='#hexcolor', width=2)
    ))
```

## Шаг 6: Backtest Framework

```python
# scripts/backtest_framework.py

# 1. Импорт
from sirena.models.new_model import NewModelForecaster

# 2. Метод прогноза
def _forecast_newmodel(self, train_df, target_date):
    model = NewModelForecaster()
    model.fit(train_df)
    return model.predict(train_df, target_date)['prediction'] - 100

# 3. Вызов в _run_rolling
results['NewModel'] = self._forecast_newmodel(train_df, target_date)
```

## После добавления — ОБЯЗАТЕЛЬНО

```bash
# 1. Перезапустить ВСЕ бэктесты
python3 scripts/run_backtest_h1.py
python3 scripts/run_backtest_h2.py
python3 scripts/run_backtest_h12.py

# 2. Проверить чеклист
python3 scripts/add_model_checklist.py ModelName

# 3. Обновить прогнозы
python3 scripts/precompute_forecasts.py

# 4. Перегенерировать графики
python3 scripts/generate_charts.py
```

### Результаты бэктестов (где сохраняются)

После запуска бэктестов результаты сохраняются в:

```
archive/results/
├── backtest_h1_predictions.csv          # Прогнозы h=1 всех моделей
├── backtest_h1_metrics.csv              # Метрики h=1 (MAE, RMSE, KPI)
├── backtest_h1_summary.md               # Сводка h=1 в Markdown
├── backtest_h2_predictions.csv          # Прогнозы h=2
├── backtest_h2_metrics.csv              # Метрики h=2
├── backtest_h2_summary.md               # Сводка h=2
├── backtest_h12_predictions.csv         # Прогнозы h=12
├── backtest_h12_metrics.csv             # Метрики h=12
└── backtest_h12_summary.md              # Сводка h=12
```

### Графики и визуализации (где сохраняются)

Интерактивные графики сохраняются в:

```
assets/charts/
├── backtest_h1_predictions.html         # Графики прогнозов h=1
├── backtest_h1_errors.html              # Графики ошибок h=1
├── backtest_h2_predictions.html         # Графики прогнозов h=2
├── backtest_h12_predictions.html        # Графики прогнозов h=12
├── forecasts.html                       # Общие прогнозы
└── model_comparison.html                # Сравнение моделей
```

### Проверка результатов

```bash
# Проверить CSV-результаты
ls -la archive/results/backtest_h*.csv

# Проверить графики
ls -la assets/charts/*.html

# Просмотреть метрики
cat archive/results/backtest_h1_metrics.csv
```

Результат должен быть: **ВСЕ ПРОВЕРКИ ПРОЙДЕНЫ: 11/11**
