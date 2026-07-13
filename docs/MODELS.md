# Модели прогнозирования СИРЕНА-КБР v5.2

## Содержание

- [Production Ensemble](#production-ensemble)
- [Топ Модели](#топ-модели-v52)
- [Сценарные Модели](#сценарные-модели)
- [Nowcasting Models](#nowcasting-models)
- [Auxiliary Models](#auxiliary-models)
- [Добавление новой модели](#добавление-новой-модели)

---

## Production Ensemble

Ансамбль объединяет модели с весами, оптимизированными на исторических данных (v5.2):

| Модель | Вес | Тип | Роль |
|--------|-----|-----|------|
| **Subcomponent Multi** | **15%** | Bottom-up | Учет специфики 45 групп товаров |
| **Huber** | **15%** | Robust Linear | Стабильность на h=1 |
| **Ridge Shock** | **14%** | Linear + Dummies | Учет шоков (2014, 2022) |
| **ElasticNet** | **14%** | Regularized Linear | Отбор признаков |
| **NGBoost Shock** | **14%** | Gradient Boosting | Probabilistic forecast |
| **NGBoost** | **10%** | Gradient Boosting | Baseline boosting |
| **Ridge** | **7%** | Linear | Baseline |
| **Prohpet** | **4%** | Additive | Долгосрочный тренд (h=12) |
| **EBM** | **3%** | Additive Tree | Интерпретируемость |

---

## Топ Модели v5.2

### 1. Subcomponent Multi (Главная модель)

**Файл:** `sirena/models/subcomponent_multi.py`

Агрегирует прогнозы наилучших моделей для каждого из 45 субкомпонентов.
Для каждой группы (продукты, услуги, непродовольственные) и подгруппы выбирается своя спецификация.

**Особенности v2.3:**
- Использует rate-признаки (`ki_lag`, `spread`) для чувствительных товаров.
- **Результат**: MAE 0.236 на h=1 (лучшая в системе).

### 2. Huber Forecaster

**Файл:** `sirena/models/huber.py`

Линейная модель с функцией потерь Huber Loss, устойчивая к выбросам.
Идеальна для краткосрочного прогнозирования в условиях волатильности.

### 3. Horizon Ensemble

**Файл:** `sirena/models/horizon_ensemble.py`

Адаптивный ансамбль, меняющий веса в зависимости от горизонта:
- h=1: Больше вес у **Huber** (тренд)
- h=12: Больше вес у **Micro** (сезонность компонентов)

---

## Сценарные Модели

### Scenario Rate Model v2.0

**Файл:** `sirena/models/scenario_rate.py`

Модель трансмиссионного механизма ДКП. Используется для вкладки "Сценарии Ki".

**Возможности:**
- Оценка влияния повышения/понижения ключевой ставки.
- Асимметричные эффекты (повышение ставки действует сильнее).
- Калибровка на исторических данных SVAR.

### Subcomponent Scenario

**Файл:** `sirena/models/subcomponent_scenario.py`

Декомпозиция эффекта ставки по 45 субкомпонентам. Позволяет видеть, какие именно товары отреагируют на изменение ставки (например, авто и техника), а какие нет (мясо, молоко).

---

## Nowcasting Models

### VolatilityWeightedNowcaster

**Файл:** `sirena/models/volatility_weighted_nowcaster.py`

**Класс:** `VolatilityWeightedNowcaster`

#### Описание

Nowcasting-модель, использующая обратную волатильность в качестве весов для агрегации недельных сигналов цен.

**Гипотеза:** Продукты с низкой волатильностью (более стабильные) должны иметь больший вес при формировании сигнала.

**Методология:**
1. Для каждого продукта рассчитать историческую волатильность (стандартное отклонение WoW-роста цен)
2. Вычислить обратную волатильность: `weight_i = 1 / std_i`
3. Нормализовать веса, чтобы сумма равнялась 1
4. Агрегировать недельные сигналы с использованием этих весов
5. Использовать взвешенный сигнал для прогнозирования месячной инфляции

**Находки исследований (Task 411):**
- ❌ Обратная волатильность НЕ улучшает точность nowcasting по сравнению со стандартными весами ИПЦ
- ❌ Продукт-специфическая настройка волатильности ухудшает MAE на 5%
- ✅ Рекомендация: Использовать стандартные веса корзины ИПЦ, а не волатильностные веса

Модель реализована для полноты и исторического сравнения.

#### Параметры

| Параметр | Значение | Описание |
|-----------|-----------|-----------|
| `alpha` | 1.0 | Сила регуляризации Ridge |
| `min_samples_per_product` | 20 | Минимум наблюдений для расчёта волатильности |
| `volatility_window` | 52 | Окно для расчёта волатильности (недель) |

#### Источники данных

- Недельные цены: `data/kbr_weekly_prices_2008_2026.csv`
- Месячная инфляция: `data/inflation_data.csv`

#### Использование

```python
from sirena.models import VolatilityWeightedNowcaster

model = VolatilityWeightedNowcaster(alpha=1.0)
model.fit(df)

# Прогноз (только h=1 - это nowcasting модель)
forecast = model.forecast(horizon=1)

# Получить веса по продуктам
weights = model.get_volatility_weights()
print(weights)  # {product_code: weight}

# Получить волатильность продуктов
volatility = model.get_product_volatility()
print(volatility)  # {product_code: std_dev}

# Бэктест
results = model.backtest(df, start_date="2024-01-01")
mae = results["error"].abs().mean()
print(f"Backtest MAE: {mae:.4f}%")
```

---

### RegimeAdaptiveNowcaster

**Файл:** `sirena/models/regime_adaptive_nowcaster.py`

**Класс:** `RegimeAdaptiveNowcaster`

#### Описание

Nowcasting-модель с переключением весов в зависимости от макроэкономического режима.

**Гипотеза:** Различные экономические режимы (shock/normal/high_inflation) требуют разных весов продуктов для точного nowcasting.

**Режимы:**
- **SHOCK**: Резкие изменения ставок (|ΔKi| > 0.5 или |ΔRuonia| > 0.5)
- **HIGH_INFLATION**: Ускорение инфляции (ΔYoY > 1.5 п.п.)
- **NORMAL**: Стабильные условия

**Методология:**
1. Определить текущий макроэкономический режим (Ki, Ruonia, инфляция)
2. Выбрать режим-специфические веса продуктов
3. Агрегировать недельные сигналы с использованием этих весов
4. Прогнозировать месячную инфляцию с использованием сигнала + лагированной ИПЦ + сезонности

**Находки исследований (Task 414):**
- ✅ Режим-специфические веса значительно отличаются от фиксированных
- ✅ Shock режим: Равномерное взвешивание (все продукты ~11%) работает лучше
- ✅ Normal режим: Продукт-специфические веса на основе исторической производительности
- ✅ High inflation режим: Некоторые продукты получают большие веса (нефть, яйца и т.д.)

#### Параметры

| Параметр | Значение | Описание |
|-----------|-----------|-----------|
| `alpha` | 1.0 | Сила регуляризации Ridge |
| `regime_weights_path` | `data/weekly_regime_weights.csv` | Путь к CSV с режим-специфическими весами |

#### Источники данных

- Недельные цены: `data/kbr_weekly_prices_2008_2026.csv`
- Месячная инфляция: `data/inflation_data.csv`
- Режим-специфические веса: `data/weekly_regime_weights.csv` (из Task 414)
- Макро данные: `data/inflation_data.csv` (Ki, Ruonia и др.)

#### Использование

```python
from sirena.models import RegimeAdaptiveNowcaster

model = RegimeAdaptiveNowcaster(alpha=1.0)
model.fit(df)

# Определить текущий режим
regime = model.detect_current_regime()
print(f"Current regime: {regime[0]}")  # 'normal', 'shock', или 'high_inflation'
print(f"Diagnostics: {regime[1]}")

# Прогноз (только h=1 - это nowcasting модель)
forecast = model.forecast(horizon=1)

# Получить режим-специфические веса
regime_weights = model.get_regime_weights()
print(regime_weights)  # {'normal': {code: weight}, 'shock': {...}, 'high_inflation': {...}}

# Бэктест
results = model.backtest(df, start_date="2024-01-01")
mae = results["error"].abs().mean()
print(f"Overall MAE: {mae:.4f}%")

# Производительность по режимам
regime_perf = results.groupby("regime")["error"].agg(["count", lambda x: x.abs().mean()])
regime_perf.columns = ["count", "mae"]
print(regime_perf)

# Save results
results.to_csv(args.output, index=False)
print(f"Results saved to: {args.output}")
else:
    print("\nNo backtest results generated (insufficient data)")

if __name__ == "__main__":
    main()
```

### Nowcasting Performance

**Период бэктеста:** 2024-2025 (24 месяца)

| Модель | MAE | vs Fixed Weights | Рекомендация |
|--------|-----|-----------------|-------------|
| **WeeklyPriceNowcaster** | 0.043% | baseline | ✅ Лучший nowcasting (цель <0.10%) |
| **VolatilityWeightedNowcaster** | ~0.045% | +5% хуже | ❌ Использовать фиксированные веса |
| **RegimeAdaptiveNowcaster** | ~0.043% | ~0% изменение | ✅ Опционально для режимозависимого анализа |

**Ключевые находки:**

1. **VolatilityWeightedNowcaster** — обратная волатильность НЕ улучшает точность (Task 411)
   - Продукт-специфическая настройка волатильности ухудшает MAE на 5%
   - Рекомендация: Использовать стандартные веса корзины ИПЦ

2. **RegimeAdaptiveNowcaster** — режимозависимые веса работают (Task 414)
   - Minimal overall improvement: +0.07% (2.8713 vs 2.8693 MAE)
   - Shock режим: равномерное взвешивание (~11% на продукт)
   - Normal режим: оптимизированные веса на основе истории
   - High Inflation режим: фокус на волатильных продуктах

**Вывод:** WeeklyPriceNowcaster с фиксированными весами остаётся лучшим nowcasting-решением. VolatilityWeightedNowcaster и RegimeAdaptiveNowcaster предоставлены для исследовательских целей и сценарного анализа.

---

## Сравнение моделей

### Метрики качества (бэктест 2019-2024)

| Модель | MAE | KPI (≤0.5) | Макро |
|--------|-----|------------|-------|
| **Ridge** | 0.38 | 78.9% | ✅ вкл |
| Ridge (без макро) | 0.36 | 77.5% | ❌ |
| BVAR | 0.35 | 78% | ✅ встроен |
| LightGBM | 0.43 | 71.1% | ❌ откл |
| LightGBM (с макро) | 0.43 | 68.7% | ✅ |
| Prophet | 0.38 | 75% | ❌ |
| SARIMA | 0.40 | 72% | ❌ |
| ETS | 0.42 | 70% | ❌ |
| LSTM | 0.45 | 68% | ❌ |

### Влияние макро-признаков

| Модель | Без макро | С макро | Δ KPI |
|--------|-----------|---------|-------|
| Ridge | MAE=0.36, KPI=77.5% | MAE=0.38, KPI=78.9% | **+1.4 п.п.** |
| LightGBM | MAE=0.43, KPI=71.1% | MAE=0.43, KPI=68.7% | -2.4 п.п. |

**Вывод:** Ridge лучше использует макро-признаки (линейные зависимости), LightGBM переобучается.

### Время выполнения

| Модель | fit() | forecast(12) |
|--------|-------|--------------|
| Ridge | ~100ms | ~10ms |
| BVAR | ~200ms | ~500ms |
| LightGBM | ~500ms | ~50ms |
| Prophet | ~5s | ~100ms |
| SARIMA | ~1s | ~10ms |
| ETS | ~200ms | ~10ms |
| LSTM | ~30s | ~100ms |

### Когда какую модель использовать

| Сценарий | Рекомендация |
|----------|--------------|
| Быстрый прогноз | Ridge |
| Учёт макро-шоков | BVAR |
| Нелинейные эффекты | LightGBM |
| Автоматическая сезонность | Prophet |
| Baseline (инерция) | SARIMA |
| Простая сезонность | ETS |
| Эксперименты с DL | LSTM |
| **Nowcasting (недельные данные)** | **WeeklyPriceNowcaster** |
| Режимозависимый анализ | RegimeAdaptiveNowcaster |
| Максимальная точность | **Ансамбль** |

---

## Добавление новой модели

### 1. Создайте файл

```python
# sirena/models/my_model.py

from .base import BaseForecaster
from .registry import ModelRegistry

@ModelRegistry.register("my_model")
class MyModelForecaster(BaseForecaster):
    name = "my_model"
    MIN_TRAIN_SIZE = 24

    def fit(self, df, target_col='Все товары и услуги'):
        series = self._validate_data(df, target_col)
        # ... обучение ...
        self._is_fitted = True
        return self

    def forecast(self, horizon=12):
        self._check_fitted()
        # ... прогноз ...
        return np.array([...])

    def backtest(self, df, start_date='2019-01-01', target_col='...'):
        # ... бэктест ...
        return pd.DataFrame({...})
```

### 2. Импортируйте в `__init__.py`

```python
# sirena/models/__init__.py
from .my_model import MyModelForecaster
```

### 3. Установите вес (опционально)

```python
ModelRegistry.set_default_weight("my_model", 0.05)
```

### 4. Добавьте тесты

```python
# tests/test_models.py
class TestMyModel:
    def test_fit(self, sample_data):
        model = MyModelForecaster()
        model.fit(sample_data)
        assert model.is_fitted
```
