# Журнал изменений СИРЕНА-КБР

Все значимые изменения в проекте документируются в этом файле.

Формат основан на [Keep a Changelog](https://keepachangelog.com/ru/1.0.0/).

---

## [4.0.1] - 2025-12-07

### Добавлено

**Макро-признаки Ki и Ruonia:**
- Новые признаки на основе ключевой ставки и RUONIA:
  - `ruonia_diff_lag1` (r=0.477) — изменение RUONIA за месяц
  - `spread_lag4` (r=0.444) — спред Ki-Ruonia
  - `ki_diff_lag6` (r=0.300) — изменение ключевой ставки
  - `ki_vol` (r=0.256) — волатильность ставки
- Модуль `sirena/macro_features.py` для работы с макро-признаками
- Параметр `use_macro` в Ridge и LightGBM

### Изменено

**Ridge:**
- `use_macro=True` по умолчанию (улучшает KPI на 1.4 п.п.)
- Добавлен метод `get_feature_importance()` с флагом `is_macro`
- Результат `predict()` включает `has_macro`

**LightGBM:**
- `use_macro=False` по умолчанию (ухудшает KPI на 2.4 п.п.)
- Макро-признаки доступны для экспериментов

**Документация:**
- Обновлён `docs/MODELS.md` с описанием макро-признаков
- Добавлена таблица влияния макро-признаков на качество

### Результаты бэктеста

| Модель | Без макро | С макро | Δ KPI |
|--------|-----------|---------|-------|
| Ridge | MAE=0.36, KPI=77.5% | MAE=0.38, KPI=78.9% | +1.4 п.п. |
| LightGBM | MAE=0.43, KPI=71.1% | MAE=0.43, KPI=68.7% | -2.4 п.п. |

---

## [4.0.0] - 2025-12-07

### Добавлено

**Архитектура:**
- `BaseForecaster` — абстрактный базовый класс для всех моделей
- `ModelRegistry` — Factory Pattern для управления моделями
- `ForecastResult` — структура результата прогноза

**REST API (FastAPI):**
- `POST /forecast` — прогноз ансамблем моделей
- `GET /forecast/quick` — быстрый прогноз только Ridge
- `POST /backtest` — бэктестирование модели
- `GET /backtest/metrics/{model}` — метрики качества
- `GET /models` — список моделей и весов
- `GET /models/{name}` — информация о модели
- `GET /health` — проверка статуса
- Swagger UI (`/docs`) и ReDoc (`/redoc`)

**Производительность:**
- `AsyncModelRunner` — параллельный запуск моделей
- `run_ensemble_async()` — асинхронный ансамбль
- `run_ensemble_parallel()` — многопроцессный ансамбль
- `ForecastCache` — кэширование с TTL (memory/file backends)

**Документация:**
- `docs/API.md` — полная документация REST API
- `docs/MODELS.md` — описание всех 7 моделей
- `docs/DEVELOPMENT.md` — руководство разработчика

### Изменено

**Структура проекта:**
```
sirena/
├── models/
│   ├── base.py          # BaseForecaster
│   ├── registry.py      # ModelRegistry
│   ├── ridge.py         # RidgeForecaster
│   ├── bvar.py          # BVARForecaster
│   ├── lightgbm.py      # LightGBMForecaster
│   ├── prophet.py       # ProphetForecaster
│   ├── arima.py         # SARIMAForecaster
│   ├── ets.py           # ETSForecaster
│   └── lstm.py          # LSTMForecaster
├── async_runner.py      # Параллельный запуск
├── cache.py             # Кэширование
├── config.py            # Конфигурация
├── data_loader.py       # Загрузка данных
└── forecast.py          # EnsembleForecaster

api/
├── main.py              # FastAPI приложение
├── routes/
│   ├── forecast.py
│   ├── backtest.py
│   └── models.py
└── schemas/
    ├── forecast.py
    ├── backtest.py
    └── models.py
```

**Модели:**
- Все модели наследуют `BaseForecaster`
- Единый интерфейс: `fit()`, `forecast()`, `backtest()`
- Регистрация через декоратор `@ModelRegistry.register("name")`

### Исправлено

- Унифицированы интерфейсы всех моделей
- Улучшена обработка ошибок в моделях
- Добавлена валидация входных данных

---

## [3.2.0] - 2025-12-06

### Добавлено

- **4 новых модели:** LightGBM, Prophet, ETS, LSTM
- **7-модельный ансамбль** с оптимизированными весами
- Новые веса: Ridge 40%, BVAR 20%, LightGBM 15%, Prophet 10%, SARIMA/ETS/LSTM 5%
- LSTM Fallback для систем без TensorFlow
- Автоматическое обучение всех моделей

### Изменено

- Обновлена структура ансамбля (с 3 до 7 моделей)
- Перераспределены веса для учёта новых моделей

---

## [3.1.0] - 2025-12-05

### Добавлено

- 3-модельный ансамбль: Ridge, BVAR, SARIMA
- Бэктестирование с 2019 года
- Страница "История (Opus)" в dashboard

---

## [3.0.0] - 2025-12-04

### Добавлено

- Ridge модель с ETS-компонентой
- BVAR с Minnesota Prior
- SARIMA(1,0,1)(1,0,1,12)
- Streamlit Dashboard

---

## [2.4.0] - 2025-12-03

### Добавлено

- Адаптивный механизм ETS
- Месячные веса комбинирования
- Улучшенная сезонность

---

## Веса моделей (v4.0)

| Модель | Вес | Причина |
|--------|-----|---------|
| Ridge | 40% | Лучший MAE, стабильность |
| BVAR | 20% | Макро-зависимости |
| LightGBM | 15% | Нелинейности |
| Prophet | 10% | Авто-сезонность |
| SARIMA | 5% | Baseline |
| ETS | 5% | Простая сезонность |
| LSTM | 5% | DL эксперименты |

---

## Миграция с v3.2 на v4.0

### Использование моделей

**Было (v3.2):**
```python
# Отдельные функции в dashboard
from sirena_kbr_v2_4_auto import predict_kbr
from sirena_bvar import run_bvar_forecast
```

**Стало (v4.0):**
```python
from sirena.models import ModelRegistry, RidgeForecaster

# Через реестр
model = ModelRegistry.get("ridge")
model.fit(df)
forecast = model.forecast(horizon=12)

# Напрямую
model = RidgeForecaster(alpha=0.3)
model.fit(df)
```

### Ансамблевое прогнозирование

**Было:**
```python
# Ручной расчёт в dashboard.py
ensemble = ridge * 0.4 + bvar * 0.2 + ...
```

**Стало:**
```python
from sirena import EnsembleForecaster

ensemble = EnsembleForecaster()
ensemble.fit(df)
result = ensemble.forecast(horizon=12)
```

### REST API

**Новое:**
```bash
# Запуск API
uvicorn api.main:app --reload

# Прогноз
curl -X POST http://localhost:8000/forecast \
  -H "Content-Type: application/json" \
  -d '{"horizon": 12}'
```

---

## Авторы

- **СИРЕНА Team** — разработка и поддержка
