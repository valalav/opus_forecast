# СИРЕНА-КБР v5.3 — Система прогнозирования инфляции

**Версия:** 5.3  
**Дата обновления:** 30 января 2026  
**Статус:** Production  
**Dashboard:** http://localhost:8503

---

## О системе

**СИРЕНА-КБР** (Система Интеллектуального Регионального Анализа) — платформа для прогнозирования индекса потребительских цен (ИПЦ) в Кабардино-Балкарской Республике.

### Что нового в v5.3

- **Weekly Prices Tab** — полноценный nowcasting с анализом недельных цен
  - Волатильность по товарам
  - Алерты на аномалии
  - Тренды цен
- **Актуализированный ансамбль** — 9 моделей с оптимизированными весами
- **SubcomponentMulti** — лучшая модель по SIRENA Score (0.515)
- **Улучшенная визуализация** — 13 вкладок: 5 прогнозов + Сценарии + Weekly + 5 бэктестов
- **🧪 Эксперименты:**
  - **Rolling Seasonality Ridge** — скользящая сезонность (24 мес.) показывает MAE 0.314, лучше Ridge на 6.4%
  - Подробнее: `experiments/rolling_seasonality/`

### Что нового в v5.0

- **Unified Architecture** — единый API для всех моделей
- **Ki Trajectory Model** — эндогенный прогноз ключевой ставки
- **Regime Detector** — адаптивные лаги на основе макрорежима
- **Conformal Prediction** — калиброванные доверительные интервалы

### Что было в v4.8

- **Production Ensemble (9 моделей)** — оптимизированные веса по h=1 MAE:
  - Huber: 18% (MAE 0.289) — лучшая на h=1
  - RidgeShockDummies: 17% (MAE 0.299)
  - ElasticNet: 17% (MAE 0.301)
  - NGBoostShock: 16% (MAE 0.291) — лучшая на h=2
  - NGBoost: 12% (MAE 0.312)
  - Ridge: 8% (MAE 0.310) — baseline
  - RidgeExtended: 5% (MAE 0.318)
  - Prophet: 4% (MAE 0.277) — лучшая на h=12
  - EBM: 3% (MAE 0.340)
- **Удалено из ансамбля:** BVAR (катастрофический h=12 MAE 5.714), ETS, SARIMA, CatBoost

---

## Быстрый старт

### Установка

```bash
cd /home/valalav/_projects/sirena-kbr
pip install -r requirements.txt
```

### Запуск Dashboard (Streamlit)

```bash
# Локально (порт 8503!)
streamlit run dashboard.py --server.port 8503

# В локальной сети (LAN)
streamlit run dashboard.py --server.address 0.0.0.0 --server.port 8503
```
- Локально: http://localhost:8503
- LAN: http://<IP-адрес>:8503

### Запуск REST API

```bash
uvicorn api.main:app --reload --port 8000
```
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

### Запуск Backtesting

Система автоматизированного бэктестирования с 5 горизонтами прогноза:

```bash
# h=1 (1 месяц вперед) — САМЫЙ ВАЖНЫЙ КПЭ
python3 scripts/run_backtest_h1.py

# h=2, h=3, h=6, h=12
python3 scripts/run_backtest_h2.py
python3 scripts/run_backtest_h3.py
python3 scripts/run_backtest_h6.py
python3 scripts/run_backtest_h12.py

# Результаты сохраняются в:
ls archive/results/backtest_h*.csv
```

**Лучшие модели (декабрь 2024 — ноябрь 2025):**
- h=1: Huber (MAE 0.289), Ridge Shock (0.299), ElasticNet (0.301)
- h=2: NGBoost Shock (MAE 0.291), NGBoost (0.302), EBM (0.305)
- h=12: Prophet (MAE 0.277), Ridge (0.338), Ridge Extended (0.339)

См. подробную методику в [docs/BACKTEST_METHODOLOGY.md](docs/BACKTEST_METHODOLOGY.md)

### SIRENA Score — Комплексная метрика

Единая метрика для оценки моделей на всех горизонтах:

```
SIRENA_Score = (0.50×MAE_h1 + 0.30×MAE_h2 + 0.20×MAE_h12) × (2 - KPI_rate)
```

```bash
# Расчёт SIRENA Score за 2020-2025
python3 scripts/sirena_score.py
```

**Топ-3 модели по SIRENA Score (ноябрь 2025):**

| Ранг | Модель | Score | MAE h=1 | MAE h=2 | MAE h=12 |
|------|--------|-------|---------|---------|----------|
| 🥇 1 | Subcomp_Multi | **0.515** | 0.265 | 0.278 | 0.264 |
| 🥈 2 | EBM | 0.592 | 0.309 | 0.309 | 0.309 |
| 🥉 3 | Subcomp | 0.607 | 0.285 | 0.379 | 0.302 |

Графики и результаты: `assets/charts/sirena_score_dynamics.html`

---

## Модели

### Production Ensemble (9 моделей) — v4.8

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

### Experimental Models (новые, v5.x)

| Модель | Файл | MAE h=1 | SIRENA Score | Описание |
|--------|------|---------|--------------|----------|
| **SubcomponentMulti** | `sirena/models/subcomponent_multi.py` | 0.265 | **0.515 (#1)** | Лучшая модель! Multi-horizon bottom-up |
| **Subcomponent** | `sirena/models/subcomponent.py` | 0.285 | 0.607 (#3) | Bottom-up по 3 компонентам |
| **Microcomponent** | `sirena/models/microcomponent.py` | — | — | 537 микрокомпонентов |
| **HierarchicalMicro** | `sirena/models/hierarchical_micro.py` | — | — | Полная иерархия micro → subcomp → comp → total |
| **MicroOptimized** | `sirena/models/micro_optimized.py` | — | — | Оптимизированный micro (Huber для stable, Ridge для volatile) |
| **MIDAS** | `sirena/models/midas.py` | — | — | Mixed Data Sampling (высокочастотные данные) |
| **TFT** | `sirena/models/tft.py` | — | — | Temporal Fusion Transformer (attention) |
| **Conformal** | `sirena/models/conformal.py` | — | — | Калиброванные доверительные интервалы |
| **KiTrajectory** | `sirena/models/ki_trajectory.py` | — | — | Эндогенный прогноз ключевой ставки |
| **UnifiedSubcomponent** | `sirena/models/unified_subcomp.py` | — | — | Интегрированная baseline + scenario |
| **ScenarioRate** | `sirena/models/scenario_rate.py` | — | — | Модель трансмиссии ставки |
| **RegimeDetector** | `sirena/models/regime_detector.py` | — | — | Адаптивные лаги по макрорежиму |
| **HorizonEnsemble** | `sirena/models/horizon_ensemble.py` | — | — | Адаптивный Huber + Micro |
| **StackingRegressor** | `sirena/models/stacking_regressor.py` | — | — | Meta-learning регрессор |

### Auxiliary Models (не в ансамбле)

| Модель | Файл | Описание |
|--------|------|----------|
| SARIMA | `sirena/models/arima.py` | Сезонная ARIMA |
| AR1 | `sirena/models/arima.py` | Авторегрессия первого порядка |
| ETS | `sirena/models/ets.py` | Exponential Smoothing |
| HoltWinters | `sirena/models/holt_winters.py` | Holt-Winters сезонность |
| BVAR | `sirena/models/bvar.py` | Байесовская VAR |
| BVARRate | `sirena/models/bvar_rate.py` | BVAR с ключевой ставкой |
| BayesianRidge | `sirena/models/bayesian_ridge.py` | Байесовская Ridge с CI |
| LightGBM | `sirena/models/lightgbm.py` | Gradient boosting |
| CatBoost | `sirena/models/catboost_model.py` | CatBoost для малых выборок |
| XGBoost | `sirena/models/xgboost_model.py` | XGBoost |
| NaiveSeasonal | `sirena/models/naive_seasonal.py` | Наивная сезонная модель |
| RidgeMacro | `sirena/models/ridge_macro.py` | Ridge с оптимальными макро-признаками |
| BudgetRidge | `sirena/models/budget_ridge.py` | Ridge с бюджетными данными |

### Nowcasting Models

| Модель | Файл | Описание |
|--------|------|----------|
| VolatilityWeightedNowcaster | `sirena/models/volatility_weighted_nowcaster.py` | Взвешенный по волатильности |
| RegimeAdaptiveNowcaster | `sirena/models/regime_adaptive_nowcaster.py` | Режим-адаптивный nowcaster |

---

## Примеры использования

### Huber (лучшая на h=1)

```python
from sirena.models import HuberForecaster

model = HuberForecaster(epsilon=1.35)
model.fit(df)
pred = model.predict(df_ext, target_date)
```

### SubcomponentMulti (лучшая по SIRENA Score)

```python
from sirena.models import SubcomponentMultiForecaster

model = SubcomponentMultiForecaster()
model.fit(df)
fc = model.forecast(horizon=12)
# Автоматически использует Ridge, Prophet, NGBoost по компонентам
```

### NGBoost с шоковыми дамми (лучшая на h=2)

```python
from sirena.models import NGBoostShockForecaster

model = NGBoostShockForecaster(n_estimators=200)
model.fit(df)
fc = model.forecast(horizon=12)
```

### Prophet (лучшая на h=12)

```python
from sirena.models import ProphetForecaster

model = ProphetForecaster()
model.fit(df)
fc = model.forecast(horizon=12)
```

---

## REST API v5.0 (Unified Architecture)

### Эндпоинты

| Метод | Путь | Описание |
|-------|------|----------|
| GET | `/health` | Статус сервера |
| GET | `/models` | Список моделей и весов |
| GET | `/models/{name}` | Информация о модели |
| POST | `/forecast` | Прогноз ансамбля или отдельной модели |
| POST | `/forecast/unified` | Unified прогноз (все модели) |
| GET | `/forecast/quick` | Быстрый прогноз (только Ridge) |
| POST | `/backtest` | Бэктестирование модели |
| GET | `/backtest/metrics/{model}` | Метрики качества |
| GET | `/weekly/signal` | Nowcasting сигнал |
| POST | `/weekly/forecast` | Прогноз недельных цен |

### Примеры запросов

```bash
# Unified прогноз (все модели)
curl -X POST http://localhost:8000/forecast/unified \
  -H "Content-Type: application/json" \
  -d '{"horizon": 12}'

# Прогноз конкретной моделью
curl -X POST http://localhost:8000/forecast \
  -H "Content-Type: application/json" \
  -d '{"model": "subcomponent_multi", "horizon": 12}'

# Быстрый прогноз
curl "http://localhost:8000/forecast/quick?horizon=6"

# Бэктест
curl -X POST http://localhost:8000/backtest \
  -H "Content-Type: application/json" \
  -d '{"model": "huber", "start_date": "2023-01-01", "horizon": 1}'
```

---

## Архитектура

### Структура проекта

```
sirena-kbr/
├── api/                      # REST API (FastAPI)
│   ├── main.py
│   ├── routes/               # forecast, backtest, weekly, health
│   └── schemas/              # Pydantic модели
│
├── sirena/                   # Ядро системы
│   ├── models/               # 40+ моделей прогнозирования
│   │   ├── base.py           # BaseForecaster (ABC)
│   │   ├── registry.py       # ModelRegistry (декоратор @register)
│   │   ├── ridge.py          # Ridge (baseline)
│   │   ├── huber.py          # Huber (лучшая h=1)
│   │   ├── subcomponent_multi.py  # Лучшая по SIRENA Score
│   │   ├── midas.py          # MIDAS
│   │   ├── tft.py            # Temporal Fusion Transformer
│   │   ├── conformal.py      # Conformal Prediction
│   │   └── ...               # 40+ моделей
│   ├── forecast.py           # EnsembleForecaster
│   ├── unified_api.py        # Unified API (v5.0)
│   └── data_loader.py        # Загрузка данных
│
├── pages/                    # Dashboard pages (Streamlit)
│   ├── constants.py          # ALL_MODELS, MODEL_COLORS
│   ├── 1_Forecast.py
│   ├── 2_Backtest.py
│   ├── 3_Weekly.py           # Nowcasting
│   └── ...
│
├── scripts/                  # 120+ скриптов
│   ├── precompute_forecasts.py
│   ├── run_backtest_h*.py    # Бэктесты для всех горизонтов
│   ├── sirena_score.py
│   └── ...
│
├── docs/                     # Документация
│   ├── MODEL_CATALOG.md      # Каталог моделей
│   ├── ADDING_MODEL_GUIDE.md # Гайд по добавлению моделей
│   ├── NOWCASTING.md         # Методика nowcasting
│   └── ...
│
├── data/                     # Данные
│   ├── infl_kbr.csv          # Месячная инфляция КБР
│   ├── kbr_weekly_prices_2008_2026.csv  # Недельные цены
│   ├── precomputed_forecasts.json
│   └── ...
│
├── dashboard.py              # Streamlit UI (v5.3, порт 8503)
├── requirements.txt
└── README.md
```

### Добавление новой модели (через ModelRegistry)

```python
from sirena.models import BaseForecaster, ModelRegistry

@ModelRegistry.register("my_model")
class MyModel(BaseForecaster):
    name = "my_model"

    def fit(self, df, target_col='Все товары и услуги'):
        self._is_fitted = True
        return self

    def predict(self, df, target_date):
        return {'prediction': value, 'model': 'MyModel'}

    def forecast(self, horizon=12):
        self._check_fitted()
        return np.array([...])  # MoM values

    def backtest(self, df, start_date='2019-01-01', target_col='...'):
        return pd.DataFrame({'date': ..., 'actual': ..., 'prediction': ..., 'error': ...})
```

---

## Данные

### Входные форматы

**Месячные данные** `data/infl_kbr.csv`:
```csv
Day;Товар;MoM
01.01.2020;Все товары и услуги;100.5
01.01.2020;Продовольственные товары;100.8
```

**Недельные данные** `data/kbr_weekly_prices_2008_2026.csv`:
```csv
Date;Component;Product;Price;Change
2026-01-12;Продовольственные товары;Говядина;649.55;101.28
```

### Веса компонентов

| Компонент | Вес |
|-----------|-----|
| Продовольственные товары | 39.48% |
| Непродовольственные товары | 36.53% |
| Услуги | 23.42% |

---

## Тестирование

```bash
# Все тесты
pytest tests/ -v

# Только API
pytest tests/test_api.py -v

# Только модели
pytest tests/test_models.py -v

# Верификация дашборда
python3 scripts/verify_all_tabs.py
```

---

## Метрики качества (Backtest 2023-2025)

### Production Ensemble (9 моделей)

| Модель | MAE h=1 | MAE h=2 | MAE h=12 | vs Ridge |
|--------|---------|---------|----------|----------|
| **Huber** | 0.289 | — | — | -8.5% |
| **RidgeShockDummies** | 0.299 | — | — | -5.3% |
| **ElasticNet** | 0.301 | — | — | -4.7% |
| **NGBoostShock** | 0.291 | 0.291 | — | -7.9% |
| **Ridge** | 0.310 | — | 0.338 | baseline |
| **Prophet** | — | — | 0.277 | — |

### Experimental Models

| Модель | MAE h=1 | MAE h=2 | MAE h=12 | SIRENA Score |
|--------|---------|---------|----------|--------------|
| **SubcomponentMulti** | 0.265 | 0.278 | 0.264 | **0.515** |
| **Subcomponent** | 0.285 | 0.379 | 0.302 | 0.607 |
| **EBM** | 0.309 | 0.309 | 0.309 | 0.592 |

---

## Зависимости

```
# Web
streamlit>=1.28.0
fastapi>=0.104.0
uvicorn>=0.24.0

# Data
pandas>=2.0.0
numpy>=1.24.0

# ML
scikit-learn>=1.3.0
lightgbm>=4.0.0
prophet>=1.1.0
statsmodels>=0.14.0
interpret>=0.6.0
ngboost>=0.5.0
catboost>=1.2.0
xgboost>=2.0.0

# Deep Learning (опционально)
torch>=2.0.0
pytorch-forecasting>=1.0.0

# External Data
yfinance>=0.2.0
```

---

## 🧪 Эксперименты

Активные исследования и прототипы в `experiments/`:

### Rolling Seasonality Ridge (2026-02-02)
- **Проблема:** Глобальная сезонность на всей истории вредит после структурных сдвигов 2022-2024
- **Решение:** Скользящая сезонность на последних 24 месяцах
- **Результат:** MAE 0.314 (6.4% лучше Ridge baseline, 1.2% лучше Huber)
- **Статус:** 🟢 Рекомендуется к внедрению
- **Подробнее:** [experiments/rolling_seasonality/](experiments/rolling_seasonality/)

Структура эксперимента:
```
experiments/rolling_seasonality/
├── README.md                    # Быстрый старт
├── docs/
│   ├── RESEARCH_PROPOSAL.md    # Предпосылки и гипотезы
│   └── RESULTS.md               # Результаты и анализ
├── models/
│   └── rolling_seasonality_ridge.py  # Модель
└── scripts/
    └── run_backtest_rolling.py  # Бэктест
```

Запуск:
```bash
cd experiments/rolling_seasonality
python3 scripts/run_backtest_rolling.py
```

---

## Changelog

### v5.3 (2026-01-30)
- Weekly Prices Tab — полноценный nowcasting
- Dashboard v5.3 (13 вкладок)
- Актуализирован ансамбль (9 моделей)

### v5.0 (2026-01-01)
- Unified Architecture — единый API
- Ki Trajectory Model
- Regime Detector
- Conformal Prediction
- TFT (Temporal Fusion Transformer)

### v4.8 (2025-12-20)
- Production Ensemble из 9 моделей с оптимизированными весами
- RidgeShockDummies и NGBoostShock (шоковые дамми)
- Удалены из ансамбля: BVAR, ETS, SARIMA, CatBoost

### v4.7 (2025-12-13)
- FundamentalForecaster
- USDForecaster
- Critical Audit Ridge

### v4.3 (2025-12-10)
- Ridge Extended v2
- ElasticNet
- Huber

### v4.0 (2025-12-07)
- 7 моделей ансамбля
- REST API
- WeeklyForecaster

---

*Разработано в рамках проекта Opus Forecast*
