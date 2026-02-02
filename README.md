# СИРЕНА-КБР v4.7 — Система прогнозирования инфляции

**Версия:** 4.7
**Дата обновления:** 13 декабря 2025
**Статус:** Production

---

## О системе

**СИРЕНА-КБР** (Система Интеллектуального Регионального Анализа) — платформа для прогнозирования индекса потребительских цен (ИПЦ) в Кабардино-Балкарской Республике.

### Что нового в v4.7

- **Fundamental Model** — модель на экономических драйверах (без hardcoded сезонности)
  - Использует прогноз USD, нефть Brent, ключевую ставку
  - MAE 0.389 (менее точная на истории, но устойчивая к структурным сдвигам)
- **USDForecaster** — ElasticNet модель для прогноза курса доллара
  - Использует нефть и дифференциалы ставок
  - MAE 2.56 (лучше naive baseline 2.70)
- **Critical Audit** — выявлена сильная зависимость Ridge от сезонных коэффициентов

### Что нового в v4.3

- **Ridge Extended v2** — лучшая модель (MAE 0.313, -0.7% vs Ridge baseline)
  - Новые календарные признаки: `is_tariff_month` (июль), `is_q1`, `is_summer`
  - Квартальная сезонность: `quarter_sin` — сильнейший новый признак
- **ElasticNet** — L1+L2 регуляризация с автоматическим feature selection
  - Отбирает 12 из 25 признаков через L1
  - CV для оптимального баланса L1/L2
- **Huber Regressor** — робастная модель к выбросам
  - Не требует исключения 2022 года вручную
  - Автоматически снижает влияние выбросов

### Что было в v4.2

- **Ridge Extended** — momentum (d_y_lag1, d_y_lag3), volatility (y_vol3, y_vol6)
- **Bayesian Ridge** — автоматическая регуляризация + доверительные интервалы
- **CatBoost** — gradient boosting оптимизированный для малых выборок (~150 точек)
- **RegimeSwitchingEnsemble** — режимозависимые веса (shock/normal режимы)
- **Dashboard v4.2** — новые модели в прогнозе и бэктесте

### Что было в v4.1

- **Адаптивные веса ансамбля** — автоматическая оптимизация весов по MAE
- **Стакинг (Meta-Learning)** — двухуровневая модель на прогнозах базовых моделей
- **Hierarchical Forecast** — MinTrace reconciliation по компонентам
- **Нефть Brent** — экзогенная переменная из Yahoo Finance
- **EBM** — заменил LSTM (InterpretML, интерпретируемый бустинг)

### Что было в v4.0

- **7 моделей ансамбля** — Ridge, BVAR, LightGBM, Prophet, SARIMA, ETS, EBM
- **REST API** — FastAPI с автодокументацией (Swagger UI)
- **WeeklyForecaster** — nowcasting на недельных ценах
- **Модульная архитектура** — BaseForecaster + ModelRegistry

---

## Быстрый старт

### Установка

```bash
cd /home/valalav/_projects/sirena-kbr
pip install -r requirements.txt
```

### Запуск Dashboard (Streamlit)

```bash
# Локально
streamlit run dashboard.py

# В локальной сети (LAN)
streamlit run dashboard.py --server.address 0.0.0.0 --server.port 8501
```
- Локально: http://localhost:8501
- LAN: http://<IP-адрес>:8501

### Запуск REST API

```bash
uvicorn api.main:app --reload --port 8000
```
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

### Запуск Backtesting

Система автоматизированного бэктестирования с 3 горизонтами прогноза:

```bash
# h=1 (1 месяц вперед) — САМЫЙ ВАЖНЫЙ КПЭ
python3 scripts/run_backtest_h1.py

# h=2 (2 месяца вперед)
python3 scripts/run_backtest_h2.py

# h=12 (12 месяцев вперед, годовая траектория)
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

### Production Ensemble (7 моделей)

| Модель | Вес | MAE | Описание |
|--------|-----|-----|----------|
| **Ridge** | 40% | 0.316 | Ridge регрессия + ETS сезонность |
| **BVAR** | 20% | 0.424 | Байесовская VAR с Minnesota Prior |
| **LightGBM** | 15% | — | Gradient Boosting |
| **Prophet** | 10% | — | Facebook Prophet |
| **SARIMA** | 5% | — | Сезонная авторегрессия |
| **ETS** | 5% | 0.412 | Exponential Smoothing |
| **EBM** | 5% | — | Explainable Boosting Machine |

### Новые модели v4.3

| Модель | MAE | vs Ridge | Описание |
|--------|-----|----------|----------|
| **Ridge Extended v2** | 0.313 | **-0.7%** | Расширенные календарные признаки (is_tariff_month, is_q1, is_summer) |
| **ElasticNet** | 0.320 | +1.4% | L1+L2 регуляризация с автоматическим feature selection |
| **Huber** | 0.335 | +6.1% | Робастная к выбросам (Huber loss) |
| **Fundamental** | 0.389 | +23% | **Structural** | Модель на экономических драйверах (без сезонных dummy) |

### Модели v4.2

| Модель | MAE | vs Ridge | Описание |
|--------|-----|----------|----------|
| **Bayesian Ridge** | 0.321 | +1.8% | Автоматическая регуляризация + доверительные интервалы |
| **CatBoost** | 0.368 | +16.7% | Gradient boosting для малых выборок |
| **RegimeSwitching** | — | — | Режимозависимые веса (shock/normal) |

### Experimental Models

| Модель | Файл | Описание |
|--------|------|----------|
| **Stacking** | `models/stacking.py` | Meta-learning на прогнозах базовых моделей |
| **Hierarchical** | `models/hierarchical.py` | MinTrace reconciliation (Total = Σ Components) |
| **Ridge SA** | `models/ridge_sa.py` | Bottom-up по 3 компонентам на SA данных |
| **Ridge SA Sub** | `models/ridge_sa_sub.py` | Bottom-up по 47 субкомпонентам |
| **Weekly** | `models/weekly.py` | Nowcasting на недельных ценах |

### Micro ARIMA (внешняя модель)

| Модель | Файл | MAE h=1 | SIRENA Score | Описание |
|--------|------|---------|--------------|----------|
| **Micro ARIMA** | `models/micro_arima.py` | 0.415 | 0.701 (#14) | Внешняя микрокомпонентная ARIMA из `micro_test.csv` |

**Вывод:** Micro ARIMA значительно уступает нашим ML-моделям (+56% MAE на h=1).

---

## Новые возможности v4.2

### 1. Ridge Extended (лучшая модель)

```python
from sirena.models import RidgeExtendedForecaster

rex = RidgeExtendedForecaster()
rex.fit(df)

# Точечный прогноз
pred = rex.predict(df, target_date)

# Важность признаков
importance = rex.get_feature_importance()
# Топ новых признаков: quarter_sin, is_jan, y_lag3, d_y_lag3
```

### 2. Bayesian Ridge с доверительными интервалами

```python
from sirena.models import BayesianRidgeForecaster

br = BayesianRidgeForecaster()
br.fit(df)

# Прогноз с CI
pred = br.predict_with_ci(df, target_date)
print(f"Прогноз: {pred['prediction']:.3f}")
print(f"95% CI: [{pred['ci_lower']:.3f}, {pred['ci_upper']:.3f}]")
print(f"Std: {pred['std']:.3f}")

# Параметры Bayesian Ridge
params = br.get_model_params()
# alpha (noise precision), lambda (weight precision), sigma (noise std)
```

### 3. CatBoost для малых выборок

```python
from sirena.models import CatBoostForecaster

cb = CatBoostForecaster(
    iterations=200,
    depth=4,
    learning_rate=0.05,
    l2_leaf_reg=5.0
)
cb.fit(df)
fc = cb.forecast(horizon=12)

# Важность признаков
importance = cb.get_feature_importance()
```

### 4. Режимозависимый ансамбль

```python
from sirena.models import RegimeSwitchingEnsemble, detect_regime

# Определение текущего режима
regime, diag = detect_regime(df)
# regime: 'normal', 'shock', или 'high_inflation'

# Ансамбль с адаптивными весами
rs = RegimeSwitchingEnsemble()
rs.fit(df)
fc = rs.forecast_with_regime(horizon=12)

print(f"Режим: {rs.current_regime}")
print(f"Веса: {rs.current_weights}")

# История режимов
history = rs.get_regime_history(df)
```

---

## Возможности v4.1

### 1. Адаптивные веса ансамбля

```python
from sirena.forecast import EnsembleForecaster

ensemble = EnsembleForecaster()
# Оптимизация весов по MAE за последние 12 месяцев
new_weights = ensemble.optimize_weights(df, lookback_months=12)
# Формула: weight_i = (1/MAE_i²) / Σ(1/MAE_j²)
```

### 2. Стакинг (Meta-Model)

```python
from sirena.models import StackingForecaster

stacking = StackingForecaster(
    base_models=['ridge', 'bvar', 'lightgbm', 'ets'],
    meta_alpha=1.0,
    oof_start='2019-01-01'
)
stacking.fit(df)
fc = stacking.forecast(horizon=12)

# Веса meta-модели
weights = stacking.get_meta_weights()
```

### 3. Hierarchical Forecast (MinTrace)

```python
from sirena.models import HierarchicalForecaster

hier = HierarchicalForecaster()
hier.fit(df)

# Согласованные прогнозы всех уровней
fc = hier.forecast_all(horizon=12)
# Возвращает: total, food, nonfood, services

# Проверка когерентности
coherence = hier.check_coherence()
# Total == 0.39*Food + 0.37*NonFood + 0.23*Services
```

### 4. Нефть Brent

```python
from sirena.macro_features import load_brent_prices, add_brent_features

# Загрузка из Yahoo Finance (с кешем)
brent = load_brent_prices()

# Добавление признаков
df = add_brent_features(df)
# brent_lag3, brent_lag6, brent_pct_lag3, brent_pct_lag6
```

---

## REST API

### Эндпоинты

| Метод | Путь | Описание |
|-------|------|----------|
| GET | `/health` | Статус сервера |
| GET | `/models` | Список моделей и весов |
| GET | `/models/{name}` | Информация о модели |
| POST | `/forecast` | Прогноз ансамбля |
| GET | `/forecast/quick` | Быстрый прогноз (только Ridge) |
| POST | `/backtest` | Бэктестирование модели |
| GET | `/backtest/metrics/{model}` | Метрики качества |
| GET | `/weekly/signal` | Nowcasting сигнал |
| POST | `/weekly/forecast` | Прогноз недельных цен |

### Примеры запросов

```bash
# Прогноз
curl -X POST http://localhost:8000/forecast \
  -H "Content-Type: application/json" \
  -d '{"horizon": 12, "models": ["ridge", "bvar"]}'

# Быстрый прогноз
curl "http://localhost:8000/forecast/quick?horizon=6"

# Бэктест
curl -X POST http://localhost:8000/backtest \
  -H "Content-Type: application/json" \
  -d '{"model": "ridge", "start_date": "2023-01-01"}'
```

---

## Архитектура

### Структура проекта

```
opus_forecast/
├── api/                      # REST API (FastAPI)
│   ├── main.py
│   ├── routes/
│   └── schemas/
│
├── sirena/                   # Ядро системы
│   ├── models/               # Модели прогнозирования
│   │   ├── base.py           # BaseForecaster (ABC)
│   │   ├── registry.py       # ModelRegistry
│   │   ├── ridge.py          # Ridge (40%)
│   │   ├── bvar.py           # BVAR (20%)
│   │   ├── lightgbm.py       # LightGBM (15%)
│   │   ├── prophet.py        # Prophet (10%)
│   │   ├── arima.py          # SARIMA (5%)
│   │   ├── ets.py            # ETS (5%)
│   │   ├── ebm.py            # EBM (5%)
│   │   ├── stacking.py       # Stacking (experimental)
│   │   ├── hierarchical.py   # Hierarchical (experimental)
│   │   ├── ridge_sa.py       # Ridge SA (experimental)
│   │   └── weekly.py         # Weekly nowcasting
│   ├── forecast.py           # EnsembleForecaster + AdaptiveWeightOptimizer
│   ├── macro_features.py     # Ki, Ruonia, Brent
│   └── sa_data_loader.py     # SA данные
│
├── data/
│   ├── infl_kbr.csv          # Месячная инфляция КБР
│   ├── sa_fl.csv             # SA данные (2016-2025)
│   ├── inflation_data.csv    # Данные с макро
│   ├── weekly_prices.csv     # Недельные цены
│   └── brent_prices.csv      # Цены нефти (auto-generated)
│
├── tests/
├── dashboard.py              # Streamlit UI
└── requirements.txt
```

### Добавление новой модели

```python
from sirena.models import BaseForecaster, ModelRegistry

@ModelRegistry.register("my_model")
class MyModel(BaseForecaster):
    name = "my_model"

    def fit(self, df, target_col='Все товары и услуги'):
        self._is_fitted = True
        return self

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

**SA данные** `data/sa_fl.csv`:
```csv
Код;Товар;Дата;Значение
1;Все товары и услуги;01.01.2016;100.2
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
```

---

## Метрики качества (Backtest 2023-2025)

| Модель | MAE | vs Ridge |
|--------|-----|----------|
| **Ridge** | 0.316 | baseline |
| BVAR | 0.435 | +38% |
| Hierarchical | 0.437 | +38% |
| Ridge SA | 0.457 | +45% |
| Stacking | 0.465 | +47% |

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

# External Data
yfinance>=0.2.0
```

---

## Changelog

### v4.7 (2025-12-13)
- FundamentalForecaster: модель без жесткой сезонности
- USDForecaster: прогноз курса на основе нефти и ставок
- Аудит кодовой базы и проверка бэктестов

### v4.1 (2025-12-11)
- Адаптивные веса ансамбля (AdaptiveWeightOptimizer)
- Стакинг модель (StackingForecaster)
- Hierarchical Forecast с MinTrace
- Нефть Brent как экзогенная переменная
- Ridge SA модели (3 компонента, 47 субкомпонентов)
- EBM заменил LSTM

### v4.0 (2025-12-07)
- 7 моделей ансамбля
- REST API на FastAPI
- WeeklyForecaster
- Модульная архитектура

### v3.2 (2025-12-06)
- Streamlit Dashboard
- BVAR с Minnesota Prior

---

*Разработано в рамках проекта Opus Forecast*
