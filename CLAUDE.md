# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## КРИТИЧЕСКАЯ ДИРЕКТИВА: НЕ ВРАТЬ

**Это самое важное правило. Нарушение = потеря доверия пользователя.**

### Что запрещено

1. **Говорить "готово/сделано/работает" без РЕАЛЬНОЙ проверки того, что видит пользователь**
   - "Скрипт запустился" ≠ "UI работает"
   - "Модель выдаёт числа" ≠ "Вкладка отображается"
   - "Код без ошибок" ≠ "Функционал доступен пользователю"

2. **Создавать "верификацию" которая проверяет не то**
   - Если пользователь жалуется на вкладки 9-12 — проверяй код вкладок 9-12
   - Не создавай отдельный скрипт который тестирует что-то другое

3. **Оптимизировать на "выглядит как работа" вместо "реально работает"**

### Что обязательно

1. **Перед словом "готово" — проверь ИМЕННО то, что просил пользователь:**
   - Если добавлял вкладку в dashboard — открой код этой вкладки и убедись что он исполняется
   - Если добавлял модель в бэктест — найди строку где эта модель вызывается в коде бэктеста
   - Если что-то должно отображаться — найди add_trace/st.write/st.plotly_chart для этого

2. **Если не можешь проверить визуально (нет доступа к браузеру) — честно скажи:**
   > "Я изменил код, но не могу проверить отображение в браузере. Проверь вкладку X."

3. **Если пользователь говорит "не работает" — сначала ПОСМОТРИ код, потом отвечай**
   - Не говори "сейчас исправлю" пока не понял что сломано
   - Не создавай новые файлы пока не разобрался в существующих

### Примеры правильного поведения

**Плохо:** "Добавил модель в dashboard. Готово!"
**Хорошо:** "Добавил модель в dashboard.py строки 1082-1089. Добавил trace на график строка 1254. Добавил в бэктест строки 1350-1360. Обнови страницу и проверь вкладку 'Прогноз'."

**Плохо:** "Верификация пройдена, всё работает!"
**Хорошо:** "Мой скрипт проверяет только работу моделей. Он НЕ проверяет отображение вкладок в UI. Для проверки UI нужно открыть dashboard в браузере."

---

## При открытии новой сессии

**ВАЖНО:** При первом сообщении пользователя в новой сессии ВСЕГДА проверяй:

1. **`CURRENT_TASK.md`** — текущая задача и прогресс работы
   - Если файл существует и содержит незавершенные задачи — сообщи об этом
   - Предложи продолжить работу с того места, где остановились

2. **Последний git commit** — что было сделано в прошлой сессии

   ```bash
   git log -1 --pretty=format:"%s%n%n%b"
   ```

3. **`task`** — файл с заданиями (если есть)

**Пример ответа пользователю:**
> "Вижу незавершенную задачу в CURRENT_TASK.md: [краткое описание].
> Прогресс: [что сделано].
> Продолжить работу?"

## Git workflow

- После завершения и проверки задачи создай сфокусированный коммит и сразу отправь его в upstream.
- Если upstream не настроен, выполни `git push --set-upstream origin HEAD`.
- Не накапливай завершённые коммиты только локально.
- При ошибке push сохрани локальный коммит и сообщи точную ошибку.

## Обязательная верификация перед "Готово"

**КРИТИЧЕСКИ ВАЖНО:** Никогда не говори "готово" или "сделано" без автоматической проверки!

### После изменения моделей или dashboard

```bash
# Запустить верификацию ВСЕХ моделей
python3 scripts/verify_dashboard.py

# Проверить результаты:
# - data/verify_forecast.csv   — прогнозы (все 10 моделей должны быть)
# - data/verify_backtest.csv   — бэктест (Actual + все модели)
# - data/verify_summary.json   — status должен быть "OK", errors: []
```

### После добавления новой модели

1. Добавить модель в `verify_dashboard.py` (секции verify_forecasts и verify_backtest)
2. Запустить верификацию
3. Проверить что модель есть в CSV файлах
4. Обновить precomputed forecasts: `python3 scripts/precompute_forecasts.py`

### Что проверять в результатах

- [ ] Все модели присутствуют в verify_forecast.csv
- [ ] Все модели присутствуют в verify_backtest.csv
- [ ] Actual колонка не пустая в backtest
- [ ] MAE рассчитан для каждой модели
- [ ] status: "OK" в verify_summary.json
- [ ] errors: [] в verify_summary.json

**Не сообщай пользователю что "всё работает" пока не увидишь эти результаты своими глазами!**

### Визуальная верификация Dashboard (скриншоты)

```bash
# Сделать скриншоты ВСЕХ 12 вкладок dashboard
python3 scripts/screenshot_dashboard.py

# Результаты в assets/screenshots/:
# - tab1__Прогноз.png
# - tab3__Бэктест.png
# - tab9__Бэктест_h=1.png
# - tab10__Бэктест_h=2.png
# - tab11__Прогноз_h=1.png
# - tab12__Прогноз_h=2.png
```

**Когда использовать:**

- После изменений в dashboard.py
- Когда пользователь говорит "вкладка не работает"
- Перед заявлением "dashboard готов"

### ОБЯЗАТЕЛЬНАЯ ПОЛНАЯ ВЕРИФИКАЦИЯ

**ПЕРЕД ЛЮБЫМ "ГОТОВО" — ВЫПОЛНИТЬ:**

```bash
python3 scripts/verify_all_tabs.py
```

Скрипт проверяет:

1. `data/precomputed_forecasts.json` — все модели есть
2. `archive/results/backtest_h1_predictions.csv` — Micro есть
3. `archive/results/backtest_h2_predictions.csv` — Micro есть
4. `dashboard.py` — ALL_MODELS определён, нет захардкоженных списков
5. `scripts/backtest_framework.py` — Micro импортирован и используется
6. Скриншоты всех 12 вкладок

**Результат должен быть: ✅ ВСЕ ПРОВЕРКИ ПРОЙДЕНЫ**

Если есть ❌ — ИСПРАВИТЬ ДО "ГОТОВО"!

### ОБЯЗАТЕЛЬНО: Обновление HTML графиков

**КРИТИЧЕСКИ ВАЖНО:** Пользователь отслеживает работу через HTML файлы в `assets/charts/`.

**После ЛЮБОГО изменения моделей или прогнозов — ОБЯЗАТЕЛЬНО выполнить:**

```bash
# 1. Обновить CSV с прогнозами (если менялись модели)
python3 scripts/precompute_forecasts.py

# 2. Перегенерировать ВСЕ HTML графики
python3 scripts/generate_charts.py
```

**Что генерируется:**

- `assets/charts/forecasts.html` — прогнозы всех моделей
- `assets/charts/backtest_h1_predictions.html` — бэктест h=1
- `assets/charts/backtest_h2_predictions.html` — бэктест h=2
- `assets/charts/backtest_h12_predictions.html` — бэктест h=12
- `assets/charts/backtest_h*_errors.html` — ошибки
- `assets/charts/metrics_comparison.html` — сравнение MAE
- `assets/charts/ranking_heatmap.html` — ранжирование
- `assets/charts/last_month_errors.html` — ошибки последнего месяца
- `assets/charts/index.html` — главная страница

**НИКОГДА не говори "готово" если не выполнил `python3 scripts/generate_charts.py`!**

Пользователь проверяет результаты ТОЛЬКО через эти HTML файлы. Если их не обновить — он увидит старые данные и будет справедливо недоволен.

### При добавлении новой модели — ОБЯЗАТЕЛЬНЫЙ ЧЕКЛИСТ

**КРИТИЧЕСКИ ВАЖНО:** Используй скрипт проверки после КАЖДОГО добавления модели!

```bash
# Проверить что модель добавлена ВЕЗДЕ (11 мест!)
python3 scripts/add_model_checklist.py ModelName
```

**11 обязательных мест для новой модели:**

| # | Файл | Что добавить |
|---|------|--------------|
| 1 | `sirena/models/{name}.py` | Файл модели |
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

**После добавления кода — ОБЯЗАТЕЛЬНО:**

```bash
# 1. Перезапустить ВСЕ бэктесты
python3 scripts/run_backtest_h1.py
python3 scripts/run_backtest_h2.py
python3 scripts/run_backtest_h12.py

# 2. Проверить что всё на месте
python3 scripts/add_model_checklist.py ModelName

# 3. Результат должен быть: ВСЕ ПРОВЕРКИ ПРОЙДЕНЫ: 11/11
```

**НЕ ГОВОРИТЬ "ГОТОВО" ПОКА НЕ ПРОЙДЕНЫ ВСЕ 11 ПРОВЕРОК!**

Полная инструкция: `.claude/skills/add-model.md`

---

## Project Overview

СИРЕНА-КБР v5.4 — система прогнозирования инфляции (ИПЦ) в Кабардино-Балкарской Республике.

**Новое в v5.4:**

- Данные обновлены до **января 2026**
- Обновлены веса компонентов корзины (Январь 2026)
- **Subcomp** — лучшая модель h=1 (MAE 0.309)
- **Subcomp_Multi** — лучшая модель h=12 (MAE 0.297)

**Лидеры по горизонтам (бэктест янв-дек 2025):**

- h=1: Subcomp (MAE 0.309)
- h=2: NGBoost (MAE 0.290)
- h=12: Subcomp_Multi (MAE 0.297)

**Ансамбль v5.0 (9 моделей, веса на основе h=1 backtest MAE):**

- Subcomp: 18% (лучшая h=1, MAE 0.309)
- Ridge_Macro: 17% (MAE 0.319)
- RidgeShockDummies: 17% (MAE 0.319)
- Ridge: 15% (baseline, MAE 0.321)
- Huber: 12% (MAE 0.324)
- NGBoost: 10% (лучшая h=2, MAE 0.290)
- Prophet: 6% (MAE 0.310 на h=12)
- EBM: 5% (MAE 0.336)

**Экспериментальные:** HorizonEnsemble, Micro, HierMicro, MicroOptimized

**Убраны из ансамбля:** BVAR, LMMR, ETS, SARIMA, LightGBM, CatBoost

## Документация

**Главная страница:** [docs/index.md](docs/index.md)

| Документ | Описание |
|----------|----------|
| [docs/MODELS.md](docs/MODELS.md) | Полный список моделей, веса в ансамбле |
| [docs/BACKTEST_METHODOLOGY.md](docs/BACKTEST_METHODOLOGY.md) | Методика бэктестов h=1, h=2, h=12 |
| [docs/NOWCASTING.md](docs/NOWCASTING.md) | **Nowcasting** — использование недельных данных |
| [docs/API.md](docs/API.md) | REST API эндпоинты |
| [docs/FORMATS.md](docs/FORMATS.md) | Форматы входных данных |

## Running Dashboard

**ВАЖНО: Dashboard запущен на порту 8503**

```bash
# Dashboard уже работает на:
http://localhost:8503

# Если нужно перезапустить:
streamlit run dashboard.py --server.port 8503

# Сброс кэша Streamlit (после изменения моделей):
# В браузере: Settings (⚙️) → Clear cache
# Или: нажать "C" на клавиатуре в dashboard
```

## Добавление новой модели в Dashboard

**ВАЖНО: При добавлении новой модели нужно изменить ТРИ места в dashboard.py!**

### 1. Прогноз (tab1) — создание прогноза модели

Найти блок `with st.spinner("Расчет ансамбля моделей..."):` (~строка 854) и добавить:

```python
# Создать DataFrame с прогнозом новой модели
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

### 2. Прогноз (tab1) — добавить в ансамбль model_weights

Найти `model_weights = {` (~строка 953) и добавить модель:

```python
model_weights = {
    'NewModel': (new_model_df['NewModel'].values if new_model_df is not None else None, 0.10),
    # ... остальные модели
}
```

### 3. Прогноз (tab1) — добавить trace на график

Найти секцию `# Add Traces` (~строка 981) и добавить:

```python
if new_model_df is not None:
    fig_fc.add_trace(go.Scatter(
        x=new_model_df['Date'], y=new_model_df['NewModel'],
        name='NewModel', line=dict(color='#hexcolor', width=2)
    ))
```

### 4. Бэктест (tab3) — импорт модели

Найти импорты в `run_comparative_backtest_cached()` (~строка 1280) и добавить:

```python
from sirena.models.new_model import NewModelForecaster
```

### 5. Бэктест (tab3) — прогнозирование в цикле

Найти цикл `for date in test_dates:` и добавить после других моделей:

```python
# NewModel
try:
    model_new = NewModelForecaster()
    model_new.fit(train_r, 'Все товары и услуги')
    pred_new = model_new.predict(train_r_ext, date)['prediction'] - 100
except: pred_new = np.nan
```

### 6. Бэктест (tab3) — добавить в results.append()

Найти `results.append({` и добавить:

```python
'NewModel': pred_new,
```

### 7. Бэктест (tab3) — добавить в all_models и model_colors

```python
all_models = ['Ridge', 'NewModel', ...]  # строка ~1465
model_colors = {
    'NewModel': '#hexcolor',  # строка ~1458
    ...
}
```

### 8. Обновить версию кэша

Изменить `BACKTEST_CACHE_VERSION` (~строка 1247):

```python
BACKTEST_CACHE_VERSION = "v4.X.X_new_model"
```

### Чеклист добавления модели

- [ ] Создать файл модели в `sirena/models/`
- [ ] Добавить импорт в `sirena/models/__init__.py`
- [ ] **Dashboard tab1**: создание прогноза
- [ ] **Dashboard tab1**: добавить в `model_weights`
- [ ] **Dashboard tab1**: добавить trace на график
- [ ] **Dashboard tab3**: импорт модели
- [ ] **Dashboard tab3**: прогноз в цикле
- [ ] **Dashboard tab3**: добавить в `results.append()`
- [ ] **Dashboard tab3**: добавить в `all_models`
- [ ] **Dashboard tab3**: добавить в `model_colors`
- [ ] **Dashboard tab3**: обновить `BACKTEST_CACHE_VERSION`
- [ ] Обновить документацию CLAUDE.md

## File Organization & Cleanliness

**CRITICAL: Do not clutter the project root!**

- **Root Directory**: Only essential files (`README.md`, `CLAUDE.md`, `dashboard.py`, `requirements.txt`, `config.json`).
- **Scripts**: All analysis, one-off scripts, and experiments must go to `archive/scripts/` or `scripts/`.
- **Data**: All CSV/Excel files must be in `data/`.
- **Images**: All plots and diagrams must be in `assets/images/`.
- **Results**: Logs and output CSVs should go to `archive/results/`.
- **Docs**: Old documentation and reports go to `archive/docs/`.

## Documentation Standards

**Source of Truth:** `docs/index.md` is the central entry point for all project documentation.

**Rules:**

1. **Always Update docs/**: When changing models or API, update the corresponding file in `docs/`.
2. **Maintain index.md**: Ensure new documentation files are linked in `docs/index.md`.
3. **Archive Obsolete**: Move outdated docs to `archive/docs/` instead of deleting them.
4. **Key Files**:
   - `docs/MODELS.md`: Active models list & description.
   - `docs/BACKTEST_METHODOLOGY.md`: Validation rules.
   - `docs/API.md`: API reference.

## Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Run Streamlit dashboard (порт 8503!)
streamlit run dashboard.py --server.port 8503

# Run Streamlit dashboard (LAN access)
streamlit run dashboard.py --server.address 0.0.0.0 --server.port 8503

# Run REST API
uvicorn api.main:app --reload --port 8000

# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_models.py -v
pytest tests/test_api.py -v

# Run single test
pytest tests/test_models.py::TestSirenaKBR::test_fit -v
```

## Architecture

### Core Package: `sirena/`

The forecasting engine follows a **Strategy + Factory** pattern:

- **`models/base.py`** — `BaseForecaster` abstract class. All models must implement `fit()`, `forecast(horizon)`, `backtest()`.
- **`models/registry.py`** — `ModelRegistry` factory. Use `@ModelRegistry.register("name")` decorator to add models.
- **`forecast.py`** — `EnsembleForecaster` combines model outputs with configurable weights + `AdaptiveWeightOptimizer`.
- **`macro_features.py`** — Макро-признаки (Ki, Ruonia, Brent).
- **`sa_data_loader.py`** — Загрузчик SA (сезонно-скорректированных) данных.

### Models (17 total)

#### Production Models (Default Ensemble)

| Model | File | Weight | Description |
|-------|------|--------|-------------|
| Ridge | `models/ridge.py` | 40% | Ridge regression + ETS seasonality |
| BVAR | `models/bvar.py` | 20% | Bayesian VAR, Minnesota Prior |
| LightGBM | `models/lightgbm.py` | 15% | Gradient Boosting |
| Prophet | `models/prophet.py` | 10% | Facebook Prophet |
| SARIMA | `models/arima.py` | 5% | Seasonal ARIMA |
| ETS | `models/ets.py` | 5% | Exponential Smoothing |
| EBM | `models/ebm.py` | 5% | Explainable Boosting Machine |

#### New in v4.7

| Model | File | MAE | Description |
|-------|------|-----|-------------|
| Fundamental | `models/fundamental.py` | 0.389 | Structural model (USD + Oil + Key Rate), no hardcoded seasonality |
| USDForecaster | `models/usd_model.py` | 2.56 | ElasticNet for USD/RUB (Oil + Rates) |

#### New in v4.3

| Model | File | MAE | Description |
|-------|------|-----|-------------|
| Ridge Extended v2 | `models/ridge_extended.py` | 0.313 | Расширенные календарные признаки (is_tariff_month, is_q1, is_summer) |
| ElasticNet | `models/elasticnet.py` | 0.320 | L1+L2 регуляризация с автоматическим feature selection |
| Huber | `models/huber.py` | 0.335 | Робастная к выбросам (Huber loss) |

#### New in v4.2

| Model | File | MAE | Description |
|-------|------|-----|-------------|
| Bayesian Ridge | `models/bayesian_ridge.py` | 0.321 | Автоматическая регуляризация + доверительные интервалы |
| CatBoost | `models/catboost_model.py` | 0.368 | Gradient boosting для малых выборок |
| Regime Switching | `models/regime_switching.py` | — | Режимозависимые веса ансамбля (shock/normal/high_inflation) |

#### Experimental Models

| Model | File | Description |
|-------|------|-------------|
| Stacking | `models/stacking.py` | Meta-learning на прогнозах базовых моделей |
| Hierarchical | `models/hierarchical.py` | MinTrace reconciliation по компонентам |
| Ridge SA | `models/ridge_sa.py` | Bottom-up по 3 компонентам на SA данных |
| Ridge SA Sub | `models/ridge_sa_sub.py` | Bottom-up по 47 субкомпонентам |
| Weekly | `models/weekly.py` | Nowcasting на недельных ценах |

| Weekly | `models/weekly.py` | Nowcasting на недельных ценах |

### New in v4.7: Structural Models

#### 1. FundamentalForecaster (`sirena/models/fundamental.py`)

Модель на экономических драйверах, устойчивая к изменению сезонности:

```python
from sirena.models import FundamentalForecaster

fund = FundamentalForecaster()
fund.fit(df)
# Uses: Forecasted USD, Brent Oil, Key Rate
```

#### 2. USDForecaster (`sirena/models/usd_model.py`)

Прогноз курса доллара для использования в других моделях:

```python
from sirena.models import USDForecaster

usd = USDForecaster()
usd.fit(df)
# Uses: Brent Oil, Key Rate differentials
```

### New in v4.2: Extended Models

#### 1. Ridge Extended (`sirena/models/ridge_extended.py`)

Лучшая модель v4.2 (MAE 0.310, -1.8% vs Ridge baseline):

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

**Новые признаки:**

- Лаги: `y_lag3`, `y_lag6`
- Momentum: `d_y_lag1`, `d_y_lag3`
- Volatility: `y_vol3`, `y_vol6`
- Календарь: `is_jan`, `is_dec`, `quarter_sin`, `quarter_cos`

#### 2. Bayesian Ridge (`sirena/models/bayesian_ridge.py`)

Ridge с автоматической регуляризацией и доверительными интервалами:

```python
from sirena.models import BayesianRidgeForecaster

br = BayesianRidgeForecaster()
br.fit(df)

# Прогноз с CI
pred = br.predict_with_ci(df, target_date)
print(f"Прогноз: {pred['prediction']:.3f}")
print(f"95% CI: [{pred['ci_lower']:.3f}, {pred['ci_upper']:.3f}]")
print(f"Std: {pred['std']:.3f}")

# Параметры модели
params = br.get_model_params()  # alpha, lambda, sigma
```

#### 3. CatBoost (`sirena/models/catboost_model.py`)

Gradient boosting оптимизированный для малых выборок (~150 точек):

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

Требует установки: `pip install catboost`

#### 4. Regime Switching (`sirena/models/regime_switching.py`)

Режимозависимый ансамбль с адаптивными весами:

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

**Режимы и веса:**

- `shock` (|ΔRuonia| > 0.5 или |ΔKi| > 0.5): BVAR 35%, Ridge 25%, SARIMA 15%
- `normal`: Ridge 50%, ETS 15%, BVAR 10%
- `high_inflation` (ΔИнфляция > 1.5pp): BVAR 30%, Ridge 30%, SARIMA 15%

---

### Systemic Repairs (v1.4)

**1. Refiner Circuit Breaker**
To prevent infinite refinement loops, the Refiner now tracks `refinement_attempts` in `prd.json`.

- **Policy**: Tasks are skipped after 3 failed refinement attempts.
- **Troubleshooting**: If a task is stuck in `BLOCKED` with logs saying "Max attempts reached", check `refinement_attempts` in JSON.

**2. Python Import Safety**
Global `edge_lab/scripts/__init__.py` exists to ensure `from scripts import x` works correctly during verification.

**3. Process Verification Protocol (CRITICAL)**
Before reporting "system active", you MUST check for duplicates:

- **Command**: `ps -ef | grep orchestrator.py | grep -v grep`
- **Rule**: If output lines > 1 (excluding children of the same tree), verify PIDs.
- **Action**: Run `pkill -f edge_lab` to kill ALL processes, then restart ONE instance.
- **Never** rely on just `ps` showing "a process". Ensure it is the **only** process tree.

---

### New in v4.1: Advanced Features

#### 1. Adaptive Weights (`sirena/forecast.py`)

```python
from sirena.forecast import EnsembleForecaster

ensemble = EnsembleForecaster()
# Оптимизация весов по MAE за последние 12 месяцев
new_weights = ensemble.optimize_weights(df, lookback_months=12)
# Формула: weight_i = (1/MAE_i²) / Σ(1/MAE_j²)
```

#### 2. Brent Oil Prices (`sirena/macro_features.py`)

```python
from sirena.macro_features import load_brent_prices, add_brent_features

# Загрузка цен нефти из Yahoo Finance
brent_df = load_brent_prices()  # с автоматическим кешем

# Добавление признаков в данные
df = add_brent_features(df)
# Признаки: brent_lag3, brent_lag6, brent_pct_lag3, brent_pct_lag6
```

#### 3. Stacking Meta-Model (`sirena/models/stacking.py`)

```python
from sirena.models import StackingForecaster

# Двухуровневая модель
stacking = StackingForecaster(
    base_models=['ridge', 'bvar', 'lightgbm', 'ets'],
    meta_alpha=1.0,  # регуляризация
    oof_start='2019-01-01'
)
stacking.fit(df)
fc = stacking.forecast(horizon=12)

# Веса meta-модели
weights = stacking.get_meta_weights()
```

#### 4. Hierarchical Forecast (`sirena/models/hierarchical.py`)

```python
from sirena.models import HierarchicalForecaster

# MinTrace reconciliation
hier = HierarchicalForecaster()
hier.fit(df)

# Прогноз всех уровней
fc = hier.forecast_all(horizon=12)
# Возвращает: total, food, nonfood, services (согласованные)

# Проверка когерентности
coherence = hier.check_coherence()
# Total == w1*Food + w2*NonFood + w3*Services
```

### REST API: `api/`

FastAPI application with routes in `api/routes/`:

- `forecast.py` — POST /forecast, GET /forecast/quick
- `backtest.py` — POST /backtest, GET /backtest/metrics/{model}
- `models.py` — GET /models, GET /models/{name}
- `weekly.py` — GET /weekly/signal, POST /weekly/forecast, GET /weekly/products

Pydantic schemas in `api/schemas/`.

### Dashboard Tabs

11 tabs in `dashboard.py`:

1. 📈 Прогноз — 12-month forecast with 7-model ensemble
2. 🔧 Экзогенные — Manual USD/Ki/RUONIA input
3. ✅ Бэктест — Model comparison and metrics
4. 🛠 Методология — Model documentation
5. 🧠 История (Opus) — Forecast decomposition
6. 🗺 Регионы — Cross-regional analysis
7. 🔒 Инсайдер — Forecast with preliminary data (inflation_data.csv)
8. 🔍 EBM — EBM model feature importance and predictions
9. 📊 Бэктест h=1 — Rolling backtest 1 month ahead (MAIN KPI)
10. 📊 Бэктест h=2 — Rolling backtest 2 months ahead
11. 📊 Бэктест h=12 — 12-month trajectory backtest (fixed cutoff)

### Data Flow

```
data/infl_kbr.csv (CSV, sep=';', decimal=',')
        ↓
    DataLoader
        ↓
EnsembleForecaster → ModelRegistry.get("ridge") → RidgeForecaster.fit() → forecast()
        ↓
    API / Dashboard
```

### Input Data Format

**Monthly data** `data/infl_kbr.csv`:

- Columns: `Day` (dd.mm.yyyy), `Товар` (category), `MoM` (month-over-month index)
- Categories: "Все товары и услуги", "Продовольственные товары", "Непродовольственные товары", "Услуги"

**SA data** `data/sa_fl.csv`:

- Сезонно-скорректированные данные
- Columns: Код, Товар, Дата, Значение
- Period: 2016-01 — 2025-10

**Insider data** `data/inflation_data.csv`:

- Columns: Date, mom, Nonprod, Prod, Serv, usd_nom_i, Ki_i, Ruonia, etc.
- Contains preliminary/insider data for "Инсайдер" tab

**Weekly prices** `data/weekly_prices.csv`:

- Columns: Товары, Сведено (YYYY_WW), Значение, Rostat_code
- Used by `WeeklyForecaster` for nowcasting

**Brent prices** `data/brent_prices.csv` (auto-generated):

- Downloaded from Yahoo Finance
- Columns: Date, brent, brent_pct

## Adding a New Model

```python
from sirena.models import BaseForecaster, ModelRegistry

@ModelRegistry.register("my_model")
class MyModel(BaseForecaster):
    name = "my_model"

    def fit(self, df, target_col='Все товары и услуги'):
        # Training logic
        self._is_fitted = True
        return self

    def forecast(self, horizon=12):
        self._check_fitted()
        return np.array([...])  # MoM values

    def backtest(self, df, start_date='2019-01-01', target_col='...'):
        # Returns DataFrame with: date, actual, prediction, error
        return pd.DataFrame({...})
```

## Key Constants

- `MIN_TRAIN_SIZE = 24` — minimum observations for training
- Outlier years excluded from training: 2010, 2022
- Target column: `'Все товары и услуги'`
- Default forecast horizon: 12 months
- Component weights (January 2026): Food 39.86%, NonFood 36.38%, Services 23.76%

## Model Performance (Backtest 2025-01 — 2025-12)

### h=1 (1 месяц вперед — ГЛАВНЫЙ КПЭ)

| Model | MAE | vs Ridge | Notes |
|-------|-----|----------|-------|
| **🏆 Subcomp** | **0.309** | **-3.7%** | Best model h=1 (v5.0) |
| Ridge_Macro | 0.319 | -0.6% | Макро-признаки |
| Ridge Shock | 0.319 | -0.6% | Shock dummies из методик ЦБ |
| Ridge | 0.321 | baseline | Production baseline |
| Ridge Extended | 0.322 | +0.3% | Расширенные признаки |
| Huber | 0.324 | +0.9% | Робастная к выбросам |
| Subcomp_Multi | 0.326 | +1.6% | Лучшая на h=12 |

### h=2 (2 месяца вперед)

| Model | MAE | vs Ridge | Notes |
|-------|-----|----------|-------|
| **🏆 NGBoost** | **0.290** | **-9.6%** | Best model h=2 |
| Ridge Extended | 0.295 | -8.1% | Расширенные признаки |
| Huber | 0.297 | -7.5% | Робастная |
| ElasticNet | 0.308 | -4.0% | L1+L2 регуляризация |

### h=12 (годовая траектория)

| Model | MAE | vs Ridge | Notes |
|-------|-----|----------|-------|
| **🏆 Subcomp_Multi** | **0.297** | **-17.3%** | Best model h=12 (v5.0) |
| Prophet | 0.310 | -13.6% | Стабильная (1 KPI violation) |
| Subcomp | 0.324 | -9.7% | Субкомпоненты |
| EBM | 0.336 | -6.4% | Интерпретируемая |

## v4.4 New Models

### NGBoost (лучшая модель!)

```python
from sirena.models import NGBoostForecaster

ngb = NGBoostForecaster()
ngb.fit(df)

# Прогноз с вероятностным распределением
pred = ngb.predict(df_ext, target_date)
print(f"Прогноз: {pred['prediction']:.2f}")
print(f"90% CI: [{pred['ci_lower']:.2f}, {pred['ci_upper']:.2f}]")
print(f"Std: {pred['std']:.3f}")
```

### Quantile Ridge (асимметричные CI)

```python
from sirena.models import QuantileRidgeForecaster

qr = QuantileRidgeForecaster()
qr.fit(df)
pred = qr.predict(df_ext, target_date)
# ci_lower = 10% квантиль, ci_upper = 90% квантиль
```

### Constrained Components

```python
from sirena.models import ConstrainedComponentsForecaster

cc = ConstrainedComponentsForecaster()
cc.fit(df)
# Прогноз Total + 3 компонента с ограничением когерентности
```

## v4.5 New Models

### Deep Ensemble (лучшая модель!)

```python
from sirena.models import DeepEnsembleForecaster

de = DeepEnsembleForecaster(n_models=5)
de.fit(df)

# Прогноз с агрегированием по ансамблю
pred = de.predict(df_ext, target_date)
print(f"Прогноз: {pred['prediction']:.2f}")
print(f"Epistemic std: {pred['epistemic_std']:.3f}")  # uncertainty между моделями
print(f"Aleatoric std: {pred['aleatoric_std']:.3f}")  # внутренняя uncertainty
```

### Conformal Prediction (калиброванные CI)

```python
from sirena.models import ConformalForecaster

cf = ConformalForecaster(coverage_target=0.90)
cf.fit(df)

# Прогноз с гарантированным 90% coverage
pred = cf.predict(df_ext, target_date)
print(f"Прогноз: {pred['prediction']:.2f}")
print(f"90% CI: [{pred['ci_lower']:.2f}, {pred['ci_upper']:.2f}]")
# CI калиброваны — 91% точек реально попадают в интервал!
```

### XGBoost

```python
from sirena.models import XGBoostForecaster

xgb = XGBoostForecaster()
xgb.fit(df)
pred = xgb.predict(df_ext, target_date)
```

## v4.6 New Models (Эксперименты из методик ЦБ)

### Ridge Shock Dummies (лучшая модель!)

Модель с dummy-переменными для исторических шоков (из R-кода методик):

```python
from sirena.models import RidgeShockDummiesForecaster

# use_2022_dummy=False — исключаем 2022, но используем dummies для 2014-2015
model = RidgeShockDummiesForecaster(use_2022_dummy=False)
model.fit(df)
pred = model.predict(df_ext, target_date)

# Важность shock dummies
importance = model.get_feature_importance()
# is_shock_jan2015: +1.96 — инфляция на ~2 п.п. выше нормы
# is_shock_dec2014: +0.90 — валютный кризис
# is_tariff_jul2017: +0.18 — июльская индексация тарифов
```

**Shock dummies:**

- `is_shock_dec2014`: Декабрь 2014 (валютный кризис)
- `is_shock_jan2015`: Январь 2015 (продолжение шока)
- `is_tariff_jul2017`: Июль — индексация тарифов ЖКХ (каждый год)
- `is_shock_mar2022`, `is_shock_apr2022`: Санкционный шок (опционально)

### Ridge Base Index (НЕ РАБОТАЕТ)

Модель на базисных индексах из методик ЦБ:

```python
from sirena.models import RidgeBaseIndexForecaster

model = RidgeBaseIndexForecaster(use_log=False)  # или use_log=True для логарифма
model.fit(df)
```

**Вывод:** Базисные индексы ухудшают прогноз (+14% MAE), несмотря на использование в методиках.

## v4.6 Experiments Summary

**Что сработало:**

- ✅ **Ridge Shock Dummies: MAE 0.300 (-5.1% vs baseline) — НОВАЯ ЛУЧШАЯ МОДЕЛЬ!**
- ✅ Dummy-переменные для исторических шоков (2014-2015) помогают модели
- ✅ Июльский dummy (is_tariff_jul2017) для индексации тарифов

**Что не сработало:**

- ❌ Ridge Base Index: MAE +14.1% — базисные индексы не улучшают прогноз
- ❌ Ridge Base Index + Log: MAE +79.8% — логарифмирование ещё хуже
- ❌ Shock Dummies с включением 2022: MAE -2.4% — лучше исключать 2022 полностью

**Ключевой вывод из методик ЦБ:**
Dummy-переменные для исторических шоков (2014-2015) работают ЛУЧШЕ чем:

- Исключение 2022 года целиком (текущий подход)
- Включение 2022 с dummy-переменными

## v4.5 Experiments Summary

**Что сработало:**

- ✅ Deep Ensemble: MAE 0.305 (-3.3% vs baseline) — лучшая модель!
- ✅ Conformal Prediction: 91% CI coverage (vs target 90%)
- ✅ XGBoost: MAE -0.2% — небольшое улучшение

**Что не сработало:**

- ❌ Ridge Components: MAE +2.1% — слишком много признаков (overfitting)
- ❌ MSTL + Ridge: MAE +44% — декомпозиция не подходит для этих данных
- ❌ Theta Method: MAE +26% — слишком простая модель
- ❌ DMA: MAE +3.2% — адаптивные веса нестабильны

## v4.4 Experiments Summary

**Что сработало:**

- ✅ NGBoost: MAE 0.306 (-3.0% vs baseline)
- ✅ Probabilistic boosting с естественными CI

**Что не сработало:**

- ❌ Ridge v3 с dummy-переменными: MAE +8% — лучше исключать выбросы полностью
- ❌ Constrained Components: MAE +5% — накопление ошибок компонентов
- ❌ Quantile Ridge: MAE +7.4% — полезен для CI, но не для точности

## v4.3 Experiments Summary

**Что сработало:**

- ✅ Календарные признаки: `is_tariff_month` (июль), `is_q1`, `is_summer` → MAE -0.7%
- ✅ ElasticNet с CV feature selection (12 из 25 признаков)

**Что не сработало:**

- ❌ Sample Weighting (2022 с весом 0.25): MAE +8.4% — хуже полного исключения
- ❌ Huber: MAE +6.0% — исключение 2022 лучше чем Huber loss

---

## v4.8 МАСШТАБНОЕ ИССЛЕДОВАНИЕ ПРИЗНАКОВ И МОДЕЛЕЙ (2025-12-28)

**Цель:** Раз и навсегда определить оптимальные комбинации для прогнозирования инфляции КБР.

**Объём:** 588+ комбинаций, 17 моделей, 23 набора признаков, 3 горизонта.

### Лучшие комбинации по горизонтам

| Горизонт | Модель | Признаки | MAE | vs Baseline |
|----------|--------|----------|-----|-------------|
| **h=1** | Voting (Ridge+Lasso+Huber) | Components_momentum | **0.461** | **-15.7%** |
| **h=2** | NGBoost | Brent_focus | **0.512** | **-12.0%** |
| **h=12** | Stacking | USD_focus | **0.343** | **-12.7%** |

### Рейтинг моделей (по среднему MAE)

| Модель | Средний MAE | Комментарий |
|--------|-------------|-------------|
| Lasso | 0.482 | Простая и эффективная |
| Voting | 0.484 | Лучший выбор для стабильности |
| ElasticNet | 0.492 | L1+L2 регуляризация |
| Stacking | 0.493 | Хорош для h=12 |
| Ridge_100 | 0.490 | Сильная регуляризация |
| Huber | 0.498 | Робастен к выбросам |
| NGBoost | 0.523 | Хорош для h=2, нестабилен на h=1 |

### Рейтинг признаков (по среднему MAE)

| Набор | Средний MAE | Улучшение |
|-------|-------------|-----------|
| **Components** | 0.505 | -12.3% |
| **Components_momentum** | 0.510 | -11.5% |
| AR_extended | 0.512 | -11.1% |
| USD_focus | 0.515 | -10.6% |
| Brent_focus | 0.514 | -10.8% |
| IBVED_focus | 0.527 | -8.5% |

### Что работает ✓

**Признаки:**

- **Компоненты инфляции** (Prod, Nonprod, Serv) — САМЫЕ ВАЖНЫЕ
- **Momentum** (mom_D1, mom_D3) — улучшает на 1-2%
- **Курс доллара** — особенно важен для h=12
- **Нефть Brent** — важна для h=2

**Модели:**

- **Voting/Stacking ансамбли** — лучший выбор
- **Lasso/ElasticNet** — простые и эффективные

### Что НЕ работает ✗

**Модели (избегать):**

- ❌ XGBoost — переобучение, MAE +12%
- ❌ RandomForest — нестабилен, MAE +2%
- ❌ CatBoost — переобучение, MAE +3%
- ❌ GradientBoosting — переобучение, MAE +8%

**Признаки (бесполезны):**

- ❌ Ki (ключевая ставка) — нет улучшения
- ❌ Ruonia — нет улучшения
- ❌ Региональные показатели КБР (month.csv) — ухудшают на 7-15%
- ❌ **ИБВЭД** (quart.csv indicator 20, month.csv indicator 21) — **НЕТ КОРРЕЛЯЦИИ** с инфляцией (|r| < 0.2, p > 0.2)

### Рекомендуемые наборы признаков

```python
# h=1: Components_momentum (лучший)
features_h1 = ['mom_L1', 'mom_L2', 'mom_D1', 'mom_D3',
               'Prod_L1', 'Nonprod_L1', 'Serv_L1',
               'Prod_D1', 'Nonprod_D1', 'Serv_D1',
               'month_sin', 'month_cos', 'is_jan', 'is_jul']

# h=2: Brent_focus
features_h2 = ['mom_L1', 'mom_L2',
               'brent_L1', 'brent_L3', 'brent_L6',
               'brent_D1', 'brent_D3',
               'month_sin', 'month_cos']

# h=12: USD_focus
features_h12 = ['mom_L1', 'mom_L2',
                'usd_nom_i_L1', 'usd_nom_i_L2', 'usd_nom_i_L6',
                'usd_nom_i_D1', 'usd_nom_i_D3',
                'month_sin', 'month_cos']
```

### Результаты сохранены

- `archive/results/research/FINAL_RESEARCH_REPORT.md` — полный отчёт
- `archive/results/research/model_comparison_full.csv` — 273 комбинации
- `archive/results/research/extended_comparison.csv` — 315 комбинаций

---

## Субкомпоненты (3-й уровень)

### Структура данных

- **45 субкомпонентов** с кодами 11-67
- **Веса** в `data/raw/sub_weight.csv`
- **MoM данные** в `data/raw/sub_mom.csv` (с 2010-01)
- **Справочник** в `data/raw/subcomp_sprav.csv`

### Группировка по компонентам

| Компонент | Количество | Суммарный вес |
|-----------|------------|---------------|
| Продовольственные товары | 15 | 39.48% |
| Непродовольственные товары | 20 | 36.53% |
| Услуги | 10 | 23.99% |

### Топ субкомпоненты по весу

| Код | Название | Вес |
|-----|----------|-----|
| 26 | Мясопродукты | 9.90% |
| 14 | Жилищные и коммунальные услуги | 9.09% |
| 29 | Одежда и белье | 6.47% |
| 33 | Плодоовощная продукция | 5.89% |
| 12 | Бытовые услуги | 4.37% |
| 54 | Другие непродовольственные товары | 4.45% |

### Эксперимент: 45 субкомпонентов с индивидуальными подходами (2025-12-29)

**Идея:** Для каждого из 45 субкомпонентов подобрать оптимальный набор признаков (USD, Brent, сезонность, тарифы, ДКП), затем агрегировать по весам.

**Результаты:**

| Горизонт | Оптимизированный | Baseline Subcomp | Direct | vs Baseline | vs Direct |
|----------|------------------|------------------|--------|-------------|-----------|
| **h=1** | **0.426** | 0.566 | 0.572 | **+24.7%** ✅ | **+25.5%** ✅ |
| h=2 | 0.556 | 0.571 | 0.566 | +2.5% | +1.7% |
| h=12 | 0.410 | 0.373 | 0.439 | -9.9% | +6.7% |

**Оптимальные подходы по субкомпонентам (h=1):**

| Подход | Вес | Субкомпоненты | Ср. улучшение |
|--------|-----|---------------|---------------|
| **usd** | 32% | Одежда, Электротовары, Молоко, Кондитерские... | +3.1% |
| baseline | 31% | ЖКХ, Мясо, Телекоммуникации, Мебель... | 0% |
| **brent** | 13% | Бытовые услуги, Транспорт, Масло, Рыба... | **+13.6%** |
| seasonal | 9% | Автомобили, Кондитерские, Стройматериалы... | +0.9% |
| all | 9% | Плодоовощи, Табак, Телерадиотовары | +11.4% |
| monetary | 6% | Топливо, Компьютеры, Моющие, Связь | +5.1% |

**Топ улучшений (отдельные субкомпоненты):**

- Масло и жиры (brent): +28.9%
- Меха и меховые изделия (brent): +21.7%
- Топливо моторное (brent): +15.5%
- Плодоовощи (all): +15.1%
- Персональные компьютеры (monetary): +14.7%
- Бытовые услуги (brent): +14.1%
- ЖКХ (seasonal): +13.6%

**Модель:** `sirena/models/subcomponent.py` — `SubcomponentForecaster`

```python
from sirena.models import SubcomponentForecaster

# h=1: MAE 0.426 (-25% vs baseline!)
model = SubcomponentForecaster(horizon=1)
model.fit(df)
forecast = model.forecast()

# h=12: для годовых прогнозов
model12 = SubcomponentForecaster(horizon=12)
```

**Выводы:**

- ✅ **h=1 с индивидуальными подходами: +25% улучшения!**
- Ключевые драйверы: USD (32% веса), Brent (13% веса, +13.6% улучшение)
- Для h=12 baseline subcomp лучше оптимизированного (возможно переобучение)

---

## Период обучения: 2010 vs 2016

**Тестирование показало:**

| Горизонт | Лучший период | Комментарий |
|----------|---------------|-------------|
| h=1 | 2016 | Данные 2010-2015 добавляют шум |
| h=2 | 2010 (Full) | Больше данных помогает |
| h=12 | 2016 | Режим инфляции изменился |

**Вывод:** Для h=1 и h=12 рекомендуется использовать данные с 2016 года.

---

## v4.9 МИКРОКОМПОНЕНТНОЕ ИССЛЕДОВАНИЕ (2025-12-29)

### Цель

Исследовать возможность улучшения прогноза через bottom-up агрегацию 537 микрокомпонентов ИПЦ.

### Созданные модели

| Модель | Файл | Описание |
|--------|------|----------|
| MicrocomponentForecaster | `sirena/models/microcomponent.py` | 497 моделей, Ridge для топ-100, VotingRegressor для остальных |
| HierarchicalMicroForecaster | `sirena/models/hierarchical_micro.py` | Иерархия: микро → субкомп → комп → total, Prophet для ЖКХ |
| MicroOptimizedForecaster | `sirena/models/micro_optimized.py` | Huber для stable/medium, Ridge для volatile |
| **HorizonEnsembleForecaster** | `sirena/models/horizon_ensemble.py` | **Адаптивный ансамбль Huber + Micro** |

### Результаты бэктеста (Dec 2024 — Nov 2025)

| Model | h=1 | h=2 | h=3 | h=6 | h=12 | **Avg** |
|-------|-----|-----|-----|-----|------|---------|
| **HorizEns** | 0.301 | **0.247** | **0.314** | 0.323 | 0.298 | **0.297** |
| Huber | **0.288** | 0.267 | 0.318 | 0.333 | 0.331 | 0.307 |
| Ridge | 0.290 | 0.289 | 0.339 | 0.356 | 0.326 | 0.320 |
| HierMicro | 0.321 | 0.342 | 0.331 | **0.322** | 0.332 | 0.330 |
| Micro | 0.361 | 0.369 | 0.350 | 0.331 | **0.297** | 0.341 |

### Ключевые находки

**1. Зависимость от горизонта:**

- **h=1,2,3**: Простые модели (Huber) побеждают
- **h=6,12**: Микрокомпонентные модели становятся конкурентоспособными
- **h=12**: Micro показал лучший результат (MAE 0.297)!

**2. Почему микро хуже на коротких горизонтах:**

- Накопление ошибок при агрегации 497 моделей
- Ошибки коррелированы (общие шоки, сезонность)
- Сильная автокорреляция total inflation проще использовать напрямую

**3. Почему микро лучше на длинных горизонтах:**

- Автокорреляция ослабевает к h=12
- Структурная информация (сезонность ЖКХ, туризма) становится ценной
- Ошибки лучше усредняются на длинных периодах

**4. HorizonEnsemble — лучший средний результат:**

- **MAE 0.297** — на 3.6% лучше Huber
- **h=2: MAE 0.247** — на 7.2% лучше Huber!
- Адаптивные веса: h=1 (80% Huber), h=12 (70% Micro)

### Использование HorizonEnsembleForecaster

```python
from sirena.models import HorizonEnsembleForecaster

# Автоматически подбирает веса под горизонт
model = HorizonEnsembleForecaster(horizon=12)
model.fit(df)

# Прогноз с decomposition
result = model.predict(df, target_date)
print(f"Ensemble: {result['prediction']}")
print(f"Huber: {result['huber_pred']}, weight: {result['weights']['huber']}")
print(f"Micro: {result['micro_pred']}, weight: {result['weights']['micro']}")
```

### Структура данных микрокомпонентов

| Категория | Товаров | Вес | Волатильность | Рекомендация |
|-----------|---------|-----|---------------|--------------|
| stable | 198 | 39.3% | std < 2 | VotingRegressor |
| medium | 266 | 46.4% | 2 ≤ std < 5 | VotingRegressor |
| volatile | 61 | 11.2% | 5 ≤ std < 15 | Ridge_500 |
| ultra_volatile | 12 | 2.6% | std ≥ 15 | Субкомп fallback |

**Топ-10 волатильных (22% потенциала ошибки):**

- Картофель (std=19.8), Помидоры (std=19.5), Огурцы (std=21.7)
- Маршрутное такси, Яйца, Апельсины, Яблоки, Капуста, Виноград

### Рекомендации для production

| Горизонт | Рекомендация | MAE |
|----------|--------------|-----|
| h=1 | Huber | 0.288 |
| h=2 | **HorizonEnsemble** | **0.247** |
| h=3 | HorizonEnsemble | 0.314 |
| h=6 | HierMicro или HorizonEnsemble | 0.322-0.323 |
| h=12 | Micro или HorizonEnsemble | 0.297-0.298 |

---

## Backtesting Framework

### Описание

Система автоматизированного бэктестирования моделей с 3 горизонтами прогноза:

- **h=1**: 1 месяц вперед (главный КПЭ)
- **h=2**: 2 месяца вперед
- **h=12**: 12 месяцев вперед (годовая траектория)

### Запуск бэктестов

```bash
# h=1 (1 month ahead) — САМЫЙ ВАЖНЫЙ КПЭ
python3 scripts/run_backtest_h1.py

# h=2 (2 months ahead)
python3 scripts/run_backtest_h2.py

# h=12 (12 months ahead, годовая траектория)
python3 scripts/run_backtest_h12.py
```

### Результаты

Результаты сохраняются в `archive/results/`:

- `backtest_h{X}_predictions.csv` — все прогнозы всех моделей
- `backtest_h{X}_metrics.csv` — метрики по каждой модели (MAE, RMSE, KPI violations)
- `backtest_h{X}_summary.md` — markdown отчет с топ моделями

### Лучшие модели (январь 2025 — декабрь 2025)

**h=1 (1 месяц вперед — главный КПЭ):**

1. Subcomp: MAE 0.309 (2 KPI violations из 12)
2. Ridge_Macro: MAE 0.319 (4 KPI violations)
3. Ridge_Shock: MAE 0.319 (3 KPI violations)

**h=2 (2 месяца вперед):**

1. NGBoost: MAE 0.290 (3 KPI violations)
2. Ridge_Ext: MAE 0.295 (2 KPI violations)
3. Huber: MAE 0.297 (2 KPI violations)

**h=12 (годовая траектория):**

1. Subcomp_Multi: MAE 0.297 (3 KPI violations)
2. Prophet: MAE 0.310 (1 KPI violation)
3. Subcomp: MAE 0.324 (4 KPI violations)

### Методика

Подробное описание логики бэктестов см. в **[docs/BACKTEST_METHODOLOGY.md](docs/BACKTEST_METHODOLOGY.md)**

### KPI Metrics

- **MAE** — Mean Absolute Error (главная метрика)
- **KPI Violations** — количество месяцев с |error| > 0.5
- **Coverage 50%** — % точек, где |error| <= 0.5
- **RMSE** — Root Mean Squared Error

### Архитектура

**Файлы:**

- `scripts/backtest_framework.py` — класс BacktestRunner (ядро системы)
- `scripts/run_backtest_h1.py` — запуск h=1
- `scripts/run_backtest_h2.py` — запуск h=2
- `scripts/run_backtest_h12.py` — запуск h=12

**19 моделей в бэктесте:**
Ridge, Ridge Extended, Ridge Shock, Ridge_Macro, Bayesian Ridge, ElasticNet, Huber, NGBoost, NGBoost Shock, LMMR Claude, LMMR Hybrid, BVAR, SARIMA, LightGBM, Prophet, ETS, EBM, CatBoost, Ensemble

**Source of Truth:**
`data/inflation_data.csv` — единственный источник актуальных значений для сравнения

---

## SIRENA Score — Комплексная метрика качества моделей

### Описание

**SIRENA Score** — единая метрика для оценки моделей на всех горизонтах прогноза.

**Формула:**

```
SIRENA_Score = Weighted_MAE × KPI_Penalty

где:
  Weighted_MAE = 0.50 × MAE_h1 + 0.30 × MAE_h2 + 0.20 × MAE_h12
  KPI_Penalty = 2.0 - KPI_rate  (диапазон 1.0 — 2.0)
  KPI_rate = доля попаданий в |error| <= 0.5
```

**Чем ниже Score — тем лучше модель.**

### Запуск расчёта

```bash
# Расширенный бэктест 2020-2025 с SIRENA Score
python3 scripts/sirena_score.py
```

### Результаты (ноябрь 2025, rolling 12 мес)

| Ранг | Модель | SIRENA Score | MAE h=1 | MAE h=2 | MAE h=12 | KPI Rate |
|------|--------|--------------|---------|---------|----------|----------|
| 🥇 1 | **Subcomp_Multi** | **0.515** | 0.265 | 0.278 | 0.264 | 8.3% |
| 🥈 2 | EBM | 0.592 | 0.309 | 0.309 | 0.309 | 8.3% |
| 🥉 3 | Subcomp | 0.607 | 0.285 | 0.379 | 0.302 | 8.3% |
| 4 | Huber | 0.629 | 0.348 | 0.269 | 0.366 | 8.3% |
| 5 | Ridge | 0.639 | 0.314 | 0.335 | 0.378 | 8.3% |
| 6 | NGBoost | 0.640 | 0.326 | 0.314 | 0.382 | 8.3% |
| ... | | | | | | |
| 14 | **Micro ARIMA** | 0.701 | 0.415 | 0.439 | 1.000 | 70.0% |
| 16 | SARIMA | 0.795 | 0.356 | 0.448 | 0.513 | 8.3% |

### Выходные файлы

- `archive/results/sirena_score_summary.csv` — финальный рейтинг
- `archive/results/sirena_score_history.csv` — история по месяцам (rolling 12 мес)
- `archive/results/sirena_score_raw.csv` — все прогнозы (71 дат × 16 моделей × 3 горизонта)
- `assets/charts/sirena_score_dynamics.html` — интерактивный график динамики

### Ключевые выводы

1. **Subcomp_Multi** — лидер по совокупному качеству (Score 0.515)
2. **Micro ARIMA** (#14 из 16) — значительно уступает нашим моделям на h=1 и h=2
3. **EBM** — стабильная модель с одинаковым MAE на всех горизонтах
4. **Prophet** — отличная на h=12 (MAE 0.318), но слабая на h=1/h=2

---

## Micro ARIMA (внешняя модель пользователя)

### Описание

Микрокомпонентная ARIMA модель, которую пользователь использовал ранее как основную. Загружается из внешнего файла `micro_test.csv`.

**Файл:** `sirena/models/micro_arima.py`

### Формат данных `micro_test.csv`

```
,2024-11-01,2024-12-01,2025-01-01,...
2024-12-01,101.43,,,
2025-01-01,100.68,100.52,,
2025-02-01,100.87,100.71,100.59,
...
```

- **Строки** — целевые даты прогноза
- **Столбцы** — даты cutoff (когда был сделан прогноз)
- **Значения** — индекс MoM в формате 100.xx (101.43 = +1.43%)

### Использование

```python
from sirena.models.micro_arima import MicroARIMAForecaster

# h=1 (1 месяц вперед)
model = MicroARIMAForecaster(horizon=1, file_path='micro_test.csv')
model.fit()

# Получить прогноз на конкретную дату
result = model.predict(df, target_date=pd.Timestamp('2025-06-01'))
print(f"Прогноз: {result['prediction'] - 100:.2f}%")  # MoM в процентах
```

### Результаты бэктеста (Dec 2024 — Nov 2025)

| Горизонт | Micro ARIMA | Лучшая модель | Разница |
|----------|-------------|---------------|---------|
| h=1 | 0.415 | Subcomp_Multi 0.265 | **+56.6%** ❌ |
| h=2 | 0.439 | NGBoost 0.251 | **+74.9%** ❌ |
| h=12 | 0.292* | Prophet 0.281 | +3.9% |

*h=12 данные неполные (window_size=10 vs 12)

### Вывод

Micro ARIMA **значительно уступает** нашим моделям машинного обучения:

- На h=1: +56.6% MAE vs лучшая модель
- На h=2: +74.9% MAE vs лучшая модель
- SIRENA Score: #14 из 16 моделей

**Рекомендация:** Использовать Subcomp_Multi или Huber вместо Micro ARIMA.

---

## Weekly Prices Nowcasting (v5.3, 2026-01-23)

### Описание

Система nowcasting на основе недельных цен Rosstat для прогнозирования месячной инфляции ИПЦ.

**Источник данных:** `data/kbr_weekly_prices_2008_2026.csv`

- 142,135 строк
- 155 продуктов
- Период: 2008-2026 (18 лет)
- Тренировочный период: 2016-2026 (post-Crimea)

### Модули

| Модуль | Файл | Назначение |
|--------|------|------------|
| Weekly Loader | `sirena/data/weekly_loader.py` | Загрузка данных, классификация продуктов |
| Weekly Nowcaster | `sirena/models/weekly_prices.py` | Nowcasting текущего месяца |
| Leading Indicators | `sirena/models/leading_indicators.py` | Опережающие индикаторы (Granger) |
| Volatility Monitor | `sirena/models/volatility_monitor.py` | Детектор аномалий |

### Результаты бэктеста (2024-2025)

| Модель | MAE | KPI Violations | vs Target |
|--------|-----|----------------|-----------|
| **WeeklyPriceNowcaster** | **0.043%** | **0/24** | **-57%** ✅ |
| Target | <0.10% | minimal | baseline |

### High-Quality Products (22 товара, <5% missing)

```python
HIGH_QUALITY_PRODUCTS = {
    111: 'Говядина',      # weight: 0.0158
    114: 'Куры',          # weight: 0.0095
    701: 'Масло сливочное', # weight: 0.0088
    1501: 'Яйца',         # weight: 0.006
    1601: 'Сахар',        # weight: 0.004
    2501: 'Картофель',    # weight: 0.004
    7802: 'Бензин АИ-92', # weight: 0.0101
    # ... и еще 15 продуктов
}
```

### Leading Indicators (33 значимых)

| Продукт | Lead (мес) | Correlation | p-value |
|---------|------------|-------------|---------|
| Масло сливочное | 2 | +0.40 | 0.0004 |
| Поездка в Турцию | 1 | +0.52 | 0.0000 |
| Хлеб ржаной | 2 | -0.36 | 0.0000 |
| Огурцы | 3 | +0.36 | 0.0161 |

### Сезонные паттерны (High-Quality Products)

| Месяц | Mean WoW | Std | Комментарий |
|-------|----------|-----|-------------|
| Январь | +0.36% | 2.66 | Тарифные повышения |
| Июнь | -0.00% | 3.17 | Стабильность |
| Июль | -0.28% | 3.06 | Урожай |
| Декабрь | +0.25% | 1.99 | Предпраздничный |

### Dashboard Tab

Вкладка **📈 Weekly** в dashboard.py:

- Nowcast signal (overall, food, non-food)
- Прогноз текущего месяца
- Volatility alerts (critical/warning)
- График динамики цен

### Использование

```python
# 1. Загрузка данных
from sirena.data.weekly_loader import load_weekly_prices, compute_basket_signal
weekly_df = load_weekly_prices()  # 79,360 rows (2016-2026)
signal = compute_basket_signal()  # {'signal': 1.74, 'food_signal': 1.92, ...}

# 2. Nowcasting
from sirena.models.weekly_prices import WeeklyPriceNowcaster
model = WeeklyPriceNowcaster(use_macro=False, use_components=True)
model.fit()
nowcast = model.nowcast()  # {'prediction': 0.45, 'coverage': 0.95, ...}

# 3. Volatility monitoring
from sirena.models.volatility_monitor import VolatilityMonitor
monitor = VolatilityMonitor()
monitor.initialize()
anomalies = monitor.check_anomalies()  # List of critical/warning alerts

# 4. Leading indicators
from sirena.models.leading_indicators import LeadingIndicatorDetector
detector = LeadingIndicatorDetector()
results = detector.analyze()
signal = detector.get_current_signal()  # {'signal': 0.12, 'n_products': 33, ...}
```

### Верификация

```bash
# Test data loader
python3 -c "from sirena.data.weekly_loader import load_weekly_prices; print(load_weekly_prices().shape)"
# Expected: (79360, 9)

# Test nowcaster
python3 sirena/models/weekly_prices.py
# Expected: MAE < 0.05

# Test volatility
python3 sirena/models/volatility_monitor.py
# Shows alerts and seasonal patterns
```

### TODO: Гипотезы для исследования (Ralph)

1. **Взвешивание по волатильности**: Использовать 1/std как вес — стабильные продукты важнее
2. **Лаги по продуктам**: Оптимизировать лаг для каждого продукта индивидуально
3. **Сезонная корректировка**: Применить X-13 к недельным данным перед агрегацией
4. **Режимозависимые веса**: Разные веса для shock/normal режимов
5. **Ensemble с месячными моделями**: Оптимальный blend weekly + monthly

---

## ИССЛЕДОВАНИЕ ЭКЗОГЕННЫХ ПРИЗНАКОВ (2025-12-28/29)

### Источники данных

| Источник | Период | Частота | Показатели |
|----------|--------|---------|------------|
| inflation_data.csv | 2010-2025 | месяц | mom, USD, Ki, Ruonia, компоненты |
| month.csv | 2016-2025 | месяц | 13 региональных показателей КБР |
| quart.csv | 2016-2025 | квартал | 5 показателей (включая ИБВЭД с 2019) |
| brent_prices.csv | 2010-2025 | месяц | Нефть Brent |

### Тесты Грейнджера (причинность)

**Статистически значимые опережающие индикаторы:**

| Показатель | Оптимальный лаг | p-value | Вывод |
|------------|-----------------|---------|-------|
| Ki (ключевая ставка) | 6 мес | 0.0000 *** | **ИСПОЛЬЗОВАТЬ** |
| USD (доллар) | 2 мес | 0.0020 ** | **ИСПОЛЬЗОВАТЬ** |
| Ruonia | 2 мес | 0.0005 *** | Коррелирует с Ki — НЕ использовать вместе |
| Brent (нефть) | 5 мес | 0.0080 ** | **ИСПОЛЬЗОВАТЬ** |
| reg_ppi (цены производителей КБР) | 3 мес | 0.0057 ** | Опционально |

### Корреляции с целевой переменной

**Топ-10 признаков:**

| Признак | Pearson r | Описание |
|---------|-----------|----------|
| reg_unknown20_MA3 (ИБВЭД) | +0.524 | Квартальный с 2019, мало данных |
| ki_i | +0.515 | Ключевая ставка |
| Ruonia_D1 | +0.492 | Первая разность Ruonia |
| brent_STD3 | +0.461 | Волатильность нефти |
| mom_L0 | +0.436 | Авторегрессия |
| prod | +0.431 | Продовольствие |

### Результаты бэктестов наборов признаков

| Набор | MAE | vs Baseline | Вывод |
|-------|-----|-------------|-------|
| Best_correlated | 0.462 | **-15.7%** | Лучший! |
| Components | 0.508 | -7.3% | Хороший |
| Federal_macro | 0.528 | -3.8% | Работает |
| Minimal (AR only) | 0.548 | 0% | Baseline |
| Regional_macro | 0.549 | +0.1% | Бесполезен |
| Kitchen_sink | 1.591 | **+190%** | ПЕРЕОБУЧЕНИЕ! |

### Ключевые выводы

**ЧТО РАБОТАЕТ:**

- Компоненты инфляции (Prod, Nonprod, Serv) — самые полезные
- USD с лагом 2 месяца — значимо помогает
- Ki с лагом 6 месяцев — значимо помогает
- Brent с лагом 5 месяцев — для h=2 особенно полезен

**ЧТО НЕ РАБОТАЕТ:**

- Ki + Ruonia вместе — мультиколлинеарность (r > 0.9)
- Региональные показатели КБР — не добавляют информации
- ИБВЭД — слишком мало данных (с 2019), нестабильно
- Kitchen sink (все признаки) — катастрофическое переобучение (+190% MAE)

---

## v5.0 RidgeMacroForecaster (2025-12-29)

### Описание

Модель на основе результатов исследования признаков. Использует оптимальные лаги макро-показателей.

**Файл:** `sirena/models/ridge_macro.py`

### Признаки

```python
FEATURES = [
    # Авторегрессия (L1+ - L0 неизвестен при прогнозе!)
    'mom_L1', 'mom_L2', 'mom_L3',

    # Федеральные макро с оптимальными лагами (Грейнджер)
    'ki_L6',        # Ключевая ставка, лаг 6, p=0.0000
    'Ruonia_D1',    # Первая разность, r=0.492
    'usd_L2',       # Доллар, лаг 2, p=0.0020
    'brent_L5',     # Нефть, лаг 5, p=0.0080
    'brent_STD3',   # Волатильность нефти, r=0.461

    # Компоненты инфляции
    'prod_L1',      # Продовольствие
    'serv_L1',      # Услуги

    # Сезонность
    'month_sin', 'month_cos',
]
```

### Результаты бэктеста

| Горизонт | MAE | vs Baseline | Позиция |
|----------|-----|-------------|---------|
| **h=1** | **0.300** | **-3.2%** | **#3** |
| h=2 | 0.321 | +3.2% | #8 |
| h=12 | 0.435 | +28% | #9 |

**Вывод:** Макро-признаки наиболее полезны на коротком горизонте (h=1).

### Использование

```python
from sirena.models import RidgeMacroForecaster

model = RidgeMacroForecaster()
model.fit(df, 'Все товары и услуги')

# Прогноз
result = model.predict(df_ext, target_date)
print(f"Прогноз: {result['prediction']:.2f}")
print(f"Признаки: {result['features_used']}")
```

### Важно

- **L0 признаки НЕ используются** — при прогнозе значение текущего месяца неизвестно
- Требует наличия макро-данных в DataFrame (Ki, Ruonia, usd_nom_i, brent)
- Данные загружаются из `inflation_data.csv` (не из `infl_kbr.csv`!)

---

## Исследование: Плодоовощи и формула агрегации Росстата

### Ключевое открытие (декабрь 2024)

**Росстат использует ЦЕПНЫЕ ИНДЕКСЫ для агрегации микрокомпонентов, а НЕ прямое взвешенное среднее MoM!**

```python
# Формула агрегации Росстата
Cum_i(t) = Cum_i(t-1) × (1 + MoM_i(t)/100)   # Кумулятивный индекс к декабрю
Agg_Cum(t) = Σ(Cum_i(t) × W_i) / Σ(W_i)      # Взвешенная агрегация
Sub_MoM(t) = (Agg_Cum(t) / Agg_Cum(t-1) - 1) × 100  # MoM из кумулятива
```

**Верификация:** 2025 год (все 21 товар) — MAD = **0.003 п.п.** (идеальное совпадение!)

### Документация исследования

| Файл | Описание |
|------|----------|
| **[docs/RESEARCH_PLODOVOSHCHI_DIVERGENCE.md](docs/RESEARCH_PLODOVOSHCHI_DIVERGENCE.md)** | Постановка задачи исследования |
| **[archive/results/plodov_divergence_report.md](archive/results/plodov_divergence_report.md)** | Полный отчёт с результатами |
| `archive/results/plodov_kbr_comparison.csv` | КБР: Rosstat vs Chain Index vs Direct (2021-2025) |
| `archive/results/plodov_summary_2024_2025.csv` | Сводка по регионам 2024-2025 |

### MicroPlodovoshchi v2.0

Модель прогнозирования плодоовощей через 21 микрокомпонент с формулой цепных индексов.

**Файл:** `sirena/models/micro_plodovoshchi.py`

**Результаты бэктеста (субкомпонент 33):**

| Метод | MAE | vs MicroPlod |
|-------|-----|--------------|
| **MicroPlodovoshchi v2.0** | **1.756 pp** | baseline |
| NGBoost (прямой) | 4.047 pp | +130% хуже |
| Ridge (прямой) | 3.783 pp | +115% хуже |

**Микрокомпонентная модель в 2+ раза лучше для прогноза субкомпонента 33!**

### Почему не интегрировано в SubcomponentMulti

Интеграция **ухудшает общий MAE** (0.308 vs 0.274) из-за:

1. Расхождения данных `kbr_micro_full.csv` и `sub_mom.csv`
2. Региональной специфики КБР (позиция 88/106 по сходимости данных)

**Рекомендуемое использование:**

- Анализ динамики отдельных товаров (картофель, огурцы, бананы)
- Прогноз по РФ в целом (там сходимость лучше: MAD=1.05 vs 1.93 в КБР)
- Nowcasting с оперативными данными по микрокомпонентам

### Региональная специфика

Анализ **106 регионов** показал неравномерность сходимости данных:

| Регион | MAD | Позиция |
|--------|-----|---------|
| РФ в целом | 1.05 | 19/106 (топ-18%) |
| КБР | 1.93 | 88/106 (худшие 17%) |
| Лучший (code=14) | 0.43 | 1/106 |
| Худший (code=8) | 3.18 | 106/106 |

### Бэктест-файлы

| Файл | Описание |
|------|----------|
| `archive/results/micro_plodov_v2_backtest.csv` | MicroPlodovoshchi v2.0 |
| `archive/results/sub33_direct_backtest.csv` | Ridge прямой прогноз sub 33 |
| `archive/results/sub33_ngboost_backtest.csv` | NGBoost прямой прогноз sub 33 |

---

# Ralph Universal: Autonomous Optimization Agent

## Core Identity

You are **Ralph**, an advanced autonomous AI agent operating within the **Opus Edge Lab**.
Your purpose is **Autopoiesis**: Self-creation, self-maintenance, and continuous evolution of the forecasting system.

## Prime Directives (The "Bulletproof" Protocol)

1. **Trust But Verify**: Never assume a task is done. Verify it with code execution (Worker) or rigorous review (Critic).
2. **No Fake Work**: Do not mark tasks as DONE unless acceptance criteria are met and verified.
3. **Evolution**: If you find a better way, update the documentation (`GEMINI.md`, `README.md`) to reflect reality.

## Context Loading Rules

**CRITICAL**: Do NOT load all files at once. Use "Lazy Loading":

- **Forecasting Tasks**: Load `sirena/models/` only when working on specific models.
- **Infrastructure**: Load `system/` only when debugging the orchestrator.
- **Documentation**: Refer to `docs/` for standards.

## Data Mining Protocol (For "Honest" Extraction)

When performing Data Mining or Parsing tasks (e.g., Task 113, 114):

1. **Safety First**: Never ping thousands of URLs blindly. Always implement rate limiting or local parsing first.
2. **Verify, Don't Assume**:
    - ❌ Bad Criterion: "Parse file X"
    - ✅ Good Criterion: "Output `data/result.csv` exists AND size > 100KB AND has > 1000 rows"
3. **Sample the Goods**: The Critic agent MUST read the first 5 lines of any generated CSV (`head -n 5`) to confirm the data is not garbage.
4. **Handle Huge Files**: For >100MB files, never use `pd.read_csv()` without `chunksize`. Prove memory safety.

## External References

- **Project Rules**: @../GEMINI.md (Strict adherence required)
- **Task List**: @tasks/prd.json
- **Opencode Reference**: @docs/opencode_reference.md

## Modes of Operation

- **Worker**: Generates code, runs tests, fixes bugs. Output: `COMPLETED_TASK`.
- **Critic**: Reviews code, checks logic, verifies outputs. Output: `APPROVE` or `REJECT`.
- **Refiner**: Analyzes BLOCKED tasks, researches files, creates subtasks. Output: Subtasks in `prd.json`.

## Ralph Configuration (Edge Lab)

- **Codebase**: `edge_lab/` (Sandbox Environment)
- **Config**: `edge_lab/system/config.py`
- **Primary Model**: `zai-coding-plan/glm-4.7` (PAID TIER) - **Do NOT use free model**
- **Critic Model**: `zai-coding-plan/glm-4.7` (PAID TIER)
- **Task Source**: `edge_lab/tasks/prd.json`

## Edge Lab: Структура директорий

```
edge_lab/
├── system/                      # Ядро Agent System (Worker + Critic)
│   ├── orchestrator.py         # Главный оркестратор (двухпроцессный цикл)
│   ├── worker.py               # Worker — выполняет задачи (Red-Green-Refactor)
│   ├── critic.py               # Critic — верифицирует результаты
│   ├── config.py               # Конфиг (модели, пути, параметры)
│   ├── state.py            # Thread-safe управление PRD/progress
│       └── agent_wrapper.py    # CLI интерфейс к LLM (opencode run)
│   ├── refiner.py              # Refiner — анализирует BLOCKED, создаёт подзадачи
│   └── config.py               # Конфиг (модели, пути, Safety Limits)
│
├── agents/                      # Автономные агенты (см. ниже)
│   ├── hypothesis_generator.py
│   ├── news_sentiment.py
│   ├── immune_system.py
│   └── rosstat_ingester.py
│
├── tasks/                       # Control Center
│   ├── prd.json                # PRD v3.2 (30 задач)
│   └── progress.txt            # Лог выполнения (timestamped)
│
├── data/                        # Выходные данные агентов
│   ├── extracted_kbr/          # 53 файла из Rosstat
│   ├── schema_registry.json    # Mapping файлов Rosstat
│   └── sentiment_index.csv     # Индекс hawkishness из CBR
│
├── MANIFESTO.md                # Философия Autopoiesis (Epoch 1-4)
├── AGENTS.md                   # Документация агентов
└── README.md                   # Quick Start
```

## Команды запуска Edge Lab

```bash
# Запуск системы (Worker + Critic параллельно)
python3 edge_lab/system/orchestrator.py

# Статус задач (30 user stories)
cat edge_lab/tasks/prd.json | jq '.user_stories[] | {id, title, status}'

# Лог прогресса
tail -50 edge_lab/tasks/progress.txt

# Проверка конфигурации
cat edge_lab/system/config.py | grep -E "MODEL|CLI"
```

## CLI Tools (Edge Lab)

### add_task.py — Task Creator

Быстрое добавление задач в `prd.json` без ручного редактирования JSON:

```bash
cd edge_lab

# Тестовая задача (auto-MVAC)
python3 add_task.py -t "Test NewModel" --type test -p high

# Data mining
python3 add_task.py -t "Mining: GDP Data" --type mining -p high

# Кастомные MVAC
python3 add_task.py -t "Custom Task" -m "@file: foo.py" -m "@functional: runs"

# Интерактивный режим
python3 add_task.py --interactive

# Справка по типам
python3 add_task.py --list-types
```

**Типы задач**: `test`, `model`, `script`, `mining`, `docs`, `integration`, `custom`

### Task Management Commands

```bash
# Показать BLOCKED задачи (требуют доработки)
python3 add_task.py --blocked

# Разблокировать задачу для повторной попытки
python3 add_task.py --unblock 124
```

### Архитектура v1.2: Worker-Critic-Refiner

```
┌─────────┐     ┌─────────┐     ┌──────────┐
│ Worker  │────▶│ Critic  │────▶│ Refiner  │
│ (does)  │     │(reviews)│     │(improves)│
└─────────┘     └─────────┘     └──────────┘
     │               │               │
     ▼               ▼               ▼
   TODO ──▶ PENDING_REVIEW ──▶ BLOCKED ──▶ Subtasks (TODO)
                    │
                    ▼
                   DONE
```

**Статусы задач:**

- `TODO` — ожидает выполнения
- `PENDING_REVIEW` — Worker закончил, Critic проверяет
- `DONE` — Critic одобрил
- `BLOCKED` — 3 неудачные попытки, требует доработки
- `DECOMPOSED` — разбит на подзадачи Refiner'ом

### Safety Limits (`system/config.py`)

```python
SAFETY_LIMITS = {
    'max_task_duration_seconds': 1800,  # 30 мин/задача
    'max_retries_per_task': 3,          # → BLOCKED после 3 попыток
    'max_file_size_mb': 100,            # Лимит для парсеров
}
```

## Автономные агенты Edge Lab

| Агент | Файл | Назначение | Статус |
|-------|------|------------|--------|
| Hypothesis Generator | `agents/hypothesis_generator.py` | Автоматическая генерация гипотез о корреляциях (лаги, признаки) | ✅ |
| News Sentiment | `agents/news_sentiment.py` | CBR press releases + BERT Hawkishness index | ✅ (mock-данные) |
| Immune System | `agents/immune_system.py` | Стресс-тестирование на Black Swan events | ✅ |
| Rosstat Ingester | `agents/rosstat_ingester.py` | Парсинг 53 файлов региональной статистики КБР | ✅ |
| Regime Detector | `agents/regime_detector.py` | Детектор экономических режимов (shock/normal/high_inflation) | ✅ |

### News Sentiment Agent (BERT Hawkishness)

Классифицирует тон пресс-релизов ЦБ:

- **Hawkish**: "повысить", "ужесточение", "борьба с инфляцией"
- **Dovish**: "снизить", "смягчение", "поддержка роста"
- **Neutral**: нейтральные формулировки

Выход: `sentiment_index.csv` — временной ряд hawkishness score (0-1).

## Связь ralph_universal/ vs edge_lab/

| Инстанс | Путь | Назначение | Статус |
|---------|------|------------|--------|
| Main | `ralph_universal/` | Оригинальный инстанс для SIRENA | ⚠️ Устаревший конфиг |
| **Edge Lab** | `edge_lab/` | **Актуальная** sandbox для экспериментов | ✅ Используйте этот |

**ВАЖНО:** Используйте `edge_lab/` — там актуальная конфигурация с `zai-coding-plan/glm-4.7`!

## Текущий прогресс задач (январь 2026)

- **24/30 выполнено** — production модели, Rosstat integration, агенты
- **6 TODO/REJECTED**:
  - MIDAS (ID 21): REJECTED — MAE +32% vs baseline
  - TFT (ID 22): TODO — требует доработки
  - Conformal (ID 23): TODO — не интегрирован
  - ExogProphet (ID 24): REJECTED — MAE +158%
  - Report generator (ID 25): TODO
  - API/Nowcasting (ID 26-30): TODO

## Opencode Documentation

For CLI usage, model configuration, and troubleshooting (including the `zai-coding-plan` prefix), refer to the Edge Lab reference:

- **[Opencode CLI Reference](edge_lab/docs/opencode_reference.md)**
- Official Docs: [opencode.ai/docs](https://opencode.ai/docs/)

## Ralph Best Practices

### Постановка задач (Rule 3-2-1)

Каждая задача ДОЛЖНА иметь:

- **3 критерия минимум**
- **2 критерия автоматически проверяемые** (exit code, file exists, metric)
- **1 критерий качественный** (code review, documentation)

**Пример хорошего acceptance_criteria:**

```json
{
  "acceptance_criteria": [
    "@file: tests/test_new_model.py exists (>50 lines)",
    "@metric: MAE < 0.35 (python3 scripts/evaluate.py)",
    "@functional: pytest tests/test_new_model.py -v passes"
  ]
}
```

**Плохо:**

```json
{"acceptance_criteria": ["pytest passes"]}
```

### Верификация (Critic)

1. Critic ДОЛЖЕН запускать команды, не верить на слово Worker
2. Используйте JSON output schema для структурированных ответов
3. Логируйте причины rejection с конкретными числами

**Critic Output Format:**

```json
{
  "decision": "APPROVE",
  "reason": "All criteria verified",
  "criteria_results": [
    {"criterion": "File exists", "passed": true, "evidence": "ls output"}
  ],
  "confidence": 0.95
}
```

### Мониторинг

```bash
# Метрики проекта
python3 edge_lab/scripts/metrics_dashboard.py

# Последние события
tail -50 edge_lab/tasks/progress.txt

# Статус задач
cat edge_lab/tasks/prd.json | jq '.user_stories[] | {id, status, title}'
```

### Персоны агентов

Персоны хранятся в `edge_lab/.opencode/agents/`:

- `worker.md` — Worker: Red-Green-Refactor, COMPLETED_TASK output
- `critic.md` — Critic: Trust But Verify, JSON decisions
- `ralph.md` — Ralph meta-agent: координация, метрики

### Health Indicators (targets)

| Метрика | Target | Action if below |
|---------|--------|-----------------|
| Completion Rate | >= 80% | Review task complexity |
| Rejection Rate | <= 20% | Improve acceptance criteria |
| Avg Criteria/Task | >= 3.0 | Add more criteria to tasks |

### Эскалация

- Task rejected 3+ times: flag для human review
- Worker blocked >30 min: эскалация с деталями
- Critic timeout: retry с упрощенной верификацией
