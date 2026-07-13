# Методика бэктестирования СИРЕНА-КБР v5.2

## Описание

Данная методика описывает 3 варианта бэктестирования моделей прогнозирования инфляции КБР с разными горизонтами:
- **h=1**: Прогноз на 1 месяц вперед (главный КПЭ)
- **h=2**: Прогноз на 2 месяца вперед
- **h=12**: Прогноз на 12 месяцев вперед (годовая траектория)

Все бэктесты автоматизированы через скрипты в `scripts/`.

---

## h=1: Прогноз на 1 месяц вперед (ГЛАВНЫЙ КПЭ)

### Описание

Rolling window backtest за последние 12 месяцев. Для каждого месяца модель обучается на всех доступных данных до `t-1` и прогнозирует месяц `t`.

### Логика

```
Для каждого месяца t (декабрь 2024 → ноябрь 2025):
  cutoff = t - 1 месяц
  train = все данные до cutoff
  прогноз = forecast(1)[0]
  actual = inflation_data.csv[t, 'mom'] - 100
  error = actual - прогноз
```

### Пример

- **Ноябрь 2025**: train до октября 2025 → прогноз на ноябрь
- **Октябрь 2025**: train до сентября 2025 → прогноз на октябрь
- **Сентябрь 2025**: train до августа 2025 → прогноз на сентябрь
- ... и так далее

**Итого**: 12 независимых обучений, 12 прогнозов

### Метрики

- **MAE** — Mean Absolute Error (главная метрика)
- **RMSE** — Root Mean Squared Error
- **KPI Violations** — количество месяцев с |error| > 0.5
- **Coverage 50%** — % точек, где |error| <= 0.5

### Результаты (декабрь 2024 — ноябрь 2025)

**Top 3 модели:**

1. **SubcomponentMulti v2.3**: MAE 0.236 (1 KPI violation) — **Лучшая модель!**
2. **Huber**: MAE 0.289 (4 KPI violations)
3. **Ridge Shock**: MAE 0.299 (4 KPI violations)

**Файлы:**
- `archive/results/backtest_h1_predictions.csv` — все прогнозы
- `archive/results/backtest_h1_metrics.csv` — метрики моделей
- `archive/results/backtest_h1_summary.md` — markdown отчет

---

## h=2: Прогноз на 2 месяца вперед

### Описание

Rolling window backtest за последние 12 месяцев. Для каждого месяца модель обучается на данных до `t-2` и прогнозирует месяц `t` (второй месяц из `forecast(2)`).

### Логика

```
Для каждого месяца t (декабрь 2024 → ноябрь 2025):
  cutoff = t - 2 месяца
  train = все данные до cutoff
  прогноз = forecast(2)[1]  # ВТОРОЙ месяц (индекс 1)
  actual = inflation_data.csv[t, 'mom'] - 100
  error = actual - прогноз
```

### Результаты (декабрь 2024 — ноябрь 2025)

**Top 3 модели:**

1. **HorizonEnsemble**: MAE 0.247
2. **NGBoost Shock**: MAE 0.291 (2 KPI violations)
3. **NGBoost**: MAE 0.302 (3 KPI violations)

**Файлы:**
- `archive/results/backtest_h2_predictions.csv`
- `archive/results/backtest_h2_metrics.csv`
- `archive/results/backtest_h2_summary.md`

---

## h=12: Прогноз на 12 месяцев вперед (годовая траектория)

### Описание

Backtest на основе фиксированного cutoff. Модель обучается один раз на данных до ноября 2024 и прогнозирует траекторию на 12 месяцев вперед (декабрь 2024 — ноябрь 2025).

### Результаты (декабрь 2024 — ноябрь 2025)

**Top 3 модели:**

1. **Prophet**: MAE 0.277 (0 KPI violations из 12) — **Лучшая на длинном горизонте!**
2. **Microcomponent**: MAE 0.297
3. **Ridge**: MAE 0.338 (4 KPI violations)

**Файлы:**
- `archive/results/backtest_h12_predictions.csv`
- `archive/results/backtest_h12_metrics.csv`
- `archive/results/backtest_h12_summary.md`

---

## Source of Truth

**Единственный источник актуальных значений:** `data/inflation_data.csv`

- Формат: sep=';', decimal=','
- Колонки: Date, mom, Prod, Nonprod, Serv, usd_nom_i, Ki_i, Ruonia
- Используется для получения актуальных значений для сравнения

---

## Модели

**Список моделей в бэктесте (v5.2):**

| Модель | Файл | Описание |
|--------|------|----------|
| Ridge | `sirena/models/ridge.py` | Baseline модель |
| Ridge Extended | `sirena/models/ridge_extended.py` | С дополнительными признаками |
| Ridge Shock | `sirena/models/ridge_shock_dummies.py` | С dummy-переменными для шоков |
| ElasticNet | `sirena/models/elasticnet.py` | L1+L2 регуляризация |
| Huber | `sirena/models/huber.py` | Робастная к выбросам |
| NGBoost | `sirena/models/ngboost_model.py` | Probabilistic boosting |
| NGBoost Shock | `sirena/models/ngboost_shock.py` | NGBoost с shock dummies |
| Prophet | `sirena/models/prophet.py` | Facebook Prophet |
| EBM | `sirena/models/ebm.py` | Explainable Boosting |
| **Subcomponent Multi** | `sirena/models/subcomponent_multi.py` | Оптимальные модели по 45 субкомпонентам |
| **Scenario Rate** | `sirena/models/scenario_rate.py` | Сценарная модель с калибровкой |
| Horizon Ensemble | `sirena/models/horizon_ensemble.py` | Адаптивный ансамбль (Huber + Micro) |
| Micro Component | `sirena/models/microcomponent.py` | Bottom-up (537 товаров) |

---

## Как запустить бэктесты

```bash
cd /home/valalav/_projects/sirena-kbr

# h=1 (САМЫЙ ВАЖНЫЙ КПЭ)
python3 scripts/run_backtest_h1.py

# h=2
python3 scripts/run_backtest_h2.py

# h=12
python3 scripts/run_backtest_h12.py

# Результаты сохраняются в:
ls archive/results/backtest_h*.csv
```

---

## Автор

Claude Code / Gemini
Дата: 2026-01-21

**Версия системы**: СИРЕНА-КБР v5.2

## Описание

Данная методика описывает 3 варианта бэктестирования моделей прогнозирования инфляции КБР с разными горизонтами:
- **h=1**: Прогноз на 1 месяц вперед (главный КПЭ)
- **h=2**: Прогноз на 2 месяца вперед
- **h=12**: Прогноз на 12 месяцев вперед (годовая траектория)

Все бэктесты автоматизированы через скрипты в `scripts/`.

---

## h=1: Прогноз на 1 месяц вперед (ГЛАВНЫЙ КПЭ)

### Описание

Rolling window backtest за последние 12 месяцев. Для каждого месяца модель обучается на всех доступных данных до `t-1` и прогнозирует месяц `t`.

### Логика

```
Для каждого месяца t (декабрь 2024 → ноябрь 2025):
  cutoff = t - 1 месяц
  train = все данные до cutoff
  прогноз = forecast(1)[0]
  actual = inflation_data.csv[t, 'mom'] - 100
  error = actual - прогноз
```

### Пример

- **Ноябрь 2025**: train до октября 2025 → прогноз на ноябрь
- **Октябрь 2025**: train до сентября 2025 → прогноз на октябрь
- **Сентябрь 2025**: train до августа 2025 → прогноз на сентябрь
- ... и так далее

**Итого**: 12 независимых обучений, 12 прогнозов

### Метрики

- **MAE** — Mean Absolute Error (главная метрика)
- **RMSE** — Root Mean Squared Error
- **KPI Violations** — количество месяцев с |error| > 0.5
- **Coverage 50%** — % точек, где |error| <= 0.5

### Результаты (декабрь 2024 — ноябрь 2025)

**Top 3 модели:**

1. **Huber**: MAE 0.289 (4 KPI violations из 12)
2. **Ridge Shock**: MAE 0.299 (4 KPI violations)
3. **ElasticNet**: MAE 0.301 (5 KPI violations)

**Файлы:**
- `archive/results/backtest_h1_predictions.csv` — все прогнозы
- `archive/results/backtest_h1_metrics.csv` — метрики моделей
- `archive/results/backtest_h1_summary.md` — markdown отчет

---

## h=2: Прогноз на 2 месяца вперед

### Описание

Rolling window backtest за последние 12 месяцев. Для каждого месяца модель обучается на данных до `t-2` и прогнозирует месяц `t` (второй месяц из `forecast(2)`).

### Логика

```
Для каждого месяца t (декабрь 2024 → ноябрь 2025):
  cutoff = t - 2 месяца
  train = все данные до cutoff
  прогноз = forecast(2)[1]  # ВТОРОЙ месяц (индекс 1)
  actual = inflation_data.csv[t, 'mom'] - 100
  error = actual - прогноз
```

### Пример

- **Ноябрь 2025**: train до сентября → прогноз [октябрь, **ноябрь**] → берем ноябрь
- **Октябрь 2025**: train до августа → прогноз [сентябрь, **октябрь**] → берем октябрь
- **Сентябрь 2025**: train до июля → прогноз [август, **сентябрь**] → берем сентябрь

**Итого**: 12 независимых обучений, 12 прогнозов

### Особенности

- **Меньше данных** чем h=1 (cutoff на 2 месяца раньше)
- **Более сложный прогноз** — нужно предсказать через 1 промежуточный месяц
- **Сравнение с h=1** показывает потерю точности при увеличении горизонта

### Результаты (декабрь 2024 — ноябрь 2025)

**Top 3 модели:**

1. **NGBoost Shock**: MAE 0.291 (2 KPI violations из 12)
2. **NGBoost**: MAE 0.302 (3 KPI violations)
3. **EBM**: MAE 0.305 (3 KPI violations)

**Файлы:**
- `archive/results/backtest_h2_predictions.csv`
- `archive/results/backtest_h2_metrics.csv`
- `archive/results/backtest_h2_summary.md`

---

## h=12: Прогноз на 12 месяцев вперед (годовая траектория)

### Описание

Backtest на основе фиксированного cutoff. Модель обучается один раз на данных до ноября 2024 и прогнозирует траекторию на 12 месяцев вперед (декабрь 2024 — ноябрь 2025).

### Логика

```
cutoff = ноябрь 2024 (фиксированный)
train = все данные до cutoff
прогноз = forecast(12)  # траектория [декабрь 2024, ..., ноябрь 2025]

Для каждого месяца в траектории (i = 0..11):
  prediction = forecast(12)[i]
  actual = inflation_data.csv[декабрь 2024 + i месяцев, 'mom'] - 100
  error = actual - prediction
```

### Пример

- **Декабрь 2024**: `forecast(12)[0]` vs факт
- **Январь 2025**: `forecast(12)[1]` vs факт
- **Февраль 2025**: `forecast(12)[2]` vs факт
- ...
- **Ноябрь 2025**: `forecast(12)[11]` vs факт

**Итого**: 1 обучение, 12 прогнозов из одной траектории

### Особенности

- **Фиксированный cutoff** — все модели используют одни и те же данные для обучения
- **Долгосрочный прогноз** — оценка стабильности модели на длинном горизонте
- **Нет промежуточного дообучения** — модель не обновляется между месяцами

### Результаты (декабрь 2024 — ноябрь 2025)

**Top 3 модели:**

1. **Prophet**: MAE 0.277 (0 KPI violations из 12) — **Лучшая на длинном горизонте!**
2. **Ridge**: MAE 0.338 (4 KPI violations)
3. **Ridge Extended**: MAE 0.339 (4 KPI violations)

**Важно**: BVAR показал нестабильность на длинном горизонте (выдал только 4 точки вместо 12).

**Файлы:**
- `archive/results/backtest_h12_predictions.csv`
- `archive/results/backtest_h12_metrics.csv`
- `archive/results/backtest_h12_summary.md`

---

## Source of Truth

**Единственный источник актуальных значений:** `data/inflation_data.csv`

- Формат: sep=';', decimal=','
- Колонки: Date, mom, Prod, Nonprod, Serv, usd_nom_i, Ki_i, Ruonia
- Используется для:
  - BVAR моделей (прямо)
  - Ridge/ML моделей (через pivot)
  - Получения актуальных значений для сравнения

---

## Модели

**18 моделей в бэктесте:**

| Модель | Файл | Описание |
|--------|------|----------|
| Ridge | `archive/scripts/sirena_kbr_v2_4_auto.py` | Baseline модель v2.4 |
| Ridge Extended | `sirena/models/ridge_extended.py` | С дополнительными признаками |
| Ridge Shock | `sirena/models/ridge_shock_dummies.py` | С dummy-переменными для шоков |
| Bayesian Ridge | `sirena/models/bayesian_ridge.py` | С доверительными интервалами |
| ElasticNet | `sirena/models/elasticnet.py` | L1+L2 регуляризация |
| Huber | `sirena/models/huber.py` | Робастная к выбросам |
| NGBoost | `sirena/models/ngboost_model.py` | Probabilistic boosting |
| NGBoost Shock | `sirena/models/ngboost_shock.py` | NGBoost с shock dummies |
| LMMR Claude | `sirena/models/lmmr_claude.py` | Linear Mixed Model |
| LMMR Hybrid | `sirena/models/lmmr_hybrid.py` | Гибридная LMMR |
| BVAR | `sirena/models/bvar.py` | Bayesian VAR |
| SARIMA | `archive/scripts/sirena_arima.py` | Seasonal ARIMA |
| LightGBM | `sirena/models/lightgbm.py` | Gradient Boosting |
| Prophet | `sirena/models/prophet.py` | Facebook Prophet |
| ETS | `sirena/models/ets.py` | Exponential Smoothing |
| EBM | `sirena/models/ebm.py` | Explainable Boosting |
| CatBoost | `sirena/models/catboost_model.py` | Опционально |
| Ensemble | — | Взвешенная комбинация 7 моделей |

---

## Как запустить бэктесты

```bash
cd /home/valalav/_projects/sirena-kbr

# h=1 (САМЫЙ ВАЖНЫЙ КПЭ)
python3 scripts/run_backtest_h1.py

# h=2
python3 scripts/run_backtest_h2.py

# h=12
python3 scripts/run_backtest_h12.py

# Результаты сохраняются в:
ls archive/results/backtest_h*.csv
```

---

## Выводы

### Лучшие модели по горизонтам:

**h=1 (1 месяц вперед — главный КПЭ):**
1. Huber (MAE 0.289)
2. Ridge Shock (MAE 0.299)
3. ElasticNet (MAE 0.301)

**h=2 (2 месяца вперед):**
1. NGBoost Shock (MAE 0.291)
2. NGBoost (MAE 0.302)
3. EBM (MAE 0.305)

**h=12 (годовая траектория):**
1. Prophet (MAE 0.277) — **Лучшая на длинном горизонте!**
2. Ridge (MAE 0.338)
3. Ridge Extended (MAE 0.339)

### Ключевые наблюдения:

1. **Huber** лучше всего на h=1 (главный КПЭ)
2. **NGBoost Shock** хорош на h=2
3. **Prophet** превосходит все модели на h=12
4. **BVAR** нестабилен на длинных горизонтах
5. MAE растет с увеличением горизонта (0.289 → 0.291 → 0.277), но h=12 показывает низкую MAE благодаря Prophet

---

## Автор

Claude Code
Дата: 2025-12-25

**Версия системы**: СИРЕНА-КБР v4.7
