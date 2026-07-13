# GEMINI.md

This file provides guidance to Gemini when working with code in this repository.

## Архитектура документации (Универсальные принципы)

> **Эти принципы применимы ко ВСЕМ проектам для оптимизации контекстного окна.**

### 1. Lazy Loading (Ленивая загрузка)

Не читай всю документацию сразу. Открывай только то, что нужно для текущей задачи:

- Добавляешь модель → `docs/ADDING_MODEL_GUIDE.md`
- Проверяешь систему → `docs/VERIFICATION_GUIDE.md`
- Нужен API → `docs/API.md`

### 2. Index Pattern (Индексный паттерн)

`docs/index.md` — навигационный хаб с кратким описанием каждого файла. Начинай с индекса.

### 3. Skill Pattern (Паттерн скилов)

Сложные workflow в `.agent/workflows/`:

- `/update-nowcast` — обновление nowcast
- `/add-model` — добавление модели
- `/run-backtest` — запуск бэктеста

### 4. Компактный GEMINI.md

GEMINI.md содержит только:

- Критические директивы (не врать)
- Ссылки на документацию
- Базовые команды

**Цель:** < 400 строк

---

## КРИТИЧЕСКАЯ ДИРЕКТИВА: НЕ ВРАТЬ

**Это самое важное правило. Нарушение = потеря доверия.**

### Что запрещено

1. **Говорить "готово" без РЕАЛЬНОЙ проверки**
   - "Скрипт запустился" ≠ "UI работает"
   - "Код без ошибок" ≠ "Функционал доступен"

2. **Создавать верификацию которая проверяет не то**

3. **Оптимизировать на "выглядит как работа"**

### Что обязательно

1. **Перед "готово" — проверь ИМЕННО то, что просил пользователь**
2. **Если не можешь проверить — честно скажи**
3. **Если "не работает" — сначала ПОСМОТРИ код, потом отвечай**

---

## При открытии новой сессии

1. Проверь `CURRENT_TASK.md` — незавершенные задачи
2. Проверь `git log -1` — последний коммит
3. Предложи продолжить работу

---

## Git workflow

- После завершения и проверки задачи создай сфокусированный коммит и сразу отправь его в upstream.
- Если upstream не настроен, выполни `git push --set-upstream origin HEAD`.
- Не накапливай завершённые коммиты только локально.
- При ошибке push сохрани локальный коммит и сообщи точную ошибку.

---

## Project Overview

**СИРЕНА-КБР v5.4** — система прогнозирования инфляции (ИПЦ) в Кабардино-Балкарской Республике.

**Данные:** до **Февраля 2026** (из `data/db_cpi_store.accdb`), недельные до **30.03.2026**

**Production Ensemble:** 9 моделей (Huber, RidgeShockDummies, ElasticNet, NGBoostShock, NGBoost, Ridge, RidgeExtended, Prophet, EBM)

**Dashboard:** `http://localhost:8503` (порт 8503!)

**Лучшая модель (SIRENA Score):** SubcomponentMulti (Score 0.515, MAE h=1: 0.265)

**Веса компонентов КБР (Январь 2026):** Прод 39.86%, Непрод 36.38%, Услуги 23.76%

---

## 🎯 КПЭ (Ключевые Показатели Эффективности)

**Главный КПЭ:** Отклонение прогноза MoM ИПЦ от факта **не более ±0.5 п.п.** (по модулю).

| Параметр | Значение |
|----------|----------|
| **Порог** | ±0.5 п.п. |
| **Горизонт** | h=1 (1 месяц вперёд) — главный |
| **Целевой MAE** | ≤ 0.35 |
| **Метод оценки** | % месяцев попадания в коридор |

**Пример:** Прогноз +0.70% → факт должен быть в диапазоне +0.20% — +1.20%.

**Историческая статистика:**
- 2024: КПЭ выполнен ~100% (12/12 месяцев)
- 2025: КПЭ не выполнен — промахи в 4-5 месяцах из-за роста волатильности (+35.7% σ)

> **При формировании прогнозов ВСЕГДА проверяй попадание в коридор ±0.5 п.п.**

---

## Ключевая документация

| Документ | Когда открывать |
|----------|----------------|
| **[docs/index.md](docs/index.md)** | Навигация по всей документации |
| **[docs/ADDING_MODEL_GUIDE.md](docs/ADDING_MODEL_GUIDE.md)** | Добавление новой модели |
| **[docs/VERIFICATION_GUIDE.md](docs/VERIFICATION_GUIDE.md)** | Верификация системы |
| **[docs/MODEL_CATALOG.md](docs/MODEL_CATALOG.md)** | Каталог моделей с примерами |
| **[docs/EDGE_LAB_REFERENCE.md](docs/EDGE_LAB_REFERENCE.md)** | Ralph / Edge Lab |
| **[docs/NOWCASTING.md](docs/NOWCASTING.md)** | Недельный nowcast |
| **[docs/FREEZE_ANALYSIS.md](docs/FREEZE_ANALYSIS.md)** | Анализ заморозки цен (PSI, FDI) |
| **[README.md](README.md)** | Полная документация проекта |

---

## Основные команды

```bash
# Dashboard (порт 8503!)
streamlit run dashboard.py --server.port 8503

# Пересчёт прогнозов
python3 scripts/precompute_forecasts.py

# Генерация графиков
python3 scripts/generate_charts.py

# Бэктест (5 горизонтов)
python3 scripts/run_backtest_h1.py  # h=1 (самый важный)
python3 scripts/run_backtest_h2.py  # h=2
python3 scripts/run_backtest_h3.py  # h=3
python3 scripts/run_backtest_h6.py  # h=6
python3 scripts/run_backtest_h12.py # h=12 (годовая траектория)

# Тесты
pytest tests/ -v

# Верификация
python3 scripts/verify_all_tabs.py

# Обработка обновления db_cpi_store.accdb
python3 scripts/process_accdb_update.py
```

---

## 📁 Структура результатов и артефактов

### 🔄 Синхронизация (Syncthing) — ГЛАВНАЯ ПАПКА

```
sync/                                    # ← Добавьте в Syncthing
├── charts/                              # 📊 PNG графики
├── csv/                                 # 📄 CSV-результаты
├── html/                                # 🌐 HTML-визуализации
├── experiments/                         # 🧪 Результаты экспериментов
└── reports/                             # 📑 Markdown отчёты

# ✅ Обновление АВТОМАТИЧЕСКОЕ после каждого бэктеста
# Ручное обновление: python3 scripts/sync_to_share.py
```

### Бэктесты и прогнозы (Production)

```
archive/results/                         # Исходные результаты
├── backtest_h1_predictions.csv          # Прогнозы h=1
├── backtest_h1_metrics.csv              # Метрики h=1
├── backtest_h1_summary.md               # Сводка h=1
├── backtest_h2_predictions.csv          # Прогнозы h=2
├── backtest_h2_metrics.csv              # Метрики h=2
├── backtest_h12_predictions.csv         # Прогнозы h=12
├── backtest_h12_metrics.csv             # Метрики h=12
├── forecasts_current.csv                # Текущие прогнозы
└── model_comparison.csv                 # Сравнение моделей
```

### Графики и визуализации (Production)

```
assets/charts/                           # Исходные графики
├── backtest_h1_predictions.html         # Интерактивные графики h=1
├── backtest_h1_errors.html              # Графики ошибок h=1
├── backtest_h12_predictions.html        # Интерактивные графики h=12
├── forecasts.html                       # Графики прогнозов
├── model_comparison.html                # Сравнение моделей
├── nowcast.html                         # Nowcast визуализация
└── sirena_score_dynamics.html           # Динамика SIRENA Score
```

### Данные

```
data/
├── inflation_data.csv                   # Основные данные (источник)
├── infl_kbr.csv                         # Альтернативный формат
├── precomputed_forecasts.json           # Предвычисленные прогнозы
├── kbr_weekly_prices_2008_2026.csv     # Недельные данные
└── raw/                                 # Сырые данные
```

### Эксперименты

```
experiments/
└── {experiment_name}/
    ├── results/                         # Результаты эксперимента
    │   ├── backtest_summary_*.csv
    │   ├── predictions_*.csv
    │   └── *.png                        # Графики
    └── docs/
        ├── RESEARCH_PROPOSAL.md
        └── RESULTS.md
```

---

## Workflows (Slash Commands)

| Команда | Описание |
|---------|----------|
| `/update-nowcast` | Обновить nowcast из недельных данных |
| `/add-model` | Добавить новую модель (11 шагов) |
| `/run-backtest` | Запустить бэктест |

---

## File Organization

- **Root**: Только `README.md`, `GEMINI.md`, `dashboard.py`, `requirements.txt`
- **sirena/models/**: 40+ моделей прогнозирования
- **Scripts**: `scripts/` — 120+ скриптов
- **Data**: `data/` — все CSV/Excel/JSON
- **Docs**: `docs/` — вся документация
- **Pages**: `pages/` — страницы дашборда Streamlit
- **Archive**: `archive/` — устаревшее и результаты бэктестов
- **Assets**: `assets/charts/` — графики и визуализации
- **Experiments**: `experiments/` — изолированные эксперименты

---

## Верификация перед "Готово"

```bash
# Полная проверка
python3 scripts/verify_all_tabs.py

# Результат должен быть: ✅ ВСЕ ПРОВЕРКИ ПРОЙДЕНЫ
```

Детали: [docs/VERIFICATION_GUIDE.md](docs/VERIFICATION_GUIDE.md)

---

## Nowcast (Оперативная корректировка)

Формула: `Nowcast = W_weekly × Weekly_Signal + W_model × Ensemble_h1`

Веса зависят от числа недель: 1 нед → 60/40, 2 нед → 70/30, 3 нед → 80/20, 4 нед → 90/10.

Обновление: `/update-nowcast` или:

```bash
python3 scripts/precompute_forecasts.py
python3 scripts/generate_charts.py
```

**ВАЖНО:** Использовать столбец `Справка_нед.Компоненты` (НЕ `Компонент`).

### Текущий Nowcast: Март 2026 (обновлено 08.04.2026)

| Показатель | Значение |
|---|---|
| **Nowcast MoM** | **+0.62%** |
| Weekly Signal | +0.77% (кумулятивный, 5 недель) |
| Ensemble h=1 | +0.26% |
| Веса | 70% weekly / 30% model |
| Недели | 02.03(+0.28%), 10.03(+0.12%), 16.03(+0.50%), 23.03(+0.40%), 30.03(-0.52%) |

> ⚠️ Последняя неделя 30.03 дефляционная (-0.52%). Драйверы: мясопереработка (колбасы/сосиски -0.077 п.п.) и плодоовощи (огурцы/картофель -0.044 п.п.) — сезонная коррекция.

### Верифицированные Закономерности (6 марта 2026)

- **Цена последней недели = Месячная цена** в 95% случаев (2896 сравнений, |diff| ≤ 1%)
- **Кумулятивный недельный индекс ≈ MoM**: корреляция 0.81, медиана |diff| = 0.40 п.п.
- **Bias:** недельные данные слегка занижают факт (-0.09 п.п.)

### Structural Shock Verification (Q1 2026)

При выявлении аномалий (>1.5σ) сверяйтесь с **[Commodity Risk Ledger](docs/research/commodity_risk_ledger_2026.md)**.
Если прогноз Deep Research не сбылся — фиксируйте причину в столбце Verdict.

Детали: [docs/NOWCASTING.md](docs/NOWCASTING.md)

---

## Ralph Edge Lab

Автономная система Worker-Critic-Refiner.

```bash
cd edge_lab
python3 system/orchestrator.py
```

---

## 🔄 СинХронизация результатов

Когда пользователь говорит **"синхронизируй"** или **"обнови sync"**:

```bash
python3 scripts/sync_to_share.py
```

Этот скрипт копирует актуальные результаты, графики и отчёты в папку `sync/` для Syncthing.

**Папка для Syncthing:** `sync/` (добавьте её в Syncthing)

---

## 🛡️ Critical Review Protocol

For financial or legal analysis, use the `/critical-review` skill.

- **Trigger**: "Check this critically", "Verify sources".
- **Action**: Enforces Tier 1 Source verification and Counter-Argument search.

### ✅ Verification Protocol (New)

- **Use `/critical-review`**: For any sensitive task (Legal, Financial, Medical), invoke this skill to enforce adversarial verification.
- **Protocol**: Source Hierarchy Audit -> Devil's Advocate Loop -> Verified Output.

## 🧮 Методология (Methodology)

### Расчет вклада в инфляцию (Contribution)
>
> **Формула:** `Вклад (п.п.) = Прирост цены (%) × Вес в корзине (доля)`

**Пример:**

- Товар: Говядина
- Рост цены: +29.6%
- Вес в корзине: 0.0158 (1.58%)
- Вклад: `29.6 * 0.0158 = 0.467 п.п.` (в общем индексе)

**Источники весов (обновлены 19 февраля 2026):**

1. **`data/micro_sprav.csv`** — Первичный источник (537 товаров, веса Росстата Январь 2026). Колонки: `Item_code; Товар; Weight`.
2. **`data/access_weights.csv`** — Полный справочник весов из ACCDB (820K строк). Колонки: `Code, Day, Region_code, Item_code, Weight_horizontal, Weight_vertical, Weight_gross`.

**Правило:** При анализе драйверов инфляции **ВСЕГДА** учитывать вес. Товар с ростом +100% и весом 0.0001% (Спички) менее важен, чем товар с ростом +10% и весом 1.5% (Мясо).

---

## 📊 Сезонное сглаживание (X-13ARIMA-SEATS)

**Бинарник:** `bin/linux/x13as_ascii` (US Census Bureau, кросс-платформенный)

**Архитектура:**
```
inflation_data.csv → п.п.→индекс (cumprod) → X-13 SEATS → п.п. (pct_change) → кэш SHA256
```

| Файл | Назначение |
|---|---|
| `import/x13.py` | Драйвер X-13 (Plan A: automdl → Plan B: airline 011·011) |
| `import/seasonal_adjustment_x13.py` | Менеджер с SHA256-кэшированием |
| `import/seasonality_analyzer.py` | STL + classical decompose (statsmodels) |
| `import/SEASONAL_ADJUSTMENT_GUIDE.md` | Документация v2.0 |

**Использование:** Автоматически через `load_data()` → `perform_seasonal_adjustment()`. Кэш: `data/cache/sa_cache_{hash}.csv`.

> При работе с сглаживанием ВСЕГДА сначала читай `import/SEASONAL_ADJUSTMENT_GUIDE.md`.

## 📊 Сезонное сглаживание (JDemetra+ v3 TRAMO-SEATS)

**JAR:** `bin/linux/jdemetra_sa.jar` (JDemetra+ v3.7.1, fat JAR ~4 МБ)
**Java:** Требуется Java 21+ (`openjdk-21-jdk-headless`)

**Архитектура:**
```
CSV (date,value) → java -jar jdemetra_sa.jar --method tramo-seats → JSON (seasadj, trend, seasonal, irregular)
```

| Файл | Назначение |
|---|---|
| `tools/jdemetra/JDemetraSA.java` | Java CLI (TRAMO-SEATS + X-13 через JDemetra+ API) |
| `tools/jdemetra/pom.xml` | Maven проект (зависимости eu.europa.ec.joinup.sat) |
| `import/jdemetra.py` | Python обёртка (subprocess → JSON → pandas) |

**Использование из Python:**
```python
from import.jdemetra import jdemetra_tramo_seats
result = jdemetra_tramo_seats(series_index)  # pd.Series с DatetimeIndex
# result.seasadj, result.trend, result.seasonal, result.irregular
```

**Пересборка JAR:** `cd tools/jdemetra && mvn package -B && cp target/jdemetra-sa-1.0.jar ../../bin/linux/jdemetra_sa.jar`

---

## 📈 Прогнозы и предположения

**Папка:** `forecasts/`

| Файл | Описание |
|---|---|
| `2026_mom_forecast_v1.md` | MoM прогноз 2026, v1 (10.03.2026). Тарифы 2-шаговые, НДС, исторический анализ |
| `2027_mom_forecast_v1.md` | MoM прогноз 2027, v1 (10.03.2026). Дезинфляция до 4.0%, сезонный профиль |
| `SA_forecast_2026_2027.xlsx` | SA/SAAR расчёт (JDemetra+ TRAMO-SEATS) на весь горизонт 2010-2027 |
| `subcomponent_forecast_2026_2027.xlsx` | Подкомпонентный прогноз: продовольствие/непроды/услуги |
| `scenario_analysis_2026_2027.xlsx` | Сценарный анализ: base/low/high |
| `charts/forecast_overview_2026_2027.png` | 4-панельный обзор: MoM, подкомпоненты, YoY, SAAR |
| `charts/full_timeline_mom.png` | Полная история MoM 2010-2027 с прогнозом |
| `nowcast_log.csv` | Лог nowcast обновлений |

### Скрипты прогнозирования

| Файл | Назначение |
|---|---|
| `scripts/nowcast_update.py` | Автообновление nowcast из еженедельных данных Росстата |

```bash
# Nowcast текущего месяца
python3 scripts/nowcast_update.py

# Nowcast конкретного месяца
python3 scripts/nowcast_update.py --month 3
```

### Изменение правил индексации тарифов (с 2026)

- **Январь:** Частичная индексация ~1.7%
- **Июль:** Индексации **НЕ БУДЕТ** (обычно +2.5-3.7% услуги = +0.8 п.п. к ИПЦ)
- **Октябрь:** Основная индексация **10%+** (~+2.4 п.п. к ИПЦ через услуги)

> ⚠️ Модели ансамбля **не знают** об изменении правил индексации. Экспертная коррекция обязательна для Июл/Окт.

### Целевые YoY

| Год | Base | Low | High |
|---|---|---|---|
| 2026 | 5.68% | 5.05% | 6.31% |
| 2027 | 4.02% | 3.40% | 4.64% |

---

## Production Models (Актуальный ансамбль v4.8)

| Модель | Вес | MAE h=1 | Файл |
|--------|-----|---------|------|
| Huber | 18% | 0.289 | `sirena/models/huber.py` |
| RidgeShockDummies | 17% | 0.299 | `sirena/models/ridge_shock_dummies.py` |
| ElasticNet | 17% | 0.301 | `sirena/models/elasticnet.py` |
| NGBoostShock | 16% | 0.291 | `sirena/models/ngboost_shock.py` |
| NGBoost | 12% | 0.312 | `sirena/models/ngboost_model.py` |
| Ridge | 8% | 0.310 | `sirena/models/ridge.py` |
| RidgeExtended | 5% | 0.318 | `sirena/models/ridge_extended.py` |
| Prophet | 4% | 0.277 | `sirena/models/prophet.py` |
| EBM | 3% | 0.340 | `sirena/models/ebm.py` |

**Лучшая модель (SIRENA Score):** SubcomponentMulti (0.515) — `sirena/models/subcomponent_multi.py`

> **Известная проблема (10.03.2026):** EBM выдаёт MoM в формате индекса (~201) вместо п.п. (~1.0). Исключать из ансамбля или пересчитывать.

> ✅ **NumPy fix (10.03.2026):** Даунгрейд numpy до 1.26.4 (`pip install "numpy<2" --break-system-packages`). Все модели теперь импортируются корректно.

---

*Последнее обновление: 8 апреля 2026*
