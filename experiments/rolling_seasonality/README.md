# Эксперимент: Rolling Seasonality Ridge

## 🎯 Цель

Проверить гипотезу: **скользящая сезонность** (на последних 24-36 месяцах) работает лучше **глобальной сезонности** (на всей истории) после структурных сдвигов 2022-2024.

---

## 📁 Структура эксперимента

```
experiments/rolling_seasonality/
├── README.md                          # Этот файл
├── docs/
│   ├── RESEARCH_PROPOSAL.md          # Предпосылки и гипотезы
│   └── RESULTS.md                     # Результаты бэктеста
├── models/
│   └── rolling_seasonality_ridge.py   # Модель
├── scripts/
│   ├── run_backtest_rolling.py        # Скрипт бэктеста
│   └── plot_results.py                # Генерация графиков
└── results/                           # 📊 РЕЗУЛЬТАТЫ (создаётся автоматически)
    ├── backtest_summary_YYYYMMDD_HHMMSS.csv
    ├── predictions_{model}_YYYYMMDD_HHMMSS.csv
    ├── forecast_comparison.png
    ├── mae_comparison.png
    ├── cumulative_errors.png
    ├── error_distribution.png
    ├── seasonal_norms_comparison.png
    └── summary_table.png
```

---

## 🚀 Быстрый старт

### 1. Запуск бэктеста

```bash
cd /home/valalav/_projects/sirena-kbr/experiments/rolling_seasonality
python3 scripts/run_backtest_rolling.py
```

**Вывод:**
- CSV-файлы сохраняются в: `experiments/rolling_seasonality/results/`
- Сводка: `backtest_summary_YYYYMMDD_HHMMSS.csv`
- Предсказания: `predictions_{model}_YYYYMMDD_HHMMSS.csv`

### 2. Генерация графиков

```bash
python3 scripts/plot_results.py
```

**Вывод (6 графиков):**
- `results/forecast_comparison.png` — сравнение прогнозов vs факт
- `results/mae_comparison.png` — сравнение MAE
- `results/cumulative_errors.png` — кумулятивные ошибки
- `results/error_distribution.png` — распределение ошибок
- `results/seasonal_norms_comparison.png` — сравнение сезонных норм
- `results/summary_table.png` — таблица результатов

### 3. Просмотр результатов

```bash
# CSV-файлы
ls -la results/*.csv

# Графики
ls -la results/*.png

# Последние результаты
ls -t results/*.csv | head -1
```

---

## 📊 Что тестируется

Скрипт автоматически сравнивает:
- **Ridge (baseline)** — глобальная сезонность
- **Huber (best)** — лучшая production модель
- **Rolling_24m** — сезонность на 24 месяцах
- **Rolling_36m** — сезонность на 36 месяцах
- **Rolling_48m** — сезонность на 48 месяцах

---

## 📈 Метрики и результаты

### Метрики
- **MAE** — главная метрика (Mean Absolute Error)
- **RMSE** — Root Mean Squared Error
- **KPI Rate** — % прогнозов с |error| ≤ 0.5
- **KPI Violations** — количество "промахов" > 0.5

### Последние результаты (2025-01 — 2025-12)

| Модель | MAE | vs Ridge | KPI Violations | Путь к результатам |
|--------|-----|----------|----------------|-------------------|
| **Rolling_24m** | **0.314** | **+6.4%** ✅ | **2/12** | `results/predictions_Rolling_24m_*.csv` |
| Huber | 0.318 | +5.2% | 3/12 | `results/predictions_Huber_*.csv` |
| Ridge (baseline) | 0.335 | — | 3/12 | `results/predictions_Ridge_*.csv` |
| Rolling_48m | 0.366 | -9.3% ❌ | 3/12 | `results/predictions_Rolling_48m_*.csv` |
| Rolling_36m | 0.370 | -10.3% ❌ | 4/12 | `results/predictions_Rolling_36m_*.csv` |

---

## 🔍 Ключевые отличия от базовой Ridge

| Аспект | Базовая Ridge | RollingSeasonalityRidge |
|--------|---------------|------------------------|
| Сезонность | Глобальная (вся история) | Rolling window (24-48 мес.) |
| Исключения | 2022, 2010 | Только 2010 |
| Адаптация | Нет | Да, к последним данным |

---

## 📖 Документация

- [docs/RESEARCH_PROPOSAL.md](docs/RESEARCH_PROPOSAL.md) — детальное описание проблемы и гипотезы
- [docs/RESULTS.md](docs/RESULTS.md) — полные результаты и анализ

---

## 🔄 Интеграция с основным проектом

Модель зарегистрирована в `ModelRegistry`:
```python
from sirena.models import ModelRegistry

model = ModelRegistry.get("rolling_seasonality_ridge")
# или
from experiments.rolling_seasonality.models import RollingSeasonalityRidge
model = RollingSeasonalityRidge(seasonality_window=36)
```

---

## 🧪 Параметры для экспериментов

```python
# Разные окна сезонности
RollingSeasonalityRidge(seasonality_window=24)  # 2 года
RollingSeasonalityRidge(seasonality_window=36)  # 3 года
RollingSeasonalityRidge(seasonality_window=48)  # 4 года

# С макро-признаками (Ki, Ruonia)
RollingSeasonalityRidge(seasonality_window=36, use_macro=True)

# Без макро-признаков
RollingSeasonalityRidge(seasonality_window=36, use_macro=False)
```

---

## 📁 Пути к файлам (для справки)

### Модель
```
experiments/rolling_seasonality/models/rolling_seasonality_ridge.py
```

### Скрипты
```
experiments/rolling_seasonality/scripts/run_backtest_rolling.py
experiments/rolling_seasonality/scripts/plot_results.py
```

### Результаты (после запуска)
```
experiments/rolling_seasonality/results/
├── backtest_summary_YYYYMMDD_HHMMSS.csv
├── predictions_Ridge_baseline_YYYYMMDD_HHMMSS.csv
├── predictions_Huber_best_YYYYMMDD_HHMMSS.csv
├── predictions_Rolling_24m_YYYYMMDD_HHMMSS.csv
├── predictions_Rolling_36m_YYYYMMDD_HHMMSS.csv
├── predictions_Rolling_48m_YYYYMMDD_HHMMSS.csv
├── forecast_comparison.png
├── mae_comparison.png
├── cumulative_errors.png
├── error_distribution.png
├── seasonal_norms_comparison.png
└── summary_table.png
```

---

## 📝 Лог изменений

- **2026-02-02** — Создание эксперимента, базовая модель, бэктест, визуализация

---

## 👤 Автор

Claude Code

---

*Эксперимент изолирован в `experiments/` и не влияет на production код до явного решения о внедрении.*  
*Результаты сохраняются в `experiments/rolling_seasonality/results/`*
