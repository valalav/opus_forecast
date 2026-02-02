# Эксперимент: Rolling Seasonality Ridge

## 🎯 Цель

Проверить гипотезу: **скользящая сезонность** (на последних 24-36 месяцах) работает лучше **глобальной сезонности** (на всей истории) после структурных сдвигов 2022-2024.

## 📁 Структура

```
experiments/rolling_seasonality/
├── README.md                          # Этот файл
├── docs/
│   ├── RESEARCH_PROPOSAL.md          # Предпосылки и гипотезы
│   └── RESULTS.md                     # Результаты бэктеста (заполняется после запуска)
├── models/
│   └── rolling_seasonality_ridge.py   # Модель
├── scripts/
│   └── run_backtest_rolling.py        # Скрипт бэктеста
└── results/                           # Результаты (создаётся автоматически)
    ├── backtest_summary_YYYYMMDD_HHMMSS.csv
    └── predictions_*_YYYYMMDD_HHMMSS.csv
```

## 🚀 Быстрый старт

### 1. Запуск бэктеста

```bash
cd /home/valalav/_projects/sirena-kbr/experiments/rolling_seasonality
python3 scripts/run_backtest_rolling.py
```

### 2. Что тестируется

Скрипт автоматически сравнивает:
- **Ridge (baseline)** — глобальная сезонность
- **Huber (best)** — лучшая production модель
- **Rolling_24m** — сезонность на 24 месяцах
- **Rolling_36m** — сезонность на 36 месяцах
- **Rolling_48m** — сезонность на 48 месяцах

### 3. Результаты

После запуска в `results/` появятся:
- `backtest_summary_*.csv` — сводная таблица метрик
- `predictions_*.csv` — детальные предсказания по месяцам

## 📊 Метрики

- **MAE** — главная метрика (Mean Absolute Error)
- **RMSE** — Root Mean Squared Error
- **KPI Rate** — % прогнозов с |error| ≤ 0.5
- **KPI Violations** — количество "промахов" > 0.5

## 🔍 Ключевые отличия от базовой Ridge

| Аспект | Базовая Ridge | RollingSeasonalityRidge |
|--------|---------------|------------------------|
| Сезонность | Глобальная (вся история) | Rolling window (24-48 мес.) |
| Исключения | 2022, 2010 | Только 2010 |
| Адаптация | Нет | Да, к последним данным |

## 📖 Документация

- [RESEARCH_PROPOSAL.md](docs/RESEARCH_PROPOSAL.md) — детальное описание проблемы и гипотезы
- [RESULTS.md](docs/RESULTS.md) — результаты и анализ (обновляется после запуска)

## 🔄 Интеграция с основным проектом

Модель зарегистрирована в `ModelRegistry`:
```python
from sirena.models import ModelRegistry

model = ModelRegistry.get("rolling_seasonality_ridge")
# или
from experiments.rolling_seasonality.models import RollingSeasonalityRidge
model = RollingSeasonalityRidge(seasonality_window=36)
```

## 🧪 Параметры для экспериментов

```python
# Разные окна сезонности
RollingSeasonalityRidge(seasonality_window=24)  # 2 года
RollingSeasonalityRidge(seasonality_window=36)  # 3 года (рекомендуется)
RollingSeasonalityRidge(seasonality_window=48)  # 4 года

# С макро-признаками (Ki, Ruonia)
RollingSeasonalityRidge(seasonality_window=36, use_macro=True)

# Без макро-признаков
RollingSeasonalityRidge(seasonality_window=36, use_macro=False)
```

## 📝 Лог изменений

- **2026-02-02** — Создание эксперимента, базовая модель, скрипт бэктеста

## 👤 Автор

Claude Code

---

*Эксперимент изолирован в `experiments/` и не влияет на production код до явного решения о внедрении.*
