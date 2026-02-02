# Документация СИРЕНА-КБР v5.3

Добро пожаловать в документацию системы прогнозирования инфляции.

## 📚 Основная документация

| Документ | Описание |
|----------|----------|
| **[MODEL_CATALOG.md](MODEL_CATALOG.md)** | **Каталог моделей** — все 40+ моделей с весами, MAE, примерами кода. |
| **[ADDING_MODEL_GUIDE.md](ADDING_MODEL_GUIDE.md)** | **Добавление модели** — 11-точечный чеклист. |
| **[VERIFICATION_GUIDE.md](VERIFICATION_GUIDE.md)** | **Верификация** — скрипты проверки, чеклисты. |
| **[BACKTEST_METHODOLOGY.md](BACKTEST_METHODOLOGY.md)** | **Методика бэктеста** — h=1, h=2, h=3, h=6, h=12, метрики. |
| **[NOWCASTING.md](NOWCASTING.md)** | **Nowcasting** — недельные данные, формула расчёта. |
| **[API.md](API.md)** | REST API v5.0 — эндпоинты, схемы. |
| **[DASHBOARD.md](DASHBOARD.md)** | **Дашборд** — вкладки, порт 8503. |
| **[FORMATS.md](FORMATS.md)** | Форматы данных (`infl_kbr.csv`, шоки, веса). |
| **[../README.md](../README.md)** | **Полная документация проекта** — быстрый старт, архитектура. |

## 🔧 Для разработчиков

| Документ | Описание |
|----------|----------|
| **[DEVELOPMENT.md](DEVELOPMENT.md)** | Установка, тесты, структура кода. |
| **[DEVELOPER_GUIDE.md](DEVELOPER_GUIDE.md)** | Расширенный гайд разработчика. |
| **[USER_GUIDE.md](USER_GUIDE.md)** | Руководство пользователя. |

## 🔬 Исследования

| Документ | Описание |
|----------|----------|
| [RESEARCH_PLODOVOSHCHI_DIVERGENCE.md](RESEARCH_PLODOVOSHCHI_DIVERGENCE.md) | Расхождение плодоовощей (цепные индексы). |
| [SEASONALITY_ANALYSIS_JAN_FEB.md](SEASONALITY_ANALYSIS_JAN_FEB.md) | Январский эффект, волатильность. |
| [SUBCOMP_MULTI_ANALYSIS.md](SUBCOMP_MULTI_ANALYSIS.md) | Лучшая модель SubcomponentMulti (SIRENA Score 0.515). |
| [BVAR_ANALYSIS.md](BVAR_ANALYSIS.md) | Анализ BVAR (удалена из ансамбля). |
| [WEEKLY_RESEARCH.md](WEEKLY_RESEARCH.md) | Недельные исследования. |
| [NEW_FINDINGS_2025.md](NEW_FINDINGS_2025.md) | Новые находки 2025 года. |
| [ANOMALY_ANALYSIS_2025.md](ANOMALY_ANALYSIS_2025.md) | Анализ аномалий 2025 года. |

## 🤖 Ralph Edge Lab

| Документ | Описание |
|----------|----------|
| **[EDGE_LAB_REFERENCE.md](EDGE_LAB_REFERENCE.md)** | **Справочник Edge Lab** — архитектура, CLI, принципы. |
| [../edge_lab/docs/ARCHITECTURE.md](../edge_lab/docs/ARCHITECTURE.md) | Полная архитектура системы. |
| [../edge_lab/AGENTS.md](../edge_lab/AGENTS.md) | Конституция агента. |

## 🗄️ Архив

Устаревшие документы в `archive/docs/`.

---

## Быстрые ссылки

### Production Models (9)
- [Huber](../sirena/models/huber.py) — лучшая на h=1 (MAE 0.289)
- [RidgeShockDummies](../sirena/models/ridge_shock_dummies.py) — MAE 0.299
- [ElasticNet](../sirena/models/elasticnet.py) — MAE 0.301
- [NGBoostShock](../sirena/models/ngboost_shock.py) — лучшая на h=2 (MAE 0.291)
- [Ridge](../sirena/models/ridge.py) — baseline (MAE 0.310)
- [Prophet](../sirena/models/prophet.py) — лучшая на h=12 (MAE 0.277)

### Best Experimental Model
- [SubcomponentMulti](../sirena/models/subcomponent_multi.py) — лучшая по SIRENA Score (0.515)

### Key Scripts
- `scripts/precompute_forecasts.py` — пересчёт прогнозов
- `scripts/run_backtest_h1.py` — бэктест h=1
- `scripts/sirena_score.py` — расчёт SIRENA Score
- `scripts/verify_all_tabs.py` — верификация дашборда

---

*Версия документации: v5.3 (2 февраля 2026)*
