# Документация СИРЕНА-КБР v5.3

Добро пожаловать в документацию системы прогнозирования инфляции.

## 📚 Основная документация

| Документ                                                                                                                                                   | Описание                                                                                                               |
| ---------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------- |
| **[MODEL_CATALOG.md](MODEL_CATALOG.md)**                                                                                                                   | **Каталог моделей** — все 40+ моделей с весами, MAE, примерами кода.                                                   |
| **[ADDING_MODEL_GUIDE.md](ADDING_MODEL_GUIDE.md)**                                                                                                         | **Добавление модели** — 11-точечный чеклист.                                                                           |
| **[VERIFICATION_GUIDE.md](VERIFICATION_GUIDE.md)**                                                                                                         | **Верификация** — скрипты проверки, чеклисты.                                                                          |
| **[BACKTEST_METHODOLOGY.md](BACKTEST_METHODOLOGY.md)**                                                                                                     | **Методика бэктеста** — h=1, h=2, h=3, h=6, h=12, метрики.                                                             |
| **[NOWCASTING.md](NOWCASTING.md)**                                                                                                                         | **Nowcasting** — недельные данные, формула расчёта.                                                                    |
| **[OPR_FORECAST_LINKAGE.md](OPR_FORECAST_LINKAGE.md)**                                                                                                     | **Связь с ОПР** — отправочная форма, PR3/докладный контекст, май-июнь 2026 и правила ручной policy-траектории.         |
| **[FORECAST_FACT_ANALYSIS.md](FORECAST_FACT_ANALYSIS.md)**                                                                                                 | **План-факт анализ** — отчёты об отклонении прогноза от факта, компонентные вклады, исторические outliers, DOCX/XLSX.  |
| **[FOOD_TARIFF_FORECAST_2026_2027.md](FOOD_TARIFF_FORECAST_2026_2027.md)**                                                                                 | **Прогноз 2026–2027** — food shock, плодоовощи, SA-субкомпоненты, перенос индексации услуг, полный пакет CSV/DOCX/ZIP. |
| **[ANALYSIS_NOTES.md](ANALYSIS_NOTES.md)**                                                                                                                 | **Реестр сценарных заметок** — где искать сохранённые краткие анализы по бензину, тарифам, micro-сценариям и контрольным точкам. |
| **[TASK_LOG.md](TASK_LOG.md)**                                                                                                                             | **Журнал завершённых задач** — дата, задача, краткий итог и ссылка на основной артефакт.                                      |
| **[EXTERNAL_MODEL_ROADMAP.md](EXTERNAL_MODEL_ROADMAP.md)**                                                                                                 | **Roadmap внешних модельных идей** — что из каталога кодов полезно для СИРЕНА-КБР, плюсы/минусы и приоритеты внедрения. |
| **[EXTERNAL_CODE_INTEGRATION_PLAN.md](EXTERNAL_CODE_INTEGRATION_PLAN.md)**                                                                                 | **План интеграции внешнего репозитория кодов** — единый маршрут от внешней модели к prototype/diagnostic/challenger/production. |
| **[VAR_MODEL_RESEARCH.md](VAR_MODEL_RESEARCH.md)**                                                                                                         | **VAR-family research** — обязательная VAR-линия: RegimeMacroVARX для h=1, SeasonalVAR для h=12, robust/shock handling. |
| **[FACTOR_MODEL_RESEARCH.md](FACTOR_MODEL_RESEARCH.md)**                                                                                                   | **Factor-family research** — обязательная факторная модель: Robust seasonal FAVAR, PCA-факторы, rolling h=1/h=2/h=12.   |
| **[FACTOR_MODEL_PRESENTATION.md](FACTOR_MODEL_PRESENTATION.md)**                                                                                           | **Factor model presentation** — готовое описание стационарной блочной FAVAR для доклада коллегам.                       |
| **[FACTOR_MODEL_AGENT_PRD.md](FACTOR_MODEL_AGENT_PRD.md)**                                                                                                 | **Factor model PRD** — агентная дорожная карта, роли MiniMax/Qwen/Gemini/Nemotron и acceptance gates.                   |
| **[../archive/results/full_forecast_package_2026_2027/verification_report.md](../archive/results/full_forecast_package_2026_2027/verification_report.md)** | **Контрольная проверка отправочного пакета** — сверка план-факт расчётов, товарных драйверов и ZIP-состава.            |
| **[API.md](API.md)**                                                                                                                                       | REST API v5.0 — эндпоинты, схемы.                                                                                      |
| **[DASHBOARD.md](DASHBOARD.md)**                                                                                                                           | **Дашборд** — вкладки, порт 8503.                                                                                      |
| **[FORMATS.md](FORMATS.md)**                                                                                                                               | Форматы данных (`infl_kbr.csv`, шоки, веса).                                                                           |
| **[../README.md](../README.md)**                                                                                                                           | **Полная документация проекта** — быстрый старт, архитектура.                                                          |

## 🔧 Для разработчиков

| Документ                                     | Описание                          |
| -------------------------------------------- | --------------------------------- |
| **[DEVELOPMENT.md](DEVELOPMENT.md)**         | Установка, тесты, структура кода. |
| **[DEVELOPER_GUIDE.md](DEVELOPER_GUIDE.md)** | Расширенный гайд разработчика.    |
| **[USER_GUIDE.md](USER_GUIDE.md)**           | Руководство пользователя.         |

## 🔬 Исследования

| Документ                                                                    | Описание                                                                       |
| --------------------------------------------------------------------------- | ------------------------------------------------------------------------------ |
| **[FREEZE_ANALYSIS.md](FREEZE_ANALYSIS.md)**                                | **Анализ заморозки цен** — PSI, пружинный эффект, сезонность, FDI для Nowcast. |
| [RESEARCH_PLODOVOSHCHI_DIVERGENCE.md](RESEARCH_PLODOVOSHCHI_DIVERGENCE.md)  | Расхождение плодоовощей (цепные индексы).                                      |
| [SEASONALITY_ANALYSIS_JAN_FEB.md](SEASONALITY_ANALYSIS_JAN_FEB.md)          | Январский эффект, волатильность.                                               |
| [SUBCOMP_MULTI_ANALYSIS.md](SUBCOMP_MULTI_ANALYSIS.md)                      | Лучшая модель SubcomponentMulti (SIRENA Score 0.515).                          |
| [BVAR_ANALYSIS.md](BVAR_ANALYSIS.md)                                        | Анализ BVAR (удалена из ансамбля).                                             |
| [VAR_MODEL_RESEARCH.md](VAR_MODEL_RESEARCH.md)                              | Итоги исследования обязательной VAR-family модели и финальный rolling backtest. |
| [FACTOR_MODEL_RESEARCH.md](FACTOR_MODEL_RESEARCH.md)                        | Итоги исследования обязательной факторной модели и финальный rolling backtest.  |
| [WEEKLY_RESEARCH.md](WEEKLY_RESEARCH.md)                                    | Недельные исследования.                                                        |
| [NEW_FINDINGS_2025.md](NEW_FINDINGS_2025.md)                                | Новые находки 2025 года.                                                       |
| [ANOMALY_ANALYSIS_2025.md](ANOMALY_ANALYSIS_2025.md)                        | Анализ аномалий 2025 года.                                                     |
| **[commodity_risk_ledger_2026.md](research/commodity_risk_ledger_2026.md)** | **Реестр Рисков Q1 2026** — План-факт верификация Deep Research.               |

## 🤖 Ralph Edge Lab

| Документ                                                             | Описание                                              |
| -------------------------------------------------------------------- | ----------------------------------------------------- |
| **[EDGE_LAB_REFERENCE.md](EDGE_LAB_REFERENCE.md)**                   | **Справочник Edge Lab** — архитектура, CLI, принципы. |
| [../edge_lab/docs/ARCHITECTURE.md](../edge_lab/docs/ARCHITECTURE.md) | Полная архитектура системы.                           |
| [../edge_lab/AGENTS.md](../edge_lab/AGENTS.md)                       | Конституция агента.                                   |

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
- `archive/results/april_2026_deviation_analysis/build_april_deviation_report.py` — пример сборки DOCX/XLSX план-факт анализа
- `archive/results/analysis_notes/analysis_index.csv` — реестр сохранённых кратких сценарных анализов и ссылок на них
- `archive/results/full_forecast_package_2026_2027/build_package.py` — сборка отправочного CSV/DOCX/ZIP пакета прогноза 2026–2027
- `archive/results/full_forecast_package_2026_2027/verification_scripts/` — контрольные скрипты проверки месячных расчётов, товарных драйверов и ZIP-состава

---

_Версия документации: v5.3 (2 февраля 2026)_
