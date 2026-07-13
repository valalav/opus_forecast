# Roadmap: оценка внешнего каталога моделей для СИРЕНА-КБР

Дата первичной версии: 2026-06-25

Источник: список моделей из прикрепленного файла
`/home/valalav/.codex/attachments/90c6ffbf-45c1-42b5-8107-e67f67e3909f/pasted-text.txt`.

Коды из внешнего репозитория распакованы в
`experiments/code_repository_20260625/`. Рабочая карта:
`experiments/code_repository_20260625/README.md`; короткий перечень файлов для
изучения: `experiments/code_repository_20260625/STUDY_FILES.md`; полный реестр:
`experiments/code_repository_20260625/inventory/file_inventory.csv`.

Цель: не копировать внешний каталог механически, а выбрать идеи, которые могут
улучшить прогноз КБР с учетом уже существующих моделей СИРЕНА.

## Короткий Вывод

Самые полезные направления для нас:

1. **Недельная -> месячная инфляция**: усилить nowcast, especially June-July
   2026, бензин, недельные товары, неполный месяц.
2. **Дезагрегированный прогноз компонентов и micro**: довести до production
   сценарные override-правила для тарифов, топлива, плодоовощей.
3. **Отбор переменных pseudo-OOS**: использовать как gate перед добавлением
   макро-признаков в Ridge/Huber/VAR/Factor.
4. **ARIMAX/ARDL/SARIMAX с поиском спецификации**: брать не как замену
   ансамблю, а как прозрачные challenger-модели и диагностические бенчмарки.
5. **PCA/FAMIDAS/MIDAS**: развивать только через уже существующие
   `FactorPolicy`, `FactorBridge`, `MIDAS`, но с жесткими leakage checks.
6. **ML-модели**: полезны только после правильного feature store и rolling
   validation; готовые ML-шаблоны без признаков не являются evidence.

Направления низкой пользы сейчас:

- банковские ставки, ипотека, морские грузоперевозки;
- DSGE/KANK, КПМ и тяжелые квартальные макромодели;
- PVAR по регионам без полноценной панельной задачи;
- HP-фильтр/output gap как самостоятельный драйвер месячного ИПЦ.

## Приоритеты

| Приоритет | Направление | Польза для КБР | Решение |
|---|---|---:|---|
| P0 | Weekly-to-monthly nowcast | Очень высокая | Делать |
| P0 | Micro/subcomponent scenario overrides | Очень высокая | Делать |
| P0 | Pseudo-OOS variable selection | Высокая | Делать |
| P1 | ARIMAX/ARDL/SARIMAX search | Средне-высокая | Делать как challenger |
| P1 | STL/seasonal component decomposition | Средняя | Использовать для diagnostics |
| P1 | PCA / clustering of components | Средняя | Интегрировать в factor/micro diagnostics |
| P2 | ML RF/XGB/SVR/NN | Средняя | Только после feature store |
| P2 | FAMIDAS/MIDAS | Средняя | Исследовать через existing MIDAS |
| P3 | BVAR / VAR family variants | Ограниченная | Только через текущий VARPolicy research gate |
| P3 | Kalman / output gap / potential output | Низкая для месячного ИПЦ | Не в первой волне |

## P0. Недельная -> Месячная Инфляция

Внешние аналоги:

- `Прогноз ежемесячной инфляции на основе еженедельной инфляции`
- `Модель "Недельная - месячная инфляция"`
- `Загрузка недельных данных ИПЦ`

Что у нас уже есть:

- `data/Сравнение еженедельных цен_01.csv`
- `data/kbr_weekly_prices_2008_2026.csv`
- `scripts/weekly_bridge_nowcast.py`
- `scripts/nowcast_update.py`
- `sirena/data/weekly_loader.py`
- `Nowcast (Weekly Bridge)` в `scripts/precompute_forecasts.py`

Плюсы:

- самый актуальный сигнал для текущего месяца;
- уже доказал важность по июню 2026;
- позволяет отдельно видеть бензин, плодоовощи, услуги.

Минусы / риски:

- недельные данные не являются официальным месячным фактом;
- дубли и разные уровни агрегирования требуют аккуратной очистки;
- простое суммирование недель может переоценивать итог месяца.

Roadmap:

1. Сделать отдельный `WeeklyMonthlyBridge` с режимами:
   - chain weekly;
   - last-week-to-last-week;
   - historical weekly-to-monthly calibration.
2. Добавить item-level contribution table для бензина, плодоовощей, тарифных
   услуг.
3. Сохранять сценарные заметки в `archive/results/analysis_notes/`.
4. Оценивать h=0/h=1 nowcast backtest отдельно от месячных моделей.

Критерий продвижения:

- rolling nowcast MAE ниже простого monthly model proxy;
- стабильная работа на неполном месяце;
- объяснимые contributions по товарам.

## P0. Micro И Дезагрегированный Прогноз

Внешние аналоги:

- `Дезагрегированный подход к прогнозированию инфляции (Python/Eviews)`
- `Прогноз инфляции на основе прогноза ее субкомпонентов`
- `Прогноз инфляции в регионе на основе прогноза по 45 компонентам`
- `Макроэкономический прогноз инфляции из микроэкономических факторов`

Что у нас уже есть:

- `Micro_SM` через `sirena/models/micro_statsmodels_external.py`
- `data/external/micro_cpi_region_export/`
- `SubcomponentMulti`
- `Microcomponent`, `MicroOptimized`, `HierarchicalMicro`
- свежий импорт `data/ИПЦПолныйРегион.xlsx` до мая 2026

Плюсы:

- лучший путь для тарифов, бензина, плодоовощей и нестандартных шоков;
- можно делать сценарии на item-level, а не ломать headline вручную;
- прозрачно объясняется руководству.

Минусы / риски:

- нужны чистые веса и контроль агрегатов/дочерних позиций;
- если смешать агрегаты и leaf-items, можно удвоить вклад;
- ARIMA/SARIMA по сотням рядов без fallback может быть нестабильна.

Roadmap:

1. Зафиксировать справочник leaf/aggregate для ЖКУ, топлива, плодоовощей.
2. Сделать сценарный слой поверх `Micro_SM`:
   - тарифы July=100 / October=110;
   - бензин reversion;
   - плодоовощные rebound/deflation paths.
3. Сохранять `micro_scenario_summary.csv` и `micro_scenario_details.csv`.
4. Сравнить:
   - baseline `Micro_SM`;
   - leaf override;
   - aggregate override;
   - official component aggregation.

Критерий продвижения:

- item-level сценарий воспроизводит headline без двойного счета;
- backtest на исторических tariff/fuel episodes лучше или объяснимее baseline.

## P0. Отбор Переменных Pseudo-OOS

Внешние аналоги:

- `Методика отбора переменных, обеспечивающих надежные прогнозы`
- `Поиск лучшей спецификации модели для прогноза`
- `Моделирование и прогнозирование регионального ИПЦ: ARDL`

Что у нас уже есть:

- пилот `archive/results/variable_selection_pilot_20260625/`
- `sirena/macro_features.py`
- `RidgeMacro`, `FactorPolicy`, `VARPolicy`

Плюсы:

- защищает от добавления слабых макро-признаков;
- дает понятный ranking по h=1/h=2/h=12;
- хорошо подходит для выбора блоков: monetary, demand, components, weekly.

Минусы / риски:

- одиночные признаки могут не пройти, хотя полезны в блоке;
- нужно строго соблюдать cutoff;
- нельзя подбирать переменные на том же окне, где заявляем качество.

Roadmap:

1. Перенести pilot в `experiments/variable_selection/`.
2. Добавить блоковый отбор:
   - monetary: USD, Ki, Ruonia, spread;
   - demand: deposits, retail/all_real, production proxies;
   - components: food/nonfood/services;
   - weekly/fuel: calibrated weekly signals.
3. Оценивать h=1, h=2, h=12 отдельно.
4. Выдавать `selected_features_by_horizon.csv`.

Критерий продвижения:

- признаки или блоки проходят RRMSE/outperform gate;
- selected features улучшают Huber/Ridge challenger в настоящем rolling
  backtest.

## P1. ARIMAX / ARDL / SARIMAX Search

Внешние аналоги:

- `Прогноз ИПЦ на основе ARIMAX модели`
- `Построение ARIMAX моделей для прогнозирования ИПЦ`
- `Моделирование и прогнозирование регионального ИПЦ: ARDL`
- `Прогнозирование региональной инфляции: SARIMAX, RandomForest`

Что у нас уже есть:

- `sarima`, `ar1`, `ets`, `holt_winters`, `naive_seasonal`
- `Ridge`/`Huber` family как более устойчивые regularized alternatives
- `RidgeMacro`, `Ridge_AsymERPT`, production proxy variants

Плюсы:

- прозрачные модели;
- понятный лаговый механизм;
- хороший challenger для макро/курсовых эффектов.

Минусы / риски:

- на короткой региональной выборке легко переобучаются;
- SARIMAX с future exog path может протекать, если брать фактический будущий
  USD/Ki/Ruonia;
- ARDL требует аккуратной стационарности и lag selection.

Roadmap:

1. Делать не production replacement, а `ARIMAXPolicy` challenger.
2. Exog future paths только deterministic:
   - last observed;
   - AR forecast;
   - documented scenario.
3. Сравнить против Huber/RidgeShockDummies/SubcomponentMulti.

Критерий продвижения:

- h=1/h=2 улучшаются без ухудшения h=12 trajectory realism;
- residual diagnostics не проваливаются критично.

## P1. STL / Сезонность / SA Компонентов

Внешние аналоги:

- `Прогнозирование ИПЦ_STL метод`
- `Сезонная корректировка компонент ИПЦ`
- `Автоматизация сезонного сглаживания нескольких рядов`
- `Подготовка рядов данных ... стационарность и сезонность`

Что у нас уже есть:

- `data/mom_sa_kbr.csv`
- `rolling_seasonality` experiments
- `Ridge_Shock_Roll24`
- seasonal diagnostics in factor/VAR research

Плюсы:

- важно после 2022 и сдвига тарифов;
- полезно для компонентных и micro-моделей;
- помогает объяснять, где модель путает сезонность и шок.

Минусы / риски:

- revised SA history может давать нереалистичную real-time оценку;
- full-sample seasonality нельзя использовать как доказательство rolling
  качества.

Roadmap:

1. Использовать rolling/train-only seasonal norms.
2. Отдельно проверить июль/октябрь 2026 после тарифного сдвига.
3. Добавить seasonality diagnostics к micro/subcomponent scenarios.

## P1. PCA / Кластеризация Компонентов

Внешние аналоги:

- `Применение метода главных компонент и кластеризации данных`
- `FAMIDAS`
- `FAR модель`

Что у нас уже есть:

- `FactorPolicy`
- `FactorBridge`
- `StationaryBlockFAVAR`
- `factor_model_research`

Плюсы:

- помогает с большим числом micro/subcomponent series;
- может создавать компактные факторы вместо десятков слабых признаков;
- полезно для diagnostics and reporting.

Минусы / риски:

- PCA часто сложно интерпретировать;
- full-sample PCA leakage недопустим;
- факторная модель может ухудшать h=12 trajectory realism.

Roadmap:

1. Делать rolling PCA only.
2. Кластеры использовать сначала для micro diagnostics:
   - fuel;
   - ЖКУ;
   - плодоовощи;
   - tradables/non-tradables.
3. В production не продвигать без factor loading stability.

## P2. ML: RF / XGB / SVR / NN

Внешние аналоги:

- `Прогнозирование инфляции на основе ML`
- `Прогноз ИПЦ на основе моделей машинного обучения`
- `Модель нейронной сети для прогнозирования инфляции`
- `SARIMAX, RandomForest`

Что у нас уже есть:

- `xgboost_model.py`
- `lightgbm.py`
- `catboost_model.py`
- `tft.py`
- `stacking_regressor.py`
- `ngboost`, `ebm`

Плюсы:

- может ловить нелинейности;
- полезно для item-level и large feature store;
- EBM/NGBoost уже подтверждают пользу некоторых ML-подходов.

Минусы / риски:

- месячная выборка КБР мала;
- нейросети почти наверняка переобучатся;
- ML без продуманного feature store слабее robust linear models;
- tuning must be nested/cutoff-safe.

Roadmap:

1. Не начинать с новой NN.
2. Сначала сделать feature store и variable-selection gate.
3. Потом протестировать:
   - XGB/LightGBM with monotone/regularized settings;
   - EBM for interpretability;
   - NGBoost for uncertainty.
4. Stacking only with out-of-fold predictions.

## P2. MIDAS / FAMIDAS

Внешние аналоги:

- `Модель прогнозирования экономической активности на основе FAMIDAS`
- mixed-frequency weekly/monthly/quarterly models

Что у нас уже есть:

- `midas.py`
- weekly data;
- macro monthly data.

Плюсы:

- естественный мост недельных цен в месячный CPI;
- может лучше использовать неполный месяц.

Минусы / риски:

- сложно валидировать;
- высокая опасность leakage при агрегации недель;
- мало наблюдений для стабильной нелинейной MIDAS.

Roadmap:

1. Начать не с full MIDAS, а с calibrated weekly bridge.
2. Если bridge стабилен, сделать `MIDASWeeklyFuelFood` challenger.
3. Отдельно оценить h=0 nowcast и h=1 forecast.

## P3. VAR / BVAR / PVAR / КПМ / DSGE

Внешние аналоги:

- regional VAR models;
- BVAR with disaggregated production/inflation;
- PVAR;
- КПМ/DSGE/KANK.

Что у нас уже есть:

- `VARPolicy`
- `BVAR`, `BVARRate`
- `VAR_MODEL_RESEARCH.md`
- `FactorPolicy`

Плюсы:

- полезно как policy/story model;
- VARPolicy уже встроен как обязательный контрольный элемент.

Минусы / риски:

- старый BVAR уже показал плохую h=12 траекторию;
- PVAR/DSGE тяжелые и не решают текущий июль-август;
- макро VARX может улучшать h=1 и портить h=12.

Roadmap:

1. Не добавлять новый VAR без чтения `docs/VAR_MODEL_RESEARCH.md`.
2. Любая VAR/BVAR идея проходит:
   - stationarity;
   - shock handling;
   - h=1/h=2/h=12;
   - trajectory realism.
3. PVAR/DSGE пока не брать.

## Что Делать Сразу

### Шаг 1

Оформить полноценный `experiments/variable_selection/` из текущего pilot.

Артефакты:

- `selected_features_by_horizon.csv`
- `rolling_predictions.csv`
- `variable_selection_report.md`

### Шаг 2

Доработать weekly nowcast:

- бензин contribution;
- last-week-to-last-week bridge;
- item-level drivers;
- сценарии для июля/августа.

### Шаг 3

Доработать `Micro_SM` scenario layer:

- регулируемые тарифы;
- бензин reversion;
- плодоовощи;
- aggregate-vs-leaf guard.

### Шаг 4

ARIMAX/ARDL challenger только после variable-selection gate.

## Что Не Делать Сейчас

- Не переносить готовый Волгоградский ML ensemble как модель.
- Не добавлять новую нейросеть без feature store и nested rolling validation.
- Не строить новый BVAR/PVAR без строгого VAR workflow.
- Не использовать future actual exogenous paths.
- Не объявлять улучшение модели только по одному окну или in-sample fit.

## Критерии Принятия Roadmap-Идей

Идея допускается к реализации, если:

1. есть источник данных и понятный cutoff;
2. есть baseline для сравнения;
3. есть rolling h=1/h=2/h=12 или явное ограничение на nowcast-only;
4. есть контроль leakage;
5. есть объяснимость вклада в текущие контрольные точки.

Идея допускается в production только если:

1. проходит backtest;
2. не ломает h=12 trajectory realism;
3. обновлены `precomputed_forecasts.json` и графики, если прогноз меняется;
4. есть verification note или report.
