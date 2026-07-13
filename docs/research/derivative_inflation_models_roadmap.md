# Roadmap: производные модели инфляции на локальных данных

## Цель

Зафиксировать пошаговый план разработки **новых производных моделей** для СИРЕНА-КБР без изменения существующих исходных моделей.

Правила этой волны работ:

- документировать каждый шаг;
- не править существующие production- и experimental-модели в месте их текущего расположения;
- каждую модифицированную идею оформлять отдельным файлом модели;
- использовать только данные, уже доступные в репозитории;
- отдельно опираться на `infostat.csv` как локальный прокси-файл спроса/услуг.

## Наблюдения, на которых основан план

### Паттерн добавления моделей

На основе `docs/ADDING_MODEL_GUIDE.md` новая модель добавляется как минимум через:

1. отдельный файл в `sirena/models/{name}.py`;
2. импорт и экспорт в `sirena/models/__init__.py`;
3. подключение в `scripts/backtest_framework.py`;
4. прогон бэктестов и обновление артефактов.

### Базовый контракт модели

На основе `sirena/models/base.py` новые total-CPI модели должны соответствовать `BaseForecaster` и реализовывать:

- `fit()`;
- `forecast()`;
- `backtest()`;
- совместимый `predict()` через базовый класс.

### Повторно используемые источники логики

- `sirena/models/ridge_shock_dummies.py` — основной шаблон ridge-модели с feature engineering и backtest-compatible поведением;
- `experiments/rolling_seasonality/models/rolling_seasonality_ridge.py` — донор логики rolling seasonality;
- `sirena/macro_features.py` — источник macro features и production proxy features;
- `sirena/models/subcomponent_multi.py` — источник идей по использованию `Torg`/`pp` и сегментированию признаков;
- `scripts/backtest_framework.py` — подтверждает доступность локальных колонок `usd_nom_i`, `Ki`, `Ruonia`, `Ki_i` в train pipeline.

## Данные, которые считаются допустимыми в этой волне

### Уже доступны локально

- `data/inflation_data.csv`;
- `data/infl_kbr.csv`;
- `data/kbr_weekly_prices_2008_2026.csv`;
- `data/raw/infostat.csv` и дублирующий `data/infostat.csv`.

### Что именно даёт `infostat.csv`

Через `load_production_proxies()` и `add_production_features()` доступны признаки:

- `torg_lag3`;
- `torg_lag6`;
- `torg_diff_lag3`;
- `torg_ma3`;
- `pp_lag3`;
- `pp_lag6`;
- `pp_diff_lag3`.

### Что не закладывается в первую волну

Пока не использовать как обязательную основу:

- внешние survey expectations;
- отдельные кредитные ряды вне уже имеющихся `Ki` / `Ruonia`;
- данные, которых нет в основном дереве `data/`.

## Фаза 0 — документация до кода

### Задача

Сначала формализовать решения и порядок внедрения, не меняя код моделей.

### Артефакт

Этот файл является дорожной картой первой волны.

### Критерий завершения

- есть список новых моделей;
- есть обоснование по данным;
- есть порядок реализации;
- зафиксирован принцип «новый файл вместо правки исходного».

## Фаза 1 — первая волна новых производных моделей

Первая волна специально выбрана как наименее рискованная: все модели строятся вокруг уже работающего ridge-пайплайна.

### 1. RidgeShockRollingForecaster

**Новый файл:** `sirena/models/ridge_shock_rolling.py`

**Registry name:** `ridge_shock_rolling_24m`

**Гипотеза:**
Сочетание shock dummies из `ridge_shock_dummies.py` и rolling seasonality из экспериментальной rolling-модели должно быть устойчивее после структурных сдвигов 2022+.

**Откуда берётся логика:**

- каркас fit/predict/features — из `ridge_shock_dummies.py`;
- расчёт rolling seasonal norm — из `rolling_seasonality_ridge.py`.

**Используемые данные:**

- CPI total;
- food / nonfood / services;
- `Ki`, `Ruonia`;
- уже существующие shock dummies.

**Ожидаемая роль:**

- основной кандидат на улучшение total CPI для `h=1`;
- максимально близкий к текущей production-логике, но более адаптивный.

**Статус реализации на текущую сессию:**

- создан отдельный файл `sirena/models/ridge_shock_rolling.py`;
- модель зарегистрирована как `ridge_shock_rolling_24m`;
- экспортирована через `sirena/models/__init__.py`;
- добавлен backtest hook в `scripts/backtest_framework.py` с именем колонки `Ridge_Shock_Roll24`;
- добавлена в `pages/constants.py` для dashboard-видимости;
- выполнен smoke-check: импорт через `ModelRegistry` успешен, локальный `backtest(..., start_date='2024-01-01')` отработал на 26 точках.

### 2. RidgeProductionProxyForecaster

**Новый файл:** `sirena/models/ridge_production_proxy.py`

**Registry name:** `ridge_production_proxy`

**Гипотеза:**
Добавление локальных demand/services proxy features из `infostat.csv` может усилить краткосрочный сигнал без необходимости во внешних survey-данных.

**Откуда берётся логика:**

- базовый ridge-style fit/predict — из `ridge_shock_dummies.py` или близкого ridge-шаблона;
- production features — через `add_production_features()` из `sirena/macro_features.py`.

**Используемые данные:**

- total CPI и основные компоненты;
- `Torg`, `pp` производные признаки.

**Ожидаемая роль:**

- тест локального спросового сигнала на total CPI;
- отдельный честный сравнительный baseline против моделей без `infostat`.

**Статус реализации на текущую сессию:**

- создан отдельный файл `sirena/models/ridge_production_proxy.py`;
- модель зарегистрирована как `ridge_production_proxy`;
- экспортирована через `sirena/models/__init__.py`;
- добавлен backtest hook в `scripts/backtest_framework.py` с именем колонки `Ridge_ProdProxy`;
- добавлена в `pages/constants.py` для dashboard-видимости;
- локальная `lsp_diagnostics` для `sirena/models/ridge_production_proxy.py` очищена до нуля;
- выполнен smoke-check: прямой импорт модели успешен, пакетный импорт через `sirena.models` успешен, локальный `backtest(..., start_date='2024-01-01')` отработал на 22 точках.

### 3. RidgeAsymmetricERPTProxyForecaster

**Новый файл:** `sirena/models/ridge_asymmetric_erpt_proxy.py`

**Registry name:** `ridge_asymmetric_erpt_proxy`

**Гипотеза:**
Прокси-асимметрия переноса курса может быть полезна даже без импортных цен, если разделить движение курса на положительные и отрицательные шоки и связать их с regime/shock контекстом.

**Откуда берётся логика:**

- ridge-шаблон с устойчивым feature pipeline;
- локальные макроданные из уже загружаемых колонок `usd_nom_i`, `Ki`, `Ruonia`, `Ki_i`.

**Используемые признаки (планируемые):**

- signed USD change proxy;
- separate depreciation / appreciation features;
- взаимодействия с shock/regime features.

**Ожидаемая роль:**

- проверка гипотезы NotebookLM, но строго на текущем локальном наборе данных.

**Статус реализации на текущую сессию:**

- создан отдельный файл `sirena/models/ridge_asymmetric_erpt_proxy.py`;
- модель зарегистрирована как `ridge_asymmetric_erpt_proxy`;
- экспортирована через `sirena/models/__init__.py`;
- добавлен backtest hook в `scripts/backtest_framework.py` с именем колонки `Ridge_AsymERPT`;
- добавлена в `pages/constants.py` для dashboard-видимости;
- локальная `lsp_diagnostics` для `sirena/models/ridge_asymmetric_erpt_proxy.py` очищена до нуля;
- выполнен smoke-check: прямой импорт модели успешен, пакетный импорт через `sirena.models` успешен, локальный `backtest(..., start_date='2024-01-01')` отработал на 26 точках.

## Фаза 2 — более тяжёлая grouped-модель

### 4. TradablesNonTradablesRidgeForecaster

**Новый файл:** `sirena/models/tradables_nontradables_ridge.py`

**Registry name:** `tradables_nontradables_ridge`

**Гипотеза:**
Разделение на условно tradables и non-tradables / services может дать более устойчивую реакцию на локальные demand и FX proxy features, чем одна агрегированная total-модель.

**Источник идей:**

- grouping intuition из `subcomponent_multi.py`;
- локальные агрегаты из уже имеющихся компонентных рядов.

**Ограничение:**

Это должна быть **новая total-модель**, а не переписывание `SubcomponentMultiForecaster`.

## Порядок исполнения

1. оформить roadmap-документ;
2. реализовать `ridge_shock_rolling.py`;
3. реализовать `ridge_production_proxy.py`;
4. реализовать `ridge_asymmetric_erpt_proxy.py`;
5. после этого решать, делать ли grouped-модель `tradables_nontradables_ridge.py`.

## Правила реализации каждой новой модели

Для каждой модели обязательно:

1. создать новый файл в `sirena/models/`;
2. зарегистрировать модель через `@ModelRegistry.register(...)`;
3. не изменять исходный файл базовой модели-донора;
4. задокументировать гипотезу и используемые признаки;
5. подключить модель в `sirena/models/__init__.py`;
6. подключить модель в `scripts/backtest_framework.py`;
7. после верификации только затем решать вопрос о dashboard / forecast visibility.

## Критерии сравнения

Новые модели сравниваются прежде всего против:

- `Ridge`;
- `RidgeShockDummies`;
- экспериментального rolling ridge;
- при необходимости `Huber`;
- при необходимости `SubcomponentMulti` как сильного альтернативного семейства, но не как прямого ridge-бенчмарка.

Главный приоритет:

- `h=1`;
- устойчивость после 2022;
- отсутствие деградации на `h=2` и `h=12` без явного выигрыша на `h=1`.

## План верификации

После каждой новой модели:

1. проверить регистрацию и интеграцию;
2. прогнать `scripts/add_model_checklist.py`;
3. прогнать релевантные бэктесты (`h=1`, `h=2`, `h=12`);
4. при изменении выходов обновить forecast/chart артефакты;
5. зафиксировать результат в документации.

### Зафиксированное расхождение docs vs repo

- `docs/ADDING_MODEL_GUIDE.md` по-прежнему ссылается на `dashboard.py` и `scripts/add_model_checklist.py`.
- В реальном рабочем дереве текущая dashboard-видимость проходит через `pages/constants.py`, а файла `scripts/add_model_checklist.py` нет.
- Поэтому для этой сессии верификация ведётся по фактическим точкам интеграции: импорт/реестр, `pages/constants.py`, `scripts/backtest_framework.py`, локальные smoke-backtest проверки.

## Итоги repo-level верификации первой волны

### Что реально было прогнано

- `python3 scripts/run_backtest_h1.py`
- `python3 scripts/run_backtest_h2.py`
- `python3 scripts/run_backtest_h12.py`
- `python3 scripts/precompute_forecasts.py`
- `python3 scripts/generate_charts.py`
- `python3 scripts/verify_all_tabs.py`
- отдельный прогон `python3 scripts/screenshot_dashboard.py` без и затем с поднятым dashboard на `http://localhost:8503`

### Результаты по новым производным моделям

Ниже зафиксированы **актуальные результаты после review-driven fixes**: новые модели были добавлены в `scripts/precompute_forecasts.py`, а в `scripts/backtest_framework.py` для всего shock-family был выровнен `use_2022_dummy=False`, чтобы сравнение было честным относительно базового `RidgeShockDummies`.

#### RidgeShockRollingForecaster / `Ridge_Shock_Roll24`

- `h=1`: MAE `0.317`, KPI violations `3/12`
- `h=2`: MAE `0.355`, KPI violations `3/12`
- `h=12`: MAE `0.347`, KPI violations `3/12`

Вывод: после выравнивания условий сравнения модель осталась рабочей, но уже не выглядит сильным кандидатом на продвижение. Её роль сейчас — аккуратный сравнительный derivative-вариант, а не лидер первой волны.

#### RidgeProductionProxyForecaster / `Ridge_ProdProxy`

- `h=1`: MAE `0.294`, KPI violations `2/12` — лучший результат в текущем прогоне
- `h=2`: MAE `0.298`, KPI violations `2/12` — лучший результат в текущем прогоне
- `h=12`: MAE `0.306`, KPI violations `4/12` — 2-е место по MAE в текущем прогоне

Вывод: это самый сильный результат первой волны. Гипотеза о полезности локальных `infostat`-прокси подтвердилась как минимум на `h=1` и `h=2`, а на `h=12` модель осталась конкурентной.

#### RidgeAsymmetricERPTProxyForecaster / `Ridge_AsymERPT`

- `h=1`: MAE `0.346`, KPI violations `5/12`
- `h=2`: MAE `0.384`, KPI violations `4/12`
- `h=12`: MAE `0.385`, KPI violations `4/12`

Вывод: локальный proxy-asymmetric ERPT не дал выигрыша в первой волне. Модель не сломана и проходит runtime/backtest-путь, но по качеству пока выглядит скорее как экспериментальная, а не production-кандидат.

### Что подтвердилось по артефактам

- backtest CSV/summary файлы обновлены в `archive/results/`
- chart-артефакты пересобраны в `assets/charts/`
- backtest framework реально отрабатывает с тремя новыми производными моделями
- `data/precomputed_forecasts.json` пересобран и теперь содержит `Ridge_Shock_Roll24`, `Ridge_ProdProxy`, `Ridge_AsymERPT` (по 12 значений на модель)

### Что было исправлено по итогам review-work

- в `scripts/precompute_forecasts.py` добавлены все три модели первой волны, поэтому forecast-facing dashboard surfaces больше не теряют их из-за отсутствия ключей в precompute JSON
- в `scripts/backtest_framework.py` выровнен `use_2022_dummy=False` для `Ridge_Shock`, `Ridge_Shock_Roll24`, `Ridge_ProdProxy`, `Ridge_AsymERPT`, чтобы сравнительные бэктесты были методологически честнее
- concern про future forecast contract для proxy-моделей отдельно проверен runtime-прогнозами на 12 месяцев; сбой не воспроизвёлся, поэтому отдельной правки в модельный код не потребовалось

### Что осталось блокером именно на repo-level verification

- `verify_all_tabs.py` падал не на backtests или charts, а на screenshot-stage
- после review-fix rerun script уже видит `15` моделей в `precomputed_forecasts.json`, включая три новых derivative-модели
- без поднятого dashboard причина screenshot-failure остаётся тривиальной: `ERR_CONNECTION_REFUSED` на `http://localhost:8503`
- с поднятым dashboard screenshot-stage всё равно не стал полностью зелёным из-за уже существующих проблем verification-скриптов:
  - устаревшие ожидаемые названия вкладок
  - конфликт locator для `📊 Бэктест h=1` vs `📊 Бэктест h=12`
  - существующие dashboard-alerts вне области первой волны (`BVAR`, `SubcompMulti`, weekly/nowcast loader issues)

### Практическая интерпретация

- первая волна новых ridge-производных моделей **функционально интегрирована и проверена реальными backtest/forecast/chart-прогонами**
- strongest candidate для дальнейшего продвижения: `RidgeProductionProxyForecaster`
- `RidgeShockRollingForecaster` стоит сохранить как стабильный сравнительный вариант
- `RidgeAsymmetricERPTProxyForecaster` следует оставить документированным как экспериментальную ветку, пока нет более сильного сигнала на локальных данных

## Фаза 1.5 — следующая derivative-ветка над текущим лидером

### RidgeProductionProxyRollingForecaster / `Ridge_ProdProxy_Roll24`

**Новый файл:** `sirena/models/ridge_production_proxy_rolling.py`

**Registry name:** `ridge_production_proxy_rolling_24m`

**Гипотеза:**
Сильнейший текущий short-horizon ridge-кандидат (`Ridge_ProdProxy`) можно усилить, если сохранить его proven `infostat` demand/services proxy block и заменить только global seasonal norm на 24-месячную rolling seasonality из семейства `Rolling_Ridge`.

**Откуда берётся логика:**

- production-proxy block и весь основной feature pipeline — из `sirena/models/ridge_production_proxy.py`;
- rolling seasonal norm — из `experiments/rolling_seasonality/models/rolling_seasonality_ridge.py`;
- исходные working models не правятся in-place, используется отдельный sibling-wrapper.

**Что было сделано:**

- создан новый файл `sirena/models/ridge_production_proxy_rolling.py`;
- модель зарегистрирована как `ridge_production_proxy_rolling_24m`;
- экспортирована через `sirena/models/__init__.py`;
- добавлен backtest-only hook в `scripts/backtest_framework.py` с колонкой `Ridge_ProdProxy_Roll24`;
- добавлен отдельный тест `tests/test_ridge_production_proxy_rolling.py`;
- `pytest tests/test_ridge_production_proxy_rolling.py -v` проходит (`4 passed`);
- `lsp_diagnostics` для новой модели и её теста очищены до нуля.

**Фактические результаты по бэктестам:**

- `h=1`: `Ridge_ProdProxy_Roll24` MAE `0.267`, KPI `1/12`
  - parent `Ridge_ProdProxy`: MAE `0.294`, KPI `2/12`
  - `Rolling_Ridge`: MAE `0.333`, KPI `2/12`
- `h=2`: `Ridge_ProdProxy_Roll24` MAE `0.280`, KPI `0/12`
  - parent `Ridge_ProdProxy`: MAE `0.298`, KPI `2/12`
  - `Rolling_Ridge`: MAE `0.300`, KPI `1/12`
- `h=12`: `Ridge_ProdProxy_Roll24` MAE `0.308`, KPI `2/12`
  - `Rolling_Ridge`: MAE `0.290`, KPI `2/12`
  - parent `Ridge_ProdProxy`: MAE `0.306`, KPI `4/12`

**Вердикт по ветке:**

- **advance** для `h=1` и `h=2`: ветка улучшает и MAE, и KPI относительно родителя, а также обходит `Rolling_Ridge` на коротких горизонтах;
- **qualified result** для `h=12`: ветка не становится победителем по MAE против `Rolling_Ridge`, но улучшает KPI-профиль относительно родителя (`2/12` vs `4/12`);
- на текущем этапе ветка оставлена **backtest-only**, без автоматического продвижения в `pages/constants.py` и `scripts/precompute_forecasts.py`, пока не завершён финальный review и не принято отдельное решение о wider exposure.

**Практическая интерпретация:**

- это первая post-first-wave derivative-ветка, которая не просто повторяет already known winners, а реально усиливает текущего short-horizon лидера;
- сочетание `infostat` proxy block + rolling seasonality выглядит сильнее, чем каждый из этих блоков по отдельности в коротком горизонте;
- при этом долгий горизонт по-прежнему нельзя считать выигранным: `Rolling_Ridge` остаётся основным `h=12` anchor.

## Фаза 1.6 — low-risk ветка вокруг Huber

### HuberProductionProxyForecaster / `Huber_ProdProxy`

**Новый файл:** `sirena/models/huber_production_proxy.py`

**Registry name:** `huber_production_proxy`

**Гипотеза:**
Перенести уже доказанный `infostat` demand/services proxy block в robust-family ветку `Huber`, не меняя исходный `huber.py`, и проверить, даёт ли это локальное усиление сильной production-модели.

**Откуда берётся логика:**

- базовый robust pipeline — из `sirena/models/huber.py`
- production-proxy block — из `sirena/macro_features.py` через `add_production_features()` и `PRODUCTION_FEATURES`
- реализация идёт отдельным sibling-wrapper без правки рабочего исходного файла.

**Что было сделано:**

- создан новый файл `sirena/models/huber_production_proxy.py`
- модель зарегистрирована как `huber_production_proxy`
- экспортирована через `sirena/models/__init__.py`
- добавлен backtest-only hook в `scripts/backtest_framework.py` с колонкой `Huber_ProdProxy`
- добавлен отдельный тест `tests/test_huber_production_proxy.py`
- `pytest tests/test_huber_production_proxy.py -v` проходит (`4 passed`)
- `lsp_diagnostics` для новой модели и её теста очищены до нуля
- ветка пока не продвигалась в `pages/constants.py` и `scripts/precompute_forecasts.py`

**Фактические результаты по бэктестам:**

- `h=1`: `Huber_ProdProxy` MAE `0.319`, KPI `3/12`
  - parent `Huber`: MAE `0.345`, KPI `3/12`
  - `Ridge_ProdProxy_Roll24`: MAE `0.267`, KPI `1/12`
  - `Ridge_ProdProxy`: MAE `0.294`, KPI `2/12`
- `h=2`: `Huber_ProdProxy` MAE `0.331`, KPI `2/12`
  - parent `Huber`: MAE `0.343`, KPI `3/12`
  - `Ridge_ProdProxy_Roll24`: MAE `0.280`, KPI `0/12`
  - `Ridge_ProdProxy`: MAE `0.298`, KPI `2/12`
- `h=12`: `Huber_ProdProxy` MAE `0.348`, KPI `4/12`
  - parent `Huber`: MAE `0.331`, KPI `4/12`
  - `Rolling_Ridge`: MAE `0.290`, KPI `2/12`
  - `Ridge_ProdProxy`: MAE `0.306`, KPI `4/12`

**Вердикт по ветке:**

- локально усиливает родителя `Huber` на `h=1` и `h=2`
- не становится новым short-horizon лидером относительно уже более сильных ridge-family веток
- деградирует на `h=12` относительно родителя и далека от `Rolling_Ridge`
- итоговое решение: **stop / keep backtest-only**, без wider promotion

**Практическая интерпретация:**

- перенос `infostat` proxy block в robust-family технически чистый и полезный как derivative experiment
- но на текущем ландшафте СИРЕНЫ эта ветка не даёт достаточно сильного улучшения, чтобы обгонять лучшие уже найденные short-horizon ridge-кандидаты
- therefore ветка фиксируется как корректный, но не priority-promote результат.

## Фаза 1.7 — low-risk ветка вокруг RidgeExtended

### RidgeExtendedProductionProxyForecaster / `Ridge_Ext_ProdProxy`

**Новый файл:** `sirena/models/ridge_extended_production_proxy.py`

**Registry name:** `ridge_extended_production_proxy`

**Гипотеза:**
Перенести уже доказанный `infostat` demand/services proxy block в более сильную ridge-family ветку `RidgeExtended`, не меняя исходный `ridge_extended.py`, и проверить, сможет ли это усилить уже конкурентную extended-модель.

**Откуда берётся логика:**

- базовый extended ridge pipeline — из `sirena/models/ridge_extended.py`
- production-proxy block — из `sirena/macro_features.py` через `add_production_features()` и `PRODUCTION_FEATURES`
- реализация идёт отдельным sibling-wrapper без правки рабочего исходного файла.

**Что было сделано:**

- создан новый файл `sirena/models/ridge_extended_production_proxy.py`
- модель зарегистрирована как `ridge_extended_production_proxy`
- экспортирована через `sirena/models/__init__.py`
- добавлен backtest-only hook в `scripts/backtest_framework.py` с колонкой `Ridge_Ext_ProdProxy`
- добавлен отдельный тест `tests/test_ridge_extended_production_proxy.py`
- `pytest tests/test_ridge_extended_production_proxy.py -v` проходит (`4 passed`)
- `lsp_diagnostics` для новой модели и её теста очищены до нуля
- ветка пока не продвигалась в `pages/constants.py` и `scripts/precompute_forecasts.py`

**Фактические результаты по бэктестам:**

- `h=1`: `Ridge_Ext_ProdProxy` MAE `0.370`, KPI `4/12`
  - parent `Ridge_Ext`: MAE `0.333`, KPI `3/12`
  - `Ridge_ProdProxy_Roll24`: MAE `0.267`, KPI `1/12`
  - `Ridge_ProdProxy`: MAE `0.294`, KPI `2/12`
- `h=2`: `Ridge_Ext_ProdProxy` MAE `0.344`, KPI `3/12`
  - parent `Ridge_Ext`: MAE `0.334`, KPI `3/12`
  - `Ridge_ProdProxy_Roll24`: MAE `0.280`, KPI `0/12`
  - `Ridge_ProdProxy`: MAE `0.298`, KPI `2/12`
- `h=12`: `Ridge_Ext_ProdProxy` MAE `0.339`, KPI `3/12`
  - `Rolling_Ridge`: MAE `0.290`, KPI `2/12`
  - parent `Ridge_Ext` не входит в верхний слой лидеров, а ветка всё равно не становится новым `h=12` challenger
  - `Ridge_ProdProxy`: MAE `0.306`, KPI `4/12`

**Вердикт по ветке:**

- не проходит локальный parent-gate на `h=1`: хуже `Ridge_Ext` по MAE и KPI
- не проходит локальный parent-gate на `h=2`: хуже `Ridge_Ext` по MAE и не улучшает KPI
- на `h=12` не становится новым лидером и остаётся заметно слабее сильнейших текущих веток
- итоговое решение: **stop / keep backtest-only**, без wider promotion

**Практическая интерпретация:**

- перенос `infostat` proxy block в `RidgeExtended` как отдельный sibling-wrapper технически выполнен корректно и полезен как закрывающий derivative check
- но эта ветка не только не усиливает текущих short-horizon лидеров, а даже не улучшает собственного прямого родителя `Ridge_Ext`
- therefore дальнейшие циклы нужно продолжать не внутри `RidgeExtended + infostat`, а по другим low-risk гипотезам вокруг уже более сильных семейств.

## Фаза 1.8 — low-risk ветка вокруг Huber через rolling seasonality

### HuberRollingForecaster / `Huber_Roll24`

**Новый файл:** `sirena/models/huber_rolling.py`

**Registry name:** `huber_rolling_24m`

**Гипотеза:**
Перенести только 24-месячный rolling seasonal norm из семейства `Rolling_Ridge` в robust-family `Huber`, не меняя `huber.py`, и проверить, даёт ли это более устойчивую short-horizon версию `Huber`.

**Откуда берётся логика:**

- базовый robust pipeline, `HuberRegressor`, `RobustScaler`, macro-support — из `sirena/models/huber.py`
- rolling seasonal norm — из `experiments/rolling_seasonality/models/rolling_seasonality_ridge.py`
- реализация идёт отдельным sibling-wrapper без правки рабочего исходного файла.

**Что было сделано:**

- создан новый файл `sirena/models/huber_rolling.py`
- модель зарегистрирована как `huber_rolling_24m`
- экспортирована через `sirena/models/__init__.py`
- добавлен backtest-only hook в `scripts/backtest_framework.py` с колонкой `Huber_Roll24`
- добавлен отдельный тест `tests/test_huber_rolling.py`
- `pytest tests/test_huber_rolling.py -v` проходит (`4 passed`)
- `lsp_diagnostics` для новой модели и её теста очищены до нуля
- ветка пока не продвигалась в `pages/constants.py` и `scripts/precompute_forecasts.py`

**Фактические результаты по бэктестам:**

- `h=1`: `Huber_Roll24` MAE `0.338`, KPI `2/12`
  - parent `Huber`: MAE `0.345`, KPI `3/12`
  - `Huber_ProdProxy`: MAE `0.319`, KPI `3/12`
  - `Ridge_ProdProxy_Roll24`: MAE `0.267`, KPI `1/12`
- `h=2`: `Huber_Roll24` MAE `0.358`, KPI `3/12`
  - parent `Huber`: MAE `0.343`, KPI `3/12`
  - `Huber_ProdProxy`: MAE `0.331`, KPI `2/12`
  - `Ridge_ProdProxy_Roll24`: MAE `0.280`, KPI `0/12`
- `h=12`: `Huber_Roll24` MAE `0.343`, KPI `2/12`
  - parent `Huber`: MAE `0.331`, KPI `4/12`
  - `Rolling_Ridge`: MAE `0.290`, KPI `2/12`
  - `Ridge_ProdProxy_Roll24`: MAE `0.308`, KPI `2/12`

**Вердикт по ветке:**

- локально усиливает parent `Huber` только на `h=1`
- не проходит parent-gate на `h=2`: хуже `Huber` по MAE при том же KPI
- на `h=12` хуже parent по MAE и неконкурентна относительно лучших ridge-family веток
- итоговое решение: **stop / keep backtest-only**, без wider promotion

**Практическая интерпретация:**

- rolling seasonality сама по себе не даёт достаточного усиления внутри `Huber`-семейства
- перенос этого блока в `Huber` оказался слабее, чем `Huber + production proxies`, и заметно слабее ветки `Ridge_ProdProxy_Roll24`
- therefore дальнейшие low-risk циклы логичнее продолжать в более сильных ridge-family/seam-ветках, а не тратить ещё итерации на `Huber + rolling`.

## Фаза 1.9 — low-risk ветка вокруг RidgeExtended через rolling seasonality

### RidgeExtendedRollingForecaster / `Ridge_Ext_Roll24`

**Новый файл:** `sirena/models/ridge_extended_rolling.py`

**Registry name:** `ridge_extended_rolling_24m`

**Гипотеза:**
Перенести только 24-месячный rolling seasonal norm из семейства `Rolling_Ridge` в более сильную ridge-family ветку `RidgeExtended`, не меняя `ridge_extended.py`, и проверить, даст ли это устойчивое улучшение без нового data seam.

**Откуда берётся логика:**

- базовый extended ridge pipeline, macro-support и ETS blending — из `sirena/models/ridge_extended.py`
- rolling seasonal norm — из `experiments/rolling_seasonality/models/rolling_seasonality_ridge.py`
- реализация идёт отдельным sibling-wrapper без правки рабочего исходного файла.

**Что было сделано:**

- создан новый файл `sirena/models/ridge_extended_rolling.py`
- модель зарегистрирована как `ridge_extended_rolling_24m`
- экспортирована через `sirena/models/__init__.py`
- добавлен backtest-only hook в `scripts/backtest_framework.py` с колонкой `Ridge_Ext_Roll24`
- добавлен отдельный тест `tests/test_ridge_extended_rolling.py`
- `pytest tests/test_ridge_extended_rolling.py -v` проходит (`4 passed`)
- `lsp_diagnostics` для новой модели и её теста очищены до нуля
- ветка пока не продвигалась в `pages/constants.py` и `scripts/precompute_forecasts.py`

**Фактические результаты по бэктестам:**

- `h=1`: `Ridge_Ext_Roll24` MAE `0.318`, KPI `1/12`
  - parent `Ridge_Ext`: MAE `0.333`, KPI `3/12`
  - `Ridge_ProdProxy_Roll24`: MAE `0.267`, KPI `1/12`
  - `Ridge_ProdProxy`: MAE `0.294`, KPI `2/12`
- `h=2`: `Ridge_Ext_Roll24` MAE `0.291`, KPI `2/12`
  - parent `Ridge_Ext`: MAE `0.334`, KPI `3/12`
  - `Ridge_ProdProxy_Roll24`: MAE `0.280`, KPI `0/12`
  - `Ridge_ProdProxy`: MAE `0.298`, KPI `2/12`
- `h=12`: `Ridge_Ext_Roll24` MAE `0.281`, KPI `2/12`
  - `Rolling_Ridge`: MAE `0.290`, KPI `2/12`
  - `Ridge_ProdProxy`: MAE `0.306`, KPI `4/12`
  - `Ridge_ProdProxy_Roll24`: MAE `0.308`, KPI `2/12`

**Вердикт по ветке:**

- проходит локальный parent-gate на всех трёх горизонтах: лучше `Ridge_Ext` по MAE и не хуже по KPI
- на `h=1` и `h=2` не становится новым абсолютным лидером, но входит в верхний конкурентный слой
- на `h=12` становится новым лучшим результатом в текущем прогоне (`0.281` vs `0.290` у `Rolling_Ridge`) при том же KPI (`2/12`)
- итоговое решение на текущем этапе: **advance candidate**, ветка пока остаётся backtest-only до отдельного branch-review и решения о wider promotion

**Практическая интерпретация:**

- это первая derivative-ветка в текущей sirena-side серии, которая не просто проходит локальный parent-gate, но и выглядит как реальный новый `h=12` лидер
- rolling seasonal norm оказывается особенно сильным внутри `RidgeExtended`, а не только как отдельная experimental-линия `Rolling_Ridge`
- при этом short-horizon crown всё ещё у `Ridge_ProdProxy_Roll24`, поэтому возможное дальнейшее продвижение этой ветки должно аккуратно разделять: `h=12` strength vs `h=1/h=2` top layer, а не объявлять её универсальным победителем.

## Правило принятия решений

Если новая производная модель:

- не улучшает `h=1`,
- или даёт нестабильный результат,
- или ухудшает общую картину без понятной компенсации,

то она остаётся задокументированной как экспериментальная и не продвигается дальше по стеку.

## Текущий статус

- Фаза 0: завершена и документирована этим файлом.
- Первая волна моделей реализована, интегрирована в registry/backtest/precompute/chart surfaces и перепроверена после review-fixes.
- Сильнейший кандидат первой волны: `RidgeProductionProxyForecaster`.
- Следующая derivative-ветка `RidgeProductionProxyRollingForecaster` уже реализована и прошла backtest-gate как сильный short-horizon candidate.
- Следующая low-risk ветка `HuberProductionProxyForecaster` реализована и проверена, но зафиксирована как backtest-only `stop`, а не как новый лидер.
- Следующая low-risk ветка `RidgeExtendedProductionProxyForecaster` реализована и проверена, но также зафиксирована как backtest-only `stop`, потому что не проходит даже локальный parent-gate.
- Следующая low-risk ветка `HuberRollingForecaster` реализована и проверена, но тоже зафиксирована как backtest-only `stop`: локально улучшает `Huber` только на `h=1`, но не проходит parent-gate на `h=2` и не конкурентна по `h=12`.
- Следующая low-risk ветка `RidgeExtendedRollingForecaster` реализована и проверена; это первый текущий `advance candidate`, который проходит parent-gate на всех горизонтах и выглядит как новый `h=12` лидер, но решение о wider promotion ещё не принято.
- `RidgeShockRollingForecaster` и `RidgeAsymmetricERPTProxyForecaster` пока остаются документированными сравнительно-экспериментальными ветками.
- Оставшийся repo-level verification blocker относится не к новым моделям, а к legacy screenshot/dashboard verifier path.
