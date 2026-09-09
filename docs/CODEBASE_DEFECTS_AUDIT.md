# Аудит дефектов кодовой базы `sirena-kbr`

Дата: `2026-09-09`

## Критерии аудита

Проверены 4 направления:

1. Связки импортов/регистраций в `sirena/models/` и `sirena/models/__init__.py`.
2. Архитектурная консистентность `scripts/run_backtest_*.py` относительно `ModelRegistry`.
3. Сопоставимость документации (`GEMINI.md`, `docs/MODEL_CATALOG.md`) и фактической модели/реестра.
4. Состояние импортов/путей в `scripts/` (битые импорты, несовместимости).

## Перечень найденных дефектов

### 1) `scripts` используют несуществующие модули LMMR (блокирующий импорт)

**Серьезность:** Высокая

**Ключевая несоответствие**

- `scripts/backtest_lmmr_gemini.py:7` — `from sirena.models.lmmr import LMMRForecaster`
- `scripts/backtest_lmmr_gemini.py:9` — `from sirena.models.lmmr_claude import LMMRForecasterClaude` в блоке `try`
- `scripts/backtest_lmmr_comparison.py:111` — `from sirena.models.lmmr import LMMRForecaster`
- `scripts/backtest_lmmr_comparison.py:125` — `from sirena.models.lmmr_claude import LMMRForecasterClaude`
- `scripts/backtest_lmmr_comparison.py:139` — `from sirena.models.lmmr_hybrid import LMMRHybridForecaster`

**Доказательство отсутствия файлов в кодовой базе** (выполнено из терминала):

- `missing: sirena/models/lmmr.py`
- `missing: sirena/models/lmmr_claude.py`
- `missing: sirena/models/lmmr_hybrid.py`

**Воздействие:**

1) `scripts/backtest_lmmr_gemini.py` падает на импорте модуля **до запуска** (ошибка на этапе инициализации).
2) `scripts/backtest_lmmr_comparison.py` в `try/except` не падает сразу, но фактически не выполнит бэктест по запрошенным LMMR-моделям.
3) Расхождение между фактической инфраструктурой и описанием моделей, которые должны поддерживаться скриптами.


### 2) `ModelRegistry` содержит дефолтный вес для несуществующей модели `lstm`

**Серьезность:** Средняя

**Ключевая несоответствие**

- `sirena/models/registry.py:31-43` — в `_default_weights` задано:
  - `'lstm': 0.05`

**Доказательство отсутствия регистрации/реальной модели**

- Поиск по реестру и моделям: `ModelRegistry.register("lstm")` отсутствует в `sirena/models`.
- `docs/MODEL_CATALOG.md:62` называет `LSTM` как архивную модель в `archive/models/lstm_model.py`, а не как активную модель в `sirena/models`.

**Воздействие:**

`ModelRegistry.create_ensemble()` использует веса как входной список кандидатов, но `ModelRegistry.get("lstm")` будет выбрасывать `KeyError`; модель фактически никогда не инициализируется, хотя вес в конфигурации присутствует.


### 3) Несогласованность реестра/экспорта: модели импортируются как API, но не зарегистрированы в `ModelRegistry`

**Серьезность:** Средняя

**Ключевые участки кода**

- `sirena/models/__init__.py:146` — экспорт `SubcomponentScenarioForecaster`
- `sirena/models/__init__.py:149` — экспорт `KiTrajectoryForecaster`
- `sirena/models/__init__.py:152` — экспорт `UnifiedSubcomponentForecaster`
- `sirena/models/__init__.py:178` — экспорт `VolatilityWeightedNowcaster`
- `sirena/models/__init__.py:179` — экспорт `RegimeAdaptiveNowcaster`

**Доказательство отсутствия декораторов регистрации**

- В `sirena/models/ki_trajectory.py:34` класс `KiTrajectoryForecaster` объявлен без `@ModelRegistry.register` и без импорта `ModelRegistry`.
- В `sirena/models/subcomponent_scenario.py:136` класс `SubcomponentScenarioForecaster` объявлен без `@ModelRegistry.register`.
- В `sirena/models/unified_subcomp.py:36` класс `UnifiedSubcomponentForecaster` объявлен без `@ModelRegistry.register`.
- В `sirena/models/volatility_weighted_nowcaster.py:60` и `sirena/models/regime_adaptive_nowcaster.py:60` классы объявлены, но также без `ModelRegistry.register`.
- Поиск `@ModelRegistry.register` в перечисленных файлах выдаёт ноль совпадений.

**Воздействие:**

Документация описывает единый механизм регистрации моделей, но часть экспортируемых модулей фактически недоступна через реестр (`ModelRegistry.get("..." )`) и не может участвовать в реестровых пайплайнах без ручной инстанцировки.


### 4) Документация `docs/MODEL_CATALOG.md` формально противоречит текущему коду реестра

**Серьезность:** Средняя

**Ключевые фрагменты документации**

- `docs/MODEL_CATALOG.md:218-220`: "Регистр моделей (ModelRegistry) ... все модели регистрируются через декоратор".
- `docs/MODEL_CATALOG.md:246`: ссылка на полный экспорт из `sirena/models/__init__.py` как «полный список".

**Воздействие:**

Литералы и примеры в каталоге дают ожидание, что все перечисленные модели имеют единый путь регистрации через декоратор. Фактически так не выполняется для ряда экспортируемых классов (см. дефект 3), что вводит в заблуждение API-инженеров и нарушает ожидания при добавлении/вызове моделей через реестр.


### 5) Архитектурное расхождение: `scripts/run_backtest_*.py` не используют `ModelRegistry`, а работают через `BacktestRunner` с ручным списком моделей

**Серьезность:** Средняя

**Ключевые участки**

- `scripts/run_backtest_h1.py:17` — импорт `BacktestRunner` напрямую.
- `scripts/run_backtest_h2.py:17` — аналогично.
- `scripts/run_backtest_h3.py:17` — аналогично.
- `scripts/run_backtest_h6.py:17` — аналогично.
- `scripts/run_backtest_h12.py:17` — аналогично.
- `scripts/backtest_framework.py:890-903` — `run()` явно делегирует в `_run_rolling`/`_run_h12`, без посредничества реестра.
- `scripts/backtest_framework.py:923-990` — в `_run_rolling` ручное перечисление `_forecast_*`.
- `scripts/backtest_framework.py:1023-1160` — в `_run_h12` также ручная инициализация моделей.

**Воздействие:**

Даже при корректной регистрации новой модели в `ModelRegistry`, скрипты бэктеста её не увидят и не запустят автоматически. Отсюда расхождение между реестровым API и фактическим производственным/backtest-пайплайном.


### 6) Дублирующийся/неиспользуемый импорт `LMMRHybridForecaster` через `try/except` в `backtest_framework.py`

**Серьезность:** Низкая

**Ключевой участок:**

- `scripts/backtest_framework.py:63-68` — `from sirena.models.lmmr_hybrid import LMMRHybridForecaster` внутри `try`.
- `scripts/backtest_framework.py:559-569` — `_forecast_lmmr_hybrid` всегда возвращает `np.nan`, если модуль недоступен.

**Воздействие:**

Даже не найденный импорт не останавливает весь бэктест (только предупреждение), но в отчете продолжаются колонки типа `LMMR_Hybrid` с пустыми значениями, создавая ложное впечатление поддерживаемости модели.


## Итоговая оценка рисков

- Имеются **три высокоопасных несоответствия**, влияющие на корректность запуска скриптов и целостность API-моделей:
  1. Ломанные LMMR-импорты в `scripts/backtest_lmmr_*.py`.
  2. Несогласованность весов реестра и набора зарегистрированных моделей (`lstm`).
  3. Расхождение между реестровой моделью и фактической модельной эксплуатацией бэктестов.

- Рекомендуется после устранения дефектов провести повторный аудит `ModelRegistry` против:
  - `docs/MODEL_CATALOG.md`
  - `docs/MODEL_CATALOG.md` — разделов по регистру и экспортам
  - скриптов бэктеста (`run_backtest_*.py` + `scripts/backtest_framework.py`)

---

## 3. Независимая состязательная верификация дефектов (Ревизор)

Дата верификации: `2026-09-09`  
Аудитор: Antigravity Code & Architecture Verifier (состязательный режим)

Проведена независимая проверка каждого из 6 дефектов, выявленных моделью, через прямое инспектирование файловой системы и статический анализ кода.

### Результаты проверки по пунктам

| № | Выявленный дефект | Статус проверки | Фактическое доказательство |
|---|---|---|---|
| **1** | Несуществующие модули LMMR (`lmmr.py`, `lmmr_claude.py`, `lmmr_hybrid.py`) в `scripts/backtest_lmmr_*.py` | **ПОДТВЕРЖДЕНО** | `[извлечено]` Файлы `sirena/models/lmmr*.py` физически отсутствуют в репозитории (`os.path.exists == False`). При запуске `python3 scripts/backtest_lmmr_gemini.py` процесс падает с `ModuleNotFoundError: No module named 'sirena.models.lmmr'`. Скрипты полностью неработоспособны. |
| **2** | Дефолтный вес несуществующей модели `lstm` в `ModelRegistry._default_weights` | **ПОДТВЕРЖДЕНО** | `[извлечено]` В [`sirena/models/registry.py:38`](file:///home/valalav/_projects/sirena-kbr/sirena/models/registry.py#L38) жестко прописано `'lstm': 0.05`. При этом ни в одном файле `sirena/models/*.py` нет декоратора `@ModelRegistry.register("lstm")` (модель перемещена в `archive/models/lstm_model.py`). Вызов `ModelRegistry.create_ensemble()` без явного списка весов приводит к `KeyError: "Model 'lstm' not found in registry"`. |
| **3** | Экспорт классов в `sirena/models/__init__.py` без регистрации в `ModelRegistry` | **ПОДТВЕРЖДЕНО С УТОЧНЕНИЕМ** | `[извлечено]` Классы `SubcomponentScenarioForecaster`, `KiTrajectoryForecaster`, `UnifiedSubcomponentForecaster`, `VolatilityWeightedNowcaster`, `RegimeAdaptiveNowcaster` экспортируются через `__init__.py`, но не содержат декоратора `@ModelRegistry.register`.<br>`[интерпретировано]` Уточнение: данные классы представляют собой композитные сервисы и nowcaster-пайплайны с нестандартным интерфейсом (например, метод `forecast_scenario` с параметрами шоков ставки). Однако факт отсутствия в реестре нарушает заявленный в документации контракт единообразия. |
| **4** | Противоречие в `docs/MODEL_CATALOG.md` (декларация регистрации всех моделей) | **ПОДТВЕРЖДЕНО** | `[извлечено]` В [`docs/MODEL_CATALOG.md:218-220`](file:///home/valalav/_projects/sirena-kbr/docs/MODEL_CATALOG.md#L218-L220) утверждается: *«все модели регистрируются через декоратор @ModelRegistry.register("model_name")»*. Фактически это не выполняется для композитных классов и забытого файла `ridge_extended_production_proxy_rolling.py`. |
| **5** | Архитектурный разрыв: `scripts/run_backtest_*.py` игнорируют `ModelRegistry` | **ПОДТВЕРЖДЕНО** | `[извлечено]` В [`scripts/backtest_framework.py`](file:///home/valalav/_projects/sirena-kbr/scripts/backtest_framework.py#L860-L1020) запуск моделей в `_run_rolling` и `_run_h12` зашит через жестко закодированные методы (`_forecast_huber`, `_forecast_ridge` и др.) со статическими импортами. Вызовы `ModelRegistry.get()` или `ModelRegistry.list_models()` в основном цикле бэктеста отсутствуют. Добавление новой модели в реестр не приводит к ее участию в бэктесте h=1..h=12. |
| **6** | Фиктивный импорт `LMMRHybridForecaster` в `scripts/backtest_framework.py` | **ПОДТВЕРЖДЕНО** | `[извлечено]` В [`scripts/backtest_framework.py:63-68`](file:///home/valalav/_projects/sirena-kbr/scripts/backtest_framework.py#L63-L68) импорт `LMMRHybridForecaster` обернут в `try/except Exception: LMMRHybridForecaster = None`. На строках 559–569 метод `_forecast_lmmr_hybrid` при отсутствии класса молча возвращает `np.nan`. В результате бэктест тратит ресурсы на прогон фиктивной колонки с пустыми значениями. |

---

## 4. Сводный вердикт и инженерные рекомендации

* **Статус аудита Codex:** Все 6 пунктов дефектов признаны состоятельными и подтверждены кодовой базой (5 подтверждены полностью, 1 — с архитектурным уточнением назначения nowcaster-классов).
* **Ложных срабатываний (False Positives):** 0 из 6.
* **Главный риск:** Попытка запустить `scripts/backtest_lmmr_gemini.py` гарантированно приводит к краху выполнения. Попытка вызвать `ModelRegistry.create_ensemble()` с дефолтными параметрами гарантированно приводит к аварийному завершению из-за `'lstm'`.
* **Правило невмешательства:** В соответствии с регламентом аудита, ни один файл исходного кода в рамках данной проверки изменен не был.


