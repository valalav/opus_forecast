# Проект: инструмент анализа устойчивой инфляции КБР

## Миссия

Создать независимый, воспроизводимый инструмент анализа устойчивой инфляции для КБР, который рассчитывает несколько устойчивых CPI-индикаторов, объясняет месячные скачки и автоматически ловит ошибки входных данных вроде смешения MoM/YoY.

## Почему это нужно

Существующий Excel-инструмент полезен как исторический прототип, но аудит показал критический риск: сохраненный лист `ИПЦ исходный mom` совпал с `ИПЦ исходный yoy`, а формулы аннуализации обработали YoY как MoM. Новый инструмент должен быть проверяемым, конфигурируемым и независимым от состояния `.xlsm`.

## Принципы

- Не менять production-данные в `data/`.
- Не ломать существующие модели СИРЕНА.
- Сначала делать экспериментальный CLI/отчет, потом интеграцию.
- Все показатели считать из явных входов и конфигов.
- Любой итоговый показатель сопровождать диагностикой качества.
- Не говорить "проверено", пока не пройдены конкретные gates.

## Предлагаемая структура будущей реализации

На этапе MVP:

```text
experiments/core_inflation_tool/
  README.md
  config/
    core_inflation_config.yaml
    exclusion_groups.yaml
  core_inflation/
    __init__.py
    loaders.py
    weights.py
    indicators.py
    diagnostics.py
    contributions.py
    report.py
    cli.py
  tests/
    test_indicators.py
    test_diagnostics.py
    test_weights.py
  outputs/
    .gitkeep
```

После проверки можно перенести стабильные части в `sirena/` и `scripts/`.

## Этапы

### Этап 0. Инвентаризация источников

Цель: определить лучший месячный источник для компонент, весов и справочников.

Файлы-кандидаты:

- `data/inflation_data.csv`
- `data/mom_sa_kbr.csv`
- `data/kbr_indices.csv`
- `data/access_weights.csv`
- `data/items_names.csv`
- `data/micro_sprav.csv`
- `data/raw/sub_mom.csv`
- `data/raw/subcomp_sprav.csv`
- `data/trimmed_mean_cpi.csv`
- `data/sticky_price_index.csv`
- `data/inflation_persistence.csv`

Результат:

- таблица источников;
- выбранные canonical inputs для MVP;
- список известных ограничений.

### Этап 1. Ядро расчета

Реализовать функции:

- перевод индекса `100.xx` в MoM-прирост;
- весовая агрегация;
- перенормировка весов;
- exclusion core;
- weighted trimmed mean;
- weighted median;
- простые сравнения с headline.

Результат:

- чистые функции без зависимости от Excel;
- unit-тесты на синтетических данных;
- проверка edge cases: пустые веса, нулевые веса, NaN, экстремальные значения.

### Этап 2. Диагностики качества

Реализовать проверки:

- MoM/YoY не идентичны;
- диапазон MoM разумный;
- сумма весов стабильна;
- нет смешения уровней иерархии;
- итоговые ряды не содержат технических ошибок;
- скачки выше порога объясняются компонентами.

Результат:

- `core_inflation_diagnostics.csv`;
- machine-readable statuses: `pass`, `warning`, `fail`, `expected_skip`;
- Markdown-сводка по проблемам.

### Этап 3. Вклады и объяснение скачков

Реализовать:

- вклад каждой компоненты в exclusion core;
- вклад включенных/исключенных компонент;
- топ положительных/отрицательных драйверов;
- gap между headline и core;
- jump report за последние 24-36 месяцев.

Результат:

- `core_inflation_contributions.csv`;
- `core_inflation_jump_report.md`;
- таблицы, пригодные для управленческой записки.

### Этап 4. CLI и воспроизводимые артефакты

Сделать команду:

```bash
python3 -m experiments.core_inflation_tool.core_inflation.cli \
  --config experiments/core_inflation_tool/config/core_inflation_config.yaml \
  --output experiments/core_inflation_tool/outputs/latest
```

Результат:

- единая команда пересчета;
- config snapshot в output;
- лог проверок;
- стабильные CSV/MD outputs.

### Этап 5. Сравнение с существующими заготовками

Сравнить новый расчет с:

- `data/trimmed_mean_cpi.csv`;
- `data/sticky_price_index.csv`;
- существующим Excel-аудитом;
- headline CPI из `data/inflation_data.csv`.

Результат:

- отчет о расхождениях;
- решение, какие старые артефакты можно считать reference, а какие нет.

### Этап 6. Решение об интеграции

После MVP решить:

- оставить инструмент в `experiments/`;
- перенести ядро в `sirena/core_inflation.py`;
- добавить operational script в `scripts/`;
- добавить вкладку/графики в dashboard;
- добавить регулярный экспорт в `assets/charts/`.

## Acceptance Criteria MVP

MVP считается готовым, если:

1. Одна команда строит все выходные артефакты в отдельной output-папке.
2. `core_inflation_series.csv` содержит минимум:
   - `date`;
   - `headline_mom`;
   - `exclusion_core_mom`;
   - `trimmed_mean_mom`;
   - `weighted_median_mom`;
   - `headline_core_gap`;
   - diagnostic flags.
3. Диагностика ловит искусственный fixture, где MoM полностью равен YoY.
4. Weighted aggregation тестируется на синтетическом примере с ожидаемым числом.
5. Trimmed mean и weighted median тестируются на синтетических весах.
6. Нет записи в production `data/` и `assets/`.
7. Отчет явно перечисляет непроверенные источники и ограничения.

## Риски

- Неоднозначная иерархия компонент и `Item_type`.
- Разные источники весов дают разные суммы.
- Микрокомпонентные ряды могут не агрегироваться в опубликованные субкомпоненты.
- Weekly-данные полезны для nowcast, но не должны подменять monthly facts.
- Слишком агрессивные исключения могут сделать "устойчивую" инфляцию менее репрезентативной.

## Рекомендуемый первый development run

Первый run должен быть ограниченным:

- не интегрировать в dashboard;
- не менять модели;
- не переписывать существующие scripts;
- сделать автономный MVP в `experiments/core_inflation_tool/`;
- проверить на синтетических fixtures и на текущих CSV;
- подготовить отчет о качестве данных.
