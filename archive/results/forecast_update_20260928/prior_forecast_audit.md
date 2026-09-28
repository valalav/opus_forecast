# Сохранённый прогноз августа 2026: provenance и workflow

Дата фиксации: 2026-09-28. Сравнение read-only; исходная модельная выборка зафиксирована по git commit `2f2f7e8275a0a2a8c8d317e3208a8c6d94d2b081`.

## Августовский forecast-vs-fact

В коммите прогнозный снимок `data/precomputed_forecasts.json` создан 2026-08-17T11:40:14.242222 при последнем входном месяце `2026-07-01`; целевой первый месяц — `2026-08-01`. Это ex ante прогноз на июльских данных, а не backtest, пересчитанный после факта.

Для Micro точное сохранённое значение: **0.09395587896391% м/м** (индекс 100.09395587896391); ячейка workbook показывает округлённые **0.094**. Прогноз сохранён в `data/precomputed_forecasts.json` (git blob `1d6cb289e16032c1806c1cc217935ed170c8cc03`), зафиксированном commit `2f2f7e8275a0a2a8c8d317e3208a8c6d94d2b081`. Код этого snapshot: `scripts/precompute_forecasts.py` blob `c1c6b4cb2ad409dc1a7039a000b9a4aa9bcc42ac`, `sirena/models/micro_optimized.py` blob `5585f5a6f35b749945bf270252113fdb51d0fa12`; его вход `data/inflation_data.csv` blob `3e58df09c675c94379d6f8989c90b4721f56f562` (до июля). В старом JSON нет source/code manifest; внешний файл Micro_SM не был tracked в git. Сохранённая workbook создана отдельным 17-месячным workflow (`scripts/build_forecast_workbook_2027.py`) тем же днём. В ней Micro также равен 0.094 после округления, но NGBoostShock, NGBoost и Ensemble отличаются от 12-месячного JSON; поэтому CSV хранит обе версии и их ошибки, а точный Micro взят из исходного JSON. Это различие не меняет лидера августа.

Текущая валидная запись факта в `data/inflation_data.csv` за 2026-08: `mom=100,12`, то есть **+0.12% м/м**. Наименьшая абсолютная ошибка у Micro: **0.0260 п.п.**; следующая Prophet (0.313%, ошибка 0.193 п.п.). Ensemble прогнозировал 0.509507% (ошибка 0.389507 п.п.). Все модели и подписанные значения — в [`august_model_comparison.csv`](august_model_comparison.csv).

Точность выше относится к одному августовскому исходу, не к общей стабильности модели. Модельный снимок был сделан до месячного факта; он основан на данных по июль 2026, горизонт 12 месяцев. Текущий кэш `data/precomputed_forecasts.json` после последующих пересчётов другой и не использовался для сравнения.

## Обновление ОПР книги и аналитического артефакта

- Основная отправочная книга — `assets/06_2026_02_Прогноз.xlsx`, лист `Прогноз`, актуальный MoM индекс в столбце E и YoY в F; контракт записан в `docs/OPR_FORECAST_LINKAGE.md`.
- Конкретное обновление формы 2026-09-01 было записано отдельным скриптом `archive/results/opr_august_preliminary_20260901/update_pr3_forecast.py`: он создаёт backup и пересчитывает траекторию CSV, но жёстко закодирован на исходные значения августа 100.50/сентября 100.65. Запуск с `--help` не поддержан; при вызове скрипт проверил текущую книгу и остановился на несовпадении августа (там уже 100.00), до изменения/копирования. Его можно брать как шаблон защитных проверок, но для нового обновления параметры и базовые значения должны соответствовать текущей книге.
- Единственный найденный общий генератор прогноза/книги/ZIP для старого обновления: `scripts/build_forecast_workbook_2027.py`; CLI — `python3 scripts/build_forecast_workbook_2027.py`, без аргументов. Он пересобирает архивную модельную книгу из `data/precomputed_forecasts_h17.json` и константы `DECIDED_2026`; на саму ОПР форму `assets/06_2026_02_Прогноз.xlsx` не пишет. Не запускал его, потому что он перезаписывает workbook/ZIP и исходные константы отражают старое решение.
- `archive/results/aug_2026_update/august_2026_update_report.md` — сохранённый аналитический markdown без отдельного генератора. Для новой траектории нужен свежий report/артефакт с объяснением предпосылок; `build_forecast_workbook_2027.py` можно переиспользовать/расширить для форматированной книги и ZIP, но он не генерирует этот отчёт.

## Перестроение Micro_SM внешней матрицы

Внешняя матрица живёт в `data/external/micro_cpi_region_export/`. При последнем обновлении metadata говорит latest_month=`2026-08-01`, регион КБР (`7`), данные raw и единицы index_previous_month_100. Штатный пайплайн использует `scripts/run_statsmodels.py`, затем `scripts/export_micro_test_matrix.py`, которые формируют `statsmodels_forecast.csv` и вход адаптера `micro_test_statsmodels.csv`. Из корня репозитория, последовательный CLI:

```bash
python3 data/external/micro_cpi_region_export/scripts/run_statsmodels.py \
  --input data/external/micro_cpi_region_export/region_cpi_long.csv \
  --metadata data/external/micro_cpi_region_export/metadata.json \
  --config data/external/micro_cpi_region_export/config/model.yml \
  --out data/external/micro_cpi_region_export/statsmodels_forecast.csv \
  --diagnostics data/external/micro_cpi_region_export/statsmodels_diagnostics.csv
python3 data/external/micro_cpi_region_export/scripts/export_micro_test_matrix.py \
  --forecast data/external/micro_cpi_region_export/statsmodels_forecast.csv \
  --metadata data/external/micro_cpi_region_export/metadata.json \
  --out data/external/micro_cpi_region_export/micro_test_statsmodels.csv
```

Both CLIs were read via `--help`; they default to aggregate `item_code=1` and use `metadata.latest_month` as cutoff. Adapter: `sirena/models/micro_statsmodels_external.py`; caller in `scripts/precompute_forecasts.py` lines 440–448. `precompute_forecasts.py` has no matrix-rebuild flag: it consumes this CSV and will stop if the model cutoff is behind the latest training cutoff. The alternative rolling matrix builder is `data/external/micro_cpi_region_export/scripts/build_rolling_micro_test_matrix.py`; it accepts `--input`, `--metadata`, `--out`, `--diagnostics`, `--config`, `--horizon`, `--item-code`, and `--cutoff-start/--cutoff-end` and creates historical cutoff columns for backtesting.
