# External Code Repository Intake, 2026-06-25

This folder contains the user's copied Bank of Russia code repository archive
and the extracted nested materials.

## Layout

- `raw/repo_codes.rar` - original archive moved from
  `~/_work/cb/_00_INBOX/Репозиторий кодов.rar`.
- `extracted/` - top-level archive extraction.
- `nested_extracted/` - nested archives extracted into short ID folders.
- `inventory/archive_extraction_map.csv` - map from each nested archive to its
  short extraction folder.
- `inventory/file_inventory.csv` - full file inventory.
- `inventory/code_files.csv` - source/script-like files only.
- `inventory/model_folder_summary.csv` - counts by source folder.
- `inventory/study_candidates.csv` - ranked candidate files for model review.
- `STUDY_FILES.md` - short human-readable list of files to inspect first.
- `agy/` - read-only subagent reports.

## Extraction Summary

- Top-level archive: `raw/repo_codes.rar`, about 605 MiB.
- Nested archives extracted: 63.
- Full extracted file inventory: 2659 files.
- Script/code-like files: 1544.

`7z` could not extract 36 RAR5 entries due to an unsupported method, but
`unrar` successfully re-extracted the top-level archive and recovered them.

## Highest-Value Files To Study First

### Weekly-to-Monthly Nowcast

- `nested_extracted/a24cd898db7c9/5 Недельная инфляция/weekCpiR/R/scripts/main.R`
- `nested_extracted/a24cd898db7c9/5 Недельная инфляция/weekCpiR/R/common/data_cpi_week_wow_nowcast.R`
- `nested_extracted/a24cd898db7c9/5 Недельная инфляция/weekCpiR/R/common/data_cpi_month_mom_from_wow_calculate.R`
- `nested_extracted/a24cd898db7c9/5 Недельная инфляция/weekCpiR/R/common/data_cpi_week_bi_calculate.R`
- `nested_extracted/a24cd898db7c9/5 Недельная инфляция/weekCpiR/R/common/data_cpi_week_bi_bias_correct.R`
- `nested_extracted/a17db49e525af/Описание и код модели недельная - месячная инфляция.docx`

### Micro / Subcomponent Forecasting

- `nested_extracted/a0268154c4591/ВВГУ_Мордовия_КСП инфляции Python/infl.py`
- `nested_extracted/a2363877985ef/Прогноз_ИПЦ_по компонентам_Хабаровск/khab_mod.prg`
- `nested_extracted/afa0cc7c1c88b/ARIMA-45 (Омск, СГУ)/Скрипт/arima_omsk.prg`
- `nested_extracted/a8dceef0607a0/ИПЦ_Магадан/Magadan_sripts_cpi.prg`
- `nested_extracted/ae00677697d94/ARIMAX.R`

### Variable Selection, ARDL, ARIMAX, ML

- `nested_extracted/a0ac088b0b1be/Методика отбора переменных, обеспечивающих надежные прогнозы/variable_models_code.prg`
- `nested_extracted/ac2294d8a53ba/ARDL_Челябинск/model_cfr_10_2023.wf1`
- `nested_extracted/ae00677697d94/ARIMAX.R`
- `nested_extracted/a9c66f72e388c/Комбинированный прогноз ИПЦ моделей машинного обучения/methodology.Rmd`
- `nested_extracted/a9c66f72e388c/Комбинированный прогноз ИПЦ моделей машинного обучения/forecasts/XGB.Rmd`
- `nested_extracted/a9c66f72e388c/Комбинированный прогноз ИПЦ моделей машинного обучения/forecasts/Random forest.Rmd`

### Seasonality / Diagnostics

- `nested_extracted/aa972a9a24e63/Сглаживание ИПЦ/main.R`
- `nested_extracted/a5331c69b69ea/Применение метода главных компонент и кластеризации данных/Код_кластеризация.ipynb`

## Initial Judgement

The most actionable materials are:

1. The `weekCpiR` R pipeline for weekly-to-monthly nowcast mechanics,
   especially item-level weekly indices, bias correction, and monthly
   accumulation.
2. Khabarovsk/Omsk 45-component ARIMA aggregation logic for a cleaner
   medium-granularity bottom-up model.
3. Mordovia Python `auto_arima` loop as a simple Python baseline, but not as a
   complete model because it lacks weighted headline reconstruction.
4. The Volgograd variable-selection methodology as a metric/gate concept,
   not as a direct EViews port.

Do not treat the external code as verified production evidence. Any port must
be run through SIRENA-KBR rolling backtests and leakage checks.
