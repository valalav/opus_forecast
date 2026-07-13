# External Code Repository Intake

Date: 2026-06-25

## What Was Added

The user copied a large external code archive to:

```text
~/_work/cb/_00_INBOX/Репозиторий кодов.rar
```

It was moved into the SIRENA-KBR workspace as:

```text
experiments/code_repository_20260625/raw/repo_codes.rar
```

The archive and its nested archives were unpacked under:

```text
experiments/code_repository_20260625/
```

## Inventory Artifacts

- `experiments/code_repository_20260625/README.md` - operational map and short
  first-study list.
- `experiments/code_repository_20260625/STUDY_FILES.md` - candidate files by
  topic.
- `experiments/code_repository_20260625/inventory/file_inventory.csv` - full
  file inventory.
- `experiments/code_repository_20260625/inventory/code_files.csv` - code/script
  files.
- `experiments/code_repository_20260625/inventory/study_candidates.csv` -
  ranked study candidates.
- `experiments/code_repository_20260625/inventory/model_folder_summary.csv` -
  model/folder counts.
- `experiments/code_repository_20260625/inventory/archive_extraction_map.csv` -
  nested archive extraction map.

## Extraction Result

- Top-level archive size: about 605 MiB.
- Nested archives extracted: 63.
- Full extracted inventory: 2659 files.
- Code/script-like files: 1544.

`7z` initially failed on 36 RAR5 entries due to unsupported compression, but
`unrar` recovered the top-level archive entries. Nested RAR files were then
extracted into short-ID folders under `nested_extracted/`.

## Subagent Reports

Read-only `agy` research reports were saved at:

- `experiments/code_repository_20260625/agy/weekly_nowcast_report.md`
- `experiments/code_repository_20260625/agy/micro_subcomponent_report.md`
- `experiments/code_repository_20260625/agy/challenger_models_report.md`

## Current Priority Findings

Highest priority:

1. `weekCpiR` weekly inflation pipeline:
   `nested_extracted/a24cd898db7c9/5 Недельная инфляция/weekCpiR/`
2. Khabarovsk/Omsk 45-component bottom-up logic:
   `nested_extracted/a2363877985ef/Прогноз_ИПЦ_по компонентам_Хабаровск/khab_mod.prg`
   and
   `nested_extracted/afa0cc7c1c88b/ARIMA-45 (Омск, СГУ)/Скрипт/arima_omsk.prg`
3. Mordovia Python component ARIMA baseline:
   `nested_extracted/a0268154c4591/ВВГУ_Мордовия_КСП инфляции Python/infl.py`
4. Volgograd variable-selection gate:
   `nested_extracted/a0ac088b0b1be/Методика отбора переменных, обеспечивающих надежные прогнозы/variable_models_code.prg`

Important caution:

- External code is not production evidence.
- Any port must use SIRENA-KBR loaders, cutoff-safe exogenous paths, rolling
  backtests, and recursive/vintage-safe seasonal adjustment.
