# Experimental Core Inflation Tool

This isolated tool builds core inflation analysis artifacts without touching
production `data/`, `sirena/`, `scripts/`, or `assets/`.

## Approved methodology

The approved Excel workbook is the source of truth. The filtered indicator
below is an analytical experiment and MUST NOT replace the approved result.
Extract and independently reproduce the official row-4 median with:

```bash
python3 -m experiments.core_inflation_tool.core_inflation.official_workbook \
  --workbook _00_inbox/Расчет_устойчивой_инфляции.xlsm \
  --database _00_inbox/database.xlsx \
  --regsa _00_inbox/RegSA.xlsx \
  --output experiments/core_inflation_tool/outputs/official/latest
```

The official indicator is the median of 16 annualized estimates selected by
the formula on `Результаты!4`. The official series CSV also publishes a
transparent, non-official analytical layer: the implied SA monthly rate, its
3MMA and annualized 3MMA, the min/Q1/Q3/max/IQR of all 16 estimates, and the
two central estimates that form each month's median. Validation artifacts
record the exact median and formula reproduction, monthly-sheet references,
database/YoY and RegSA transfers, MoM/YoY separation, `RAND()` tie-breaker
count, and SHA-256 fingerprints of all three source workbooks.

Official-workbook outputs:

- `official_stable_inflation_series.csv` — official values plus the analytical layer;
- `official_stable_inflation_components.csv` — all 16 estimates and labels by month;
- `official_validation.json` — machine-readable calculation controls;
- `official_source_manifest.json` — source paths, sizes, and SHA-256 fingerprints;
- `official_reproduction_report.md` — Russian audit and interpretation report.

### Native Linux calculation

Calculate the same 16 approved estimates and their median without executing
Excel or LibreOffice:

```bash
python3 -m experiments.core_inflation_tool.core_inflation.native_official \
  --regsa _00_inbox/RegSA.xlsx \
  --weights data/access_weights.csv \
  --raw-mom _00_inbox/Расчет_устойчивой_инфляции.xlsm \
  --reference-workbook _00_inbox/Расчет_устойчивой_инфляции.xlsm \
  --output experiments/core_inflation_tool/outputs/native/latest
```

`--reference-workbook` is optional and is read only to compare cached approved
results. It never supplies any of the 16 calculated values. `--raw-mom`
supplies the raw credit-price observation needed to build the residual
“Другие услуги” component; it may be either a fresh `kbr_indices.csv`-format
CSV or a source workbook containing the `ИПЦ_все` input sheet. The calculation
itself uses 45 non-overlapping RegSA components, annual `Weight_vertical`
weights, deterministic code-based tie breaking, weighted partial-boundary
trimming, 3/24-month sample volatility, declared fixed exclusions, and the
median of all 16 SAAR estimates.

Native outputs:

- `native_official_stable_inflation_series.csv`;
- `native_official_stable_inflation_components.csv`;
- `native_official_diagnostics.json`;
- `native_official_source_manifest.json`;
- `native_official_report.md`;
- `native_official_comparison.csv` and `native_official_validation.json` when
  a reference workbook is supplied.

The tool currently provides:

- component contribution tables;
- jump-focused Markdown report;
- ordinary and seasonally adjusted raw stable-core signals and filtered indicators;
- 3MMA, 3MMA annualized, and rolling 12-month annual stable-core metrics;
- long-run smoothness and practical usefulness diagnostics;
- Russian dynamics report for analytical review;
- CLI artifact writer;
- config snapshot for each run.

Repository-input runs enforce one hierarchy level per series:

- ordinary indicators use the 537 leaf items listed in `data/micro_sprav.csv`;
- SA indicators use the configured 44 non-overlapping subcomponents;
- every month must cover at least 98% of the eligible annual basket weight;
- fruit and vegetables plus motor fuel are excluded by named micro-basket groups;
- SA exclusions use the matching aggregate groups: housing and utilities,
  fruit and vegetables, and motor fuel;
- `exclusion_core` winsorizes the lower and upper 5% weighted tails. Raw
  component values and raw exclusion core remain in audit outputs.

`stable_core_signal` is the monthly average of exclusion core and the weighted
trimmed mean. Published `stable_core` applies a causal robust filter to that
signal: each month it absorbs `alpha` of the current innovation after clipping
the innovation to `max_innovation_pp`. With the default `alpha: 0.35` and
`max_innovation_pp: 1.0`, the indicator can change by at most 0.35 percentage
points in one month. The filter uses no future observations.

Run shape:

```bash
python3 -m experiments.core_inflation_tool.core_inflation.cli \
  --config experiments/core_inflation_tool/config/core_inflation_config.yaml \
  --output experiments/core_inflation_tool/outputs/latest
```

The CLI refuses output paths outside `experiments/core_inflation_tool/outputs`
unless a test fixture explicitly sets `CORE_INFLATION_ALLOW_EXTERNAL_OUTPUT=1`
or calls `run(..., allow_output_outside_experiment=True)`.

Minimal config:

```yaml
input:
  components_csv: components.csv
columns:
  date: date
  component: component
  mom: mom
  weight: weight
  excluded: excluded
trim_lower: 0.10
trim_upper: 0.10
winsor:
  lower: 0.05
  upper: 0.05
smoothing:
  alpha: 0.35
  max_innovation_pp: 1.0
report:
  jump_threshold: 0.50
```

Outputs:

- `core_inflation_series.csv`
- `core_inflation_diagnostics.csv`
- `core_inflation_contributions.csv`
- `core_inflation_sa_contributions.csv`
- `core_inflation_longrun_metrics.csv`
- `core_inflation_jump_report.md`
- `core_inflation_dynamics_report.md`
- `core_inflation_config_used.json`

Limitations:

- The exclusion list is transparent but still requires business approval
  before the indicator can be promoted to production.
- Failed diagnostics are intentionally propagated into the report and series;
  they mean the numeric results should not be used analytically.
- `data/mom_sa_kbr.csv` is revised history unless real-time vintages are
  established; SA results are analytical rather than vintage-safe backtests.
- This experiment is not integrated into the dashboard or production models.
