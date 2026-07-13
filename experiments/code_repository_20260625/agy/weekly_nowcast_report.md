# Weekly Nowcast Research Report

This report summarizes the most useful weekly-to-monthly nowcast code and files for SIRENA-KBR, extracted from the repositories `a24cd898db7c9` and `a17db49e525af`.

## 1. Tyumen EViews Model (`a17db49e525af`)

This directory contains a simplified regression-based approach for regional nowcasting.

**Algorithm:**
- A single-equation regression (estimated via EViews `ls`) predicting regional monthly inflation.
- Uses accumulated weekly inflation of sub-regions (Tyumen south, KhMAO, YNAO) as independent variables.
- Incorporates non-linear terms (e.g., squared accumulated weekly inflation).
- Uses the Hodrick-Prescott (HP) filter to compute gaps in relative prices.
- Includes Autoregressive (AR) terms.

**Inputs:**
- Accumulated weekly inflation from Rosstat's weekly monitoring (narrow basket).
- Regional monthly inflation facts (wide basket).

**Leakage Risks:**
- **High Risk:** The Hodrick-Prescott (HP) filter is a two-sided filter. Using it on the full dataset during historical evaluation introduces forward-looking bias (future data leaks into past estimates of the relative price gap).
- The model relies heavily on non-linear transformations to compensate for a small sample size, increasing overfitting risks.

**Key File for Codex:**
- `experiments/code_repository_20260625/nested_extracted/a17db49e525af/Описание и код модели недельная - месячная инфляция.docx`

---

## 2. CBR R Pipeline (`a24cd898db7c9` - 5 Недельная инфляция)

This directory contains a mature, item-level bottom-up pipeline written in R. It models week-over-week (wow) growth and aggregates it into monthly (mom) metrics.

**Algorithm:**
- **Nowcast Models (`data_cpi_week_wow_nowcast.R`):** Uses CatBoost and Linear Models (`lm`) to predict item-level weekly inflation (`wow`).
- **Aggregation (`data_cpi_week_bi_calculate.R`):** Aggregates item-level indices up to broader categories (levels 7 down to 1) using a modified Laspeyres index formula. It incorporates a bias-correction step (`data_cpi_week_bi_bias_correct.R`) to reconcile high-frequency weekly signals with actual monthly reports.
- **Monthly Accumulation (`data_cpi_month_mom_from_wow_calculate.R`):** Computes monthly MoM inflation by accumulating daily/weekly indices (`bi_daily`). It handles specific Rosstat price registration days and dynamically shifts dates (e.g., treating the 25th-28th as the boundary for the month).

**Inputs:**
- Item-level weekly price changes (`wow`).
- Reference dictionaries (weights, OKATO region codes, item codes).
- Exchange rates (`exr`).

**Leakage Risks:**
- **CatBoost Model:** If trained on the full panel without strict time-based expanding window splits, it will suffer from time-leakage during historical backtesting.
- **Exchange Rates:** Weekly average exchange rates are used. Need to ensure the timing is strictly contemporaneous or lagged (no future exchange rates from the prediction week).
- **Rosstat Registration Days:** Hardcoded date boundaries for price registration (`25` or `rosstat` actual day) require careful alignment. Mismatches during historical backtesting versus real-time operation can introduce subtle data availability leaks.

**Key Files for Codex to Read Next:**
- `experiments/code_repository_20260625/nested_extracted/a24cd898db7c9/5 Недельная инфляция/weekCpiR/R/common/data_cpi_week_wow_nowcast.R`
- `experiments/code_repository_20260625/nested_extracted/a24cd898db7c9/5 Недельная инфляция/weekCpiR/R/common/data_cpi_month_mom_from_wow_calculate.R`
- `experiments/code_repository_20260625/nested_extracted/a24cd898db7c9/5 Недельная инфляция/weekCpiR/R/common/data_cpi_week_bi_calculate.R`
