# VAR-family model research

This document summarizes the 2026-06 VAR-family research track and the final mandatory VAR policy.

## Goal

The project needed a defensible VAR-family model. The goal was not to prove that VAR beats every production ML model, but to build a professional, leakage-free, interpretable VAR benchmark/control model with honest limitations.

## Main conclusion

Use a horizon-specific VAR policy:

| Horizon | Recommended VAR-family model | Rationale |
| --- | --- | --- |
| h=1 | `RegimeMacroVARX_l1` | Best short-horizon VAR signal: macro VARX in normal regimes, robust Huber VAR in shock regimes. |
| h=12 | `SeasonalVAR_CPI_F_NF_S` | Best deterministic long-horizon trajectory: explicit train-only seasonality, realistic path shape, no random noise. |

If a single VAR model is required across horizons, use `SeasonalVAR_CPI_F_NF_S` as the stable trajectory fallback.

## Final rolling backtest

Command:

```bash
python3 experiments/var_sa_research/final_var_policy_backtest.py --run-name final_var_policy_rolling
```

Artifacts:

- `experiments/var_sa_research/final_var_policy_report.md`
- `experiments/var_sa_research/final_var_policy_backtest.py`
- `experiments/var_sa_research/runs/final_var_policy_rolling/`

Final h=1/h=12 rolling metrics:

| Horizon | Model | All-window MAE | OOS non-shock MAE | 2022 shock MAE | KPI violations |
| ---: | --- | ---: | ---: | ---: | ---: |
| h=1 | `RegimeMacroVARX_l1` | 0.379 | 0.304 | 0.831 | 24 |
| h=1 | `SeasonalVAR_CPI_F_NF_S` | 0.396 | 0.347 | 0.754 | 29 |
| h=1 | `Hybrid_VAR_Policy` | 0.379 | 0.304 | 0.831 | 24 |
| h=12 | `RegimeMacroVARX_l1` | 0.547 | 0.433 | 1.236 | 39 |
| h=12 | `SeasonalVAR_CPI_F_NF_S` | 0.440 | 0.374 | 0.909 | 32 |
| h=12 | `Hybrid_VAR_Policy` | 0.440 | 0.374 | 0.909 | 32 |

Trajectory diagnostics from the same run:

| Model | Mean path MAE | Mean vol ratio | Mean flatness | Mean seasonal correlation |
| --- | ---: | ---: | ---: | ---: |
| `RegimeMacroVARX_l1` | 0.522 | 0.150 | 0.841 | 0.090 |
| `SeasonalVAR_CPI_F_NF_S` | 0.445 | 0.553 | 0.144 | 0.973 |
| `Hybrid_VAR_Policy` | 0.445 | 0.553 | 0.144 | 0.973 |

Interpretation: `RegimeMacroVARX_l1` is useful for h=1, but its recursive h=12 path is too flat and weakly seasonal. `SeasonalVAR_CPI_F_NF_S` is the appropriate long-horizon trajectory model.

## Research path and lessons

### 1. Naive VAR/BVAR was not enough

Initial tests of plain VAR/BVAR, including archived BVAR variants, showed weak or unstable results. The old BVAR did not provide a defensible mandatory model by itself.

### 2. Single-window tuning overfit

`fine_seasonal_resid_var_tc_roll42_l5` reached h=1 MAE around 0.223 on the selected 2025-04..2026-03 window. Independent robustness checks showed this was a single-window grid-selection artifact. It is rejected for production and for mandatory VAR reporting.

### 3. Plain VAR(1) became the clean baseline

`PlainVAR_BIC` on `CPI + Food + NonFood + Services` is effectively stable VAR(1), selected by BIC in all h=1 cutoffs. It beats archived BVAR and remains the clean baseline.

### 4. Outlier and shock handling matters

Proper robust handling improved the VAR line. The useful pattern was not hard-coded 2022 dummies; those often made forecasts worse. The useful model was:

- normal regime: VARX with `USD`, `Ruonia`, `Ki_i`;
- shock regime: robust Huber VAR without macro exog;
- regime rule from cutoff-observed CPI volatility only.

This is `RegimeMacroVARX_l1`.

### 5. More macro variables are not automatically better

Variable subset tests showed:

- `Ki_i` helps h=1.
- `USD` is often harmful at h=12.
- `Ruonia` is less stable than `Ki_i`.
- all-macro VARX is not better than a smaller macro set.

Future VAR work must test macro variables with and without each block; never assume more variables improve a VAR.

### 6. Long-horizon trajectory must be evaluated separately

Models with acceptable h=12 MAE can still produce bad trajectories: flat, anti-seasonal, or unrealistic paths. Do not add random noise to make paths look realistic.

For h=12, explicit deterministic seasonality is required. `SeasonalVAR_CPI_F_NF_S` uses expanding train-only month-of-year means, VAR(1) on residuals, and deterministic seasonal reconstruction. This is not the rejected roll-window tuning path.

## Current recommended status

- `Hybrid_VAR_Policy`: recommended mandatory VAR policy when horizon-specific models are allowed.
- `SeasonalVAR_CPI_F_NF_S`: recommended single-model fallback and long-horizon trajectory VAR.
- `RegimeMacroVARX_l1`: recommended h=1 mandatory VAR component.

Production integration status:

- registered model: `var_policy`
- implementation: `sirena/models/var_policy.py`
- export: `sirena/models/__init__.py`
- forecast cache key: `VARPolicy`
- ensemble participation: added to `scripts/precompute_forecasts.py` with a modest relative weight (`0.03`) as an interpretable mandatory VAR-family control.

This is not claimed to supersede the strongest ML models. It is included to satisfy the mandatory VAR-family requirement and to add a leakage-free macro/seasonal econometric control signal.

## Cleanup / archive

Historical one-off research scripts, task prompts, and previous run artifacts were moved under:

- `archive/results/var_mandatory_research_20260607/`

The active research folder keeps only the final runner, final report, and final rolling run artifacts.

## Rules for future agents

- Start from `final_var_policy_backtest.py` and this document before proposing another VAR model.
- Always test outliers/shocks before advanced VAR variants.
- Always test macro subsets, not only all-macro specifications.
- Always test **endogenous** subsets as well, not only macro (exogenous) subsets. See the
  grid-search study below — a smaller endogenous set (e.g. `{mom, Nonprod}`) can beat the
  full `{CPI, Food, NonFood, Services}` specification on h=1 AFE/MAE.
- Always evaluate h=12 trajectory realism separately from point MAE.
- Always report RMSE, MAE, and Theil together when comparing VAR candidates;
  always report MAPE only with a clipped denominator.
- Always compare a candidate VAR against an **ARMA(1,1) baseline** on the same
  cut-offs and horizons. A VAR that does not beat ARMA(1,1) on RMSE/MAE is not
  worth promoting.
- For every backtest report, also report the 6 naïve baselines (target, naive, mean6,
  ar1, argap, direct) so reviewers can see whether the production models are actually
  beating drift/random-walk — see `scripts/run_baselines.py`.
- Never add stochastic noise to point forecasts for visual realism.

## Baselines reference: `scripts/run_baselines.py` (2026-06)

After `run_backtest_h1.py` (or h2 / h12) writes `archive/results/backtest_h{H}_predictions.csv`,
run:

```bash
python3 scripts/run_baselines.py --horizon 1
```

This adds 6 naïve baseline columns (`target`, `naive`, `mean6`, `ar1`, `argap`, `direct`)
to the predictions CSV (computed from the same train window up to each cut-off) and
produces an extended metrics CSV with **Theil U** for every model — including the
existing ones. Implementation is a port of the ВВГУ `cpi_fcst.R` `fsct_simple()` and
uses `benchmarks_cpi.py:BENCHMARKS` directly.

Why Theil U is informative here: the standard Theil U = RMSE(model) / RMSE(actual.shift(1))
is < 1 for drift-style models (`naive`, `ar1`, `direct` at h=1) and typically > 1 for
multi-step VAR-style models that do not condition on the last actual. Reporting both
together lets the reviewer see which kind of model is being compared.

## Side study: UGU `enum.prg` grid search (2026-06)

External reference material: `_inbox/var_extracted/var/Поиск лучшей спецификации модели для
прогноза/Поиск лучшей спецификации_УГУ/`. Source code: `enum.prg` (EViews 11, 103 lines,
CP1251), data: `data.wf1` (not opened on Linux; `pyeviews` is COM-only and is unusable here).
Full read-through and methodology comparison: `summary.md` next to the source files.

Their approach: VAR grid search over (a) all `C(n, k)` endogenous subsets for k=1..n,
(b) VAR lags 1..6, (c) expanding-window pseudo-out-of-sample forecast, with three
metrics per horizon — AFE, MSFE, MAPFE.

The two reusable patterns from that work:

1. **MAPFE with `mom_y` in the denominator**, not `y` — this matches our existing
   `mom`-based series and avoids near-zero division at low inflation rates.
2. **Explicit grid over endogenous subsets** — we had not done this systematically for
   the four CPI components. A smoke test of the port (`scripts/grid_search_var.py`,
   196 KBR observations 2010-01..2026-04, min_window=60, max_lag=3, horizon=3) confirmed
   the pattern: `{mom, Nonprod}` with lag=1 wins h=1 AFE (~0.403), and the full
   `{mom, Prod, Nonprod, Serv}` specification is 15th (~0.574). Result is stable across
   lag=1..3 and min_window=48..60.

Caveats — this is a smoke test and **not** production evidence:

- it does not separate shock-regime and non-shock windows;
- it does not test the `h=2+` trajectory realism criterion;
- it is the kind of grid that produced the rejected `fine_seasonal_resid_var_tc_roll42_l5`
  overfit earlier — promote only with shock-regime split and `h=12` realism checks.

Conclusion: a subset search across endogens is a legitimate research tool, but the
mandatory policy stays `var_policy` (RegimeMacroVARX_l1 for h=1, SeasonalVAR_CPI_F_NF_S
for h=12+). The grid script is kept as a research reference under `scripts/`, not
promoted to a registered model.

## Side study: DVMR (ДВМР) method recommendations (2026-06)

External reference: `_inbox/var_extracted/var/Прогнозирование основных
макроэкономических переменных с помощью модели векторной авторегрессии/ВАР_ДВМР/`.
Full read-through: [`docs/METODICHKA_ДВМР_analysis.md`](METODICHKA_ДВМР_analysis.md).

Authors: В.П. Жураковский, А.Н. Новопашина, Д.В. Гришин, Е.С. Булыга — ДВМР ГУ Банка
России. This is a self-evaluation document for the OPR (Основные направления развития)
submission, not a student project. It is treated here as the Bank-of-Russia standard
for a defensible regional VAR self-evaluation.

Their model: 4-quarterly endogenous (proxy GRP, CPI, USD/RUB, key rate) + 8 COVID-period
DUM, sample 2016Q3-2023Q1 (27 obs), lag=2 with zero-restrictions on lag 2, in-sample
fit, baseline comparison to ARMA(1,1).

The four residual tests they perform that we currently do not:

1. **Stability** (inverse roots inside the unit circle).
2. **Portmanteau** (multivariate autocorrelation, lags > VAR order).
3. **LM test for serial correlation** (Breusch-Godfrey style).
4. **Normality** (multivariate Jarque-Bera).

Plus the metric panel they report that we do not: **RMSE, MAPE, Theil**, alongside
the MAE we already have, and a mandatory **ARMA(1,1) baseline** as a sanity check.

Conclusions for our `var_policy`:

- The four residual tests are cheap to add (`statsmodels`-backed, ~50 lines total) and
  belong in `sirena/models/var_policy.py:diagnostics()`. They do not change forecasts
  but raise the defensibility bar of the model when reporting to a non-econometric
  audience.
- Theil's U is a useful normalized companion to MAE (range 0..1, "how much worse than
  naive"). Worth adding to the production backtest output.
- ARMA(1,1) baseline is the right **lower bound** for "is my VAR worth it". Worth
  running once and reporting; we do not need it on every rolling forecast.
- **Do not** copy their hard-coded COVID DUM approach. Our `var_policy` uses an
  automatic shock-regime switch (Huber regression when CPI abs or 12m vol exceed a
  threshold) — this is a structurally more robust alternative and is documented as
  such in section 4 above.
- Their MAPE is computed naively and gives nonsensical values (279% for GRP) because
  the denominator is near zero. If we add MAPE to our reports, **clip the denominator**
  (`max(|y|, 1.0)`) — the UGU-style `MAPFE` with `mom_y` in the denominator (see the
  grid_search_var study above) is the safer pattern.

Numerical reminder: DVMR has 27 quarterly observations. We have 196 monthly
observations = 49 quarters equivalent. Our out-of-sample rolling is **7× longer**,
which is why we can afford to be stricter on overfitting than DVMR can.
