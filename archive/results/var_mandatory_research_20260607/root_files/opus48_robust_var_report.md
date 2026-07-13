# Opus 4.8 — Robust / Outlier-Aware VAR Report

**Agent:** opus48
**Date:** 2026-06-07
**Task file:** `experiments/var_sa_research/robust_outlier_var_task.md`
**Report path:** `experiments/var_sa_research/opus48_robust_var_report.md`
**Run dir:** `experiments/var_sa_research/runs/opus48_robust_var_main/`

## TL;DR

Proper outlier/shock handling **does** improve the mandatory VAR on the primary horizon — but only
the *right* kind. A **regime-aware Macro VARX** (`RegimeMacroVARX_l1`: VARX with USD/Ruonia/Ki in
normal regimes, robust Huber VAR with no macro in shock regimes, regime decided from cutoff-only CPI
volatility) beats `PlainVAR_BIC` on h=1 across the board — all windows, out-of-selection, non-shock,
**and** the 2022 shock — while staying leakage-free. A **robust (winsorized/MAD) FAVAR** gives the
best 2022-shock accuracy and is horizon-stable. Conversely, **hard-coded 2022 intervention dummies
make things worse**, and winsorizing the MoM series does nothing.

**Final status: `recommended robust mandatory VAR`** — `RegimeMacroVARX_l1` for h=1 (the main KPI),
with `RobustFAVAR_f2_l1` as the horizon-stable shock-robust alternative. It still does not beat
production ML (RidgeShockDummies ≈ 0.349 all / Huber); this is the strongest *VAR-family* option.

---

## 1. Methods tested (all VAR-family, official data, leakage-free)

| Candidate | Outlier/shock mechanism | Train-only? |
|---|---|---|
| HuberVAR_l1 | equation-by-equation HuberRegressor (robust loss) | yes, future-blind |
| WinsorizedVAR_l1 | clip each series at train-only 2.5/97.5% before OLS VAR(1) | yes |
| PulseOutlierVAR_l1 | flag train months with robust \|z\|>3.5 (MAD) of VAR(1) resid; pulse dummies; future pulse=0 | yes |
| InterventionVAR_l1 | OLS VAR(1) + **pre-declared** dummies: sanctions 2022-03..08, COVID 2020-04..05, July tariff | declared calendar |
| RegimeMacroVARX_l1 | normal→VARX(USD,Ruonia,Ki, exog=last obs); shock→Huber VAR (no macro); regime from cutoff CPI vol | yes |
| RobustFAVAR_f2_l1 | train-only winsorized + MAD-standardized PCA (2 factors) + VAR(1) | yes |

Baselines: `PlainVAR_BIC` (= plain VAR(1) tc), `FAVAR_f2_l1`, `VARX_last_exog_l1`, `ArchivedBVAR`,
`RW`, `SeasonalNaive`. Windows: 2018_2019, 2020_2021, 2022_shock, 2023, 2024_2025q1,
selection_2025-04_2026-03. Horizons 1 (primary), 2, 12.

---

## 2. h=1 metrics (primary)

| model | 2022_shock | AGG_all | out-of-selection | oos_non-shock | KPI(all) |
|---|---:|---:|---:|---:|---:|
| **RegimeMacroVARX_l1** | 0.831 | **0.379** | **0.376** | **0.304** | 24 |
| RobustFAVAR_f2_l1 | **0.783** | 0.380 | 0.384 | 0.320 | 25 |
| FAVAR_f2_l1 | 0.839 | 0.387 | 0.388 | 0.315 | 23 |
| HuberVAR_l1 | 0.853 | 0.397 | 0.399 | 0.326 | 25 |
| PulseOutlierVAR_l1 | 0.871 | 0.400 | 0.403 | 0.328 | 25 |
| **PlainVAR_BIC** (baseline) | 0.918 | 0.401 | 0.401 | 0.318 | 28 |
| VARX_last_exog_l1 | 1.025 | 0.405 | 0.408 | 0.309 | 21 |
| WinsorizedVAR_l1 | 0.952 | 0.408 | 0.410 | 0.323 | — |
| ArchivedBVAR | 1.039 | 0.477 | 0.472 | 0.382 | 34 |
| SeasonalNaive | 0.908 | 0.440 | 0.447 | 0.373 | 33 |
| InterventionVAR_l1 | **1.711** | 0.507 | 0.519 | 0.328 | — |

`RegimeMacroVARX_l1` vs `PlainVAR_BIC`: h=1 all −5.4% (0.379 vs 0.401), out-of-selection −6.2%,
non-shock −4.6% (0.304 vs 0.318), 2022 shock −9.5% (0.831 vs 0.918), KPI 24 vs 28. It improves
**both** regimes, so the gain is not a shock-only artifact.

---

## 3. h=2 / h=12 metrics (best candidates)

| model | h2 all | h2 oos_nonshock | h12 all | h12 oos_nonshock |
|---|---:|---:|---:|---:|
| RegimeMacroVARX_l1 | 0.476 | 0.351 | **0.547 (worse)** | 0.433 |
| RobustFAVAR_f2_l1 | **0.470** | 0.367 | 0.483 | 0.387 |
| PulseOutlierVAR_l1 | 0.476 | 0.345 | 0.471 | 0.375 |
| PlainVAR_BIC | 0.485 | 0.338 | 0.483 | 0.387 |
| HuberVAR_l1 | 0.477 | 0.345 | **7.259 (explodes)** | 9.335 |

`RegimeMacroVARX_l1` is competitive on h=2 but **degrades at h=12** (recursive VARX with constant
exog drifts). `RobustFAVAR_f2_l1` is the only robust model that is both shock-best and horizon-stable.
`HuberVAR_l1` is unsafe beyond short horizons (explosive recursion).

---

## 4. Recommended robust VAR

- **Primary (h=1 KPI): `RegimeMacroVARX_l1`** — recommended *robust mandatory VAR* for h=1. It
  **replaces** `PlainVAR_BIC` for h=1 (and is fine for h=2), but **not** for h=12, where
  `PlainVAR_BIC` / `RobustFAVAR_f2_l1` / `PulseOutlierVAR_l1` should be used.
- **Horizon-stable shock-robust alternative: `RobustFAVAR_f2_l1`** — best 2022-shock MAE (0.783),
  ≈ baseline on h=2/h=12, single config across horizons. Choose this if one model must serve all
  horizons.

Both are VAR-family (lagged-endogenous dynamics; VARX and FAVAR are standard VAR extensions).

---

## 5. Outlier dates / dummies / regimes

- **Pulse detection** (robust z>3.5): consistently flags the historical KBR shocks — 2010-09, 2011-01,
  2012-07, 2013-07, 2014-10/12, 2015-01/04/07, 2016-12 (each flagged in ~all cutoffs), i.e. the
  2014–2015 currency crisis and July tariff spikes. ~11–17 pulses per cutoff (`outlier_log.csv`).
- **Declared dummies** (InterventionVAR): sanctions 2022-03..08, COVID 2020-04..05, July tariff.
  Result: net harmful (see §6).
- **Regime split** (`selection_log.csv`): shock regime fires 10/12 months in 2022 and 8/12 in 2023,
  2–4/12 in calm years — i.e. the cutoff-only volatility rule correctly localizes the shocks without
  ever being told the 2022 calendar.

---

## 6. Why some "robust" ideas failed (blunt)

- **InterventionVAR (declared 2022/COVID/July dummies): worse everywhere, catastrophic on the shock
  (1.71).** Hard-coding the shock calendar backfires — the dummy coefficients are estimated from a
  handful of volatile months and overcorrect the forecast. Hard-coding 2022 does **not** even help.
- **WinsorizedVAR: no gain** (0.408 vs 0.401). Clipping total/component MoM removes genuine signal
  along with outliers.
- **HuberVAR: small h=1 gain, explodes at h=12** (7.26). Unrestricted robust equations admit
  explosive VAR roots; usable only at short horizons (and inside the regime model as the shock leg,
  where it is only ever used for short steps).
- **PulseOutlierVAR: only marginal** (0.400). Removing past-outlier leverage barely changes the
  one-step forecast because VAR(1) on these series is already near a random walk.

The honest lesson: the shock improvement that *generalizes* comes from **regime-conditional model
choice driven by cutoff-observable volatility** (RegimeMacroVARX) and **robust factor preprocessing**
(RobustFAVAR), **not** from calendar dummies that encode 2022.

---

## 7. Leakage audit

All candidates leakage-free. Numerical probe: corrupting every row at/after the target date (`+999`
on all columns) leaves each model's prediction **bit-identical** across normal (2019-07), shock
(2022-04), and recent (2025-09) cutoffs — see `leakage_checks.csv` (`leakage_free = True` for all 7
models incl. PlainVAR_BIC). Mechanisms: regime from cutoff CPI only; VARX exog held at last observed
value (never future actuals); pulse dummies train-only with future pulse=0; declared dummies are
deterministic date functions (they encode calendar knowledge but do not read the target actual — so
leakage-free yet still inaccurate). No SA/revised data used.

---

## 8. Acceptance check (vs task criteria) for RegimeMacroVARX_l1

- leakage-free — ✔
- beats `PlainVAR_BIC` on h=1 all and out-of-selection — ✔ (0.379/0.376 vs 0.401)
- does not sacrifice non-shock — ✔ (improves it, 0.304 vs 0.318)
- materially improves 2022 shock / KPI — ✔ (0.831 vs 0.918; KPI 24 vs 28)
- beats archived BVAR — ✔ (0.477)
- explainable VAR-family — ✔ (regime-switched VARX / robust VAR)
- caveat: degrades at h=12 → recommendation is horizon-scoped (h=1/h=2).

---

## 9. Exact commands run

```bash
cd /home/valalav/_projects/sirena-kbr
python3 -m py_compile experiments/var_sa_research/opus48_robust_var.py
python3 experiments/var_sa_research/opus48_robust_var.py --run-name opus48_robust_var_main --horizons 1,2,12
# leakage_checks.csv generated by inline +999 future-corruption probe (see report section 7)
```

Artifacts (only under `runs/opus48_robust_var_*`, nothing else overwritten):
`config.json, metrics.csv, predictions.csv, comparison.csv, outlier_log.csv, selection_log.csv,
leakage_checks.csv, notes.md, opus48_robust_var.py`.

---

## 10. Final status and tone

**`recommended robust mandatory VAR`** = `RegimeMacroVARX_l1` (h=1 primary; replaces PlainVAR_BIC on
the main KPI), with `RobustFAVAR_f2_l1` as the horizon-stable shock-robust alternative.

Honest framing: this is the strongest *VAR-family* benchmark/control model with proper shock handling;
the improvement is real and leakage-free but modest (~5% on h=1), it degrades at h=12, and it remains
behind production ML forecasters. It is offered as the mandatory VAR / interpretable macro-component
control, not as the main production forecaster.

## 11. What remains unchecked

- Student-t / heavy-tailed BVAR (Direction 5) was not implemented (kept to faster robust-loss and
  regime approaches); given Huber/pulse gains were modest, a t-error BVAR is unlikely to beat the
  regime model but is not ruled out.
- Regime rule thresholds (|CPI|≥1.0, std≥0.55) were declared, not tuned; a small inner-validation
  sweep could refine them but risks the same overfitting flagged in earlier reports.
- h=12 used a rolling protocol for cross-window comparability, not the production fixed-cutoff
  trajectory backtest.
- No ensemble of RegimeMacroVARX + RobustFAVAR was fit (would leave the pure-VAR framing).
