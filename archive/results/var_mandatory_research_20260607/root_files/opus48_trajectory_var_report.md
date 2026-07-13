# Opus 4.8 — Parsimonious Multi-Horizon VAR / Trajectory Report

**Agent:** opus48
**Date:** 2026-06-07
**Task file:** `experiments/var_sa_research/parsimonious_trajectory_var_task.md`
**Report path:** `experiments/var_sa_research/opus48_trajectory_var_report.md`
**Run dir:** `experiments/var_sa_research/runs/opus48_trajectory_var_main/`

## TL;DR

"More variables" is **not** better. Macro (USD/Ki/Ruonia) helps **only h=1** (and only `Ki`); it is
neutral on h=2 and **catastrophic on h=12** (constant-exog propagation drifts the path). No single VARX
serves all horizons. The only model that gives a **realistic deterministic 12-month trajectory** —
correct seasonal shape, realistic volatility, no explosion — is a **seasonal-deterministic VAR**
(`SeasonalVAR_CPI_F_NF_S`: expanding train-only month means + VAR(1) on the residual + deterministic
seasonal reconstruction), which is also the **best h=12 point model**. Plain VAR and macro VARX, despite
similar average MAE, produce economically **wrong-shaped (anti-seasonal), nearly flat** paths.

**Final status: `recommended horizon-specific VAR policy`** (with `SeasonalVAR_CPI_F_NF_S` as the h=12 /
trajectory model and the single-model fallback). All deterministic; no random noise; leakage-free.

---

## 1. Variable subsets tested

Endogenous sets: `CPI`; `CPI+Food+NonFood+Services`; `Food+NonFood+Services` (CPI reconstructed by
official weights); `CPI+Food`; `CPI+NonFood`; `CPI+Services`; `CPI+Food+NonFood`; `CPI+Food+Services`;
`CPI+NonFood+Services`.
Macro/exog sets: `none, USD, Ki_i, Ruonia, USD+Ki, USD+Ruonia, Ki+Ruonia, USD+Ki+Ruonia` (cutoff-safe
exogenous, last-observed path for Part A). Grid = 65 specs × 3 horizons × 6 windows.

---

## 2. Parsimony results (Part A, AGG out-of-selection non-shock MAE)

**Best per horizon:**

| Horizon | Best specs | MAE |
|---|---|---:|
| h=1 | `CPI_F_NF+Ki`, `CPI_NF_S+Ki`, `CPI_F_S+Ki` | 0.299–0.300 |
| h=2 | `CPI_NF+none`, `CPI_F_NF+none` | 0.336–0.337 |
| h=12 | `SeasonalVAR_CPI`, `SeasonalVAR_CPI_F_NF_S`, `CPI+none` | 0.371–0.384 |

**Macro effect on `CPI_F_NF_S` (the canonical 4-var set):**

| macro | h1 | h2 | h12 |
|---|---:|---:|---:|
| none | 0.318 | 0.338 | 0.387 |
| **Ki** | **0.302** | 0.351 | 0.605 |
| USD | 0.342 | 0.396 | 0.926 |
| Ruonia | 0.328 | 0.369 | 0.442 |
| USD+Ki+Ruonia | 0.309 | 0.375 | 0.703 |

**Answers to the core questions:**
1. Best subset per horizon — h=1: CPI + 2 components + **Ki**; h=2: small component VAR, **no macro**;
   h=12: **seasonal-deterministic / univariate CPI**, no macro.
2. Does all-macro hurt vs a smaller subset? **Yes.** `USD+Ki+Ruonia` is worse than `Ki`-only on h=1 and
   far worse on h=12. USD is the single most damaging macro on the long horizon.
3. Keep the h=1 macro gain without degrading h=12? **Not in one VARX** — the same last-obs exog that
   helps h=1 destroys h=12 (0.387→0.605 with Ki). Requires a horizon-specific policy.
4/5: see trajectory section and recommendation.

---

## 3. Exogenous path assumptions (Part B, CPI_F_NF_S + USD_Ki_Ruonia, h=12 MAE)

| exog path | MAE | RMSE | KPI |
|---|---:|---:|---:|
| AR(1) forecast | 0.508 | 0.790 | 38 |
| recent-12m mean | 0.547 | 0.830 | 40 |
| **last observed (constant)** | **0.780** | 1.454 | 48 |

Holding exog constant for 12 months is the worst assumption — it propagates the last macro level and
drifts the path. AR(1) mean-reversion roughly halves the damage, but a **macro-free** seasonal VAR still
beats all macro exog-path variants at h=12. This is exactly why the earlier `RegimeMacroVARX_l1`
(last-obs exog) degraded at h=12.

---

## 4. Trajectory realism (Part C, deterministic fixed-cutoff 12-month paths)

Cutoffs: 2019-12, 2021-12 (→2022 shock), 2024-11 (production-style), 2025-03.

| model | traj_MAE 2024-11 | seasonal_corr (range) | vol_ratio (range) | explosive |
|---|---:|---:|---:|:--:|
| **SeasonalVAR_CPI_F_NF_S** | **0.326** | **0.95–0.99** | 0.33–0.68 | no |
| SeasonalVAR_CPI | 0.354 | 0.96–1.00 | 0.37–0.69 | no |
| SeasonalNaive | 0.329 | 1.00 | 0.32–0.80 | no |
| VAR_CPI_only | 0.396 | **−0.69 … 0.44** | 0.04–0.13 | no |
| PlainVAR_CPI_F_NF_S | 0.410 | **−0.63 … 0.32** | 0.02–0.14 | no |
| VARX_USDKiRuonia_lastobs | 1.149 | **−0.65 … 0.58** | 0.07–0.30 | no |
| RW | 1.017 | NaN (flat) | 0.00 | no |

traj_MAE by cutoff for the winner `SeasonalVAR_CPI_F_NF_S`: **0.252 / 0.969 / 0.326 / 0.348** — best at
3 of 4 cutoffs (beating even the seasonal-naive), with the 2021-12→2022 shock window unforecastable by
all (seasonal-naive 0.908 marginally best there).

Diagnostics (full set in `trajectory_metrics.csv`): path_std, vol_ratio (vs trailing-36m std),
sign_changes, max month-to-month jump, seasonal amplitude, **seasonal_corr** (corr of the path's
month pattern with historical month-of-year means), flatness, n_identical, explosive flag.

**Rejected unrealistic paths (blunt):**
- **Plain VAR / VAR_CPI_only**: *anti-seasonal* (negative seasonal_corr in 2019/2021) and **too flat**
  (vol_ratio 0.02–0.14, i.e. ~5–15% of historical volatility). Their decent average MAE hides an
  economically wrong trajectory.
- **VARX with last-obs macro**: wrong seasonal shape (corr ≈ −0.6) and the worst traj_MAE (1.15) —
  macro propagation pushes the path the wrong way.
- **RW**: a dead-flat horizontal line (vol_ratio 0).
- No explosive paths were produced by any candidate (all VAR roots stable at lag 1).

The seasonal shape that makes a 12-month path realistic comes **only** from explicit deterministic
seasonality, not from adding endogenous components or macro variables.

---

## 5. Final recommendation — horizon-specific VAR-family policy

| Horizon | Recommended VAR | oos non-shock MAE | Rationale |
|---|---|---:|---|
| **h=1** | `CPI_F_NF+Ki` VARX (or robust `RegimeMacroVARX_l1`) | 0.299 (0.304) | Ki is the only helpful macro; 3-var component set |
| **h=2** | `CPI_NF` / `CPI_F_NF` VAR, no macro | 0.336 | macro adds nothing at h=2 |
| **h=12 / trajectory** | **`SeasonalVAR_CPI_F_NF_S`** | 0.374 | best h=12 point error AND only realistic deterministic path |

**Single-model fallback** (if one stable model is required across horizons):
`SeasonalVAR_CPI_F_NF_S` — h=1 0.347, h=2 0.347, h=12 0.374, and the only model with a realistic
seasonal trajectory. It trades ~0.04 of h=1 MAE for correct long-horizon behaviour.

`SeasonalVAR` is leakage-free and parsimonious: it uses **expanding** train-only month-of-year means
(no roll-window hyperparameter), so it is **not** the previously-rejected tuned `roll42_l5` — there is
nothing to overfit.

---

## 6. Leakage audit

Leakage-free. `+999` future-corruption probe (corrupt every column for all rows ≥ target) leaves
predictions bit-identical for SeasonalVAR and the VARX/FAVAR candidates (`leakage_checks.csv`).
Mechanisms: seasonal means and VAR coefficients from train-only data; exog future paths deterministic
(last-obs / AR1 fit on train / recent mean), never future actual macro. No SA/revised data used. All
point forecasts deterministic for fixed config (no random noise anywhere).

---

## 7. Exact commands run

```bash
cd /home/valalav/_projects/sirena-kbr
python3 -m py_compile experiments/var_sa_research/opus48_trajectory_var.py
python3 experiments/var_sa_research/opus48_trajectory_var.py --run-name opus48_trajectory_var_main
```

Artifacts (only under `runs/opus48_trajectory_var_*`, nothing else overwritten):
`config.json, metrics.csv, predictions.csv, comparison.csv, selection_log.csv, exog_path_metrics.csv,
trajectory_metrics.csv, leakage_checks.csv, notes.md, trajectory_{2019-12,2021-12,2024-11,2025-03}.png,
opus48_trajectory_var.py`.

---

## 8. Final status

**`recommended horizon-specific VAR policy`**:
- h=1 → `CPI_F_NF+Ki` VARX (or `RegimeMacroVARX_l1` for shock-robustness),
- h=2 → parsimonious component VAR (no macro),
- h=12 / trajectory → **`SeasonalVAR_CPI_F_NF_S`** (also the recommended single-model fallback).

Honest framing: this is the strongest *VAR-family* multi-horizon policy — parsimonious, leakage-free,
deterministic, with realistic 12-month trajectories. It is the mandatory VAR benchmark / interpretable
control; production ML models remain better on raw h=1 point accuracy.

## 9. What remains unchecked

- Lag orders beyond 1 were not swept in the parsimony grid (BIC consistently selects lag 1 for these
  series, per the mandatory-VAR report; higher lags were not expected to help and were skipped for
  combinatorial economy).
- Tariff/admin-price seasonal dummies (vs the expanding-mean seasonality used here) were not separately
  tested; the expanding month-mean reconstruction already attains seasonal_corr ≈ 0.97–0.99.
- BVARX heavy-tail/shrinkage variants of the seasonal model were not run; the OLS seasonal VAR already
  wins the trajectory and is the most transparent.
- h=12 used a rolling protocol for point MAE comparability across windows; trajectory realism used true
  fixed-cutoff 12-step paths (Part C).
