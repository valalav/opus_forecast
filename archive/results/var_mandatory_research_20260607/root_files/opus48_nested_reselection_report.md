# Opus 4.8 — Nested Re-Selection Report: Seasonal-Residual VAR Family

**Agent:** opus48
**Date:** 2026-06-07
**Task file:** `experiments/var_sa_research/parallel_nested_reselection_task.md`
**Report path:** `experiments/var_sa_research/opus48_nested_reselection_report.md`
**Run dir:** `experiments/var_sa_research/runs/opus48_nested_main/`

## TL;DR

Under **honest nested re-selection** (hyperparameters chosen by an inner historical validation
*inside each cutoff*, never on the outer target), the seasonal-residual VAR family has **no
deployable signal**. It is worse than a plain VAR, worse than a trivial seasonal-naive baseline,
and far worse than production models (Huber, RidgeShockDummies) out of the previous selection
window. The honest method **never reproduces `roll42_l5`** and, on the very selection window,
returns h=1 MAE **0.405** instead of the overfit 0.223.

**Final status: `rejected`** (standalone and as an ensemble member). Methodology is leakage-free;
the failure is genuine lack of predictive value.

---

## 1. Was nested re-selection implemented correctly?

Yes. For each outer target date and horizon `h`:

1. `outer_cutoff = target_date − h months`; `train_full = official[index ≤ outer_cutoff]`.
2. **Inner validation:** the last `inner_k = 12` months (all ≤ outer_cutoff) are inner targets.
   For each candidate `(roll ∈ {24,30,36,42,48,60,72}) × (lags ∈ {1..6}) × (train_mode ∈
   {expanding, rolling120})`, and each inner target `t_in`:
   `inner_cutoff = t_in − h`; fit on `train_full[index ≤ inner_cutoff]`; predict `t_in`; error vs
   `actual(t_in)` (available because `t_in ≤ outer_cutoff`). Config score = inner MAE
   (min `min_valid = 6` valid inner points).
3. Pick lowest-inner-MAE config (deterministic tie-break `(mae, lags, roll, train_mode)`).
4. Refit on `train_full`, forecast the outer target.
5. Record chosen `(roll, lags, train_mode)` and inner score (`selection_log.csv`).

The outer target actual is never used in selection. The same train-only seasonal residualization
path audited earlier (proven leakage-free in `opus48_review_report.md`) is reused unchanged.

**Inner validation scheme:** rolling, horizon-matched, last 12 months, scored by inner MAE.

---

## 2. Out-of-selection performance (the decisive test)

**h=1 MAE by window** (84-config inner grid re-selected at every cutoff):

| Window | NestedSeasonalVAR | NestedPlainVAR | RW | SeasonalNaive | Ridge_ProdProxy_Roll24 | Huber | RidgeShock |
|---|---:|---:|---:|---:|---:|---:|---:|
| 2018_2019 | 0.380 | 0.335 | 0.325 | 0.426 | 0.201 | 0.327 | 0.341 |
| 2020_2021 | 0.294 | 0.264 | 0.313 | 0.298 | 0.321 | 0.326 | 0.308 |
| 2022_shock | 0.812 | 1.098 | 1.197 | 0.908 | 1.313 | 0.548 | 0.561 |
| 2023 | 0.685 | 0.535 | 0.647 | 0.410 | 0.638 | 0.390 | 0.342 |
| 2024_2025q1 | 0.465 | 0.397 | 0.491 | 0.379 | 0.506 | 0.323 | 0.302 |
| **selection 2025-04..2026-03** | **0.405** | 0.376 | 0.592 | 0.388 | 0.248 | 0.329 | 0.297 |
| **AGG nonsel/nonshock** | **0.418** | 0.357 | 0.406 | 0.373 | 0.424 | 0.336 | 0.323 |
| **AGG all** | 0.464 | 0.449 | 0.524 | 0.440 | 0.548 | 0.361 | 0.349 |

**Multi-horizon (AGG nonsel/nonshock MAE):**

| Horizon | NestedSeasonalVAR | NestedPlainVAR | RW | SeasonalNaive |
|---|---:|---:|---:|---:|
| h=1 | 0.418 | 0.357 | 0.406 | 0.373 |
| h=2 | 0.443 | 0.356 | 0.417 | 0.373 |
| h=12 | 0.442 | 0.391 | 0.535 | 0.373 |

---

## 3. Did it beat the baselines? (direct answers)

- **vs simple baselines, out of selection:** NO. NestedSeasonalVAR (h=1 nonsel 0.418) is beaten by
  NestedPlainVAR (0.357), SeasonalNaive (0.373), and even RW (0.406). The result holds on h=2 and
  h=12 too. The seasonal-residual transform is a *net negative*: removing it (NestedPlainVAR)
  improves every aggregate.
- **vs project baselines (h=1):** NO. Huber (0.336) and RidgeShockDummies (0.323) beat it
  decisively out of selection; only the heavily-tuned `Ridge_ProdProxy_Roll24` is volatile (great
  on its 2018/selection windows, catastrophic in the 2022 shock at 1.313).
- **On the previous selection window:** honest nested selection yields **0.405**, vs the overfit
  single-window grid number **0.223**. This is the cleanest possible demonstration that the 0.223
  was selection-window overfitting.

---

## 4. Are the selected hyperparameters stable?

No — they are **noisy and never converge on `roll42_l5`**:

- Across 99 h=1 cutoffs: `roll42` chosen 15 times; **`roll42 AND lags5` chosen 0 times.**
- Inner validation gravitates to a *different* region: `roll72` (45/99) and `lags1` (41/99).
- **29 distinct `(roll, lags, train_mode)` configs** are selected across 99 cutoffs.
- Even within the 12-month selection window the choice churns: roll `72→72→72→72→72→72→30→60→30→30→30→48`,
  lags `4→…→2→4→2→2→2→3`, flipping train_mode repeatedly.

A genuine signal would produce stable, repeatable hyperparameters. This is the footprint of fitting
short-window noise.

---

## 5. Is there evidence of deployable signal?

No.
- It does not beat a plain VAR (its own ablation) → the seasonal mechanism adds nothing.
- It does not beat a one-line seasonal-naive or random walk out of selection.
- It is materially worse than existing production h=1 leaders.
- Its only "win" (0.223) is non-reproducible under honest selection.
- As an ensemble member it would inject correlated VAR error without diversifying benefit
  (it tracks plain VAR but worse), so it is not an ensemble candidate either.

---

## 6. Final status

**`rejected`** — for standalone production and for ensemble use.

This confirms and strengthens the earlier audit (`opus48_review_report.md`): the
`fine_seasonal_resid_var_tc_roll42_l5` h=1 = 0.223 was single-window grid-selection overfit.
The underlying *idea* (train-only seasonal residual VAR) is leakage-free and methodologically
clean, but provides no out-of-sample value once hyperparameters are chosen honestly.

---

## 7. Exact commands run

```bash
cd /home/valalav/_projects/sirena-kbr
python3 -m py_compile experiments/var_sa_research/run_var_sa_backtests.py
python3 -m py_compile experiments/var_sa_research/opus48_nested_reselection.py
python3 experiments/var_sa_research/opus48_nested_reselection.py \
    --run-name opus48_nested_main --horizons 1,2,12 --inner-k 12 --min-valid 6 --project-baselines
```

Artifacts (only under `runs/opus48_nested_*`, nothing else overwritten):
- `runs/opus48_nested_main/config.json`
- `runs/opus48_nested_main/predictions.csv`  (all models × windows × horizons, per-target)
- `runs/opus48_nested_main/metrics.csv`       (per window/horizon + AGG rows)
- `runs/opus48_nested_main/selection_log.csv` (chosen roll/lags/train_mode + inner score per cutoff)
- `runs/opus48_nested_main/notes.md`
- `experiments/var_sa_research/opus48_nested_reselection.py` (research-only runner)

---

## 8. What remains unchecked

- **Inner scheme sensitivity:** only `inner_k=12`, rolling, MAE-selection was used. A longer inner
  window or expanding inner validation could shift selections, but given the family loses to plain
  VAR and seasonal-naive everywhere, a different inner scheme is very unlikely to rescue it.
- **Component bottom-up seasonal-residual variant** was not nested-selected (only total/components
  CPI VAR). The earlier fixed-config evidence already showed bottom-up did not help; not re-run here.
- **BVAR seasonal-residual** under nested selection was not run (slow; the VAR family already fails,
  and the BVAR best was also single-window selected).
- **Project baselines** were run only for h=1 (runtime). h=2/h=12 project comparisons rely on the
  archived backtests, which are themselves tuned on the 2025 window.
- **SA / revised-history data** intentionally excluded (revision leakage), per task rules.
