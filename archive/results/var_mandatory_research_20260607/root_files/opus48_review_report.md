# Opus 4.8 — Independent VAR/SA Review Report

**Audit target:** `fine_seasonal_resid_var_tc_roll42_l5` (Codex CLI "best" candidate)
**Auditor:** Claude Opus 4.8, independent re-run
**Date:** 2026-06-07
**Task file:** `experiments/var_sa_research/opus48_review_task.md`
**Report path:** `experiments/var_sa_research/opus48_review_report.md`
**Verdict (one line):** Metrics reproduced exactly, **no leakage found**, but the headline
h=1 MAE 0.223 is **window-specific overfit from grid selection** — not a robust improvement.
**Final status: `experimental only` (leaning toward `rejected` as a standalone production model).**

---

## 1. Was the Codex CLI claim reproduced?

**Yes, bit-for-bit.** Two independent paths confirm it:

- The saved `iter03_seasonal_fine/metrics.csv` already contains the exact claimed numbers.
- A fresh re-run (`opus48_reproduce_iter03`, identical preset/seed) reproduced **every**
  candidate's MAE/RMSE/KPI with `max |Δ| = 0.0` (VAR families *and* BVAR families — the BVAR
  seed is deterministic).

| Horizon | Claimed MAE | Reproduced MAE | KPI Viol | Match |
|--------:|------------:|---------------:|---------:|:-----:|
| h=1 | 0.223115 | 0.223115 | 2 | ✅ |
| h=2 | 0.308349 | 0.308349 | 2 | ✅ |
| h=12 | 0.413011 | 0.413011 | 4 | ✅ |

The runner saved with iter03 is byte-identical (md5) to the current `run_var_sa_backtests.py`,
so the artifacts match the code. **The numbers are real arithmetic.**

---

## 2. Leakage / methodology audit

Verdict: **no future leakage in the seasonal-residual VAR path.** Confirmed by code review and
a numerical probe.

**Code review** (`run_var_sa_backtests.py`):
- `evaluate_candidates`: `train = source[source.index <= cutoff]`, `cutoff = target_date − horizon`. ✔
- `_seasonal_means`: trailing-window month-of-year means computed from `train` only; target
  month's actual is never in `train` (train ends at `cutoff`). ✔
- `_seasonal_residual_frame`: subtracts train-only means from train rows. ✔
- `forecast_var` on residuals: forecasts from `data.values[-k_ar:]` of the residualized train. ✔
- `_future_seasonal`: target-month factor taken from the train-only means. ✔
- Reconstruction: `residual_forecast + target_month_seasonal_mean`. ✔

**Numerical probe** (`opus48_robustness.py`): corrupting **every** row at/after the target date
(`+999` on all columns) left the prediction **bit-identical** (`0.228626868... == 0.228626868...`).
Reconstructed target month equals the intended target for h=1/h=2/h=12 (cutoff+horizon check
passes), so the multi-step target index is correct.

Caveat (already documented by the original author): SA-data candidates use revised full-history
SA series → revision leakage risk. That does **not** affect `roll42_l5`, which is official-data only.

---

## 3. Robustness without retuning (the decisive test)

The config was selected by a fine grid scored on **one** 12-month window (`2025-04..2026-03`,
the archive h=1 window). I froze it and evaluated the identical config on 7 other 12-month
windows with the same cutoff discipline.

**Frozen `roll42_l5`, h=1 MAE by window:**

| Window | h=1 MAE | h=2 MAE | h=12 MAE | h1 KPI |
|---|---:|---:|---:|---:|
| **selection 2025-04..2026-03** | **0.223** | 0.308 | 0.413 | 2 |
| 2020 | 0.295 | 0.221 | 0.247 | 1 |
| 2021 | 0.295 | 0.330 | 0.291 | 1 |
| 2019 | 0.382 | 0.359 | 0.393 | 4 |
| 2018 | 0.464 | 0.599 | 0.420 | 4 |
| 2024 | 0.562 | 0.664 | 0.588 | 6 |
| 2023 | 0.614 | 0.617 | 0.543 | 5 |
| 2022 (shock) | 0.693 | 0.872 | 0.947 | 3 |
| **mean of 6 non-selection non-shock** | **0.435** | 0.465 | 0.414 | — |

The selection-window h=1 MAE (0.223) is **~half** the average over the other non-shock windows
(0.435). The "best ever" window is precisely the one used to pick the config. This is the
textbook footprint of window-specific overfitting.

---

## 4. Comparison vs baselines (same windows, same cutoffs)

Because archived production metrics only exist for the selection window, I computed honest
baselines myself on **every** window (`SeasonalNaive_roll42`, `RW`, `RW12`, `PlainVAR_l5`).

**h=1 MAE, roll42_l5 vs self-computed baselines:**

| Window | roll42_l5 | PlainVAR_l5 | RW | SeasonalNaive |
|---|---:|---:|---:|---:|
| selection 2025 | **0.223** | 0.329 | 0.593 | 0.342 |
| 2019 | 0.382 | **0.286** | 0.249 | 0.306 |
| 2020 | 0.295 | 0.318 | 0.299 | **0.273** |
| 2021 | **0.295** | 0.239 | 0.328 | 0.411 |
| 2023 | 0.614 | 0.624 | 0.647 | **0.621** |
| 2024 | 0.562 | **0.359** | 0.522 | 0.648 |

roll42_l5 **only** dominates on the selection window. Off-selection it is frequently beaten by a
plain VAR(5) or even a random walk (2019, 2024).

**Archived production baselines on the selection window** (for reference — these were also tuned
on this window, so not a clean out-of-sample comparison either):

| Model | h=1 | h=2 | h=12 |
|---|---:|---:|---:|
| Ridge_ProdProxy_Roll24 | 0.248 | 0.279 | 0.304 |
| Ridge_Shock | 0.304 | 0.362 | 0.382 |
| Huber | 0.335 | 0.343 | 0.436 |
| Subcomp_Multi | 0.386 | 0.352 | 0.331 |
| BVAR (archived) | 0.444 | 0.634 | 0.480 |
| **roll42_l5 (selection)** | **0.223** | 0.308 | 0.413 |

roll42_l5 beats `Ridge_ProdProxy_Roll24` on h=1 *only on the selection window*; it is worse on
h=2 and clearly worse on h=12 everywhere. Its off-selection h=1 mean (0.435) is worse than every
production model in the table.

---

## 5. Plateau vs lucky point

**Neighborhood grid `roll∈{30,36,42,48,60} × lags∈{1..6}`, h=1 MAE on the selection window:**

| roll\lags | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---:|---:|---:|---:|---:|---:|
| 30 | 0.282 | 0.271 | 0.311 | 0.332 | 0.343 | 0.342 |
| 36 | 0.335 | 0.336 | 0.346 | 0.326 | 0.359 | 0.389 |
| **42** | 0.269 | 0.238 | 0.227 | 0.237 | **0.223** | 0.255 |
| 48 | 0.335 | 0.320 | 0.299 | 0.289 | 0.336 | 0.351 |
| 60 | 0.358 | 0.327 | 0.311 | 0.280 | 0.314 | 0.347 |

`roll42` is an **isolated trough**: its immediate neighbor `roll36` is the *worst* row (0.33–0.39),
while `roll42` is the *best* row (0.22–0.27). A minimum flanked by the maximum is not a plateau —
it is fitting noise.

- roll42_l5 rank on the **selection** grid: **#1 of 30**.
- roll42_l5 rank averaged over the **other** windows: **#17 of 30** (below median, avg MAE 0.435).
- Configs that actually generalize: `roll60_l1`, `roll48_l1` — a different region of the grid.

**Conclusion: `roll42_l5` is a lucky isolated point, not a stable plateau.**

---

## 6. Usefulness assessment

- **Standalone h=1 specialist?** No. Out-of-window h=1 ≈ 0.435 is worse than current production
  h=1 leaders (Ridge_ProdProxy_Roll24 0.248, Ridge_Shock 0.304, even baseline Ridge 0.323).
- **Ensemble candidate?** Weak. It is not a leakage-free reliable signal; off-window it tracks a
  plain VAR/RW and adds correlated error during shocks (2022/2023/2024 MAE 0.56–0.69).
- **Experimental only?** Yes — the *idea* (train-only seasonal residualization before VAR) is
  methodologically sound and clean, and is a reasonable research direction. The *specific tuned
  config* is not.
- **Rejected for production:** Yes, as a standalone model — due to grid-selection overfit and weak
  out-of-window / h=12 performance, not due to leakage.

---

## 7. Final status

**`experimental only`** for the seasonal-residual-VAR *family*; **`rejected`** for the *specific*
`fine_seasonal_resid_var_tc_roll42_l5` config as a standalone production model.

The Codex CLI's own iteration log already reached the cautious "experimental, not production-ready"
conclusion and explicitly flagged the single-window selection risk. This audit **confirms the
numbers, clears the leakage concern, and upgrades the warning to a finding**: the 0.223 is an
overfit artifact, demonstrated by 7 out-of-selection windows and a neighborhood map.

---

## 8. Exact commands used

```bash
cd /home/valalav/_projects/sirena-kbr

# Compile + reproduce iter03 (determinism / claim verification)
python3 -m py_compile experiments/var_sa_research/run_var_sa_backtests.py
python3 experiments/var_sa_research/run_var_sa_backtests.py \
    --run-name opus48_reproduce_iter03 --preset seasonal_fine --n-draws 160 --seed 20260607

# Independent robustness audit (frozen config, multi-window, baselines, neighborhood, leakage probe)
python3 -m py_compile experiments/var_sa_research/opus48_robustness.py
python3 experiments/var_sa_research/opus48_robustness.py
```

Artifacts produced (all under `runs/opus48_*`, nothing overwritten):
- `runs/opus48_reproduce_iter03/` — exact re-run of iter03 (config/metrics/predictions/comparison/notes).
- `runs/opus48_robustness/` — `robustness_by_window.csv`, `neighborhood_h1.csv`, `metrics.csv`,
  `predictions.csv`, `config.json`, `notes.md`.
- `experiments/var_sa_research/opus48_robustness.py` — research-only runner extension.

---

## 9. Remaining unchecked items

- BVAR seasonal-residual variants were reproduced but not separately robustness-tested across
  windows (only the VAR `roll42_l5` was frozen and stress-tested; the BVAR best `roll42_l2_lam0p05`
  is very likely subject to the same single-window selection bias).
- No real-time SA vintage files were available; SA-based candidates remain untrustworthy by the
  original author's own note (revision leakage) and were out of scope here.
- Did not test the seasonal-residual *idea* under a proper nested/rolling re-selection (i.e.
  re-pick roll/lags inside each cutoff). That would be the fair way to estimate the family's true
  deployable value and is the recommended next step if anyone wants to pursue it.
