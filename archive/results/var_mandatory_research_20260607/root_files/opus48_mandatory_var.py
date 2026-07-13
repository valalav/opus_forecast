#!/usr/bin/env python3
"""
Opus 4.8 — Mandatory VAR-family benchmark search.

Goal: find the BEST methodologically defensible VAR-family model (not to re-prove that
roll42_l5 is overfit). All variants are official-data-only, leakage-free, and either use
honest nested inner-MAE selection or transparent fixed rules. ML models are NOT mixed in.

VAR-family variants evaluated:
  1. PlainVAR_BIC          : classical VAR[CPI,Food,NonFood,Services], lag chosen by BIC
                             inside each cutoff (conservative, no grid). Clean baseline.
  2. PlainVAR_nested       : VAR[CPI,Food,NonFood,Services], lags 1..6 x {expanding,rolling120}
                             selected by inner MAE inside each cutoff.
  3. CompConstrainedVAR    : VAR[Food,NonFood,Services] (NO total), CPI = sum(w_i * comp_i)
                             with official weights; lags x train_mode by inner MAE.
                             Tests whether dropping the accounting-identity total helps.
  4. BVAR_avg              : Bayesian VAR model averaging over lags{1,2} x lambda1{0.05,.1,.2,.5}
                             (Minnesota prior, analytic). Per cutoff: softmax weights by inner MAE.
  5. VARfamily_combo       : forecast combination of {PlainVAR_nested, CompConstrainedVAR,
                             BVAR_avg}; softmax weights from each model's honest inner MAE.

Controls / baselines (acceptance bar):
  - ArchivedBVAR : BVAR(lags=4, lambda1=1.0, [CPI,Food,USD,Ruonia]) — the thing to beat.
  - RW, SeasonalNaive — trivial floor (must not be worse than these).

Outer windows: 2018_2019, 2020_2021, 2022_shock, 2023, 2024_2025q1, selection_2025-04_2026-03.
Horizons: 1 (primary), 2, 12.

Research-only. Writes nothing outside runs/opus48_mandatory_var_*. No production changes.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.tsa.api import VAR

RESEARCH_DIR = Path(__file__).resolve().parent
ROOT = RESEARCH_DIR.parents[1]
sys.path.append(str(RESEARCH_DIR))
sys.path.append(str(ROOT))

from run_var_sa_backtests import (  # noqa: E402
    Candidate, forecast_var, forecast_bvar, load_official_data,
    _seasonal_means, _future_seasonal, OFFICIAL_COMPONENT_WEIGHTS,
)

warnings.filterwarnings("ignore")

TOTAL_VARS = ("CPI", "Food", "NonFood", "Services")
COMP_VARS = ("Food", "NonFood", "Services")
LAG_GRID = [1, 2, 3, 4, 5, 6]
TRAIN_MODES = ["expanding", "rolling120"]
BVAR_CONFIGS = [(l, lam) for l in (1, 2) for lam in (0.05, 0.1, 0.2, 0.5)]


# ----------------------------- low-level predictors -----------------------
def _tm(train, mode):
    return train.iloc[-120:].copy() if (mode == "rolling120" and len(train) > 120) else train


def plain_cand(lags):
    return Candidate("plain", "VAR", "official", TOTAL_VARS, {"lags": lags, "target": "CPI"}, forecast_var, "")


def comp_cand(lags):
    return Candidate("comp", "VAR", "official", COMP_VARS, {"lags": lags, "target": "bottom_up"}, forecast_var, "")


def predict_plain(train, horizon, lags, mode):
    return forecast_var(_tm(train, mode).loc[:, list(TOTAL_VARS)], horizon, plain_cand(lags), 0)


def predict_comp(train, horizon, lags, mode):
    return forecast_var(_tm(train, mode).loc[:, list(COMP_VARS)], horizon, comp_cand(lags), 0)


def predict_plain_bic(train, horizon):
    data = train.loc[:, list(TOTAL_VARS)].dropna()
    if len(data) < 40:
        return np.nan
    try:
        order = VAR(data).select_order(maxlags=6)
        lags = int(getattr(order, "bic", 1)) or 1
    except Exception:
        lags = 1
    lags = max(1, min(6, lags))
    return forecast_var(data, horizon, plain_cand(lags), 0)


def bvar_cand(lags, lam, draws):
    return Candidate("bv", "BVAR", "official", TOTAL_VARS,
                     {"lags": lags, "lambda1": lam, "lambda2": 0.5, "lambda3": 1.0,
                      "target": "CPI", "n_draws": draws}, forecast_bvar, "")


def predict_archived_bvar(train, horizon, draws=120):
    cand = Candidate("abv", "BVAR", "official", ("CPI", "Food", "USD", "Ruonia"),
                     {"lags": 4, "lambda1": 1.0, "lambda2": 0.5, "lambda3": 1.0,
                      "target": "CPI", "n_draws": draws}, forecast_bvar, "")
    return forecast_bvar(train, horizon, cand, 12345)


def predict_rw(train, horizon):
    s = train["CPI"].dropna()
    return float(s.iloc[-1]) if len(s) else np.nan


def predict_seasonal_naive(train, horizon):
    mm = _seasonal_means(train, list(TOTAL_VARS), None)
    tm = (train.index.max() + pd.DateOffset(months=horizon)).month
    return float(_future_seasonal(mm, list(TOTAL_VARS), tm)[0])


# ----------------------------- inner validation ---------------------------
def inner_targets(train_full, inner_k):
    return train_full.index[-inner_k:]


def inner_errors(train_full, horizon, inner_k, predict_at):
    """predict_at(tr, horizon) -> float ; returns list of abs errors over inner targets."""
    errs = []
    for t_in in inner_targets(train_full, inner_k):
        ic = t_in - pd.DateOffset(months=horizon)
        tr = train_full[train_full.index <= ic]
        if len(tr) < 30:
            continue
        p = predict_at(tr, horizon)
        if np.isfinite(p) and t_in in train_full.index:
            errs.append(abs(float(train_full.loc[t_in, "CPI"]) - p))
    return errs


def select_nested(train_full, horizon, inner_k, min_valid, grid_predict, grid):
    """grid: list of param tuples; grid_predict(tr,h,*params). Returns (best_params, inner_mae)."""
    best = None
    for params in grid:
        errs = inner_errors(train_full, horizon, inner_k, lambda tr, h: grid_predict(tr, h, *params))
        if len(errs) >= min_valid:
            mae = float(np.mean(errs))
            key = (mae,) + tuple(params if isinstance(params, tuple) else (params,))
            if best is None or key < best[0]:
                best = (key, params, mae)
    if best is None:
        return grid[0], np.nan
    return best[1], best[2]


def bvar_avg_predict_and_inner(train_full, horizon, inner_k, min_valid, draws, tau=0.10):
    """Return (outer_pred, inner_mae, weights_dict). Softmax-by-inner-MAE BVAR model averaging."""
    # inner predictions per config over inner block
    cfg_inner_preds = {}  # cfg -> {t_in: pred}
    cfg_inner_mae = {}
    inner_t = inner_targets(train_full, inner_k)
    for (lags, lam) in BVAR_CONFIGS:
        preds = {}
        errs = []
        for t_in in inner_t:
            ic = t_in - pd.DateOffset(months=horizon)
            tr = train_full[train_full.index <= ic]
            if len(tr) < 30:
                continue
            p = forecast_bvar(tr, horizon, bvar_cand(lags, lam, draws), 777)
            if np.isfinite(p):
                preds[t_in] = p
                if t_in in train_full.index:
                    errs.append(abs(float(train_full.loc[t_in, "CPI"]) - p))
        if len(errs) >= min_valid:
            cfg_inner_preds[(lags, lam)] = preds
            cfg_inner_mae[(lags, lam)] = float(np.mean(errs))
    if not cfg_inner_mae:
        return np.nan, np.nan, {}
    # softmax weights by inner MAE
    cfgs = list(cfg_inner_mae.keys())
    maes = np.array([cfg_inner_mae[c] for c in cfgs])
    w = np.exp(-(maes - maes.min()) / max(tau, 1e-6))
    w = w / w.sum()
    weights = {c: float(wi) for c, wi in zip(cfgs, w)}
    # combined inner MAE (using these weights on the inner block)
    comb_errs = []
    for t_in in inner_t:
        num = 0.0
        wsum = 0.0
        for c in cfgs:
            p = cfg_inner_preds[c].get(t_in)
            if p is not None:
                num += weights[c] * p
                wsum += weights[c]
        if wsum > 0 and t_in in train_full.index:
            comb_errs.append(abs(float(train_full.loc[t_in, "CPI"]) - num / wsum))
    inner_mae = float(np.mean(comb_errs)) if comb_errs else np.nan
    # outer prediction: refit each config on full train, weighted average
    num = 0.0
    wsum = 0.0
    for c in cfgs:
        lags, lam = c
        p = forecast_bvar(train_full, horizon, bvar_cand(lags, lam, draws), 777)
        if np.isfinite(p):
            num += weights[c] * p
            wsum += weights[c]
    outer = num / wsum if wsum > 0 else np.nan
    return outer, inner_mae, weights


# ----------------------------- windows / metrics --------------------------
def build_windows():
    def rng(s, n):
        return pd.date_range(s, periods=n, freq="MS")
    return {
        "2018_2019": rng("2018-01-01", 24),
        "2020_2021": rng("2020-01-01", 24),
        "2022_shock": rng("2022-01-01", 12),
        "2023": rng("2023-01-01", 12),
        "2024_2025q1": rng("2024-01-01", 15),
        "selection_2025-04_2026-03": rng("2025-04-01", 12),
    }


def metrics_from(g):
    v = g.dropna(subset=["abs_error"])
    n = len(v)
    if n == 0:
        return dict(n=0, MAE=np.nan, RMSE=np.nan, Bias=np.nan, MaxAbs=np.nan, KPI_Viol=np.nan, Coverage=np.nan)
    return dict(n=n, MAE=v["abs_error"].mean(), RMSE=np.sqrt((v["error"] ** 2).mean()),
                Bias=v["error"].mean(), MaxAbs=v["abs_error"].max(),
                KPI_Viol=int((v["abs_error"] > 0.5).sum()),
                Coverage=float((v["abs_error"] <= 0.5).mean() * 100))


# ----------------------------- main ---------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-name", default="opus48_mandatory_var_main")
    ap.add_argument("--horizons", default="1,2,12")
    ap.add_argument("--inner-k", type=int, default=12)
    ap.add_argument("--min-valid", type=int, default=6)
    ap.add_argument("--bvar-draws", type=int, default=100)
    args = ap.parse_args()

    horizons = [int(x) for x in args.horizons.split(",")]
    out_dir = RESEARCH_DIR / "runs" / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    official = load_official_data()
    windows = build_windows()

    pred_rows, sel_rows = [], []
    t0 = time.time()

    for hz in horizons:
        for wname, wdates in windows.items():
            for td in wdates:
                if td not in official.index:
                    continue
                actual = float(official.loc[td, "CPI"])
                cutoff = td - pd.DateOffset(months=hz)
                tf = official[official.index <= cutoff].copy()
                if len(tf) < 40:
                    continue
                common = dict(horizon=hz, window=wname, target_date=td, actual=actual,
                              cutoff=cutoff, train_n=len(tf))

                # 1. PlainVAR_BIC
                p_bic = predict_plain_bic(tf, hz)

                # 2. PlainVAR_nested
                grid_plain = [(l, m) for l in LAG_GRID for m in TRAIN_MODES]
                bp, bp_inner = select_nested(tf, hz, args.inner_k, args.min_valid,
                                             lambda tr, h, l, m: predict_plain(tr, h, l, m), grid_plain)
                p_plain = predict_plain(tf, hz, bp[0], bp[1])

                # 3. CompConstrainedVAR_nested
                bc, bc_inner = select_nested(tf, hz, args.inner_k, args.min_valid,
                                             lambda tr, h, l, m: predict_comp(tr, h, l, m), grid_plain)
                p_comp = predict_comp(tf, hz, bc[0], bc[1])

                # 4. BVAR_avg
                p_bavg, bavg_inner, bavg_w = bvar_avg_predict_and_inner(
                    tf, hz, args.inner_k, args.min_valid, args.bvar_draws)

                # 5. VARfamily_combo (softmax by inner MAE of the 3 base models)
                base = [("PlainVAR_nested", p_plain, bp_inner),
                        ("CompConstrainedVAR", p_comp, bc_inner),
                        ("BVAR_avg", p_bavg, bavg_inner)]
                valid = [(nm, pr, im) for nm, pr, im in base if np.isfinite(pr) and np.isfinite(im)]
                if valid:
                    maes = np.array([im for _, _, im in valid])
                    w = np.exp(-(maes - maes.min()) / 0.10)
                    w = w / w.sum()
                    p_combo = float(sum(wi * pr for (_, pr, _), wi in zip(valid, w)))
                    combo_w = {nm: float(wi) for (nm, _, _), wi in zip(valid, w)}
                else:
                    p_combo, combo_w = np.nan, {}

                # controls
                p_arch = predict_archived_bvar(tf, hz, draws=120)
                p_rw = predict_rw(tf, hz)
                p_sn = predict_seasonal_naive(tf, hz)

                for mname, pred in [
                    ("PlainVAR_BIC", p_bic), ("PlainVAR_nested", p_plain),
                    ("CompConstrainedVAR", p_comp), ("BVAR_avg", p_bavg),
                    ("VARfamily_combo", p_combo), ("ArchivedBVAR", p_arch),
                    ("RW", p_rw), ("SeasonalNaive", p_sn),
                ]:
                    err = actual - pred if np.isfinite(pred) else np.nan
                    pred_rows.append({**common, "model": mname, "prediction": pred, "error": err,
                                      "abs_error": abs(err) if np.isfinite(err) else np.nan})

                sel_rows.append({**common,
                                 "plain_lags": bp[0], "plain_tm": bp[1], "plain_inner": bp_inner,
                                 "comp_lags": bc[0], "comp_tm": bc[1], "comp_inner": bc_inner,
                                 "bvar_avg_inner": bavg_inner,
                                 "bvar_avg_weights": json.dumps({f"l{l}_lam{lam}": round(v, 3) for (l, lam), v in bavg_w.items()}),
                                 "combo_weights": json.dumps({k: round(v, 3) for k, v in combo_w.items()})})
        print(f"[h={hz}] done, elapsed {time.time()-t0:.1f}s", flush=True)

    predictions = pd.DataFrame(pred_rows)
    selection = pd.DataFrame(sel_rows)
    predictions.to_csv(out_dir / "predictions.csv", index=False)
    selection.to_csv(out_dir / "selection_log.csv", index=False)

    mrows = []
    for (hz, wname, mname), g in predictions.groupby(["horizon", "window", "model"]):
        mrows.append({"horizon": hz, "window": wname, "model": mname, **metrics_from(g)})
    nonsel = predictions[~predictions["window"].isin(["selection_2025-04_2026-03", "2022_shock"])]
    for (hz, mname), g in nonsel.groupby(["horizon", "model"]):
        mrows.append({"horizon": hz, "window": "AGG_nonsel_nonshock", "model": mname, **metrics_from(g)})
    for (hz, mname), g in predictions.groupby(["horizon", "model"]):
        mrows.append({"horizon": hz, "window": "AGG_all", "model": mname, **metrics_from(g)})
    metrics = pd.DataFrame(mrows).sort_values(["horizon", "window", "MAE"])
    metrics.to_csv(out_dir / "metrics.csv", index=False)

    # comparison.csv: wide MAE table per horizon (AGG rows)
    comp = metrics[metrics.window.str.startswith("AGG")].pivot_table(
        index=["horizon", "window"], columns="model", values="MAE").round(4)
    comp.to_csv(out_dir / "comparison.csv")

    config = {
        "audit": "opus48 mandatory VAR-family benchmark",
        "variants": ["PlainVAR_BIC", "PlainVAR_nested", "CompConstrainedVAR", "BVAR_avg", "VARfamily_combo"],
        "controls": ["ArchivedBVAR", "RW", "SeasonalNaive"],
        "official_weights": OFFICIAL_COMPONENT_WEIGHTS,
        "lag_grid": LAG_GRID, "train_modes": TRAIN_MODES, "bvar_configs": [list(c) for c in BVAR_CONFIGS],
        "bvar_draws": args.bvar_draws, "inner_k": args.inner_k, "min_valid": args.min_valid,
        "inner_validation": "rolling last inner_k months, horizon-matched, inner-MAE selection / softmax averaging",
        "horizons": horizons,
        "windows": {k: [str(v[0].date()), str(v[-1].date())] for k, v in windows.items()},
    }
    (out_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    pd.set_option("display.width", 240)
    print("\n=== h=1 MAE (AGG rows) ===")
    h1 = metrics[(metrics.horizon == 1) & (metrics.window.str.startswith("AGG"))]
    print(h1.pivot_table(index="window", columns="model", values="MAE").round(3).to_string())
    print("\nSaved to", out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
