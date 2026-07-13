#!/usr/bin/env python3
"""
Opus 4.8 independent robustness audit for fine_seasonal_resid_var_tc_roll42_l5.

This is a research-only extension of run_var_sa_backtests.py. It does NOT modify
production code or data, and writes only under runs/opus48_*.

What it does, none of which the original runner can do directly:
  1. Freezes the reported best config and evaluates it on multiple historical
     12-month windows (not just the single archive selection window).
  2. Evaluates a small neighborhood (roll in {30,36,42,48,60} x lags in {1..6})
     on every window to check whether roll42_l5 is a stable plateau or a lucky
     isolated point.
  3. Adds honest, self-computed baselines on the *same* windows and cutoff
     discipline (so the comparison is apples-to-apples, unlike archive metrics
     which only exist for the 2025-04..2026-03 window):
       - SeasonalNaive: trailing-42m month-of-year mean (the seasonal factor
         alone, i.e. residual prediction forced to 0).
       - RW (random walk / last observed MoM).
       - RW12 (value 12 months earlier, seasonal random walk).
       - PlainVAR_l5: VAR(5) on raw MoM, no seasonal residualization.
  4. Re-derives the archive selection-window metrics for roll42_l5 as an
     independent reproduction check.

Usage:
  python3 experiments/var_sa_research/opus48_robustness.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.tsa.api import VAR

RESEARCH_DIR = Path(__file__).resolve().parent
ROOT = RESEARCH_DIR.parents[1]
sys.path.append(str(RESEARCH_DIR))

from run_var_sa_backtests import (  # noqa: E402
    Candidate,
    forecast_seasonal_var,
    forecast_var,
    load_official_data,
    _seasonal_means,
    _future_seasonal,
)

OUT_DIR = RESEARCH_DIR / "runs" / "opus48_robustness"
VARIABLES = ("CPI", "Food", "NonFood", "Services")


def make_seasonal_candidate(window_months: int, lags: int) -> Candidate:
    return Candidate(
        name=f"fine_seasonal_resid_var_tc_roll{window_months}_l{lags}",
        family="VAR_seasonal_residual",
        data_source="official",
        variables=VARIABLES,
        params={"lags": lags, "target": "CPI", "seasonality_window_months": window_months},
        forecast_fn=forecast_seasonal_var,
        notes="frozen / neighborhood",
    )


# ---- baselines (same cutoff discipline) ----------------------------------
def predict_seasonal_naive(train: pd.DataFrame, horizon: int) -> float:
    mm = _seasonal_means(train, list(VARIABLES), 42)
    target_month = (train.index.max() + pd.DateOffset(months=horizon)).month
    return float(_future_seasonal(mm, list(VARIABLES), target_month)[0])


def predict_rw(train: pd.DataFrame, horizon: int) -> float:
    s = train["CPI"].dropna()
    return float(s.iloc[-1]) if len(s) else np.nan


def predict_rw12(train: pd.DataFrame, horizon: int) -> float:
    s = train["CPI"].dropna()
    target_date = train.index.max() + pd.DateOffset(months=horizon)
    src = target_date - pd.DateOffset(months=12)
    if src in s.index:
        return float(s.loc[src])
    return np.nan


def predict_plain_var_l5(train: pd.DataFrame, horizon: int) -> float:
    cand = Candidate(
        name="plain_var_l5", family="VAR", data_source="official",
        variables=VARIABLES, params={"lags": 5, "target": "CPI"},
        forecast_fn=forecast_var, notes="",
    )
    return forecast_var(train.loc[:, list(VARIABLES)], horizon, cand, 0)


def eval_model(official, predict_fn, window_dates, horizon) -> dict:
    rows = []
    for target_date in window_dates:
        if target_date not in official.index:
            continue
        actual = float(official.loc[target_date, "CPI"])
        cutoff = target_date - pd.DateOffset(months=horizon)
        train = official[official.index <= cutoff].copy()
        try:
            pred = predict_fn(train, horizon)
        except Exception:
            pred = np.nan
        err = actual - pred if np.isfinite(pred) else np.nan
        rows.append({"target_date": target_date, "actual": actual, "prediction": pred,
                     "error": err, "abs_error": abs(err) if np.isfinite(err) else np.nan})
    df = pd.DataFrame(rows)
    valid = df.dropna(subset=["abs_error"])
    n = len(valid)
    return {
        "n": n,
        "MAE": valid["abs_error"].mean() if n else np.nan,
        "RMSE": np.sqrt((valid["error"] ** 2).mean()) if n else np.nan,
        "Bias": valid["error"].mean() if n else np.nan,
        "MaxAbs": valid["abs_error"].max() if n else np.nan,
        "KPI_Viol": int((valid["abs_error"] > 0.5).sum()) if n else np.nan,
        "_pred_df": df,
    }


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    official = load_official_data()

    # Define evaluation windows (each is a 12-month block of target dates).
    windows = {
        "2018": pd.date_range("2018-01-01", periods=12, freq="MS"),
        "2019": pd.date_range("2019-01-01", periods=12, freq="MS"),
        "2020": pd.date_range("2020-01-01", periods=12, freq="MS"),
        "2021": pd.date_range("2021-01-01", periods=12, freq="MS"),
        "2023": pd.date_range("2023-01-01", periods=12, freq="MS"),
        "2024": pd.date_range("2024-01-01", periods=12, freq="MS"),
        # selection window used by the archive + Codex CLI
        "selection_2025-04_2026-03": pd.date_range("2025-04-01", periods=12, freq="MS"),
    }
    # NOTE: 2022 deliberately included separately because CLAUDE notes 2022 as a
    # shock/outlier year; we report it but flag it.
    windows["2022_shock"] = pd.date_range("2022-01-01", periods=12, freq="MS")

    horizons = [1, 2, 12]

    # ---- 1. Frozen roll42_l5 across windows ----
    frozen = make_seasonal_candidate(42, 5)

    def frozen_predict(train, horizon, _c=frozen):
        run_seed = 0
        return forecast_seasonal_var(train, horizon, _c, run_seed)

    baselines = {
        "SeasonalNaive_roll42": predict_seasonal_naive,
        "RW": predict_rw,
        "RW12": predict_rw12,
        "PlainVAR_l5": predict_plain_var_l5,
        "roll42_l5": frozen_predict,
    }

    robustness_rows = []
    all_pred_rows = []
    for wname, wdates in windows.items():
        for hz in horizons:
            for mname, fn in baselines.items():
                res = eval_model(official, lambda tr, h: fn(tr, h), wdates, hz)
                robustness_rows.append({
                    "window": wname, "horizon": hz, "model": mname,
                    "n": res["n"], "MAE": res["MAE"], "RMSE": res["RMSE"],
                    "Bias": res["Bias"], "MaxAbs": res["MaxAbs"], "KPI_Viol": res["KPI_Viol"],
                })
                pdf = res["_pred_df"].copy()
                pdf["window"] = wname
                pdf["horizon"] = hz
                pdf["model"] = mname
                all_pred_rows.append(pdf)

    robustness = pd.DataFrame(robustness_rows)
    robustness.to_csv(OUT_DIR / "robustness_by_window.csv", index=False)
    pd.concat(all_pred_rows, ignore_index=True).to_csv(OUT_DIR / "predictions.csv", index=False)

    # ---- 2. Neighborhood plateau check (h=1 only, all windows) ----
    nb_rows = []
    nb_windows = {k: v for k, v in windows.items() if k != "2022_shock"}
    for wname, wdates in nb_windows.items():
        for roll in [30, 36, 42, 48, 60]:
            for lags in [1, 2, 3, 4, 5, 6]:
                cand = make_seasonal_candidate(roll, lags)
                res = eval_model(
                    official,
                    lambda tr, h, c=cand: forecast_seasonal_var(tr, h, c, 0),
                    wdates, 1,
                )
                nb_rows.append({"window": wname, "roll": roll, "lags": lags,
                                "MAE_h1": res["MAE"], "KPI_Viol": res["KPI_Viol"], "n": res["n"]})
    neighborhood = pd.DataFrame(nb_rows)
    neighborhood.to_csv(OUT_DIR / "neighborhood_h1.csv", index=False)

    # ---- metrics.csv: frozen roll42_l5 summary per window/horizon ----
    metrics = robustness[robustness["model"] == "roll42_l5"].copy()
    metrics.to_csv(OUT_DIR / "metrics.csv", index=False)

    # ---- config.json ----
    config = {
        "audit": "opus48 independent robustness",
        "frozen_config": "fine_seasonal_resid_var_tc_roll42_l5",
        "frozen_params": {"lags": 5, "seasonality_window_months": 42, "variables": list(VARIABLES)},
        "windows": {k: [str(v[0].date()), str(v[-1].date())] for k, v in windows.items()},
        "horizons": horizons,
        "baselines": list(baselines.keys()),
        "neighborhood_grid": {"roll": [30, 36, 42, 48, 60], "lags": [1, 2, 3, 4, 5, 6]},
    }
    (OUT_DIR / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    # ---- console summary ----
    pd.set_option("display.width", 200)
    print("=== Frozen roll42_l5 by window (h=1) ===")
    print(metrics[metrics["horizon"] == 1][["window", "MAE", "RMSE", "Bias", "KPI_Viol", "n"]].to_string(index=False))
    print("\n=== roll42_l5 vs baselines, h=1 MAE by window ===")
    piv = robustness[robustness["horizon"] == 1].pivot_table(index="window", columns="model", values="MAE")
    print(piv.to_string())
    print("\nSaved to", OUT_DIR)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
