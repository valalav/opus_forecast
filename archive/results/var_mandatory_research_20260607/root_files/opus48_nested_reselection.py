#!/usr/bin/env python3
"""
Opus 4.8 — Nested rolling re-selection audit of the seasonal-residual VAR family.

Core question: does seasonal-residual VAR have *deployable* value when the
hyperparameters (seasonal roll window, VAR lags, train mode) are chosen honestly
*inside each cutoff* by an inner historical validation, instead of being picked on
the final 12-month evaluation window?

Method (proper nested selection), for each outer target date and horizon h:
  1. outer cutoff = target_date - h months.
  2. train_full = official[index <= cutoff]  (no future data).
  3. INNER validation: take the last `inner_k` months that are all <= cutoff as
     inner targets. For each candidate config and each inner target t_in:
        inner_cutoff = t_in - h ; fit on data <= inner_cutoff ; predict t_in ;
        error vs actual(t_in)  (available because t_in <= cutoff).
     Score = inner MAE over valid inner points. Outer target actual is NEVER used.
  4. Pick the config with the lowest inner MAE (deterministic tie-break).
  5. Refit that config on train_full and forecast the outer target.
  6. Record selected (roll, lags, train_mode) and inner score.

Baselines on the SAME outer dates:
  - RW                  : last observed MoM.
  - SeasonalNaive       : expanding month-of-year mean (residual forced to 0).
  - NestedPlainVAR      : VAR with lags chosen by the same inner scheme, NO seasonal
                          residualization  -> ablation isolating the seasonal benefit.
  - NestedSeasonalVAR   : the family under test.
Optional project baselines (h=1 only): Ridge_ProdProxy_Roll24, Huber, RidgeShockDummies.

Research-only. Writes nothing outside runs/opus48_nested_*. Does not touch production.
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

RESEARCH_DIR = Path(__file__).resolve().parent
ROOT = RESEARCH_DIR.parents[1]
sys.path.append(str(RESEARCH_DIR))
sys.path.append(str(ROOT))

from run_var_sa_backtests import (  # noqa: E402
    Candidate,
    forecast_seasonal_var,
    forecast_var,
    load_official_data,
    _seasonal_means,
    _future_seasonal,
)

warnings.filterwarnings("ignore")

VARIABLES = ("CPI", "Food", "NonFood", "Services")
ROLL_GRID = [24, 30, 36, 42, 48, 60, 72]
LAG_GRID = [1, 2, 3, 4, 5, 6]
TRAIN_MODES = ["expanding", "rolling120"]


# ----------------------------- helpers ------------------------------------
def apply_train_mode(train: pd.DataFrame, train_mode: str) -> pd.DataFrame:
    if train_mode == "rolling120" and len(train) > 120:
        return train.iloc[-120:].copy()
    return train


def seasonal_candidate(roll: int, lags: int) -> Candidate:
    return Candidate(
        name=f"sresid_roll{roll}_l{lags}", family="VAR_seasonal_residual",
        data_source="official", variables=VARIABLES,
        params={"lags": lags, "target": "CPI", "seasonality_window_months": roll},
        forecast_fn=forecast_seasonal_var, notes="",
    )


def plain_candidate(lags: int) -> Candidate:
    return Candidate(
        name=f"plain_l{lags}", family="VAR", data_source="official",
        variables=VARIABLES, params={"lags": lags, "target": "CPI"},
        forecast_fn=forecast_var, notes="",
    )


def predict_seasonal(train: pd.DataFrame, horizon: int, roll: int, lags: int, train_mode: str) -> float:
    tr = apply_train_mode(train, train_mode)
    return forecast_seasonal_var(tr, horizon, seasonal_candidate(roll, lags), 0)


def predict_plain(train: pd.DataFrame, horizon: int, lags: int, train_mode: str) -> float:
    tr = apply_train_mode(train, train_mode)
    return forecast_var(tr.loc[:, list(VARIABLES)], horizon, plain_candidate(lags), 0)


# ----------------------------- baselines ----------------------------------
def predict_rw(train: pd.DataFrame, horizon: int) -> float:
    s = train["CPI"].dropna()
    return float(s.iloc[-1]) if len(s) else np.nan


def predict_seasonal_naive(train: pd.DataFrame, horizon: int) -> float:
    mm = _seasonal_means(train, list(VARIABLES), None)  # expanding, all history
    target_month = (train.index.max() + pd.DateOffset(months=horizon)).month
    return float(_future_seasonal(mm, list(VARIABLES), target_month)[0])


# ------------------------- inner re-selection -----------------------------
def inner_select_seasonal(train_full: pd.DataFrame, horizon: int, inner_k: int,
                          min_valid: int) -> tuple[int, int, str, float]:
    """Return (roll, lags, train_mode, inner_mae) chosen by inner validation."""
    inner_targets = train_full.index[-inner_k:]
    best = None  # (mae, lags, roll, train_mode_rank, train_mode)
    tm_rank = {"expanding": 0, "rolling120": 1}
    for train_mode in TRAIN_MODES:
        for roll in ROLL_GRID:
            for lags in LAG_GRID:
                errs = []
                for t_in in inner_targets:
                    inner_cutoff = t_in - pd.DateOffset(months=horizon)
                    tr = train_full[train_full.index <= inner_cutoff]
                    if len(tr) < 30:
                        continue
                    pred = predict_seasonal(tr, horizon, roll, lags, train_mode)
                    if np.isfinite(pred) and t_in in train_full.index:
                        errs.append(abs(float(train_full.loc[t_in, "CPI"]) - pred))
                if len(errs) >= min_valid:
                    mae = float(np.mean(errs))
                    key = (mae, lags, roll, tm_rank[train_mode])
                    if best is None or key < best[0]:
                        best = (key, roll, lags, train_mode, mae)
    if best is None:
        return 42, 1, "expanding", np.nan
    return best[1], best[2], best[3], best[4]


def inner_select_plain(train_full: pd.DataFrame, horizon: int, inner_k: int,
                       min_valid: int) -> tuple[int, str, float]:
    inner_targets = train_full.index[-inner_k:]
    best = None
    tm_rank = {"expanding": 0, "rolling120": 1}
    for train_mode in TRAIN_MODES:
        for lags in LAG_GRID:
            errs = []
            for t_in in inner_targets:
                inner_cutoff = t_in - pd.DateOffset(months=horizon)
                tr = train_full[train_full.index <= inner_cutoff]
                if len(tr) < 30:
                    continue
                pred = predict_plain(tr, horizon, lags, train_mode)
                if np.isfinite(pred) and t_in in train_full.index:
                    errs.append(abs(float(train_full.loc[t_in, "CPI"]) - pred))
            if len(errs) >= min_valid:
                mae = float(np.mean(errs))
                key = (mae, lags, tm_rank[train_mode])
                if best is None or key < best[0]:
                    best = (key, lags, train_mode, mae)
    if best is None:
        return 1, "expanding", np.nan
    return best[1], best[2], best[3]


# ------------------------- project baselines ------------------------------
def load_model_frame() -> pd.DataFrame:
    raw = pd.read_csv(ROOT / "data" / "inflation_data.csv", sep=";", decimal=",")
    raw["Date"] = pd.to_datetime(raw["Date"], format="%d.%m.%Y", errors="coerce")
    raw["Date"] = raw["Date"].dt.to_period("M").dt.to_timestamp()
    raw = raw.set_index("Date").sort_index()
    df = pd.DataFrame(index=raw.index)
    df["Все товары и услуги"] = pd.to_numeric(raw["mom"], errors="coerce")
    df["Продовольственные товары"] = pd.to_numeric(raw["Prod"], errors="coerce")
    df["Непродовольственные товары"] = pd.to_numeric(raw["Nonprod"], errors="coerce")
    df["Услуги"] = pd.to_numeric(raw["Serv"], errors="coerce")
    for col in ["usd_nom_i", "Ki", "Ki_i", "Ruonia", "Ruonia_i", "fl_potrb_zad", "fl_dep", "all_real"]:
        if col in raw.columns:
            df[col] = pd.to_numeric(raw[col], errors="coerce")
    return df


def project_baseline_predictors():
    from sirena.models.ridge_production_proxy_rolling import RidgeProductionProxyRollingForecaster
    from sirena.models.huber import HuberForecaster
    from sirena.models.ridge_shock_dummies import RidgeShockDummiesForecaster

    def _pt(model, train_model_df, target_date):
        ext = train_model_df.copy()
        ext.loc[target_date] = np.nan
        model.fit(train_model_df, "Все товары и услуги")
        return float(model.predict(ext, target_date)["prediction"] - 100)

    return {
        "Ridge_ProdProxy_Roll24": lambda tr, td: _pt(
            RidgeProductionProxyRollingForecaster(use_2022_dummy=False, seasonality_window=24), tr, td),
        "Huber": lambda tr, td: _pt(HuberForecaster(), tr, td),
        "RidgeShockDummies": lambda tr, td: _pt(RidgeShockDummiesForecaster(use_2022_dummy=False), tr, td),
    }


# ----------------------------- windows ------------------------------------
def build_windows() -> dict[str, pd.DatetimeIndex]:
    def rng(start, periods):
        return pd.date_range(start, periods=periods, freq="MS")
    return {
        "2018_2019": rng("2018-01-01", 24),
        "2020_2021": rng("2020-01-01", 24),
        "2022_shock": rng("2022-01-01", 12),
        "2023": rng("2023-01-01", 12),
        "2024_2025q1": rng("2024-01-01", 15),
        "selection_2025-04_2026-03": rng("2025-04-01", 12),
    }


def metrics_from(df: pd.DataFrame) -> dict:
    valid = df.dropna(subset=["abs_error"])
    n = len(valid)
    if n == 0:
        return dict(n=0, MAE=np.nan, RMSE=np.nan, Bias=np.nan, MaxAbs=np.nan, KPI_Viol=np.nan, Coverage=np.nan)
    return dict(
        n=n, MAE=valid["abs_error"].mean(), RMSE=np.sqrt((valid["error"] ** 2).mean()),
        Bias=valid["error"].mean(), MaxAbs=valid["abs_error"].max(),
        KPI_Viol=int((valid["abs_error"] > 0.5).sum()),
        Coverage=float((valid["abs_error"] <= 0.5).mean() * 100),
    )


# ------------------------------- main -------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-name", default="opus48_nested_main")
    ap.add_argument("--horizons", default="1,2,12")
    ap.add_argument("--inner-k", type=int, default=12)
    ap.add_argument("--min-valid", type=int, default=6)
    ap.add_argument("--project-baselines", action="store_true",
                    help="Also run Ridge_ProdProxy/Huber/RidgeShock (h=1 only).")
    args = ap.parse_args()

    horizons = [int(x) for x in args.horizons.split(",")]
    out_dir = RESEARCH_DIR / "runs" / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    official = load_official_data()
    model_frame = load_model_frame() if args.project_baselines else None
    proj = project_baseline_predictors() if args.project_baselines else {}
    windows = build_windows()

    pred_rows = []
    sel_rows = []
    t0 = time.time()

    for hz in horizons:
        for wname, wdates in windows.items():
            for target_date in wdates:
                if target_date not in official.index:
                    continue
                actual = float(official.loc[target_date, "CPI"])
                cutoff = target_date - pd.DateOffset(months=hz)
                train_full = official[official.index <= cutoff].copy()
                if len(train_full) < 40:
                    continue

                # --- nested seasonal VAR ---
                roll_s, lags_s, tm_s, inner_s = inner_select_seasonal(
                    train_full, hz, args.inner_k, args.min_valid)
                pred_s = predict_seasonal(train_full, hz, roll_s, lags_s, tm_s)

                # --- nested plain VAR (ablation) ---
                lags_p, tm_p, inner_p = inner_select_plain(
                    train_full, hz, args.inner_k, args.min_valid)
                pred_p = predict_plain(train_full, hz, lags_p, tm_p)

                # --- simple baselines ---
                pred_rw = predict_rw(train_full, hz)
                pred_sn = predict_seasonal_naive(train_full, hz)

                row_common = dict(horizon=hz, window=wname, target_date=target_date, actual=actual,
                                  cutoff=cutoff, train_n=len(train_full))
                for mname, pred, extra in [
                    ("NestedSeasonalVAR", pred_s, dict(sel_roll=roll_s, sel_lags=lags_s, sel_train_mode=tm_s, inner_mae=inner_s)),
                    ("NestedPlainVAR", pred_p, dict(sel_lags=lags_p, sel_train_mode=tm_p, inner_mae=inner_p)),
                    ("RW", pred_rw, {}),
                    ("SeasonalNaive", pred_sn, {}),
                ]:
                    err = actual - pred if np.isfinite(pred) else np.nan
                    pred_rows.append({**row_common, "model": mname, "prediction": pred,
                                      "error": err, "abs_error": abs(err) if np.isfinite(err) else np.nan,
                                      **extra})

                sel_rows.append({**row_common, "fam": "seasonal", "sel_roll": roll_s, "sel_lags": lags_s,
                                 "sel_train_mode": tm_s, "inner_mae": inner_s})
                sel_rows.append({**row_common, "fam": "plain", "sel_roll": np.nan, "sel_lags": lags_p,
                                 "sel_train_mode": tm_p, "inner_mae": inner_p})

                # --- project baselines (h=1 only) ---
                if args.project_baselines and hz == 1:
                    train_model = model_frame[model_frame.index <= cutoff].copy()
                    for mname, fn in proj.items():
                        try:
                            pred = fn(train_model, target_date)
                        except Exception:
                            pred = np.nan
                        err = actual - pred if np.isfinite(pred) else np.nan
                        pred_rows.append({**row_common, "model": mname, "prediction": pred,
                                          "error": err, "abs_error": abs(err) if np.isfinite(err) else np.nan})

        print(f"[h={hz}] done, elapsed {time.time()-t0:.1f}s")

    predictions = pd.DataFrame(pred_rows)
    selection = pd.DataFrame(sel_rows)
    predictions.to_csv(out_dir / "predictions.csv", index=False)
    selection.to_csv(out_dir / "selection_log.csv", index=False)

    # metrics per model/window/horizon + aggregate (excl shock) + all
    mrows = []
    for (hz, wname, mname), g in predictions.groupby(["horizon", "window", "model"]):
        mrows.append({"horizon": hz, "window": wname, "model": mname, **metrics_from(g)})
    # aggregate across non-selection non-shock windows
    nonsel = predictions[~predictions["window"].isin(["selection_2025-04_2026-03", "2022_shock"])]
    for (hz, mname), g in nonsel.groupby(["horizon", "model"]):
        mrows.append({"horizon": hz, "window": "AGG_nonsel_nonshock", "model": mname, **metrics_from(g)})
    allw = predictions
    for (hz, mname), g in allw.groupby(["horizon", "model"]):
        mrows.append({"horizon": hz, "window": "AGG_all", "model": mname, **metrics_from(g)})
    metrics = pd.DataFrame(mrows).sort_values(["horizon", "window", "MAE"])
    metrics.to_csv(out_dir / "metrics.csv", index=False)

    config = {
        "audit": "opus48 nested rolling re-selection",
        "family": "seasonal-residual VAR (official data)",
        "variables": list(VARIABLES),
        "roll_grid": ROLL_GRID, "lag_grid": LAG_GRID, "train_modes": TRAIN_MODES,
        "inner_validation": {"scheme": "rolling last inner_k months, horizon-matched cutoffs",
                             "inner_k": args.inner_k, "min_valid": args.min_valid,
                             "selection_metric": "inner MAE", "tie_break": "(mae, lags, roll, train_mode)"},
        "horizons": horizons,
        "windows": {k: [str(v[0].date()), str(v[-1].date())] for k, v in windows.items()},
        "baselines": ["RW", "SeasonalNaive", "NestedPlainVAR"] + (list(proj.keys()) if proj else []),
        "project_baselines_enabled": bool(args.project_baselines),
    }
    (out_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    # console summary
    pd.set_option("display.width", 220)
    print("\n=== h=1 MAE by window (key models) ===")
    h1 = metrics[metrics.horizon == 1]
    piv = h1.pivot_table(index="window", columns="model", values="MAE")
    cols = [c for c in ["NestedSeasonalVAR", "NestedPlainVAR", "RW", "SeasonalNaive",
                        "Ridge_ProdProxy_Roll24", "Huber", "RidgeShockDummies"] if c in piv.columns]
    print(piv[cols].round(3).to_string())
    print("\nSaved to", out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
