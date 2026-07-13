#!/usr/bin/env python3
"""
Opus 4.8 — Parsimonious multi-horizon VAR / trajectory quality.

Two parts:
  PART A (parsimony): grid of endogenous subsets x macro(exog) subsets, evaluated on h=1/h=2/h=12
    rolling MAE over 6 windows. Tests whether smaller variable sets beat macro-heavy ones, and
    whether USD/Ki_i/Ruonia help. VARX exog = last observed value (deterministic) for Part A.
  PART B (exog paths): for the best macro subset, compare deterministic exog path scenarios
    (last_obs, AR1 forecast, recent-12m mean) on h=12 point error.
  PART C (trajectory realism): deterministic fixed-cutoff 12-month trajectories from several
    cutoffs; realism diagnostics (path std, vol ratio, sign changes, max jump, seasonal amplitude,
    seasonal-shape correlation, flatness, explosive flag, component consistency). Charts for best.

All forecasts are deterministic (no random noise in point forecasts). Leakage-free.
Research-only; writes only under runs/opus48_trajectory_var_*.
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

RESEARCH_DIR = Path(__file__).resolve().parent
ROOT = RESEARCH_DIR.parents[1]
sys.path.append(str(RESEARCH_DIR))
sys.path.append(str(ROOT))

from run_var_sa_backtests import (  # noqa: E402
    load_official_data, _seasonal_means, _future_seasonal, OFFICIAL_COMPONENT_WEIGHTS,
)
from opus48_robust_var import _design, _fit_eqs, predict_archived_bvar  # noqa: E402
from statsmodels.tsa.api import VAR  # noqa: E402

warnings.filterwarnings("ignore")
W = OFFICIAL_COMPONENT_WEIGHTS

ENDOG_SETS = {
    "CPI": ["CPI"],
    "CPI_F_NF_S": ["CPI", "Food", "NonFood", "Services"],
    "F_NF_S_bottomup": ["Food", "NonFood", "Services"],
    "CPI_F": ["CPI", "Food"],
    "CPI_NF": ["CPI", "NonFood"],
    "CPI_S": ["CPI", "Services"],
    "CPI_F_NF": ["CPI", "Food", "NonFood"],
    "CPI_F_S": ["CPI", "Food", "Services"],
    "CPI_NF_S": ["CPI", "NonFood", "Services"],
}
MACRO_SETS = {
    "none": [], "USD": ["USD"], "Ki": ["Ki_i"], "Ruonia": ["Ruonia"],
    "USD_Ki": ["USD", "Ki_i"], "USD_Ruonia": ["USD", "Ruonia"],
    "Ki_Ruonia": ["Ki_i", "Ruonia"], "USD_Ki_Ruonia": ["USD", "Ki_i", "Ruonia"],
}


# ----------------------- deterministic exog future paths ------------------
def exog_future_builder(train_macro: pd.DataFrame, macro: list[str], mode: str):
    """Return fn(step:int)->np.array of exog values for that future step (deterministic)."""
    if not macro:
        return lambda step: np.array([])
    last = train_macro.iloc[-1][macro].values.astype(float)
    if mode == "last_obs":
        return lambda step: last
    if mode == "recent_mean":
        rm = train_macro.iloc[-12:][macro].mean().values.astype(float)
        return lambda step: rm
    if mode == "ar1":
        # per-variable AR(1): x_t = c + phi x_{t-1}; deterministic forward path
        params = {}
        for m in macro:
            s = train_macro[m].dropna().values.astype(float)
            if len(s) < 24:
                params[m] = (s[-1], 0.0, s[-1])
                continue
            y = s[1:]; x = s[:-1]
            X = np.column_stack([np.ones_like(x), x])
            b = np.linalg.lstsq(X, y, rcond=None)[0]
            params[m] = (b[0], b[1], s[-1])
        def fn(step, params=params, macro=macro):
            out = []
            for m in macro:
                c, phi, last_v = params[m]
                v = last_v
                for _ in range(step):
                    v = c + phi * v
                out.append(v)
            return np.array(out, dtype=float)
        return fn
    raise ValueError(mode)


def varx_path(train, endog, macro, lags, horizon, exog_mode="last_obs", target="auto", min_train=36):
    """Deterministic recursive VARX path. Returns np.array of length `horizon` of CPI MoM."""
    cols = endog + macro
    data = train.loc[:, cols].dropna()
    if len(data) < max(min_train, lags + 18):
        return np.full(horizon, np.nan)
    arr_endog = data.loc[:, endog].values.astype(float)
    exog_hist = data.loc[:, macro].values.astype(float) if macro else None
    X, Y = _design(arr_endog, lags, exog_hist)
    if len(X) < lags + 10:
        return np.full(horizon, np.nan)
    B = _fit_eqs(X, Y, "ols")  # (k, 1+k*lags+nexog)
    fut_fn = exog_future_builder(data.loc[:, macro] if macro else data, macro, exog_mode)
    hist = list(arr_endog[-lags:])
    path = []
    cpi_idx = endog.index("CPI") if "CPI" in endog else None
    for step in range(1, horizon + 1):
        row = [1.0]
        for lag in range(1, lags + 1):
            row.extend(hist[-lag])
        if macro:
            row.extend(fut_fn(step))
        pv = (np.array([row]) @ B.T).ravel()
        hist.append(pv)
        if cpi_idx is not None:
            path.append(float(pv[cpi_idx]))
        else:
            d = dict(zip(endog, pv))
            path.append(sum(d[n] * W[n] for n in ("Food", "NonFood", "Services")))
    return np.array(path, dtype=float)


def varx_point(train, endog, macro, lags, horizon, exog_mode="last_obs"):
    p = varx_path(train, endog, macro, lags, horizon, exog_mode)
    return p[horizon - 1] if len(p) >= horizon else np.nan


def seasonal_var_path(train, endog, lags, horizon, min_train=40):
    """Deterministic seasonal VAR path: deseasonalize by EXPANDING train-only month means
    (no roll-window tuning -> parsimonious, not the rejected roll42 approach), VAR(1) on the
    residual, then add the target-month seasonal mean back for each step. Returns CPI path."""
    data = train.loc[:, endog].dropna()
    if len(data) < min_train:
        return np.full(horizon, np.nan)
    month_means = data.groupby(data.index.month).mean()
    resid = data.copy()
    for m in range(1, 13):
        mask = data.index.month == m
        if m in month_means.index:
            resid.loc[mask, endog] = data.loc[mask, endog] - month_means.loc[m, endog].values
    arr = resid.values.astype(float)
    X, Y = _design(arr, lags, None)
    if len(X) < lags + 10:
        return np.full(horizon, np.nan)
    B = _fit_eqs(X, Y, "ols")
    hist = list(arr[-lags:])
    last = data.index.max()
    cpi_idx = endog.index("CPI") if "CPI" in endog else None
    path = []
    for step in range(1, horizon + 1):
        row = [1.0]
        for lag in range(1, lags + 1):
            row.extend(hist[-lag])
        pv = (np.array([row]) @ B.T).ravel()
        hist.append(pv)
        m = (last + pd.DateOffset(months=step)).month
        seas = month_means.loc[m, endog].values if m in month_means.index else np.zeros(len(endog))
        recon = pv + seas
        if cpi_idx is not None:
            path.append(float(recon[cpi_idx]))
        else:
            d = dict(zip(endog, recon))
            path.append(sum(d[n] * W[n] for n in ("Food", "NonFood", "Services")))
    return np.array(path, dtype=float)


def seasonal_var_point(train, endog, lags, horizon):
    p = seasonal_var_path(train, endog, lags, horizon)
    return p[horizon - 1] if len(p) >= horizon else np.nan


# ----------------------- baselines (paths) --------------------------------
def rw_path(train, horizon):
    s = train["CPI"].dropna()
    return np.full(horizon, float(s.iloc[-1]) if len(s) else np.nan)


def seasonal_naive_path(train, horizon):
    mm = _seasonal_means(train, ["CPI"], None)
    last = train.index.max()
    out = []
    for step in range(1, horizon + 1):
        m = (last + pd.DateOffset(months=step)).month
        out.append(float(_future_seasonal(mm, ["CPI"], m)[0]))
    return np.array(out)


# ----------------------- windows / metrics --------------------------------
def build_windows():
    def rng(s, n): return pd.date_range(s, periods=n, freq="MS")
    return {
        "2018_2019": rng("2018-01-01", 24), "2020_2021": rng("2020-01-01", 24),
        "2022_shock": rng("2022-01-01", 12), "2023": rng("2023-01-01", 12),
        "2024_2025q1": rng("2024-01-01", 15), "selection_2025-04_2026-03": rng("2025-04-01", 12),
    }


def metrics_from(g):
    v = g.dropna(subset=["abs_error"]); n = len(v)
    if n == 0:
        return dict(n=0, MAE=np.nan, RMSE=np.nan, KPI_Viol=np.nan)
    return dict(n=n, MAE=v["abs_error"].mean(), RMSE=np.sqrt((v["error"]**2).mean()),
                KPI_Viol=int((v["abs_error"] > 0.5).sum()))


# ----------------------- trajectory realism -------------------------------
def trajectory_diag(path, train, target_dates):
    """Deterministic realism diagnostics for a 12-step CPI MoM path."""
    p = np.asarray(path, dtype=float)
    if not np.all(np.isfinite(p)):
        return dict(path_std=np.nan, vol_ratio=np.nan, sign_changes=np.nan, max_jump=np.nan,
                    flatness=np.nan, seasonal_corr=np.nan, seasonal_amp=np.nan, explosive=True,
                    n_identical=np.nan)
    hist = train["CPI"].dropna().iloc[-36:].values
    hist_std = float(np.std(hist)) if len(hist) else np.nan
    diffs = np.diff(p)
    # seasonal shape: correlate path month pattern with historical month-of-year means
    mm = train.assign(_m=train.index.month).groupby("_m")["CPI"].mean()
    months = [d.month for d in target_dates]
    hist_season = np.array([mm.get(m, np.nan) for m in months])
    if np.all(np.isfinite(hist_season)) and np.std(hist_season) > 1e-6 and np.std(p) > 1e-6:
        seasonal_corr = float(np.corrcoef(p, hist_season)[0, 1])
    else:
        seasonal_corr = np.nan
    return dict(
        path_std=float(np.std(p)),
        vol_ratio=float(np.std(p) / hist_std) if hist_std and hist_std > 1e-9 else np.nan,
        sign_changes=int(np.sum(np.diff(np.sign(p)) != 0)),
        max_jump=float(np.max(np.abs(diffs))) if len(diffs) else 0.0,
        flatness=float(np.mean(np.abs(diffs) < 0.05)) if len(diffs) else np.nan,
        seasonal_corr=seasonal_corr,
        seasonal_amp=float(p.max() - p.min()),
        explosive=bool(np.max(np.abs(p)) > 5.0 or (hist_std and np.std(p) > 4 * hist_std)),
        n_identical=int(np.sum(np.abs(diffs) < 1e-9)),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-name", default="opus48_trajectory_var_main")
    args = ap.parse_args()
    out_dir = RESEARCH_DIR / "runs" / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    official = load_official_data()
    windows = build_windows()
    t0 = time.time()

    # ---------------- PART A: parsimony grid (h=1/2/12, last_obs exog) ----------------
    specs = []
    for ename, endog in ENDOG_SETS.items():
        for mname, macro in MACRO_SETS.items():
            if ename == "F_NF_S_bottomup" and macro:
                continue  # bottom-up tested macro-free only (no CPI eq for macro to feed)
            specs.append((f"{ename}+{mname}", endog, macro))

    rows = []
    for hz in [1, 2, 12]:
        for wname, wdates in windows.items():
            for td in wdates:
                if td not in official.index:
                    continue
                actual = float(official.loc[td, "CPI"])
                cutoff = td - pd.DateOffset(months=hz)
                tf = official[official.index <= cutoff]
                if len(tf) < 40:
                    continue
                for sname, endog, macro in specs:
                    pred = varx_point(tf, endog, macro, 1, hz, "last_obs")
                    err = actual - pred if np.isfinite(pred) else np.nan
                    rows.append(dict(horizon=hz, window=wname, target_date=td, actual=actual,
                                     spec=sname, prediction=pred, error=err,
                                     abs_error=abs(err) if np.isfinite(err) else np.nan))
        print(f"[PartA h={hz}] {time.time()-t0:.1f}s", flush=True)
    # seasonal-deterministic specs point error (same windows/horizons)
    for hz in [1, 2, 12]:
        for wname, wdates in windows.items():
            for td in wdates:
                if td not in official.index:
                    continue
                actual = float(official.loc[td, "CPI"]); cutoff = td - pd.DateOffset(months=hz)
                tf = official[official.index <= cutoff]
                if len(tf) < 40:
                    continue
                for sname, sendog in {"SeasonalVAR_CPI": ENDOG_SETS["CPI"],
                                      "SeasonalVAR_CPI_F_NF_S": ENDOG_SETS["CPI_F_NF_S"]}.items():
                    pred = seasonal_var_point(tf, sendog, 1, hz)
                    err = actual - pred if np.isfinite(pred) else np.nan
                    rows.append(dict(horizon=hz, window=wname, target_date=td, actual=actual,
                                     spec=sname, prediction=pred, error=err,
                                     abs_error=abs(err) if np.isfinite(err) else np.nan))
    grid = pd.DataFrame(rows)
    grid.to_csv(out_dir / "predictions.csv", index=False)

    mrows = []
    for (hz, sname), g in grid.groupby(["horizon", "spec"]):
        mrows.append(dict(horizon=hz, window="AGG_all", spec=sname, **metrics_from(g)))
    nonshock = grid[~grid.window.isin(["selection_2025-04_2026-03", "2022_shock"])]
    for (hz, sname), g in nonshock.groupby(["horizon", "spec"]):
        mrows.append(dict(horizon=hz, window="AGG_oos_nonshock", spec=sname, **metrics_from(g)))
    for (hz, wname, sname), g in grid.groupby(["horizon", "window", "spec"]):
        mrows.append(dict(horizon=hz, window=wname, spec=sname, **metrics_from(g)))
    metrics = pd.DataFrame(mrows)
    metrics.to_csv(out_dir / "metrics.csv", index=False)

    # ---------------- PART B: exog path scenarios (best macro subset) ----------------
    # use a representative macro spec CPI_F_NF_S + USD_Ki_Ruonia, compare exog paths on h=12
    bpb = []
    endogB, macroB = ENDOG_SETS["CPI_F_NF_S"], MACRO_SETS["USD_Ki_Ruonia"]
    for mode in ["last_obs", "ar1", "recent_mean"]:
        for wname, wdates in windows.items():
            for td in wdates:
                if td not in official.index:
                    continue
                actual = float(official.loc[td, "CPI"]); cutoff = td - pd.DateOffset(months=12)
                tf = official[official.index <= cutoff]
                if len(tf) < 40:
                    continue
                pred = varx_point(tf, endogB, macroB, 1, 12, mode)
                err = actual - pred if np.isfinite(pred) else np.nan
                bpb.append(dict(exog_mode=mode, window=wname, target_date=td,
                                error=err, abs_error=abs(err) if np.isfinite(err) else np.nan))
    exogB = pd.DataFrame(bpb)
    exog_metrics = exogB.groupby("exog_mode").apply(
        lambda g: pd.Series(metrics_from(g.rename(columns={})))).reset_index()
    exog_metrics.to_csv(out_dir / "exog_path_metrics.csv", index=False)
    print(f"[PartB] {time.time()-t0:.1f}s", flush=True)

    # ---------------- PART C: trajectory realism (fixed cutoffs) ----------------
    traj_models = {
        "PlainVAR_CPI_F_NF_S": (ENDOG_SETS["CPI_F_NF_S"], [], "last_obs"),
        "VAR_CPI_only": (ENDOG_SETS["CPI"], [], "last_obs"),
        "CompConstrained_F_NF_S": (ENDOG_SETS["F_NF_S_bottomup"], [], "last_obs"),
        "VARX_USD_lastobs": (ENDOG_SETS["CPI_F_NF_S"], ["USD"], "last_obs"),
        "VARX_USDKiRuonia_lastobs": (ENDOG_SETS["CPI_F_NF_S"], ["USD", "Ki_i", "Ruonia"], "last_obs"),
        "VARX_USDKiRuonia_ar1": (ENDOG_SETS["CPI_F_NF_S"], ["USD", "Ki_i", "Ruonia"], "ar1"),
        "CPI_Services": (ENDOG_SETS["CPI_S"], [], "last_obs"),
    }
    # seasonal-deterministic candidates (computed via seasonal_var_path)
    seasonal_traj = {
        "SeasonalVAR_CPI": ENDOG_SETS["CPI"],
        "SeasonalVAR_CPI_F_NF_S": ENDOG_SETS["CPI_F_NF_S"],
    }
    fixed_cutoffs = [pd.Timestamp("2024-11-01"), pd.Timestamp("2025-03-01"),
                     pd.Timestamp("2021-12-01"), pd.Timestamp("2019-12-01")]
    traj_rows = []
    chart_data = {}
    for cutoff in fixed_cutoffs:
        if cutoff not in official.index:
            continue
        tf = official[official.index <= cutoff]
        target_dates = [cutoff + pd.DateOffset(months=s) for s in range(1, 13)]
        actuals = np.array([float(official.loc[d, "CPI"]) if d in official.index else np.nan
                            for d in target_dates])
        chart_data[str(cutoff.date())] = {"target_dates": target_dates, "actual": actuals, "paths": {}}
        for mname, (endog, macro, mode) in traj_models.items():
            path = varx_path(tf, endog, macro, 1, 12, mode)
            diag = trajectory_diag(path, tf, target_dates)
            valid = np.isfinite(path) & np.isfinite(actuals)
            mae = float(np.mean(np.abs(actuals[valid] - path[valid]))) if valid.any() else np.nan
            traj_rows.append(dict(cutoff=str(cutoff.date()), model=mname, traj_MAE=mae, **diag))
            chart_data[str(cutoff.date())]["paths"][mname] = path
        for smname, sendog in seasonal_traj.items():
            spath = seasonal_var_path(tf, sendog, 1, 12)
            diag = trajectory_diag(spath, tf, target_dates)
            valid = np.isfinite(spath) & np.isfinite(actuals)
            mae = float(np.mean(np.abs(actuals[valid] - spath[valid]))) if valid.any() else np.nan
            traj_rows.append(dict(cutoff=str(cutoff.date()), model=smname, traj_MAE=mae, **diag))
            chart_data[str(cutoff.date())]["paths"][smname] = spath
        # add RW + seasonal naive
        for nm, pth in [("RW", rw_path(tf, 12)), ("SeasonalNaive", seasonal_naive_path(tf, 12))]:
            diag = trajectory_diag(pth, tf, target_dates)
            valid = np.isfinite(pth) & np.isfinite(actuals)
            mae = float(np.mean(np.abs(actuals[valid] - pth[valid]))) if valid.any() else np.nan
            traj_rows.append(dict(cutoff=str(cutoff.date()), model=nm, traj_MAE=mae, **diag))
            chart_data[str(cutoff.date())]["paths"][nm] = pth
    traj = pd.DataFrame(traj_rows)
    traj.to_csv(out_dir / "trajectory_metrics.csv", index=False)

    # charts
    for cstr, cd in chart_data.items():
        fig, ax = plt.subplots(figsize=(11, 6))
        x = [d.strftime("%Y-%m") for d in cd["target_dates"]]
        ax.plot(x, cd["actual"], "k-o", lw=2.5, label="Actual", zorder=10)
        for mname in ["VAR_CPI_only", "SeasonalVAR_CPI", "VARX_USDKiRuonia_lastobs",
                      "VARX_USDKiRuonia_ar1", "SeasonalNaive", "RW"]:
            if mname in cd["paths"]:
                ax.plot(x, cd["paths"][mname], "--", alpha=0.8, label=mname)
        ax.axhline(0, color="gray", lw=0.6)
        ax.set_title(f"Deterministic 12-month CPI MoM trajectory — cutoff {cstr}")
        ax.set_ylabel("CPI MoM (%)"); ax.tick_params(axis="x", rotation=45)
        ax.legend(fontsize=8, ncol=2); fig.tight_layout()
        fig.savefig(out_dir / f"trajectory_{cstr}.png", dpi=110); plt.close(fig)

    # comparison.csv: h=1/2/12 AGG MAE for shortlist specs + baselines
    short = ["CPI+none", "CPI_F_NF_S+none", "CPI_F_NF_S+USD", "CPI_F_NF_S+USD_Ki_Ruonia",
             "CPI_S+none", "CPI_F+none", "F_NF_S_bottomup+none", "CPI_F_NF_S+Ki", "CPI_F_NF_S+Ruonia"]
    comp = metrics[(metrics.spec.isin(short)) & (metrics.window.isin(["AGG_all", "AGG_oos_nonshock"]))]
    comp = comp.pivot_table(index="spec", columns=["window", "horizon"], values="MAE").round(4)
    comp.to_csv(out_dir / "comparison.csv")

    # leakage checks
    lk = []
    for mname, (endog, macro, mode) in list(traj_models.items())[:4]:
        td = pd.Timestamp("2024-06-01"); cutoff = td - pd.DateOffset(months=1)
        tf = official[official.index <= cutoff]
        p = varx_point(tf, endog, macro, 1, 1, mode)
        tamp = official.copy()
        for c in official.columns:
            tamp.loc[tamp.index >= td, c] += 999
        p2 = varx_point(tamp[tamp.index <= cutoff], endog, macro, 1, 1, mode)
        lk.append(dict(model=mname, pred_clean=round(float(p), 6), pred_tampered=round(float(p2), 6),
                       leakage_free=bool(np.isclose(p, p2))))
    pd.DataFrame(lk).to_csv(out_dir / "leakage_checks.csv", index=False)
    # selection_log: best spec per horizon
    sel = []
    for hz in [1, 2, 12]:
        sub = metrics[(metrics.horizon == hz) & (metrics.window == "AGG_oos_nonshock")].sort_values("MAE")
        for rank, (_, r) in enumerate(sub.head(5).iterrows(), 1):
            sel.append(dict(horizon=hz, rank=rank, spec=r["spec"], MAE_oos_nonshock=round(r["MAE"], 4)))
    pd.DataFrame(sel).to_csv(out_dir / "selection_log.csv", index=False)

    config = dict(audit="opus48 parsimonious trajectory VAR", endog_sets=ENDOG_SETS, macro_sets=MACRO_SETS,
                  partA="endog x macro grid, VARX lag1 last_obs exog, h=1/2/12",
                  partB="exog path modes last_obs/ar1/recent_mean on CPI_F_NF_S+USD_Ki_Ruonia h=12",
                  partC=dict(models=list(traj_models.keys()),
                             fixed_cutoffs=[str(c.date()) for c in fixed_cutoffs]),
                  windows={k: [str(v[0].date()), str(v[-1].date())] for k, v in windows.items()})
    (out_dir / "config.json").write_text(json.dumps(config, indent=2, default=str), encoding="utf-8")

    pd.set_option("display.width", 240)
    print("\n=== PART A: best specs by horizon (AGG_oos_nonshock MAE) ===")
    for hz in [1, 2, 12]:
        sub = metrics[(metrics.horizon == hz) & (metrics.window == "AGG_oos_nonshock")].sort_values("MAE")
        print(f"-- h={hz} top6 --")
        print(sub[["spec", "MAE", "KPI_Viol"]].head(6).to_string(index=False))
    print("\n=== PART B exog path (h=12) ===")
    print(exog_metrics.to_string(index=False))
    print("\n=== PART C trajectory realism (cutoff 2024-11) ===")
    print(traj[traj.cutoff == "2024-11-01"][["model", "traj_MAE", "path_std", "vol_ratio",
          "sign_changes", "max_jump", "seasonal_corr", "explosive"]].round(3).to_string(index=False))
    print("\nSaved to", out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
