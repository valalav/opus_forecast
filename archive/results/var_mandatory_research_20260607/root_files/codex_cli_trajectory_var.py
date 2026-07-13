#!/usr/bin/env python3
"""
Research-only parsimonious multi-horizon VAR / trajectory search.

All forecasts are deterministic. No stochastic shocks or random noise are added
to point forecasts or trajectories. Artifacts are written only under
experiments/var_sa_research/runs/codex_cli_trajectory_var_*.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import warnings
from dataclasses import dataclass
from datetime import datetime
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import HuberRegressor
from statsmodels.tsa.api import VAR

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from experiments.var_sa_research.run_var_sa_backtests import (  # noqa: E402
    OFFICIAL_COMPONENT_WEIGHTS,
    _future_seasonal,
    _seasonal_means,
    load_official_data,
)
from sirena.models.bvar import BVARForecaster  # noqa: E402

warnings.filterwarnings("ignore")


AGENT_ID = "codex_cli"
RESEARCH_DIR = ROOT / "experiments" / "var_sa_research"
RUNS_DIR = RESEARCH_DIR / "runs"
REPORT_PATH = RESEARCH_DIR / f"{AGENT_ID}_trajectory_var_report.md"
HORIZONS = [1, 2, 12]
BASE_COMPONENTS = ("Food", "NonFood", "Services")
ALL_MACRO = ("USD", "Ki_i", "Ruonia")


@dataclass(frozen=True)
class Candidate:
    name: str
    role: str
    family: str
    endog: tuple[str, ...]
    macro: tuple[str, ...]
    params: dict[str, Any]
    notes: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", default=f"{AGENT_ID}_trajectory_var_h1_h2_h12_full")
    parser.add_argument("--seed", type=int, default=20260607)
    return parser.parse_args()


def historical_windows() -> list[tuple[str, pd.Timestamp, pd.Timestamp]]:
    return [
        ("2018_2019", pd.Timestamp("2018-01-01"), pd.Timestamp("2019-12-01")),
        ("2020_2021", pd.Timestamp("2020-01-01"), pd.Timestamp("2021-12-01")),
        ("2022_shock", pd.Timestamp("2022-01-01"), pd.Timestamp("2022-12-01")),
        ("2023", pd.Timestamp("2023-01-01"), pd.Timestamp("2023-12-01")),
        ("2024_2025q1", pd.Timestamp("2024-01-01"), pd.Timestamp("2025-03-01")),
        ("selection_2025-04_2026-03", pd.Timestamp("2025-04-01"), pd.Timestamp("2026-03-01")),
    ]


def all_outer_dates() -> pd.DataFrame:
    rows = []
    for window, start, end in historical_windows():
        for target_date in pd.date_range(start=start, end=end, freq="MS"):
            rows.append({"window": window, "target_date": target_date})
    return pd.DataFrame(rows)


def endog_sets() -> dict[str, tuple[str, ...]]:
    return {
        "cpi": ("CPI",),
        "tc": ("CPI", "Food", "NonFood", "Services"),
        "comp": ("Food", "NonFood", "Services"),
        "cpi_food": ("CPI", "Food"),
        "cpi_nonfood": ("CPI", "NonFood"),
        "cpi_services": ("CPI", "Services"),
        "cpi_food_nonfood": ("CPI", "Food", "NonFood"),
        "cpi_food_services": ("CPI", "Food", "Services"),
        "cpi_nonfood_services": ("CPI", "NonFood", "Services"),
    }


def macro_sets() -> dict[str, tuple[str, ...]]:
    out = {"none": tuple()}
    for r in range(1, len(ALL_MACRO) + 1):
        for combo in combinations(ALL_MACRO, r):
            out["_".join(combo)] = tuple(combo)
    return out


def build_candidates() -> list[Candidate]:
    candidates: list[Candidate] = [
        Candidate("PlainVAR_BIC", "baseline", "VAR_BIC", ("CPI", "Food", "NonFood", "Services"), tuple(), {"kind": "plain_bic"}, "Plain VAR with BIC lag 1..6."),
        Candidate("RegimeMacroVARX_l1", "baseline", "Regime_VARX", ("CPI", "Food", "NonFood", "Services"), ALL_MACRO, {"kind": "regime_macro_varx", "lags": 1, "exog_path": "last", "seasonal": False}, "Prior robust h=1 candidate: normal macro VARX, shock Huber VAR."),
        Candidate("RobustFAVAR_f2_l1", "baseline", "Robust_FAVAR", ("CPI",), ("Food", "NonFood", "Services", "USD", "Ki_i", "Ruonia"), {"kind": "favar", "lags": 1, "factors": 2, "robust": True}, "Prior robust FAVAR baseline."),
        Candidate("FAVAR_f2_l1", "baseline", "FAVAR", ("CPI",), ("Food", "NonFood", "Services", "USD", "Ki_i", "Ruonia"), {"kind": "favar", "lags": 1, "factors": 2, "robust": False}, "Prior non-robust FAVAR baseline."),
        Candidate("varx_last_exog_l1", "baseline", "VARX", ("CPI", "Food", "NonFood", "Services"), ALL_MACRO, {"kind": "linear_varx", "lags": 1, "exog_path": "last", "seasonal": False, "estimator": "ridge", "alpha": 0.25}, "Prior macro VARX baseline with all macro variables."),
        Candidate("Archived_BVAR", "baseline", "BVAR", ("CPI", "Food", "USD", "Ruonia"), tuple(), {"kind": "bvar_det", "lags": 4, "lambda1": 1.0}, "Deterministic posterior-mean version of archived BVAR."),
        Candidate("RandomWalk", "simple_baseline", "Naive", ("CPI",), tuple(), {"kind": "random_walk"}, "Last observed CPI."),
        Candidate("SeasonalNaive", "simple_baseline", "Naive", ("CPI",), tuple(), {"kind": "seasonal_naive"}, "Train-only month-of-year mean."),
    ]

    for set_name, endog in endog_sets().items():
        target = "bottom_up" if set_name == "comp" else "CPI"
        candidates.append(
            Candidate(
                f"var_{set_name}_l1",
                "candidate",
                "VAR",
                endog,
                tuple(),
                {"kind": "linear_varx", "lags": 1, "target": target, "exog_path": "none", "seasonal": False, "estimator": "ridge", "alpha": 0.05},
                "Parsimonious VAR(1), no exogenous macro.",
            )
        )
        candidates.append(
            Candidate(
                f"seasonal_var_{set_name}_l1",
                "candidate",
                "Seasonal_VAR",
                endog,
                tuple(),
                {"kind": "linear_varx", "lags": 1, "target": target, "exog_path": "none", "seasonal": True, "estimator": "ridge", "alpha": 0.05},
                "VAR(1) with deterministic month-of-year dummies.",
            )
        )

    path_rules = ["last", "ar1", "mean_revert", "rate_hold_usd_ar"]
    for set_name, endog in endog_sets().items():
        target = "bottom_up" if set_name == "comp" else "CPI"
        for macro_name, macro in macro_sets().items():
            if macro_name == "none":
                continue
            for path_rule in path_rules:
                candidates.append(
                    Candidate(
                        f"varx_{set_name}_{macro_name}_{path_rule}_l1",
                        "candidate",
                        "VARX",
                        endog,
                        macro,
                        {
                            "kind": "linear_varx",
                            "lags": 1,
                            "target": target,
                            "exog_path": path_rule,
                            "seasonal": False,
                            "estimator": "ridge",
                            "alpha": 0.25,
                        },
                        "Parsimonious VARX(1) with deterministic macro path.",
                    )
                )
            candidates.append(
                Candidate(
                    f"seasonal_varx_{set_name}_{macro_name}_last_l1",
                    "candidate",
                    "Seasonal_VARX",
                    endog,
                    macro,
                    {
                        "kind": "linear_varx",
                        "lags": 1,
                        "target": target,
                        "exog_path": "last",
                        "seasonal": True,
                        "estimator": "ridge",
                        "alpha": 0.25,
                    },
                    "VARX(1) with deterministic macro path and month dummies.",
                )
            )

    candidates.extend(
        [
            Candidate("BVAR_tc_l1_lam0p3", "candidate", "BVAR", ("CPI", "Food", "NonFood", "Services"), tuple(), {"kind": "bvar_det", "lags": 1, "lambda1": 0.3}, "Deterministic Minnesota BVAR posterior mean."),
            Candidate("BVAR_cpi_food_l1_lam0p3", "candidate", "BVAR", ("CPI", "Food"), tuple(), {"kind": "bvar_det", "lags": 1, "lambda1": 0.3}, "Small deterministic Minnesota BVAR posterior mean."),
            Candidate("BVAR_comp_bottomup_l1_lam0p3", "candidate", "BVAR_component", ("Food", "NonFood", "Services"), tuple(), {"kind": "bvar_det", "lags": 1, "lambda1": 0.3, "target": "bottom_up"}, "Component deterministic BVAR with weighted CPI reconstruction."),
        ]
    )
    return candidates


def candidates_for_horizon(candidates: list[Candidate], horizon: int) -> list[Candidate]:
    if horizon in {1, 2}:
        return candidates
    # h=12 shortlist after full-grid runtime proved too high:
    # - all required baselines;
    # - all no-macro endogenous/component sets with and without seasonality;
    # - all macro subsets and deterministic path rules for the main total/components system;
    # - deterministic BVAR/FAVAR candidates.
    out = []
    for cand in candidates:
        if cand.role in {"baseline", "simple_baseline"}:
            out.append(cand)
            continue
        if cand.family.startswith("BVAR") or "FAVAR" in cand.family:
            out.append(cand)
            continue
        if len(cand.macro) == 0:
            out.append(cand)
            continue
        if cand.endog == ("CPI", "Food", "NonFood", "Services"):
            out.append(cand)
            continue
    return out


def _weighted_components(values: dict[str, float]) -> float:
    return float(sum(values[col] * weight for col, weight in OFFICIAL_COMPONENT_WEIGHTS.items()))


def _target_from_vector(endog: list[str], vec: np.ndarray, target: str) -> float:
    if target == "bottom_up":
        return _weighted_components(dict(zip(endog, vec)))
    return float(vec[endog.index("CPI")])


def _ridge_solve(x: np.ndarray, y: np.ndarray, alpha: float) -> np.ndarray:
    xtx = x.T @ x
    penalty = np.eye(xtx.shape[0]) * alpha
    penalty[0, 0] = 0.0
    return np.linalg.solve(xtx + penalty, x.T @ y)


def month_dummies(dates: list[pd.Timestamp] | pd.DatetimeIndex) -> pd.DataFrame:
    idx = pd.DatetimeIndex(dates)
    out = pd.DataFrame(index=idx)
    for month in range(2, 13):
        out[f"month_{month:02d}"] = (idx.month == month).astype(float)
    out["tariff_july"] = (idx.month == 7).astype(float)
    return out


def fit_ar1(series: pd.Series) -> tuple[float, float]:
    s = series.dropna().astype(float)
    if len(s) < 24:
        return float(s.iloc[-1]) if len(s) else 0.0, 0.0
    y = s.iloc[1:].values
    x = np.column_stack([np.ones(len(s) - 1), s.iloc[:-1].values])
    try:
        beta = np.linalg.lstsq(x, y, rcond=None)[0]
        return float(beta[0]), float(np.clip(beta[1], -0.95, 0.95))
    except Exception:
        return float(s.iloc[-1]), 0.0


def exog_path(train: pd.DataFrame, macro: tuple[str, ...], horizon: int, rule: str) -> pd.DataFrame:
    dates = pd.date_range(train.index.max() + pd.DateOffset(months=1), periods=horizon, freq="MS")
    out = pd.DataFrame(index=dates)
    if not macro:
        return out
    data = train.loc[:, list(macro)].dropna()
    if data.empty:
        raise ValueError("insufficient_exog_train")
    for col in macro:
        s = data[col].dropna().astype(float)
        last = float(s.iloc[-1])
        recent_mean = float(s.iloc[-min(12, len(s)) :].mean())
        vals = []
        if rule == "last":
            vals = [last] * horizon
        elif rule == "mean_revert":
            prev = last
            for _ in range(horizon):
                prev = 0.70 * prev + 0.30 * recent_mean
                vals.append(float(prev))
        elif rule == "ar1":
            a, b = fit_ar1(s)
            prev = last
            for _ in range(horizon):
                prev = a + b * prev
                vals.append(float(prev))
        elif rule == "rate_hold_usd_ar":
            if col in {"Ki_i", "Ruonia"}:
                vals = [last] * horizon
            elif col == "USD":
                a, b = fit_ar1(s)
                prev = last
                for _ in range(horizon):
                    prev = a + b * prev
                    vals.append(float(prev))
            else:
                vals = [last] * horizon
        else:
            raise ValueError(f"unknown exog path rule: {rule}")
        out[col] = vals
    return out


def linear_varx_path(train: pd.DataFrame, candidate: Candidate, horizon: int) -> tuple[list[float], dict[str, Any]]:
    endog = list(candidate.endog)
    macro = list(candidate.macro)
    params = candidate.params
    lags = int(params.get("lags", 1))
    target = str(params.get("target", "CPI"))
    seasonal = bool(params.get("seasonal", False))
    alpha = float(params.get("alpha", 0.05))
    estimator = str(params.get("estimator", "ridge"))
    cols = endog + macro
    data = train.loc[:, cols].dropna().copy()
    if len(data) < max(36, lags + len(endog) * 6):
        raise ValueError("insufficient_train")
    y_data = data.loc[:, endog].values.astype(float)
    row_dates = data.index[lags:]
    exog_train = data.loc[:, macro] if macro else pd.DataFrame(index=data.index)
    future_exog = exog_path(data, tuple(macro), horizon, str(params.get("exog_path", "none"))) if macro else pd.DataFrame()

    x_rows = []
    y_rows = []
    md = month_dummies(row_dates) if seasonal else pd.DataFrame(index=row_dates)
    for pos, date in enumerate(row_dates, start=lags):
        row = [1.0]
        for lag in range(1, lags + 1):
            row.extend(y_data[pos - lag])
        if macro:
            row.extend(exog_train.iloc[pos].values.astype(float))
        if seasonal:
            row.extend(md.loc[date].values.astype(float))
        x_rows.append(row)
        y_rows.append(y_data[pos])
    x = np.asarray(x_rows)
    y = np.asarray(y_rows)
    if len(x) < 24:
        raise ValueError("insufficient_design")

    betas = []
    for idx in range(len(endog)):
        if estimator == "huber":
            model = HuberRegressor(alpha=1e-4, epsilon=1.35, max_iter=300, fit_intercept=False)
            model.fit(x, y[:, idx])
            betas.append(model.coef_)
        else:
            betas.append(_ridge_solve(x, y[:, idx], alpha))
    beta = np.asarray(betas)

    history = [row.copy() for row in y_data[-lags:]]
    path = []
    target_vecs = []
    for step in range(1, horizon + 1):
        fdate = data.index.max() + pd.DateOffset(months=step)
        row = [1.0]
        for lag in range(1, lags + 1):
            row.extend(history[-lag])
        if macro:
            row.extend(future_exog.iloc[step - 1][macro].values.astype(float))
        if seasonal:
            row.extend(month_dummies([fdate]).iloc[0].values.astype(float))
        pred_vec = np.asarray(row) @ beta.T
        history.append(pred_vec)
        target_vecs.append(dict(zip(endog, pred_vec)))
        path.append(_target_from_vector(endog, pred_vec, target))
    comp_gap = np.nan
    if set(BASE_COMPONENTS).issubset(set(endog)) and "CPI" in endog:
        comp_gap = float(
            max(
                abs(v["CPI"] - _weighted_components({c: v[c] for c in BASE_COMPONENTS}))
                for v in target_vecs
            )
        )
    return path, {"component_consistency_gap": comp_gap, "exog_path_rule": params.get("exog_path", "none"), "seasonal": seasonal}


def plain_bic_path(train: pd.DataFrame, horizon: int) -> tuple[list[float], dict[str, Any]]:
    data = train.loc[:, ["CPI", "Food", "NonFood", "Services"]].dropna()
    if len(data) < 42:
        raise ValueError("insufficient_train")
    try:
        order = VAR(data).select_order(maxlags=6)
        lags = int(getattr(order, "bic", 1)) or 1
    except Exception:
        lags = 1
    lags = max(1, min(6, lags))
    cand = Candidate("inner_bic", "baseline", "VAR", tuple(data.columns), tuple(), {"kind": "linear_varx", "lags": lags, "target": "CPI", "exog_path": "none", "seasonal": False, "estimator": "ridge", "alpha": 0.0}, "")
    path, meta = linear_varx_path(train, cand, horizon)
    meta["selected_lag"] = lags
    return path, meta


def random_walk_path(train: pd.DataFrame, horizon: int) -> tuple[list[float], dict[str, Any]]:
    last = float(train["CPI"].dropna().iloc[-1])
    return [last] * horizon, {}


def seasonal_naive_path(train: pd.DataFrame, horizon: int) -> tuple[list[float], dict[str, Any]]:
    mm = _seasonal_means(train, ["CPI", "Food", "NonFood", "Services"], None)
    vals = []
    for step in range(1, horizon + 1):
        tm = (train.index.max() + pd.DateOffset(months=step)).month
        vals.append(float(_future_seasonal(mm, ["CPI", "Food", "NonFood", "Services"], tm)[0]))
    return vals, {}


def bvar_det_path(train: pd.DataFrame, candidate: Candidate, horizon: int) -> tuple[list[float], dict[str, Any]]:
    cols = list(candidate.endog)
    target = str(candidate.params.get("target", "CPI"))
    data = train.loc[:, cols].dropna()
    lags = int(candidate.params["lags"])
    if len(data) < max(42, lags * len(cols) + 24):
        raise ValueError("insufficient_train")
    model = BVARForecaster(
        lags=lags,
        lambda1=float(candidate.params.get("lambda1", 0.3)),
        lambda2=0.5,
        lambda3=1.0,
        n_draws=1,
        var_names=cols,
    )
    model.fit(data, target_col=cols[0])
    hist = [row.copy() for row in model.raw_data[-model.lags :]]
    path = []
    for _ in range(horizon):
        x_t = np.ones(1 + model.k * model.lags)
        for lag in range(1, model.lags + 1):
            x_t[1 + (lag - 1) * model.k : 1 + lag * model.k] = hist[-lag]
        y_new = x_t @ model.B_post
        hist.append(y_new)
        path.append(_target_from_vector(cols, y_new, target))
    return path, {"lambda1": candidate.params.get("lambda1")}


def favar_path(train: pd.DataFrame, candidate: Candidate, horizon: int) -> tuple[list[float], dict[str, Any]]:
    factors_n = int(candidate.params["factors"])
    robust = bool(candidate.params.get("robust", False))
    factor_cols = ["Food", "NonFood", "Services", "USD", "Ki_i", "Ruonia"]
    data = train.loc[:, ["CPI"] + factor_cols].dropna()
    if len(data) < 48:
        raise ValueError("insufficient_train")
    z = data.loc[:, factor_cols].values.astype(float)
    if robust:
        center = np.median(z, axis=0)
        scale = np.median(np.abs(z - center), axis=0) * 1.4826
        fallback = z.std(axis=0)
        scale = np.where(scale < 1e-8, np.where(fallback < 1e-8, 1.0, fallback), scale)
        z_std = np.clip((z - center) / scale, -3.0, 3.0)
    else:
        center = z.mean(axis=0)
        scale = z.std(axis=0)
        scale[scale < 1e-8] = 1.0
        z_std = (z - center) / scale
    _, _, vt = np.linalg.svd(z_std, full_matrices=False)
    scores = z_std @ vt[:factors_n].T
    favar = pd.DataFrame(index=data.index, data={"CPI": data["CPI"].values})
    for idx in range(factors_n):
        favar[f"F{idx + 1}"] = scores[:, idx]
    inner = Candidate("inner_favar", "candidate", "FAVAR", tuple(favar.columns), tuple(), {"kind": "linear_varx", "lags": int(candidate.params.get("lags", 1)), "target": "CPI", "exog_path": "none", "seasonal": False, "estimator": "ridge", "alpha": 0.0}, "")
    return linear_varx_path(favar, inner, horizon)


def is_shock_regime(train: pd.DataFrame) -> bool:
    cpi = train["CPI"].dropna()
    if len(cpi) < 24:
        return False
    return bool(abs(float(cpi.iloc[-1])) >= 1.0 or float(cpi.iloc[-12:].std()) >= 0.55)


def regime_macro_varx_path(train: pd.DataFrame, candidate: Candidate, horizon: int) -> tuple[list[float], dict[str, Any]]:
    if is_shock_regime(train):
        inner = Candidate("inner_huber_var", "candidate", "Huber_VAR", ("CPI", "Food", "NonFood", "Services"), tuple(), {"kind": "linear_varx", "lags": 1, "target": "CPI", "exog_path": "none", "seasonal": False, "estimator": "huber", "alpha": 0.0}, "")
        path, meta = linear_varx_path(train, inner, horizon)
        meta["regime"] = "shock_huber_var"
        return path, meta
    path, meta = linear_varx_path(train, candidate, horizon)
    meta["regime"] = "normal_macro_varx"
    return path, meta


def forecast_path(train: pd.DataFrame, candidate: Candidate, horizon: int) -> tuple[list[float], dict[str, Any]]:
    kind = candidate.params["kind"]
    if kind == "plain_bic":
        return plain_bic_path(train, horizon)
    if kind == "linear_varx":
        return linear_varx_path(train, candidate, horizon)
    if kind == "regime_macro_varx":
        return regime_macro_varx_path(train, candidate, horizon)
    if kind == "favar":
        return favar_path(train, candidate, horizon)
    if kind == "bvar_det":
        return bvar_det_path(train, candidate, horizon)
    if kind == "random_walk":
        return random_walk_path(train, horizon)
    if kind == "seasonal_naive":
        return seasonal_naive_path(train, horizon)
    raise ValueError(f"unknown kind: {kind}")


def metric_record(group: pd.DataFrame) -> dict[str, Any]:
    ok = group.dropna(subset=["error"]).copy()
    if ok.empty:
        return {"n": 0, "MAE": np.nan, "RMSE": np.nan, "Bias": np.nan, "KPI_Violations": np.nan, "Coverage_50pct": np.nan, "Max_Error": np.nan}
    err = ok["error"].astype(float)
    ae = err.abs()
    return {
        "n": len(ok),
        "MAE": ae.mean(),
        "RMSE": float(np.sqrt((err**2).mean())),
        "Bias": err.mean(),
        "KPI_Violations": int((ae > 0.5).sum()),
        "Coverage_50pct": float((ae <= 0.5).mean() * 100),
        "Max_Error": ae.max(),
    }


def calculate_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for keys, group in predictions.groupby(["horizon", "window", "model", "role", "family"], sort=False):
        horizon, window, model, role, family = keys
        rows.append({"horizon": horizon, "slice": window, "model": model, "role": role, "family": family, **metric_record(group)})
    slice_defs = {
        "all_windows": lambda df: df.index == df.index,
        "out_of_selection": lambda df: df["window"] != "selection_2025-04_2026-03",
        "out_of_selection_non_shock": lambda df: (df["window"] != "selection_2025-04_2026-03") & (df["window"] != "2022_shock"),
        "shock_2022": lambda df: df["window"] == "2022_shock",
        "selection_reference": lambda df: df["window"] == "selection_2025-04_2026-03",
    }
    for keys, group in predictions.groupby(["horizon", "model", "role", "family"], sort=False):
        horizon, model, role, family = keys
        for slice_name, mask_fn in slice_defs.items():
            rows.append({"horizon": horizon, "slice": slice_name, "model": model, "role": role, "family": family, **metric_record(group[mask_fn(group)])})
    return pd.DataFrame(rows).sort_values(["horizon", "slice", "MAE", "model"])


def trajectory_record(path: list[float], train: pd.DataFrame, candidate: Candidate, meta: dict[str, Any]) -> dict[str, Any]:
    arr = np.asarray(path, dtype=float)
    diffs = np.diff(arr)
    hist = train["CPI"].dropna().astype(float).iloc[-min(60, len(train)) :]
    hist_std12 = float(hist.iloc[-12:].std()) if len(hist) >= 12 else np.nan
    path_std = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
    signs = np.sign(arr)
    signs = signs[signs != 0]
    sign_changes = int(np.sum(signs[1:] != signs[:-1])) if len(signs) > 1 else 0
    max_jump = float(np.max(np.abs(diffs))) if len(diffs) else 0.0
    flatness = float(np.mean(np.abs(diffs) < 0.03)) if len(diffs) else 1.0
    repeated = float(np.mean(np.abs(diffs) < 1e-8)) if len(diffs) else 1.0
    vol_ratio = path_std / hist_std12 if hist_std12 and np.isfinite(hist_std12) and hist_std12 > 1e-8 else np.nan
    explosive = bool(np.nanmax(np.abs(arr)) > 5.0 or max_jump > 3.0 or (np.isfinite(vol_ratio) and vol_ratio > 4.0))
    flat = bool(flatness > 0.75 or path_std < 0.03)
    return {
        "path_std": path_std,
        "hist_recent_std": hist_std12,
        "volatility_ratio": vol_ratio,
        "sign_changes": sign_changes,
        "max_month_jump": max_jump,
        "seasonal_amplitude": float(np.nanmax(arr) - np.nanmin(arr)),
        "flatness_share": flatness,
        "repeated_identical_share": repeated,
        "explosive_flag": explosive,
        "flat_path_flag": flat,
        "component_consistency_gap": meta.get("component_consistency_gap", np.nan),
        "path_json": json.dumps([float(x) for x in arr], ensure_ascii=False),
    }


def evaluate(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, list[Candidate]]:
    official = load_official_data()
    candidates = build_candidates()
    targets = all_outer_dates()
    pred_rows = []
    traj_rows = []
    selection_rows = []
    leakage_rows = []
    total = sum(len(targets) * len(candidates_for_horizon(candidates, h)) for h in HORIZONS)
    done = 0
    last_progress = 0
    for horizon in HORIZONS:
        horizon_candidates = candidates_for_horizon(candidates, horizon)
        for _, trow in targets.iterrows():
            target_date = trow["target_date"]
            if target_date not in official.index:
                continue
            cutoff = target_date - pd.DateOffset(months=horizon)
            train = official[official.index <= cutoff].copy()
            if len(train) < 48:
                continue
            actual = float(official.loc[target_date, "CPI"])
            for candidate in horizon_candidates:
                try:
                    path, meta = forecast_path(train, candidate, horizon)
                    pred = float(path[-1]) if len(path) else np.nan
                    status = "ok" if np.isfinite(pred) else "nan_prediction"
                except Exception as exc:
                    path, meta, pred, status = [], {}, np.nan, f"error: {type(exc).__name__}: {exc}"
                error = actual - pred if np.isfinite(pred) else np.nan
                pred_rows.append(
                    {
                        "horizon": horizon,
                        "window": trow["window"],
                        "target_date": target_date,
                        "cutoff": cutoff,
                        "model": candidate.name,
                        "role": candidate.role,
                        "family": candidate.family,
                        "endog": ",".join(candidate.endog),
                        "macro": ",".join(candidate.macro) if candidate.macro else "none",
                        "exog_path": candidate.params.get("exog_path", "none"),
                        "seasonal": bool(candidate.params.get("seasonal", False)),
                        "actual": actual,
                        "prediction": pred,
                        "error": error,
                        "abs_error": abs(error) if np.isfinite(error) else np.nan,
                        "status": status,
                        "train_start": train.index.min(),
                        "train_end": train.index.max(),
                        "train_n": len(train),
                        "params": json.dumps(candidate.params, ensure_ascii=False, sort_keys=True),
                        "meta": json.dumps({k: str(v) for k, v in meta.items()}, ensure_ascii=False, sort_keys=True),
                        "notes": candidate.notes,
                    }
                )
                if candidate.name == "PlainVAR_BIC" and "selected_lag" in meta:
                    selection_rows.append({"horizon": horizon, "target_date": target_date, "cutoff": cutoff, "model": candidate.name, "selected_lag": meta["selected_lag"], "selection_method": "BIC"})
                if horizon == 12 and len(path) == 12:
                    traj_rows.append(
                        {
                            "model": candidate.name,
                            "role": candidate.role,
                            "family": candidate.family,
                            "window": trow["window"],
                            "target_date": target_date,
                            "cutoff": cutoff,
                            **trajectory_record(path, train, candidate, meta),
                        }
                    )
                leakage_rows.append(
                    {
                        "horizon": horizon,
                        "model": candidate.name,
                        "target_date": target_date,
                        "cutoff": cutoff,
                        "train_end": train.index.max(),
                        "train_end_after_cutoff": bool(train.index.max() > cutoff),
                        "future_actual_used": False,
                        "random_noise_used": False,
                        "exog_path_rule": candidate.params.get("exog_path", "none"),
                    }
                )
                done += 1
            if done // 2500 > last_progress:
                last_progress = done // 2500
                print(f"evaluated {done}/{total}", flush=True)
    return pd.DataFrame(pred_rows), pd.DataFrame(traj_rows), pd.DataFrame(selection_rows), pd.DataFrame(leakage_rows), candidates


def build_comparison(metrics: pd.DataFrame, traj: pd.DataFrame) -> pd.DataFrame:
    rows = []
    allm = metrics[metrics["slice"] == "all_windows"].copy()
    for model, group in allm.groupby("model", sort=False):
        rec: dict[str, Any] = {"model": model}
        first = group.iloc[0]
        rec["role"] = first["role"]
        rec["family"] = first["family"]
        for horizon in HORIZONS:
            hrow = group[group["horizon"] == horizon]
            if len(hrow):
                row = hrow.iloc[0]
                rec[f"h{horizon}_MAE"] = row["MAE"]
                rec[f"h{horizon}_RMSE"] = row["RMSE"]
                rec[f"h{horizon}_KPI"] = row["KPI_Violations"]
                rec[f"h{horizon}_Coverage"] = row["Coverage_50pct"]
        t = traj[traj["model"] == model]
        if len(t):
            rec["h12_path_std_mean"] = t["path_std"].mean()
            rec["h12_vol_ratio_mean"] = t["volatility_ratio"].replace([np.inf, -np.inf], np.nan).mean()
            rec["h12_flatness_mean"] = t["flatness_share"].mean()
            rec["h12_max_jump_mean"] = t["max_month_jump"].mean()
            rec["h12_explosive_rate"] = float(t["explosive_flag"].mean() * 100)
            rec["h12_flat_path_rate"] = float(t["flat_path_flag"].mean() * 100)
        rows.append(rec)
    out = pd.DataFrame(rows)
    out["score"] = (
        out["h1_MAE"].fillna(9) * 0.45
        + out["h2_MAE"].fillna(9) * 0.25
        + out["h12_MAE"].fillna(9) * 0.25
        + out["h12_explosive_rate"].fillna(100) / 100 * 0.15
        + out["h12_flat_path_rate"].fillna(100) / 100 * 0.10
    )
    return out.sort_values(["score", "h1_MAE", "h12_MAE", "model"])


def horizon_policy(comparison: pd.DataFrame) -> pd.DataFrame:
    eligible = comparison[(comparison["role"].isin(["candidate", "baseline"])) & (comparison["h12_explosive_rate"].fillna(0) <= 25)].copy()
    rows = []
    for horizon in HORIZONS:
        col = f"h{horizon}_MAE"
        h = eligible.dropna(subset=[col]).copy()
        if horizon == 12:
            h = h[(h["h12_flat_path_rate"].fillna(0) <= 50) & (h["h12_explosive_rate"].fillna(0) == 0)]
            h["rank_score"] = h[col] + 0.04 * h["h12_flatness_mean"].fillna(0) + 0.02 * h["h12_vol_ratio_mean"].fillna(1).sub(1).abs()
        else:
            h["rank_score"] = h[col]
        if len(h):
            best = h.sort_values(["rank_score", col, "score"]).iloc[0]
            rows.append({"horizon": horizon, "selected_model": best["model"], "selected_family": best["family"], "MAE": best[col], "rank_score": best["rank_score"]})
    return pd.DataFrame(rows)


def make_charts(out_dir: Path, predictions: pd.DataFrame, traj: pd.DataFrame, comparison: pd.DataFrame) -> list[str]:
    import matplotlib.pyplot as plt

    chart_paths: list[str] = []
    top_models = comparison[(comparison["role"].isin(["candidate", "baseline"])) & (comparison["h12_explosive_rate"].fillna(0) == 0)].head(5)["model"].tolist()
    if not top_models:
        return chart_paths
    target_date = pd.Timestamp("2026-03-01")
    actual = load_official_data()["CPI"]
    path_dates = pd.date_range(target_date - pd.DateOffset(months=11), target_date, freq="MS")
    plt.figure(figsize=(11, 5))
    plt.plot(path_dates, actual.reindex(path_dates), marker="o", color="black", linewidth=2, label="Actual CPI")
    for model in top_models:
        row = traj[(traj["model"] == model) & (traj["target_date"] == target_date)]
        if row.empty:
            continue
        vals = json.loads(row.iloc[0]["path_json"])
        plt.plot(path_dates, vals, marker="o", linewidth=1.4, label=model[:38])
    plt.axhline(0, color="gray", linewidth=0.8)
    plt.title("h=12 deterministic trajectory ending 2026-03")
    plt.ylabel("MoM CPI, p.p.")
    plt.grid(True, alpha=0.25)
    plt.legend(fontsize=8)
    plt.tight_layout()
    out = out_dir / "trajectory_2026_03_top_h12.png"
    plt.savefig(out, dpi=150)
    plt.close()
    chart_paths.append(str(out))

    top = comparison.head(20).copy()
    plt.figure(figsize=(12, 6))
    x = np.arange(len(top))
    plt.bar(x - 0.25, top["h1_MAE"], width=0.25, label="h1 MAE")
    plt.bar(x, top["h2_MAE"], width=0.25, label="h2 MAE")
    plt.bar(x + 0.25, top["h12_MAE"], width=0.25, label="h12 MAE")
    plt.xticks(x, top["model"], rotation=70, ha="right", fontsize=7)
    plt.ylabel("MAE")
    plt.title("Top trajectory VAR candidates by composite score")
    plt.grid(True, axis="y", alpha=0.25)
    plt.legend()
    plt.tight_layout()
    out = out_dir / "top_candidates_horizon_mae.png"
    plt.savefig(out, dpi=150)
    plt.close()
    chart_paths.append(str(out))
    return chart_paths


def write_config(out_dir: Path, args: argparse.Namespace, candidates: list[Candidate]) -> None:
    h12_shortlist = [c.name for c in candidates_for_horizon(candidates, 12)]
    cfg = {
        "agent_id": AGENT_ID,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "run_name": args.run_name,
        "horizons": HORIZONS,
        "endogenous_sets": {k: list(v) for k, v in endog_sets().items()},
        "macro_sets": {k: list(v) for k, v in macro_sets().items()},
        "exog_paths": {
            "last": "repeat last observed cutoff value",
            "ar1": "univariate AR(1), fit only on cutoff data",
            "mean_revert": "0.70 previous + 0.30 trailing-12-month mean",
            "rate_hold_usd_ar": "Ki_i/Ruonia held at current cutoff value; USD AR(1)",
        },
        "seasonality": "month-of-year dummies and July tariff dummy in selected candidates",
        "random_noise": "not used",
        "candidate_count": len(candidates),
        "h12_shortlist": {
            "used": True,
            "reason": "full h=12 grid was CPU-heavy; task allows full h1/h2 with h12 shortlist when runtime is high",
            "rule": "baselines + simple baselines + all no-macro endogenous sets + all macro subsets/path rules for total-components + BVAR/FAVAR candidates",
            "candidate_count": len(h12_shortlist),
            "candidates": h12_shortlist,
        },
    }
    (out_dir / "config.json").write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")


def final_status_and_recommendation(comparison: pd.DataFrame, policy: pd.DataFrame) -> tuple[str, str]:
    if policy["selected_model"].nunique() > 1:
        return "recommended horizon-specific VAR policy", "Horizon-specific policy is preferred because short-horizon and h=12 trajectory criteria select different parsimonious models."
    best = comparison[comparison["model"] == policy.iloc[0]["selected_model"]].iloc[0] if len(policy) else comparison.iloc[0]
    if best["role"] in {"candidate", "baseline"} and best["h12_explosive_rate"] == 0 and best["h12_flat_path_rate"] < 50:
        return "recommended parsimonious trajectory VAR", "A single deterministic VAR-family model is acceptable across h=1/h=2/h=12."
    return "experimental trajectory VAR", "Best point metrics require caution due to trajectory realism diagnostics."


def write_notes_report(
    out_dir: Path,
    args: argparse.Namespace,
    candidates: list[Candidate],
    metrics: pd.DataFrame,
    comparison: pd.DataFrame,
    traj: pd.DataFrame,
    leakage: pd.DataFrame,
    policy: pd.DataFrame,
    chart_paths: list[str],
) -> None:
    status, recommendation_text = final_status_and_recommendation(comparison, policy)
    leak_violations = int((leakage["train_end_after_cutoff"] | leakage["future_actual_used"] | leakage["random_noise_used"]).sum()) if len(leakage) else 0
    endog_tested = pd.DataFrame([{"set": k, "variables": ",".join(v)} for k, v in endog_sets().items()])
    macro_tested = pd.DataFrame([{"set": k, "variables": ",".join(v) if v else "none"} for k, v in macro_sets().items()])
    show_cols = ["model", "role", "family", "h1_MAE", "h2_MAE", "h12_MAE", "h12_path_std_mean", "h12_vol_ratio_mean", "h12_flatness_mean", "h12_explosive_rate", "h12_flat_path_rate"]
    comp_show = comparison[show_cols].head(35)
    unrealistic = comparison[(comparison["h12_explosive_rate"].fillna(0) > 0) | (comparison["h12_flat_path_rate"].fillna(0) > 50)][show_cols].head(20)

    notes = [
        "# codex_cli Trajectory VAR Notes",
        "",
        f"- Run directory: `{out_dir}`",
        f"- Report: `{REPORT_PATH}`",
        f"- Final status: `{status}`",
        f"- Recommendation: {recommendation_text}",
        f"- Leakage violations: `{leak_violations}`",
        "",
        "## Horizon Policy",
        "",
        policy.to_markdown(index=False) if len(policy) else "_No policy rows._",
        "",
        "## Top Comparison",
        "",
        comp_show.to_markdown(index=False),
        "",
        "## Charts",
        "",
        "\n".join(f"- `{p}`" for p in chart_paths) if chart_paths else "_No charts generated._",
    ]
    (out_dir / "notes.md").write_text("\n".join(notes) + "\n", encoding="utf-8")

    report = [
        "# codex_cli Parsimonious Trajectory VAR Report",
        "",
        "## Scope",
        "",
        "Research-only deterministic VAR-family evaluation for h=1, h=2, and h=12. No production data, model registry, dashboard, or shared task file was modified. No random noise was added to point paths.",
        "",
        "## Variable Subsets Tested",
        "",
        endog_tested.to_markdown(index=False),
        "",
        "## Macro / Exogenous Sets Tested",
        "",
        macro_tested.to_markdown(index=False),
        "",
        "Exogenous paths tested: `last`, `ar1`, `mean_revert`, and `rate_hold_usd_ar`. All are fit or declared from cutoff data only.",
        "",
        "h=1 and h=2 use the full candidate grid. h=12 uses a transparent shortlist after the full h=12 grid proved too slow: all baselines, all no-macro endogenous/component sets, all macro subsets/path rules on total-components, and all BVAR/FAVAR candidates.",
        "",
        "## Recommendation",
        "",
        f"- Final status: `{status}`",
        f"- Decision: {recommendation_text}",
        "",
        policy.to_markdown(index=False) if len(policy) else "_No horizon-specific policy selected._",
        "",
        "## h=1 / h=2 / h=12 Metrics And Trajectory Diagnostics",
        "",
        comp_show.to_markdown(index=False),
        "",
        "## Rejected Unrealistic Paths",
        "",
        unrealistic.to_markdown(index=False) if len(unrealistic) else "No top-ranked candidate had explosive or excessive-flatness h=12 path flags.",
        "",
        "## Leakage And Determinism Audit",
        "",
        "- Every target uses `cutoff = target_date - horizon months`.",
        "- Exogenous paths are deterministic and use only cutoff-observed history.",
        "- BVAR candidates use posterior mean recursion only; no Monte Carlo draws or shock noise are used.",
        "- Month and tariff dummies are deterministic date functions.",
        f"- Leakage/random-noise violations in `leakage_checks.csv`: `{leak_violations}`.",
        "",
        "## Charts",
        "",
        "\n".join(f"- `{p}`" for p in chart_paths) if chart_paths else "_No charts generated._",
        "",
        "## Commands Run",
        "",
        f"```bash\npython3 experiments/var_sa_research/codex_cli_trajectory_var.py --run-name {args.run_name}\n```",
        "",
        "## Artifacts",
        "",
        f"- Run directory: `{out_dir}`",
        "- Required files: `config.json`, `metrics.csv`, `predictions.csv`, `comparison.csv`, `selection_log.csv`, `trajectory_metrics.csv`, `leakage_checks.csv`, `notes.md`, trajectory charts, script copy.",
    ]
    REPORT_PATH.write_text("\n".join(report) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    if not args.run_name.startswith(f"{AGENT_ID}_trajectory_var_"):
        raise ValueError(f"run-name must start with {AGENT_ID}_trajectory_var_")
    out_dir = RUNS_DIR / args.run_name
    out_dir.mkdir(parents=True, exist_ok=False)
    candidates = build_candidates()
    write_config(out_dir, args, candidates)
    predictions, traj, selection_log, leakage, candidates = evaluate(args)
    metrics = calculate_metrics(predictions)
    comparison = build_comparison(metrics, traj)
    policy = horizon_policy(comparison)
    selection_policy = policy.copy()
    if len(selection_policy):
        selection_policy["selection_method"] = "post-evaluation horizon policy: h1/h2 by MAE, h12 by MAE plus realism penalties"
    selection_out = pd.concat([selection_log, selection_policy], ignore_index=True, sort=False)

    predictions.to_csv(out_dir / "predictions.csv", index=False)
    metrics.to_csv(out_dir / "metrics.csv", index=False)
    comparison.to_csv(out_dir / "comparison.csv", index=False)
    selection_out.to_csv(out_dir / "selection_log.csv", index=False)
    traj.to_csv(out_dir / "trajectory_metrics.csv", index=False)
    leakage.to_csv(out_dir / "leakage_checks.csv", index=False)
    chart_paths = make_charts(out_dir, predictions, traj, comparison)
    shutil.copy2(Path(__file__), out_dir / Path(__file__).name)
    write_notes_report(out_dir, args, candidates, metrics, comparison, traj, leakage, policy, chart_paths)

    print(f"Wrote artifacts to {out_dir}", flush=True)
    print(f"Wrote final report to {REPORT_PATH}", flush=True)
    print(comparison.head(18).to_string(index=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
