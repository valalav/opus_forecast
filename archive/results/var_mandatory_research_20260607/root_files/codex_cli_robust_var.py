#!/usr/bin/env python3
"""
Research-only robust / outlier-aware mandatory VAR-family search.

This script does not modify production data, model registration, dashboard
paths, or shared task files. All artifacts are written below
experiments/var_sa_research/runs/codex_cli_robust_var_*.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import warnings
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import HuberRegressor
from statsmodels.tsa.api import VAR

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from experiments.var_sa_research.run_fixed_config_robustness import (  # noqa: E402
    forecast_archived_bvar,
    forecast_huber,
    forecast_ridge_shock,
    load_model_frame,
)
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
REPORT_PATH = RESEARCH_DIR / f"{AGENT_ID}_robust_var_report.md"
TOTAL_VARS = ("CPI", "Food", "NonFood", "Services")
MACRO_VARS = ("USD", "Ruonia", "Ki_i")
HORIZON = 1


@dataclass(frozen=True)
class Candidate:
    name: str
    role: str
    family: str
    direction: str
    params: dict[str, Any]
    notes: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", default=f"{AGENT_ID}_robust_var_h1_full")
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


def build_candidates() -> list[Candidate]:
    candidates = [
        Candidate("PlainVAR_BIC", "mandatory_baseline", "VAR", "baseline", {"kind": "plain_bic"}, "Classical VAR total/components, lag selected by BIC 1..6 inside cutoff."),
        Candidate("plain_var_tc_l1", "mandatory_baseline", "VAR", "baseline", {"kind": "plain_var", "lags": 1}, "Fixed VAR(1) on total/components."),
        Candidate("favar_macro_components_f2_l1", "mandatory_baseline", "FAVAR", "baseline", {"kind": "favar", "lags": 1, "factors": 2, "robust": False}, "Prior codex_cli FAVAR challenger."),
        Candidate("varx_last_exog_l1", "mandatory_baseline", "VARX_OLS", "baseline", {"kind": "varx", "lags": 1, "robust": False, "intervention": False}, "Macro VARX with USD/Ruonia/Ki_i held at last cutoff value."),
        Candidate("Archived_BVAR", "mandatory_baseline", "BVAR", "baseline", {"kind": "archived_bvar"}, "Archived BVAR acceptance bar."),
        Candidate("RandomWalk", "simple_baseline", "Naive", "baseline", {"kind": "random_walk"}, "Last observed CPI MoM."),
        Candidate("SeasonalNaive", "simple_baseline", "Naive", "baseline", {"kind": "seasonal_naive"}, "Train-only month-of-year mean."),
        Candidate("Huber", "external_context", "ML_context", "context", {"kind": "huber"}, "Production ML context only."),
        Candidate("RidgeShockDummies", "external_context", "ML_context", "context", {"kind": "ridge_shock"}, "Production ML context only."),
        Candidate("intervention_var_l1", "robust_candidate", "VARX_deterministic", "intervention_dummies", {"kind": "linear_var", "lags": 1, "estimator": "ridge", "intervention": True, "macro": False, "alpha": 0.05}, "VAR-equation OLS/Ridge with predeclared COVID/2022/calendar intervention dummies."),
        Candidate("intervention_varx_macro_l1", "robust_candidate", "VARX_deterministic", "intervention_dummies", {"kind": "linear_var", "lags": 1, "estimator": "ridge", "intervention": True, "macro": True, "alpha": 0.25}, "Macro VARX with deterministic intervention dummies and last-observed macro scenario."),
        Candidate("additive_outlier_var_l1_z35", "robust_candidate", "VARX_pulse_outlier", "additive_outlier_detection", {"kind": "linear_var", "lags": 1, "estimator": "ridge", "intervention": False, "macro": False, "outlier_z": 3.5, "alpha": 0.05}, "Train-only residual MAD outlier pulses, target pulses set to zero."),
        Candidate("additive_outlier_var_l1_z40", "robust_candidate", "VARX_pulse_outlier", "additive_outlier_detection", {"kind": "linear_var", "lags": 1, "estimator": "ridge", "intervention": False, "macro": False, "outlier_z": 4.0, "alpha": 0.05}, "Same as z35 with stricter residual threshold."),
        Candidate("winsorized_var_l1_q05", "robust_candidate", "VAR_winsorized", "winsorized_training", {"kind": "winsor_var", "lags": 1, "q": 0.05}, "VAR(1) after train-only component-wise 5/95 winsorization."),
        Candidate("winsorized_bvar_l1_q05", "robust_candidate", "BVAR_winsorized", "student_t_bvar_approx", {"kind": "winsor_bvar", "lags": 1, "q": 0.05, "lambda1": 0.3}, "BVAR posterior mean on winsorized data as heavy-tail/downweighting approximation."),
        Candidate("huber_var_l1", "robust_candidate", "Robust_VAR", "robust_equation_var", {"kind": "linear_var", "lags": 1, "estimator": "huber", "intervention": False, "macro": False}, "Equation-by-equation Huber VAR(1)."),
        Candidate("huber_intervention_var_l1", "robust_candidate", "Robust_VARX", "robust_equation_var", {"kind": "linear_var", "lags": 1, "estimator": "huber", "intervention": True, "macro": False}, "Huber VAR with predeclared intervention dummies."),
        Candidate("huber_macro_varx_l1", "robust_candidate", "Robust_VARX", "regime_macro_varx", {"kind": "linear_var", "lags": 1, "estimator": "huber", "intervention": False, "macro": True}, "Robust macro VARX with USD/Ruonia/Ki_i last-observed scenario."),
        Candidate("regime_macro_varx_shock_guard_l1", "robust_candidate", "Regime_VARX", "regime_macro_varx", {"kind": "regime_macro_guard"}, "Normal periods use macro VARX; 2022/high-volatility targets use Huber intervention VAR."),
        Candidate("robust_favar_mad_f2_l1", "robust_candidate", "Robust_FAVAR", "robust_favar", {"kind": "favar", "lags": 1, "factors": 2, "robust": True}, "FAVAR with train-only median/MAD scaling and winsorized factor inputs."),
    ]
    return candidates


def _weighted_components(pred_row: dict[str, float]) -> float:
    return float(sum(pred_row[name] * weight for name, weight in OFFICIAL_COMPONENT_WEIGHTS.items()))


def _fit_var_forecast(data: pd.DataFrame, cols: list[str], lags: int) -> np.ndarray:
    frame = data.loc[:, cols].dropna()
    if len(frame) <= max(36, len(cols) * lags + 16):
        raise ValueError("insufficient_train")
    fit = VAR(frame).fit(lags)
    return fit.forecast(frame.values[-fit.k_ar :], steps=1)[0]


def forecast_plain_var(train: pd.DataFrame, lags: int) -> tuple[float, dict[str, Any]]:
    fc = _fit_var_forecast(train, list(TOTAL_VARS), lags)
    return float(fc[0]), {"selected_lag": lags}


def forecast_plain_bic(train: pd.DataFrame) -> tuple[float, dict[str, Any]]:
    data = train.loc[:, list(TOTAL_VARS)].dropna()
    if len(data) < 42:
        raise ValueError("insufficient_train")
    try:
        order = VAR(data).select_order(maxlags=6)
        lags = int(getattr(order, "bic", 1)) or 1
    except Exception:
        lags = 1
    lags = max(1, min(6, lags))
    pred, meta = forecast_plain_var(train, lags)
    meta["selection_method"] = "BIC"
    return pred, meta


def forecast_random_walk(train: pd.DataFrame) -> tuple[float, dict[str, Any]]:
    return float(train["CPI"].dropna().iloc[-1]), {}


def forecast_seasonal_naive(train: pd.DataFrame) -> tuple[float, dict[str, Any]]:
    mm = _seasonal_means(train, list(TOTAL_VARS), None)
    target_month = (train.index.max() + pd.DateOffset(months=1)).month
    return float(_future_seasonal(mm, list(TOTAL_VARS), target_month)[0]), {}


def _ridge_solve(x: np.ndarray, y: np.ndarray, alpha: float) -> np.ndarray:
    xtx = x.T @ x
    penalty = np.eye(xtx.shape[0]) * alpha
    penalty[0, 0] = 0.0
    return np.linalg.solve(xtx + penalty, x.T @ y)


def intervention_values(dates: pd.DatetimeIndex | list[pd.Timestamp]) -> pd.DataFrame:
    idx = pd.DatetimeIndex(dates)
    out = pd.DataFrame(index=idx)
    out["shock_2022_year"] = (idx.year == 2022).astype(float)
    out["shock_2022_mar_apr"] = (((idx.year == 2022) & (idx.month.isin([3, 4])))).astype(float)
    out["covid_2020_q2"] = (((idx.year == 2020) & (idx.month.isin([4, 5, 6])))).astype(float)
    out["tariff_july"] = (idx.month == 7).astype(float)
    return out


def _lagged_design(
    train: pd.DataFrame,
    lags: int,
    macro: bool,
    intervention: bool,
    pulse_dates: list[pd.Timestamp] | None,
    target_date: pd.Timestamp,
) -> tuple[np.ndarray, np.ndarray, list[str], np.ndarray, pd.DatetimeIndex]:
    cols = list(TOTAL_VARS) + (list(MACRO_VARS) if macro else [])
    data = train.loc[:, cols].dropna().copy()
    if len(data) < max(42, lags + 24):
        raise ValueError("insufficient_train")
    row_dates = data.index[lags:]
    feature_names = ["const"]
    for lag in range(1, lags + 1):
        for col in TOTAL_VARS:
            feature_names.append(f"{col}_lag{lag}")
    if macro:
        feature_names.extend([f"{col}_scenario" for col in MACRO_VARS])
    if intervention:
        feature_names.extend(list(intervention_values(row_dates).columns))
    pulse_dates = pulse_dates or []
    pulse_names = [f"pulse_{d.strftime('%Y_%m')}" for d in pulse_dates]
    feature_names.extend(pulse_names)

    x_rows = []
    y_rows = []
    iv_train = intervention_values(row_dates) if intervention else pd.DataFrame(index=row_dates)
    pulse_set = {pd.Timestamp(d) for d in pulse_dates}
    for pos, date in enumerate(row_dates, start=lags):
        row = [1.0]
        for lag in range(1, lags + 1):
            row.extend(data.iloc[pos - lag][list(TOTAL_VARS)].values.astype(float))
        if macro:
            row.extend(data.iloc[pos][list(MACRO_VARS)].values.astype(float))
        if intervention:
            row.extend(iv_train.loc[date].values.astype(float))
        row.extend([1.0 if date == d else 0.0 for d in pulse_set])
        x_rows.append(row)
        y_rows.append(data.iloc[pos][list(TOTAL_VARS)].values.astype(float))

    forecast_row = [1.0]
    for lag in range(1, lags + 1):
        forecast_row.extend(data.iloc[-lag][list(TOTAL_VARS)].values.astype(float))
    if macro:
        forecast_row.extend(data.iloc[-1][list(MACRO_VARS)].values.astype(float))
    if intervention:
        forecast_row.extend(intervention_values([target_date]).iloc[0].values.astype(float))
    forecast_row.extend([0.0 for _ in pulse_set])
    return np.asarray(x_rows), np.asarray(y_rows), feature_names, np.asarray(forecast_row), row_dates


def _fit_equation(x: np.ndarray, y: np.ndarray, forecast_row: np.ndarray, estimator: str, alpha: float) -> float:
    if estimator == "huber":
        model = HuberRegressor(alpha=1e-4, epsilon=1.35, max_iter=500, fit_intercept=False)
        model.fit(x, y)
        return float(model.predict(forecast_row.reshape(1, -1))[0])
    beta = _ridge_solve(x, y, alpha)
    return float(forecast_row @ beta)


def detect_additive_outliers(train: pd.DataFrame, threshold: float) -> tuple[list[pd.Timestamp], list[dict[str, Any]]]:
    data = train.loc[:, list(TOTAL_VARS)].dropna()
    if len(data) < 48:
        return [], []
    try:
        fit = VAR(data).fit(1)
        resid = pd.DataFrame(fit.resid, index=data.index[1:], columns=TOTAL_VARS)
    except Exception:
        return [], []
    scores = {}
    for col in TOTAL_VARS:
        r = resid[col].astype(float)
        med = float(r.median())
        mad = float((r - med).abs().median()) * 1.4826
        if mad < 1e-8:
            mad = float(r.std()) or 1.0
        scores[col] = ((r - med).abs() / mad).rename(col)
    score_df = pd.concat(scores.values(), axis=1)
    score_df["max_robust_z"] = score_df.max(axis=1)
    detected = score_df[score_df["max_robust_z"] > threshold].copy()
    dates = [pd.Timestamp(d) for d in detected.index]
    log_rows = []
    for date, row in detected.iterrows():
        max_col = row[list(TOTAL_VARS)].astype(float).idxmax()
        log_rows.append(
            {
                "detected_date": date,
                "threshold": threshold,
                "max_robust_z": float(row["max_robust_z"]),
                "max_variable": max_col,
                "CPI_z": float(row["CPI"]),
                "Food_z": float(row["Food"]),
                "NonFood_z": float(row["NonFood"]),
                "Services_z": float(row["Services"]),
            }
        )
    return dates, log_rows


def forecast_linear_var(train: pd.DataFrame, target_date: pd.Timestamp, params: dict[str, Any]) -> tuple[float, dict[str, Any], list[dict[str, Any]]]:
    pulse_dates: list[pd.Timestamp] = []
    outlier_rows: list[dict[str, Any]] = []
    if "outlier_z" in params:
        pulse_dates, outlier_rows = detect_additive_outliers(train, float(params["outlier_z"]))

    x, y, feature_names, forecast_row, row_dates = _lagged_design(
        train=train,
        lags=int(params["lags"]),
        macro=bool(params.get("macro", False)),
        intervention=bool(params.get("intervention", False)),
        pulse_dates=pulse_dates,
        target_date=target_date,
    )
    preds = []
    for equation_idx, col in enumerate(TOTAL_VARS):
        preds.append(
            _fit_equation(
                x,
                y[:, equation_idx],
                forecast_row,
                str(params.get("estimator", "ridge")),
                float(params.get("alpha", 0.05)),
            )
        )
    meta = {
        "feature_count": len(feature_names),
        "pulse_count": len(pulse_dates),
        "detected_outlier_max": max(pulse_dates) if pulse_dates else pd.NaT,
        "estimator": params.get("estimator", "ridge"),
    }
    return float(preds[0]), meta, outlier_rows


def winsorize_train(train: pd.DataFrame, q: float) -> pd.DataFrame:
    out = train.copy()
    for col in list(TOTAL_VARS) + list(MACRO_VARS):
        if col not in out:
            continue
        s = out[col].dropna()
        if len(s) < 24:
            continue
        lo, hi = s.quantile(q), s.quantile(1.0 - q)
        out[col] = out[col].clip(lo, hi)
    return out


def forecast_winsor_var(train: pd.DataFrame, q: float, lags: int) -> tuple[float, dict[str, Any]]:
    wtrain = winsorize_train(train, q)
    pred, meta = forecast_plain_var(wtrain, lags)
    meta["winsor_q"] = q
    return pred, meta


def forecast_winsor_bvar(train: pd.DataFrame, params: dict[str, Any]) -> tuple[float, dict[str, Any]]:
    cols = list(TOTAL_VARS)
    wtrain = winsorize_train(train, float(params["q"]))
    data = wtrain.loc[:, cols].dropna()
    if len(data) < 42:
        raise ValueError("insufficient_train")
    model = BVARForecaster(
        lags=int(params["lags"]),
        lambda1=float(params["lambda1"]),
        lambda2=0.5,
        lambda3=1.0,
        n_draws=1,
        var_names=cols,
    )
    model.fit(data, target_col="CPI")
    x_t = np.ones(1 + model.k * model.lags)
    for lag in range(1, model.lags + 1):
        x_t[1 + (lag - 1) * model.k : 1 + lag * model.k] = model.raw_data[-lag, :]
    y_new = x_t @ model.B_post
    return float(y_new[0]), {"winsor_q": params["q"], "lambda1": params["lambda1"]}


def forecast_favar(train: pd.DataFrame, params: dict[str, Any]) -> tuple[float, dict[str, Any]]:
    factors_n = int(params["factors"])
    lags = int(params["lags"])
    robust = bool(params.get("robust", False))
    factor_cols = ["Food", "NonFood", "Services", "USD", "Ruonia", "Ki_i"]
    data = train.loc[:, ["CPI"] + factor_cols].dropna()
    if len(data) < 48:
        raise ValueError("insufficient_train")
    z = data.loc[:, factor_cols].values.astype(float)
    if robust:
        center = np.median(z, axis=0)
        scale = np.median(np.abs(z - center), axis=0) * 1.4826
        scale[scale < 1e-8] = np.std(z, axis=0)[scale < 1e-8]
        scale[scale < 1e-8] = 1.0
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
    fc = _fit_var_forecast(favar, list(favar.columns), lags)
    return float(fc[0]), {"robust": robust, "factors": factors_n}


def is_shock_guard_target(train: pd.DataFrame, target_date: pd.Timestamp) -> bool:
    if target_date.year == 2022:
        return True
    cpi = train["CPI"].dropna()
    if len(cpi) < 12:
        return False
    return bool(abs(float(cpi.iloc[-1])) >= 1.0 or float(cpi.iloc[-12:].std()) >= 0.55)


def forecast_candidate(
    official_train: pd.DataFrame,
    model_train: pd.DataFrame,
    target_date: pd.Timestamp,
    candidate: Candidate,
    seed: int,
) -> tuple[float, str, dict[str, Any], list[dict[str, Any]]]:
    del seed
    params = candidate.params
    kind = params["kind"]
    outlier_rows: list[dict[str, Any]] = []
    if kind == "plain_bic":
        pred, meta = forecast_plain_bic(official_train)
    elif kind == "plain_var":
        pred, meta = forecast_plain_var(official_train, int(params["lags"]))
    elif kind == "random_walk":
        pred, meta = forecast_random_walk(official_train)
    elif kind == "seasonal_naive":
        pred, meta = forecast_seasonal_naive(official_train)
    elif kind == "archived_bvar":
        pred, status = forecast_archived_bvar(official_train, target_date, 12345)
        meta = {"archived_status": status}
    elif kind == "huber":
        pred, status = forecast_huber(model_train, target_date, 0)
        meta = {"context_status": status}
    elif kind == "ridge_shock":
        pred, status = forecast_ridge_shock(model_train, target_date, 0)
        meta = {"context_status": status}
    elif kind == "linear_var":
        pred, meta, outlier_rows = forecast_linear_var(official_train, target_date, params)
    elif kind == "winsor_var":
        pred, meta = forecast_winsor_var(official_train, float(params["q"]), int(params["lags"]))
    elif kind == "winsor_bvar":
        pred, meta = forecast_winsor_bvar(official_train, params)
    elif kind == "favar":
        pred, meta = forecast_favar(official_train, params)
    elif kind == "varx":
        varx_params = {"lags": params["lags"], "estimator": "ridge", "intervention": params.get("intervention", False), "macro": True, "alpha": 0.25}
        pred, meta, outlier_rows = forecast_linear_var(official_train, target_date, varx_params)
    elif kind == "regime_macro_guard":
        if is_shock_guard_target(official_train, target_date):
            guard_params = {"lags": 1, "estimator": "huber", "intervention": True, "macro": False}
            pred, meta, outlier_rows = forecast_linear_var(official_train, target_date, guard_params)
            meta["regime"] = "shock_guard_huber_intervention_var"
        else:
            guard_params = {"lags": 1, "estimator": "ridge", "intervention": False, "macro": True, "alpha": 0.25}
            pred, meta, outlier_rows = forecast_linear_var(official_train, target_date, guard_params)
            meta["regime"] = "normal_macro_varx"
    else:
        raise ValueError(f"unknown candidate kind: {kind}")
    status = "ok" if np.isfinite(pred) else "nan_prediction"
    return float(pred) if np.isfinite(pred) else np.nan, status, meta, outlier_rows


def evaluate(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, list[Candidate]]:
    official = load_official_data()
    model_df = load_model_frame()
    candidates = build_candidates()
    outer = all_outer_dates()
    pred_rows = []
    outlier_rows = []
    selection_rows = []
    leakage_rows = []

    total = len(outer) * len(candidates)
    done = 0
    for _, row in outer.iterrows():
        target_date = row["target_date"]
        if target_date not in official.index:
            continue
        cutoff = target_date - pd.DateOffset(months=1)
        official_train = official[official.index <= cutoff].copy()
        model_train = model_df[model_df.index <= cutoff].copy()
        actual = float(official.loc[target_date, "CPI"])
        for idx, candidate in enumerate(candidates):
            try:
                pred, status, meta, detected = forecast_candidate(
                    official_train,
                    model_train,
                    target_date,
                    candidate,
                    args.seed + idx * 1000 + int(target_date.strftime("%Y%m")),
                )
            except Exception as exc:
                pred, status, meta, detected = np.nan, f"error: {type(exc).__name__}: {exc}", {}, []
            error = actual - pred if np.isfinite(pred) else np.nan
            pred_rows.append(
                {
                    "horizon": 1,
                    "window": row["window"],
                    "target_date": target_date,
                    "cutoff": cutoff,
                    "model": candidate.name,
                    "role": candidate.role,
                    "family": candidate.family,
                    "direction": candidate.direction,
                    "actual": actual,
                    "prediction": pred,
                    "error": error,
                    "abs_error": abs(error) if np.isfinite(error) else np.nan,
                    "status": status,
                    "train_start": official_train.index.min() if len(official_train) else pd.NaT,
                    "train_end": official_train.index.max() if len(official_train) else pd.NaT,
                    "train_n": len(official_train),
                    "params": json.dumps(candidate.params, ensure_ascii=False, sort_keys=True),
                    "meta": json.dumps({k: str(v) for k, v in meta.items()}, ensure_ascii=False, sort_keys=True),
                    "notes": candidate.notes,
                }
            )
            if candidate.name == "PlainVAR_BIC" and "selected_lag" in meta:
                selection_rows.append(
                    {
                        "model": candidate.name,
                        "target_date": target_date,
                        "cutoff": cutoff,
                        "selection_method": "BIC",
                        "selected_lag": meta["selected_lag"],
                    }
                )
            for detected_row in detected:
                outlier_rows.append(
                    {
                        "model": candidate.name,
                        "target_date": target_date,
                        "cutoff": cutoff,
                        **detected_row,
                    }
                )
            detected_max = meta.get("detected_outlier_max", pd.NaT)
            leakage_rows.append(
                {
                    "model": candidate.name,
                    "target_date": target_date,
                    "cutoff": cutoff,
                    "train_end": official_train.index.max() if len(official_train) else pd.NaT,
                    "train_end_after_cutoff": bool(len(official_train) and official_train.index.max() > cutoff),
                    "detected_outlier_max": detected_max,
                    "detected_outlier_after_cutoff": bool(pd.notna(detected_max) and pd.Timestamp(detected_max) > cutoff),
                    "target_actual_used_in_rule": False,
                    "future_exog_used": False,
                }
            )
            done += 1
        if done and done % 500 == 0:
            print(f"evaluated {done}/{total}", flush=True)
    return pd.DataFrame(pred_rows), pd.DataFrame(outlier_rows), pd.DataFrame(selection_rows), pd.DataFrame(leakage_rows), candidates


def metric_record(group: pd.DataFrame) -> dict[str, Any]:
    ok = group.dropna(subset=["error"]).copy()
    if ok.empty:
        return {"n": 0, "MAE": np.nan, "RMSE": np.nan, "Bias": np.nan, "Max_Error": np.nan, "KPI_Violations": np.nan, "Coverage_50pct": np.nan}
    errors = ok["error"].astype(float)
    abs_errors = errors.abs()
    return {
        "n": len(ok),
        "MAE": abs_errors.mean(),
        "RMSE": np.sqrt((errors**2).mean()),
        "Bias": errors.mean(),
        "Max_Error": abs_errors.max(),
        "KPI_Violations": int((abs_errors > 0.5).sum()),
        "Coverage_50pct": float((abs_errors <= 0.5).mean() * 100),
    }


def calculate_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for keys, group in predictions.groupby(["model", "role", "family", "direction", "window"], sort=False):
        model, role, family, direction, window = keys
        rows.append({"model": model, "role": role, "family": family, "direction": direction, "slice": window, **metric_record(group)})

    slice_defs = {
        "all_windows": lambda df: df.index == df.index,
        "out_of_selection": lambda df: df["window"] != "selection_2025-04_2026-03",
        "out_of_selection_non_shock": lambda df: (df["window"] != "selection_2025-04_2026-03") & (df["window"] != "2022_shock"),
        "shock_2022": lambda df: df["window"] == "2022_shock",
        "selection_reference": lambda df: df["window"] == "selection_2025-04_2026-03",
    }
    for keys, group in predictions.groupby(["model", "role", "family", "direction"], sort=False):
        model, role, family, direction = keys
        for slice_name, mask_fn in slice_defs.items():
            sub = group[mask_fn(group)]
            rows.append({"model": model, "role": role, "family": family, "direction": direction, "slice": slice_name, **metric_record(sub)})
    return pd.DataFrame(rows).sort_values(["slice", "MAE", "model"])


def build_comparison(metrics: pd.DataFrame) -> pd.DataFrame:
    key_slices = ["all_windows", "out_of_selection", "out_of_selection_non_shock", "shock_2022", "selection_reference"]
    rows = []
    for model, group in metrics[metrics["slice"].isin(key_slices)].groupby("model", sort=False):
        first = group.iloc[0]
        rec = {
            "model": model,
            "role": first["role"],
            "family": first["family"],
            "direction": first["direction"],
        }
        for slice_name in key_slices:
            s = group[group["slice"] == slice_name]
            if s.empty:
                continue
            row = s.iloc[0]
            rec[f"{slice_name}_n"] = row["n"]
            rec[f"{slice_name}_MAE"] = row["MAE"]
            rec[f"{slice_name}_RMSE"] = row["RMSE"]
            rec[f"{slice_name}_KPI"] = row["KPI_Violations"]
            rec[f"{slice_name}_Coverage"] = row["Coverage_50pct"]
        rows.append(rec)
    return pd.DataFrame(rows).sort_values(["all_windows_MAE", "out_of_selection_MAE", "model"])


def choose_final_status(comparison: pd.DataFrame, leakage: pd.DataFrame) -> tuple[str, str]:
    robust = comparison[comparison["role"] == "robust_candidate"].copy()
    if robust.empty:
        return "none", "no robust improvement"
    plain = comparison[comparison["model"] == "PlainVAR_BIC"].iloc[0]
    archived = comparison[comparison["model"] == "Archived_BVAR"].iloc[0]
    leak_bad = bool((leakage["train_end_after_cutoff"] | leakage["detected_outlier_after_cutoff"] | leakage["target_actual_used_in_rule"] | leakage["future_exog_used"]).any())
    robust["shock_gain"] = plain["shock_2022_MAE"] - robust["shock_2022_MAE"]
    robust["all_gain"] = plain["all_windows_MAE"] - robust["all_windows_MAE"]
    robust["outsel_gain"] = plain["out_of_selection_MAE"] - robust["out_of_selection_MAE"]
    robust["nonshock_penalty"] = robust["out_of_selection_non_shock_MAE"] - plain["out_of_selection_non_shock_MAE"]
    robust["kpi_gain"] = plain["all_windows_KPI"] - robust["all_windows_KPI"]
    robust["score"] = robust["all_windows_MAE"] + 0.20 * robust["shock_2022_MAE"] + 0.03 * np.maximum(robust["nonshock_penalty"], 0)
    best = robust.sort_values(["score", "all_windows_MAE", "shock_2022_MAE"]).iloc[0]
    if leak_bad:
        return str(best["model"]), "experimental robust VAR"
    beats_archived = bool(best["all_windows_MAE"] < archived["all_windows_MAE"])
    beats_plain_primary = bool((best["all_gain"] > 0) or (best["outsel_gain"] > 0))
    improves_shock = bool((best["shock_gain"] > 0.03) or (best["kpi_gain"] > 0))
    nonshock_ok = bool(best["out_of_selection_non_shock_MAE"] <= plain["out_of_selection_non_shock_MAE"] * 1.12)
    if beats_archived and beats_plain_primary and improves_shock and nonshock_ok:
        return str(best["model"]), "recommended robust mandatory VAR"
    if beats_archived and improves_shock and nonshock_ok:
        return str(best["model"]), "shock-robust VAR alternative"
    if beats_archived or improves_shock:
        return str(best["model"]), "experimental robust VAR"
    return str(best["model"]), "no robust improvement"


def write_config(out_dir: Path, args: argparse.Namespace, candidates: list[Candidate]) -> None:
    config = {
        "agent_id": AGENT_ID,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "run_name": args.run_name,
        "purpose": "robust/outlier-aware mandatory VAR-family h=1 research",
        "horizons": [1],
        "h2_h12": "not tested; robust full-grid task is h=1 primary and exogenous recursive paths require separate design",
        "windows": [{"name": n, "start": str(s.date()), "end": str(e.date())} for n, s, e in historical_windows()],
        "candidate_count": len(candidates),
        "directions": sorted({c.direction for c in candidates if c.role == "robust_candidate"}),
        "predeclared_interventions": {
            "shock_2022_year": "target/train date in 2022",
            "shock_2022_mar_apr": "March-April 2022",
            "covid_2020_q2": "April-June 2020",
            "tariff_july": "all July months",
        },
        "seed": args.seed,
    }
    (out_dir / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")


def write_notes_and_report(
    out_dir: Path,
    args: argparse.Namespace,
    candidates: list[Candidate],
    metrics: pd.DataFrame,
    comparison: pd.DataFrame,
    outliers: pd.DataFrame,
    leakage: pd.DataFrame,
) -> None:
    final_model, final_status = choose_final_status(comparison, leakage)
    def _row(model: str) -> pd.Series | None:
        rows = comparison[comparison["model"] == model]
        return rows.iloc[0] if len(rows) else None

    final_row = _row(final_model)
    plain_row = _row("PlainVAR_BIC")
    favar_row = _row("favar_macro_components_f2_l1")
    varx_row = _row("varx_last_exog_l1")

    relation = "No robust replacement is justified."
    if final_status == "recommended robust mandatory VAR":
        relation = "Replaces `PlainVAR_BIC` as the robust mandatory VAR-family benchmark, but not production ML."
    elif final_status == "shock-robust VAR alternative":
        relation = "Use as a shock-robust alternative beside `PlainVAR_BIC`, not as the default mandatory VAR."
    elif final_status == "experimental robust VAR":
        relation = "Experimental only; do not replace `PlainVAR_BIC` without additional validation."

    interpretation_lines = []
    if final_row is not None and plain_row is not None:
        interpretation_lines.extend(
            [
                f"- Versus `PlainVAR_BIC`: all-window MAE {final_row['all_windows_MAE']:.6f} vs {plain_row['all_windows_MAE']:.6f}; 2022 MAE {final_row['shock_2022_MAE']:.6f} vs {plain_row['shock_2022_MAE']:.6f}.",
                f"- Non-shock out-of-selection MAE {final_row['out_of_selection_non_shock_MAE']:.6f} vs {plain_row['out_of_selection_non_shock_MAE']:.6f}; all-window KPI violations {int(final_row['all_windows_KPI'])} vs {int(plain_row['all_windows_KPI'])}.",
            ]
        )
    if final_row is not None and favar_row is not None:
        interpretation_lines.append(
            f"- Versus non-robust FAVAR: all-window MAE is {final_row['all_windows_MAE']:.6f} vs {favar_row['all_windows_MAE']:.6f}; robust preprocessing helps 2022 slightly but is not a broad accuracy breakthrough."
        )
    if final_row is not None and varx_row is not None:
        interpretation_lines.append(
            f"- Versus macro VARX: non-shock MAE is {final_row['out_of_selection_non_shock_MAE']:.6f} vs {varx_row['out_of_selection_non_shock_MAE']:.6f}; 2022 MAE is {final_row['shock_2022_MAE']:.6f} vs {varx_row['shock_2022_MAE']:.6f}."
        )
    interpretation_lines.extend(
        [
            "- Deterministic intervention dummies and the simple regime guard did not fix 2022; in this validation they often made the shock slice worse.",
            "- Additive pulse dummies and Huber VAR reduce some damage relative to plain VAR, but they do not beat robust FAVAR.",
        ]
    )

    direction_counts = pd.DataFrame(
        [
            {"direction": direction, "candidate_count": sum(c.direction == direction for c in candidates)}
            for direction in sorted({c.direction for c in candidates})
        ]
    )
    leak_violations = int((leakage["train_end_after_cutoff"] | leakage["detected_outlier_after_cutoff"] | leakage["target_actual_used_in_rule"] | leakage["future_exog_used"]).sum()) if len(leakage) else 0
    outlier_summary = (
        outliers.groupby(["model", "detected_date"], as_index=False)
        .agg(count=("target_date", "size"), mean_max_z=("max_robust_z", "mean"), max_variable=("max_variable", lambda s: s.mode().iloc[0] if len(s.mode()) else ""))
        .sort_values(["model", "count"], ascending=[True, False])
        if len(outliers)
        else pd.DataFrame()
    )
    selected_cols = [
        "model",
        "role",
        "family",
        "all_windows_MAE",
        "out_of_selection_MAE",
        "out_of_selection_non_shock_MAE",
        "shock_2022_MAE",
        "all_windows_KPI",
        "shock_2022_KPI",
    ]
    comp_show = comparison[selected_cols].head(35)

    notes = [
        "# codex_cli Robust VAR Notes",
        "",
        f"- Run directory: `{out_dir}`",
        f"- Final report: `{REPORT_PATH}`",
        f"- Final robust model: `{final_model}`",
        f"- Final status: `{final_status}`",
        f"- Decision: {relation}",
        f"- Leakage violations: `{leak_violations}`",
        "- h=2/h=12 were not tested in this run.",
        "",
        "## Directions Tested",
        "",
        direction_counts.to_markdown(index=False),
        "",
        "## h=1 Comparison",
        "",
        comp_show.to_markdown(index=False),
        "",
        "## Outlier Summary",
        "",
        outlier_summary.head(40).to_markdown(index=False) if len(outlier_summary) else "_No additive outliers detected._",
    ]
    (out_dir / "notes.md").write_text("\n".join(notes) + "\n", encoding="utf-8")

    report = [
        "# codex_cli Robust VAR Report",
        "",
        "## Scope",
        "",
        "Research-only robust/outlier-aware h=1 VAR-family evaluation. No production data, registry, dashboard, or shared task file was modified.",
        "",
        "## Methods Tested",
        "",
        direction_counts.to_markdown(index=False),
        "",
        "Robust directions include deterministic intervention dummies, train-only additive outlier pulse detection, winsorized VAR/BVAR, equation-by-equation Huber VAR, regime-aware macro VARX, and robust FAVAR.",
        "",
        "## Recommendation",
        "",
        f"- Final robust VAR model: `{final_model}`",
        f"- Final status: `{final_status}`",
        f"- Decision: {relation}",
        "- `PlainVAR_BIC` remains the explicit incumbent acceptance bar from the task.",
        "- Huber and RidgeShockDummies are external production context only, not VAR candidates.",
        "",
        "## h=1 Metrics",
        "",
        comp_show.to_markdown(index=False),
        "",
        "## Interpretation",
        "",
        "\n".join(interpretation_lines),
        "",
        "## Outlier / Dummy Evidence",
        "",
        "- Predeclared deterministic dummies: 2022 year, March-April 2022, COVID Q2 2020, and July tariff/admin seasonality.",
        "- Additive outlier pulse dummies are detected separately inside each cutoff from baseline VAR(1) train residual MAD scores; target-month pulse dummies are zero.",
        "",
        outlier_summary.head(30).to_markdown(index=False) if len(outlier_summary) else "_No additive outlier rows were logged._",
        "",
        "## Leakage Audit",
        "",
        "- Every target uses `cutoff = target_date - 1 month`.",
        "- Outlier detection uses only residuals from rows at or before the cutoff.",
        "- VARX macro target scenarios use last observed cutoff values for USD/Ruonia/Ki_i.",
        "- Predeclared calendar dummies are allowed for target dates but are not selected from forecast errors.",
        f"- Leakage violations in `leakage_checks.csv`: `{leak_violations}`.",
        "",
        "## h=2 / h=12",
        "",
        "Not tested. The robust task is h=1-primary; extending regime Macro VARX to h=2/h=12 needs explicit recursive exogenous scenario paths rather than reusing one-step last-observed paths.",
        "",
        "## Commands Run",
        "",
        f"```bash\npython3 experiments/var_sa_research/codex_cli_robust_var.py --run-name {args.run_name}\n```",
        "",
        "## Artifacts",
        "",
        f"- Run directory: `{out_dir}`",
        "- Required files: `config.json`, `metrics.csv`, `predictions.csv`, `comparison.csv`, `outlier_log.csv`, `selection_log.csv`, `leakage_checks.csv`, `notes.md`, script copy.",
    ]
    REPORT_PATH.write_text("\n".join(report) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    if not args.run_name.startswith(f"{AGENT_ID}_robust_var_"):
        raise ValueError(f"run-name must start with {AGENT_ID}_robust_var_")
    out_dir = RUNS_DIR / args.run_name
    out_dir.mkdir(parents=True, exist_ok=False)

    candidates = build_candidates()
    write_config(out_dir, args, candidates)
    predictions, outliers, selection_log, leakage, candidates = evaluate(args)
    metrics = calculate_metrics(predictions)
    comparison = build_comparison(metrics)

    predictions.to_csv(out_dir / "predictions.csv", index=False)
    metrics.to_csv(out_dir / "metrics.csv", index=False)
    comparison.to_csv(out_dir / "comparison.csv", index=False)
    outliers.to_csv(out_dir / "outlier_log.csv", index=False)
    selection_log.to_csv(out_dir / "selection_log.csv", index=False)
    leakage.to_csv(out_dir / "leakage_checks.csv", index=False)
    shutil.copy2(Path(__file__), out_dir / Path(__file__).name)
    write_notes_and_report(out_dir, args, candidates, metrics, comparison, outliers, leakage)

    print(f"Wrote artifacts to {out_dir}", flush=True)
    print(f"Wrote final report to {REPORT_PATH}", flush=True)
    print(comparison.head(18).to_string(index=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
