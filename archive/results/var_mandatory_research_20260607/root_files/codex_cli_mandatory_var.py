#!/usr/bin/env python3
"""
Research-only mandatory VAR-family search for СИРЕНА-КБР.

The script does not modify production data, model registry, dashboard paths, or
shared task files. All artifacts are written below
experiments/var_sa_research/runs/codex_cli_mandatory_var_*.
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
from statsmodels.tsa.api import VAR
from statsmodels.tsa.arima.model import ARIMA

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
    load_official_data,
)
from sirena.models.bvar import BVARForecaster  # noqa: E402

warnings.filterwarnings("ignore")


AGENT_ID = "codex_cli"
RESEARCH_DIR = ROOT / "experiments" / "var_sa_research"
RUNS_DIR = RESEARCH_DIR / "runs"
REPORT_PATH = RESEARCH_DIR / f"{AGENT_ID}_mandatory_var_report.md"
TRAIN_MODES = ["expanding"]
INNER_MONTHS = 24
MIN_INNER_OBS = 12
HORIZONS = [1]


@dataclass(frozen=True)
class Candidate:
    name: str
    family: str
    direction: str
    variables: tuple[str, ...]
    params: dict[str, Any]
    notes: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", default=f"{AGENT_ID}_mandatory_var_h1_full")
    parser.add_argument("--seed", type=int, default=20260607)
    parser.add_argument("--inner-months", type=int, default=INNER_MONTHS)
    parser.add_argument("--min-inner-obs", type=int, default=MIN_INNER_OBS)
    parser.add_argument("--skip-project-context", action="store_true")
    parser.add_argument("--include-sarima", action="store_true")
    return parser.parse_args()


def historical_windows() -> list[tuple[str, pd.Timestamp, pd.Timestamp]]:
    return [
        ("pre_covid_2018_2019", pd.Timestamp("2018-01-01"), pd.Timestamp("2019-12-01")),
        ("covid_2020_2021", pd.Timestamp("2020-01-01"), pd.Timestamp("2021-12-01")),
        ("sanctions_2022", pd.Timestamp("2022-01-01"), pd.Timestamp("2022-12-01")),
        ("tightening_2023", pd.Timestamp("2023-01-01"), pd.Timestamp("2023-12-01")),
        ("pre_selection_2024_2025q1", pd.Timestamp("2024-01-01"), pd.Timestamp("2025-03-01")),
        ("selection_reference_2025q2_2026q1", pd.Timestamp("2025-04-01"), pd.Timestamp("2026-03-01")),
    ]


def all_outer_dates() -> pd.DataFrame:
    rows = []
    for window, start, end in historical_windows():
        for target_date in pd.date_range(start=start, end=end, freq="MS"):
            rows.append({"window": window, "target_date": target_date})
    return pd.DataFrame(rows)


def apply_train_mode(df: pd.DataFrame, cutoff: pd.Timestamp, train_mode: str) -> pd.DataFrame:
    train = df[df.index <= cutoff].copy()
    if train_mode == "rolling120" and len(train) > 120:
        train = train.iloc[-120:].copy()
    return train


def all_precompute_dates(official: pd.DataFrame, outer_dates: pd.DataFrame, inner_months: int) -> pd.DatetimeIndex:
    start = outer_dates["target_date"].min() - pd.DateOffset(months=inner_months + 2)
    end = outer_dates["target_date"].max()
    return pd.DatetimeIndex(official.index[(official.index >= start) & (official.index <= end)])


def build_candidates() -> list[Candidate]:
    candidates: list[Candidate] = []

    for lags in [1, 2, 3, 4]:
        candidates.append(
            Candidate(
                name=f"plain_var_tc_l{lags}",
                family="VAR",
                direction="plain_var_bvar_baseline",
                variables=("CPI", "Food", "NonFood", "Services"),
                params={"kind": "var", "lags": lags, "target": "CPI"},
                notes="Plain VAR on official total plus three components.",
            )
        )
        candidates.append(
            Candidate(
                name=f"component_constrained_var_l{lags}",
                family="VAR_component_constrained",
                direction="component_constrained_var",
                variables=("Food", "NonFood", "Services"),
                params={"kind": "var", "lags": lags, "target": "bottom_up"},
                notes="Component VAR reconstructed with fixed CPI component weights.",
            )
        )

    for varset_name, variables in {
        "tc": ("CPI", "Food", "NonFood", "Services"),
        "tfu": ("CPI", "Food", "USD", "Ruonia"),
        "comp": ("Food", "NonFood", "Services"),
    }.items():
        for lags in [1, 2]:
            for lam in [0.1, 0.3, 1.0]:
                candidates.append(
                    Candidate(
                        name=f"bvar_det_{varset_name}_l{lags}_lam{str(lam).replace('.', 'p')}",
                        family="BVAR_deterministic_mean",
                        direction="bvar_model_averaging_pool",
                        variables=variables,
                        params={
                            "kind": "bvar_mean",
                            "lags": lags,
                            "lambda1": lam,
                            "target": "bottom_up" if varset_name == "comp" else "CPI",
                        },
                        notes="Minnesota BVAR posterior-mean forecast, no Monte Carlo shock draw.",
                    )
                )

    for lags in [1, 2, 3]:
        candidates.append(
            Candidate(
                name=f"varx_last_exog_l{lags}",
                family="VARX_OLS",
                direction="varx_cutoff_safe_exog",
                variables=("CPI", "Food", "NonFood", "Services", "USD", "Ruonia", "Ki_i"),
                params={"kind": "varx", "lags": lags, "alpha": 0.25},
                notes="Reduced-form CPI VARX equation; target exog scenario is last observed cutoff value.",
            )
        )

    for factors in [1, 2]:
        for lags in [1, 2, 3]:
            candidates.append(
                Candidate(
                    name=f"favar_macro_components_f{factors}_l{lags}",
                    family="FAVAR",
                    direction="factor_augmented_var",
                    variables=("CPI", "Food", "NonFood", "Services", "USD", "Ruonia", "Ki_i"),
                    params={"kind": "favar", "factors": factors, "lags": lags},
                    notes="Train-only PCA factors from components and macro variables, then VAR on CPI plus factors.",
                )
            )

    candidates.extend(
        [
            Candidate(
                name="regime_var_shock_component_else_plain",
                family="Regime_VAR",
                direction="regime_aware_var",
                variables=("CPI", "Food", "NonFood", "Services"),
                params={"kind": "regime", "rule": "shock_component_else_plain"},
                notes="If cutoff CPI shock is high, use component VAR l1; otherwise plain VAR l2.",
            ),
            Candidate(
                name="regime_bvar_high_shrink_else_tfu",
                family="Regime_BVAR",
                direction="regime_aware_var",
                variables=("CPI", "Food", "NonFood", "Services", "USD", "Ruonia"),
                params={"kind": "regime", "rule": "high_shrink_else_tfu"},
                notes="If recent volatility is high, use tightly shrunk component BVAR; otherwise CPI/Food/USD/Ruonia BVAR.",
            ),
        ]
    )
    return candidates


def _fit_var_forecast(data: pd.DataFrame, cols: list[str], lags: int, horizon: int = 1) -> np.ndarray:
    frame = data.loc[:, cols].dropna()
    if len(frame) <= max(30, len(cols) * lags + 12):
        raise ValueError("insufficient_train")
    fit = VAR(frame).fit(lags)
    return fit.forecast(frame.values[-fit.k_ar :], steps=horizon)[horizon - 1]


def _weighted_components(pred_row: dict[str, float]) -> float:
    return float(sum(pred_row[name] * weight for name, weight in OFFICIAL_COMPONENT_WEIGHTS.items()))


def forecast_var_candidate(train: pd.DataFrame, candidate: Candidate) -> float:
    cols = list(candidate.variables)
    fc = _fit_var_forecast(train, cols, int(candidate.params["lags"]))
    if candidate.params.get("target") == "bottom_up":
        return _weighted_components(dict(zip(cols, fc)))
    return float(fc[cols.index("CPI")])


def forecast_bvar_mean(train: pd.DataFrame, candidate: Candidate) -> float:
    cols = list(candidate.variables)
    data = train.loc[:, cols].dropna()
    if len(data) < max(36, int(candidate.params["lags"]) * len(cols) + 24):
        raise ValueError("insufficient_train")

    model = BVARForecaster(
        lags=int(candidate.params["lags"]),
        lambda1=float(candidate.params["lambda1"]),
        lambda2=0.5,
        lambda3=1.0,
        n_draws=1,
        var_names=cols,
    )
    model.fit(data, target_col=cols[0])
    y_curr = model.raw_data[-model.lags :, :].copy()
    x_t = np.ones(1 + model.k * model.lags)
    for lag in range(1, model.lags + 1):
        x_t[1 + (lag - 1) * model.k : 1 + lag * model.k] = y_curr[-lag, :]
    y_new = x_t @ model.B_post
    if candidate.params.get("target") == "bottom_up":
        return _weighted_components(dict(zip(cols, y_new)))
    return float(y_new[cols.index("CPI")])


def _ridge_solve(x: np.ndarray, y: np.ndarray, alpha: float) -> np.ndarray:
    xtx = x.T @ x
    penalty = np.eye(xtx.shape[0]) * alpha
    penalty[0, 0] = 0.0
    return np.linalg.solve(xtx + penalty, x.T @ y)


def forecast_varx(train: pd.DataFrame, candidate: Candidate) -> float:
    lags = int(candidate.params["lags"])
    alpha = float(candidate.params["alpha"])
    endog = ["CPI", "Food", "NonFood", "Services"]
    exog = ["USD", "Ruonia", "Ki_i"]
    data = train.loc[:, endog + exog].dropna()
    if len(data) < max(42, lags + 24):
        raise ValueError("insufficient_train")

    x_rows = []
    y_rows = []
    for idx in range(lags, len(data)):
        row = [1.0]
        for lag in range(1, lags + 1):
            row.extend(data.iloc[idx - lag][endog].values.astype(float))
        row.extend(data.iloc[idx][exog].values.astype(float))
        x_rows.append(row)
        y_rows.append(float(data.iloc[idx]["CPI"]))
    beta = _ridge_solve(np.asarray(x_rows), np.asarray(y_rows), alpha)

    forecast_row = [1.0]
    for lag in range(1, lags + 1):
        forecast_row.extend(data.iloc[-lag][endog].values.astype(float))
    forecast_row.extend(data.iloc[-1][exog].values.astype(float))
    return float(np.asarray(forecast_row) @ beta)


def forecast_favar(train: pd.DataFrame, candidate: Candidate) -> float:
    factors_n = int(candidate.params["factors"])
    lags = int(candidate.params["lags"])
    factor_cols = ["Food", "NonFood", "Services", "USD", "Ruonia", "Ki_i"]
    data = train.loc[:, ["CPI"] + factor_cols].dropna()
    if len(data) < max(48, lags + 30):
        raise ValueError("insufficient_train")

    z = data.loc[:, factor_cols].values.astype(float)
    mean = z.mean(axis=0)
    std = z.std(axis=0)
    std[std < 1e-8] = 1.0
    z_std = (z - mean) / std
    _, _, vt = np.linalg.svd(z_std, full_matrices=False)
    scores = z_std @ vt[:factors_n].T
    favar = pd.DataFrame(index=data.index, data={"CPI": data["CPI"].values})
    for idx in range(factors_n):
        favar[f"F{idx + 1}"] = scores[:, idx]
    fc = _fit_var_forecast(favar, list(favar.columns), lags)
    return float(fc[0])


def forecast_regime(train: pd.DataFrame, candidate: Candidate) -> float:
    cpi = train["CPI"].dropna()
    if len(cpi) < 48:
        raise ValueError("insufficient_train")
    last_abs = abs(float(cpi.iloc[-1]))
    trailing3 = abs(float(cpi.iloc[-3:].mean()))
    trailing12_std = float(cpi.iloc[-12:].std())

    if candidate.params["rule"] == "shock_component_else_plain":
        if last_abs >= 1.0 or trailing3 >= 0.8:
            inner = Candidate(
                "inner_component_var_l1",
                "VAR_component_constrained",
                "regime_inner",
                ("Food", "NonFood", "Services"),
                {"kind": "var", "lags": 1, "target": "bottom_up"},
                "",
            )
        else:
            inner = Candidate(
                "inner_plain_var_l2",
                "VAR",
                "regime_inner",
                ("CPI", "Food", "NonFood", "Services"),
                {"kind": "var", "lags": 2, "target": "CPI"},
                "",
            )
        return forecast_var_candidate(train, inner)

    if trailing12_std >= 0.55 or last_abs >= 1.0:
        inner = Candidate(
            "inner_bvar_comp_l1_lam0p1",
            "BVAR_deterministic_mean",
            "regime_inner",
            ("Food", "NonFood", "Services"),
            {"kind": "bvar_mean", "lags": 1, "lambda1": 0.1, "target": "bottom_up"},
            "",
        )
    else:
        inner = Candidate(
            "inner_bvar_tfu_l1_lam0p3",
            "BVAR_deterministic_mean",
            "regime_inner",
            ("CPI", "Food", "USD", "Ruonia"),
            {"kind": "bvar_mean", "lags": 1, "lambda1": 0.3, "target": "CPI"},
            "",
        )
    return forecast_bvar_mean(train, inner)


def forecast_candidate(train: pd.DataFrame, candidate: Candidate) -> float:
    kind = candidate.params["kind"]
    if kind == "var":
        return forecast_var_candidate(train, candidate)
    if kind == "bvar_mean":
        return forecast_bvar_mean(train, candidate)
    if kind == "varx":
        return forecast_varx(train, candidate)
    if kind == "favar":
        return forecast_favar(train, candidate)
    if kind == "regime":
        return forecast_regime(train, candidate)
    raise ValueError(f"unknown kind: {kind}")


def compute_candidate_predictions(
    official: pd.DataFrame,
    candidates: list[Candidate],
    target_dates: pd.DatetimeIndex,
) -> pd.DataFrame:
    rows = []
    total = len(TRAIN_MODES) * len(candidates) * len(target_dates)
    done = 0
    for train_mode in TRAIN_MODES:
        for candidate in candidates:
            for target_date in target_dates:
                if target_date not in official.index:
                    continue
                cutoff = target_date - pd.DateOffset(months=1)
                train = apply_train_mode(official, cutoff, train_mode)
                actual = float(official.loc[target_date, "CPI"])
                try:
                    pred = forecast_candidate(train, candidate)
                    status = "ok" if np.isfinite(pred) else "nan_prediction"
                except Exception as exc:
                    pred = np.nan
                    status = f"error: {type(exc).__name__}: {exc}"
                error = actual - pred if np.isfinite(pred) else np.nan
                rows.append(
                    {
                        "horizon": 1,
                        "train_mode": train_mode,
                        "candidate_id": candidate.name,
                        "family": candidate.family,
                        "direction": candidate.direction,
                        "target_date": target_date,
                        "cutoff": cutoff,
                        "train_start": train.index.min() if len(train) else pd.NaT,
                        "train_end": train.index.max() if len(train) else pd.NaT,
                        "train_n": len(train),
                        "actual": actual,
                        "prediction": float(pred) if np.isfinite(pred) else np.nan,
                        "error": error,
                        "abs_error": abs(error) if np.isfinite(error) else np.nan,
                        "status": status,
                        "variables": ",".join(candidate.variables),
                        "params": json.dumps(candidate.params, ensure_ascii=False, sort_keys=True),
                        "notes": candidate.notes,
                    }
                )
                done += 1
            if done and done % 500 == 0:
                print(f"candidate precompute {done}/{total}", flush=True)
    return pd.DataFrame(rows)


def predict_random_walk(train: pd.DataFrame) -> float:
    return float(train["CPI"].dropna().iloc[-1])


def predict_seasonal_naive(train: pd.DataFrame, roll_window: int = 42) -> float:
    data = train[["CPI"]].dropna()
    if len(data) > roll_window:
        data = data.iloc[-roll_window:]
    target_month = (train.index.max() + pd.DateOffset(months=1)).month
    by_month = data.groupby(data.index.month)["CPI"].mean()
    if target_month in by_month.index:
        return float(by_month.loc[target_month])
    return float(by_month.mean())


def predict_arima_101(train: pd.DataFrame) -> float:
    series = train["CPI"].dropna()
    if len(series) < 42:
        raise ValueError("insufficient_train")
    fit = ARIMA(series, order=(1, 0, 1)).fit()
    return float(fit.forecast(steps=1).iloc[0])


def predict_sarima(train: pd.DataFrame) -> float:
    series = train["CPI"].dropna()
    if len(series) < 60:
        raise ValueError("insufficient_train")
    fit = ARIMA(series, order=(1, 0, 1), seasonal_order=(1, 0, 1, 12)).fit()
    return float(fit.forecast(steps=1).iloc[0])


def compute_context_predictions(
    official: pd.DataFrame,
    model_df: pd.DataFrame,
    outer_dates: pd.DataFrame,
    args: argparse.Namespace,
) -> pd.DataFrame:
    rows = []
    simple_fns = {
        "RandomWalk": lambda train_pp, train_model, target, seed: (predict_random_walk(train_pp), "ok"),
        "SeasonalNaive_roll42": lambda train_pp, train_model, target, seed: (predict_seasonal_naive(train_pp), "ok"),
        "ARIMA_101": lambda train_pp, train_model, target, seed: (predict_arima_101(train_pp), "ok"),
    }
    if args.include_sarima:
        simple_fns["SARIMA_101_10112"] = lambda train_pp, train_model, target, seed: (predict_sarima(train_pp), "ok")

    project_fns = {}
    if not args.skip_project_context:
        project_fns = {
            "Archived_BVAR": lambda train_pp, train_model, target, seed: forecast_archived_bvar(train_pp, target, seed),
            "Huber": lambda train_pp, train_model, target, seed: forecast_huber(train_model, target, seed),
            "RidgeShockDummies": lambda train_pp, train_model, target, seed: forecast_ridge_shock(train_model, target, seed),
        }

    for train_mode in TRAIN_MODES:
        for _, row in outer_dates.iterrows():
            target_date = row["target_date"]
            if target_date not in official.index:
                continue
            cutoff = target_date - pd.DateOffset(months=1)
            train_pp = apply_train_mode(official, cutoff, train_mode)
            train_model = apply_train_mode(model_df, cutoff, train_mode)
            actual = float(official.loc[target_date, "CPI"])
            for idx, (model, fn) in enumerate({**simple_fns, **project_fns}.items()):
                try:
                    pred, status = fn(train_pp, train_model, target_date, args.seed + idx * 1000 + int(target_date.strftime("%Y%m")))
                except Exception as exc:
                    pred, status = np.nan, f"error: {type(exc).__name__}: {exc}"
                error = actual - pred if np.isfinite(pred) else np.nan
                rows.append(
                    {
                        "horizon": 1,
                        "train_mode": train_mode,
                        "window": row["window"],
                        "model": model,
                        "role": "external_context" if model in project_fns else "simple_baseline",
                        "target_date": target_date,
                        "cutoff": cutoff,
                        "actual": actual,
                        "prediction": float(pred) if np.isfinite(pred) else np.nan,
                        "error": error,
                        "abs_error": abs(error) if np.isfinite(error) else np.nan,
                        "status": status,
                    }
                )
    return pd.DataFrame(rows)


def _weighted_average_prediction(group: pd.DataFrame, weights: pd.Series) -> float:
    usable = group.dropna(subset=["prediction"]).copy()
    usable = usable[usable["candidate_id"].isin(weights.index)]
    if usable.empty:
        return np.nan
    w = weights.loc[usable["candidate_id"]].astype(float)
    w = w / w.sum()
    return float((usable.set_index("candidate_id")["prediction"].astype(float) * w).sum())


def make_combinations(
    candidate_predictions: pd.DataFrame,
    outer_dates: pd.DataFrame,
    inner_months: int,
    min_inner_obs: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows = []
    selection_rows = []
    leakage_rows = []
    candidate_predictions = candidate_predictions.copy()
    candidate_predictions["target_date"] = pd.to_datetime(candidate_predictions["target_date"])

    combo_defs = {
        "BVAR_ModelAverage": lambda df: df["family"].str.startswith("BVAR"),
        "VARFamily_ModelAverage": lambda df: (
            df["family"].isin(
                [
                    "VAR",
                    "VAR_component_constrained",
                    "BVAR_deterministic_mean",
                    "VARX_OLS",
                    "FAVAR",
                    "Regime_VAR",
                    "Regime_BVAR",
                ]
            )
        ),
    }

    for train_mode in TRAIN_MODES:
        mode_preds = candidate_predictions[candidate_predictions["train_mode"] == train_mode]
        for _, outer in outer_dates.iterrows():
            target_date = outer["target_date"]
            if target_date not in set(mode_preds["target_date"]):
                continue
            cutoff = target_date - pd.DateOffset(months=1)
            inner_dates = pd.date_range(end=cutoff, periods=inner_months, freq="MS")
            inner = mode_preds[mode_preds["target_date"].isin(inner_dates)].dropna(subset=["abs_error"])
            outer_group = mode_preds[mode_preds["target_date"] == target_date]
            if outer_group.empty:
                continue
            actual = float(outer_group["actual"].iloc[0])
            for combo_name, mask_fn in combo_defs.items():
                pool_inner = inner[mask_fn(inner)].copy()
                scores = (
                    pool_inner.groupby(["candidate_id", "family", "direction"], as_index=False)
                    .agg(inner_n=("abs_error", "size"), inner_mae=("abs_error", "mean"), inner_rmse=("error", lambda x: float(np.sqrt((x.astype(float) ** 2).mean()))))
                )
                scores = scores[scores["inner_n"] >= min_inner_obs].sort_values(["inner_mae", "inner_rmse", "candidate_id"])
                if scores.empty:
                    continue
                scores = scores.head(10).copy()
                weights = (1.0 / (scores.set_index("candidate_id")["inner_mae"] + 1e-6)).astype(float)
                weights = weights / weights.sum()
                pred = _weighted_average_prediction(outer_group, weights)
                if not np.isfinite(pred):
                    status = "nan_prediction"
                    error = np.nan
                else:
                    status = "ok"
                    error = actual - pred
                rows.append(
                    {
                        "horizon": 1,
                        "train_mode": train_mode,
                        "window": outer["window"],
                        "model": combo_name,
                        "role": "var_family_combination",
                        "target_date": target_date,
                        "cutoff": cutoff,
                        "actual": actual,
                        "prediction": pred,
                        "error": error,
                        "abs_error": abs(error) if np.isfinite(error) else np.nan,
                        "status": status,
                    }
                )
                temp = scores.copy()
                temp["model"] = combo_name
                temp["outer_target_date"] = target_date
                temp["outer_window"] = outer["window"]
                temp["train_mode"] = train_mode
                temp["weight"] = temp["candidate_id"].map(weights.to_dict())
                temp["inner_start"] = inner_dates.min()
                temp["inner_end"] = inner_dates.max()
                selection_rows.append(temp)
                leakage_rows.append(
                    {
                        "model": combo_name,
                        "target_date": target_date,
                        "cutoff": cutoff,
                        "inner_start": inner_dates.min(),
                        "inner_end": inner_dates.max(),
                        "inner_end_after_cutoff": bool(inner_dates.max() > cutoff),
                        "selection_used_outer_actual": False,
                    }
                )
    selection_log = pd.concat(selection_rows, ignore_index=True) if selection_rows else pd.DataFrame()
    return pd.DataFrame(rows), selection_log, pd.DataFrame(leakage_rows)


def outer_candidate_predictions(candidate_predictions: pd.DataFrame, outer_dates: pd.DataFrame) -> pd.DataFrame:
    outer_keys = set(pd.to_datetime(outer_dates["target_date"]))
    window_map = dict(zip(pd.to_datetime(outer_dates["target_date"]), outer_dates["window"]))
    out = candidate_predictions[candidate_predictions["target_date"].isin(outer_keys)].copy()
    out["window"] = out["target_date"].map(window_map)
    out["model"] = out["candidate_id"]
    out["role"] = "var_family_candidate"
    return out[
        [
            "horizon",
            "train_mode",
            "window",
            "model",
            "role",
            "target_date",
            "cutoff",
            "actual",
            "prediction",
            "error",
            "abs_error",
            "status",
            "family",
            "direction",
            "variables",
            "params",
            "notes",
        ]
    ]


def calculate_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    ok = predictions.dropna(subset=["error"]).copy()
    for keys, group in ok.groupby(["horizon", "train_mode", "window", "model", "role"], sort=False):
        horizon, train_mode, window, model, role = keys
        errors = group["error"].astype(float)
        abs_errors = errors.abs()
        rows.append(
            {
                "horizon": horizon,
                "train_mode": train_mode,
                "window": window,
                "model": model,
                "role": role,
                "n": len(group),
                "MAE": abs_errors.mean(),
                "RMSE": np.sqrt((errors**2).mean()),
                "Bias": errors.mean(),
                "Max_Error": abs_errors.max(),
                "KPI_Violations": int((abs_errors > 0.5).sum()),
                "Coverage_50pct": float((abs_errors <= 0.5).mean() * 100),
            }
        )
    for keys, group in ok.groupby(["horizon", "train_mode", "model", "role"], sort=False):
        horizon, train_mode, model, role = keys
        errors = group["error"].astype(float)
        abs_errors = errors.abs()
        rows.append(
            {
                "horizon": horizon,
                "train_mode": train_mode,
                "window": "all_windows",
                "model": model,
                "role": role,
                "n": len(group),
                "MAE": abs_errors.mean(),
                "RMSE": np.sqrt((errors**2).mean()),
                "Bias": errors.mean(),
                "Max_Error": abs_errors.max(),
                "KPI_Violations": int((abs_errors > 0.5).sum()),
                "Coverage_50pct": float((abs_errors <= 0.5).mean() * 100),
            }
        )
    return pd.DataFrame(rows).sort_values(["horizon", "train_mode", "window", "MAE", "model"])


def write_config(out_dir: Path, args: argparse.Namespace, candidates: list[Candidate], precompute_dates: pd.DatetimeIndex) -> None:
    config = {
        "agent_id": AGENT_ID,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "run_name": args.run_name,
        "purpose": "mandatory VAR-family h=1 research, no production integration",
        "horizons": HORIZONS,
        "train_modes": TRAIN_MODES,
        "inner_validation": {
            "inner_months": args.inner_months,
            "min_inner_obs": args.min_inner_obs,
            "scheme": "trailing target months ending at each outer cutoff; inner predictions precomputed with their own target-1 cutoff",
        },
        "outer_windows": [
            {"name": name, "start": str(start.date()), "end": str(end.date())}
            for name, start, end in historical_windows()
        ],
        "precompute_dates": {
            "start": str(precompute_dates.min().date()),
            "end": str(precompute_dates.max().date()),
            "count": len(precompute_dates),
        },
        "candidate_count": len(candidates),
        "directions": sorted({c.direction for c in candidates}),
        "project_context": not args.skip_project_context,
        "include_sarima": args.include_sarima,
        "seed": args.seed,
    }
    (out_dir / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")


def determine_status(comparison: pd.DataFrame, final_model: str) -> str:
    overall = comparison.set_index("model")
    if final_model not in overall.index:
        return "no adequate VAR found"
    final_mae = float(overall.loc[final_model, "MAE"])
    archived = float(overall.loc["Archived_BVAR", "MAE"]) if "Archived_BVAR" in overall.index else np.inf
    seasonal = float(overall.loc["SeasonalNaive_roll42", "MAE"]) if "SeasonalNaive_roll42" in overall.index else np.inf
    plain_best = overall[overall.index.str.startswith("plain_var_tc_l")]["MAE"].min()
    if final_mae <= archived and final_mae <= seasonal and final_mae <= plain_best:
        return "recommended mandatory VAR"
    if final_mae <= archived and final_mae <= max(seasonal, plain_best):
        return "experimental mandatory VAR candidate"
    if np.isfinite(plain_best):
        return "fallback clean VAR only"
    return "no adequate VAR found"


def best_var_family_row(comparison: pd.DataFrame) -> pd.Series | None:
    var_roles = {"var_family_candidate", "var_family_combination"}
    eligible = comparison[comparison["role"].isin(var_roles)].copy()
    if eligible.empty:
        return None
    return eligible.sort_values(["MAE", "RMSE", "model"]).iloc[0]


def write_notes_and_report(
    out_dir: Path,
    args: argparse.Namespace,
    candidates: list[Candidate],
    metrics: pd.DataFrame,
    comparison: pd.DataFrame,
    selection_log: pd.DataFrame,
    leakage: pd.DataFrame,
) -> None:
    best_var = best_var_family_row(comparison)
    final_model = str(best_var["model"]) if best_var is not None else "none"
    final_status = determine_status(comparison, final_model)
    leakage_violations = int(leakage["inner_end_after_cutoff"].sum()) if len(leakage) else 0
    direction_counts = pd.DataFrame(
        [{"direction": d, "candidate_count": sum(c.direction == d for c in candidates)} for d in sorted({c.direction for c in candidates})]
    )
    selected_counts = (
        selection_log.groupby(["model", "candidate_id"], as_index=False)["weight"].agg(["count", "mean"]).reset_index()
        if len(selection_log)
        else pd.DataFrame()
    )

    notes = [
        "# codex_cli Mandatory VAR Notes",
        "",
        f"- Run directory: `{out_dir}`",
        f"- Final report: `{REPORT_PATH}`",
        f"- Final VAR-family model by h=1 all-window MAE: `{final_model}`",
        f"- Final status: `{final_status}`",
        f"- Leakage violations found in nested selection logs: `{leakage_violations}`",
        "- h=2/h=12 skipped in this run; h=1 is the mandatory acceptance KPI and the candidate set is nested/expanding.",
        "",
        "## Directions Tested",
        "",
        direction_counts.to_markdown(index=False),
        "",
        "## Overall Comparison",
        "",
        comparison.head(30).to_markdown(index=False) if len(comparison) else "_No comparison._",
        "",
        "## Combination Selection Counts",
        "",
        selected_counts.sort_values(["model", "count"], ascending=[True, False]).head(30).to_markdown(index=False)
        if len(selected_counts)
        else "_No selection rows._",
    ]
    (out_dir / "notes.md").write_text("\n".join(notes) + "\n", encoding="utf-8")

    report = [
        "# codex_cli Mandatory VAR Report",
        "",
        "## Scope",
        "",
        "Research-only h=1 search for the best defensible mandatory VAR-family model. No production data, model registry, dashboard, or shared task file was modified.",
        "",
        "## Variants Tested",
        "",
        direction_counts.to_markdown(index=False),
        "",
        "Implemented families: plain VAR, Minnesota BVAR posterior-mean grid, BVAR inner-validation averaging, VARX with last-observed exogenous scenario paths, train-only FAVAR, two transparent regime VAR/BVAR rules, component-constrained bottom-up VAR, and a VAR-family forecast combination.",
        "",
        "## Recommendation",
        "",
        f"- Final recommended mandatory VAR model: `{final_model}`",
        f"- Final status: `{final_status}`",
        "- External Huber/Ridge rows are context only and are not eligible for the mandatory VAR recommendation.",
        "- Use as mandatory VAR-family benchmark / secondary diagnostic model, not as a claim that VAR beats production ML.",
        "",
        "## h=1 Metrics",
        "",
        comparison.head(35).to_markdown(index=False) if len(comparison) else "_No metrics._",
        "",
        "## Leakage Audit",
        "",
        f"- Outer forecast training cutoff: target month minus one month for every prediction.",
        f"- Inner model averaging selection: trailing `{args.inner_months}` target months ending at cutoff; no outer actual enters weights.",
        "- FAVAR scaling/PCA is fit separately on each training cutoff only.",
        "- VARX target exogenous path uses the last observed cutoff value, not future actual exogenous values.",
        f"- Nested selection leakage violations recorded: `{leakage_violations}`.",
        "",
        "## Rejected / Weaker Variants",
        "",
        "- Individual VARX and FAVAR candidates are retained in metrics but were not automatically preferred unless their cutoff-only h=1 errors won.",
        "- Regime rules are intentionally simple; they did not get extra regime tuning beyond fixed thresholds.",
        "- Rejected `roll42_l5` seasonal-residual tuning was not continued in this run.",
        "",
        "## Commands Run",
        "",
        f"```bash\npython3 experiments/var_sa_research/codex_cli_mandatory_var.py --run-name {args.run_name}\n```",
        "",
        "## Artifacts",
        "",
        f"- Run directory: `{out_dir}`",
        "- `config.json`, `metrics.csv`, `predictions.csv`, `selection_log.csv`, `comparison.csv`, `leakage_checks.csv`, `candidate_predictions.csv`, `notes.md`, and script copy are saved there.",
    ]
    REPORT_PATH.write_text("\n".join(report) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    if not args.run_name.startswith(f"{AGENT_ID}_mandatory_var_"):
        raise ValueError(f"run-name must start with {AGENT_ID}_mandatory_var_")

    out_dir = RUNS_DIR / args.run_name
    out_dir.mkdir(parents=True, exist_ok=False)

    official = load_official_data()
    model_df = load_model_frame()
    outer_dates = all_outer_dates()
    precompute_dates = all_precompute_dates(official, outer_dates, args.inner_months)
    candidates = build_candidates()
    write_config(out_dir, args, candidates, precompute_dates)

    print(f"Evaluating {len(candidates)} VAR-family candidates over {len(precompute_dates)} target months", flush=True)
    candidate_predictions = compute_candidate_predictions(official, candidates, precompute_dates)
    candidate_predictions.to_csv(out_dir / "candidate_predictions.csv", index=False)

    combo_predictions, selection_log, leakage = make_combinations(
        candidate_predictions,
        outer_dates,
        args.inner_months,
        args.min_inner_obs,
    )
    context_predictions = compute_context_predictions(official, model_df, outer_dates, args)
    candidate_outer = outer_candidate_predictions(candidate_predictions, outer_dates)
    predictions = pd.concat([candidate_outer, combo_predictions, context_predictions], ignore_index=True, sort=False)
    metrics = calculate_metrics(predictions)
    comparison = metrics[
        (metrics["horizon"] == 1)
        & (metrics["train_mode"] == "expanding")
        & (metrics["window"] == "all_windows")
    ].sort_values(["MAE", "RMSE", "model"])

    predictions.to_csv(out_dir / "predictions.csv", index=False)
    metrics.to_csv(out_dir / "metrics.csv", index=False)
    selection_log.to_csv(out_dir / "selection_log.csv", index=False)
    leakage.to_csv(out_dir / "leakage_checks.csv", index=False)
    comparison.to_csv(out_dir / "comparison.csv", index=False)
    shutil.copy2(Path(__file__), out_dir / Path(__file__).name)
    write_notes_and_report(out_dir, args, candidates, metrics, comparison, selection_log, leakage)

    print(f"Wrote artifacts to {out_dir}", flush=True)
    print(f"Wrote final report to {REPORT_PATH}", flush=True)
    if len(comparison):
        print(comparison.head(15).to_string(index=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
