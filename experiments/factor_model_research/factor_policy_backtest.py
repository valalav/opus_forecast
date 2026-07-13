#!/usr/bin/env python3
"""
Factor-family research backtest for the mandatory factor model.

This runner compares leakage-safe PCA/FAVAR and statsmodels DFM variants on the
same source-of-truth monthly facts. It saves diagnostics, predictions, metrics,
trajectory checks, and a concise report for management-facing evidence.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.linear_model import HuberRegressor
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sirena.models.factor_bridge import FactorBridgeForecaster
from sirena.models.factor_policy import FactorPolicyForecaster

try:
    from statsmodels.tsa.statespace.dynamic_factor import DynamicFactor

    STATSMODELS_DFM_AVAILABLE = True
except Exception:
    STATSMODELS_DFM_AVAILABLE = False


TARGET = "CPI"
COMPONENTS = ["Food", "NonFood", "Services"]
MACRO = ["USD", "Ki_i", "Ruonia", "Ruonia_i", "Deposits", "RetailReal"]
ALL_INFO = COMPONENTS + MACRO
OUTLIER_YEARS = {2010, 2022}


@dataclass(frozen=True)
class Candidate:
    name: str
    kind: str
    info_cols: tuple[str, ...]
    factors: int
    lags: int
    seasonal: bool
    robust: bool
    note: str
    estimator: str = "huber"
    factor_method: str = "pca"
    window: str = "expanding"
    publication_lag: int = 0
    regime_split: str = "none"


def load_official() -> pd.DataFrame:
    path = ROOT / "data" / "inflation_data.csv"
    raw = pd.read_csv(path, sep=";", decimal=",")
    raw["Date"] = pd.to_datetime(raw["Date"], format="%d.%m.%Y", errors="coerce")
    raw["Date"] = raw["Date"].dt.to_period("M").dt.to_timestamp()
    raw = raw.set_index("Date").sort_index()

    df = pd.DataFrame(index=raw.index)
    df["CPI"] = pd.to_numeric(raw["mom"], errors="coerce") - 100
    df["Food"] = pd.to_numeric(raw["Prod"], errors="coerce") - 100
    df["NonFood"] = pd.to_numeric(raw["Nonprod"], errors="coerce") - 100
    df["Services"] = pd.to_numeric(raw["Serv"], errors="coerce") - 100
    df["USD"] = pd.to_numeric(raw["usd_nom_i"], errors="coerce") - 100
    df["Ki_i"] = pd.to_numeric(raw["Ki_i"], errors="coerce")
    df["Ruonia"] = pd.to_numeric(raw["Ruonia"], errors="coerce")
    df["Ruonia_i"] = pd.to_numeric(raw["Ruonia_i"], errors="coerce")
    df["Deposits"] = pd.to_numeric(raw["fl_dep"], errors="coerce") - 100
    df["RetailReal"] = pd.to_numeric(raw["all_real"], errors="coerce") - 100
    return df.dropna(subset=[TARGET])


def robust_z(series: pd.Series) -> pd.Series:
    x = series.dropna()
    med = x.median()
    mad = (x - med).abs().median()
    if not mad or np.isnan(mad):
        return pd.Series(index=series.index, dtype=float)
    return 0.6745 * (series - med) / mad


def save_diagnostics(data: pd.DataFrame, out_dir: Path) -> dict:
    diagnostics = {
        "source": "data/inflation_data.csv",
        "start": data.index.min().strftime("%Y-%m"),
        "end": data.index.max().strftime("%Y-%m"),
        "n_months": int(len(data)),
        "columns": list(data.columns),
        "scale": {
            "CPI/components/USD/Deposits/RetailReal": "MoM p.p. (index minus 100)",
            "Ki_i/Ruonia/Ruonia_i": "raw source scale, standardized before PCA/DFM",
        },
        "missing_by_column": data.isna().sum().astype(int).to_dict(),
        "robust_abs_z_gt_3_5_by_column": {
            col: int((robust_z(data[col]).abs() > 3.5).sum()) for col in data.columns
        },
        "shock_months_cpi_abs_gt_1": [
            idx.strftime("%Y-%m") for idx, val in data[TARGET].items() if abs(val) > 1.0
        ],
        "years_excluded_only_in_seasonal_diagnostics": sorted(OUTLIER_YEARS),
    }
    (out_dir / "data_diagnostics.json").write_text(
        json.dumps(diagnostics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return diagnostics


def build_candidates(include_dfm: bool = True, include_quantile_bridge: bool = True) -> list[Candidate]:
    candidates: list[Candidate] = [
        Candidate(
            name="SeasonalNaive_mean",
            kind="baseline",
            info_cols=(),
            factors=0,
            lags=0,
            seasonal=True,
            robust=False,
            estimator="mean",
            note="Train-only month-of-year mean baseline.",
        ),
        Candidate(
            name="SeasonalNaive_lastyear",
            kind="baseline",
            info_cols=(),
            factors=0,
            lags=0,
            seasonal=True,
            robust=False,
            estimator="lastyear",
            note="Last observed same calendar month baseline.",
        ),
        Candidate(
            name="SeasonalAR1",
            kind="baseline",
            info_cols=(),
            factors=0,
            lags=1,
            seasonal=True,
            robust=False,
            estimator="seasonal_ar1",
            note="Simple CPI AR(1) with train-only month seasonality.",
        ),
    ]
    info_sets = {
        "components": tuple(COMPONENTS),
        "macro": tuple(MACRO),
        "all": tuple(ALL_INFO),
        "lean": ("Food", "NonFood", "Services", "USD", "Ki_i", "Ruonia"),
    }
    for label, cols in info_sets.items():
        max_factors = min(3, len(cols))
        for factors in range(1, max_factors + 1):
            for lags in (1, 2):
                for seasonal in (False, True):
                    candidates.append(
                        Candidate(
                            name=f"FAVAR_{label}_f{factors}_l{lags}_{'seasonal' if seasonal else 'raw'}_ols",
                            kind="favar",
                            info_cols=cols,
                            factors=factors,
                            lags=lags,
                            seasonal=seasonal,
                            robust=False,
                            note=f"PCA factors from {label}; VAR on CPI+factors.",
                        )
                    )
                    candidates.append(
                        Candidate(
                            name=f"RobustFAVAR_{label}_f{factors}_l{lags}_{'seasonal' if seasonal else 'raw'}",
                            kind="favar",
                            info_cols=cols,
                            factors=factors,
                            lags=lags,
                            seasonal=seasonal,
                            robust=True,
                            note=f"Huber equations; PCA factors from {label}.",
                        )
                    )

    if include_dfm and STATSMODELS_DFM_AVAILABLE:
        for label, cols in {"components": tuple(COMPONENTS), "lean": ("Food", "NonFood", "Services", "USD", "Ki_i", "Ruonia")}.items():
            for factors in (1, 2):
                for seasonal in (False, True):
                    candidates.append(
                        Candidate(
                            name=f"DFM_{label}_f{factors}_q1_{'seasonal' if seasonal else 'raw'}",
                            kind="dfm",
                            info_cols=cols,
                            factors=factors,
                            lags=1,
                            seasonal=seasonal,
                            robust=False,
                            note="statsmodels DynamicFactor on CPI plus information set.",
                        )
                    )
    bridge_estimators = [("huber", True), ("ridge", False)]
    if include_quantile_bridge:
        bridge_estimators.append(("quantile", True))
    for estimator, robust in bridge_estimators:
        factor_methods = ("pca",) if estimator == "quantile" else ("pca", "average", "weighted_pca", "sparse_pca")
        lag_grid = (1,) if estimator == "quantile" else (1, 2, 3)
        for factor_method in factor_methods:
            for lags in lag_grid:
                candidates.append(
                    Candidate(
                        name=f"BlockBridge_{factor_method}_l{lags}_{estimator}_seasonal",
                        kind="bridge",
                        info_cols=("Food", "NonFood", "Services", "USD", "Ki_i", "Ruonia"),
                        factors=2,
                        lags=lags,
                        seasonal=True,
                        robust=robust,
                        estimator=estimator,
                        factor_method=factor_method,
                        note="Direct block-factor bridge: component factor + macro factor, horizon-specific equation.",
                    )
                )
    for rolling_window in ("rolling84", "rolling120"):
        candidates.append(
            Candidate(
                name=f"BlockBridge_pca_l1_huber_seasonal_{rolling_window}",
                kind="bridge",
                info_cols=("Food", "NonFood", "Services", "USD", "Ki_i", "Ruonia"),
                factors=2,
                lags=1,
                seasonal=True,
                robust=True,
                estimator="huber",
                factor_method="pca",
                window=rolling_window,
                note=f"Block bridge with {rolling_window} train window.",
            )
        )
        candidates.append(
            Candidate(
                name=f"RobustFAVAR_lean_f2_l1_seasonal_{rolling_window}",
                kind="favar",
                info_cols=("Food", "NonFood", "Services", "USD", "Ki_i", "Ruonia"),
                factors=2,
                lags=1,
                seasonal=True,
                robust=True,
                window=rolling_window,
                note=f"Selected FAVAR with {rolling_window} train window.",
            )
        )
    candidates.extend(
        [
            Candidate(
                name="BlockBridge_pca_l1_huber_seasonal_pubLag1",
                kind="bridge",
                info_cols=("Food", "NonFood", "Services", "USD", "Ki_i", "Ruonia"),
                factors=2,
                lags=1,
                seasonal=True,
                robust=True,
                estimator="huber",
                factor_method="pca",
                publication_lag=1,
                note="Publication-lag proxy: macro/financial columns shifted by one month inside each train window.",
            ),
            Candidate(
                name="RobustFAVAR_lean_f2_l1_seasonal_pubLag1",
                kind="favar",
                info_cols=("Food", "NonFood", "Services", "USD", "Ki_i", "Ruonia"),
                factors=2,
                lags=1,
                seasonal=True,
                robust=True,
                publication_lag=1,
                note="Selected FAVAR with one-month macro publication lag proxy.",
            ),
            Candidate(
                name="BlockBridge_pca_l1_huber_regimeMedian",
                kind="regime_bridge",
                info_cols=("Food", "NonFood", "Services", "USD", "Ki_i", "Ruonia"),
                factors=2,
                lags=1,
                seasonal=True,
                robust=True,
                estimator="huber",
                factor_method="pca",
                regime_split="median",
                note="Train-only median CPI regime split; falls back to full train if subset is too short.",
            ),
        ]
    )
    return candidates


def _seasonal_adjust(train: pd.DataFrame, columns: Iterable[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    work = train.loc[:, list(columns)].copy().ffill().bfill()
    month_means = work.groupby(work.index.month).mean()
    for month, means in month_means.iterrows():
        mask = work.index.month == month
        work.loc[mask, :] = work.loc[mask, :] - means.values
    return work, month_means


def _design(values: np.ndarray, lags: int) -> tuple[np.ndarray, np.ndarray]:
    x_rows, y_rows = [], []
    for t in range(lags, len(values)):
        row = [1.0]
        for lag in range(1, lags + 1):
            row.extend(values[t - lag])
        x_rows.append(row)
        y_rows.append(values[t])
    return np.asarray(x_rows), np.asarray(y_rows)


def _fit_equations(x: np.ndarray, y: np.ndarray, robust: bool) -> np.ndarray:
    betas = []
    for j in range(y.shape[1]):
        if robust:
            try:
                model = HuberRegressor(
                    alpha=0.0,
                    epsilon=1.35,
                    fit_intercept=False,
                    max_iter=300,
                )
                model.fit(x, y[:, j])
                betas.append(model.coef_)
                continue
            except Exception:
                pass
        betas.append(np.linalg.lstsq(x, y[:, j], rcond=None)[0])
    return np.asarray(betas)


def _apply_window(train: pd.DataFrame, candidate: Candidate) -> pd.DataFrame:
    if candidate.window.startswith("rolling"):
        try:
            size = int(candidate.window.replace("rolling", ""))
        except ValueError:
            return train
        if len(train) > size:
            return train.tail(size)
    return train


def _apply_publication_lag(train: pd.DataFrame, candidate: Candidate) -> pd.DataFrame:
    if candidate.publication_lag <= 0:
        return train
    lagged = train.copy()
    lag_cols = [col for col in MACRO if col in lagged.columns]
    if lag_cols:
        lagged[lag_cols] = lagged[lag_cols].shift(candidate.publication_lag)
    return lagged.ffill().bfill()


def prepare_train(train: pd.DataFrame, candidate: Candidate) -> pd.DataFrame:
    return _apply_publication_lag(_apply_window(train, candidate), candidate)


def baseline_forecast(train: pd.DataFrame, candidate: Candidate, horizon: int) -> np.ndarray:
    data = train[[TARGET]].copy().ffill().bfill().dropna()
    if len(data) < 36:
        return np.full(horizon, np.nan)
    last_date = data.index.max()

    if candidate.estimator == "lastyear":
        path = []
        for step in range(1, horizon + 1):
            target_date = last_date + pd.DateOffset(months=step)
            same_month = data[data.index.month == target_date.month][TARGET]
            path.append(float(same_month.iloc[-1]) if not same_month.empty else float(data[TARGET].iloc[-1]))
        return np.asarray(path)

    if candidate.estimator == "seasonal_ar1":
        seasonal, month_means = _seasonal_adjust(data, [TARGET])
        y = seasonal[TARGET].values.astype(float)
        if len(y) < 24 or np.std(y[:-1]) < 1e-9:
            return baseline_forecast(train, Candidate("SeasonalNaive_mean", "baseline", (), 0, 0, True, False, "fallback", estimator="mean"), horizon)
        x = np.column_stack([np.ones(len(y) - 1), y[:-1]])
        beta = np.linalg.lstsq(x, y[1:], rcond=None)[0]
        current = float(y[-1])
        path = []
        for step in range(1, horizon + 1):
            current = float(beta[0] + beta[1] * current)
            month = (last_date + pd.DateOffset(months=step)).month
            path.append(current + float(month_means.loc[month, TARGET]))
        return np.asarray(path)

    month_means = data.groupby(data.index.month)[TARGET].mean()
    return np.asarray(
        [
            float(month_means.get((last_date + pd.DateOffset(months=step)).month, data[TARGET].mean()))
            for step in range(1, horizon + 1)
        ],
        dtype=float,
    )


def favar_forecast(train: pd.DataFrame, candidate: Candidate, horizon: int) -> np.ndarray:
    train = prepare_train(train, candidate)
    cols = [TARGET] + [col for col in candidate.info_cols if col in train.columns]
    if len(cols) - 1 < candidate.factors:
        return np.full(horizon, np.nan)
    data = train.loc[:, cols].copy().ffill().bfill().dropna()
    if len(data) < 60:
        return np.full(horizon, np.nan)

    month_means = None
    if candidate.seasonal:
        data, month_means = _seasonal_adjust(data, cols)

    info_cols = cols[1:]
    nonconstant = [col for col in info_cols if data[col].nunique(dropna=True) > 1]
    if len(nonconstant) < candidate.factors:
        return np.full(horizon, np.nan)
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(data[nonconstant].values.astype(float))
    pca = PCA(n_components=candidate.factors)
    factors = pca.fit_transform(x_scaled)

    factor_cols = [f"F{i + 1}" for i in range(candidate.factors)]
    var_data = pd.DataFrame(factors, index=data.index, columns=factor_cols)
    var_data.insert(0, TARGET, data[TARGET].astype(float).values)
    x, y = _design(var_data.values.astype(float), candidate.lags)
    if len(x) < 24:
        return np.full(horizon, np.nan)
    beta = _fit_equations(x, y, candidate.robust)

    hist = [row for row in var_data.values[-candidate.lags :]]
    path = []
    last_date = data.index.max()
    for step in range(1, horizon + 1):
        row = [1.0]
        for lag in range(1, candidate.lags + 1):
            row.extend(hist[-lag])
        pred_vec = (np.asarray([row]) @ beta.T).ravel()
        hist.append(pred_vec)
        pred = float(pred_vec[0])
        if candidate.seasonal and month_means is not None:
            month = (last_date + pd.DateOffset(months=step)).month
            pred += float(month_means.loc[month, TARGET])
        path.append(pred)
    return np.asarray(path)


def dfm_forecast(train: pd.DataFrame, candidate: Candidate, horizon: int) -> np.ndarray:
    train = prepare_train(train, candidate)
    if not STATSMODELS_DFM_AVAILABLE:
        return np.full(horizon, np.nan)
    cols = [TARGET] + [col for col in candidate.info_cols if col in train.columns]
    if len(cols) < candidate.factors + 1:
        return np.full(horizon, np.nan)
    data = train.loc[:, cols].copy().ffill().bfill().dropna()
    if len(data) < 72:
        return np.full(horizon, np.nan)

    month_means = None
    if candidate.seasonal:
        data, month_means = _seasonal_adjust(data, cols)

    nonconstant = [col for col in cols if data[col].nunique(dropna=True) > 1]
    if TARGET not in nonconstant or len(nonconstant) < candidate.factors + 1:
        return np.full(horizon, np.nan)
    data = data[nonconstant]

    scaler = StandardScaler()
    scaled = scaler.fit_transform(data.values.astype(float))
    try:
        model = DynamicFactor(
            scaled,
            k_factors=candidate.factors,
            factor_order=candidate.lags,
            error_cov_type="diagonal",
            enforce_stationarity=True,
        )
        result = model.fit(disp=False, maxiter=250)
        pred_scaled = np.asarray(result.forecast(steps=horizon))
        pred = scaler.inverse_transform(pred_scaled)
    except Exception:
        return np.full(horizon, np.nan)

    cpi_idx = data.columns.get_loc(TARGET)
    path = pred[:, cpi_idx].astype(float)
    if candidate.seasonal and month_means is not None:
        last_date = data.index.max()
        for idx in range(horizon):
            month = (last_date + pd.DateOffset(months=idx + 1)).month
            path[idx] += float(month_means.loc[month, TARGET])
    return path


def bridge_forecast(train: pd.DataFrame, candidate: Candidate, horizon: int) -> np.ndarray:
    train = prepare_train(train, candidate)
    try:
        model = FactorBridgeForecaster(
            lags=candidate.lags,
            estimator=candidate.estimator,
            seasonal=candidate.seasonal,
            factor_method=candidate.factor_method,
            component_cols=COMPONENTS,
            macro_cols=[col for col in candidate.info_cols if col not in COMPONENTS],
            max_horizon=max(12, horizon),
            rolling_window=None,
        )
        model.fit(train)
        return model.forecast(horizon)
    except Exception:
        return np.full(horizon, np.nan)


def regime_bridge_forecast(train: pd.DataFrame, candidate: Candidate, horizon: int) -> np.ndarray:
    prepared = prepare_train(train, candidate)
    threshold = float(prepared[TARGET].median())
    current_high = float(prepared[TARGET].iloc[-1]) > threshold
    subset = prepared[prepared[TARGET] > threshold] if current_high else prepared[prepared[TARGET] <= threshold]
    if len(subset) >= FactorBridgeForecaster.MIN_TRAIN_SIZE:
        prepared = subset
    return bridge_forecast(prepared, candidate, horizon)


def forecast_candidate(train: pd.DataFrame, candidate: Candidate, horizon: int) -> np.ndarray:
    if candidate.kind == "baseline":
        return baseline_forecast(prepare_train(train, candidate), candidate, horizon)
    if candidate.kind == "regime_bridge":
        return regime_bridge_forecast(train, candidate, horizon)
    if candidate.kind == "bridge":
        return bridge_forecast(train, candidate, horizon)
    if candidate.kind == "dfm":
        return dfm_forecast(train, candidate, horizon)
    return favar_forecast(train, candidate, horizon)


def evaluate_candidates(data: pd.DataFrame, candidates: list[Candidate], horizons: tuple[int, ...]) -> pd.DataFrame:
    rows = []
    for horizon in horizons:
        target_dates = pd.date_range(start="2018-01-01", end=data.index.max(), freq="MS")
        for target_date in target_dates:
            if target_date not in data.index:
                continue
            cutoff = target_date - pd.DateOffset(months=horizon)
            train = data[data.index <= cutoff].copy()
            if len(train) < 72:
                continue
            actual = float(data.loc[target_date, TARGET])
            prev_date = target_date - pd.DateOffset(months=1)
            prev_actual = float(data.loc[prev_date, TARGET]) if prev_date in data.index else np.nan
            for candidate in candidates:
                try:
                    path = forecast_candidate(train, candidate, horizon)
                    pred = float(path[horizon - 1]) if len(path) >= horizon else np.nan
                except Exception:
                    pred = np.nan
                rows.append(
                    {
                        "horizon": horizon,
                        "target_date": target_date,
                        "cutoff": cutoff,
                        "candidate": candidate.name,
                        "actual": actual,
                        "prev_actual": prev_actual,
                        "prediction": pred,
                        "actual_change": actual - prev_actual if np.isfinite(prev_actual) else np.nan,
                        "predicted_change": pred - prev_actual if np.isfinite(prev_actual) and not np.isnan(pred) else np.nan,
                        "error": actual - pred if not np.isnan(pred) else np.nan,
                    }
                )
    return pd.DataFrame(rows)


def calculate_metrics(predictions: pd.DataFrame, candidates: list[Candidate]) -> pd.DataFrame:
    config = {candidate.name: candidate for candidate in candidates}
    rows = []
    for (horizon, candidate_name), group in predictions.groupby(["horizon", "candidate"]):
        valid = group.dropna(subset=["prediction", "actual"])
        if valid.empty:
            continue
        errors = valid["actual"] - valid["prediction"]
        nonshock = valid[~valid["target_date"].dt.year.isin(OUTLIER_YEARS)]
        nonshock_errors = nonshock["actual"] - nonshock["prediction"] if not nonshock.empty else pd.Series(dtype=float)
        directional = valid.dropna(subset=["actual_change", "predicted_change"])
        if directional.empty:
            directional_accuracy = np.nan
        else:
            directional_accuracy = float(
                (np.sign(directional["actual_change"]) == np.sign(directional["predicted_change"])).mean() * 100
            )
        candidate = config[candidate_name]
        rows.append(
            {
                "candidate": candidate_name,
                "horizon": int(horizon),
                "kind": candidate.kind,
                "factors": candidate.factors,
                "lags": candidate.lags,
                "seasonal": candidate.seasonal,
                "robust": candidate.robust,
                "estimator": candidate.estimator,
                "factor_method": candidate.factor_method,
                "window": candidate.window,
                "publication_lag": candidate.publication_lag,
                "regime_split": candidate.regime_split,
                "info_cols": ",".join(candidate.info_cols),
                "n": int(len(valid)),
                "mae": float(errors.abs().mean()),
                "rmse": float(np.sqrt((errors**2).mean())),
                "bias": float(errors.mean()),
                "kpi_violations": int((errors.abs() > 0.5).sum()),
                "coverage_50pct": float((errors.abs() <= 0.5).mean() * 100),
                "max_abs_error": float(errors.abs().max()),
                "nonshock_mae": float(nonshock_errors.abs().mean()) if len(nonshock_errors) else np.nan,
                "directional_accuracy": directional_accuracy,
            }
        )
    metrics = pd.DataFrame(rows)
    wide = metrics.pivot(index="candidate", columns="horizon", values="mae")
    for h in (1, 2, 12):
        metrics[f"mae_h{h}"] = metrics["candidate"].map(wide.get(h, pd.Series(dtype=float)))
    metrics["weighted_score"] = (
        0.50 * metrics["mae_h1"].fillna(9.0)
        + 0.30 * metrics["mae_h2"].fillna(9.0)
        + 0.20 * metrics["mae_h12"].fillna(9.0)
    )
    for baseline_name, prefix in (
        ("RobustFAVAR_lean_f2_l1_seasonal", "rel_favar"),
        ("SeasonalNaive_mean", "rel_seasonal"),
    ):
        baseline_wide = metrics[metrics["candidate"] == baseline_name].set_index("horizon")["mae"]
        metrics[prefix] = metrics.apply(
            lambda row: float(row["mae"] / baseline_wide.loc[row["horizon"]])
            if row["horizon"] in baseline_wide.index and baseline_wide.loc[row["horizon"]] > 0
            else np.nan,
            axis=1,
        )
    return metrics.sort_values(["weighted_score", "horizon", "mae"])


def residual_diagnostics(predictions: pd.DataFrame, reference: str = "SeasonalNaive_mean") -> pd.DataFrame:
    rows = []
    ref = predictions[predictions["candidate"] == reference][
        ["horizon", "target_date", "error"]
    ].rename(columns={"error": "ref_error"})
    for (horizon, candidate_name), group in predictions.groupby(["horizon", "candidate"]):
        valid = group.dropna(subset=["error", "prediction", "actual"]).sort_values("target_date")
        if len(valid) < 20:
            continue
        errors = valid["error"].astype(float)
        lb_p = np.nan
        arch_p = np.nan
        try:
            lb_p = float(acorr_ljungbox(errors, lags=[min(12, len(errors) // 3)], return_df=True)["lb_pvalue"].iloc[0])
        except Exception:
            pass
        try:
            arch_p = float(het_arch(errors, nlags=min(4, max(1, len(errors) // 10)))[1])
        except Exception:
            pass

        merged = valid[["horizon", "target_date", "error"]].merge(ref, on=["horizon", "target_date"], how="inner")
        dm_t = np.nan
        dm_p = np.nan
        dm_loss_delta = np.nan
        if len(merged) >= 20 and candidate_name != reference:
            loss_delta = merged["error"].abs() - merged["ref_error"].abs()
            dm_loss_delta = float(loss_delta.mean())
            se = float(loss_delta.std(ddof=1) / math.sqrt(len(loss_delta))) if len(loss_delta) > 1 else np.nan
            if se and np.isfinite(se) and se > 1e-12:
                dm_t = float(loss_delta.mean() / se)
                dm_p = float(2 * stats.t.sf(abs(dm_t), df=len(loss_delta) - 1))

        rows.append(
            {
                "candidate": candidate_name,
                "horizon": int(horizon),
                "n": int(len(valid)),
                "ljung_box_p": lb_p,
                "arch_lm_p": arch_p,
                "dm_abs_loss_delta_vs_seasonal": dm_loss_delta,
                "dm_t_vs_seasonal": dm_t,
                "dm_p_vs_seasonal": dm_p,
                "residual_std": float(errors.std(ddof=1)),
            }
        )
    return pd.DataFrame(rows)


def trajectory_diagnostics(data: pd.DataFrame, candidates: list[Candidate]) -> pd.DataFrame:
    cutoffs = pd.date_range(start="2018-12-01", end=data.index.max() - pd.DateOffset(months=12), freq="6MS")
    rows = []
    for candidate in candidates:
        for cutoff in cutoffs:
            train = data[data.index <= cutoff].copy()
            actual_dates = pd.date_range(start=cutoff + pd.DateOffset(months=1), periods=12, freq="MS")
            if actual_dates[-1] > data.index.max() or len(train) < 72:
                continue
            actual = data.loc[actual_dates, TARGET].values.astype(float)
            pred = forecast_candidate(train, candidate, 12)
            if len(pred) < 12 or np.isnan(pred).all():
                continue
            actual_diff_std = np.std(np.diff(actual))
            pred_diff_std = np.std(np.diff(pred))
            vol_ratio = pred_diff_std / actual_diff_std if actual_diff_std > 1e-9 else np.nan
            flatness = float((np.abs(np.diff(pred)) < 0.03).mean())
            path_mae = float(np.mean(np.abs(actual - pred)))
            seasonal = train.groupby(train.index.month)[TARGET].mean()
            seasonal_path = np.asarray([seasonal.get(date.month, np.nan) for date in actual_dates], dtype=float)
            if np.isfinite(seasonal_path).all() and np.std(seasonal_path) > 1e-9 and np.std(pred) > 1e-9:
                seasonal_corr = float(np.corrcoef(pred, seasonal_path)[0, 1])
            else:
                seasonal_corr = np.nan
            rows.append(
                {
                    "candidate": candidate.name,
                    "cutoff": cutoff,
                    "path_mae": path_mae,
                    "vol_ratio": float(vol_ratio) if not np.isnan(vol_ratio) else np.nan,
                    "flatness": flatness,
                    "seasonal_corr": seasonal_corr,
                    "max_abs_jump": float(np.max(np.abs(np.diff(pred)))),
                    "explosive": bool(np.max(np.abs(pred)) > 5.0),
                }
            )
    if not rows:
        return pd.DataFrame()
    diag = pd.DataFrame(rows)
    return diag.groupby("candidate", as_index=False).agg(
        mean_path_mae=("path_mae", "mean"),
        mean_vol_ratio=("vol_ratio", "mean"),
        mean_flatness=("flatness", "mean"),
        mean_seasonal_corr=("seasonal_corr", "mean"),
        max_abs_jump=("max_abs_jump", "max"),
        explosive_rate=("explosive", "mean"),
        n_paths=("candidate", "size"),
    )


def write_report(
    out_dir: Path,
    diagnostics: dict,
    metrics: pd.DataFrame,
    trajectory: pd.DataFrame,
    residuals: pd.DataFrame,
    selected_name: str,
    candidates: list[Candidate],
) -> None:
    selected_cfg = next(candidate for candidate in candidates if candidate.name == selected_name)
    selected_rows = metrics[metrics["candidate"] == selected_name].sort_values("horizon")
    best_by_score = (
        metrics.drop_duplicates("candidate")
        .sort_values("weighted_score")
        .head(12)[["candidate", "kind", "weighted_score", "mae_h1", "mae_h2", "mae_h12"]]
    )
    best_bridge = (
        metrics[metrics["candidate"].str.startswith("BlockBridge")]
        .drop_duplicates("candidate")
        .sort_values("weighted_score")
        .head(1)
    )
    best_baselines = (
        metrics[metrics["kind"] == "baseline"]
        .drop_duplicates("candidate")
        .sort_values("weighted_score")
        .head(5)
    )
    selected_traj = trajectory[trajectory["candidate"] == selected_name] if not trajectory.empty else pd.DataFrame()
    bridge_estimators = sorted(
        {
            candidate.estimator
            for candidate in candidates
            if candidate.kind == "bridge"
        }
    )
    bridge_estimator_text = ", ".join(bridge_estimators) if bridge_estimators else "none"
    implementation = (
        "`sirena.models.factor_policy.FactorPolicyForecaster` and registered under `factor_policy`"
        if selected_cfg.kind != "bridge"
        else "`sirena.models.factor_bridge.FactorBridgeForecaster` and registered under `factor_bridge`"
    )

    lines = [
        "# Factor Model Research Report",
        "",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Goal",
        "",
        "Build a defensible mandatory factor-family model. The goal is not to beat the production ensemble, but to provide a real, leakage-safe factor benchmark/control model with documented limitations.",
        "",
        "## Data Diagnostics",
        "",
        f"- Source: `{diagnostics['source']}`",
        f"- Coverage: {diagnostics['start']} to {diagnostics['end']} ({diagnostics['n_months']} months)",
        "- Scale: CPI/components/USD/deposits/retail are MoM p.p.; rates/index auxiliaries are standardized before factor extraction.",
        f"- CPI shock months with |MoM| > 1 p.p.: {len(diagnostics['shock_months_cpi_abs_gt_1'])}",
        "",
        "## Tested Variants",
        "",
        "- FAVAR: PCA factors from components, macro, lean, and full information sets; factors=1..3; lags=1..2; raw vs train-only seasonal residuals; OLS vs Huber equations.",
        f"- Block factor bridge: train-only component and macro factors; direct horizon equations; factor methods include PCA/average/weighted PCA/sparse PCA; estimators={bridge_estimator_text}; lags=1..3.",
        "- Window robustness: expanding, rolling84, and rolling120 variants where configured.",
        "- Publication-lag proxy: selected macro/financial columns shifted by one month inside the train window.",
        "- Regime proxy: train-only median CPI split with full-sample fallback when the regime subset is too short.",
        "- Simple baselines: seasonal month mean, last-year same month, and seasonal AR(1).",
        "- DFM: statsmodels DynamicFactor control variants on component and lean information sets where the local environment supports it.",
        "- All rolling windows use only data available up to `target_date - horizon`; future macro actuals are never used.",
        "",
        "## Selected Model",
        "",
        f"Selected: `{selected_name}`",
        "",
        f"- Kind: `{selected_cfg.kind}`",
        f"- Factors: {selected_cfg.factors}",
        f"- VAR/factor order: {selected_cfg.lags}",
        f"- Train-only seasonality: {selected_cfg.seasonal}",
        f"- Robust equations: {selected_cfg.robust}",
        f"- Information set: {', '.join(selected_cfg.info_cols)}",
        "",
        f"This is implemented as {implementation}.",
        "",
        "## Selected Metrics",
        "",
            "| Horizon | N | MAE | RMSE | KPI Violations | Coverage <=0.5 | Non-shock MAE | Bias |",
            "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for _, row in selected_rows.iterrows():
        lines.append(
            f"| {int(row['horizon'])} | {int(row['n'])} | {row['mae']:.3f} | {row['rmse']:.3f} | {int(row['kpi_violations'])} | {row['coverage_50pct']:.1f}% | {row['nonshock_mae']:.3f} | {row['bias']:.3f} |"
        )

    lines.extend(
        [
            "",
            "## Top Configurations By Weighted Score",
            "",
            "| Candidate | Kind | Score | MAE h=1 | MAE h=2 | MAE h=12 |",
            "| --- | --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for _, row in best_by_score.iterrows():
        lines.append(
            f"| `{row['candidate']}` | {row['kind']} | {row['weighted_score']:.3f} | {row['mae_h1']:.3f} | {row['mae_h2']:.3f} | {row['mae_h12']:.3f} |"
        )

    lines.extend(["", "## Simple Baseline Checks", ""])
    if best_baselines.empty:
        lines.append("No simple baseline candidate was included in this run.")
    else:
        lines.extend(
            [
                "| Baseline | Score | MAE h=1 | MAE h=2 | MAE h=12 |",
                "| --- | ---: | ---: | ---: | ---: |",
            ]
        )
        for _, row in best_baselines.iterrows():
            lines.append(
                f"| `{row['candidate']}` | {row['weighted_score']:.3f} | {row['mae_h1']:.3f} | {row['mae_h2']:.3f} | {row['mae_h12']:.3f} |"
            )

    lines.extend(["", "## Best Block Bridge Challenger", ""])
    if best_bridge.empty:
        lines.append("No block bridge candidate was included in this run.")
    else:
        row = best_bridge.iloc[0]
        lines.extend(
            [
                f"- Candidate: `{row['candidate']}`",
                f"- Weighted score: {row['weighted_score']:.3f}",
                f"- MAE h=1/h=2/h=12: {row['mae_h1']:.3f} / {row['mae_h2']:.3f} / {row['mae_h12']:.3f}",
                "- Promotion decision: not promoted unless it beats the current factor baseline by at least 1% weighted score without h=1/h=12 degradation.",
            ]
        )

    selected_resid = residuals[residuals["candidate"] == selected_name].sort_values("horizon") if not residuals.empty else pd.DataFrame()
    lines.extend(["", "## Residual And DM-Style Diagnostics", ""])
    if selected_resid.empty:
        lines.append("Residual diagnostics were not available for the selected model.")
    else:
        lines.extend(
            [
                "| Horizon | Ljung-Box p | ARCH-LM p | DM abs-loss delta vs seasonal | DM p vs seasonal |",
                "| ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for _, row in selected_resid.iterrows():
            lines.append(
                f"| {int(row['horizon'])} | {row['ljung_box_p']:.3f} | {row['arch_lm_p']:.3f} | {row['dm_abs_loss_delta_vs_seasonal']:.3f} | {row['dm_p_vs_seasonal']:.3f} |"
            )

    lines.extend(["", "## h=12 Trajectory Diagnostics", ""])
    if selected_traj.empty:
        lines.append("Trajectory diagnostics were not available for the selected model.")
    else:
        row = selected_traj.iloc[0]
        lines.extend(
            [
                f"- Mean path MAE: {row['mean_path_mae']:.3f}",
                f"- Mean volatility ratio: {row['mean_vol_ratio']:.3f}",
                f"- Mean flatness share: {row['mean_flatness']:.3f}",
                f"- Mean seasonal correlation: {row['mean_seasonal_corr']:.3f}",
                f"- Max absolute jump: {row['max_abs_jump']:.3f}",
                f"- Explosive path rate: {row['explosive_rate']:.1%}",
            ]
        )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "The factor model is a real econometric benchmark: latent factors are extracted from observed monthly component and macro series, then forecast jointly with CPI. It is not promoted as a replacement for stronger production ML/subcomponent models unless future validation justifies that. Its role is to satisfy the mandatory factor-model requirement and add an interpretable data-rich control signal.",
            "",
            "## Artifacts",
            "",
            "- `data_diagnostics.json`",
            "- `candidate_configs.json`",
            "- `predictions.csv`",
            "- `metrics.csv`",
            "- `comparison.csv`",
            "- `trajectory_diagnostics.csv`",
            "- `residual_diagnostics.csv`",
            "- `factor_model_report.md`",
        ]
    )
    (out_dir / "factor_model_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", default="factor_policy_rolling")
    parser.add_argument("--no-dfm", action="store_true", help="Skip statsmodels DynamicFactor variants")
    parser.add_argument("--no-quantile-bridge", action="store_true", help="Skip slower QuantileRegressor bridge variants")
    parser.add_argument("--fast", action="store_true", help="Smaller candidate grid for smoke checks")
    parser.add_argument("--proposal-sweep", action="store_true", help="Compact sweep covering every agent-proposed variant family")
    args = parser.parse_args()

    run_dir = ROOT / "experiments" / "factor_model_research" / "runs" / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    data = load_official()
    diagnostics = save_diagnostics(data, run_dir)
    candidates = build_candidates(
        include_dfm=not args.no_dfm,
        include_quantile_bridge=not args.no_quantile_bridge,
    )
    if args.fast:
        candidates = [
            c
            for c in candidates
            if c.name
            in {
                "RobustFAVAR_lean_f2_l1_seasonal",
                "FAVAR_lean_f2_l1_seasonal_ols",
                "RobustFAVAR_components_f2_l1_seasonal",
                "RobustFAVAR_all_f2_l1_seasonal",
                "DFM_components_f1_q1_seasonal",
                "BlockBridge_pca_l1_huber_seasonal",
                "BlockBridge_pca_l2_huber_seasonal",
                "BlockBridge_average_l2_ridge_seasonal",
                "SeasonalNaive_mean",
                "SeasonalAR1",
                "BlockBridge_weighted_pca_l1_huber_seasonal",
                "BlockBridge_sparse_pca_l1_huber_seasonal",
                "BlockBridge_pca_l1_quantile_seasonal",
            }
        ]
    if args.proposal_sweep:
        keep = {
            "SeasonalNaive_mean",
            "SeasonalNaive_lastyear",
            "SeasonalAR1",
            "RobustFAVAR_lean_f2_l1_seasonal",
            "RobustFAVAR_lean_f2_l1_seasonal_rolling84",
            "RobustFAVAR_lean_f2_l1_seasonal_rolling120",
            "RobustFAVAR_lean_f2_l1_seasonal_pubLag1",
            "DFM_components_f1_q1_seasonal",
            "DFM_lean_f1_q1_seasonal",
            "BlockBridge_pca_l1_huber_seasonal",
            "BlockBridge_pca_l2_huber_seasonal",
            "BlockBridge_average_l1_huber_seasonal",
            "BlockBridge_weighted_pca_l1_huber_seasonal",
            "BlockBridge_sparse_pca_l1_huber_seasonal",
            "BlockBridge_pca_l1_ridge_seasonal",
            "BlockBridge_pca_l1_quantile_seasonal",
            "BlockBridge_pca_l1_huber_seasonal_rolling84",
            "BlockBridge_pca_l1_huber_seasonal_rolling120",
            "BlockBridge_pca_l1_huber_seasonal_pubLag1",
            "BlockBridge_pca_l1_huber_regimeMedian",
        }
        candidates = [candidate for candidate in candidates if candidate.name in keep]
    (run_dir / "candidate_configs.json").write_text(
        json.dumps([asdict(candidate) for candidate in candidates], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    predictions = evaluate_candidates(data, candidates, horizons=(1, 2, 12))
    predictions.to_csv(run_dir / "predictions.csv", index=False)

    metrics = calculate_metrics(predictions, candidates)
    metrics.to_csv(run_dir / "metrics.csv", index=False)

    residuals = residual_diagnostics(predictions)
    residuals.to_csv(run_dir / "residual_diagnostics.csv", index=False)

    comparison = metrics.drop_duplicates("candidate").sort_values("weighted_score")
    comparison.to_csv(run_dir / "comparison.csv", index=False)

    baseline_name = "RobustFAVAR_lean_f2_l1_seasonal"
    top_for_trajectory = comparison.head(12)["candidate"].tolist()
    if baseline_name in {candidate.name for candidate in candidates} and baseline_name not in top_for_trajectory:
        top_for_trajectory.append(baseline_name)
    candidate_map = {candidate.name: candidate for candidate in candidates}
    trajectory = trajectory_diagnostics(data, [candidate_map[name] for name in top_for_trajectory])
    trajectory.to_csv(run_dir / "trajectory_diagnostics.csv", index=False)

    # Promotion rule from the agent review: the bridge may replace the current
    # factor baseline only with a real weighted-score gain and no h=1/h=12
    # degradation. Otherwise keep the existing robust seasonal FAVAR.
    best_row = comparison.iloc[0]
    baseline_rows = comparison[comparison["candidate"] == baseline_name]
    selected = str(best_row["candidate"])
    if not baseline_rows.empty:
        baseline = baseline_rows.iloc[0]
        best_is_bridge = str(best_row["candidate"]).startswith("BlockBridge")
        weighted_gain = float(best_row["weighted_score"]) <= float(baseline["weighted_score"]) * 0.99
        h1_ok = float(best_row["mae_h1"]) <= float(baseline["mae_h1"]) * 1.02
        h12_ok = float(best_row["mae_h12"]) <= float(baseline["mae_h12"]) * 1.02
        selected = str(best_row["candidate"]) if best_is_bridge and weighted_gain and h1_ok and h12_ok else baseline_name

    selected_traj = trajectory[trajectory["candidate"] == selected] if not trajectory.empty else pd.DataFrame()
    if not selected_traj.empty and float(selected_traj.iloc[0]["explosive_rate"]) > 0:
        selected = baseline_name if not baseline_rows.empty else str(best_row["candidate"])

    write_report(run_dir, diagnostics, metrics, trajectory, residuals, selected, candidates)
    print(f"Run saved: {run_dir}")
    print(f"Selected: {selected}")
    print(comparison.head(10)[["candidate", "kind", "weighted_score", "mae_h1", "mae_h2", "mae_h12"]].to_string(index=False))


if __name__ == "__main__":
    main()
