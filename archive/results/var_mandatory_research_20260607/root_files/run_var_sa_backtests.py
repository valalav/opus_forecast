#!/usr/bin/env python3
"""
Reproducible VAR/BVAR/SA research backtests for experiments/var_sa_research.

This runner intentionally does not modify production data or model registry.
It follows the archived h=1/h=2/h=12 backtest date windows when available so
research metrics are directly comparable with archive/results/backtest_h*.csv.
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import pandas as pd
from statsmodels.tsa.api import VAR

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from sirena.models.bvar import BVARForecaster  # noqa: E402

warnings.filterwarnings("ignore")


RESEARCH_DIR = ROOT / "experiments" / "var_sa_research"
RUNS_DIR = RESEARCH_DIR / "runs"
ARCHIVE_RESULTS = ROOT / "archive" / "results"

OFFICIAL_COMPONENT_WEIGHTS = {
    "Food": 0.3986,
    "NonFood": 0.3638,
    "Services": 0.2376,
}


@dataclass(frozen=True)
class Candidate:
    name: str
    family: str
    data_source: str
    variables: tuple[str, ...]
    params: dict
    forecast_fn: Callable[[pd.DataFrame, int, "Candidate", int], float]
    notes: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-name",
        default=datetime.now().strftime("%Y%m%d_%H%M%S"),
        help="Subdirectory name under experiments/var_sa_research/runs/",
    )
    parser.add_argument(
        "--preset",
        choices=["baseline", "grid", "sa", "seasonal", "seasonal_fine", "all"],
        default="all",
        help="Candidate set to evaluate.",
    )
    parser.add_argument(
        "--n-draws",
        type=int,
        default=160,
        help="BVAR Monte Carlo draws per rolling forecast.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=20260607,
        help="Base random seed for stochastic BVAR forecasts.",
    )
    return parser.parse_args()


def load_official_data() -> pd.DataFrame:
    df = pd.read_csv(ROOT / "data" / "inflation_data.csv", sep=";", decimal=",")
    df["Date"] = pd.to_datetime(df["Date"], format="%d.%m.%Y", errors="coerce")
    df["Date"] = df["Date"].dt.to_period("M").dt.to_timestamp()
    df = df.set_index("Date").sort_index()

    out = pd.DataFrame(index=df.index)
    out["CPI"] = pd.to_numeric(df["mom"], errors="coerce") - 100
    out["Food"] = pd.to_numeric(df["Prod"], errors="coerce") - 100
    out["NonFood"] = pd.to_numeric(df["Nonprod"], errors="coerce") - 100
    out["Services"] = pd.to_numeric(df["Serv"], errors="coerce") - 100
    out["USD"] = pd.to_numeric(df["usd_nom_i"], errors="coerce") - 100
    out["Ki_i"] = pd.to_numeric(df["Ki_i"], errors="coerce")
    out["Ruonia"] = pd.to_numeric(df["Ruonia"], errors="coerce")
    return out.dropna(subset=["CPI"])


def load_sa_mom_data() -> pd.DataFrame:
    raw = pd.read_csv(ROOT / "data" / "mom_sa_kbr.csv", sep=";", decimal=",", encoding="utf-8-sig")
    date_cols = [c for c in raw.columns if isinstance(c, str) and c[:4].isdigit()]
    long = raw.melt(id_vars=["Товар"], value_vars=date_cols, var_name="Date", value_name="value")
    long["Date"] = pd.to_datetime(long["Date"], format="%Y-%m").dt.to_period("M").dt.to_timestamp()
    pivot = long.pivot_table(index="Date", columns="Товар", values="value", aggfunc="first").sort_index()

    rename = {
        "Все товары и услуги": "CPI",
        "Продовольственные товары": "Food",
        "Непродовольственные товары": "NonFood",
        "Услуги": "Services",
    }
    out = pivot.rename(columns=rename)
    keep = [c for c in ["CPI", "Food", "NonFood", "Services"] if c in out.columns]
    out = out[keep].apply(pd.to_numeric, errors="coerce") - 100
    return out


def archived_test_dates(horizon: int, official: pd.DataFrame) -> pd.DatetimeIndex:
    path = ARCHIVE_RESULTS / f"backtest_h{horizon}_predictions.csv"
    if path.exists():
        dates = pd.to_datetime(pd.read_csv(path)["Date"]).dt.to_period("M").dt.to_timestamp()
        return pd.DatetimeIndex(dates)

    last_fact = official.dropna(subset=["CPI"]).index.max()
    if horizon == 12:
        cutoff = last_fact - pd.DateOffset(months=12)
        return pd.date_range(cutoff + pd.DateOffset(months=1), periods=12, freq="MS")
    return pd.date_range(end=last_fact, periods=12, freq="MS")


def forecast_var(train: pd.DataFrame, horizon: int, candidate: Candidate, seed: int) -> float:
    del seed
    cols = list(candidate.variables)
    data = train.loc[:, cols].dropna()
    lags = int(candidate.params["lags"])
    if len(data) <= max(24, len(cols) * len(cols) * lags + len(cols) + 1):
        return np.nan

    fit = VAR(data).fit(lags)
    fc = fit.forecast(y=data.values[-fit.k_ar :], steps=horizon)
    if candidate.params.get("target") == "bottom_up":
        weights = OFFICIAL_COMPONENT_WEIGHTS
        pred_row = dict(zip(cols, fc[horizon - 1]))
        return sum(pred_row[name] * weight for name, weight in weights.items())
    return float(fc[horizon - 1, cols.index("CPI")])


def forecast_bvar(train: pd.DataFrame, horizon: int, candidate: Candidate, seed: int) -> float:
    np.random.seed(seed)
    cols = list(candidate.variables)
    data = train.loc[:, cols].dropna()
    min_train = int(candidate.params.get("min_train", 30))
    if len(data) < min_train:
        return np.nan

    model = BVARForecaster(
        lags=int(candidate.params.get("lags", 1)),
        lambda1=float(candidate.params.get("lambda1", 0.2)),
        lambda2=float(candidate.params.get("lambda2", 0.5)),
        lambda3=float(candidate.params.get("lambda3", 1.0)),
        lambda4=float(candidate.params.get("lambda4", 100.0)),
        n_draws=int(candidate.params.get("n_draws", 160)),
        sigma_prior_scale=float(candidate.params.get("sigma_prior_scale", 1.0)),
        auto_lambda=bool(candidate.params.get("auto_lambda", False)),
        auto_lags=bool(candidate.params.get("auto_lags", False)),
        max_lags=int(candidate.params.get("max_lags", 4)),
        var_names=cols,
    )
    model.fit(data, target_col=cols[0])
    fc = model.forecast_full(horizon=horizon)["median"]
    if candidate.params.get("target") == "bottom_up":
        weights = OFFICIAL_COMPONENT_WEIGHTS
        pred_row = dict(zip(cols, fc[horizon - 1]))
        return sum(pred_row[name] * weight for name, weight in weights.items())
    return float(fc[horizon - 1, 0])


def _seasonal_means(train: pd.DataFrame, variables: list[str], window_months: int | None) -> pd.DataFrame:
    data = train.loc[:, variables].dropna()
    if window_months is not None and len(data) > window_months:
        data = data.iloc[-window_months:]
    return data.groupby(data.index.month).mean()


def _seasonal_residual_frame(
    train: pd.DataFrame,
    variables: list[str],
    month_means: pd.DataFrame,
) -> pd.DataFrame:
    data = train.loc[:, variables].dropna().copy()
    for month in range(1, 13):
        mask = data.index.month == month
        if month in month_means.index:
            data.loc[mask, variables] = data.loc[mask, variables] - month_means.loc[month, variables].values
    return data


def _future_seasonal(month_means: pd.DataFrame, variables: list[str], target_month: int) -> np.ndarray:
    if target_month in month_means.index:
        return month_means.loc[target_month, variables].values.astype(float)
    return month_means.loc[:, variables].mean().values.astype(float)


def forecast_seasonal_var(train: pd.DataFrame, horizon: int, candidate: Candidate, seed: int) -> float:
    del seed
    cols = list(candidate.variables)
    window_months = candidate.params.get("seasonality_window_months")
    month_means = _seasonal_means(train, cols, window_months)
    resid_train = _seasonal_residual_frame(train, cols, month_means)
    if len(resid_train) < max(24, len(cols) * int(candidate.params["lags"]) + 12):
        return np.nan

    residual_candidate = Candidate(
        name=candidate.name + "_residual_core",
        family=candidate.family,
        data_source=candidate.data_source,
        variables=candidate.variables,
        params={**candidate.params, "target": candidate.params.get("target", "CPI")},
        forecast_fn=forecast_var,
        notes=candidate.notes,
    )
    residual_prediction = forecast_var(resid_train, horizon, residual_candidate, 0)
    target_date = train.index.max() + pd.DateOffset(months=horizon)

    if candidate.params.get("target") == "bottom_up":
        seasonal = _future_seasonal(month_means, cols, target_date.month)
        seasonal_by_col = dict(zip(cols, seasonal))
        seasonal_total = sum(seasonal_by_col[name] * weight for name, weight in OFFICIAL_COMPONENT_WEIGHTS.items())
        return residual_prediction + seasonal_total

    seasonal_cpi = float(_future_seasonal(month_means, cols, target_date.month)[cols.index("CPI")])
    return residual_prediction + seasonal_cpi


def forecast_seasonal_bvar(train: pd.DataFrame, horizon: int, candidate: Candidate, seed: int) -> float:
    cols = list(candidate.variables)
    window_months = candidate.params.get("seasonality_window_months")
    month_means = _seasonal_means(train, cols, window_months)
    resid_train = _seasonal_residual_frame(train, cols, month_means)
    if len(resid_train) < int(candidate.params.get("min_train", 30)):
        return np.nan

    residual_candidate = Candidate(
        name=candidate.name + "_residual_core",
        family=candidate.family,
        data_source=candidate.data_source,
        variables=candidate.variables,
        params={**candidate.params, "target": candidate.params.get("target", "CPI")},
        forecast_fn=forecast_bvar,
        notes=candidate.notes,
    )
    residual_prediction = forecast_bvar(resid_train, horizon, residual_candidate, seed)
    target_date = train.index.max() + pd.DateOffset(months=horizon)

    if candidate.params.get("target") == "bottom_up":
        seasonal = _future_seasonal(month_means, cols, target_date.month)
        seasonal_by_col = dict(zip(cols, seasonal))
        seasonal_total = sum(seasonal_by_col[name] * weight for name, weight in OFFICIAL_COMPONENT_WEIGHTS.items())
        return residual_prediction + seasonal_total

    seasonal_cpi = float(_future_seasonal(month_means, cols, target_date.month)[cols.index("CPI")])
    return residual_prediction + seasonal_cpi


def build_candidates(preset: str, n_draws: int) -> list[Candidate]:
    candidates: list[Candidate] = []

    def add_var(name: str, source: str, variables: Iterable[str], lags: int, target: str = "CPI", notes: str = ""):
        candidates.append(
            Candidate(
                name=name,
                family="VAR",
                data_source=source,
                variables=tuple(variables),
                params={"lags": lags, "target": target},
                forecast_fn=forecast_var,
                notes=notes,
            )
        )

    def add_bvar(
        name: str,
        source: str,
        variables: Iterable[str],
        lags: int,
        lambda1: float,
        lambda2: float = 0.5,
        lambda3: float = 1.0,
        target: str = "CPI",
        notes: str = "",
        **extra,
    ):
        params = {
            "lags": lags,
            "lambda1": lambda1,
            "lambda2": lambda2,
            "lambda3": lambda3,
            "target": target,
            "n_draws": n_draws,
            **extra,
        }
        candidates.append(
            Candidate(
                name=name,
                family="BVAR",
                data_source=source,
                variables=tuple(variables),
                params=params,
                forecast_fn=forecast_bvar,
                notes=notes,
            )
        )

    if preset in {"baseline", "all"}:
        for lags in [1, 2, 3]:
            add_var(f"var_official_total_components_l{lags}", "official", ["CPI", "Food", "NonFood", "Services"], lags)
        add_var(
            "var_official_components_bottomup_l1",
            "official",
            ["Food", "NonFood", "Services"],
            1,
            target="bottom_up",
            notes="Component-only VAR with weighted bottom-up aggregation.",
        )
        add_bvar(
            "bvar_archive_style_l4_lam1_total_food_usd_ruonia",
            "official",
            ["CPI", "Food", "USD", "Ruonia"],
            4,
            1.0,
            lambda2=0.5,
            lambda3=1.0,
            notes="Closest to scripts/backtest_framework.py h=1/h=2 BVAR settings.",
        )
        add_bvar(
            "bvar_default_total_components_l1_lam02",
            "official",
            ["CPI", "Food", "NonFood", "Services"],
            1,
            0.2,
            notes="Current BVARForecaster default Minnesota tightness on total plus components.",
        )

    if preset in {"grid", "all"}:
        for variables_name, variables in {
            "tc": ["CPI", "Food", "NonFood", "Services"],
            "tfu": ["CPI", "Food", "USD", "Ruonia"],
            "comp": ["Food", "NonFood", "Services"],
        }.items():
            for lags in [1, 2]:
                for lambda1 in [0.05, 0.1, 0.2, 0.5, 1.0]:
                    target = "bottom_up" if variables_name == "comp" else "CPI"
                    add_bvar(
                        f"bvar_grid_{variables_name}_l{lags}_lam{str(lambda1).replace('.', 'p')}",
                        "official",
                        variables,
                        lags,
                        lambda1,
                        lambda2=0.5,
                        lambda3=1.0,
                        target=target,
                        notes="Quick shrinkage/lag grid.",
                    )
        add_bvar(
            "bvar_auto_lambda_total_components_l1",
            "official",
            ["CPI", "Food", "NonFood", "Services"],
            1,
            0.2,
            auto_lambda=True,
            notes="BVARForecaster empirical-Bayes lambda1 selection.",
        )
        add_bvar(
            "bvar_auto_lags_total_components",
            "official",
            ["CPI", "Food", "NonFood", "Services"],
            1,
            0.2,
            auto_lags=True,
            max_lags=3,
            notes="BVARForecaster BIC lag selection.",
        )

    if preset in {"sa", "all"}:
        for lags in [1, 2]:
            add_var(
                f"var_sa_total_components_l{lags}",
                "sa_revised",
                ["CPI", "Food", "NonFood", "Services"],
                lags,
                notes="Uses full-history revised SA data; compared against official non-SA MoM.",
            )
            add_var(
                f"var_sa_components_bottomup_l{lags}",
                "sa_revised",
                ["Food", "NonFood", "Services"],
                lags,
                target="bottom_up",
                notes="Bottom-up SA component VAR with fixed CPI component weights.",
            )
        for lambda1 in [0.1, 0.2, 0.5]:
            add_bvar(
                f"bvar_sa_total_components_l1_lam{str(lambda1).replace('.', 'p')}",
                "sa_revised",
                ["CPI", "Food", "NonFood", "Services"],
                1,
                lambda1,
                notes="BVAR on revised SA total plus components; compared against official non-SA MoM.",
            )
            add_bvar(
                f"bvar_sa_components_bottomup_l1_lam{str(lambda1).replace('.', 'p')}",
                "sa_revised",
                ["Food", "NonFood", "Services"],
                1,
                lambda1,
                target="bottom_up",
                notes="Bottom-up BVAR on revised SA components with fixed CPI component weights.",
            )

    if preset in {"seasonal", "all"}:
        for window_name, window_months in {"expanding": None, "roll60": 60, "roll36": 36}.items():
            for lags in [1, 2, 3]:
                candidates.append(
                    Candidate(
                        name=f"seasonal_resid_var_tc_{window_name}_l{lags}",
                        family="VAR_seasonal_residual",
                        data_source="official",
                        variables=("CPI", "Food", "NonFood", "Services"),
                        params={
                            "lags": lags,
                            "target": "CPI",
                            "seasonality_window_months": window_months,
                        },
                        forecast_fn=forecast_seasonal_var,
                        notes="Forecasts residual after train-only month-of-year mean, then adds target-month seasonal mean back.",
                    )
                )
            for lags in [1, 2]:
                candidates.append(
                    Candidate(
                        name=f"seasonal_resid_var_comp_bottomup_{window_name}_l{lags}",
                        family="VAR_seasonal_residual",
                        data_source="official",
                        variables=("Food", "NonFood", "Services"),
                        params={
                            "lags": lags,
                            "target": "bottom_up",
                            "seasonality_window_months": window_months,
                        },
                        forecast_fn=forecast_seasonal_var,
                        notes="Component residual VAR with train-only seasonal means and weighted bottom-up reconstruction.",
                    )
                )
            for lambda1 in [0.1, 0.2, 0.5, 1.0]:
                candidates.append(
                    Candidate(
                        name=f"seasonal_resid_bvar_tc_{window_name}_l1_lam{str(lambda1).replace('.', 'p')}",
                        family="BVAR_seasonal_residual",
                        data_source="official",
                        variables=("CPI", "Food", "NonFood", "Services"),
                        params={
                            "lags": 1,
                            "lambda1": lambda1,
                            "lambda2": 0.5,
                            "lambda3": 1.0,
                            "target": "CPI",
                            "n_draws": n_draws,
                            "seasonality_window_months": window_months,
                        },
                        forecast_fn=forecast_seasonal_bvar,
                        notes="BVAR on train-only seasonal residuals with seasonal reconstruction.",
                    )
                )

    if preset in {"seasonal_fine"}:
        fine_windows = {
            "roll24": 24,
            "roll30": 30,
            "roll36": 36,
            "roll42": 42,
            "roll48": 48,
            "roll60": 60,
            "roll72": 72,
        }
        for window_name, window_months in fine_windows.items():
            for lags in [2, 3, 4, 5, 6]:
                candidates.append(
                    Candidate(
                        name=f"fine_seasonal_resid_var_tc_{window_name}_l{lags}",
                        family="VAR_seasonal_residual",
                        data_source="official",
                        variables=("CPI", "Food", "NonFood", "Services"),
                        params={
                            "lags": lags,
                            "target": "CPI",
                            "seasonality_window_months": window_months,
                        },
                        forecast_fn=forecast_seasonal_var,
                        notes="Fine grid around best iteration-2 total/component seasonal residual VAR.",
                    )
                )
            for lags in [1, 2, 3, 4]:
                candidates.append(
                    Candidate(
                        name=f"fine_seasonal_resid_var_comp_bottomup_{window_name}_l{lags}",
                        family="VAR_seasonal_residual",
                        data_source="official",
                        variables=("Food", "NonFood", "Services"),
                        params={
                            "lags": lags,
                            "target": "bottom_up",
                            "seasonality_window_months": window_months,
                        },
                        forecast_fn=forecast_seasonal_var,
                        notes="Fine grid around best iteration-2 component bottom-up seasonal residual VAR.",
                    )
                )
        for window_name, window_months in {"roll24": 24, "roll30": 30, "roll36": 36, "roll42": 42, "roll48": 48}.items():
            for lags in [1, 2]:
                for lambda1 in [0.05, 0.1, 0.2, 0.3, 0.5]:
                    candidates.append(
                        Candidate(
                            name=(
                                f"fine_seasonal_resid_bvar_tc_{window_name}_"
                                f"l{lags}_lam{str(lambda1).replace('.', 'p')}"
                            ),
                            family="BVAR_seasonal_residual",
                            data_source="official",
                            variables=("CPI", "Food", "NonFood", "Services"),
                            params={
                                "lags": lags,
                                "lambda1": lambda1,
                                "lambda2": 0.5,
                                "lambda3": 1.0,
                                "target": "CPI",
                                "n_draws": n_draws,
                                "seasonality_window_months": window_months,
                            },
                            forecast_fn=forecast_seasonal_bvar,
                            notes="Fine BVAR grid on train-only seasonal residuals.",
                        )
                    )

    # Keep names unique if presets overlap archive-style/default with grid variants.
    seen = set()
    unique = []
    for candidate in candidates:
        if candidate.name in seen:
            continue
        seen.add(candidate.name)
        unique.append(candidate)
    return unique


def evaluate_candidates(
    candidates: list[Candidate],
    official: pd.DataFrame,
    sa: pd.DataFrame,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    prediction_rows = []
    all_sources = {"official": official, "sa_revised": sa}

    for horizon in [1, 2, 12]:
        test_dates = archived_test_dates(horizon, official)
        for target_date in test_dates:
            if target_date not in official.index:
                continue
            actual = float(official.loc[target_date, "CPI"])
            cutoff = target_date - pd.DateOffset(months=horizon)

            for candidate_index, candidate in enumerate(candidates):
                source_df = all_sources[candidate.data_source]
                train = source_df[source_df.index <= cutoff].copy()
                run_seed = seed + candidate_index * 1000 + horizon * 100 + int(target_date.strftime("%Y%m"))
                try:
                    prediction = candidate.forecast_fn(train, horizon, candidate, run_seed)
                    error = actual - prediction if np.isfinite(prediction) else np.nan
                    status = "ok" if np.isfinite(prediction) else "nan_prediction"
                except Exception as exc:
                    prediction = np.nan
                    error = np.nan
                    status = f"error: {type(exc).__name__}: {exc}"

                prediction_rows.append(
                    {
                        "horizon": horizon,
                        "candidate": candidate.name,
                        "family": candidate.family,
                        "data_source": candidate.data_source,
                        "target_date": target_date,
                        "cutoff": cutoff,
                        "train_start": train.index.min() if len(train) else pd.NaT,
                        "train_end": train.index.max() if len(train) else pd.NaT,
                        "train_n": len(train.dropna(subset=list(candidate.variables))),
                        "actual": actual,
                        "prediction": prediction,
                        "error": error,
                        "abs_error": abs(error) if np.isfinite(error) else np.nan,
                        "status": status,
                        "variables": ",".join(candidate.variables),
                        "params": json.dumps(candidate.params, ensure_ascii=False, sort_keys=True),
                        "notes": candidate.notes,
                    }
                )

    predictions = pd.DataFrame(prediction_rows)
    metrics = calculate_metrics(predictions)
    return predictions, metrics


def calculate_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    grouped = predictions.dropna(subset=["error"]).groupby(["candidate", "horizon"], sort=False)
    for (candidate, horizon), group in grouped:
        errors = group["error"].astype(float)
        abs_errors = errors.abs()
        rows.append(
            {
                "candidate": candidate,
                "family": group["family"].iloc[0],
                "data_source": group["data_source"].iloc[0],
                "horizon": horizon,
                "n": len(group),
                "MAE": abs_errors.mean(),
                "RMSE": np.sqrt((errors**2).mean()),
                "Mean_Error": errors.mean(),
                "Max_Error": abs_errors.max(),
                "KPI_Violations": int((abs_errors > 0.5).sum()),
                "Coverage_50pct": float((abs_errors <= 0.5).mean() * 100),
            }
        )
    metrics = pd.DataFrame(rows)
    if metrics.empty:
        return metrics
    return metrics.sort_values(["horizon", "MAE", "candidate"]).reset_index(drop=True)


def load_archived_comparison() -> pd.DataFrame:
    wanted = {
        "Huber",
        "Ridge_Shock",
        "ElasticNet",
        "Subcomp_Multi",
        "BVAR",
        "Ridge_ProdProxy_Roll24",
    }
    rows = []
    for horizon in [1, 2, 12]:
        path = ARCHIVE_RESULTS / f"backtest_h{horizon}_metrics.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path)
        for _, row in df.iterrows():
            if row["Model"] in wanted:
                rows.append(
                    {
                        "candidate": f"archived_{row['Model']}",
                        "horizon": horizon,
                        "n": np.nan,
                        "MAE": row["MAE"],
                        "RMSE": row["RMSE"],
                        "Mean_Error": row.get("Mean_Error", np.nan),
                        "Max_Error": row.get("Max_Error", np.nan),
                        "KPI_Violations": row.get("KPI_Violations", np.nan),
                        "Coverage_50pct": row.get("Coverage_50pct", np.nan),
                    }
                )
    return pd.DataFrame(rows)


def make_comparison(metrics: pd.DataFrame) -> pd.DataFrame:
    archived = load_archived_comparison()
    comparison = pd.concat([metrics, archived], ignore_index=True)
    if comparison.empty:
        return comparison

    wide = comparison.pivot_table(index="candidate", columns="horizon", values="MAE", aggfunc="first")
    for horizon in [1, 2, 12]:
        if horizon not in wide.columns:
            wide[horizon] = np.nan
    wide["SIRENA_Score"] = 0.50 * wide[1] + 0.30 * wide[2] + 0.20 * wide[12]

    meta_rows = []
    for candidate, group in comparison.groupby("candidate", sort=False):
        family = group["family"].dropna().iloc[0] if "family" in group and group["family"].notna().any() else "archived"
        data_source = (
            group["data_source"].dropna().iloc[0]
            if "data_source" in group and group["data_source"].notna().any()
            else "archive/results"
        )
        meta_rows.append({"candidate": candidate, "family": family, "data_source": data_source})
    first_rows = pd.DataFrame(meta_rows)
    out = wide.rename(columns={1: "MAE_h1", 2: "MAE_h2", 12: "MAE_h12"}).reset_index()
    out = out.merge(first_rows, on="candidate", how="left")
    cols = ["candidate", "family", "data_source", "MAE_h1", "MAE_h2", "MAE_h12", "SIRENA_Score"]
    return out[cols].sort_values(["MAE_h1", "SIRENA_Score"], na_position="last")


def write_notes(
    out_dir: Path,
    args: argparse.Namespace,
    candidates: list[Candidate],
    official: pd.DataFrame,
    sa: pd.DataFrame,
    metrics: pd.DataFrame,
    comparison: pd.DataFrame,
) -> None:
    best_h1 = metrics[metrics["horizon"] == 1].sort_values("MAE").head(5)
    best_all = comparison.dropna(subset=["MAE_h1"]).head(10)
    lines = [
        "# VAR/SA Research Run Notes",
        "",
        f"- Run directory: `{out_dir}`",
        f"- Preset: `{args.preset}`",
        f"- BVAR draws: `{args.n_draws}`",
        f"- Seed: `{args.seed}`",
        f"- Official data range: `{official.index.min().date()}` to `{official.index.max().date()}`",
        f"- SA data range: `{sa.index.min().date()}` to `{sa.index.max().date()}`",
        f"- Candidates evaluated: `{len(candidates)}`",
        "",
        "## Method",
        "",
        "- h=1/h=2 use archived test dates from `archive/results/backtest_h*_predictions.csv` when present.",
        "- h=12 uses the archived 12-month trajectory when present.",
        "- Each target trains only on rows with date `<= target_date - horizon months`.",
        "- SA files are full-history revised estimates, not real-time vintages; SA results have revision leakage risk and are not production evidence.",
        "",
        "## Best h=1 Candidates",
        "",
        best_h1.to_markdown(index=False) if len(best_h1) else "_No successful h=1 predictions._",
        "",
        "## Best Overall Comparison Rows",
        "",
        best_all.to_markdown(index=False) if len(best_all) else "_No comparison rows._",
        "",
        "## Candidate Notes",
        "",
    ]
    for candidate in candidates:
        lines.append(f"- `{candidate.name}`: {candidate.notes or 'No extra notes.'}")
    (out_dir / "notes.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def save_config(out_dir: Path, args: argparse.Namespace, candidates: list[Candidate]) -> None:
    config = {
        "run_name": args.run_name,
        "preset": args.preset,
        "n_draws": args.n_draws,
        "seed": args.seed,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "candidate_count": len(candidates),
        "candidates": [
            {
                "name": c.name,
                "family": c.family,
                "data_source": c.data_source,
                "variables": c.variables,
                "params": c.params,
                "notes": c.notes,
            }
            for c in candidates
        ],
    }
    (out_dir / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> int:
    args = parse_args()
    out_dir = RUNS_DIR / args.run_name
    out_dir.mkdir(parents=True, exist_ok=False)

    official = load_official_data()
    sa = load_sa_mom_data()
    candidates = build_candidates(args.preset, args.n_draws)

    save_config(out_dir, args, candidates)
    predictions, metrics = evaluate_candidates(candidates, official, sa, args.seed)
    comparison = make_comparison(metrics)

    predictions.to_csv(out_dir / "predictions.csv", index=False)
    metrics.to_csv(out_dir / "metrics.csv", index=False)
    comparison.to_csv(out_dir / "comparison.csv", index=False)
    (out_dir / "run_var_sa_backtests.py").write_text(Path(__file__).read_text(encoding="utf-8"), encoding="utf-8")
    write_notes(out_dir, args, candidates, official, sa, metrics, comparison)

    print(f"Saved run artifacts to {out_dir}")
    if not comparison.empty:
        print(comparison.head(12).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
