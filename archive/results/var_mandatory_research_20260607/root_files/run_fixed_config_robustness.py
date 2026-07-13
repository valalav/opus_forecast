#!/usr/bin/env python3
"""
Robustness checks for the frozen fine_seasonal_resid_var_tc_roll42_l5 model.

This script intentionally fixes the selected VAR configuration and evaluates it
outside the 2025-04..2026-03 selection window. It also runs comparable h=1
baselines on historical windows and emits leakage diagnostics for the seasonal
residualization step.
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from experiments.var_sa_research.run_var_sa_backtests import (  # noqa: E402
    Candidate,
    forecast_seasonal_var,
    load_official_data,
)
from sirena.models.bvar import BVARForecaster  # noqa: E402
from sirena.models.huber import HuberForecaster  # noqa: E402
from sirena.models.ridge_production_proxy_rolling import RidgeProductionProxyRollingForecaster  # noqa: E402
from sirena.models.ridge_shock_dummies import RidgeShockDummiesForecaster  # noqa: E402
from sirena.models.subcomponent_multi import SubcomponentMultiForecaster  # noqa: E402

warnings.filterwarnings("ignore")


RESEARCH_DIR = ROOT / "experiments" / "var_sa_research"
RUNS_DIR = RESEARCH_DIR / "runs"
SELECTION_START = pd.Timestamp("2025-04-01")
SELECTION_END = pd.Timestamp("2026-03-01")

FIXED_VAR_CONFIG = {
    "name": "fine_seasonal_resid_var_tc_roll42_l5",
    "family": "VAR_seasonal_residual",
    "data_source": "official",
    "variables": ("CPI", "Food", "NonFood", "Services"),
    "lags": 5,
    "seasonality_window_months": 42,
    "target": "CPI",
}


@dataclass(frozen=True)
class BaselineSpec:
    name: str
    forecast_fn: Callable[[pd.DataFrame, pd.DataFrame, pd.Timestamp, int], tuple[float, str]]
    notes: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", default="robustness_fixed_roll42_l5")
    parser.add_argument("--seed", type=int, default=20260607)
    parser.add_argument(
        "--skip-subcomponent",
        action="store_true",
        help="Skip SubcomponentMulti if a fast diagnostic run is needed.",
    )
    parser.add_argument(
        "--profile",
        choices=["full", "representative_months"],
        default="full",
        help="Historical target profile. representative_months is for compact SubcomponentMulti checks.",
    )
    return parser.parse_args()


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


def fixed_var_candidate() -> Candidate:
    return Candidate(
        name=FIXED_VAR_CONFIG["name"],
        family=FIXED_VAR_CONFIG["family"],
        data_source=FIXED_VAR_CONFIG["data_source"],
        variables=tuple(FIXED_VAR_CONFIG["variables"]),
        params={
            "lags": FIXED_VAR_CONFIG["lags"],
            "seasonality_window_months": FIXED_VAR_CONFIG["seasonality_window_months"],
            "target": FIXED_VAR_CONFIG["target"],
        },
        forecast_fn=forecast_seasonal_var,
        notes="Frozen config from iter03; no retuning in robustness run.",
    )


def historical_windows(profile: str = "full") -> list[tuple[str, pd.Timestamp, pd.Timestamp]]:
    if profile == "representative_months":
        return [
            ("rep_pre_covid_2019_12", pd.Timestamp("2019-12-01"), pd.Timestamp("2019-12-01")),
            ("rep_covid_2021_12", pd.Timestamp("2021-12-01"), pd.Timestamp("2021-12-01")),
            ("rep_sanctions_2022_12", pd.Timestamp("2022-12-01"), pd.Timestamp("2022-12-01")),
            ("rep_tightening_2023_12", pd.Timestamp("2023-12-01"), pd.Timestamp("2023-12-01")),
            ("rep_pre_selection_2025_03", pd.Timestamp("2025-03-01"), pd.Timestamp("2025-03-01")),
        ]
    return [
        ("pre_covid_2018_2019", pd.Timestamp("2018-01-01"), pd.Timestamp("2019-12-01")),
        ("covid_2020_2021", pd.Timestamp("2020-01-01"), pd.Timestamp("2021-12-01")),
        ("sanctions_2022", pd.Timestamp("2022-01-01"), pd.Timestamp("2022-12-01")),
        ("tightening_2023", pd.Timestamp("2023-01-01"), pd.Timestamp("2023-12-01")),
        ("pre_selection_2024_2025q1", pd.Timestamp("2024-01-01"), pd.Timestamp("2025-03-01")),
    ]


def apply_train_mode(df: pd.DataFrame, cutoff: pd.Timestamp, train_mode: str) -> pd.DataFrame:
    train = df[df.index <= cutoff].copy()
    if train_mode == "rolling120" and len(train) > 120:
        train = train.iloc[-120:].copy()
    return train


def forecast_fixed_var(train_pp: pd.DataFrame, target_date: pd.Timestamp, seed: int) -> tuple[float, str]:
    del target_date
    candidate = fixed_var_candidate()
    pred = forecast_seasonal_var(train_pp, 1, candidate, seed)
    if not np.isfinite(pred):
        return np.nan, "nan_prediction"
    return float(pred), "ok"


def forecast_archived_bvar(train_pp: pd.DataFrame, target_date: pd.Timestamp, seed: int) -> tuple[float, str]:
    del target_date
    np.random.seed(seed)
    cols = ["CPI", "Food", "USD", "Ruonia"]
    data = train_pp.loc[:, cols].dropna()
    if len(data) < 36:
        return np.nan, "insufficient_train"
    model = BVARForecaster(lags=4, lambda1=1.0, n_draws=160, var_names=cols)
    model.fit(data, target_col="CPI")
    pred = float(model.forecast_full(horizon=1)["median"][0, 0])
    return pred, "ok"


def _predict_point_model(model: Any, train_model_df: pd.DataFrame, target_date: pd.Timestamp) -> float:
    train_ext = train_model_df.copy()
    train_ext.loc[target_date] = np.nan
    model.fit(train_model_df, "Все товары и услуги")
    result = model.predict(train_ext, target_date)
    return float(result["prediction"] - 100)


def forecast_ridge_prodproxy_roll24(train_model_df: pd.DataFrame, target_date: pd.Timestamp, seed: int) -> tuple[float, str]:
    del seed
    model = RidgeProductionProxyRollingForecaster(use_2022_dummy=False, seasonality_window=24)
    return _predict_point_model(model, train_model_df, target_date), "ok"


def forecast_huber(train_model_df: pd.DataFrame, target_date: pd.Timestamp, seed: int) -> tuple[float, str]:
    del seed
    model = HuberForecaster()
    return _predict_point_model(model, train_model_df, target_date), "ok"


def forecast_ridge_shock(train_model_df: pd.DataFrame, target_date: pd.Timestamp, seed: int) -> tuple[float, str]:
    del seed
    model = RidgeShockDummiesForecaster(use_2022_dummy=False)
    return _predict_point_model(model, train_model_df, target_date), "ok"


class CutoffSubcomponentMultiForecaster(SubcomponentMultiForecaster):
    """SubcomponentMulti wrapper that cuts loaded subcomponent rows at cutoff."""

    def __init__(self, cutoff: pd.Timestamp, **kwargs: Any):
        super().__init__(**kwargs)
        self.cutoff = cutoff.to_period("M").to_timestamp()
        self.loaded_subcomponent_max_date = pd.NaT

    def _load_data(self, data_dir):  # noqa: ANN001
        sub_data = super()._load_data(data_dir)
        sub_data = sub_data[sub_data.index <= self.cutoff].copy()
        self.loaded_subcomponent_max_date = sub_data.index.max() if len(sub_data) else pd.NaT
        return sub_data


def forecast_subcomponent_multi_cutoff(train_model_df: pd.DataFrame, target_date: pd.Timestamp, seed: int) -> tuple[float, str]:
    del seed
    cutoff = train_model_df.index.max()
    model = CutoffSubcomponentMultiForecaster(cutoff=cutoff, horizon=1, use_exog_forecast=False)
    model.fit(train_model_df, "Все товары и услуги")
    result = model.predict(train_model_df, target_date)
    if pd.notna(model.loaded_subcomponent_max_date) and model.loaded_subcomponent_max_date > cutoff:
        return float(result["prediction"] - 100), "leakage_violation"
    return float(result["prediction"] - 100), "ok_cutoff_subcomponents"


def baseline_specs(skip_subcomponent: bool) -> list[BaselineSpec]:
    specs = [
        BaselineSpec("Ridge_ProdProxy_Roll24", lambda train_model, train_pp, target, seed: forecast_ridge_prodproxy_roll24(train_model, target, seed), "Production proxy rolling seasonality baseline."),
        BaselineSpec("Huber", lambda train_model, train_pp, target, seed: forecast_huber(train_model, target, seed), "HuberForecaster default settings."),
        BaselineSpec("RidgeShockDummies", lambda train_model, train_pp, target, seed: forecast_ridge_shock(train_model, target, seed), "RidgeShockDummiesForecaster(use_2022_dummy=False)."),
        BaselineSpec("Archived_BVAR", lambda train_model, train_pp, target, seed: forecast_archived_bvar(train_pp, target, seed), "BVAR lags=4 lambda1=1.0 variables CPI/Food/USD/Ruonia."),
    ]
    if not skip_subcomponent:
        specs.append(
            BaselineSpec(
                "SubcomponentMulti_cutoff",
                lambda train_model, train_pp, target, seed: forecast_subcomponent_multi_cutoff(train_model, target, seed),
                "SubcomponentMulti with _load_data cut to each cutoff; use_exog_forecast=False.",
            )
        )
    return specs


def leakage_record(train_pp: pd.DataFrame, target_date: pd.Timestamp, train_mode: str) -> dict[str, Any]:
    cutoff = target_date - pd.DateOffset(months=1)
    vars_ = list(FIXED_VAR_CONFIG["variables"])
    seasonal_source = train_pp.loc[:, vars_].dropna()
    if len(seasonal_source) > FIXED_VAR_CONFIG["seasonality_window_months"]:
        seasonal_source = seasonal_source.iloc[-FIXED_VAR_CONFIG["seasonality_window_months"] :]
    max_source = seasonal_source.index.max() if len(seasonal_source) else pd.NaT
    min_source = seasonal_source.index.min() if len(seasonal_source) else pd.NaT
    return {
        "target_date": target_date,
        "cutoff": cutoff,
        "train_mode": train_mode,
        "train_start": train_pp.index.min() if len(train_pp) else pd.NaT,
        "train_end": train_pp.index.max() if len(train_pp) else pd.NaT,
        "seasonal_source_start": min_source,
        "seasonal_source_end": max_source,
        "seasonal_source_n": len(seasonal_source),
        "seasonal_window_months": FIXED_VAR_CONFIG["seasonality_window_months"],
        "uses_target_or_future": bool(pd.notna(max_source) and max_source >= target_date),
        "train_end_after_cutoff": bool(len(train_pp) and train_pp.index.max() > cutoff),
    }


def calculate_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    ok = predictions.dropna(subset=["prediction", "error"])
    group_cols = ["train_mode", "window", "model"]
    for keys, group in ok.groupby(group_cols, sort=False):
        train_mode, window, model = keys
        errors = group["error"].astype(float)
        abs_errors = errors.abs()
        rows.append(
            {
                "train_mode": train_mode,
                "window": window,
                "model": model,
                "n": len(group),
                "MAE": abs_errors.mean(),
                "RMSE": np.sqrt((errors**2).mean()),
                "Mean_Error": errors.mean(),
                "Max_Error": abs_errors.max(),
                "KPI_Violations": int((abs_errors > 0.5).sum()),
                "Coverage_50pct": float((abs_errors <= 0.5).mean() * 100),
            }
        )

    # Aggregate rows across all non-overlapping historical windows.
    for keys, group in ok.groupby(["train_mode", "model"], sort=False):
        train_mode, model = keys
        errors = group["error"].astype(float)
        abs_errors = errors.abs()
        rows.append(
            {
                "train_mode": train_mode,
                "window": "all_historical_pre_selection",
                "model": model,
                "n": len(group),
                "MAE": abs_errors.mean(),
                "RMSE": np.sqrt((errors**2).mean()),
                "Mean_Error": errors.mean(),
                "Max_Error": abs_errors.max(),
                "KPI_Violations": int((abs_errors > 0.5).sum()),
                "Coverage_50pct": float((abs_errors <= 0.5).mean() * 100),
            }
        )
    return pd.DataFrame(rows).sort_values(["train_mode", "window", "MAE", "model"])


def calculate_ensemble_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    ok = predictions[predictions["status"].str.startswith("ok", na=False)].copy()
    pivot = ok.pivot_table(
        index=["train_mode", "window", "target_date", "actual"],
        columns="model",
        values="prediction",
        aggfunc="first",
    ).reset_index()

    ensemble_defs = {
        "Blend_80_RidgeProd_20_FixedVAR": lambda df: 0.8 * df["Ridge_ProdProxy_Roll24"] + 0.2 * df["Fixed_VAR_roll42_l5"],
        "Blend_70_RidgeProd_30_FixedVAR": lambda df: 0.7 * df["Ridge_ProdProxy_Roll24"] + 0.3 * df["Fixed_VAR_roll42_l5"],
        "Blend_50_RidgeProd_50_FixedVAR": lambda df: 0.5 * df["Ridge_ProdProxy_Roll24"] + 0.5 * df["Fixed_VAR_roll42_l5"],
        "Blend_80_Huber_20_FixedVAR": lambda df: 0.8 * df["Huber"] + 0.2 * df["Fixed_VAR_roll42_l5"],
    }

    for name, fn in ensemble_defs.items():
        needed = ["Fixed_VAR_roll42_l5"]
        if "RidgeProd" in name:
            needed.append("Ridge_ProdProxy_Roll24")
        if "Huber" in name:
            needed.append("Huber")
        if not all(col in pivot.columns for col in needed):
            continue
        temp = pivot.dropna(subset=needed + ["actual"]).copy()
        if temp.empty:
            continue
        temp["prediction"] = fn(temp)
        temp["error"] = temp["actual"] - temp["prediction"]
        for keys, group in temp.groupby(["train_mode", "window"], sort=False):
            train_mode, window = keys
            errors = group["error"].astype(float)
            abs_errors = errors.abs()
            rows.append(
                {
                    "train_mode": train_mode,
                    "window": window,
                    "model": name,
                    "n": len(group),
                    "MAE": abs_errors.mean(),
                    "RMSE": np.sqrt((errors**2).mean()),
                    "Mean_Error": errors.mean(),
                    "Max_Error": abs_errors.max(),
                    "KPI_Violations": int((abs_errors > 0.5).sum()),
                    "Coverage_50pct": float((abs_errors <= 0.5).mean() * 100),
                }
            )
        for train_mode, group in temp.groupby("train_mode", sort=False):
            errors = group["error"].astype(float)
            abs_errors = errors.abs()
            rows.append(
                {
                    "train_mode": train_mode,
                    "window": "all_historical_pre_selection",
                    "model": name,
                    "n": len(group),
                    "MAE": abs_errors.mean(),
                    "RMSE": np.sqrt((errors**2).mean()),
                    "Mean_Error": errors.mean(),
                    "Max_Error": abs_errors.max(),
                    "KPI_Violations": int((abs_errors > 0.5).sum()),
                    "Coverage_50pct": float((abs_errors <= 0.5).mean() * 100),
                }
            )
    return pd.DataFrame(rows).sort_values(["train_mode", "window", "MAE", "model"]) if rows else pd.DataFrame()


def run_backtests(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    official = load_official_data()
    model_df = load_model_frame()
    baselines = baseline_specs(args.skip_subcomponent)

    prediction_rows = []
    leakage_rows = []
    target_windows = historical_windows(args.profile)
    train_modes = ["expanding", "rolling120"]

    for train_mode in train_modes:
        for window_name, start, end in target_windows:
            target_dates = pd.date_range(start=start, end=end, freq="MS")
            for target_date in target_dates:
                if target_date >= SELECTION_START or target_date > official.index.max():
                    continue
                cutoff = target_date - pd.DateOffset(months=1)
                train_pp = apply_train_mode(official, cutoff, train_mode)
                train_model = apply_train_mode(model_df, cutoff, train_mode)
                actual = float(official.loc[target_date, "CPI"])
                seed = args.seed + int(target_date.strftime("%Y%m")) + (100000 if train_mode == "rolling120" else 0)

                leakage_rows.append(leakage_record(train_pp, target_date, train_mode))

                try:
                    pred, status = forecast_fixed_var(train_pp, target_date, seed)
                except Exception as exc:
                    pred, status = np.nan, f"error: {type(exc).__name__}: {exc}"
                prediction_rows.append(
                    {
                        "train_mode": train_mode,
                        "window": window_name,
                        "model": "Fixed_VAR_roll42_l5",
                        "target_date": target_date,
                        "cutoff": cutoff,
                        "train_start": train_pp.index.min() if len(train_pp) else pd.NaT,
                        "train_end": train_pp.index.max() if len(train_pp) else pd.NaT,
                        "train_n": len(train_pp),
                        "actual": actual,
                        "prediction": pred,
                        "error": actual - pred if np.isfinite(pred) else np.nan,
                        "abs_error": abs(actual - pred) if np.isfinite(pred) else np.nan,
                        "status": status,
                    }
                )

                for spec_index, spec in enumerate(baselines):
                    try:
                        pred, status = spec.forecast_fn(train_model, train_pp, target_date, seed + spec_index * 1000)
                    except Exception as exc:
                        pred, status = np.nan, f"error: {type(exc).__name__}: {exc}"
                    prediction_rows.append(
                        {
                            "train_mode": train_mode,
                            "window": window_name,
                            "model": spec.name,
                            "target_date": target_date,
                            "cutoff": cutoff,
                            "train_start": train_model.index.min() if len(train_model) else pd.NaT,
                            "train_end": train_model.index.max() if len(train_model) else pd.NaT,
                            "train_n": len(train_model),
                            "actual": actual,
                            "prediction": pred,
                            "error": actual - pred if np.isfinite(pred) else np.nan,
                            "abs_error": abs(actual - pred) if np.isfinite(pred) else np.nan,
                            "status": status,
                            "notes": spec.notes,
                        }
                    )

    predictions = pd.DataFrame(prediction_rows)
    leakage = pd.DataFrame(leakage_rows)
    metrics = calculate_metrics(predictions)
    ensemble_metrics = calculate_ensemble_metrics(predictions)
    return predictions, metrics, ensemble_metrics, leakage


def write_config(out_dir: Path, args: argparse.Namespace) -> None:
    config = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "selection_window_excluded": {
            "start": str(SELECTION_START.date()),
            "end": str(SELECTION_END.date()),
        },
        "fixed_var_config": {
            **FIXED_VAR_CONFIG,
            "variables": list(FIXED_VAR_CONFIG["variables"]),
        },
        "historical_windows": [
            {"name": name, "start": str(start.date()), "end": str(end.date())}
            for name, start, end in historical_windows(args.profile)
        ],
        "train_modes": ["expanding", "rolling120"],
        "baselines": [spec.name for spec in baseline_specs(args.skip_subcomponent)],
        "skip_subcomponent": args.skip_subcomponent,
        "profile": args.profile,
        "seed": args.seed,
    }
    (out_dir / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")


def write_notes(out_dir: Path, metrics: pd.DataFrame, ensemble_metrics: pd.DataFrame, leakage: pd.DataFrame) -> None:
    overall = metrics[metrics["window"] == "all_historical_pre_selection"].copy()
    expanding = overall[overall["train_mode"] == "expanding"].sort_values("MAE")
    rolling = overall[overall["train_mode"] == "rolling120"].sort_values("MAE")
    ens_overall = ensemble_metrics[ensemble_metrics["window"] == "all_historical_pre_selection"].copy()
    leak_violations = int((leakage["uses_target_or_future"] | leakage["train_end_after_cutoff"]).sum()) if len(leakage) else 0

    lines = [
        "# Fixed Config Robustness Notes",
        "",
        f"- Run directory: `{out_dir}`",
        "- Frozen candidate: `fine_seasonal_resid_var_tc_roll42_l5`",
        "- Excluded selection window: `2025-04` through `2026-03`.",
        "- Historical h=1 windows/profile are recorded in `config.json`.",
        "- Train modes: expanding and rolling120.",
        f"- Leakage check violations for fixed seasonal residualization: `{leak_violations}`.",
        "",
        "## Expanding Overall",
        "",
        expanding.to_markdown(index=False) if len(expanding) else "_No expanding metrics._",
        "",
        "## Rolling120 Overall",
        "",
        rolling.to_markdown(index=False) if len(rolling) else "_No rolling120 metrics._",
        "",
        "## Ensemble Overlay Diagnostics",
        "",
        ens_overall.sort_values(["train_mode", "MAE"]).to_markdown(index=False)
        if len(ens_overall)
        else "_No ensemble metrics._",
        "",
        "## Interpretation",
        "",
        "- The frozen VAR is evaluated without retuning lag or seasonal window.",
        "- SubcomponentMulti is wrapped to cut subcomponent source rows to the cutoff; `use_exog_forecast=False` avoids forecast-helper side effects.",
        "- Ensemble overlays are diagnostic fixed blends, not optimized ensemble weights.",
    ]
    (out_dir / "notes.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def make_charts(out_dir: Path, metrics: pd.DataFrame) -> None:
    import matplotlib.pyplot as plt

    overall = metrics[
        (metrics["window"] == "all_historical_pre_selection")
        & (metrics["train_mode"] == "expanding")
    ].sort_values("MAE")
    if overall.empty:
        return
    plt.figure(figsize=(11, 5))
    plt.bar(overall["model"], overall["MAE"], color="#2f6f9f")
    plt.axhline(0.30, color="red", linestyle="--", linewidth=1, label="h=1 target 0.30")
    plt.ylabel("MAE, p.p.")
    plt.title("Historical Robustness h=1 MAE, Expanding")
    plt.xticks(rotation=35, ha="right")
    plt.grid(True, axis="y", alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "expanding_overall_mae.png", dpi=160)
    plt.close()

    fixed = metrics[
        (metrics["model"] == "Fixed_VAR_roll42_l5")
        & (metrics["window"] != "all_historical_pre_selection")
    ].copy()
    if fixed.empty:
        return
    pivot = fixed.pivot(index="window", columns="train_mode", values="MAE")
    pivot.plot(kind="bar", figsize=(11, 5))
    plt.axhline(0.30, color="red", linestyle="--", linewidth=1)
    plt.ylabel("MAE, p.p.")
    plt.title("Fixed VAR Robustness by Historical Window")
    plt.xticks(rotation=35, ha="right")
    plt.grid(True, axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_dir / "fixed_var_window_mae.png", dpi=160)
    plt.close()


def main() -> int:
    args = parse_args()
    out_dir = RUNS_DIR / args.run_name
    out_dir.mkdir(parents=True, exist_ok=False)

    write_config(out_dir, args)
    predictions, metrics, ensemble_metrics, leakage = run_backtests(args)

    predictions.to_csv(out_dir / "predictions.csv", index=False)
    metrics.to_csv(out_dir / "metrics.csv", index=False)
    ensemble_metrics.to_csv(out_dir / "ensemble_metrics.csv", index=False)
    comparison = pd.concat(
        [
            metrics.assign(kind="standalone"),
            ensemble_metrics.assign(kind="ensemble") if len(ensemble_metrics) else pd.DataFrame(),
        ],
        ignore_index=True,
    )
    comparison.to_csv(out_dir / "comparison.csv", index=False)
    leakage.to_csv(out_dir / "leakage_checks.csv", index=False)
    (out_dir / "run_fixed_config_robustness.py").write_text(Path(__file__).read_text(encoding="utf-8"), encoding="utf-8")
    write_notes(out_dir, metrics, ensemble_metrics, leakage)
    make_charts(out_dir, metrics)

    print(f"Saved robustness artifacts to {out_dir}")
    overall = metrics[metrics["window"] == "all_historical_pre_selection"].sort_values(["train_mode", "MAE"])
    print(overall.to_string(index=False))
    if len(ensemble_metrics):
        print("\nEnsemble diagnostics:")
        print(
            ensemble_metrics[ensemble_metrics["window"] == "all_historical_pre_selection"]
            .sort_values(["train_mode", "MAE"])
            .to_string(index=False)
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
