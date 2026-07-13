#!/usr/bin/env python3
"""
codex_cli nested re-selection audit for the seasonal-residual VAR family.

The runner is research-only. It does not modify production data, production
model registry, or the shared parallel task file. All run artifacts are written
under runs/codex_cli_nested_*.
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RESEARCH_DIR = ROOT / "experiments" / "var_sa_research"
RUNS_DIR = RESEARCH_DIR / "runs"
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from experiments.var_sa_research.run_var_sa_backtests import (  # noqa: E402
    Candidate,
    _future_seasonal,
    _seasonal_means,
    forecast_seasonal_var,
    forecast_var,
    load_official_data,
)
from experiments.var_sa_research.run_fixed_config_robustness import (  # noqa: E402
    forecast_archived_bvar,
    forecast_huber,
    forecast_ridge_prodproxy_roll24,
    forecast_ridge_shock,
    load_model_frame,
)

warnings.filterwarnings("ignore")


AGENT_ID = "codex_cli"
VARIABLES_TC = ("CPI", "Food", "NonFood", "Services")
VARIABLES_COMP = ("Food", "NonFood", "Services")
ROLL_WINDOWS = [24, 30, 36, 42, 48, 60, 72]
LAGS = [1, 2, 3, 4, 5, 6]
TRAIN_MODES = ["expanding", "rolling120"]
INNER_MONTHS = 24
MIN_INNER_OBS = 12
HORIZONS = [1]


@dataclass(frozen=True)
class GridConfig:
    candidate_id: str
    variant: str
    variables: tuple[str, ...]
    target: str
    roll_window: int
    lags: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", default="codex_cli_nested_h1_full")
    parser.add_argument("--seed", type=int, default=20260607)
    parser.add_argument("--inner-months", type=int, default=INNER_MONTHS)
    parser.add_argument("--min-inner-obs", type=int, default=MIN_INNER_OBS)
    parser.add_argument(
        "--skip-project-baselines",
        action="store_true",
        help="Only compute simple baselines if project baseline runtime is problematic.",
    )
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


def apply_train_mode(df: pd.DataFrame, cutoff: pd.Timestamp, train_mode: str) -> pd.DataFrame:
    train = df[df.index <= cutoff].copy()
    if train_mode == "rolling120" and len(train) > 120:
        return train.iloc[-120:].copy()
    return train


def build_grid() -> list[GridConfig]:
    grid = []
    for roll in ROLL_WINDOWS:
        for lags in LAGS:
            grid.append(
                GridConfig(
                    candidate_id=f"seasonal_resid_var_tc_roll{roll}_l{lags}",
                    variant="total_components",
                    variables=VARIABLES_TC,
                    target="CPI",
                    roll_window=roll,
                    lags=lags,
                )
            )
            grid.append(
                GridConfig(
                    candidate_id=f"seasonal_resid_var_comp_bottomup_roll{roll}_l{lags}",
                    variant="component_bottomup",
                    variables=VARIABLES_COMP,
                    target="bottom_up",
                    roll_window=roll,
                    lags=lags,
                )
            )
    return grid


def to_candidate(cfg: GridConfig) -> Candidate:
    return Candidate(
        name=cfg.candidate_id,
        family="VAR_seasonal_residual",
        data_source="official",
        variables=cfg.variables,
        params={
            "lags": cfg.lags,
            "target": cfg.target,
            "seasonality_window_months": cfg.roll_window,
        },
        forecast_fn=forecast_seasonal_var,
        notes="codex_cli nested grid candidate",
    )


def all_outer_dates() -> pd.DataFrame:
    rows = []
    for window, start, end in historical_windows():
        for target_date in pd.date_range(start=start, end=end, freq="MS"):
            rows.append({"window": window, "target_date": target_date})
    return pd.DataFrame(rows)


def all_precompute_dates(official: pd.DataFrame, outer_dates: pd.DataFrame, inner_months: int) -> pd.DatetimeIndex:
    min_outer = outer_dates["target_date"].min()
    max_outer = outer_dates["target_date"].max()
    start = min_outer - pd.DateOffset(months=inner_months + 2)
    dates = official.index[(official.index >= start) & (official.index <= max_outer)]
    return pd.DatetimeIndex(dates)


def predict_grid_candidate(
    official: pd.DataFrame,
    cfg: GridConfig,
    target_date: pd.Timestamp,
    train_mode: str,
) -> tuple[float, str, pd.Timestamp, pd.Timestamp, int]:
    cutoff = target_date - pd.DateOffset(months=1)
    train = apply_train_mode(official, cutoff, train_mode)
    candidate = to_candidate(cfg)
    try:
        prediction = forecast_seasonal_var(train, 1, candidate, 0)
    except Exception as exc:
        return np.nan, f"error: {type(exc).__name__}: {exc}", train.index.min(), train.index.max(), len(train)
    status = "ok" if np.isfinite(prediction) else "nan_prediction"
    return float(prediction) if np.isfinite(prediction) else np.nan, status, train.index.min(), train.index.max(), len(train)


def predict_random_walk(train: pd.DataFrame) -> float:
    return float(train["CPI"].dropna().iloc[-1])


def predict_seasonal_naive(train: pd.DataFrame, roll_window: int = 42) -> float:
    mm = _seasonal_means(train, list(VARIABLES_TC), roll_window)
    target_month = (train.index.max() + pd.DateOffset(months=1)).month
    return float(_future_seasonal(mm, list(VARIABLES_TC), target_month)[0])


def predict_plain_var_l5(train: pd.DataFrame) -> float:
    cand = Candidate(
        name="plain_var_l5",
        family="VAR",
        data_source="official",
        variables=VARIABLES_TC,
        params={"lags": 5, "target": "CPI"},
        forecast_fn=forecast_var,
        notes="plain VAR baseline",
    )
    return float(forecast_var(train.loc[:, list(VARIABLES_TC)], 1, cand, 0))


def compute_grid_predictions(
    official: pd.DataFrame,
    grid: list[GridConfig],
    target_dates: pd.DatetimeIndex,
) -> pd.DataFrame:
    rows = []
    total = len(TRAIN_MODES) * len(grid) * len(target_dates)
    done = 0
    for train_mode in TRAIN_MODES:
        for cfg in grid:
            for target_date in target_dates:
                if target_date not in official.index:
                    continue
                actual = float(official.loc[target_date, "CPI"])
                cutoff = target_date - pd.DateOffset(months=1)
                pred, status, train_start, train_end, train_n = predict_grid_candidate(
                    official, cfg, target_date, train_mode
                )
                error = actual - pred if np.isfinite(pred) else np.nan
                rows.append(
                    {
                        "train_mode": train_mode,
                        "candidate_id": cfg.candidate_id,
                        "variant": cfg.variant,
                        "roll_window": cfg.roll_window,
                        "lags": cfg.lags,
                        "target_date": target_date,
                        "cutoff": cutoff,
                        "train_start": train_start,
                        "train_end": train_end,
                        "train_n": train_n,
                        "actual": actual,
                        "prediction": pred,
                        "error": error,
                        "abs_error": abs(error) if np.isfinite(error) else np.nan,
                        "status": status,
                    }
                )
                done += 1
            if done and done % 1000 == 0:
                print(f"grid precompute {done}/{total}")
    return pd.DataFrame(rows)


def compute_simple_baselines(
    official: pd.DataFrame,
    outer_dates: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for train_mode in TRAIN_MODES:
        for _, row in outer_dates.iterrows():
            target_date = row["target_date"]
            if target_date not in official.index:
                continue
            window = row["window"]
            cutoff = target_date - pd.DateOffset(months=1)
            train = apply_train_mode(official, cutoff, train_mode)
            actual = float(official.loc[target_date, "CPI"])
            baseline_fns = {
                "RandomWalk": predict_random_walk,
                "SeasonalNaive_roll42": predict_seasonal_naive,
                "PlainVAR_l5": predict_plain_var_l5,
            }
            for model, fn in baseline_fns.items():
                try:
                    pred = fn(train)
                    status = "ok" if np.isfinite(pred) else "nan_prediction"
                except Exception as exc:
                    pred = np.nan
                    status = f"error: {type(exc).__name__}: {exc}"
                error = actual - pred if np.isfinite(pred) else np.nan
                rows.append(
                    {
                        "horizon": 1,
                        "train_mode": train_mode,
                        "window": window,
                        "model": model,
                        "target_date": target_date,
                        "cutoff": cutoff,
                        "actual": actual,
                        "prediction": pred,
                        "error": error,
                        "abs_error": abs(error) if np.isfinite(error) else np.nan,
                        "status": status,
                    }
                )
    return pd.DataFrame(rows)


def compute_project_baselines(
    official: pd.DataFrame,
    model_df: pd.DataFrame,
    outer_dates: pd.DataFrame,
    seed: int,
) -> pd.DataFrame:
    rows = []
    baseline_fns: dict[str, Any] = {
        "Huber": lambda train_model, train_pp, target, run_seed: forecast_huber(train_model, target, run_seed),
        "RidgeShockDummies": lambda train_model, train_pp, target, run_seed: forecast_ridge_shock(train_model, target, run_seed),
        "Archived_BVAR": lambda train_model, train_pp, target, run_seed: forecast_archived_bvar(train_pp, target, run_seed),
        "Ridge_ProdProxy_Roll24": lambda train_model, train_pp, target, run_seed: forecast_ridge_prodproxy_roll24(train_model, target, run_seed),
    }
    for train_mode in TRAIN_MODES:
        for _, row in outer_dates.iterrows():
            target_date = row["target_date"]
            if target_date not in official.index:
                continue
            window = row["window"]
            cutoff = target_date - pd.DateOffset(months=1)
            train_pp = apply_train_mode(official, cutoff, train_mode)
            train_model = apply_train_mode(model_df, cutoff, train_mode)
            actual = float(official.loc[target_date, "CPI"])
            for idx, (model, fn) in enumerate(baseline_fns.items()):
                try:
                    pred, status = fn(train_model, train_pp, target_date, seed + idx * 1000 + int(target_date.strftime("%Y%m")))
                except Exception as exc:
                    pred = np.nan
                    status = f"error: {type(exc).__name__}: {exc}"
                error = actual - pred if np.isfinite(pred) else np.nan
                rows.append(
                    {
                        "horizon": 1,
                        "train_mode": train_mode,
                        "window": window,
                        "model": model,
                        "target_date": target_date,
                        "cutoff": cutoff,
                        "actual": actual,
                        "prediction": pred,
                        "error": error,
                        "abs_error": abs(error) if np.isfinite(error) else np.nan,
                        "status": status,
                    }
                )
    return pd.DataFrame(rows)


def nested_select_predictions(
    grid_predictions: pd.DataFrame,
    outer_dates: pd.DataFrame,
    inner_months: int,
    min_inner_obs: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    prediction_rows = []
    selection_rows = []
    leakage_rows = []

    # Index for faster slices.
    grid_predictions = grid_predictions.copy()
    grid_predictions["target_date"] = pd.to_datetime(grid_predictions["target_date"])

    for train_mode in TRAIN_MODES:
        mode_preds = grid_predictions[grid_predictions["train_mode"] == train_mode]
        for _, outer in outer_dates.iterrows():
            target_date = outer["target_date"]
            window = outer["window"]
            cutoff = target_date - pd.DateOffset(months=1)
            actual = mode_preds.loc[
                (mode_preds["target_date"] == target_date),
                "actual",
            ].dropna()
            if actual.empty:
                continue
            actual_value = float(actual.iloc[0])
            inner_dates = pd.date_range(
                end=cutoff,
                periods=inner_months,
                freq="MS",
            )
            inner = mode_preds[mode_preds["target_date"].isin(inner_dates)].dropna(subset=["abs_error"])
            candidate_scores = (
                inner.groupby(["candidate_id", "variant", "roll_window", "lags"], as_index=False)
                .agg(inner_n=("abs_error", "size"), inner_mae=("abs_error", "mean"), inner_rmse=("error", lambda x: float(np.sqrt((x.astype(float) ** 2).mean()))))
            )
            candidate_scores = candidate_scores[candidate_scores["inner_n"] >= min_inner_obs].copy()
            if candidate_scores.empty:
                continue
            candidate_scores = candidate_scores.sort_values(
                ["inner_mae", "inner_rmse", "roll_window", "lags", "variant"],
                ascending=[True, True, True, True, True],
            )
            chosen = candidate_scores.iloc[0].to_dict()
            candidate_scores["selected"] = candidate_scores["candidate_id"] == chosen["candidate_id"]
            candidate_scores["outer_target_date"] = target_date
            candidate_scores["outer_window"] = window
            candidate_scores["train_mode"] = train_mode
            candidate_scores["inner_start"] = inner_dates.min()
            candidate_scores["inner_end"] = inner_dates.max()
            selection_rows.append(candidate_scores)

            outer_pred = mode_preds[
                (mode_preds["target_date"] == target_date)
                & (mode_preds["candidate_id"] == chosen["candidate_id"])
            ]
            if outer_pred.empty:
                continue
            outer_pred_row = outer_pred.iloc[0]
            pred = float(outer_pred_row["prediction"])
            error = actual_value - pred if np.isfinite(pred) else np.nan
            prediction_rows.append(
                {
                    "horizon": 1,
                    "train_mode": train_mode,
                    "window": window,
                    "model": "NestedSeasonalResidualVAR",
                    "target_date": target_date,
                    "cutoff": cutoff,
                    "actual": actual_value,
                    "prediction": pred,
                    "error": error,
                    "abs_error": abs(error) if np.isfinite(error) else np.nan,
                    "status": outer_pred_row["status"],
                    "selected_candidate_id": chosen["candidate_id"],
                    "selected_variant": chosen["variant"],
                    "selected_roll_window": int(chosen["roll_window"]),
                    "selected_lags": int(chosen["lags"]),
                    "inner_mae": float(chosen["inner_mae"]),
                    "inner_rmse": float(chosen["inner_rmse"]),
                    "inner_n": int(chosen["inner_n"]),
                }
            )

            leakage_rows.append(
                {
                    "train_mode": train_mode,
                    "window": window,
                    "target_date": target_date,
                    "cutoff": cutoff,
                    "inner_start": inner_dates.min(),
                    "inner_end": inner_dates.max(),
                    "inner_end_after_cutoff": bool(inner_dates.max() > cutoff),
                    "outer_prediction_train_end": outer_pred_row["train_end"],
                    "outer_train_end_after_cutoff": bool(pd.Timestamp(outer_pred_row["train_end"]) > cutoff),
                    "selection_used_outer_actual": False,
                    "selected_candidate_id": chosen["candidate_id"],
                }
            )

    predictions = pd.DataFrame(prediction_rows)
    selection_log = pd.concat(selection_rows, ignore_index=True) if selection_rows else pd.DataFrame()
    leakage = pd.DataFrame(leakage_rows)
    return predictions, selection_log, leakage


def calculate_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    ok = predictions.dropna(subset=["error"]).copy()
    group_cols = ["horizon", "train_mode", "window", "model"]
    for keys, group in ok.groupby(group_cols, sort=False):
        horizon, train_mode, window, model = keys
        errors = group["error"].astype(float)
        abs_errors = errors.abs()
        rows.append(
            {
                "horizon": horizon,
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
    for keys, group in ok.groupby(["horizon", "train_mode", "model"], sort=False):
        horizon, train_mode, model = keys
        errors = group["error"].astype(float)
        abs_errors = errors.abs()
        rows.append(
            {
                "horizon": horizon,
                "train_mode": train_mode,
                "window": "all_windows",
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
    return pd.DataFrame(rows).sort_values(["horizon", "train_mode", "window", "MAE", "model"])


def selection_stability(selection_log: pd.DataFrame) -> pd.DataFrame:
    selected = selection_log[selection_log["selected"]].copy()
    if selected.empty:
        return pd.DataFrame()
    return (
        selected.groupby(["train_mode", "variant", "roll_window", "lags"], as_index=False)
        .agg(count=("selected", "size"), mean_inner_mae=("inner_mae", "mean"))
        .sort_values(["train_mode", "count", "mean_inner_mae"], ascending=[True, False, True])
    )


def write_config(out_dir: Path, args: argparse.Namespace, grid: list[GridConfig], precompute_dates: pd.DatetimeIndex) -> None:
    config = {
        "agent_id": AGENT_ID,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "method": "nested h=1 rolling/expanding re-selection",
        "inner_validation": {
            "inner_months": args.inner_months,
            "min_inner_obs": args.min_inner_obs,
            "scheme": "last N target months within each outer cutoff; each inner prediction uses its own target-1 cutoff",
        },
        "outer_windows": [
            {"name": name, "start": str(start.date()), "end": str(end.date())}
            for name, start, end in historical_windows()
        ],
        "horizons": HORIZONS,
        "train_modes": TRAIN_MODES,
        "grid": {
            "roll_windows": ROLL_WINDOWS,
            "lags": LAGS,
            "variants": ["total_components", "component_bottomup"],
            "candidate_count": len(grid),
        },
        "precompute_dates": {
            "start": str(precompute_dates.min().date()),
            "end": str(precompute_dates.max().date()),
            "count": len(precompute_dates),
        },
        "baselines": {
            "simple": ["RandomWalk", "SeasonalNaive_roll42", "PlainVAR_l5"],
            "project": [] if args.skip_project_baselines else ["Huber", "RidgeShockDummies", "Archived_BVAR", "Ridge_ProdProxy_Roll24"],
            "skipped": ["SubcomponentMulti", "h=2", "h=12"],
        },
        "seed": args.seed,
    }
    (out_dir / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")


def write_notes(
    out_dir: Path,
    metrics: pd.DataFrame,
    selection_log: pd.DataFrame,
    leakage: pd.DataFrame,
    stability: pd.DataFrame,
) -> None:
    overall = metrics[metrics["window"] == "all_windows"].sort_values(["train_mode", "MAE"])
    selected = selection_log[selection_log["selected"]].copy() if len(selection_log) else pd.DataFrame()
    leakage_violations = 0
    if len(leakage):
        leakage_violations = int((leakage["inner_end_after_cutoff"] | leakage["outer_train_end_after_cutoff"]).sum())
    lines = [
        "# codex_cli Nested Re-Selection Notes",
        "",
        f"- Run directory: `{out_dir}`",
        "- h=1 only; h=2/h=12 skipped for runtime because the required core question is h=1 deployable value.",
        "- Grid: seasonal residual VAR total/components and component-bottom-up, roll windows 24..72, lags 1..6.",
        "- Inner validation: trailing 24 target months inside each outer cutoff, minimum 12 valid inner observations.",
        f"- Leakage violations found: `{leakage_violations}`.",
        "",
        "## Overall Metrics",
        "",
        overall.to_markdown(index=False) if len(overall) else "_No metrics._",
        "",
        "## Selected Hyperparameter Stability",
        "",
        stability.head(30).to_markdown(index=False) if len(stability) else "_No selected rows._",
        "",
        "## Selected Counts By Train Mode",
        "",
        selected.groupby(["train_mode", "roll_window", "lags", "variant"]).size().reset_index(name="count").sort_values(["train_mode", "count"], ascending=[True, False]).head(40).to_markdown(index=False)
        if len(selected)
        else "_No selected rows._",
    ]
    (out_dir / "notes.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def make_charts(out_dir: Path, metrics: pd.DataFrame, selection_log: pd.DataFrame) -> None:
    import matplotlib.pyplot as plt

    overall = metrics[(metrics["window"] == "all_windows") & (metrics["train_mode"] == "expanding")].sort_values("MAE")
    if len(overall):
        plt.figure(figsize=(12, 5))
        plt.bar(overall["model"], overall["MAE"], color="#2f6f9f")
        plt.axhline(0.30, color="red", linestyle="--", linewidth=1, label="0.30 target")
        plt.ylabel("MAE, p.p.")
        plt.title("codex_cli nested re-selection h=1 overall MAE, expanding")
        plt.xticks(rotation=35, ha="right")
        plt.grid(True, axis="y", alpha=0.25)
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / "overall_expanding_mae.png", dpi=160)
        plt.close()

    selected = selection_log[selection_log["selected"]].copy() if len(selection_log) else pd.DataFrame()
    if len(selected):
        for train_mode, group in selected.groupby("train_mode"):
            counts = group["roll_window"].value_counts().sort_index()
            plt.figure(figsize=(8, 4))
            plt.bar(counts.index.astype(str), counts.values, color="#6b8e23")
            plt.title(f"Selected roll windows: {train_mode}")
            plt.xlabel("roll window")
            plt.ylabel("count")
            plt.tight_layout()
            plt.savefig(out_dir / f"selected_roll_windows_{train_mode}.png", dpi=160)
            plt.close()


def main() -> int:
    args = parse_args()
    if not args.run_name.startswith(f"{AGENT_ID}_nested_"):
        raise ValueError(f"run-name must start with {AGENT_ID}_nested_")
    out_dir = RUNS_DIR / args.run_name
    out_dir.mkdir(parents=True, exist_ok=False)

    official = load_official_data()
    outer_dates = all_outer_dates()
    precompute_dates = all_precompute_dates(official, outer_dates, args.inner_months)
    grid = build_grid()

    write_config(out_dir, args, grid, precompute_dates)
    print(f"Precomputing {len(grid)} grid candidates across {len(precompute_dates)} dates and {len(TRAIN_MODES)} train modes")
    grid_predictions = compute_grid_predictions(official, grid, precompute_dates)
    grid_predictions.to_csv(out_dir / "candidate_grid_predictions.csv", index=False)

    nested_predictions, selection_log, leakage = nested_select_predictions(
        grid_predictions,
        outer_dates,
        args.inner_months,
        args.min_inner_obs,
    )
    simple_baselines = compute_simple_baselines(official, outer_dates)
    prediction_frames = [nested_predictions, simple_baselines]
    if not args.skip_project_baselines:
        model_df = load_model_frame()
        project_baselines = compute_project_baselines(official, model_df, outer_dates, args.seed)
        prediction_frames.append(project_baselines)

    predictions = pd.concat(prediction_frames, ignore_index=True)
    metrics = calculate_metrics(predictions)
    stability = selection_stability(selection_log)

    predictions.to_csv(out_dir / "predictions.csv", index=False)
    metrics.to_csv(out_dir / "metrics.csv", index=False)
    selection_log.to_csv(out_dir / "selection_log.csv", index=False)
    leakage.to_csv(out_dir / "leakage_checks.csv", index=False)
    stability.to_csv(out_dir / "selection_stability.csv", index=False)
    (out_dir / "codex_cli_nested_reselection.py").write_text(Path(__file__).read_text(encoding="utf-8"), encoding="utf-8")
    write_notes(out_dir, metrics, selection_log, leakage, stability)
    make_charts(out_dir, metrics, selection_log)

    print(f"Saved nested artifacts to {out_dir}")
    print(metrics[metrics["window"] == "all_windows"].sort_values(["train_mode", "MAE"]).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
