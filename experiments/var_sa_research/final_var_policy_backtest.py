#!/usr/bin/env python3
"""
Final rolling backtest for the mandatory VAR-family policy.

Evaluates:
- SeasonalVAR_CPI_F_NF_S: deterministic seasonal VAR(1) on CPI + Food + NonFood + Services.
- RegimeMacroVARX_l1: normal regime VARX with USD/Ruonia/Ki_i, shock regime Huber VAR.
- Hybrid_VAR_Policy: h=1 uses RegimeMacroVARX_l1, h=12 uses SeasonalVAR_CPI_F_NF_S.

The h=12 output stores both the 12th-step rolling backtest prediction and full deterministic
12-month paths for trajectory diagnostics. No random noise is added.
"""
from __future__ import annotations

import argparse
import json
import math
import warnings
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import HuberRegressor

try:
    from statsmodels.tsa.api import VAR
except Exception:  # pragma: no cover - statsmodels is expected in this project
    VAR = None


ROOT = Path(__file__).resolve().parents[2]
RESEARCH_DIR = ROOT / "experiments" / "var_sa_research"
RUNS_DIR = RESEARCH_DIR / "runs"

ENDOG = ["CPI", "Food", "NonFood", "Services"]
MACRO = ["USD", "Ruonia", "Ki_i"]


@dataclass(frozen=True)
class ForecastResult:
    prediction: float
    path: np.ndarray
    model_used: str
    regime: str


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


def design_matrix(data: np.ndarray, lags: int, exog: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    x_rows, y_rows = [], []
    for t in range(lags, len(data)):
        row = [1.0]
        for lag in range(1, lags + 1):
            row.extend(data[t - lag])
        if exog is not None:
            row.extend(exog[t])
        x_rows.append(row)
        y_rows.append(data[t])
    return np.asarray(x_rows), np.asarray(y_rows)


def fit_equations(x: np.ndarray, y: np.ndarray, estimator: str = "ols") -> np.ndarray:
    betas = []
    for j in range(y.shape[1]):
        if estimator == "huber":
            try:
                model = HuberRegressor(alpha=0.0, epsilon=1.35, max_iter=300, fit_intercept=False)
                model.fit(x, y[:, j])
                betas.append(model.coef_)
            except Exception:
                betas.append(np.linalg.lstsq(x, y[:, j], rcond=None)[0])
        else:
            betas.append(np.linalg.lstsq(x, y[:, j], rcond=None)[0])
    return np.asarray(betas)


def eq_var_path(
    train: pd.DataFrame,
    endog: list[str],
    horizon: int,
    *,
    lags: int = 1,
    estimator: str = "ols",
    exog_cols: list[str] | None = None,
    exog_future: np.ndarray | None = None,
    min_train: int = 40,
) -> np.ndarray:
    exog_cols = exog_cols or []
    data = train.loc[:, endog + exog_cols].dropna()
    if len(data) < max(min_train, lags + 24):
        return np.full(horizon, np.nan)

    y_data = data.loc[:, endog].values.astype(float)
    x_exog = data.loc[:, exog_cols].values.astype(float) if exog_cols else None
    x, y = design_matrix(y_data, lags, x_exog)
    if len(x) < lags + 12:
        return np.full(horizon, np.nan)

    beta = fit_equations(x, y, estimator)
    hist = list(y_data[-lags:])
    if exog_cols and exog_future is None:
        exog_future = data.iloc[-1][exog_cols].values.astype(float)

    cpi_idx = endog.index("CPI")
    path = []
    for _step in range(1, horizon + 1):
        row = [1.0]
        for lag in range(1, lags + 1):
            row.extend(hist[-lag])
        if exog_cols:
            row.extend(exog_future)
        pred_vec = (np.asarray([row]) @ beta.T).ravel()
        hist.append(pred_vec)
        path.append(float(pred_vec[cpi_idx]))
    return np.asarray(path)


def seasonal_var_path(train: pd.DataFrame, horizon: int, *, endog: list[str] = ENDOG, lags: int = 1) -> np.ndarray:
    data = train.loc[:, endog].dropna()
    if len(data) < 40:
        return np.full(horizon, np.nan)

    month_means = data.groupby(data.index.month).mean()
    resid = data.copy()
    for month in range(1, 13):
        mask = resid.index.month == month
        if month in month_means.index:
            resid.loc[mask, endog] = data.loc[mask, endog] - month_means.loc[month, endog].values

    arr = resid.values.astype(float)
    x, y = design_matrix(arr, lags)
    if len(x) < lags + 12:
        return np.full(horizon, np.nan)

    beta = fit_equations(x, y, "ols")
    hist = list(arr[-lags:])
    last_date = data.index.max()
    cpi_idx = endog.index("CPI")
    path = []
    for step in range(1, horizon + 1):
        row = [1.0]
        for lag in range(1, lags + 1):
            row.extend(hist[-lag])
        resid_pred = (np.asarray([row]) @ beta.T).ravel()
        hist.append(resid_pred)
        month = (last_date + pd.DateOffset(months=step)).month
        seasonal = month_means.loc[month, endog].values if month in month_means.index else np.zeros(len(endog))
        recon = resid_pred + seasonal
        path.append(float(recon[cpi_idx]))
    return np.asarray(path)


def regime_is_shock(train: pd.DataFrame) -> bool:
    cpi = train["CPI"].dropna()
    if len(cpi) < 12:
        return False
    last_abs = abs(float(cpi.iloc[-1]))
    trailing_std = float(cpi.iloc[-12:].std())
    return last_abs >= 1.0 or trailing_std >= 0.55


def regime_macro_varx_path(train: pd.DataFrame, horizon: int) -> ForecastResult:
    if regime_is_shock(train):
        path = eq_var_path(train, ENDOG, horizon, estimator="huber", min_train=40)
        return ForecastResult(float(path[horizon - 1]), path, "RegimeMacroVARX_l1", "shock_huber_var")
    path = eq_var_path(train, ENDOG, horizon, estimator="ols", exog_cols=MACRO, min_train=42)
    return ForecastResult(float(path[horizon - 1]), path, "RegimeMacroVARX_l1", "normal_macro_varx")


def seasonal_var_forecast(train: pd.DataFrame, horizon: int) -> ForecastResult:
    path = seasonal_var_path(train, horizon)
    return ForecastResult(float(path[horizon - 1]), path, "SeasonalVAR_CPI_F_NF_S", "seasonal_var")


def hybrid_policy_forecast(train: pd.DataFrame, horizon: int) -> ForecastResult:
    if horizon == 1:
        result = regime_macro_varx_path(train, horizon)
        return ForecastResult(result.prediction, result.path, "Hybrid_VAR_Policy", result.regime)
    if horizon == 12:
        result = seasonal_var_forecast(train, horizon)
        return ForecastResult(result.prediction, result.path, "Hybrid_VAR_Policy", result.regime)
    raise ValueError("Hybrid policy is defined only for h=1 and h=12 in this final backtest.")


def rolling_targets(data: pd.DataFrame) -> dict[str, pd.DatetimeIndex]:
    return {
        "2018_2019": pd.date_range("2018-01-01", "2019-12-01", freq="MS"),
        "2020_2021": pd.date_range("2020-01-01", "2021-12-01", freq="MS"),
        "2022_shock": pd.date_range("2022-01-01", "2022-12-01", freq="MS"),
        "2023": pd.date_range("2023-01-01", "2023-12-01", freq="MS"),
        "2024_2025q1": pd.date_range("2024-01-01", "2025-03-01", freq="MS"),
        "selection_2025-04_2026-03": pd.date_range("2025-04-01", "2026-03-01", freq="MS"),
    }


def error_metrics(frame: pd.DataFrame) -> dict[str, float]:
    clean = frame.dropna(subset=["error"])
    if clean.empty:
        return {"n": 0, "MAE": np.nan, "RMSE": np.nan, "Bias": np.nan, "Max_Error": np.nan,
                "KPI_Violations": np.nan, "Coverage_50pct": np.nan}
    err = clean["error"].astype(float)
    abs_err = err.abs()
    return {
        "n": int(len(clean)),
        "MAE": float(abs_err.mean()),
        "RMSE": float(math.sqrt((err ** 2).mean())),
        "Bias": float(err.mean()),
        "Max_Error": float(abs_err.max()),
        "KPI_Violations": int((abs_err > 0.5).sum()),
        "Coverage_50pct": float((abs_err <= 0.5).mean() * 100),
    }


def trajectory_metrics(path: np.ndarray, train: pd.DataFrame, target_dates: list[pd.Timestamp]) -> dict[str, float]:
    if not np.all(np.isfinite(path)):
        return {
            "path_std": np.nan, "vol_ratio": np.nan, "flatness": np.nan,
            "max_jump": np.nan, "sign_changes": np.nan, "seasonal_corr": np.nan,
            "seasonal_amplitude": np.nan, "explosive": True,
        }
    diffs = np.diff(path)
    hist = train["CPI"].dropna().iloc[-36:].values
    hist_std = float(np.std(hist)) if len(hist) else np.nan
    month_means = train.assign(_month=train.index.month).groupby("_month")["CPI"].mean()
    seasonal_shape = np.asarray([month_means.get(d.month, np.nan) for d in target_dates])
    if np.std(path) > 1e-8 and np.all(np.isfinite(seasonal_shape)) and np.std(seasonal_shape) > 1e-8:
        seasonal_corr = float(np.corrcoef(path, seasonal_shape)[0, 1])
    else:
        seasonal_corr = np.nan
    return {
        "path_std": float(np.std(path)),
        "vol_ratio": float(np.std(path) / hist_std) if hist_std and hist_std > 1e-9 else np.nan,
        "flatness": float(np.mean(np.abs(diffs) < 0.05)) if len(diffs) else np.nan,
        "max_jump": float(np.max(np.abs(diffs))) if len(diffs) else 0.0,
        "sign_changes": int(np.sum(np.diff(np.sign(path)) != 0)),
        "seasonal_corr": seasonal_corr,
        "seasonal_amplitude": float(np.max(path) - np.min(path)),
        "explosive": bool(np.max(np.abs(path)) > 5.0 or (hist_std and np.std(path) > 4 * hist_std)),
    }


def run_backtest(data: pd.DataFrame, out_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    models = {
        "SeasonalVAR_CPI_F_NF_S": seasonal_var_forecast,
        "RegimeMacroVARX_l1": regime_macro_varx_path,
        "Hybrid_VAR_Policy": hybrid_policy_forecast,
    }
    windows = rolling_targets(data)
    pred_rows, path_rows, leakage_rows = [], [], []

    for horizon in [1, 12]:
        for window_name, targets in windows.items():
            for target_date in targets:
                if target_date not in data.index:
                    continue
                cutoff = target_date - pd.DateOffset(months=horizon)
                train = data[data.index <= cutoff]
                actual = float(data.loc[target_date, "CPI"])
                if len(train) < 48:
                    continue
                for model_name, forecast_fn in models.items():
                    result = forecast_fn(train, horizon)
                    prediction = result.prediction
                    err = actual - prediction if np.isfinite(prediction) else np.nan
                    pred_rows.append({
                        "horizon": horizon,
                        "window": window_name,
                        "target_date": target_date,
                        "cutoff": cutoff,
                        "model": model_name,
                        "model_used": result.model_used,
                        "regime": result.regime,
                        "actual": actual,
                        "prediction": prediction,
                        "error": err,
                        "abs_error": abs(err) if np.isfinite(err) else np.nan,
                    })
                    if horizon == 12 and np.all(np.isfinite(result.path)):
                        future_dates = [cutoff + pd.DateOffset(months=step) for step in range(1, 13)]
                        diag = trajectory_metrics(result.path, train, future_dates)
                        valid_actuals = [
                            float(data.loc[d, "CPI"]) if d in data.index else np.nan
                            for d in future_dates
                        ]
                        valid = np.isfinite(valid_actuals)
                        path_mae = float(np.mean(np.abs(np.asarray(valid_actuals)[valid] - result.path[valid]))) if np.any(valid) else np.nan
                        path_rows.append({
                            "cutoff": cutoff,
                            "window": window_name,
                            "target_date_h12": target_date,
                            "model": model_name,
                            "regime": result.regime,
                            "path_mae_available": path_mae,
                            **diag,
                        })
                        for step, (date, value) in enumerate(zip(future_dates, result.path), 1):
                            path_rows.append({
                                "cutoff": cutoff,
                                "window": window_name,
                                "target_date_h12": target_date,
                                "model": model_name,
                                "regime": result.regime,
                                "step": step,
                                "path_date": date,
                                "path_prediction": float(value),
                            })

    predictions = pd.DataFrame(pred_rows)
    path_details = pd.DataFrame(path_rows)

    # Future-corruption leakage probe for the final models.
    probe_targets = [pd.Timestamp("2024-06-01"), pd.Timestamp("2025-09-01")]
    for horizon in [1, 12]:
        for target in probe_targets:
            if target not in data.index:
                continue
            cutoff = target - pd.DateOffset(months=horizon)
            clean_train = data[data.index <= cutoff]
            tampered = data.copy()
            for col in tampered.columns:
                tampered.loc[tampered.index >= target, col] += 999
            tampered_train = tampered[tampered.index <= cutoff]
            for model_name, forecast_fn in models.items():
                clean = forecast_fn(clean_train, horizon).prediction
                dirty = forecast_fn(tampered_train, horizon).prediction
                leakage_rows.append({
                    "horizon": horizon,
                    "target_date": target,
                    "model": model_name,
                    "clean_prediction": clean,
                    "tampered_prediction": dirty,
                    "leakage_free": bool(np.isclose(clean, dirty, equal_nan=True)),
                })

    leakage = pd.DataFrame(leakage_rows)
    predictions.to_csv(out_dir / "predictions.csv", index=False)
    path_details.to_csv(out_dir / "trajectory_paths_and_metrics.csv", index=False)
    leakage.to_csv(out_dir / "leakage_checks.csv", index=False)
    return predictions, path_details, leakage


def summarize(predictions: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    rows = []
    subsets = {
        "all_windows": predictions,
        "out_of_selection": predictions[predictions["window"] != "selection_2025-04_2026-03"],
        "out_of_selection_nonshock": predictions[
            ~predictions["window"].isin(["selection_2025-04_2026-03", "2022_shock"])
        ],
        "selection_2025-04_2026-03": predictions[predictions["window"] == "selection_2025-04_2026-03"],
        "shock_2022": predictions[predictions["window"] == "2022_shock"],
    }
    for subset_name, subset in subsets.items():
        for (horizon, model), group in subset.groupby(["horizon", "model"], sort=False):
            rows.append({
                "subset": subset_name,
                "horizon": int(horizon),
                "model": model,
                **error_metrics(group),
            })
    for (horizon, window, model), group in predictions.groupby(["horizon", "window", "model"], sort=False):
        rows.append({
            "subset": window,
            "horizon": int(horizon),
            "model": model,
            **error_metrics(group),
        })
    metrics = pd.DataFrame(rows)
    metrics = metrics.sort_values(["horizon", "subset", "MAE", "model"])
    metrics.to_csv(out_dir / "metrics.csv", index=False)
    comparison = metrics[metrics["subset"].isin(["all_windows", "out_of_selection", "out_of_selection_nonshock", "shock_2022"])]
    comparison.to_csv(out_dir / "comparison.csv", index=False)
    return metrics


def make_charts(predictions: pd.DataFrame, out_dir: Path) -> None:
    for horizon in [1, 12]:
        subset = predictions[predictions["horizon"] == horizon].copy()
        if subset.empty:
            continue
        pivot = subset.pivot_table(index="target_date", columns="model", values="prediction", aggfunc="first")
        actual = subset.drop_duplicates("target_date").set_index("target_date")["actual"].sort_index()
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(actual.index, actual.values, color="black", lw=2.2, label="Actual")
        for model in pivot.columns:
            ax.plot(pivot.index, pivot[model], lw=1.4, alpha=0.85, label=model)
        ax.axhline(0, color="gray", lw=0.6)
        ax.set_title(f"Rolling h={horizon} VAR policy backtest")
        ax.set_ylabel("CPI MoM, p.p.")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(out_dir / f"rolling_h{horizon}_predictions.png", dpi=120)
        plt.close(fig)


def write_report(out_dir: Path, metrics: pd.DataFrame, leakage: pd.DataFrame) -> None:
    def row(subset: str, horizon: int, model: str) -> pd.Series:
        m = metrics[(metrics["subset"] == subset) & (metrics["horizon"] == horizon) & (metrics["model"] == model)]
        return m.iloc[0] if not m.empty else pd.Series(dtype=object)

    summary_rows = []
    for horizon in [1, 12]:
        for model in ["SeasonalVAR_CPI_F_NF_S", "RegimeMacroVARX_l1", "Hybrid_VAR_Policy"]:
            r = row("all_windows", horizon, model)
            if not r.empty:
                summary_rows.append({
                    "horizon": horizon,
                    "model": model,
                    "all_MAE": r["MAE"],
                    "all_KPI": r["KPI_Violations"],
                    "oos_MAE": row("out_of_selection", horizon, model).get("MAE", np.nan),
                    "nonshock_MAE": row("out_of_selection_nonshock", horizon, model).get("MAE", np.nan),
                    "shock2022_MAE": row("shock_2022", horizon, model).get("MAE", np.nan),
                })
    summary = pd.DataFrame(summary_rows)
    best_h1 = summary[summary["horizon"] == 1].sort_values("all_MAE").head(1)
    best_h12 = summary[summary["horizon"] == 12].sort_values("all_MAE").head(1)
    leakage_ok = bool(leakage["leakage_free"].all()) if not leakage.empty else False

    report = [
        "# Final Mandatory VAR Policy Rolling Backtest",
        "",
        f"Run directory: `{out_dir}`",
        "",
        "## Models",
        "",
        "- `SeasonalVAR_CPI_F_NF_S`: deterministic expanding month-of-year seasonal reconstruction plus VAR(1) on residuals.",
        "- `RegimeMacroVARX_l1`: cutoff-only normal/shock regime; normal uses VARX with `USD`, `Ruonia`, `Ki_i`; shock uses robust Huber VAR without macro exog.",
        "- `Hybrid_VAR_Policy`: h=1 uses `RegimeMacroVARX_l1`; h=12 uses `SeasonalVAR_CPI_F_NF_S`.",
        "",
        "No random noise is added to point forecasts.",
        "",
        "## Summary Metrics",
        "",
        summary.to_markdown(index=False),
        "",
        "## Recommendation",
        "",
        f"- Best h=1 by all-window MAE: `{best_h1.iloc[0]['model']}` ({best_h1.iloc[0]['all_MAE']:.3f})." if not best_h1.empty else "- h=1 unavailable.",
        f"- Best h=12 by all-window MAE: `{best_h12.iloc[0]['model']}` ({best_h12.iloc[0]['all_MAE']:.3f})." if not best_h12.empty else "- h=12 unavailable.",
        "- Recommended reporting policy: use `Hybrid_VAR_Policy` when a horizon-specific mandatory VAR is allowed; use `SeasonalVAR_CPI_F_NF_S` as the single-model trajectory fallback.",
        "",
        "## Verification",
        "",
        f"- Leakage probe passed: `{leakage_ok}`.",
        "- Rolling h=1 and h=12 predictions, metrics, trajectory paths, and charts are saved in the run directory.",
        "",
        "## Files",
        "",
        "- `predictions.csv`",
        "- `metrics.csv`",
        "- `comparison.csv`",
        "- `trajectory_paths_and_metrics.csv`",
        "- `leakage_checks.csv`",
        "- `rolling_h1_predictions.png`",
        "- `rolling_h12_predictions.png`",
    ]
    (out_dir / "final_var_policy_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    (RESEARCH_DIR / "final_var_policy_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", default="final_var_policy_rolling")
    return parser.parse_args()


def main() -> int:
    warnings.filterwarnings("ignore")
    args = parse_args()
    out_dir = RUNS_DIR / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    data = load_official_data()
    config = {
        "run_name": args.run_name,
        "models": ["SeasonalVAR_CPI_F_NF_S", "RegimeMacroVARX_l1", "Hybrid_VAR_Policy"],
        "horizons": [1, 12],
        "data": str(ROOT / "data" / "inflation_data.csv"),
        "no_random_noise": True,
        "regime_rule": "shock if abs(last CPI) >= 1.0 or trailing 12-month CPI std >= 0.55",
    }
    (out_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    predictions, _paths, leakage = run_backtest(data, out_dir)
    metrics = summarize(predictions, out_dir)
    make_charts(predictions, out_dir)
    write_report(out_dir, metrics, leakage)

    print(f"Saved final VAR policy backtest to {out_dir}")
    print(metrics[metrics["subset"].isin(["all_windows", "out_of_selection_nonshock", "shock_2022"])].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
