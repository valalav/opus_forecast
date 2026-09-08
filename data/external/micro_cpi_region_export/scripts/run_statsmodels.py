from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from statsmodels.tsa.holtwinters import ExponentialSmoothing


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run independent statsmodels CPI forecast.")
    parser.add_argument("--input", required=True, help="Path to region_cpi_long.csv")
    parser.add_argument("--metadata", required=True, help="Path to metadata.json")
    parser.add_argument("--out", required=True, help="Output forecast_long.csv path")
    parser.add_argument("--diagnostics", default="", help="Optional diagnostics CSV path")
    parser.add_argument("--config", default="config/model.yml", help="YAML config path")
    parser.add_argument("--horizon", type=int, default=0, help="Override forecast horizon in months")
    return parser.parse_args()


def read_config(path: str) -> dict:
    config_path = Path(path)
    if not config_path.exists():
        return {}
    return yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}


def seasonal_naive(series: pd.Series, horizon: int) -> np.ndarray:
    values = series.dropna().to_numpy(dtype=float)
    if len(values) == 0:
        return np.full(horizon, np.nan)
    if len(values) >= 12:
        pattern = values[-12:]
        return np.resize(pattern, horizon)
    return np.full(horizon, values[-1])


def forecast_one(series: pd.Series, horizon: int, config: dict) -> tuple[np.ndarray, dict]:
    clean = series.astype(float)
    if clean.isna().any():
        clean = clean.interpolate(limit_direction="both")
    clean = clean.dropna()
    if isinstance(clean.index, pd.DatetimeIndex):
        clean = clean.asfreq("MS")
    min_history = int(config.get("input", {}).get("min_history_months", 36))
    seasonal_periods = int(config.get("statsmodels", {}).get("seasonal_periods", 12))
    damped_trend = bool(config.get("statsmodels", {}).get("damped_trend", True))

    diagnostics = {
        "method": "ets_additive_trend",
        "status": "ok",
        "history_months": int(clean.shape[0]),
        "aic": np.nan,
        "sse": np.nan,
        "error": "",
    }

    if clean.shape[0] < min_history:
        diagnostics["method"] = "seasonal_naive_12"
        diagnostics["status"] = "fallback_short_history"
        return seasonal_naive(clean, horizon), diagnostics

    try:
        model = ExponentialSmoothing(
            clean,
            trend="add",
            damped_trend=damped_trend,
            seasonal="add" if clean.shape[0] >= seasonal_periods * 2 else None,
            seasonal_periods=seasonal_periods if clean.shape[0] >= seasonal_periods * 2 else None,
            initialization_method="estimated",
        )
        fit = model.fit(optimized=True)
        forecast = fit.forecast(horizon).to_numpy(dtype=float)
        diagnostics["aic"] = float(getattr(fit, "aic", np.nan))
        diagnostics["sse"] = float(getattr(fit, "sse", np.nan))
        return forecast, diagnostics
    except Exception as exc:  # keep batch production alive; diagnostics records the failure
        diagnostics["method"] = "seasonal_naive_12"
        diagnostics["status"] = "fallback_model_error"
        diagnostics["error"] = str(exc)
        return seasonal_naive(clean, horizon), diagnostics


def main() -> None:
    args = parse_args()
    config = read_config(args.config)
    horizon = args.horizon or int(config.get("forecast", {}).get("horizon_months", 12))

    data = pd.read_csv(
        args.input,
        dtype={"region_code": "string", "item_code": "string"},
        encoding="utf-8-sig",
    )
    metadata = json.loads(Path(args.metadata).read_text(encoding="utf-8-sig"))
    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    data["mom_index"] = pd.to_numeric(data["mom_index"], errors="coerce")
    data = data.dropna(subset=["date", "item_code"]).sort_values(["item_code", "date"])

    latest = pd.to_datetime(metadata.get("latest_month"), errors="coerce")
    if pd.isna(latest):
        latest = data["date"].max()

    rows = []
    diagnostics_rows = []
    run_id = pd.Timestamp.utcnow().strftime("%Y%m%dT%H%M%SZ")
    region_code = str(metadata.get("region_code") or data["region_code"].dropna().iloc[0])
    region_name = str(metadata.get("region_name") or data["region_name"].dropna().iloc[0])
    forecast_dates = pd.date_range(latest + pd.DateOffset(months=1), periods=horizon, freq="MS")

    for item_code, item_df in data.groupby("item_code", dropna=True):
        item_df = item_df[item_df["date"] <= latest]
        series = item_df.set_index("date")["mom_index"].asfreq("MS")
        forecast, diag = forecast_one(series, horizon, config)

        item_name = item_df["item_name"].dropna().iloc[0] if item_df["item_name"].notna().any() else ""
        item_rosstat = (
            item_df["item_rosstat"].dropna().iloc[0] if item_df["item_rosstat"].notna().any() else ""
        )
        diagnostics_rows.append(
            {
                "run_id": run_id,
                "model_name": "statsmodels",
                "region_code": region_code,
                "region_name": region_name,
                "item_code": item_code,
                "item_rosstat": item_rosstat,
                "history_months": diag["history_months"],
                "method": diag["method"],
                "status": diag["status"],
                "aic": diag["aic"],
                "sse": diag["sse"],
                "error": diag["error"],
            }
        )
        for forecast_date, value in zip(forecast_dates, forecast, strict=True):
            rows.append(
                {
                    "run_id": run_id,
                    "model_name": "statsmodels",
                    "region_code": region_code,
                    "region_name": region_name,
                    "item_code": item_code,
                    "item_rosstat": item_rosstat,
                    "item_name": item_name,
                    "series_type": "mom_index",
                    "forecast_date": forecast_date.date().isoformat(),
                    "forecast_value": round(float(value), 6) if pd.notna(value) else np.nan,
                    "lower_bound": np.nan,
                    "upper_bound": np.nan,
                    "method": diag["method"],
                    "status": diag["status"],
                }
            )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_path, index=False, encoding="utf-8")

    diagnostics_path = Path(args.diagnostics) if args.diagnostics else out_path.with_name(
        out_path.stem + "_diagnostics.csv"
    )
    pd.DataFrame(diagnostics_rows).to_csv(diagnostics_path, index=False, encoding="utf-8")


if __name__ == "__main__":
    main()
