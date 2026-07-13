#!/usr/bin/env python3
"""
Export separate per-model CSV files for the Stavropol package.

Builds a forecast round anchored to the March 2026 workflow:
- latest available actual data cut: February 2026
- internal forecast horizon: March 2026 .. December 2027
- exported horizon: April 2026 .. December 2027

Output schema for every model file:
    Date,Model,Forecast_MoM
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, cast

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from sirena.sa_data_loader import get_sa_with_total
from sirena.models.ridge import RidgeForecaster
from sirena.models.ridge_extended import RidgeExtendedForecaster
from sirena.models.ridge_macro import RidgeMacroForecaster
from sirena.models.ridge_shock_dummies import RidgeShockDummiesForecaster
from sirena.models.bayesian_ridge import BayesianRidgeForecaster
from sirena.models.elasticnet import ElasticNetForecaster
from sirena.models.huber import HuberForecaster
from sirena.models.ngboost_model import NGBoostForecaster
from sirena.models.ngboost_shock import NGBoostShockForecaster
from sirena.models.bvar import BVARForecaster
from sirena.models.arima import SARIMAForecaster
from sirena.models.lightgbm import LightGBMForecaster
from sirena.models.prophet import ProphetForecaster
from sirena.models.ets import ETSForecaster
from sirena.models.ebm import EBMForecaster
from sirena.models.catboost_model import CatBoostForecaster
from sirena.models.subcomponent import SubcomponentForecaster
from sirena.models.subcomponent_multi import SubcomponentMultiForecaster
from sirena.models.microcomponent import MicrocomponentForecaster


CUTOFF_MONTH: pd.Timestamp = cast(pd.Timestamp, pd.Timestamp("2026-02-01"))
EXPORT_START: pd.Timestamp = cast(pd.Timestamp, pd.Timestamp("2026-04-01"))
EXPORT_END: pd.Timestamp = cast(pd.Timestamp, pd.Timestamp("2027-12-01"))
OUTPUT_DIR = PROJECT_ROOT / "archive" / "results" / "stavropol_model_csv"
MANIFEST_PATH = OUTPUT_DIR / "manifest.json"
TARGET_COL = "Все товары и услуги"


@dataclass(frozen=True)
class ModelSpec:
    name: str
    factory: Callable[[], Any]
    dataset: str
    method: str


MODEL_SPECS: list[ModelSpec] = [
    ModelSpec("Ridge", RidgeForecaster, "sa", "iterative"),
    ModelSpec("Ridge_Ext", RidgeExtendedForecaster, "sa", "iterative"),
    ModelSpec("Ridge_Shock", RidgeShockDummiesForecaster, "sa", "iterative"),
    ModelSpec("Bayes_Ridge", BayesianRidgeForecaster, "sa", "iterative"),
    ModelSpec("ElasticNet", ElasticNetForecaster, "sa", "iterative"),
    ModelSpec("Huber", HuberForecaster, "sa", "iterative"),
    ModelSpec("NGBoost", NGBoostForecaster, "sa", "iterative"),
    ModelSpec("NGBoost_Shock", NGBoostShockForecaster, "sa", "iterative"),
    ModelSpec("LightGBM", LightGBMForecaster, "sa", "iterative"),
    ModelSpec("EBM", EBMForecaster, "sa", "iterative"),
    ModelSpec("CatBoost", CatBoostForecaster, "sa", "iterative"),
    ModelSpec("Prophet", ProphetForecaster, "sa", "forecast"),
    ModelSpec("ETS", ETSForecaster, "sa", "forecast"),
    ModelSpec("Subcomp", SubcomponentForecaster, "sa", "forecast"),
    ModelSpec("Subcomp_Multi", SubcomponentMultiForecaster, "sa", "forecast"),
    ModelSpec("Micro", MicrocomponentForecaster, "sa", "forecast"),
    ModelSpec("Ridge_Macro", RidgeMacroForecaster, "macro", "forecast"),
    ModelSpec("BVAR", BVARForecaster, "bvar", "forecast"),
    ModelSpec("SARIMA", SARIMAForecaster, "sa", "forecast"),
]


def load_sa_round_data(cutoff_month: pd.Timestamp) -> pd.DataFrame:
    df = get_sa_with_total().copy()
    df.index = pd.to_datetime(df.index).to_period("M").to_timestamp()
    return df.loc[df.index <= cutoff_month].sort_index()


def load_macro_round_data(cutoff_month: pd.Timestamp) -> tuple[pd.DataFrame, pd.DataFrame]:
    macro_df = pd.read_csv(PROJECT_ROOT / "data" / "inflation_data.csv", sep=";", decimal=",")
    for column in macro_df.columns:
        if column != "Date":
            macro_df[column] = pd.to_numeric(macro_df[column], errors="coerce")
    macro_df["Date"] = pd.to_datetime(macro_df["Date"], format="%d.%m.%Y", errors="coerce")
    macro_df["Date"] = macro_df["Date"].dt.to_period("M").dt.to_timestamp()
    macro_df = macro_df.set_index("Date").sort_index()
    macro_df = macro_df.loc[macro_df.index <= cutoff_month].copy()

    brent_path = PROJECT_ROOT / "data" / "brent_prices.csv"
    if brent_path.exists():
        brent_df = pd.read_csv(brent_path)
        brent_df["Date"] = pd.to_datetime(brent_df["Date"], errors="coerce")
        brent_df["Date"] = brent_df["Date"].dt.to_period("M").dt.to_timestamp()
        brent_df = brent_df.set_index("Date").sort_index()
        if "brent" in brent_df.columns:
            macro_df = macro_df.join(brent_df[["brent"]], how="left")

    bvar_data = pd.DataFrame(index=macro_df.index)
    bvar_data["CPI"] = macro_df["mom"] - 100
    bvar_data["Food"] = macro_df["Prod"] - 100
    bvar_data["NonFood"] = macro_df["Nonprod"] - 100
    bvar_data["Services"] = macro_df["Serv"] - 100
    if "usd_nom_i" in macro_df.columns:
        bvar_data["USD"] = macro_df["usd_nom_i"] - 100
    if "Ruonia" in macro_df.columns:
        bvar_data["RUONIA"] = macro_df["Ruonia"]
    bvar_data = bvar_data.dropna()

    return macro_df, bvar_data


def ensure_timestamp(value: Any) -> pd.Timestamp:
    return cast(pd.Timestamp, pd.to_datetime(value))


def normalize_forecast(values: Any) -> np.ndarray:
    raw = values.values if hasattr(values, "values") else values
    forecast = np.asarray(raw, dtype=float)
    if forecast.ndim != 1:
        forecast = forecast.reshape(-1)
    if forecast.size and float(np.nanmean(np.abs(forecast))) > 50:
        forecast = forecast - 100
    return forecast


def iterative_forecast(model: Any, df: pd.DataFrame, horizon: int, target_col: str = TARGET_COL) -> np.ndarray:
    last_date = ensure_timestamp(df.index.max())
    predictions: list[float] = []
    df_work = df.copy()

    for step in range(horizon):
        target_date = last_date + pd.DateOffset(months=step + 1)
        df_ext = df_work.copy()
        prev_date = df_ext.index[-1]
        df_ext.loc[target_date] = df_ext.loc[prev_date].copy()
        df_ext.loc[target_date, target_col] = np.nan
        df_ext = df_ext.sort_index()

        try:
            pred_result = model.predict(df_ext, target_date)
            pred = float(pred_result["prediction"])
            if abs(pred) > 50:
                pred -= 100
        except Exception:
            seasonal_norm = getattr(model, "seasonal_norm", None)
            if seasonal_norm:
                pred = float(seasonal_norm.get(target_date.month, 100.0) - 100)
            else:
                pred = float(df_work[target_col].tail(12).mean() - 100)

        predictions.append(pred)

        if target_date not in df_work.index:
            df_work.loc[target_date] = df_work.loc[prev_date].copy()
        df_work.loc[target_date, target_col] = pred + 100

    return np.asarray(predictions, dtype=float)


def build_forecast_dates(last_actual: pd.Timestamp, horizon: int) -> pd.DatetimeIndex:
    return pd.date_range(start=last_actual + pd.DateOffset(months=1), periods=horizon, freq="MS")


def get_export_horizon(last_actual: pd.Timestamp) -> int:
    return int((EXPORT_END.year - last_actual.year) * 12 + (EXPORT_END.month - last_actual.month))


def determine_source_last_date(spec: ModelSpec, sa_df: pd.DataFrame, macro_df: pd.DataFrame) -> str:
    if spec.dataset == "macro":
        return ensure_timestamp(macro_df.index.max()).strftime("%Y-%m-%d")
    if spec.dataset == "bvar":
        return ensure_timestamp(macro_df.index.max()).strftime("%Y-%m-%d")
    if spec.name in {"Subcomp", "Subcomp_Multi"}:
        data_dir = PROJECT_ROOT / "data"
        helper: Any = spec.factory()
        internal_df = helper._load_data(data_dir)  # type: ignore[attr-defined]
        return ensure_timestamp(pd.to_datetime(internal_df.index).max()).strftime("%Y-%m-%d")
    if spec.name == "Micro":
        data_dir = PROJECT_ROOT / "data"
        helper: Any = spec.factory()
        internal_df = helper._load_data(data_dir)  # type: ignore[attr-defined]
        return ensure_timestamp(pd.to_datetime(internal_df.index).max()).strftime("%Y-%m-%d")
    return ensure_timestamp(sa_df.index.max()).strftime("%Y-%m-%d")


def run_model(spec: ModelSpec, sa_df: pd.DataFrame, macro_df: pd.DataFrame, bvar_df: pd.DataFrame, horizon: int) -> np.ndarray:
    model: Any = spec.factory()

    if spec.dataset == "macro":
        macro_no_brent = macro_df.drop(columns=["brent"], errors="ignore")
        model.fit(macro_no_brent, "mom")
        return normalize_forecast(model.forecast(horizon))

    if spec.dataset == "bvar":
        var_names = ["CPI", "Food", "NonFood", "Services"]
        if "USD" in bvar_df.columns:
            var_names.append("USD")
        if "RUONIA" in bvar_df.columns:
            var_names.append("RUONIA")
        np.random.seed(42)
        model = BVARForecaster(lags=1, lambda1=0.2, var_names=var_names)
        model.fit(bvar_df, "CPI")
        return normalize_forecast(model.forecast(horizon))

    model.fit(sa_df, TARGET_COL)
    if spec.method == "iterative":
        return iterative_forecast(model, sa_df, horizon, TARGET_COL)
    return normalize_forecast(model.forecast(horizon))


def write_model_csv(model_name: str, forecast_dates: pd.DatetimeIndex, forecast_values: np.ndarray) -> None:
    export_df = pd.DataFrame(
        {
            "Date": forecast_dates.strftime("%Y-%m-%d"),
            "Model": model_name,
            "Forecast_MoM": forecast_values,
        }
    )
    export_df = export_df.loc[export_df["Date"] >= EXPORT_START.strftime("%Y-%m-%d")].copy()
    export_df.to_csv(OUTPUT_DIR / f"{model_name}.csv", index=False)


def build_warnings(source_last_date: str, forecast_values: np.ndarray) -> list[str]:
    warnings: list[str] = []
    if pd.Timestamp(source_last_date) < CUTOFF_MONTH:
        warnings.append("source_data_older_than_round_cutoff")
    if forecast_values.size and np.allclose(forecast_values, 0.0):
        warnings.append("flat_zero_forecast")
    return warnings


def main() -> int:
    np.random.seed(42)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    sa_df = load_sa_round_data(CUTOFF_MONTH)
    macro_df, bvar_df = load_macro_round_data(CUTOFF_MONTH)

    if sa_df.empty or macro_df.empty:
        raise RuntimeError("Required March 2026 round data could not be loaded")

    last_actual = ensure_timestamp(sa_df.index.max())
    if last_actual != CUTOFF_MONTH:
        raise RuntimeError(f"Unexpected SA cutoff: {last_actual:%Y-%m-%d} != {CUTOFF_MONTH:%Y-%m-%d}")

    horizon = get_export_horizon(last_actual)
    forecast_dates = build_forecast_dates(last_actual, horizon)

    manifest: dict[str, Any] = {
        "generated_at": datetime.now().isoformat(),
        "forecast_round_label": "2026-03",
        "actual_data_cutoff": CUTOFF_MONTH.strftime("%Y-%m-%d"),
        "internal_forecast_start": ensure_timestamp(forecast_dates.min()).strftime("%Y-%m-%d"),
        "export_start": EXPORT_START.strftime("%Y-%m-%d"),
        "export_end": EXPORT_END.strftime("%Y-%m-%d"),
        "models": {},
        "failures": {},
    }

    for spec in MODEL_SPECS:
        source_last_date = determine_source_last_date(spec, sa_df, macro_df)
        try:
            forecast_values = run_model(spec, sa_df, macro_df, bvar_df, horizon)
            if len(forecast_values) != horizon:
                raise RuntimeError(f"Expected {horizon} values, got {len(forecast_values)}")
            write_model_csv(spec.name, forecast_dates, forecast_values)
            warnings = build_warnings(source_last_date, forecast_values)
            manifest["models"][spec.name] = {
                "status": "ok",
                "rows_exported": int((forecast_dates >= EXPORT_START).sum()),
                "source_last_date": source_last_date,
                "dataset": spec.dataset,
                "method": spec.method,
                "warnings": warnings,
            }
            print(f"OK  {spec.name}")
        except Exception as exc:
            manifest["failures"][spec.name] = {
                "status": "failed",
                "source_last_date": source_last_date,
                "dataset": spec.dataset,
                "method": spec.method,
                "error": str(exc),
            }
            print(f"FAIL {spec.name}: {exc}")

    with open(MANIFEST_PATH, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False)

    failed = len(manifest["failures"])
    if failed:
        print(f"Completed with {failed} model failures. See {MANIFEST_PATH}")
        return 1

    print(f"Exported {len(MODEL_SPECS)} model CSV files to {OUTPUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
