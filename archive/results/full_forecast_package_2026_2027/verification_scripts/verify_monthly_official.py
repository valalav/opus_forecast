#!/usr/bin/env python3
"""Independent verification of official monthly forecast-vs-fact calculations.

Read-only: prints JSON to stdout and does not modify project files.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[4]
WEIGHTS = {"Prod": 0.3986, "Nonprod": 0.3638, "Serv": 0.2376}
FORECAST_INDEX = 100.45


def trim_mean_10(values: pd.Series) -> float:
    arr = np.sort(values.dropna().to_numpy(dtype=float))
    k = int(np.floor(len(arr) * 0.1))
    trimmed = arr[k : len(arr) - k] if len(arr) - 2 * k > 0 else arr
    return float(np.mean(trimmed))


def main() -> None:
    df = pd.read_csv(ROOT / "data" / "inflation_data.csv", sep=";", decimal=",", encoding="utf-8-sig")
    df["Date"] = pd.to_datetime(df["Date"], format="%d.%m.%Y")
    for col in ["mom", "Prod", "Nonprod", "Serv"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    april_2026 = df[df["Date"].dt.strftime("%Y-%m") == "2026-04"].iloc[0]
    forecast_mom = FORECAST_INDEX - 100
    fact_mom = float(april_2026["mom"] - 100)
    error_pp = forecast_mom - fact_mom

    contributions = {}
    for component, weight in WEIGHTS.items():
        component_mom = float(april_2026[component] - 100)
        contributions[component] = {
            "mom_pct": component_mom,
            "weight": weight,
            "contribution_pp": component_mom * weight,
        }
    component_sum = float(sum(item["contribution_pp"] for item in contributions.values()))

    aprils = df[df["Date"].dt.month == 4].copy().sort_values(by="Date")
    previous_aprils = aprils[aprils["Date"].dt.year < 2026]
    april_mom_pct = previous_aprils["mom"] - 100
    q1 = float(april_mom_pct.quantile(0.25))
    q3 = float(april_mom_pct.quantile(0.75))
    iqr = q3 - q1
    lower_fence = q1 - 1.5 * iqr
    upper_fence = q3 + 1.5 * iqr
    median = float(april_mom_pct.median())
    mad = float(np.median(np.abs(april_mom_pct - median)))
    robust_z = float((fact_mom - median) / (1.4826 * mad)) if mad else None

    result = {
        "source": "data/inflation_data.csv",
        "forecast_index": FORECAST_INDEX,
        "forecast_mom_pct": forecast_mom,
        "fact_index": float(april_2026["mom"]),
        "fact_mom_pct": fact_mom,
        "error_pp": error_pp,
        "abs_error_pp": abs(error_pp),
        "excess_over_0_5_pp": abs(error_pp) - 0.5,
        "kpi_0_5_pass": abs(error_pp) <= 0.5,
        "component_contributions": contributions,
        "component_sum_pp": component_sum,
        "component_residual_vs_fact_pp": fact_mom - component_sum,
        "april_history_pre2026": {
            "n": int(len(previous_aprils)),
            "mean_pct": float(april_mom_pct.mean()),
            "median_pct": median,
            "trim10_pct": trim_mean_10(april_mom_pct),
            "std_pct": float(april_mom_pct.std(ddof=1)),
            "q1_pct": q1,
            "q3_pct": q3,
            "iqr_pct": iqr,
            "lower_fence_pct": lower_fence,
            "upper_fence_pct": upper_fence,
            "outlier_years": previous_aprils.loc[(april_mom_pct < lower_fence) | (april_mom_pct > upper_fence), "Date"].dt.year.astype(int).tolist(),
            "min_pct": float(april_mom_pct.min()),
            "min_year": int(previous_aprils.loc[april_mom_pct.idxmin(), "Date"].year),
            "deflation_count_before_2026": int((previous_aprils["mom"] < 100).sum()),
            "april_2026_is_first_deflation": bool((previous_aprils["mom"] < 100).sum() == 0 and fact_mom < 0),
            "robust_z": robust_z,
        },
        "mismatches": [],
    }

    checks = [
        (round(error_pp, 2) == 0.73, "error_pp should round to 0.73"),
        (round(component_sum, 3) == -0.279, "component sum should round to -0.279"),
        (int((previous_aprils["mom"] < 100).sum()) == 0, "no previous April deflation"),
        (round(float(april_mom_pct.mean()), 3) == 0.922, "April mean should round to 0.922"),
        (round(float(april_mom_pct.median()), 3) == 0.675, "April median should round to 0.675"),
    ]
    result["mismatches"] = [message for ok, message in checks if not ok]
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
