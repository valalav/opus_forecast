#!/usr/bin/env python3
"""Build a compact KBR45 comparison package for control-point discussion."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "archive" / "results" / "kbr45_forecast_comparison_20260625"

PRECOMPUTED = ROOT / "data" / "precomputed_forecasts.json"
KBR45_HEADLINE = (
    ROOT
    / "archive"
    / "results"
    / "kbr45_forecast_prototype_20260625"
    / "kbr45_headline_forecast_latest_weights.csv"
)
KBR45_COMPONENTS = (
    ROOT
    / "archive"
    / "results"
    / "kbr45_forecast_prototype_20260625"
    / "kbr45_component_forecast.csv"
)
WEEKLY_CONTRIB = (
    ROOT
    / "archive"
    / "results"
    / "weekly_laspeyres_nowcast_20260625"
    / "weekly_laspeyres_contributions.csv"
)


CONTROL_DATES = pd.to_datetime(["2026-06-01", "2026-07-01", "2026-08-01"])
FOCUS_MODELS = [
    "Ensemble",
    "Nowcast",
    "Huber",
    "RidgeShockDummies",
    "NGBoostShock",
    "Prophet",
    "SubcomponentMulti",
    "Micro_SM",
]


def load_precomputed() -> pd.DataFrame:
    data = json.loads(PRECOMPUTED.read_text(encoding="utf-8"))
    rows = []
    for model, values in data.get("forecasts", {}).items():
        if not isinstance(values, list):
            continue
        for date, value in zip(data["forecast_dates"], values):
            rows.append(
                {
                    "source": "production_cache",
                    "model": model,
                    "date": pd.Timestamp(date),
                    "mom_pp": value,
                    "index": None if value is None else 100.0 + float(value),
                    "status": "production_or_existing_diagnostic",
                }
            )
    return pd.DataFrame(rows)


def load_kbr45() -> tuple[pd.DataFrame, pd.DataFrame]:
    headline = pd.read_csv(KBR45_HEADLINE, parse_dates=["date"])
    components = pd.read_csv(KBR45_COMPONENTS, parse_dates=["date"])
    rows = []
    for row in headline.itertuples(index=False):
        rows.append(
            {
                "source": "kbr45_prototype",
                "model": f"KBR45_{row.scenario}",
                "date": row.date,
                "mom_pp": row.headline_mom_pp,
                "index": row.headline_index,
                "status": "experimental_not_production",
            }
        )
    return pd.DataFrame(rows), components


def weekly_fuel_overlay(kbr45: pd.DataFrame, components: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    weekly = pd.read_csv(WEEKLY_CONTRIB, parse_dates=["date"])
    bridge = weekly[weekly["period_type"] == "month_end_bridge"].copy()
    latest_date = bridge["date"].max()
    latest = bridge[bridge["date"] == latest_date].copy()

    fuel_mask = latest["product_name"].astype(str).str.lower().str.contains(
        "бензин|дизель|газовое моторное|топлив",
        regex=True,
        na=False,
    )
    fuel = latest[fuel_mask].dropna(subset=["headline_contribution_pp"]).copy()
    weekly_fuel_contribution = float(fuel["headline_contribution_pp"].sum())
    weekly_matched_contribution = float(latest["headline_contribution_pp"].sum())
    weekly_fuel_weight = float(fuel["weight"].sum())

    june = pd.Timestamp("2026-06-01")
    kbr45_fuel = components[
        (components["scenario"] == "baseline")
        & (components["date"] == june)
        & (components["external_var"] == "n_topl")
    ].iloc[0]
    kbr45_fuel_contribution = float(
        kbr45_fuel["scenario_mom_pp"] * kbr45_fuel["latest_region_weight"]
    )
    fuel_delta = weekly_fuel_contribution - kbr45_fuel_contribution

    overlay_rows = []
    for scenario in ["KBR45_baseline", "KBR45_tariff_july100_oct110"]:
        base = kbr45[kbr45["model"] == scenario].copy()
        base["model"] = scenario + "_weekly_fuel_overlay"
        base["source"] = "kbr45_plus_weekly_laspeyres"
        base["status"] = "experimental_overlay_not_production"
        june_mask = base["date"] == june
        base.loc[june_mask, "mom_pp"] = base.loc[june_mask, "mom_pp"] + fuel_delta
        base.loc[june_mask, "index"] = 100.0 + base.loc[june_mask, "mom_pp"]
        overlay_rows.append(base)

    overlay = pd.concat(overlay_rows, ignore_index=True)
    details = pd.DataFrame(
        [
            {
                "latest_weekly_date": latest_date,
                "weekly_matched_contribution_pp": weekly_matched_contribution,
                "weekly_fuel_contribution_pp": weekly_fuel_contribution,
                "weekly_fuel_weight": weekly_fuel_weight,
                "kbr45_june_fuel_contribution_pp": kbr45_fuel_contribution,
                "fuel_overlay_delta_pp": fuel_delta,
                "interpretation": (
                    "June-only overlay replaces KBR45 mechanical fuel contribution "
                    "with observed weekly month-end fuel pressure."
                ),
            }
        ]
    )
    fuel.to_csv(OUT_DIR / "weekly_fuel_overlay_items.csv", index=False, encoding="utf-8")
    return overlay, details


def pivot_control_points(df: pd.DataFrame) -> pd.DataFrame:
    focus = df[df["date"].isin(CONTROL_DATES)].copy()
    focus["month"] = focus["date"].dt.strftime("%Y-%m")
    pivot = (
        focus.pivot_table(
            index=["source", "model", "status"],
            columns="month",
            values="index",
            aggfunc="last",
        )
        .reset_index()
        .sort_values(["source", "model"])
    )
    return pivot


def write_report(control: pd.DataFrame, overlay_details: pd.DataFrame) -> None:
    detail = overlay_details.iloc[0]
    selected = control[
        control["model"].isin(
            [
                "Ensemble",
                "Nowcast",
                "Micro_SM",
                "KBR45_baseline",
                "KBR45_tariff_july100_oct110",
                "KBR45_baseline_weekly_fuel_overlay",
                "KBR45_tariff_july100_oct110_weekly_fuel_overlay",
            ]
        )
    ].copy()

    report = f"""# KBR45 Forecast Comparison Package

Date: 2026-06-25

Status: diagnostic comparison, not a production forecast update.

## What This Answers

This package places the new KBR45 scenario layer next to the existing production
forecast cache and the June weekly gasoline signal. It is the next step after
building the 45-component map and prototype.

## Control Points

{selected.to_markdown(index=False, floatfmt=".3f")}

## Weekly Fuel Overlay

- Latest weekly observation used: `{pd.Timestamp(detail["latest_weekly_date"]).date()}`
- Weekly matched-basket contribution: `{detail["weekly_matched_contribution_pp"]:.3f}` p.p.
- Weekly fuel contribution: `{detail["weekly_fuel_contribution_pp"]:.3f}` p.p.
- KBR45 mechanical June fuel contribution: `{detail["kbr45_june_fuel_contribution_pp"]:.3f}` p.p.
- June fuel overlay delta: `{detail["fuel_overlay_delta_pp"]:.3f}` p.p.

## Reading The Numbers

- Production `Ensemble` remains the current cache baseline.
- `Nowcast` is available only for June in the cache and is higher because it
  already reflects short-term weekly pressure.
- Mechanical KBR45 baseline is too calm for June unless the weekly fuel overlay
  is added.
- The tariff-shift result is the clean policy insight: setting July ЖКУ to 100
  lowers July KBR45 by about 0.25 p.p.; moving the 110 indexation to October
  pushes the October point by about 0.89 p.p. in the prototype.

## Next Operational Step

Build a proper challenger backtest against `SubcomponentMulti`, `Micro_SM`,
Huber and Ensemble, then decide whether KBR45 should remain a diagnostic layer
or be registered as a model.
"""
    (OUT_DIR / "kbr45_forecast_comparison_report.md").write_text(report, encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    precomputed = load_precomputed()
    kbr45, components = load_kbr45()
    overlay, overlay_details = weekly_fuel_overlay(kbr45, components)

    combined = pd.concat([precomputed, kbr45, overlay], ignore_index=True)
    combined_focus = combined[
        (combined["date"].isin(CONTROL_DATES))
        & (
            combined["model"].isin(FOCUS_MODELS)
            | combined["model"].str.startswith("KBR45_")
        )
    ].copy()
    control = pivot_control_points(combined_focus)

    combined_focus.to_csv(OUT_DIR / "control_point_forecasts_long.csv", index=False, encoding="utf-8")
    control.to_csv(OUT_DIR / "control_point_forecasts_wide.csv", index=False, encoding="utf-8")
    overlay_details.to_csv(OUT_DIR / "weekly_fuel_overlay_summary.csv", index=False, encoding="utf-8")
    write_report(control, overlay_details)

    print((OUT_DIR / "kbr45_forecast_comparison_report.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
