#!/usr/bin/env python3
"""Prototype a 45-component SIRENA-KBR forecast layer.

This is an experiment, not a production model. It uses the canonical KBR45
mapping, current official long regional data, robust seasonal component
forecasts, and transparent scenario overrides.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
MAPPING = ROOT / "experiments" / "kbr_45_component_mapping" / "kbr_45_component_mapping.csv"
REGION_LONG = ROOT / "data" / "external" / "micro_cpi_region_export" / "region_cpi_long.csv"
OUT_DIR = ROOT / "archive" / "results" / "kbr45_forecast_prototype_20260625"


@dataclass(frozen=True)
class ForecastConfig:
    start: pd.Timestamp
    periods: int
    seasonal_years: int = 5
    recent_months: int = 3
    seasonal_weight: float = 0.75


def month_range(start: pd.Timestamp, periods: int) -> list[pd.Timestamp]:
    return [start + pd.DateOffset(months=i) for i in range(periods)]


def load_mapping() -> pd.DataFrame:
    mapping = pd.read_csv(MAPPING)
    mapping["subcomponent_code"] = mapping["subcomponent_code"].astype(int)
    return mapping


def load_component_history(mapping: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    region = pd.read_csv(REGION_LONG, parse_dates=["date"])
    codes = set(mapping["subcomponent_code"].astype(int))

    components = region[region["item_code"].isin(codes)].copy()
    components["subcomponent_code"] = components["item_code"].astype(int)
    components["mom_pp"] = components["mom_index"] - 100.0
    panel = components.pivot_table(
        index="date",
        columns="subcomponent_code",
        values="mom_pp",
        aggfunc="last",
    ).sort_index()

    headline = (
        region[region["item_code"] == 1]
        .drop_duplicates("date", keep="last")
        .set_index("date")["mom_index"]
        .sort_index()
        - 100.0
    )
    return panel, headline


def robust_component_forecast(
    series: pd.Series,
    target: pd.Timestamp,
    config: ForecastConfig,
) -> tuple[float, str]:
    hist = series.loc[series.index < target].dropna()
    if hist.empty:
        return 0.0, "zero_no_history"

    same_month = hist[hist.index.month == target.month].tail(config.seasonal_years)
    recent = hist.tail(config.recent_months)

    if len(same_month) >= 3:
        seasonal = float(same_month.median())
        method = f"seasonal_median_{len(same_month)}y"
    elif len(same_month) > 0:
        seasonal = float(same_month.median())
        method = f"short_seasonal_median_{len(same_month)}y"
    else:
        seasonal = float(hist.tail(12).median())
        method = "rolling_12m_median"

    recent_value = float(recent.median()) if len(recent) else seasonal
    pred = config.seasonal_weight * seasonal + (1.0 - config.seasonal_weight) * recent_value

    clip_base = same_month if len(same_month) >= 5 else hist.tail(60)
    q05 = float(clip_base.quantile(0.05))
    q95 = float(clip_base.quantile(0.95))
    iqr = float(clip_base.quantile(0.75) - clip_base.quantile(0.25))
    pad = max(0.5, iqr)
    clipped = float(np.clip(pred, q05 - pad, q95 + pad))
    if clipped != pred:
        method += "_clipped"
    return clipped, method


def forecast_components(
    panel: pd.DataFrame,
    mapping: pd.DataFrame,
    config: ForecastConfig,
    cutoff: pd.Timestamp | None = None,
) -> pd.DataFrame:
    targets = month_range(config.start, config.periods)
    history = panel if cutoff is None else panel.loc[panel.index <= cutoff]
    rows = []

    for target in targets:
        for row in mapping.itertuples(index=False):
            code = int(row.subcomponent_code)
            pred, method = robust_component_forecast(history[code], target, config)
            rows.append(
                {
                    "date": target,
                    "subcomponent_code": code,
                    "external_var": row.external_var,
                    "subcomponent_name": row.subcomponent_name,
                    "parent_component": row.parent_component,
                    "scenario_tags": row.scenario_tags,
                    "baseline_mom_pp": pred,
                    "forecast_method": method,
                    "canonical_weight": row.canonical_weight,
                    "latest_region_weight": row.latest_region_weight,
                }
            )

    return pd.DataFrame(rows)


def apply_tariff_scenario(
    forecast: pd.DataFrame,
    july_index: float,
    october_index: float,
) -> pd.DataFrame:
    out = forecast.copy()
    out["scenario"] = "baseline"
    out["scenario_mom_pp"] = out["baseline_mom_pp"]
    out["override_reason"] = ""

    scenario = forecast.copy()
    scenario["scenario"] = "tariff_july100_oct110"
    scenario["scenario_mom_pp"] = scenario["baseline_mom_pp"]
    scenario["override_reason"] = ""

    july_mask = (scenario["subcomponent_code"] == 14) & (scenario["date"].dt.month == 7)
    october_mask = (scenario["subcomponent_code"] == 14) & (scenario["date"].dt.month == 10)
    scenario.loc[july_mask, "scenario_mom_pp"] = july_index - 100.0
    scenario.loc[july_mask, "override_reason"] = f"u_gkh July set to index {july_index:.1f}"
    scenario.loc[october_mask, "scenario_mom_pp"] = october_index - 100.0
    scenario.loc[october_mask, "override_reason"] = f"u_gkh October set to index {october_index:.1f}"

    return pd.concat([out, scenario], ignore_index=True)


def aggregate_headline(component_forecast: pd.DataFrame, weight_col: str) -> pd.DataFrame:
    df = component_forecast.copy()
    df["contribution_pp"] = df[weight_col] * df["scenario_mom_pp"]
    headline = (
        df.groupby(["scenario", "date"], as_index=False)
        .agg(
            headline_mom_pp=("contribution_pp", "sum"),
            component_weight_sum=(weight_col, "sum"),
        )
        .sort_values(["scenario", "date"])
    )
    headline["headline_index"] = 100.0 + headline["headline_mom_pp"]
    headline["weight_set"] = weight_col
    return headline


def backtest(
    panel: pd.DataFrame,
    headline: pd.Series,
    mapping: pd.DataFrame,
    horizons: tuple[int, ...] = (1, 2, 12),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    first_cutoff = max(panel.index.min() + pd.DateOffset(years=5), pd.Timestamp("2015-12-01"))
    cutoffs = panel.loc[(panel.index >= first_cutoff) & (panel.index < panel.index.max())].index

    for cutoff in cutoffs:
        for h in horizons:
            target = cutoff + pd.DateOffset(months=h)
            if target not in headline.index:
                continue
            config = ForecastConfig(start=target, periods=1)
            comp = forecast_components(panel, mapping, config, cutoff=cutoff)
            comp["scenario"] = "baseline"
            comp["scenario_mom_pp"] = comp["baseline_mom_pp"]
            pred = aggregate_headline(comp, "canonical_weight").iloc[0]
            actual = float(headline.loc[target])
            rows.append(
                {
                    "cutoff": cutoff,
                    "target": target,
                    "horizon": h,
                    "forecast_mom_pp": float(pred["headline_mom_pp"]),
                    "actual_mom_pp": actual,
                    "error_pp": float(pred["headline_mom_pp"]) - actual,
                }
            )

    preds = pd.DataFrame(rows)
    if preds.empty:
        return pd.DataFrame(), preds

    metrics = (
        preds.assign(abs_error=lambda df: df["error_pp"].abs(), sq_error=lambda df: df["error_pp"] ** 2)
        .groupby("horizon", as_index=False)
        .agg(
            observations=("error_pp", "size"),
            mae_pp=("abs_error", "mean"),
            rmse_pp=("sq_error", lambda s: float(np.sqrt(s.mean()))),
            bias_pp=("error_pp", "mean"),
        )
    )
    return metrics, preds


def write_report(
    config: ForecastConfig,
    mapping: pd.DataFrame,
    component_scenarios: pd.DataFrame,
    headline_latest: pd.DataFrame,
    headline_canonical: pd.DataFrame,
    metrics: pd.DataFrame,
    last_fact: pd.Timestamp,
) -> None:
    baseline = headline_latest[headline_latest["scenario"] == "baseline"].copy()
    tariff = headline_latest[headline_latest["scenario"] == "tariff_july100_oct110"].copy()
    focus_dates = baseline[baseline["date"].isin(month_range(config.start, min(3, config.periods)))]

    scenario_delta = tariff.merge(
        baseline[["date", "headline_mom_pp"]],
        on="date",
        suffixes=("_tariff", "_baseline"),
    )
    scenario_delta["delta_pp"] = (
        scenario_delta["headline_mom_pp_tariff"] - scenario_delta["headline_mom_pp_baseline"]
    )

    overrides = component_scenarios[component_scenarios["override_reason"] != ""][
        [
            "scenario",
            "date",
            "external_var",
            "subcomponent_name",
            "baseline_mom_pp",
            "scenario_mom_pp",
            "latest_region_weight",
            "override_reason",
        ]
    ]

    report = f"""# KBR45 Forecast Prototype

Date: 2026-06-25

Status: experimental forecast prototype, not production.

Last official fact in source data: `{last_fact.date()}`

## Inputs

- 45-component map:
  `experiments/kbr_45_component_mapping/kbr_45_component_mapping.csv`
- Official regional long data:
  `data/external/micro_cpi_region_export/region_cpi_long.csv`

## Method

- For each of 45 components, forecast monthly MoM p.p. by a robust blend of
  same-calendar-month median and recent median.
- Aggregate by `latest_region_weight` for the current forecast and by
  `canonical_weight` for compatibility checks.
- Apply transparent scenario overrides only after the baseline forecast.
- Weekly nowcast signals are not yet integrated into this prototype. June 2026
  fuel pressure should therefore be read from the separate weekly Laspeyres
  nowcast until a proper KBR45 weekly overlay is added.

## Current Forecast: Latest Weights

{focus_dates[["scenario", "date", "headline_index", "headline_mom_pp"]].to_markdown(index=False, floatfmt=".3f")}

## Tariff Scenario Delta: Latest Weights

{scenario_delta[["date", "headline_mom_pp_baseline", "headline_mom_pp_tariff", "delta_pp"]].to_markdown(index=False, floatfmt=".3f")}

## Scenario Overrides

{overrides.to_markdown(index=False, floatfmt=".3f")}

## Backtest Metrics

{metrics.to_markdown(index=False, floatfmt=".3f") if not metrics.empty else "No backtest observations produced."}

## Interpretation

- This prototype is useful as a transparent scenario layer for fuel, ЖКУ,
  плодоовощи and other tagged components.
- The June baseline is intentionally mechanical and does not supersede the
  existing weekly-nowcast evidence on gasoline.
- It should not replace `SubcomponentMulti`, `Micro_SM`, Huber or Ensemble until
  its backtest is compared against those production candidates.
- July tariff behavior is handled explicitly: July ЖКУ can be set to 100.0 and
  October ЖКУ to 110.0 without changing the baseline component logic.
"""
    (OUT_DIR / "kbr45_forecast_report.md").write_text(report, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="2026-06-01")
    parser.add_argument("--periods", type=int, default=7)
    parser.add_argument("--tariff-july-index", type=float, default=100.0)
    parser.add_argument("--tariff-october-index", type=float, default=110.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    config = ForecastConfig(start=pd.Timestamp(args.start), periods=args.periods)
    mapping = load_mapping()
    panel, headline = load_component_history(mapping)

    component_forecast = forecast_components(panel, mapping, config)
    component_scenarios = apply_tariff_scenario(
        component_forecast,
        july_index=args.tariff_july_index,
        october_index=args.tariff_october_index,
    )
    headline_latest = aggregate_headline(component_scenarios, "latest_region_weight")
    headline_canonical = aggregate_headline(component_scenarios, "canonical_weight")
    metrics, preds = backtest(panel, headline, mapping)

    component_scenarios.to_csv(OUT_DIR / "kbr45_component_forecast.csv", index=False, encoding="utf-8")
    headline_latest.to_csv(OUT_DIR / "kbr45_headline_forecast_latest_weights.csv", index=False, encoding="utf-8")
    headline_canonical.to_csv(
        OUT_DIR / "kbr45_headline_forecast_canonical_weights.csv",
        index=False,
        encoding="utf-8",
    )
    metrics.to_csv(OUT_DIR / "kbr45_backtest_metrics.csv", index=False, encoding="utf-8")
    preds.to_csv(OUT_DIR / "kbr45_backtest_predictions.csv", index=False, encoding="utf-8")

    write_report(
        config,
        mapping,
        component_scenarios,
        headline_latest,
        headline_canonical,
        metrics,
        last_fact=panel.index.max(),
    )

    print((OUT_DIR / "kbr45_forecast_report.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
