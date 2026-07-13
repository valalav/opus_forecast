#!/usr/bin/env python3
"""Build July-August 2026 management control-point scenarios.

This is a decision artifact, not a production forecast update.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "archive" / "results" / "july_august_control_points_20260625"

PRECOMPUTED = ROOT / "data" / "precomputed_forecasts.json"
KBR45_COMPARISON = (
    ROOT
    / "archive"
    / "results"
    / "kbr45_forecast_comparison_20260625"
    / "control_point_forecasts_wide.csv"
)


GASOLINE_REVERSION_DRAG_PP = -0.4275
GASOLINE_DIESEL_REVERSION_DRAG_PP = -0.4637


def load_production_ensemble() -> dict[str, float]:
    data = json.loads(PRECOMPUTED.read_text(encoding="utf-8"))
    dates = data["forecast_dates"]
    ensemble = data["forecasts"]["Ensemble"]
    values = {pd.Timestamp(d).strftime("%Y-%m"): 100.0 + float(v) for d, v in zip(dates, ensemble)}
    return values


def load_policy_delta() -> float:
    wide = pd.read_csv(KBR45_COMPARISON)
    baseline = wide[wide["model"] == "KBR45_baseline"].iloc[0]
    tariff = wide[wide["model"] == "KBR45_tariff_july100_oct110"].iloc[0]
    return float(tariff["2026-07"] - baseline["2026-07"])


def build_scenarios() -> tuple[pd.DataFrame, pd.DataFrame]:
    production = load_production_ensemble()
    policy_delta = load_policy_delta()

    july_prod = production["2026-07"]
    august_prod = production["2026-08"]
    july_policy = july_prod + policy_delta
    august_policy = august_prod
    july_fuel_down = july_policy + GASOLINE_REVERSION_DRAG_PP
    july_hard_down = july_policy + GASOLINE_DIESEL_REVERSION_DRAG_PP

    rows = [
        {
            "scenario_id": "production_cache",
            "scenario_name": "Production cache Ensemble",
            "july_index": july_prod,
            "august_index": august_prod,
            "july_mom_pp": july_prod - 100.0,
            "august_mom_pp": august_prod - 100.0,
            "decision_use": "Reference only unless no expert override is allowed",
            "key_assumption": "Current precomputed Ensemble, no explicit July tariff correction",
        },
        {
            "scenario_id": "policy_adjusted",
            "scenario_name": "Policy-adjusted baseline",
            "july_index": july_policy,
            "august_index": august_policy,
            "july_mom_pp": july_policy - 100.0,
            "august_mom_pp": august_policy - 100.0,
            "decision_use": "Recommended working baseline before final June fact",
            "key_assumption": "No July ЖКУ indexation; KBR45 tariff delta applied to Ensemble",
        },
        {
            "scenario_id": "policy_plus_gasoline_reversion",
            "scenario_name": "Policy-adjusted + gasoline reversion",
            "july_index": july_fuel_down,
            "august_index": august_policy,
            "july_mom_pp": july_fuel_down - 100.0,
            "august_mom_pp": august_policy - 100.0,
            "decision_use": "Lower-risk bound if July gasoline returns to early-June level",
            "key_assumption": "Policy-adjusted baseline plus direct gasoline drag -0.4275 p.p.",
        },
        {
            "scenario_id": "policy_plus_gasoline_diesel_reversion",
            "scenario_name": "Policy-adjusted + gasoline/diesel reversion",
            "july_index": july_hard_down,
            "august_index": august_policy,
            "july_mom_pp": july_hard_down - 100.0,
            "august_mom_pp": august_policy - 100.0,
            "decision_use": "Hard downside sensitivity, not central",
            "key_assumption": "Policy-adjusted baseline plus motor-fuel drag -0.4637 p.p.",
        },
    ]

    bridge = pd.DataFrame(
        [
            {
                "item": "Production Ensemble July",
                "value_pp": july_prod - 100.0,
                "source": "data/precomputed_forecasts.json",
                "comment": "Current production cache before expert policy correction",
            },
            {
                "item": "Production Ensemble August",
                "value_pp": august_prod - 100.0,
                "source": "data/precomputed_forecasts.json",
                "comment": "Current production cache",
            },
            {
                "item": "KBR45 July tariff delta",
                "value_pp": policy_delta,
                "source": "archive/results/kbr45_forecast_comparison_20260625/control_point_forecasts_wide.csv",
                "comment": "KBR45 baseline minus KBR45 July ЖКУ=100 scenario",
            },
            {
                "item": "Gasoline reversion drag",
                "value_pp": GASOLINE_REVERSION_DRAG_PP,
                "source": "archive/results/analysis_notes/2026-06-25_july_2026_gasoline_reversion_floor.md",
                "comment": "Direct July drag if gasoline returns to 2026-06-01 level",
            },
            {
                "item": "Gasoline + diesel reversion drag",
                "value_pp": GASOLINE_DIESEL_REVERSION_DRAG_PP,
                "source": "archive/results/analysis_notes/2026-06-25_july_2026_gasoline_reversion_floor.md",
                "comment": "Hard downside sensitivity",
            },
        ]
    )
    return pd.DataFrame(rows), bridge


def write_report(scenarios: pd.DataFrame, bridge: pd.DataFrame) -> None:
    central = scenarios[scenarios["scenario_id"] == "policy_adjusted"].iloc[0]
    lower = scenarios[scenarios["scenario_id"] == "policy_plus_gasoline_reversion"].iloc[0]
    hard = scenarios[scenarios["scenario_id"] == "policy_plus_gasoline_diesel_reversion"].iloc[0]

    report = f"""# July-August 2026 Control Points

Date: 2026-06-25

Status: management decision artifact, not production cache update.

## Recommendation

Use **policy-adjusted baseline** as the working July-August control point until
the official June fact and additional July weekly fuel data arrive.

- July working point: `{central["july_index"]:.3f}`.
- August working point: `{central["august_index"]:.3f}`.
- July downside if gasoline reverts: `{lower["july_index"]:.3f}`.
- Hard July downside if gasoline and diesel revert: `{hard["july_index"]:.3f}`.

## Scenario Table

{scenarios[["scenario_id", "scenario_name", "july_index", "august_index", "decision_use"]].to_markdown(index=False, floatfmt=".3f")}

## Assumption Bridge

{bridge.to_markdown(index=False, floatfmt=".4f")}

## Interpretation

- The production Ensemble remains the reference cache path: July 100.649,
  August 100.340.
- Because July 2026 regulated-tariff indexation is assumed absent, the raw
  Ensemble July point is too high for the working policy path.
- Applying the KBR45 tariff delta lowers July by about 0.248 p.p., giving a
  working July point around 100.400.
- If gasoline returns to early-June levels in July, the direct drag can push the
  July point to about 99.97. Diesel reversion makes the hard downside about
  99.94.
- August should not mechanically inherit the July fuel downside. Keep August at
  the policy-adjusted baseline until new weekly July data appear.

## Operational Rule

For current discussion, carry three numbers:

1. Production reference: July 100.65, August 100.34.
2. Working policy-adjusted baseline: July 100.40, August 100.34.
3. Downside fuel-risk case: July 99.97, August 100.34.
"""
    (OUT_DIR / "july_august_control_points_report.md").write_text(report, encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    scenarios, bridge = build_scenarios()
    scenarios.to_csv(OUT_DIR / "july_august_control_points.csv", index=False, encoding="utf-8")
    bridge.to_csv(OUT_DIR / "assumption_bridge.csv", index=False, encoding="utf-8")
    write_report(scenarios, bridge)
    print((OUT_DIR / "july_august_control_points_report.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
