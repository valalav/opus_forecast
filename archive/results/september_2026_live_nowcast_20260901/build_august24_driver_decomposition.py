#!/usr/bin/env python3
"""Build the 24 August weekly decomposition used for the September protocol."""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from sirena.data.weekly_bridge import load_semicolon_weekly_prices  # noqa: E402
from sirena.data.weekly_loader import COMPONENT_WEIGHTS  # noqa: E402


OUTPUT = Path(__file__).with_name("august24_driver_decomposition.csv")
WEEK_DATE = pd.Timestamp("2026-08-24")
PRODUCE_TERMS = (
    "картоф",
    "капуст",
    "лук реп",
    "свёкл",
    "морков",
    "яблок",
    "помидор",
    "огур",
    "банан",
    "апельс",
    "груш",
    "виноград",
)


def main() -> None:
    weekly = load_semicolon_weekly_prices(
        ROOT / "data" / "Сравнение еженедельных цен_01.csv"
    )
    week = weekly[weekly["date"] == WEEK_DATE].copy()
    if week.empty:
        raise RuntimeError("Weekly source does not contain 2026-08-24")

    weights = {
        component: weight / sum(COMPONENT_WEIGHTS.values())
        for component, weight in COMPONENT_WEIGHTS.items()
    }
    rows: list[dict[str, str | int | float]] = []
    weekly_total = 0.0

    for component, weight in weights.items():
        component_rows = week[
            (week["component"] == component) & week["change_index"].notna()
        ].copy()
        component_mom = float(component_rows["change_index"].mean() - 100.0)
        contribution = component_mom * weight
        weekly_total += contribution
        rows.append(
            {
                "level": "component",
                "name": component,
                "component": component,
                "wow_pp": component_mom,
                "approx_headline_contribution_pp": contribution,
                "n_items": len(component_rows),
            }
        )

    rows.insert(
        0,
        {
            "level": "summary",
            "name": "weekly_bridge_total",
            "component": "all",
            "wow_pp": weekly_total,
            "approx_headline_contribution_pp": weekly_total,
            "n_items": len(week),
        },
    )

    food = week[(week["component"] == "food") & week["change_index"].notna()].copy()
    food_count = len(food)
    food["wow_pp"] = food["change_index"] - 100.0
    food["approx_headline_contribution_pp"] = (
        food["wow_pp"] * weights["food"] / food_count
    )
    produce = food[
        food["product_name"]
        .str.casefold()
        .map(lambda name: any(term in name for term in PRODUCE_TERMS))
    ].copy()

    rows.append(
        {
            "level": "summary",
            "name": "selected_fruit_and_vegetables",
            "component": "food",
            "wow_pp": float(produce["wow_pp"].sum()),
            "approx_headline_contribution_pp": float(
                produce["approx_headline_contribution_pp"].sum()
            ),
            "n_items": len(produce),
        }
    )
    for _, item in produce.sort_values("approx_headline_contribution_pp").iterrows():
        rows.append(
            {
                "level": "item",
                "name": str(item["product_name"]),
                "component": "food",
                "wow_pp": float(item["wow_pp"]),
                "approx_headline_contribution_pp": float(
                    item["approx_headline_contribution_pp"]
                ),
                "n_items": 1,
            }
        )

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(
            output,
            fieldnames=[
                "level",
                "name",
                "component",
                "wow_pp",
                "approx_headline_contribution_pp",
                "n_items",
            ],
            lineterminator="\n",
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    **row,
                    "wow_pp": f"{float(row['wow_pp']):.6f}",
                    "approx_headline_contribution_pp": (
                        f"{float(row['approx_headline_contribution_pp']):.6f}"
                    ),
                }
            )

    print(f"Wrote {OUTPUT}")
    print(f"Weekly bridge total: {weekly_total:+.6f} pp")
    print(
        "Selected fruit and vegetables: "
        f"{produce['approx_headline_contribution_pp'].sum():+.6f} pp"
    )


if __name__ == "__main__":
    main()
