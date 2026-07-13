#!/usr/bin/env python3
"""Independent verification of April 2026 weekly product-driver calculations.

Read-only: prints JSON to stdout and does not modify project files.
"""
from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[4]
W = {"food": 0.3986, "nonfood": 0.3638, "services": 0.2376}


def norm(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value).strip().lower().replace("ё", "е"))


def parse_num(value: Any) -> float:
    try:
        return float(str(value).strip().replace("\xa0", "").replace(",", "."))
    except Exception:
        return np.nan


def classify_component(value: Any) -> str | None:
    text = norm(value)
    if "непродовольств" in text:
        return "nonfood"
    if "продовольств" in text:
        return "food"
    if "услуг" in text:
        return "services"
    return None


def load_fresh_weekly() -> pd.DataFrame:
    rows = []
    path = ROOT / "data" / "Сравнение еженедельных цен_01.csv"
    with path.open(encoding="utf-8-sig") as file:
        reader = csv.DictReader(file, delimiter=";")
        for raw_row in reader:
            row = {str(key).strip(): val for key, val in raw_row.items()}
            date = pd.to_datetime(row.get("Name", ""), format="%d.%m.%Y", errors="coerce")
            name = str(row.get("Наименование", "")).strip()
            key = str(row.get("№", "")).strip() or name
            price = parse_num(row.get("Средние цены, рублей", ""))
            component = classify_component(row.get("Справка_нед.Компоненты", row.get("Компонент", "")))
            if pd.notna(date) and name and component and np.isfinite(price):
                rows.append({"date": date, "key": key, "name": name, "component": component, "price": price})
    return pd.DataFrame(rows).drop_duplicates(["date", "key", "name", "price"])


def main() -> None:
    df = load_fresh_weekly()
    start = pd.Timestamp("2026-03-30")
    end = pd.Timestamp("2026-04-27")
    p0 = df[df["date"] == start].set_index("key")
    p1 = df[df["date"] == end].set_index("key")
    common_keys = sorted(set(p0.index) & set(p1.index))

    counts = {component: 0 for component in W}
    for key in common_keys:
        component = p1.loc[key, "component"]
        if isinstance(component, pd.Series):
            component = component.iloc[0]
        if component in counts:
            counts[component] += 1

    records = []
    for key in common_keys:
        component = p1.loc[key, "component"]
        name = p1.loc[key, "name"]
        price_start = p0.loc[key, "price"]
        price_end = p1.loc[key, "price"]
        if isinstance(component, pd.Series):
            component = component.iloc[0]
            name = name.iloc[0]
            price_start = price_start.iloc[0]
            price_end = price_end.iloc[0]
        if component in W and price_start > 0 and price_end > 0 and counts[component] > 0:
            change_pct = (price_end / price_start - 1) * 100
            contribution = change_pct * W[component] / counts[component]
            records.append(
                {
                    "product": str(name),
                    "component": str(component),
                    "price_start": float(price_start),
                    "price_end": float(price_end),
                    "change_pct": float(change_pct),
                    "approx_contribution_pp": float(contribution),
                }
            )

    product_table = pd.DataFrame(records).sort_values("approx_contribution_pp")
    result = {
        "source": "data/Сравнение еженедельных цен_01.csv",
        "period": "2026-03-30 to 2026-04-27",
        "counts": counts,
        "matched_items": int(len(product_table)),
        "approx_sum_contribution_pp": float(product_table["approx_contribution_pp"].sum()),
        "top_negative": product_table.head(15).to_dict(orient="records"),
        "top_positive": product_table.tail(12).sort_values("approx_contribution_pp", ascending=False).to_dict(orient="records"),
        "mismatches": [],
    }

    expected = {
        "Огурцы свежие, кг": (-22.63, -0.2050),
        "Помидоры свежие, кг": (-17.87, -0.1619),
        "Яйца куриные, 10 шт.": (-7.57, -0.0685),
        "Капуста белокочанная свежая, кг": (15.87, 0.1437),
    }
    for product, (expected_pct, expected_contribution) in expected.items():
        rows = product_table[product_table["product"] == product]
        if rows.empty:
            result["mismatches"].append(f"Missing expected product: {product}")
            continue
        row = rows.iloc[0]
        if round(float(row["change_pct"]), 2) != expected_pct:
            result["mismatches"].append(f"{product} change mismatch: {row['change_pct']} vs {expected_pct}")
        if round(float(row["approx_contribution_pp"]), 4) != expected_contribution:
            result["mismatches"].append(f"{product} contribution mismatch: {row['approx_contribution_pp']} vs {expected_contribution}")

    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
