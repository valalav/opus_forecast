#!/usr/bin/env python3
"""Prototype weighted weekly nowcast using KBR item weights.

This is an experiment, not a production model.  It uses the fresh weekly
semicolon file and matches weekly item names to `data/micro_sprav.csv`.

Outputs:
- weekly_laspeyres_nowcast.csv
- weekly_laspeyres_contributions.csv
- weekly_laspeyres_matches.csv
- weekly_laspeyres_summary.md
"""

from __future__ import annotations

import argparse
import re
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from sirena.data.weekly_bridge import load_semicolon_weekly_prices
from sirena.data.weekly_loader import COMPONENT_WEIGHTS


DEFAULT_OUT_DIR = PROJECT_ROOT / "archive/results/weekly_laspeyres_nowcast_20260625"
DEFAULT_WEEKLY_PATH = PROJECT_ROOT / "data/Сравнение еженедельных цен_01.csv"
DEFAULT_SPRAV_PATH = PROJECT_ROOT / "data/micro_sprav.csv"

MANUAL_ALIASES = {
    "бензин автомобильный марки аи 98 и выше": [132],
    "метамизол натрия 10 таблеток": [370],
    "нимесулид 10 таблеток": [406],
    "активированный уголь 10 таблеток": [851],
    "кроссовые туфли для детей": [319],
    "услуги по снабжению электроэнергией": [744, 745, 746, 747],
    "плата за жилье в домах государственного и муниципального жилищных фондов общей площади": [400],
    "восстановление зуба пломбой": [342],
}


def normalize_name(value: Any) -> str:
    text = str(value or "").lower().replace("ё", "е")
    text = re.sub(r"\([^)]*\)", " ", text)
    text = re.sub(
        r"\b(кг|л|м2|м3|шт|штук|услуга|поездка|пара|комплект|месяц|упаковка)\b",
        " ",
        text,
    )
    text = re.sub(r"[^a-zа-я0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def map_component_to_weekly(component: Any) -> str | None:
    text = str(component or "").lower()
    if "непродовольств" in text:
        return "nonfood"
    if "продовольств" in text:
        return "food"
    if "услуг" in text:
        return "services"
    return None


def load_weights(path: Path) -> pd.DataFrame:
    sprav = pd.read_csv(path, sep=";", decimal=",", encoding="utf-8-sig")
    sprav["component"] = sprav["Компонент"].map(map_component_to_weekly)
    sprav["norm_name"] = sprav["Товар"].map(normalize_name)
    sprav["weight"] = pd.to_numeric(sprav["Weight"], errors="coerce")
    sprav = sprav.dropna(subset=["component", "norm_name", "weight"])
    sprav = sprav[sprav["norm_name"] != ""].copy()
    sprav = sprav.sort_values("weight", ascending=False).drop_duplicates(
        subset=["component", "norm_name"], keep="first"
    )
    alias_rows = []
    by_code = sprav.set_index("Item_code", drop=False)
    for alias_name, item_codes in MANUAL_ALIASES.items():
        rows = by_code.loc[[code for code in item_codes if code in by_code.index]]
        if rows.empty:
            continue
        component = rows["component"].dropna().iloc[0]
        if rows["component"].nunique(dropna=True) != 1:
            continue
        alias_rows.append(
            {
                "Item_code": "+".join(str(int(code)) for code in rows["Item_code"]),
                "Товар": " + ".join(str(name) for name in rows["Товар"]),
                "Компонент": rows["Компонент"].dropna().iloc[0],
                "Субкомпонент": "manual_alias",
                "Weight": float(rows["weight"].sum()),
                "component": component,
                "norm_name": alias_name,
                "weight": float(rows["weight"].sum()),
            }
        )
    if alias_rows:
        sprav = pd.concat([sprav, pd.DataFrame(alias_rows)], ignore_index=True)
    return sprav


def best_fuzzy_match(
    name: str,
    component: str,
    sprav: pd.DataFrame,
    threshold: float,
    min_margin: float,
) -> tuple[pd.Series | None, float, str]:
    candidates = sprav[sprav["component"] == component]
    best_row = None
    best_score = -1.0
    second_score = -1.0
    for _, row in candidates.iterrows():
        score = SequenceMatcher(None, name, str(row["norm_name"])).ratio()
        if score > best_score:
            second_score = best_score
            best_score = score
            best_row = row
        elif score > second_score:
            second_score = score
    if best_row is not None and best_score >= threshold and best_score - second_score >= min_margin:
        return best_row, float(best_score), "fuzzy"
    return None, float(best_score), "unmatched"


def build_item_matches(
    weekly: pd.DataFrame,
    sprav: pd.DataFrame,
    fuzzy_threshold: float = 0.88,
    min_margin: float = 0.03,
) -> pd.DataFrame:
    items = (
        weekly[["item_key", "product_name", "component"]]
        .drop_duplicates(subset=["item_key"])
        .copy()
    )
    items["norm_name"] = items["product_name"].map(normalize_name)

    exact_lookup = {
        (str(row["component"]), str(row["norm_name"])): row
        for _, row in sprav.iterrows()
    }

    rows: list[dict[str, Any]] = []
    for row in items.itertuples(index=False):
        key = (str(row.component), str(row.norm_name))
        matched = exact_lookup.get(key)
        score = 1.0 if matched is not None else np.nan
        method = "exact" if matched is not None else "unmatched"
        if matched is None:
            matched, score, method = best_fuzzy_match(
                str(row.norm_name), str(row.component), sprav, fuzzy_threshold, min_margin
            )

        if matched is None:
            rows.append(
                {
                    "item_key": row.item_key,
                    "product_name": row.product_name,
                    "component": row.component,
                    "norm_name": row.norm_name,
                    "matched": False,
                    "match_method": "unmatched",
                    "match_score": score,
                    "item_code": np.nan,
                    "sprav_name": "",
                    "weight": np.nan,
                }
            )
        else:
            rows.append(
                {
                    "item_key": row.item_key,
                    "product_name": row.product_name,
                    "component": row.component,
                    "norm_name": row.norm_name,
                    "matched": True,
                    "match_method": method,
                    "match_score": score,
                    "item_code": matched["Item_code"],
                    "sprav_name": matched["Товар"],
                    "weight": matched["weight"],
                }
            )

    return pd.DataFrame(rows)


def weighted_index_for_period(data: pd.DataFrame, index_col: str) -> dict[str, Any]:
    matched = data[data["matched"] & data[index_col].notna()].copy()
    if matched.empty:
        return {
            "matched_items": 0,
            "matched_weight": 0.0,
            "observed_basket_index": np.nan,
            "headline_partial_index": np.nan,
            "component_scaled_index": np.nan,
        }

    matched["weight"] = pd.to_numeric(matched["weight"], errors="coerce")
    matched = matched.dropna(subset=["weight"])
    total_weight = float(matched["weight"].sum())
    observed_basket_index = float((matched[index_col] * matched["weight"]).sum() / total_weight)
    headline_partial_index = float(100.0 + ((matched[index_col] - 100.0) * matched["weight"]).sum())

    component_scaled_index = 0.0
    component_rows: dict[str, dict[str, Any]] = {}
    for component, component_weight in COMPONENT_WEIGHTS.items():
        comp = matched[matched["component"] == component].copy()
        comp_weight = float(comp["weight"].sum())
        if comp_weight > 0:
            comp_index = float((comp[index_col] * comp["weight"]).sum() / comp_weight)
        else:
            comp_index = 100.0
        component_scaled_index += comp_index * component_weight
        component_rows[component] = {
            "index": comp_index,
            "mom_pp": comp_index - 100.0,
            "matched_items": int(len(comp)),
            "matched_weight": comp_weight,
        }

    return {
        "matched_items": int(len(matched)),
        "matched_weight": total_weight,
        "observed_basket_index": observed_basket_index,
        "headline_partial_index": headline_partial_index,
        "component_scaled_index": component_scaled_index,
        "components": component_rows,
    }


def weekly_chain(weekly: pd.DataFrame, matches: pd.DataFrame, target_month: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    month_start = pd.Timestamp(target_month).to_period("M").to_timestamp()
    month_end = month_start + pd.DateOffset(months=1)
    month = weekly[(weekly["date"] >= month_start) & (weekly["date"] < month_end)].copy()
    month = month.merge(matches, on=["item_key", "product_name", "component"], how="left")
    month["change_index"] = pd.to_numeric(month["change_index"], errors="coerce")
    month["item_mom_pp"] = month["change_index"] - 100.0
    month["headline_contribution_pp"] = np.where(
        month["matched"],
        month["item_mom_pp"] * pd.to_numeric(month["weight"], errors="coerce"),
        np.nan,
    )

    rows = []
    cumulative_observed = 1.0
    cumulative_partial = 1.0
    cumulative_scaled = 1.0
    for date, data in month.groupby("date", sort=True):
        stats = weighted_index_for_period(data, "change_index")
        cumulative_observed *= stats["observed_basket_index"] / 100.0
        cumulative_partial *= stats["headline_partial_index"] / 100.0
        cumulative_scaled *= stats["component_scaled_index"] / 100.0
        rows.append(
            {
                "date": pd.Timestamp(date).date().isoformat(),
                "matched_items": stats["matched_items"],
                "matched_weight": stats["matched_weight"],
                "observed_basket_index": stats["observed_basket_index"],
                "observed_basket_mom_pp": stats["observed_basket_index"] - 100.0,
                "headline_partial_index": stats["headline_partial_index"],
                "headline_partial_mom_pp": stats["headline_partial_index"] - 100.0,
                "component_scaled_index": stats["component_scaled_index"],
                "component_scaled_mom_pp": stats["component_scaled_index"] - 100.0,
                "cum_observed_basket_mom_pp": (cumulative_observed - 1.0) * 100.0,
                "cum_headline_partial_mom_pp": (cumulative_partial - 1.0) * 100.0,
                "cum_component_scaled_mom_pp": (cumulative_scaled - 1.0) * 100.0,
            }
        )
    return pd.DataFrame(rows), month


def month_end_bridge(weekly: pd.DataFrame, matches: pd.DataFrame, target_month: str) -> tuple[dict[str, Any], pd.DataFrame]:
    month_start = pd.Timestamp(target_month).to_period("M").to_timestamp()
    month_end = month_start + pd.DateOffset(months=1)
    previous = weekly[weekly["date"] < month_start]
    current = weekly[(weekly["date"] >= month_start) & (weekly["date"] < month_end)]
    if previous.empty or current.empty:
        return {}, pd.DataFrame()
    base_date = previous["date"].max()
    end_date = current["date"].max()

    base = weekly[weekly["date"] == base_date][["item_key", "price"]].rename(columns={"price": "price_base"})
    end = weekly[weekly["date"] == end_date][["item_key", "product_name", "component", "price"]].rename(columns={"price": "price_end"})
    data = end.merge(base, on="item_key", how="inner")
    data = data.merge(matches, on=["item_key", "product_name", "component"], how="left")
    data = data[(data["price_base"] > 0) & (data["price_end"] > 0)].copy()
    data["price_index"] = data["price_end"] / data["price_base"] * 100.0
    data["item_mom_pp"] = data["price_index"] - 100.0
    data["headline_contribution_pp"] = np.where(
        data["matched"],
        data["item_mom_pp"] * pd.to_numeric(data["weight"], errors="coerce"),
        np.nan,
    )
    stats = weighted_index_for_period(data, "price_index")
    stats["base_date"] = pd.Timestamp(base_date).date().isoformat()
    stats["end_date"] = pd.Timestamp(end_date).date().isoformat()
    return stats, data


def write_summary(
    out_dir: Path,
    target_month: str,
    weekly_rows: pd.DataFrame,
    month_details: pd.DataFrame,
    month_end_stats: dict[str, Any],
) -> None:
    last = weekly_rows.iloc[-1] if not weekly_rows.empty else None
    month_items = month_details.drop_duplicates(subset=["item_key"])
    matched = month_items[month_items["matched"]]
    lines = [
        "# Weekly Laspeyres Nowcast Prototype",
        "",
        f"Target month: `{target_month}`",
        "",
        "## Coverage",
        "",
        f"- Weekly unique items: `{len(month_items)}`",
        f"- Matched items: `{len(matched)}`",
        f"- Matched item weight: `{matched['weight'].sum():.4f}`",
        f"- Exact matches: `{int((month_items['match_method'] == 'exact').sum())}`",
        f"- Fuzzy matches: `{int((month_items['match_method'] == 'fuzzy').sum())}`",
        "",
        f"## {pd.Timestamp(target_month).strftime('%B %Y')} Signal",
        "",
    ]
    if last is not None:
        lines.extend(
            [
                f"- Last weekly date: `{last['date']}`",
                f"- Cumulative observed matched-basket signal: `{last['cum_observed_basket_mom_pp']:+.3f}` pp",
                f"- Cumulative headline partial contribution: `{last['cum_headline_partial_mom_pp']:+.3f}` pp",
                f"- Cumulative component-scaled signal: `{last['cum_component_scaled_mom_pp']:+.3f}` pp",
            ]
        )
    if month_end_stats:
        lines.extend(
            [
                "",
                "## Month-End Price Bridge",
                "",
                f"- Window: `{month_end_stats['base_date']}` -> `{month_end_stats['end_date']}`",
                f"- Observed matched-basket price index: `{month_end_stats['observed_basket_index']:.3f}`",
                f"- Headline partial index: `{month_end_stats['headline_partial_index']:.3f}`",
                f"- Component-scaled index: `{month_end_stats['component_scaled_index']:.3f}`",
            ]
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "`headline_partial_index` treats all unmatched CPI basket weights as no change.",
            "`component_scaled_index` treats matched weekly items as representative within each broad component.",
            "Both are diagnostics, not official CPI facts.",
        ]
    )
    (out_dir / "weekly_laspeyres_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--month", default="2026-06")
    parser.add_argument("--weekly-path", default=str(DEFAULT_WEEKLY_PATH))
    parser.add_argument("--sprav-path", default=str(DEFAULT_SPRAV_PATH))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--fuzzy-threshold", type=float, default=0.88)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    weekly = load_semicolon_weekly_prices(args.weekly_path)
    sprav = load_weights(Path(args.sprav_path))
    matches = build_item_matches(weekly, sprav, fuzzy_threshold=args.fuzzy_threshold)
    weekly_rows, weekly_details = weekly_chain(weekly, matches, args.month)
    month_end_stats, month_end_details = month_end_bridge(weekly, matches, args.month)

    matches.to_csv(out_dir / "weekly_laspeyres_matches.csv", index=False, encoding="utf-8")
    weekly_rows.to_csv(out_dir / "weekly_laspeyres_nowcast.csv", index=False, encoding="utf-8")

    if not month_end_details.empty and month_end_stats:
        month_end_details = month_end_details.copy()
        month_end_details["date"] = month_end_stats.get("end_date")

    contributions = pd.concat(
        [
            weekly_details.assign(period_type="weekly_chain", index_used=weekly_details["change_index"]),
            month_end_details.assign(period_type="month_end_bridge", index_used=month_end_details.get("price_index", np.nan)),
        ],
        ignore_index=True,
        sort=False,
    )
    keep_cols = [
        "period_type",
        "date",
        "item_key",
        "product_name",
        "component",
        "matched",
        "match_method",
        "match_score",
        "item_code",
        "sprav_name",
        "weight",
        "index_used",
        "item_mom_pp",
        "headline_contribution_pp",
    ]
    contributions[[c for c in keep_cols if c in contributions.columns]].to_csv(
        out_dir / "weekly_laspeyres_contributions.csv", index=False, encoding="utf-8"
    )

    write_summary(out_dir, args.month, weekly_rows, weekly_details, month_end_stats)
    print(out_dir)
    print(weekly_rows.tail(1).to_string(index=False))
    if month_end_stats:
        print(month_end_stats)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
