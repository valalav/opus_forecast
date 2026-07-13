#!/usr/bin/env python3
"""Print diagnostic weekly bridge nowcast for a target month."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from sirena.data.weekly_bridge import compute_weekly_bridge_nowcast


def _format_percent(value: Any) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):+.4f}%"


def _print_component_table(components: Dict[str, Dict[str, Any]]) -> None:
    labels = {
        "food": "Продовольствие",
        "nonfood": "Непродовольственные",
        "services": "Услуги",
    }
    for key, label in labels.items():
        row = components.get(key, {})
        index = row.get("index")
        n_items = row.get("n_items", 0)
        if index is None:
            continue
        print(f"    {label:22s}: {float(index):8.4f} ({float(index) - 100:+.4f}%), n={n_items}")


def _print_drivers(title: str, rows: list[Dict[str, Any]]) -> None:
    print(f"\n{title}")
    for row in rows[:10]:
        print(
            "  "
            f"{row.get('product_name', '')[:42]:42s} "
            f"{row.get('index', 0):8.4f} "
            f"({row.get('mom', 0):+7.4f} п.п.)"
        )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Diagnostic weekly bridge nowcast from fresh semicolon weekly rows."
    )
    parser.add_argument("--month", default="2026-04", help="Target month, YYYY-MM")
    parser.add_argument(
        "--data-path",
        default=None,
        help="Path to fresh semicolon weekly file; defaults to data/Сравнение еженедельных цен_01.csv",
    )
    parser.add_argument("--json", action="store_true", help="Print raw JSON diagnostic")
    args = parser.parse_args()

    result = compute_weekly_bridge_nowcast(args.month, data_path=args.data_path)
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0

    print("=" * 72)
    print(f"Weekly bridge nowcast: {result['target_month']}")
    print("=" * 72)
    print(f"Source: {result.get('source_file')}")
    print(
        "Rows: "
        f"raw={result.get('raw_rows')}, deduped={result.get('deduped_rows')}, "
        f"duplicates_removed={result.get('duplicates_removed')}"
    )
    print(f"Status: {result.get('status')}")

    chain = result.get("chain")
    if chain:
        print("\nCumulative weekly chain")
        for week in chain.get("weeks", []):
            print(
                f"  {week['date']}: "
                f"index={week['index']:.4f}, "
                f"mom={week['mom']:+.4f}%, "
                f"cumulative={week['cumulative_mom']:+.4f}%, "
                f"n={week['n_items']}"
            )
        print(
            "  Итог цепочки: "
            f"index={chain.get('index'):.4f}, mom={_format_percent(chain.get('mom'))}"
        )
        print(
            "  Экстраполированный сигнал: "
            f"{_format_percent(chain.get('extrapolated_mom'))} "
            f"({chain.get('weeks_count')} недель, "
            f"remaining={chain.get('remaining_weeks')}, "
            f"decay={chain.get('decay_factor')})"
        )

    month_end = result.get("month_end")
    if month_end:
        print("\nMonth-end price-level bridge")
        print(
            f"  {month_end['base_date']} → {month_end['end_date']}: "
            f"index={month_end['index']:.4f}, mom={_format_percent(month_end.get('mom'))}, "
            f"matched_items={month_end['matched_items']}"
        )
        _print_component_table(month_end.get("components", {}))
        _print_drivers("Top decreases", month_end.get("top_decreases", []))
        _print_drivers("Top increases", month_end.get("top_increases", []))

    next_week = result.get("first_next_week_vs_month_end")
    if next_week:
        print("\nFirst next week control")
        print(
            f"  {next_week['base_date']} → {next_week['end_date']}: "
            f"index={next_week['index']:.4f}, mom={_format_percent(next_week.get('mom'))}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
