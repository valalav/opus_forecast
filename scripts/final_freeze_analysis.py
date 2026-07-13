#!/usr/bin/env python3

from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from sirena.freeze_analysis import run_final_freeze_analysis


def main() -> None:
    outputs = run_final_freeze_analysis()
    latest = outputs.monthly_summary.iloc[-1]
    print("Final exclusion-aware Level-5 freeze analysis complete")
    print(f"Included items: {outputs.item_summary['Item_code'].nunique()}")
    print(f"Excluded documented codes: {len(outputs.exclusions_summary)}")
    print(f"Latest weighted freeze share: {latest['FDI_weighted_pct']:.2f}%")
    print(f"Report: {outputs.report_path}")


if __name__ == "__main__":
    main()
