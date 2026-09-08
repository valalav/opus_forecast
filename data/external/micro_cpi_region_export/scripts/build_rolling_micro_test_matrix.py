from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from run_statsmodels import forecast_one, read_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build Sirena micro_test.csv matrix with historical cutoff columns "
            "from selected-region CPI export."
        )
    )
    parser.add_argument("--input", required=True, help="Path to region_cpi_long.csv")
    parser.add_argument("--metadata", required=True, help="Path to metadata.json")
    parser.add_argument("--out", required=True, help="Output micro_test-style CSV path")
    parser.add_argument("--diagnostics", default="", help="Optional diagnostics CSV path")
    parser.add_argument("--config", default="config/model.yml", help="YAML config path")
    parser.add_argument("--horizon", type=int, default=12, help="Forecast horizon in months")
    parser.add_argument("--item-code", default="1", help="Aggregate item code. Default: 1")
    parser.add_argument(
        "--cutoff-start",
        default="",
        help="First cutoff month, YYYY-MM-DD. Default: first month with enough history.",
    )
    parser.add_argument(
        "--cutoff-end",
        default="",
        help="Last cutoff month, YYYY-MM-DD. Default: metadata latest_month.",
    )
    return parser.parse_args()


def month_start(value: object) -> pd.Timestamp:
    return pd.Timestamp(value).to_period("M").to_timestamp()


def main() -> None:
    args = parse_args()
    config = read_config(args.config)
    min_history = int(config.get("input", {}).get("min_history_months", 36))
    horizon = int(args.horizon)

    input_path = Path(args.input)
    metadata = json.loads(Path(args.metadata).read_text(encoding="utf-8-sig"))
    latest = month_start(metadata.get("latest_month"))
    if args.cutoff_end:
        latest = min(latest, month_start(args.cutoff_end))

    data = pd.read_csv(
        input_path,
        dtype={"region_code": "string", "item_code": "string"},
        encoding="utf-8-sig",
    )
    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    data["mom_index"] = pd.to_numeric(data["mom_index"], errors="coerce")
    data = data.dropna(subset=["date", "item_code"]).sort_values(["item_code", "date"])

    item_df = data[data["item_code"].astype(str) == str(args.item_code)].copy()
    if item_df.empty:
        raise ValueError(f"No rows found for item_code={args.item_code}.")

    item_df["date"] = item_df["date"].dt.to_period("M").dt.to_timestamp()
    series = item_df.set_index("date")["mom_index"].sort_index().asfreq("MS")
    first_valid = series.dropna().index.min()
    if pd.isna(first_valid):
        raise ValueError(f"No valid mom_index values for item_code={args.item_code}.")

    if args.cutoff_start:
        cutoff_start = month_start(args.cutoff_start)
    else:
        cutoff_start = month_start(first_valid + pd.DateOffset(months=min_history - 1))
    cutoff_start = max(cutoff_start, month_start(first_valid))

    if cutoff_start > latest:
        raise ValueError(
            f"cutoff_start {cutoff_start.date()} is after cutoff_end {latest.date()}."
        )

    cutoffs = pd.date_range(cutoff_start, latest, freq="MS")
    target_dates = pd.date_range(
        cutoff_start + pd.DateOffset(months=1),
        latest + pd.DateOffset(months=horizon),
        freq="MS",
    )

    matrix_columns = {}
    diagnostics = []
    for cutoff in cutoffs:
        train = series.loc[:cutoff]
        forecast, diag = forecast_one(train, horizon, config)
        col_name = cutoff.strftime("%d.%m.%Y")
        column = pd.Series(np.nan, index=target_dates)
        for step, value in enumerate(forecast, start=1):
            target = cutoff + pd.DateOffset(months=step)
            column.loc[target] = value
        matrix_columns[col_name] = column
        diagnostics.append(
            {
                "cutoff_date": cutoff.date().isoformat(),
                "item_code": str(args.item_code),
                "history_months": diag["history_months"],
                "method": diag["method"],
                "status": diag["status"],
                "aic": diag["aic"],
                "sse": diag["sse"],
                "error": diag["error"],
            }
        )

    matrix = pd.DataFrame(matrix_columns, index=target_dates)
    out = matrix.reset_index().rename(columns={"index": "Date"})
    out["Date"] = out["Date"].dt.strftime("%d.%m.%Y")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, sep=";", decimal=",", index=False, encoding="utf-8-sig")

    diagnostics_path = Path(args.diagnostics) if args.diagnostics else out_path.with_name(
        out_path.stem + "_diagnostics.csv"
    )
    pd.DataFrame(diagnostics).to_csv(diagnostics_path, index=False, encoding="utf-8")


if __name__ == "__main__":
    main()
