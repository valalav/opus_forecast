from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a lightweight rolling backtest for Sirena Micro_SM only."
    )
    parser.add_argument(
        "--project-root",
        default=".",
        help="Sirena project root. Default: current working directory.",
    )
    parser.add_argument(
        "--actuals",
        default="data/inflation_data.csv",
        help="Actual monthly inflation CSV inside project root.",
    )
    parser.add_argument(
        "--output-dir",
        default="archive/results/micro_sm_rolling",
        help="Output directory inside project root.",
    )
    parser.add_argument(
        "--horizons",
        type=int,
        nargs="+",
        default=[1, 2, 3, 12],
        help="Horizons in months.",
    )
    parser.add_argument(
        "--windows",
        type=int,
        default=12,
        help="Number of latest actual target months per horizon.",
    )
    return parser.parse_args()


def read_actuals(path: Path) -> pd.Series:
    df = pd.read_csv(path, sep=";", decimal=",")
    df["Date"] = pd.to_datetime(df["Date"], format="%d.%m.%Y", errors="coerce")
    if df["Date"].isna().any():
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["Date"] = df["Date"].dt.to_period("M").dt.to_timestamp()
    df["mom"] = pd.to_numeric(
        df["mom"].astype(str).str.replace(",", ".", regex=False),
        errors="coerce",
    )
    series = df.dropna(subset=["Date", "mom"]).set_index("Date")["mom"].sort_index()
    return series - 100


def metrics_for(frame: pd.DataFrame) -> dict[str, float | int]:
    valid = frame.dropna(subset=["actual", "prediction"])
    if valid.empty:
        return {
            "n": 0,
            "mae": np.nan,
            "rmse": np.nan,
            "mean_error": np.nan,
            "max_abs_error": np.nan,
            "kpi_violations_abs_gt_0_5": 0,
        }

    error = valid["prediction"] - valid["actual"]
    return {
        "n": int(len(valid)),
        "mae": float(error.abs().mean()),
        "rmse": float(np.sqrt((error**2).mean())),
        "mean_error": float(error.mean()),
        "max_abs_error": float(error.abs().max()),
        "kpi_violations_abs_gt_0_5": int((error.abs() > 0.5).sum()),
    }


def main() -> None:
    args = parse_args()
    project_root = Path(args.project_root).resolve()
    sys.path.insert(0, str(project_root))

    from sirena.models.micro_statsmodels_external import (  # noqa: WPS433
        MicroStatsmodelsExternalForecaster,
    )

    actuals = read_actuals(project_root / args.actuals)
    out_dir = project_root / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "actuals_file": str(project_root / args.actuals),
        "windows": args.windows,
        "horizons": args.horizons,
        "metrics": {},
    }

    all_rows = []
    for horizon in args.horizons:
        model = MicroStatsmodelsExternalForecaster(horizon=horizon)
        model.fit()

        rows = []
        for target_date in actuals.index[-args.windows :]:
            cutoff = target_date - pd.DateOffset(months=horizon)
            train_index = pd.date_range(actuals.index.min(), cutoff, freq="MS")
            train = pd.DataFrame(index=train_index)
            result = model.predict(train, target_date)
            raw_prediction = result.get("prediction", np.nan)
            prediction = (
                float(raw_prediction) - 100
                if pd.notna(raw_prediction)
                else np.nan
            )
            rows.append(
                {
                    "horizon": horizon,
                    "cutoff_date": cutoff,
                    "target_date": target_date,
                    "actual": float(actuals.loc[target_date]),
                    "prediction": prediction,
                    "error": prediction - float(actuals.loc[target_date])
                    if pd.notna(prediction)
                    else np.nan,
                }
            )

        horizon_df = pd.DataFrame(rows)
        horizon_df.to_csv(
            out_dir / f"micro_sm_h{horizon}_predictions.csv",
            index=False,
            encoding="utf-8",
        )
        summary["metrics"][f"h{horizon}"] = metrics_for(horizon_df)
        all_rows.extend(rows)

    all_df = pd.DataFrame(all_rows)
    all_df.to_csv(out_dir / "micro_sm_all_predictions.csv", index=False, encoding="utf-8")
    (out_dir / "micro_sm_metrics.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
