"""
Расширение основного бэктеста 6 наивными бенчмарками + Theil U.

Не трогает scripts/backtest_framework.py. Использует уже посчитанный
`archive/results/backtest_h1_predictions.csv` (или _h2 / _h12) и дописывает
6 столбцов с наивными прогнозами + 1 столбец Theil U для каждой модели
(включая существующие).

Использование:

    # Стандартно — добавить бенчмарки к h=1 бэктесту:
    python3 scripts/run_baselines.py

    # К h=12:
    python3 scripts/run_baselines.py --horizon 12

    # Указать пути явно:
    python3 scripts/run_baselines.py --predictions path/to/preds.csv \\
        --out-csv path/to/out.csv

Наивные бенчмарки (port of cpi_fcst.R fsct_simple, без arima):

  target  4%/12 = 0.333% м/м на все шаги (целевой уровень ЦБ)
  naive   последнее наблюдение повторяется
  mean6   среднее последних 6 месяцев
  ar1     AR(1) с интерсептом
  argap   AR(1) вокруг target (без вычитания среднего)
  direct  d_t = a_h + b_h * d_{t-h} (для h=1 совпадает с ar1)

Theil U: sqrt(MSE(model) / MSE(actual.shift(1))).
  < 1 — лучше наивного (actual.shift(1)),
  > 1 — хуже. Для multi-step моделей без last-actual U > 1 нормально.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from benchmarks_cpi import (
    bench_target, bench_naive, bench_mean6, bench_ar1, bench_argap, bench_direct,
    load_kbr,
)


BENCH_NAMES: List[str] = ["target", "naive", "mean6", "ar1", "argap", "direct"]
BENCH_FNS = {
    "target": bench_target, "naive": bench_naive, "mean6": bench_mean6,
    "ar1": bench_ar1, "argap": bench_argap, "direct": bench_direct,
}


def add_baselines(
    predictions_csv: Path,
    data_csv: Path,
    target: str = "mom",
    period: str = "all",
) -> Tuple[pd.DataFrame, List[str]]:
    """Читает predictions, для каждой строки (Date) считает 6 наивных прогнозов
    по фактическому train-окну до Date. Возвращает (extended_df, model_columns)."""
    preds = pd.read_csv(predictions_csv)
    preds["Date"] = pd.to_datetime(preds["Date"])
    preds = preds.sort_values("Date").reset_index(drop=True)
    skip_cols = {"Date", "Actual"}

    full = load_kbr(data_csv, dropna_how="any")
    full[target] = full[target].astype(float)

    for name in BENCH_NAMES:
        col = name
        preds[col] = np.nan
        for i, row in preds.iterrows():
            cutoff = row["Date"] - pd.offsets.MonthBegin(1)
            history = full.loc[full.index < cutoff, target].dropna().to_numpy(dtype=float)
            if len(history) < 6:
                continue
            preds.at[i, col] = float(BENCH_FNS[name](history, 1)[0])

    return preds, [c for c in preds.columns if c not in skip_cols]


def theil_per_model(df: pd.DataFrame, model_cols: List[str]) -> pd.DataFrame:
    """Theil U = RMSE(model) / RMSE(actual.shift(1)).
    Возвращает DataFrame [Model, RMSE, MAE, MAPE, Theil_U, n].
    """
    rows = []
    for m in model_cols:
        valid = df[[m, "Actual"]].dropna()
        if len(valid) < 3:
            continue
        a = valid["Actual"].to_numpy(dtype=float)
        f = valid[m].to_numpy(dtype=float)
        err = a - f
        rmse = float(np.sqrt(np.mean(err ** 2)))
        mae = float(np.mean(np.abs(err)))
        denom = np.where(np.abs(a) < 1.0, 1.0, np.abs(a))
        mape = float(np.mean(np.abs(err) / denom) * 100.0)
        naive_pred = np.concatenate([[a[0]], a[:-1]])
        rmse_naive = float(np.sqrt(np.mean((a - naive_pred) ** 2)))
        theil = rmse / rmse_naive if rmse_naive > 1e-12 else np.nan
        rows.append({
            "Model": m, "RMSE": rmse, "MAE": mae, "MAPE": mape,
            "Theil_U": float(theil), "n": int(len(valid)),
        })
    return pd.DataFrame(rows).sort_values("MAE")


def main() -> int:
    p = argparse.ArgumentParser(description="Добавить 6 наивных бенчмарков + Theil U к бэктесту")
    p.add_argument("--horizon", type=int, default=1, help="1, 2 или 12 (имя файла)")
    p.add_argument("--predictions", type=Path, default=None,
                   help="CSV с предсказаниями (по умолчанию archive/results/backtest_h{H}_predictions.csv)")
    p.add_argument("--data", type=Path,
                   default=Path(__file__).resolve().parents[1] / "data" / "inflation_data.csv")
    p.add_argument("--out-csv", type=Path, default=None,
                   help="Куда сохранить расширенный predictions CSV (по умолчанию — рядом с исходным)")
    p.add_argument("--out-metrics", type=Path, default=None,
                   help="Куда сохранить сводные метрики с Theil U (по умолчанию — рядом)")
    args = p.parse_args()

    preds_path = args.predictions or Path(f"archive/results/backtest_h{args.horizon}_predictions.csv")
    if not preds_path.exists():
        print(f"Файл не найден: {preds_path}", file=sys.stderr)
        return 1

    print(f"Loading predictions: {preds_path}", file=sys.stderr)
    print(f"  Loading data: {args.data}", file=sys.stderr)
    ext, model_cols = add_baselines(preds_path, args.data)
    print(f"  Added {len(BENCH_NAMES)} baseline columns. New shape: {ext.shape}", file=sys.stderr)

    out_csv = args.out_csv or preds_path.with_name(preds_path.stem + "_with_baselines.csv")
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    ext.to_csv(out_csv, index=False)
    print(f"Saved extended predictions: {out_csv}", file=sys.stderr)

    metrics = theil_per_model(ext, model_cols)
    out_metrics = args.out_metrics or preds_path.with_name(preds_path.stem + "_metrics_with_baselines.csv")
    metrics.to_csv(out_metrics, index=False)
    print(f"Saved metrics: {out_metrics}", file=sys.stderr)

    print("\n=== Сводные метрики (отсортированы по MAE) ===", file=sys.stderr)
    show_cols = ["Model", "RMSE", "MAE", "MAPE", "Theil_U", "n"]
    print(metrics[show_cols].to_string(index=False))

    print("\n=== Бенчмарки (rank по MAE) ===", file=sys.stderr)
    bench_metrics = metrics[metrics["Model"].isin(BENCH_NAMES)].sort_values("MAE")
    print(bench_metrics[show_cols].to_string(index=False))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
