#!/usr/bin/env python3
"""Калиброванный недельный nowcast месячного ИПЦ КБР.

Сырой недельный мост нельзя подавать как прогноз: на 35 месяцах он даёт
LOO MAE 0.469 — хуже простого среднего (0.441). Так возник промах по июлю
2026 (мост +1.41% против факта +0.82%).

Здесь сигнал сжимается к среднему регрессией и дополняется плодоовощным
членом (недельная корзина переоценивает плодоовощи примерно втрое против
их веса в ИПЦ). LOO MAE падает до 0.326.

Использование:
    python3 scripts/weekly_calibrated_nowcast.py --month 2026-08
    python3 scripts/weekly_calibrated_nowcast.py --month 2026-08 --loo
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sirena.data.weekly_bridge import (  # noqa: E402
    compute_weekly_bridge_nowcast,
    load_semicolon_weekly_prices,
)

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_WEEKLY = ROOT / "data" / "Сравнение еженедельных цен_01.csv"
MICRO_SPRAV = ROOT / "data" / "raw" / "micro_sprav.csv"
MONTHLY = ROOT / "data" / "inflation_data.csv"

# Вес подкомпонента 33 «Плодоовощная продукция, включая картофель» в ИПЦ.
VEG_WEIGHT = 0.05888


def load_actual_monthly() -> pd.Series:
    """Фактический MoM в процентах, индексированный периодом месяца."""
    df = pd.read_csv(MONTHLY, sep=";", decimal=",", encoding="utf-8-sig")
    df.columns = [c.strip() for c in df.columns]
    df["per"] = pd.to_datetime(df["Date"], format="%d.%m.%Y").dt.to_period("M")
    return df.set_index("per")["mom"].astype(float) - 100.0


def compute_veg_proxy(weekly: pd.DataFrame) -> pd.Series:
    """Взвешенный плодоовощной прокси, база «конец месяца к концу месяца».

    Валидирован против официального подкомпонента 33: corr 0.944, MAE 2.94 пп.
    """
    sprav = pd.read_csv(MICRO_SPRAV, sep=";", decimal=",", encoding="utf-8-sig")
    veg = sprav[sprav["Субкомпонент"].astype(str).str.contains("Плодоовощ", na=False)]
    weights: Dict[str, float] = dict(zip(veg["Товар"], veg["Weight"]))

    prices = (
        weekly[weekly.product_name.isin(weights)]
        .pivot_table(index="date", columns="product_name", values="price")
        .sort_index()
        .ffill()
    )
    cols = [c for c in prices.columns if c in weights]
    w = np.array([weights[c] for c in cols], dtype=float)
    w = w / w.sum()

    month_end = prices.assign(per=prices.index.to_period("M")).groupby("per")[cols].last()
    mom = ((month_end[cols] / month_end[cols].shift(1) * w).sum(axis=1) - 1) * 100
    # Первый месяц выборки неполон — базы для сравнения нет.
    return mom.iloc[1:]


def build_panel(weekly_path: Path) -> pd.DataFrame:
    """Панель: сигнал первых двух недель, полный мост, плодоовощи, факт."""
    weekly = load_semicolon_weekly_prices(weekly_path)
    weekly["per"] = weekly["date"].dt.to_period("M")
    actual = load_actual_monthly()
    veg = compute_veg_proxy(weekly)

    rows: List[Dict[str, object]] = []
    for per in sorted(weekly["per"].unique()):
        dates = sorted(weekly.loc[weekly["per"] == per, "date"].unique())
        if not dates:
            continue
        row: Dict[str, object] = {
            "per": per,
            "n_weeks": len(dates),
            "actual": actual.get(per, np.nan),
            "veg": veg.get(per, np.nan),
        }
        full = compute_weekly_bridge_nowcast(str(per), df=weekly)
        row["bridge_full"] = _bridge_index(full)
        if len(dates) >= 2:
            upto_second = weekly[weekly["date"] <= pd.Timestamp(dates[1])]
            partial = compute_weekly_bridge_nowcast(str(per), df=upto_second)
            row["bridge_w2"] = _bridge_index(partial)
        rows.append(row)

    panel = pd.DataFrame(rows).set_index("per")
    for col in ("bridge_full", "bridge_w2"):
        panel[col] = pd.to_numeric(panel[col], errors="coerce") - 100.0
    return panel


def _bridge_index(bridge: Dict) -> Optional[float]:
    month_end = bridge.get("month_end")
    return month_end.get("index") if isinstance(month_end, dict) else None


def fit(train: pd.DataFrame) -> np.ndarray:
    """МНК: actual ~ 1 + bridge_w2 + veg."""
    X = np.c_[np.ones(len(train)), train["bridge_w2"].values, train["veg"].values]
    return np.linalg.lstsq(X, train["actual"].values, rcond=None)[0]


def predict(coef: np.ndarray, bridge_w2: float, veg: float) -> float:
    return float(coef[0] + coef[1] * bridge_w2 + coef[2] * veg)


def leave_one_out(panel: pd.DataFrame) -> Tuple[float, float]:
    """LOO MAE и RMSE калиброванной модели."""
    errors = []
    for idx in panel.index:
        train = panel.drop(idx)
        test = panel.loc[idx]
        coef = fit(train)
        errors.append(test["actual"] - predict(coef, test["bridge_w2"], test["veg"]))
    err = np.array(errors, dtype=float)
    return float(np.mean(np.abs(err))), float(np.sqrt(np.mean(err**2)))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--month", default="2026-08", help="Целевой месяц, YYYY-MM")
    parser.add_argument("--weekly-path", default=str(DEFAULT_WEEKLY))
    parser.add_argument("--loo", action="store_true", help="Показать leave-one-out качество")
    args = parser.parse_args()

    panel = build_panel(Path(args.weekly_path))
    target = pd.Period(args.month, freq="M")
    if target not in panel.index:
        print(f"Нет недельных данных за {target}")
        return 1

    # Обучаем только на месяцах с полным покрытием и известным фактом.
    train = panel[panel["n_weeks"] >= 3].dropna(subset=["actual", "bridge_w2", "veg"])
    coef = fit(train)
    row = panel.loc[target]
    point = predict(coef, row["bridge_w2"], row["veg"])
    mae, rmse = leave_one_out(train)

    print("=" * 66)
    print(f"КАЛИБРОВАННЫЙ NOWCAST — {target}")
    print("=" * 66)
    print(f"  Недель в месяце получено : {int(row['n_weeks'])}")
    print(f"  Сырой недельный мост     : {row['bridge_w2']:+.3f}%  <- НЕ прогноз")
    print(f"  Плодоовощной прокси      : {row['veg']:+.3f}%")
    print()
    print(f"  actual = {coef[0]:+.3f} {coef[1]:+.3f}*мост {coef[2]:+.4f}*плодоовощи")
    print(f"  (обучено на {len(train)} месяцах: {train.index.min()} .. {train.index.max()})")
    print()
    print(f"  ПРОГНОЗ MoM              : {point:+.3f}%   (индекс {100 + point:.2f})")
    print(f"  Коридор ±1 LOO MAE       : {point - mae:+.2f}% .. {point + mae:+.2f}%")
    if args.loo:
        raw = float(np.mean(np.abs(train["actual"] - train["bridge_w2"])))
        base = float(np.mean(np.abs(train["actual"] - train["actual"].mean())))
        print()
        print(f"  LOO MAE калиброванной    : {mae:.3f}   RMSE {rmse:.3f}")
        print(f"  MAE сырого моста         : {raw:.3f}")
        print(f"  MAE среднего (baseline)  : {base:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
