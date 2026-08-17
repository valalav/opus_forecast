#!/usr/bin/env python3
"""Пересобрать канонический недельный файл на свежих данных.

Проблема. `data/kbr_weekly_prices_2008_2026.csv` остановлен на 26.01.2026 и
вдобавок имеет систематический сдвиг дат на неделю вперёд: ингестер брал конец
интервала «с A по B», тогда как регистрация цены приходится на A. Свежий
операционный файл `data/Еженедельные цены.csv` (по 10.08.2026) размечен верно.

Доказательство сдвига: на общих (дата, товар) цены совпадают на 98.5% только
после сдвига свежего файла на +1 неделю к историческим меткам. Проверка против
официального подкомпонента 42 «Топливо моторное» на окне 2023-09..2025-12:

    история как есть   MAE 0.789 пп, corr 0.892
    история минус 1 нед MAE 0.558 пп, corr 0.942
    свежий файл         MAE 0.555 пп, corr 0.932

Что делает скрипт: сдвигает историю на -1 неделю, дописывает свежие недели
после конца истории, пересчитывает недельные приросты и сохраняет результат
в том же формате, что ждут остальные модули.

    python3 scripts/rebuild_weekly_canonical.py
    python3 scripts/rebuild_weekly_canonical.py --dry-run
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sirena.data.weekly_bridge import load_semicolon_weekly_prices  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
CANONICAL = ROOT / "data" / "kbr_weekly_prices_2008_2026.csv"
FRESH = ROOT / "data" / "Еженедельные цены.csv"

# Строки-агрегаты в операционном файле — это не товары.
AGGREGATE_ROWS = {"Продовольственные товары", "Непродовольственные товары", "Услуги"}

# Товары, которых нет в исторической номенклатуре; коды назначаем детерминированно.
NEW_CODE_BASE = 9000


def build(dry_run: bool = False) -> int:
    hist = pd.read_csv(CANONICAL, parse_dates=["date"])
    fresh = load_semicolon_weekly_prices(FRESH)
    fresh = fresh[~fresh.product_name.isin(AGGREGATE_ROWS)].copy()

    # 1. Исправляем сдвиг истории.
    hist["date"] = hist["date"] - pd.Timedelta(weeks=1)
    hist_end = hist["date"].max()

    # 2. Сопоставляем названия с кодами.
    name_to_code = (
        hist.dropna(subset=["product_code"])
        .groupby("product_name")["product_code"]
        .agg(lambda s: s.mode().iloc[0])
        .to_dict()
    )
    unmapped = sorted(set(fresh.product_name) - set(name_to_code))
    for offset, name in enumerate(unmapped):
        name_to_code[name] = NEW_CODE_BASE + offset

    # 3. Дописываем только то, чего в истории ещё нет.
    add = fresh[fresh["date"] > hist_end].copy()
    add["product_code"] = add["product_name"].map(name_to_code)
    add = add[["date", "product_code", "product_name", "price"]]

    combined = pd.concat(
        [hist[["date", "product_code", "product_name", "price"]], add],
        ignore_index=True,
    )
    combined = (
        combined.dropna(subset=["date", "product_name"])
        .drop_duplicates(subset=["date", "product_name"], keep="last")
        .sort_values(["product_code", "date"])
        .reset_index(drop=True)
    )

    # 4. Пересчитываем недельные приросты по всей длине ряда.
    combined["price_prev_week"] = combined.groupby("product_code")["price"].shift(1)
    combined["wow_growth"] = (
        combined["price"] / combined["price_prev_week"] - 1
    ) * 100
    combined.loc[~np.isfinite(combined["wow_growth"]), "wow_growth"] = np.nan

    print("=" * 68)
    print("ПЕРЕСБОРКА КАНОНИЧЕСКОГО НЕДЕЛЬНОГО ФАЙЛА")
    print("=" * 68)
    print(f"  История (после сдвига -1 нед): по {hist_end.date()}")
    print(f"  Свежий файл                  : по {fresh['date'].max().date()}")
    print(f"  Дописано недель              : {add['date'].nunique()}")
    print(f"  Дописано строк               : {len(add)}")
    print(f"  Новых товаров (коды {NEW_CODE_BASE}+)   : {len(unmapped)}")
    for name in unmapped:
        print(f"      {int(name_to_code[name])}  {name}")
    print(f"  Итого строк                  : {len(combined)}")
    print(f"  Итого товаров                : {combined.product_code.nunique()}")
    print(f"  Диапазон                     : {combined.date.min().date()} .. {combined.date.max().date()}")

    if dry_run:
        print("\n  --dry-run: файл не записан")
        return 0

    backup = CANONICAL.with_suffix(".csv.before_rebuild")
    if not backup.exists():
        shutil.copy2(CANONICAL, backup)
        print(f"\n  Резервная копия: {backup.name}")
    combined.to_csv(CANONICAL, index=False)
    print(f"  Записано: {CANONICAL.relative_to(ROOT)}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="Показать план без записи")
    args = parser.parse_args()
    return build(dry_run=args.dry_run)


if __name__ == "__main__":
    raise SystemExit(main())
