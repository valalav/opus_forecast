"""
Python-порт логики УГУ `enum.prg` (EViews) — grid-search VAR на расширяющемся окне.

Суть: для зависимой `y` (например, mom CPI КБР) перебираем
  * все подмножества эндогенных C(n, k) для k=1..n,
  * лаги VAR 1..max_lag,
  * расширяющееся обучающее окно, dynamic forecast на h шагов вперёд,
считаем AFE / MSFE / MAPFE для каждого горизонта и выбираем лучшую спецификацию
по заданной метрике. Никакой подстройки под шоки — это чистый research-скетч,
решение о режимах и продвижении в SIRENA принимается отдельно.

Использование:

    # smoke-режим на наших данных КБР:
    python3 scripts/grid_search_var.py

    # пример со своими параметрами:
    python3 scripts/grid_search_var.py --endog mom,Prod,Nonprod,Serv \\
        --exog usd_nom_i,Ki_i,Ruonia --max-lag 3 --horizon 3 --min-window 60

Метрики:
    AFE  = mean(|mom_y - mom_y_f|)
    MSFE = mean((mom_y - mom_y_f)^2)
    MAPFE = mean(|mom_y - mom_y_f| / |mom_y|)   <- нормировка на mom, не на y,
                                                  устойчиво при mom ~ 0
"""
from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    from statsmodels.tsa.api import VAR
except ImportError as exc:  # pragma: no cover
    print("statsmodels не установлен. Поставьте: pip install statsmodels", file=sys.stderr)
    raise


# ----------------------------------------------------------------------
# Загрузка и подготовка данных
# ----------------------------------------------------------------------

def load_kbr_inflation(path: Path) -> pd.DataFrame:
    """Читает data/inflation_data.csv в формате SIRENA (sep=';', decimal=',',
    индексы вида 100+X = MoM%). Возвращает DataFrame с DatetimeIndex,
    где mom-ряды оставлены в шкале 100+X (как в исходнике), а уровни (Ki, Ruonia) — как есть.
    """
    df = pd.read_csv(path, sep=";", decimal=",")
    df["Date"] = pd.to_datetime(df["Date"], format="%d.%m.%Y", errors="coerce")
    df = df.dropna(subset=["Date"]).set_index("Date").sort_index()
    return df


def make_mom(series: pd.Series) -> pd.Series:
    """Эквивалент EViews: series/series(-1)*100, но работает и с индексами,
    и с темпами в виде уровней. NaN-безопасно."""
    return series / series.shift(1) * 100.0


# ----------------------------------------------------------------------
# VAR: проектирование дизайн-матриц
# ----------------------------------------------------------------------

def _stack_lags(data: np.ndarray, lags: int, exog: np.ndarray | None) -> Tuple[np.ndarray, np.ndarray]:
    """X = [1, y_{t-1}, ..., y_{t-lags}, exog_t]; Y = y_t. Вектор-строки.
    Совпадает по логике с _design() в sirena/models/var_policy.py, но
    допускает общий lags>1 (как в enum.prg).
    """
    x_rows, y_rows = [], []
    for t in range(lags, len(data)):
        row = [1.0]
        for lag in range(1, lags + 1):
            row.extend(data[t - lag])
        if exog is not None:
            row.extend(exog[t])
        x_rows.append(row)
        y_rows.append(data[t])
    return np.asarray(x_rows), np.asarray(y_rows)


def _fit_var_ols(data: np.ndarray, lags: int, exog: np.ndarray | None) -> np.ndarray:
    """Возвращает betas shape (k_endog, 1 + lags*k_endog + k_exog)."""
    x, y = _stack_lags(data, lags, exog)
    if len(x) < y.shape[1] + 1:
        raise ValueError(f"Слишком короткая выборка для VAR({lags}): {len(x)} < {y.shape[1]+1}")
    return np.linalg.lstsq(x, y, rcond=None)[0]


def _dynamic_forecast(data: np.ndarray, lags: int, exog: np.ndarray | None,
                      horizon: int, exog_future: np.ndarray | None) -> np.ndarray:
    """Динамический прогноз на horizon шагов. exog_future — матрица (horizon, k_exog)
    или None, если экзогенных нет. Если экзогенные есть, а exog_future=None,
    считается прогноз «без будущего» — берётся последний наблюдаемый вектор.
    """
    betas = _fit_var_ols(data, lags, exog)
    hist = [data[t] for t in range(len(data) - lags, len(data))]
    # hist = список из lags последних векторов y; hist[-1] = y_{T}.
    out = np.empty((horizon, data.shape[1]))
    for h in range(horizon):
        row = [1.0]
        for lag in range(1, lags + 1):
            row.extend(hist[-lag])
        if exog is not None:
            row.extend(exog_future[h] if exog_future is not None else exog[-1])
        pred = np.asarray([row]) @ betas  # shape (1, k_endog)
        out[h] = pred.ravel()
        hist.append(out[h])
    return out


# ----------------------------------------------------------------------
# Backtest: расширяющееся окно
# ----------------------------------------------------------------------

def rolling_forecast(
    df_mom: pd.DataFrame,
    endog_cols: Sequence[str],
    exog_cols: Sequence[str],
    lags: int,
    horizon: int,
    min_window: int,
) -> np.ndarray:
    """Для каждой точки t >= min_window+lags+horizon оценивает VAR на
    df_mom.iloc[:t] и прогнозирует horizon шагов. Возвращает массив
    формы (n_forecasts, horizon, k_endog), где строки соответствуют
    cut-off индексам df_mom.
    """
    endog = df_mom.loc[:, list(endog_cols)].to_numpy(dtype=float)
    exog = df_mom.loc[:, list(exog_cols)].to_numpy(dtype=float) if exog_cols else None
    n = len(df_mom)
    # Число доступных cut-offs:
    #   первая валидная точка t = min_window + lags  (нужны lags прошлых и window обучающих)
    #   последняя t = n - horizon  (чтобы horizon не вышел за выборку)
    first = min_window + lags
    last = n - horizon
    if first >= last:
        raise ValueError(
            f"Недостаточно наблюдений: n={n}, min_window={min_window}, lags={lags}, "
            f"horizon={horizon} -> нужен хотя бы {min_window + lags + horizon + 1}"
        )
    out = np.empty((last - first, horizon, endog.shape[1]))
    for k, t in enumerate(range(first, last)):
        endog_train = endog[:t]
        exog_train = exog[:t] if exog is not None else None
        out[k] = _dynamic_forecast(endog_train, lags, exog_train, horizon, exog_future=None)
    return out


# ----------------------------------------------------------------------
# Метрики: AFE / MSFE / MAPFE
# ----------------------------------------------------------------------

def forecast_metrics(actuals: np.ndarray, predictions: np.ndarray) -> dict:
    """actuals: (n_cut, horizon, k_endog); predictions: то же.
    Возвращает метрики per-horizon (h=1..H), усреднённые по cut-off'ам
    и по эндогенным рядам (mean по обеим осям — соответствует
    mean(afe) в EViews-коде УГУ).
    """
    fe = actuals - predictions
    afe = np.abs(fe)
    sfe = fe ** 2
    # MAPFE: нормировка на |mom_y|, как в enum.prg:
    #   mfe_var_{!i}=@abs(fe_var_{!i})/mom_{%dep}
    # т.е. относительная ошибка к фактическому mom.
    #    protect от близких к нулю фактов клипом:
    denom = np.where(np.abs(actuals) < 1e-3, 1e-3, np.abs(actuals))
    mfe = afe / denom
    return {
        "AFE":  afe.mean(axis=(0, 2)),
        "MSFE": sfe.mean(axis=(0, 2)),
        "MAPFE": mfe.mean(axis=(0, 2)),
    }


# ----------------------------------------------------------------------
# Grid search
# ----------------------------------------------------------------------

def _all_subsets(items: Sequence[str]) -> Iterable[Tuple[str, ...]]:
    """Все непустые подмножества, по возрастанию длины (как enum.prg:
    for !k=1 to !n ... все комбинации C(n, k) для k)."""
    items = list(items)
    for k in range(1, len(items) + 1):
        for combo in itertools.combinations(items, k):
            yield combo


def grid_search(
    df_mom: pd.DataFrame,
    target: str,
    endog_pool: Sequence[str],
    exog_cols: Sequence[str],
    max_lag: int,
    horizon: int,
    min_window: int,
) -> pd.DataFrame:
    """Возвращает таблицу спецификаций с метриками AFE/MSFE/MAPFE
    на каждом горизонте. Итог: ~C(n_endog_pool,1)+...+C(n,k) строк,
    умноженное на max_lag вариантов лага.
    """
    rows = []
    actuals_cache: dict = {}

    extra_pool = [c for c in endog_pool if c != target]
    for extra_subset in itertools.chain.from_iterable(
        itertools.combinations(extra_pool, k) for k in range(0, len(extra_pool) + 1)
    ):
        endogs = (target, *extra_subset)
        for lag in range(1, max_lag + 1):
            key = (tuple(sorted(endogs)), lag)
            if key not in actuals_cache:
                # считаем прогноз один раз для (endogs, lag)
                try:
                    preds = rolling_forecast(
                        df_mom, endogs, exog_cols, lag, horizon, min_window
                    )
                except ValueError as e:
                    rows.append({
                        "endog": ",".join(endogs),
                        "exog": ",".join(exog_cols) or "—",
                        "lag": lag,
                        "status": f"skip: {e}",
                    })
                    actuals_cache[key] = None
                    continue
                # соберём actuals в той же форме:
                first = min_window + lag
                last = len(df_mom) - horizon
                # кусок df_mom, соответствующий first..last-1 (все target-ряды)
                actuals = df_mom.iloc[first:last].loc[:, list(endogs)].to_numpy(dtype=float)
                # actuals shape (n_cut, k_endog); нужно (n_cut, horizon, k_endog) —
                # берём из самой выборки, сдвигая на horizon шагов вперёд
                actuals_3d = np.empty_like(preds)
                for h in range(horizon):
                    actuals_3d[:, h, :] = df_mom.iloc[first + h:last + h].loc[:, list(endogs)].to_numpy(dtype=float)
                m = forecast_metrics(actuals_3d, preds)
                actuals_cache[key] = m
            m = actuals_cache[key]
            if m is None:
                continue
            row = {
                "endog": ",".join(endogs),
                "exog": ",".join(exog_cols) or "—",
                "lag": lag,
                "status": "ok",
            }
            for h in range(horizon):
                row[f"AFE_h{h+1}"] = float(m["AFE"][h])
                row[f"MSFE_h{h+1}"] = float(m["MSFE"][h])
                row[f"MAPFE_h{h+1}"] = float(m["MAPFE"][h])
            rows.append(row)

    out = pd.DataFrame(rows)
    return out


# ----------------------------------------------------------------------
# CLI / smoke-режим на КБР
# ----------------------------------------------------------------------

DEFAULT_ENDOG_POOL = ["mom", "Prod", "Nonprod", "Serv"]
DEFAULT_EXOG = ["usd_nom_i", "Ki_i", "Ruonia"]


def build_kbr_mom_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Готовит DataFrame в шкале mom*100 (mom, Prod, Nonprod, Serv, usd_nom_i)
    плюс уровни Ki_i, Ruonia. NaN дропаем.
    """
    out = pd.DataFrame(index=df.index)
    for col in ["mom", "Prod", "Nonprod", "Serv", "usd_nom_i"]:
        out[col] = pd.to_numeric(df[col], errors="coerce")
    out["Ki_i"] = pd.to_numeric(df["Ki_i"], errors="coerce")
    out["Ruonia"] = pd.to_numeric(df["Ruonia"], errors="coerce")
    return out.dropna()


def main() -> int:
    p = argparse.ArgumentParser(description="Grid-search VAR по мотивам УГУ enum.prg")
    p.add_argument("--data", type=Path,
                   default=Path(__file__).resolve().parents[1] / "data" / "inflation_data.csv",
                   help="Путь к data/inflation_data.csv")
    p.add_argument("--target", default="mom", help="Зависимая переменная (mom CPI)")
    p.add_argument("--endog", default=",".join(DEFAULT_ENDOG_POOL),
                   help="Пул эндогенных (через запятую)")
    p.add_argument("--exog", default=",".join(DEFAULT_EXOG),
                   help="Экзогенные (через запятую) — оставляются как есть")
    p.add_argument("--max-lag", type=int, default=3)
    p.add_argument("--horizon", type=int, default=3)
    p.add_argument("--min-window", type=int, default=60)
    p.add_argument("--out", type=Path, default=None,
                   help="Куда сохранить CSV-результат (по умолчанию — в stdout top-20)")
    args = p.parse_args()

    df = load_kbr_inflation(args.data)
    df_mom = build_kbr_mom_frame(df)
    if len(df_mom) < args.min_window + args.max_lag + args.horizon + 1:
        print(f"Слишком мало наблюдений: {len(df_mom)} < "
              f"{args.min_window + args.max_lag + args.horizon + 1}", file=sys.stderr)
        return 1

    print(f"Загружено {len(df_mom)} наблюдений "
          f"({df_mom.index.min().date()} .. {df_mom.index.max().date()})", file=sys.stderr)

    result = grid_search(
        df_mom=df_mom,
        target=args.target,
        endog_pool=args.endog.split(","),
        exog_cols=args.exog.split(",") if args.exog else [],
        max_lag=args.max_lag,
        horizon=args.horizon,
        min_window=args.min_window,
    )

    # Лучшие по AFE на h=1
    h1_col = f"AFE_h1"
    ok = result[result["status"] == "ok"].copy()
    if ok.empty:
        print("Ни одной валидной спецификации (см. status-колонку):", file=sys.stderr)
        print(result.to_string(index=False))
        return 1
    ok = ok.sort_values(h1_col, ascending=True)

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        ok.to_csv(args.out, index=False)
        print(f"Сохранено: {args.out}", file=sys.stderr)

    print(f"\nТоп-20 спецификаций по AFE на h=1 (всего валидных: {len(ok)}):\n")
    cols = ["endog", "exog", "lag"] + [c for c in ok.columns if c.startswith(("AFE_", "MSFE_", "MAPFE_"))]
    print(ok[cols].head(20).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
