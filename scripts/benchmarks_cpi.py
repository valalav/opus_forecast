"""
Python-порт логики ВВГУ `cpi_fcst.R` — rolling OOS бэктест CPI с 7 бенчмарками
и grid-VAR.

Источник: _inbox/var_extracted/var/Модели для краткосрочного прогноза инфляции (без
дезагрегации)/cpi_fcst.R (R 4.1.2, 102 строки). Автор — Напалков В.В., ВВГУ Банка
России. Оригинал тестировался на их данных (CPI в виде mom%, 2010-2023); здесь
применяется к нашим данным КБР (data/inflation_data.csv) с переводом mom из
шкалы 100+X в mom% через (x - 100).

Что портировано:

  1. Семь univariate-бенчмарков (fsct_simple в R):
       target, naive, mean6, ar1, argap, arima, direct.
  2. Grid VAR (fcst_var в R) — для каждой точки out-of-sample считается набор
     VAR-спецификаций = period × all-subsets-of-regressors × lags × const.
  3. Rolling-forecast (hist_fcsts) — для каждой даты >= start_year-1/12
     прогноз на 7 месяцев вперёд.

Что НЕ портировано (и почему):
  - пакет `seasonal` (X-13ARIMA-SEATS) — входы уже в MoM% SA, десезонирование
    на входе не нужно;
  - ARIMA(0,0,3) с seasonal=(3,0,0) — оставлен auto_arima с seasonal=False,
    см. NOTES в коде.

Результат: длинный CSV (точка × горизонт × спецификация × прогноз), плюс сводная
таблица RMSE/MAE/MAPE/Theil по (spec, hor) и текстовый вывод лучших.

Использование:

    # smoke на наших данных КБР:
    python3 scripts/benchmarks_cpi.py

    # только бенчмарки (быстрее):
    python3 scripts/benchmarks_cpi.py --no-var

    # указать окна и горизонт:
    python3 scripts/benchmarks_cpi.py --start-year 2020 --horizon 6

NOTES / VАЖНЫЕ ОТЛИЧИЯ ОТ ОРИГИНАЛА:
  - В R `cpi` идёт в виде mom% (числа). У нас `mom` в шкале 100+X — переводим в mom%.
  - VAR в R использует `type='const'` и пакет vars; здесь — `np.linalg.lstsq`,
    как в нашем grid_search_var.py, чтобы не зависеть от statsmodels.tsa.VAR.
  - `auto.arima` в R: auto.arima(d, 0, 0, 3, 3, 0, 0, ic='aic') — seasonal=False,
    max AR=0, max MA=3, max seasonal AR=3, max seasonal MA=0, IC='aic'. У нас
    pmdarima.auto_arima с seasonal=False, max_p=0, max_q=3.
"""
from __future__ import annotations

import argparse
import itertools
import sys
import time
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


# ----------------------------------------------------------------------
# Загрузка и подготовка КБР-данных
# ----------------------------------------------------------------------

def load_kbr(path: Path, dropna_how: str = "any") -> pd.DataFrame:
    """data/inflation_data.csv в шкале mom% (x - 100) для индексов.

    dropna_how='all' — оставлять наблюдения, в которых хотя бы один ряд непустой.
    dropna_how='any' — dropna-строки с любым пропуском (безопаснее для VAR).
    """
    df = pd.read_csv(path, sep=";", decimal=",")
    df["Date"] = pd.to_datetime(df["Date"], format="%d.%m.%Y", errors="coerce")
    df = df.dropna(subset=["Date"]).set_index("Date").sort_index()

    out = pd.DataFrame(index=df.index)
    # индексы 100+X -> mom%
    for col in ["mom", "Prod", "Nonprod", "Serv", "usd_nom_i"]:
        if col in df.columns:
            out[col] = pd.to_numeric(df[col], errors="coerce") - 100.0
    # уровни (не переводим)
    for col in ["Ki_i", "Ruonia"]:
        if col in df.columns:
            out[col] = pd.to_numeric(df[col], errors="coerce")

    if dropna_how == "any":
        out = out.dropna()
    else:
        out = out.dropna(how="all")
    out.index = out.index.to_period("M").to_timestamp()
    return out


# ----------------------------------------------------------------------
# Утилиты: скользящее окно данных для конкретной точки out-of-sample
# ----------------------------------------------------------------------

def _window_for_period(data: pd.DataFrame, period: str | int) -> pd.DataFrame:
    """Аналог if param$period != 'all' в R. period: 'all' | 'from YYYY' | число лет (str или int)."""
    if period == "all":
        return data
    if isinstance(period, str) and period.startswith("from "):
        year = int(period.split()[1])
        return data[data.index >= pd.Timestamp(f"{year}-01-01")]
    try:
        years = int(period)
    except (TypeError, ValueError):
        raise ValueError(f"Unknown period: {period!r}")
    n = years * 12
    return data.iloc[-n:] if n < len(data) else data


# ----------------------------------------------------------------------
# Бенчмарки (port of fsct_simple)
# ----------------------------------------------------------------------

def bench_target(d: np.ndarray, hor: int) -> np.ndarray:
    """4% годовых / 12 = 0.3333% м/м на все шаги горизонта."""
    return np.full(hor, 4.0 / 12.0)


def bench_naive(d: np.ndarray, hor: int) -> np.ndarray:
    """Последнее наблюдение повторяется на все шаги."""
    return np.full(hor, d[-1])


def bench_mean6(d: np.ndarray, hor: int) -> np.ndarray:
    return np.full(hor, d[-6:].mean())


def bench_ar1(d: np.ndarray, hor: int) -> np.ndarray:
    """AR(1) без константы вокруг среднего: d_t = c + phi * d_{t-1}, оцениваем OLS.
    Эквивалент R: ar.ols(d, aic=F, order.max=1) — fit с intercept по умолчанию.
    Возьмём вариант с intercept (как в R), т.к. demean=F только для argap.
    """
    y = d[1:]
    x = d[:-1]
    x_mean, y_mean = x.mean(), y.mean()
    x_d, y_d = x - x_mean, y - y_mean
    denom = (x_d ** 2).sum()
    if denom < 1e-12:
        return np.full(hor, d[-1])
    phi = (x_d * y_d).sum() / denom
    c = y_mean - phi * x_mean
    last = d[-1]
    out = np.empty(hor)
    for h in range(hor):
        out[h] = c + phi * last
        last = out[h]
    return out


def bench_argap(d: np.ndarray, hor: int) -> np.ndarray:
    """AR(1) вокруг 4%/12 (target), без вычета среднего (demean=False в R)."""
    target = 4.0 / 12.0
    y = d[1:] - target
    x = d[:-1] - target
    # AR(1) без интерсепта: y_t = phi * x_t
    denom = (x ** 2).sum()
    if denom < 1e-12:
        return np.full(hor, target)
    phi = (x * y).sum() / denom
    last = d[-1] - target
    out = np.empty(hor)
    for h in range(hor):
        out[h] = phi * last
        last = out[h]
    return out + target


def bench_arima(d: np.ndarray, hor: int, max_p: int = 0, max_q: int = 3) -> np.ndarray:
    """Порт R: auto.arima(d, 0, 0, 3, 3, 0, 0, ic='aic'). Сезонность выключена.
    Использует pmdarima если есть; иначе — AR(1) как fallback.
    """
    try:
        import pmdarima  # type: ignore
        model = pmdarima.auto_arima(
            d,
            start_p=0, start_q=0,
            max_p=max_p, max_q=max_q,
            seasonal=False, d=0,
            information_criterion="aic",
            suppress_warnings=True,
            error_action="ignore",
        )
        return np.asarray(model.predict(n_periods=hor))
    except ImportError:
        return bench_ar1(d, hor)


def bench_direct(d: np.ndarray, hor: int) -> np.ndarray:
    """Direct multi-step: для каждого h, d_t = a_h + b_h * d_{t-h}.
    Прогноз на h шагов: a_h + b_h * d_{last}.
    Эквивалент R: coef(lm(d_lags[,1] ~ d_lags[,1+x])) %*% c(1, tail(d_lags[,1], 1)).
    """
    n = len(d)
    out = np.empty(hor)
    for x in range(1, hor + 1):
        if n - x < 2:
            out[x - 1] = d[-1]
            continue
        y = d[x:]
        z = d[:n - x]
        z_mean, y_mean = z.mean(), y.mean()
        denom = ((z - z_mean) ** 2).sum()
        if denom < 1e-12:
            out[x - 1] = y_mean
            continue
        b = ((z - z_mean) * (y - y_mean)).sum() / denom
        a = y_mean - b * z_mean
        out[x - 1] = a + b * d[-1]
    return out


BENCHMARKS = {
    "target": bench_target,
    "naive":  bench_naive,
    "mean6":  bench_mean6,
    "ar1":    bench_ar1,
    "argap":  bench_argap,
    "arima":  bench_arima,
    "direct": bench_direct,
}


# ----------------------------------------------------------------------
# VAR (port of fcst_var + VAR subset grid)
# ----------------------------------------------------------------------

def _var_design(data: np.ndarray, lags: int, exog: np.ndarray | None) -> Tuple[np.ndarray, np.ndarray]:
    """X = [1, y_{t-1}, ..., y_{t-lags}, exog_t]; Y = y_t. Вектор-строки."""
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


def _var_dynamic_forecast(data: np.ndarray, lags: int, exog: np.ndarray | None, horizon: int) -> np.ndarray:
    """Возвращает матрицу (horizon, k_endog) — динамический прогноз."""
    x, y = _var_design(data, lags, exog)
    if len(x) < y.shape[1] + 1:
        return np.full((horizon, y.shape[1]), np.nan)
    betas = np.linalg.lstsq(x, y, rcond=None)[0]
    hist = [data[t] for t in range(len(data) - lags, len(data))]
    out = np.empty((horizon, data.shape[1]))
    for h in range(horizon):
        row = [1.0]
        for lag in range(1, lags + 1):
            row.extend(hist[-lag])
        if exog is not None:
            row.extend(exog[-1])
        pred = (np.asarray([row]) @ betas).ravel()
        out[h] = pred
        hist.append(pred)
    return out


def _all_subsets(items: Sequence[str]) -> Iterable[Tuple[str, ...]]:
    """Все непустые подмножества (как combn в R), по возрастанию длины."""
    items = list(items)
    for k in range(1, len(items) + 1):
        for combo in itertools.combinations(items, k):
            yield combo


# ----------------------------------------------------------------------
# Прогноз на конкретную точку out-of-sample
# ----------------------------------------------------------------------

def forecast_at(
    df: pd.DataFrame,
    end: pd.Timestamp,
    target: str,
    regressor_pool: Sequence[str],
    periods: Sequence,
    lags_options: Sequence[int],
    horizon: int,
) -> pd.DataFrame:
    """Прогнозы на `horizon` шагов вперёд от точки `end` (включительно)."""
    history = df.loc[df.index <= end].copy()
    target_series = history[target].to_numpy(dtype=float)
    rows: List[dict] = []

    # Бенчмарки (каждый прогоняется на каждом из 3 окон — как в R)
    for period in periods:
        window = _window_for_period(history, period)
        if target not in window.columns or len(window[target].dropna()) < 3:
            continue
        d = window[target].dropna().to_numpy(dtype=float)
        for name, fn in BENCHMARKS.items():
            try:
                fc = fn(d, horizon)
            except Exception:
                fc = np.full(horizon, np.nan)
            for h in range(horizon):
                rows.append({
                    "end": end, "hor": h + 1,
                    "period": str(period), "spec": name,
                    "const": "-", "lags": "-",
                    "fcst": float(fc[h]),
                })

    # VAR grid: все подмножества regressor_pool × лаги × 3 period
    for period in periods:
        window = _window_for_period(history, period)
        if len(window) < 8:
            continue
        for subset in _all_subsets(regressor_pool):
            endogs = [target] + list(subset)
            data_block = window[endogs].dropna()
            if len(data_block) < 8:
                continue
            data_arr = data_block.to_numpy(dtype=float)
            for lag in lags_options:
                try:
                    pred = _var_dynamic_forecast(data_arr, lag, exog=None, horizon=horizon)
                    target_idx = 0  # target идёт первым столбцом в endogs
                except Exception:
                    pred = np.full((horizon, len(endogs)), np.nan)
                    target_idx = 0
                for h in range(horizon):
                    rows.append({
                        "end": end, "hor": h + 1,
                        "period": str(period),
                        "spec": f"var_{'+'.join(subset) or 'none'}",
                        "const": "const", "lags": lag,
                        "fcst": float(pred[h, target_idx]) if np.isfinite(pred[h, target_idx]) else np.nan,
                    })

    return pd.DataFrame(rows)


def rolling_forecasts(
    df: pd.DataFrame,
    target: str,
    regressor_pool: Sequence[str],
    periods: Sequence,
    lags_options: Sequence[int],
    horizon: int,
    start_year: int,
    run_var: bool = True,
) -> pd.DataFrame:
    """Прогон по всем cut-off'ам с start_year-1/12, как hist_fcsts в R."""
    if not run_var:
        # бенчмарки считаются в forecast_at всегда (это быстро), VAR-часть отбрасываем
        pass

    cutoff_start = pd.Timestamp(f"{start_year - 1}-{12:02d}-01")
    ends = [idx for idx in df.index if idx >= cutoff_start]
    parts: List[pd.DataFrame] = []
    t0 = time.time()
    for k, end in enumerate(ends):
        df_fc = forecast_at(df, end, target, regressor_pool, periods, lags_options, horizon)
        if not run_var:
            df_fc = df_fc[df_fc["spec"].isin(BENCHMARKS.keys())]
        parts.append(df_fc)
        if (k + 1) % 12 == 0 or k == 0:
            print(f"  ... {k+1}/{len(ends)} cutoffs, "
                  f"rows={sum(len(p) for p in parts)}, "
                  f"elapsed={time.time()-t0:.1f}s", file=sys.stderr)
    out = pd.concat(parts, ignore_index=True)
    print(f"Total: {len(ends)} cutoffs × horizon={horizon} × {len(BENCHMARKS)} бенчмарков "
          f"+ VAR grid = {len(out)} rows, {time.time()-t0:.1f}s", file=sys.stderr)
    return out


# ----------------------------------------------------------------------
# Метрики
# ----------------------------------------------------------------------

def _theil_u(actual: np.ndarray, fcst: np.ndarray) -> float:
    """Theil's U = RMSE(model) / RMSE(naive=last-actual).
    < 1 — лучше наивного, > 1 — хуже. Naive baseline = последний actual."""
    if len(actual) < 2:
        return np.nan
    rmse_model = np.sqrt(np.nanmean((actual - fcst) ** 2))
    naive_pred = np.concatenate([[actual[0]], actual[:-1]])  # F_{t-1} = y_{t-1}
    rmse_naive = np.sqrt(np.nanmean((actual - naive_pred) ** 2))
    if rmse_naive < 1e-12:
        return np.nan
    return float(rmse_model / rmse_naive)


def evaluate(fcst_df: pd.DataFrame, actuals: pd.Series, target: str) -> pd.DataFrame:
    """Для каждой (period, spec, hor) считает RMSE/MAE/MAPE/Theil vs actuals.
    actuals: pd.Series с DatetimeIndex, target=mom% (тот же формат, что и fcst).
    """
    out_rows = []
    for (period, spec, hor), grp in fcst_df.groupby(["period", "spec", "hor"], observed=True):
        if "end" in grp.columns:
            ends = pd.to_datetime(grp["end"])
            fcst_dates = pd.DatetimeIndex(ends + pd.DateOffset(months=int(hor)))
            fcst_dates = fcst_dates.to_period("M").to_timestamp()
        else:
            continue
        actual_vals = actuals.reindex(fcst_dates).to_numpy(dtype=float)
        fcst_vals = grp["fcst"].to_numpy(dtype=float)
        mask = np.isfinite(actual_vals) & np.isfinite(fcst_vals)
        if mask.sum() < 3:
            continue
        a, f = actual_vals[mask], fcst_vals[mask]
        err = a - f
        rmse = float(np.sqrt(np.mean(err ** 2)))
        mae = float(np.mean(np.abs(err)))
        denom = np.where(np.abs(a) < 1.0, 1.0, np.abs(a))
        mape = float(np.mean(np.abs(err) / denom) * 100.0)
        theil = _theil_u(a, f)
        out_rows.append({
            "period": period, "spec": spec, "hor": int(hor),
            "n": int(mask.sum()),
            "rmse": rmse, "mae": mae, "mape": mape, "theil": theil,
        })
    return pd.DataFrame(out_rows).sort_values(["hor", "mae"])


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------

DEFAULT_TARGET = "mom"
DEFAULT_REGRESSORS = ["Prod", "Nonprod", "Serv", "usd_nom_i", "Ki_i", "Ruonia"]


def main() -> int:
    p = argparse.ArgumentParser(description="Port of VVGU cpi_fcst.R — rolling OOS benchmarks + VAR grid")
    p.add_argument("--data", type=Path,
                   default=Path(__file__).resolve().parents[1] / "data" / "inflation_data.csv")
    p.add_argument("--target", default=DEFAULT_TARGET)
    p.add_argument("--regressors", default=",".join(DEFAULT_REGRESSORS))
    p.add_argument("--horizon", type=int, default=7)
    p.add_argument("--start-year", type=int, default=2019)
    p.add_argument("--periods", default="all,from 2016,5",
                   help="Через запятую: 'all' | 'from YYYY' | N (лет)")
    p.add_argument("--lags", default="1,3", help="Через запятую, лаги VAR")
    p.add_argument("--no-var", action="store_true",
                   help="Не считать VAR grid (только 7 бенчмарков, быстро)")
    p.add_argument("--out-fcst", type=Path, default=None, help="CSV со всеми прогнозами")
    p.add_argument("--out-eval", type=Path, default=None, help="CSV со сводными метриками")
    args = p.parse_args()

    periods = [p.strip() for p in args.periods.split(",")]
    lags_options = [int(x.strip()) for x in args.lags.split(",")]

    print(f"Loading {args.data} ...", file=sys.stderr)
    df = load_kbr(args.data, dropna_how="any")
    print(f"  shape={df.shape}, range={df.index.min().date()}..{df.index.max().date()}", file=sys.stderr)
    print(f"  target={args.target!r}, regressors={args.regressors}", file=sys.stderr)
    print(f"  horizon={args.horizon}, start_year={args.start_year}, "
          f"periods={periods}, lags={lags_options}, no_var={args.no_var}", file=sys.stderr)

    fcst = rolling_forecasts(
        df=df,
        target=args.target,
        regressor_pool=args.regressors.split(","),
        periods=periods,
        lags_options=lags_options,
        horizon=args.horizon,
        start_year=args.start_year,
        run_var=not args.no_var,
    )

    if args.out_fcst is not None:
        args.out_fcst.parent.mkdir(parents=True, exist_ok=True)
        fcst.to_csv(args.out_fcst, index=False)
        print(f"Saved forecasts: {args.out_fcst} ({len(fcst)} rows)", file=sys.stderr)

    # Сводные метрики: actuals = mom% (target) из исходных данных
    actuals = df[args.target]
    eval_df = evaluate(fcst, actuals, args.target)
    if args.out_eval is not None:
        args.out_eval.parent.mkdir(parents=True, exist_ok=True)
        eval_df.to_csv(args.out_eval, index=False)
        print(f"Saved metrics: {args.out_eval} ({len(eval_df)} rows)", file=sys.stderr)

    print("\n=== TOP-15 specs на h=1 (по MAE, среди бенчмарков) ===", file=sys.stderr, flush=True)
    h1 = eval_df[eval_df["hor"] == 1]
    non_trivial_bench = h1["spec"].isin(BENCHMARKS.keys()) & ~((h1["spec"] == "direct") & (h1["hor"] == 1))
    bench_h1 = h1[non_trivial_bench].sort_values("mae").head(15)
    print(bench_h1.to_string(index=False), flush=True)

    print("\n=== TOP-15 specs на h=1 (по MAE, среди VAR) ===", file=sys.stderr, flush=True)
    var_mask = h1["spec"].str.startswith("var_")
    var_h1 = h1[var_mask].sort_values("mae").head(15)
    if var_h1.empty:
        print("(нет VAR — запустите без --no-var)", flush=True)
    else:
        print(var_h1.to_string(index=False), flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
