"""Long-run diagnostics for the experimental core inflation tool."""

from __future__ import annotations

import math

import pandas as pd


INDICATOR_COLUMNS = [
    "headline_mom",
    "exclusion_core_raw_mom",
    "exclusion_core_mom",
    "trimmed_mean_mom",
    "weighted_median_mom",
    "stable_core_signal_mom",
    "stable_core_mom",
    "headline_sa_mom",
    "exclusion_core_sa_raw_mom",
    "exclusion_core_sa_mom",
    "trimmed_mean_sa_mom",
    "weighted_median_sa_mom",
    "stable_core_sa_signal_mom",
    "stable_core_sa_mom",
]


def annualize_monthly_rate(monthly_rate: float) -> float:
    """Annualize a monthly percent rate."""

    if not math.isfinite(monthly_rate):
        return float("nan")
    return ((1.0 + monthly_rate / 100.0) ** 12 - 1.0) * 100.0



def causal_robust_filter(
    values: pd.Series,
    *,
    alpha: float = 0.35,
    max_innovation_pp: float = 1.0,
) -> pd.Series:
    """Estimate a slow-moving inflation rate without using future observations.

    Each update absorbs ``alpha`` of the current innovation. Large innovations
    are clipped before the update so that a one-month relative-price shock
    cannot dominate the estimated persistent rate.
    """

    if not 0.0 < alpha <= 1.0:
        raise ValueError("alpha must be in (0, 1]")
    if max_innovation_pp <= 0.0 or not math.isfinite(max_innovation_pp):
        raise ValueError("max_innovation_pp must be finite and positive")

    numeric = pd.to_numeric(values, errors="coerce")
    filtered = pd.Series(float("nan"), index=numeric.index, dtype=float)
    state = float("nan")
    for index, value in numeric.items():
        if not math.isfinite(value):
            continue
        if not math.isfinite(state):
            state = float(value)
        else:
            innovation = max(-max_innovation_pp, min(max_innovation_pp, float(value) - state))
            state += alpha * innovation
        filtered.loc[index] = state
    return filtered


def rolling_12m_rate(monthly_rates: pd.Series) -> pd.Series:
    """Return a rolling 12-month cumulative annual rate from monthly percent rates."""

    rates = pd.to_numeric(monthly_rates, errors="coerce")
    return (1.0 + rates / 100.0).rolling(12, min_periods=12).apply(lambda values: values.prod(), raw=True).sub(1.0).mul(100.0)


def add_stable_rate_metrics(series: pd.DataFrame) -> pd.DataFrame:
    """Add 3MMA and annual stable-inflation metrics to a monthly series."""

    out = series.copy()
    for column in ["stable_core_mom", "stable_core_sa_mom"]:
        if column not in out.columns:
            continue
        stem = column.removesuffix("_mom")
        mom = pd.to_numeric(out[column], errors="coerce")
        out[f"{stem}_3mma"] = mom.rolling(3, min_periods=3).mean()
        out[f"{stem}_3mma_annualized"] = out[f"{stem}_3mma"].map(annualize_monthly_rate)
        out[f"{stem}_12m"] = rolling_12m_rate(mom)
    return out


def future_average(series: pd.Series, horizon: int = 12) -> pd.Series:
    """Return the average of the next ``horizon`` observations for each row."""

    shifted = [series.shift(-step) for step in range(1, horizon + 1)]
    return pd.concat(shifted, axis=1).mean(axis=1, skipna=False)


def build_longrun_metrics(series: pd.DataFrame, horizon: int = 12) -> pd.DataFrame:
    """Measure smoothness and practical long-horizon usefulness of indicators."""

    if "date" not in series.columns or "headline_mom" not in series.columns:
        raise ValueError("series must contain date and headline_mom")

    frame = series.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame = frame.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    headline = pd.to_numeric(frame["headline_mom"], errors="coerce")
    target = future_average(headline, horizon=horizon)
    baseline_mae = (headline - target).abs().mean()

    rows: list[dict[str, object]] = []
    for column in [col for col in INDICATOR_COLUMNS if col in frame.columns]:
        values = pd.to_numeric(frame[column], errors="coerce")
        valid = values.notna()
        if not valid.any():
            continue
        changes = values.diff()
        compare_mask = valid & headline.notna()
        forecast_mask = valid & target.notna()
        mae_next_h = float((values.loc[forecast_mask] - target.loc[forecast_mask]).abs().mean()) if forecast_mask.any() else float("nan")
        corr = float("nan")
        if compare_mask.sum() >= 3:
            left = values.loc[compare_mask]
            right = headline.loc[compare_mask]
            if left.std(ddof=0) > 0 and right.std(ddof=0) > 0:
                corr = float(left.corr(right))
        rows.append(
            {
                "indicator": column,
                "start_date": frame.loc[valid, "date"].min().date().isoformat(),
                "end_date": frame.loc[valid, "date"].max().date().isoformat(),
                "n_months": int(valid.sum()),
                "mean_mom": float(values.mean()),
                "annualized_mean": annualize_monthly_rate(float(values.mean())),
                "std_mom": float(values.std(ddof=0)),
                "avg_abs_monthly_change": float(changes.abs().mean()),
                "max_abs_monthly_change": float(changes.abs().max()),
                "corr_with_headline": corr,
                f"mae_vs_next_{horizon}m_avg_headline": mae_next_h,
                f"mae_gain_vs_headline_next_{horizon}m": float(baseline_mae - mae_next_h)
                if math.isfinite(baseline_mae) and math.isfinite(mae_next_h)
                else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def build_yearly_dynamics(series: pd.DataFrame) -> pd.DataFrame:
    """Summarize yearly dynamics for headline and stable-core indicators."""

    frame = series.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame = frame.dropna(subset=["date"]).sort_values("date")
    frame["year"] = frame["date"].dt.year
    frame = add_stable_rate_metrics(frame)
    columns = [col for col in ["headline_mom", "stable_core_mom", "stable_core_sa_mom"] if col in frame.columns]
    grouped = frame.groupby("year")[columns].mean().reset_index()
    for column in columns:
        grouped[f"{column}_annualized"] = grouped[column].map(annualize_monthly_rate)
    return grouped


def render_dynamics_report(
    series: pd.DataFrame,
    metrics: pd.DataFrame,
    diagnostics: pd.DataFrame,
    *,
    smoothing_alpha: float = 0.35,
    max_innovation_pp: float = 1.0,
    winsor_lower: float = 0.05,
    winsor_upper: float = 0.05,
) -> str:
    """Render a Russian analytical note on stable inflation dynamics."""

    frame = series.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame = frame.dropna(subset=["date"]).sort_values("date")
    frame = add_stable_rate_metrics(frame)
    yearly = build_yearly_dynamics(frame)
    latest = frame.tail(12)
    failed = diagnostics.loc[diagnostics["status"].astype(str).str.lower().eq("fail")] if not diagnostics.empty else diagnostics

    def fmt(value: object, digits: int = 2) -> str:
        try:
            number = float(value)
        except (TypeError, ValueError):
            return ""
        if not math.isfinite(number):
            return ""
        return f"{number:.{digits}f}"

    def latest_value(column: str) -> str:
        if column not in frame.columns or frame[column].dropna().empty:
            return "н/д"
        row = frame.dropna(subset=[column]).iloc[-1]
        return f"{row['date'].date().isoformat()}: {fmt(row[column])}% м/м"

    def latest_rate_pack(stem: str) -> str:
        mom_column = f"{stem}_mom"
        if mom_column not in frame.columns or frame[mom_column].dropna().empty:
            return "н/д"
        row = frame.dropna(subset=[mom_column]).iloc[-1]
        parts = [f"{row['date'].date().isoformat()}: {fmt(row[mom_column])}% м/м"]
        if f"{stem}_3mma" in row.index and math.isfinite(float(row.get(f"{stem}_3mma", float("nan")))):
            parts.append(f"3MMA {fmt(row[f'{stem}_3mma'])}% м/м")
        if f"{stem}_3mma_annualized" in row.index and math.isfinite(float(row.get(f"{stem}_3mma_annualized", float("nan")))):
            parts.append(f"3MMA annualized {fmt(row[f'{stem}_3mma_annualized'])}%")
        if f"{stem}_12m" in row.index and math.isfinite(float(row.get(f"{stem}_12m", float("nan")))):
            parts.append(f"12m {fmt(row[f'{stem}_12m'])}%")
        return "; ".join(parts)

    preferred = "stable_core_sa_mom" if "stable_core_sa_mom" in frame.columns else "stable_core_mom"
    preferred_metrics = metrics.loc[metrics["indicator"].eq(preferred)] if not metrics.empty else pd.DataFrame()
    headline_metrics = metrics.loc[metrics["indicator"].eq("headline_mom")] if not metrics.empty else pd.DataFrame()

    lines = [
        "# Динамика устойчивой инфляции КБР",
        "",
        "Экспериментальная аналитическая записка. Показатель не заменяет официальный ИПЦ; он отделяет устойчивое давление от шумных товарных и тарифных скачков.",
        "",
        "## Метод",
        "",
        "- Обычный ряд строится только по 537 микропозициям `data/micro_sprav.csv`; SA-ряд — по единому непересекающемуся уровню из 44 субкомпонент.",
        "- `exclusion_core_raw` исключает заранее заданные шумные группы и перенормирует веса.",
        f"- `exclusion_core` дополнительно винзоризирует нижние {winsor_lower:.0%} и верхние {winsor_upper:.0%} взвешенного распределения, не удаляя наблюдения из аудиторской таблицы вкладов.",
        "- `trimmed_mean` каждый месяц отсекает нижние и верхние 10% распределения компонент по весу.",
        "- `weighted_median` показывает центральную компоненту распределения.",
        "- `stable_core_signal` усредняет `exclusion_core` и `trimmed_mean`; это входной месячный сигнал.",
        f"- `stable_core` — причинный робастный фильтр сигнала: за месяц он усваивает {smoothing_alpha:.0%} изменения, предварительно ограниченного ±{max_innovation_pp:.2f} п.п.; будущие наблюдения не используются.",
        "- `*_sa` использует сезонно очищенные ряды из `data/mom_sa_kbr.csv`; обычные ряды используются как контроль фактической динамики.",
        "",
        "## Диагностика",
        "",
    ]
    if failed.empty:
        lines.append("Проваленных диагностик нет; предупреждения остаются аналитическими ограничениями, а не блокером расчета.")
    else:
        lines.append("Есть проваленные диагностики, числовой результат нельзя использовать без исправления входов:")
        for _, row in failed.iterrows():
            lines.append(f"- `{row.get('check')}`: {row.get('message')}")
    lines.extend(
        [
            "",
            "## Текущее состояние",
            "",
            f"- Headline: {latest_value('headline_mom')}",
            f"- Устойчивая инфляция, обычные данные: {latest_rate_pack('stable_core')}",
            f"- Устойчивая инфляция, SA: {latest_rate_pack('stable_core_sa')}",
            "",
        ]
    )

    if not preferred_metrics.empty:
        row = preferred_metrics.iloc[0]
        lines.extend(
            [
                "## Долгий горизонт",
                "",
                f"Рабочий показатель `{preferred}` покрывает {int(row['n_months'])} месяцев ({row['start_date']} - {row['end_date']}).",
                f"Средний темп: {fmt(row['mean_mom'])}% м/м, что соответствует {fmt(row['annualized_mean'])}% SAAR/год при постоянном месячном темпе.",
                f"Месячная волатильность: {fmt(row['std_mom'])} п.п.; средний абсолютный скачок к предыдущему месяцу: {fmt(row['avg_abs_monthly_change'])} п.п.",
            ]
        )
        if not headline_metrics.empty:
            hrow = headline_metrics.iloc[0]
            lines.append(
                f"Для сравнения headline имеет волатильность {fmt(hrow['std_mom'])} п.п. и средний абсолютный скачок {fmt(hrow['avg_abs_monthly_change'])} п.п."
            )
        mae_col = next((col for col in row.index if col.startswith("mae_vs_next_")), None)
        gain_col = next((col for col in row.index if col.startswith("mae_gain_vs_headline")), None)
        if mae_col and gain_col:
            lines.append(
                f"Проверка практической полезности: MAE к среднему headline на следующие 12 месяцев = {fmt(row[mae_col])} п.п.; выигрыш относительно текущего headline как наивной оценки = {fmt(row[gain_col])} п.п."
            )
        lines.append("")

    lines.extend(["## Годовая динамика", "", "| год | headline ann. | stable ann. | stable SA ann. |", "|---:|---:|---:|---:|"])
    for _, row in yearly.tail(12).iterrows():
        lines.append(
            "| {year} | {headline} | {stable} | {stable_sa} |".format(
                year=int(row["year"]),
                headline=fmt(row.get("headline_mom_annualized")),
                stable=fmt(row.get("stable_core_mom_annualized")),
                stable_sa=fmt(row.get("stable_core_sa_mom_annualized")),
            )
        )

    if not latest.empty:
        lines.extend(
            [
                "",
                "## Последние 12 месяцев",
                "",
                "| date | headline | stable | stable 3MMA | stable 12m | stable SA | stable SA 3MMA | stable SA 12m | gap SA-headline |",
                "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for _, row in latest.iterrows():
            gap = row.get("stable_core_sa_mom", float("nan")) - row.get("headline_mom", float("nan"))
            lines.append(
                "| {date} | {headline} | {stable} | {stable_3mma} | {stable_12m} | {stable_sa} | {stable_sa_3mma} | {stable_sa_12m} | {gap} |".format(
                    date=row["date"].date().isoformat(),
                    headline=fmt(row.get("headline_mom")),
                    stable=fmt(row.get("stable_core_mom")),
                    stable_3mma=fmt(row.get("stable_core_3mma")),
                    stable_12m=fmt(row.get("stable_core_12m")),
                    stable_sa=fmt(row.get("stable_core_sa_mom")),
                    stable_sa_3mma=fmt(row.get("stable_core_sa_3mma")),
                    stable_sa_12m=fmt(row.get("stable_core_sa_12m")),
                    gap=fmt(gap),
                )
            )
    lines.extend(
        [
            "",
            "## Интерпретация",
            "",
            "Если headline резко расходится с `stable_core_sa`, скачок с большей вероятностью связан с сезонными, тарифными или товарными хвостами. Если оба ряда движутся в одну сторону несколько месяцев подряд, это сигнал более устойчивого инфляционного давления.",
            "",
        ]
    )
    return "\n".join(lines).rstrip() + "\n"
