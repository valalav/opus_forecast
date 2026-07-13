"""Weekly bridge nowcast diagnostics for fresh Rosstat semicolon files.

The canonical weekly loader works with the historical normalized CSV.  Fresh
operational uploads currently arrive as ``data/Сравнение еженедельных цен_01.csv``;
this module keeps that path separate and produces diagnostic bridge signals
without registering them as production model forecasts.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, cast

import numpy as np
import pandas as pd

from .weekly_loader import COMPONENT_WEIGHTS


DEFAULT_SEMICOLON_WEEKLY_PATH = Path("data") / "Сравнение еженедельных цен_01.csv"
DEFAULT_DECAY_FACTOR = 0.6
DEFAULT_WEEKS_PER_MONTH = 4


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def get_semicolon_weekly_path(path: Optional[str | Path] = None) -> Path:
    """Return the fresh semicolon weekly file path."""
    if path is not None:
        candidate = Path(path)
        return candidate if candidate.is_absolute() else _project_root() / candidate

    cwd_candidate = Path.cwd() / DEFAULT_SEMICOLON_WEEKLY_PATH
    if cwd_candidate.exists():
        return cwd_candidate


    root_candidate = _project_root() / DEFAULT_SEMICOLON_WEEKLY_PATH
    if root_candidate.exists():
        return root_candidate


    raise FileNotFoundError(
        f"Fresh weekly semicolon file not found: {DEFAULT_SEMICOLON_WEEKLY_PATH}"
    )


def _parse_russian_float(value: Any) -> float:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return np.nan
    text = str(value).strip().replace("\xa0", "").replace(" ", "").replace(",", ".")
    if not text:
        return np.nan
    try:
        return float(text)
    except ValueError:
        return np.nan


def classify_component(component: Any) -> Optional[str]:
    """Map Russian component labels to internal component keys."""
    text = str(component or "").strip().lower()
    if "непродовольств" in text:
        return "nonfood"
    if "продовольств" in text:
        return "food"
    if "услуг" in text:
        return "services"
    return None


def _first_existing_column(columns: Iterable[str], candidates: Iterable[str]) -> str:
    existing = {col.strip(): col for col in columns}
    for candidate in candidates:
        if candidate in existing:
            return existing[candidate]
    raise KeyError(f"None of the expected columns found: {', '.join(candidates)}")


def load_semicolon_weekly_prices(path: Optional[str | Path] = None) -> pd.DataFrame:
    """Load fresh weekly prices from the row-wise Rosstat semicolon export.

    Rows are deduplicated by ``(date, item_key)`` because recent files contain
    repeated blocks for the same week.  The returned frame uses normalized
    columns suitable for bridge calculations.
    """
    data_path = get_semicolon_weekly_path(path)
    raw = pd.read_csv(data_path, sep=";", encoding="utf-8-sig", dtype=str)
    raw.columns = [col.strip() for col in raw.columns]

    date_col = _first_existing_column(raw.columns, ["Date", "Name"])
    name_col = _first_existing_column(raw.columns, ["Наименование", "Наименование "])
    price_col = _first_existing_column(raw.columns, ["Средние цены, рублей"])
    change_col = _first_existing_column(
        raw.columns, ["Изменение цен, в % к предыдущей неделе"]
    )
    component_col = _first_existing_column(
        raw.columns, ["Справка_нед.Компоненты", "Компонент"]
    )
    item_col = "№" if "№" in raw.columns else None

    df = pd.DataFrame(
        {
            "date": pd.to_datetime(raw[date_col], format="%d.%m.%Y", errors="coerce"),
            "product_name": raw[name_col].fillna("").astype(str).str.strip(),
            "price": raw[price_col].map(_parse_russian_float),
            "change_index": raw[change_col].map(_parse_russian_float),
            "component_raw": raw[component_col].fillna("").astype(str).str.strip(),
        }
    )
    if item_col:
        item_values = raw[item_col].fillna("").astype(str).str.strip()
    else:
        item_values = pd.Series([""] * len(raw), index=raw.index)

    fallback_key = cast(pd.Series, df["product_name"]).str.casefold()
    df["item_key"] = item_values.where(item_values != "", fallback_key)
    df["component"] = cast(pd.Series, df["component_raw"]).map(classify_component)
    df["source_file"] = str(data_path)

    before = len(df)
    df = df.dropna(subset=["date"])
    df = df[df["product_name"] != ""]
    df = cast(pd.DataFrame, df[cast(pd.Series, df["component"]).notna()])
    df = df.drop_duplicates(subset=["date", "item_key"], keep="first")
    df = df.sort_values(["date", "item_key"]).reset_index(drop=True)

    df.attrs["raw_rows"] = before
    df.attrs["deduped_rows"] = len(df)
    df.attrs["duplicates_removed"] = before - len(df)
    df.attrs["source_file"] = str(data_path)
    return df


def _normalised_weights(weights: Optional[Dict[str, float]] = None) -> Dict[str, float]:
    selected = dict(weights or COMPONENT_WEIGHTS)
    total = sum(selected.get(component, 0.0) for component in COMPONENT_WEIGHTS)
    if total <= 0:
        raise ValueError("Component weights must have positive total weight")
    return {component: selected.get(component, 0.0) / total for component in COMPONENT_WEIGHTS}


def _component_index(
    data: pd.DataFrame,
    value_col: str,
    weights: Dict[str, float],
) -> Tuple[float, Dict[str, Dict[str, float]]]:
    component_stats: Dict[str, Dict[str, float]] = {}
    weighted_index = 0.0

    for component, weight in weights.items():
        raw_values = cast(
            pd.Series,
            data.loc[cast(pd.Series, data["component"]) == component, value_col],
        )
        values = cast(pd.Series, pd.to_numeric(raw_values, errors="coerce")).dropna()
        mean_index = float(values.mean()) if len(values) else 100.0
        component_stats[component] = {
            "index": mean_index,
            "mom": mean_index - 100.0,
            "n_items": int(len(values)),
        }
        weighted_index += mean_index * weight

    return float(weighted_index), component_stats


def _month_bounds(target_month: str | pd.Timestamp) -> Tuple[pd.Timestamp, pd.Timestamp]:
    timestamp = cast(pd.Timestamp, pd.Timestamp(target_month))
    month_start = timestamp.to_period("M").to_timestamp()
    month_end = month_start + pd.DateOffset(months=1)
    return month_start, month_end


def _safe_round(value: Any, digits: int = 6) -> Optional[float]:
    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(numeric):
        return None
    return round(numeric, digits)


def _top_drivers(
    matched: pd.DataFrame,
    weights: Dict[str, float],
    limit: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    details = matched.copy()
    details["n_in_component"] = details.groupby("component")["item_key"].transform("count")
    details["component_weight"] = cast(pd.Series, details["component"]).map(
        lambda component: weights.get(str(component), 0.0)
    )
    details["approx_contribution_pp"] = (
        (details["item_index"] - 100.0)
        * details["component_weight"]
        / details["n_in_component"]
    )

    def row_to_dict(row: pd.Series) -> Dict[str, Any]:
        return {
            "product_name": str(row["product_name"]),
            "component": str(row["component"]),
            "index": _safe_round(row["item_index"], 4),
            "mom": _safe_round(row["item_index"] - 100.0, 4),
            "price_base": _safe_round(row["price_base"], 4),
            "price_end": _safe_round(row["price_end"], 4),
            "approx_contribution_pp": _safe_round(row["approx_contribution_pp"], 6),
        }

    decreases = [
        row_to_dict(row)
        for _, row in details.nsmallest(limit, "item_index").iterrows()
    ]
    increases = [
        row_to_dict(row)
        for _, row in details.nlargest(limit, "item_index").iterrows()
    ]
    return decreases, increases


def _price_level_bridge(
    df: pd.DataFrame,
    base_date: pd.Timestamp,
    end_date: pd.Timestamp,
    weights: Dict[str, float],
    driver_limit: int,
) -> Dict[str, Any]:
    base_mask = cast(pd.Series, df["date"]) == base_date
    base = cast(
        pd.DataFrame,
        df.loc[base_mask, ["item_key", "product_name", "component", "price"]].copy(),
    )
    base["price_base"] = base["price"]
    base = cast(
        pd.DataFrame,
        base[["item_key", "product_name", "component", "price_base"]].copy(),
    )

    end_mask = cast(pd.Series, df["date"]) == end_date
    end = cast(pd.DataFrame, df.loc[end_mask, ["item_key", "price"]].copy())
    end["price_end"] = end["price"]
    end = cast(pd.DataFrame, end[["item_key", "price_end"]].copy())

    matched = base.merge(end, on="item_key", how="inner")
    matched = cast(
        pd.DataFrame,
        matched[
            (cast(pd.Series, matched["price_base"]) > 0)
            & (cast(pd.Series, matched["price_end"]) > 0)
        ].copy(),
    )
    matched["item_index"] = matched["price_end"] / matched["price_base"] * 100.0

    weighted_index, components = _component_index(matched, "item_index", weights)
    decreases, increases = _top_drivers(matched, weights, driver_limit)

    return {
        "base_date": base_date.strftime("%Y-%m-%d"),
        "end_date": end_date.strftime("%Y-%m-%d"),
        "index": _safe_round(weighted_index),
        "mom": _safe_round(weighted_index - 100.0),
        "matched_items": int(len(matched)),
        "components": components,
        "top_decreases": decreases,
        "top_increases": increases,
    }


def compute_weekly_bridge_nowcast(
    target_month: str | pd.Timestamp,
    df: Optional[pd.DataFrame] = None,
    data_path: Optional[str | Path] = None,
    weights: Optional[Dict[str, float]] = None,
    decay_factor: float = DEFAULT_DECAY_FACTOR,
    weeks_per_month: int = DEFAULT_WEEKS_PER_MONTH,
    driver_limit: int = 12,
) -> Dict[str, Any]:
    """Compute diagnostic weekly bridge signals for one target month."""
    weekly = df.copy() if df is not None else load_semicolon_weekly_prices(data_path)
    component_weights = _normalised_weights(weights)
    month_start, month_end = _month_bounds(target_month)

    month_data = weekly[(weekly["date"] >= month_start) & (weekly["date"] < month_end)]
    result: Dict[str, Any] = {
        "target_month": month_start.strftime("%Y-%m"),
        "source_file": str(weekly.attrs.get("source_file", data_path or DEFAULT_SEMICOLON_WEEKLY_PATH)),
        "raw_rows": int(weekly.attrs.get("raw_rows", len(weekly))),
        "deduped_rows": int(weekly.attrs.get("deduped_rows", len(weekly))),
        "duplicates_removed": int(weekly.attrs.get("duplicates_removed", 0)),
        "weights": {key: _safe_round(value) for key, value in component_weights.items()},
        "status": "ok",
    }

    if month_data.empty:
        result["status"] = "no_weekly_data_for_month"
        return result

    weekly_rows: List[Dict[str, Any]] = []
    weekly_indices: List[float] = []
    cumulative = 1.0
    month_dates = cast(pd.Series, month_data["date"]).dropna().unique()
    for week_date in sorted(month_dates):
        week_ts = cast(pd.Timestamp, pd.Timestamp(week_date))
        week_data = cast(pd.DataFrame, month_data[month_data["date"] == week_ts])
        weighted_index, components = _component_index(week_data, "change_index", component_weights)
        weekly_indices.append(weighted_index)
        cumulative *= weighted_index / 100.0
        weekly_rows.append(
            {
                "date": week_ts.strftime("%Y-%m-%d"),
                "index": _safe_round(weighted_index),
                "mom": _safe_round(weighted_index - 100.0),
                "cumulative_mom": _safe_round((cumulative - 1.0) * 100.0),
                "n_items": int(len(week_data)),
                "components": components,
            }
        )

    cumulative_mom = (cumulative - 1.0) * 100.0
    avg_weekly_change = float(np.mean([idx - 100.0 for idx in weekly_indices]))
    remaining_weeks = max(0, weeks_per_month - len(weekly_indices))
    extrapolated_mom = cumulative_mom + avg_weekly_change * decay_factor * remaining_weeks
    result["chain"] = {
        "weeks_count": int(len(weekly_indices)),
        "weeks": weekly_rows,
        "index": _safe_round(cumulative * 100.0),
        "mom": _safe_round(cumulative_mom),
        "avg_weekly_mom": _safe_round(avg_weekly_change),
        "remaining_weeks": int(remaining_weeks),
        "decay_factor": _safe_round(decay_factor),
        "extrapolated_mom": _safe_round(extrapolated_mom),
    }

    weekly_dates = cast(pd.Series, weekly["date"])
    previous_dates = cast(pd.Series, weekly.loc[weekly_dates < month_start, "date"])
    if not previous_dates.empty:
        base_date = cast(pd.Timestamp, pd.Timestamp(previous_dates.max()))
        end_date = cast(pd.Timestamp, pd.Timestamp(cast(pd.Series, month_data["date"]).max()))
        result["month_end"] = _price_level_bridge(
            weekly, base_date, end_date, component_weights, driver_limit
        )

    next_month_data = cast(
        pd.DataFrame,
        weekly[
            (weekly_dates >= month_end)
            & (weekly_dates < month_end + pd.DateOffset(months=1))
        ],
    )
    if "month_end" in result and not next_month_data.empty:
        next_date = cast(pd.Timestamp, pd.Timestamp(cast(pd.Series, next_month_data["date"]).min()))
        end_date = cast(pd.Timestamp, pd.Timestamp(cast(pd.Series, month_data["date"]).max()))
        result["first_next_week_vs_month_end"] = _price_level_bridge(
            weekly, end_date, next_date, component_weights, driver_limit
        )
        if not previous_dates.empty:
            result["first_next_week_vs_prev_month_end"] = _price_level_bridge(
                weekly,
                cast(pd.Timestamp, pd.Timestamp(previous_dates.max())),
                next_date,
                component_weights,
                driver_limit,
            )

    return result


def compute_weekly_bridge_for_months(
    months: Iterable[str | pd.Timestamp],
    data_path: Optional[str | Path] = None,
    weights: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """Compute bridge diagnostics for multiple forecast months."""
    weekly = load_semicolon_weekly_prices(data_path)
    by_month: Dict[str, Dict[str, Any]] = {}
    for month in months:
        bridge = compute_weekly_bridge_nowcast(month, df=weekly, weights=weights)
        by_month[bridge["target_month"]] = bridge

    return {
        "method": "weekly_bridge_v1",
        "description": (
            "Diagnostic weekly bridge from fresh semicolon Rosstat rows; "
            "not a production model and not included in Ensemble."
        ),
        "source_file": str(weekly.attrs.get("source_file", data_path or DEFAULT_SEMICOLON_WEEKLY_PATH)),
        "raw_rows": int(weekly.attrs.get("raw_rows", len(weekly))),
        "deduped_rows": int(weekly.attrs.get("deduped_rows", len(weekly))),
        "duplicates_removed": int(weekly.attrs.get("duplicates_removed", 0)),
        "data_date_range": {
            "start": cast(
                pd.Timestamp, pd.Timestamp(cast(pd.Series, weekly["date"]).min())
            ).strftime("%Y-%m-%d"),
            "end": cast(
                pd.Timestamp, pd.Timestamp(cast(pd.Series, weekly["date"]).max())
            ).strftime("%Y-%m-%d"),
        },
        "by_month": by_month,
    }


def weekly_model_blend_weights(weeks_count: int) -> Tuple[float, float]:
    """Return weekly/model blend weights used for the clean auxiliary Nowcast."""
    weight_map = {
        1: (0.60, 0.40),
        2: (0.70, 0.30),
        3: (0.80, 0.20),
        4: (0.90, 0.10),
    }
    return weight_map.get(weeks_count, (0.70, 0.30))
