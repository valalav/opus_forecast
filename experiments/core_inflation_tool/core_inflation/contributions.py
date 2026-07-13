"""Contribution tables for the experimental core inflation tool."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import pandas as pd


@dataclass(frozen=True)
class ContributionColumns:
    """Column names used to build component contribution tables."""

    date: str = "date"
    component: str = "component"
    mom: str = "mom"
    weight: str = "weight"
    excluded: str = "excluded"


def _require_columns(frame: pd.DataFrame, columns: Iterable[str]) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"missing required columns: {', '.join(missing)}")


def build_contribution_table(
    components: pd.DataFrame,
    columns: ContributionColumns | None = None,
) -> pd.DataFrame:
    """Return monthly headline/core contributions for component MoM values.

    Input rows must represent one component in one month. Weights may be in any
    positive scale; the function normalizes them by month for headline and by
    included components for exclusion-core contributions.
    """

    columns = columns or ContributionColumns()
    _require_columns(components, [columns.date, columns.component, columns.mom, columns.weight])

    frame = components.copy()
    if columns.excluded not in frame.columns:
        frame[columns.excluded] = False

    frame[columns.mom] = pd.to_numeric(frame[columns.mom], errors="coerce")
    frame[columns.weight] = pd.to_numeric(frame[columns.weight], errors="coerce")
    if frame[[columns.mom, columns.weight]].isna().any().any():
        raise ValueError("component mom and weight columns must be numeric and non-empty")
    if (frame[columns.weight] < 0).any():
        raise ValueError("component weights must be non-negative")

    excluded = frame[columns.excluded]
    if excluded.dtype == object:
        frame[columns.excluded] = excluded.astype(str).str.lower().isin({"1", "true", "yes", "y"})
    else:
        frame[columns.excluded] = excluded.fillna(False).astype(bool)

    total_weight = frame.groupby(columns.date)[columns.weight].transform("sum")
    included_weight = (
        frame.loc[~frame[columns.excluded]]
        .groupby(columns.date)[columns.weight]
        .sum()
        .rename("_included_weight")
    )
    frame = frame.join(included_weight, on=columns.date)

    if (total_weight <= 0).any():
        bad_dates = frame.loc[total_weight <= 0, columns.date].drop_duplicates().astype(str).tolist()
        raise ValueError(f"non-positive total component weight for dates: {', '.join(bad_dates)}")
    if frame["_included_weight"].isna().any() or (frame["_included_weight"] <= 0).any():
        bad_dates = (
            frame.loc[frame["_included_weight"].isna() | (frame["_included_weight"] <= 0), columns.date]
            .drop_duplicates()
            .astype(str)
            .tolist()
        )
        raise ValueError(f"non-positive included component weight for dates: {', '.join(bad_dates)}")

    frame["headline_weight_share"] = frame[columns.weight] / total_weight
    frame["core_weight_share"] = 0.0
    included_mask = ~frame[columns.excluded]
    frame.loc[included_mask, "core_weight_share"] = (
        frame.loc[included_mask, columns.weight] / frame.loc[included_mask, "_included_weight"]
    )
    frame["headline_contribution_pp"] = frame[columns.mom] * frame["headline_weight_share"]
    frame["core_contribution_pp"] = frame[columns.mom] * frame["core_weight_share"]
    frame["included_weight_sum"] = frame["_included_weight"]
    frame["excluded_weight_sum"] = total_weight - frame["_included_weight"]
    frame["total_weight_sum"] = total_weight

    result = frame.rename(
        columns={
            columns.date: "date",
            columns.component: "component",
            columns.mom: "mom",
            columns.weight: "weight",
            columns.excluded: "excluded",
        }
    )
    ordered = [
        "date",
        "component",
        "mom",
        "weight",
        "excluded",
        "headline_weight_share",
        "core_weight_share",
        "headline_contribution_pp",
        "core_contribution_pp",
        "total_weight_sum",
        "included_weight_sum",
        "excluded_weight_sum",
    ]
    return result[ordered].sort_values(["date", "headline_contribution_pp"], ascending=[True, False])


def build_series_from_contributions(contributions: pd.DataFrame) -> pd.DataFrame:
    """Aggregate contribution rows into the required core inflation series."""

    _require_columns(
        contributions,
        ["date", "headline_contribution_pp", "core_contribution_pp", "excluded_weight_sum"],
    )
    series = (
        contributions.groupby("date", as_index=False)
        .agg(
            headline_mom=("headline_contribution_pp", "sum"),
            exclusion_core_mom=("core_contribution_pp", "sum"),
            total_weight_sum=("total_weight_sum", "first"),
            included_weight_sum=("included_weight_sum", "first"),
            excluded_weight_sum=("excluded_weight_sum", "first"),
        )
        .sort_values("date")
    )
    series["headline_core_gap"] = series["headline_mom"] - series["exclusion_core_mom"]
    return series


def top_contributors(
    contributions: pd.DataFrame,
    date: str,
    metric: str = "headline_contribution_pp",
    n: int = 10,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return top positive and negative component drivers for one date."""

    _require_columns(contributions, ["date", "component", metric])
    month = contributions.loc[contributions["date"].astype(str) == str(date)].copy()
    if month.empty:
        return month, month
    positive = month.sort_values(metric, ascending=False).head(n)
    negative = month.sort_values(metric, ascending=True).head(n)
    return positive, negative


def component_contributions(frame: pd.DataFrame, value_col: str = "mom_growth", weight_col: str = "weight") -> pd.DataFrame:
    """Build real-data contribution rows using item labels from the CLI frame."""
    data = frame.dropna(subset=[value_col, weight_col]).copy()
    totals = data.groupby("date")[weight_col].transform("sum")
    data["weight_norm"] = data[weight_col] / totals
    data["contribution_pp"] = data[value_col] * data["weight_norm"]
    cols = ["date", "item_code", "item_name", value_col, weight_col, "weight_norm", "contribution_pp"]
    return data[[c for c in cols if c in data.columns]].sort_values(["date", "contribution_pp"], ascending=[True, False])
