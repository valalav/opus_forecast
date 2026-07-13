from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .weights import validate_weights, weight_coverage


@dataclass(frozen=True)
class ExclusionCoreResult:
    value: float
    original_weight_sum: float
    excluded_weight_sum: float
    included_weight_sum: float
    excluded_components: tuple[object, ...]


@dataclass(frozen=True)
class WeightedMedianResult:
    value: float
    component: object
    component_weight: float


@dataclass(frozen=True)
class WeightedTrimmedMeanResult:
    value: float
    included_weight_sum: float
    lower_trimmed: tuple[object, ...]
    upper_trimmed: tuple[object, ...]


def index_to_growth(index_values: pd.Series | float) -> pd.Series | float:
    return index_values - 100.0


def weighted_aggregate(values: pd.Series, weights: pd.Series) -> float:
    vals = pd.to_numeric(values, errors="coerce")
    w = validate_weights(weights, "aggregation weights")
    mask = vals.notna() & w.notna()
    if not mask.any():
        raise ValueError("no valid values for weighted aggregation")
    return float((vals.loc[mask] * w.loc[mask]).sum() / w.loc[mask].sum())



def weighted_winsorized_mean(
    values: pd.Series,
    weights: pd.Series,
    *,
    lower: float = 0.05,
    upper: float = 0.05,
) -> float:
    """Return a weighted mean after clipping both distribution tails."""

    if lower < 0 or upper < 0 or lower + upper >= 1:
        raise ValueError("winsor shares must be non-negative and sum to less than 1")
    data = pd.DataFrame({"value": values, "weight": weights}).dropna().sort_values("value")
    validated = validate_weights(data["weight"], "winsor weights")
    cumulative = validated.cumsum()
    total = float(validated.sum())
    lower_value = float(data.loc[cumulative > lower * total, "value"].iloc[0])
    upper_value = float(data.loc[cumulative >= (1.0 - upper) * total, "value"].iloc[0])
    return weighted_aggregate(data["value"].clip(lower_value, upper_value), validated)


def weighted_aggregation(values: pd.Series, weights: pd.Series) -> float:
    return weighted_aggregate(values, weights)


def exclusion_core(
    values: pd.DataFrame | pd.Series,
    weights: pd.Series | set[int],
    exclusions: list[object] | set[object] | None = None,
    value_col: str = "mom_growth",
    weight_col: str = "weight",
) -> ExclusionCoreResult | tuple[float, dict[str, float]]:
    if isinstance(values, pd.DataFrame):
        frame = values
        exclude_codes = {int(code) for code in weights}  # type: ignore[arg-type]
        included_mask = ~frame["item_code"].astype(int).isin(exclude_codes)
        if not included_mask.any():
            raise ValueError("exclusion removed all components")
        stats = weight_coverage(frame, included_mask, weight_col)
        return weighted_aggregate(frame.loc[included_mask, value_col], frame.loc[included_mask, weight_col]), stats

    if exclusions is None:
        exclusions = []
    w = validate_weights(weights, "exclusion weights")  # type: ignore[arg-type]
    excluded = tuple(exclusions)
    included_mask = ~values.index.isin(excluded)
    if not included_mask.any():
        raise ValueError("exclusion removed all components")
    return ExclusionCoreResult(
        value=weighted_aggregate(values.loc[included_mask], w.loc[included_mask]),
        original_weight_sum=float(w.sum()),
        excluded_weight_sum=float(w.loc[~included_mask].sum()),
        included_weight_sum=float(w.loc[included_mask].sum()),
        excluded_components=excluded,
    )


def weighted_median(
    values: pd.DataFrame | pd.Series,
    weights: pd.Series | None = None,
    value_col: str = "mom_growth",
    weight_col: str = "weight",
) -> WeightedMedianResult | tuple[float, int | None]:
    if isinstance(values, pd.DataFrame):
        data = values[[value_col, weight_col, "item_code"]].dropna().sort_values(value_col)
        w = validate_weights(data[weight_col], "median weights")
        cutoff = w.sum() / 2.0
        cumulative = w.cumsum()
        pos = cumulative.ge(cutoff).idxmax()
        row = data.loc[pos]
        return float(row[value_col]), int(row["item_code"]) if pd.notna(row["item_code"]) else None

    if weights is None:
        raise ValueError("weights are required")
    data = pd.DataFrame({"value": values, "weight": weights}).dropna().sort_values("value")
    weights = validate_weights(data[weight_col], "median weights")
    cutoff = weights.sum() / 2.0
    cumulative = weights.cumsum()
    pos = cumulative.ge(cutoff).idxmax()
    row = data.loc[pos]
    return WeightedMedianResult(value=float(row["value"]), component=pos, component_weight=float(row["weight"]))


def _weighted_trimmed_mean_frame(
    frame: pd.DataFrame,
    lower: float = 0.10,
    upper: float = 0.10,
    value_col: str = "mom_growth",
    weight_col: str = "weight",
) -> tuple[float, pd.DataFrame]:
    if lower < 0 or upper < 0 or lower + upper >= 1:
        raise ValueError("trim shares must be non-negative and sum to less than 1")
    data = frame[[value_col, weight_col, "item_code"]].dropna().sort_values(value_col).copy()
    weights = validate_weights(data[weight_col], "trim weights")
    total = float(weights.sum())
    data["_remaining_weight"] = weights.astype(float)
    lower_left = lower * total
    upper_left = upper * total
    for idx in data.index:
        if lower_left <= 0:
            break
        cut = min(float(data.at[idx, "_remaining_weight"]), lower_left)
        data.at[idx, "_remaining_weight"] -= cut
        lower_left -= cut
    for idx in reversed(data.index):
        if upper_left <= 0:
            break
        cut = min(float(data.at[idx, "_remaining_weight"]), upper_left)
        data.at[idx, "_remaining_weight"] -= cut
        upper_left -= cut
    kept = data.loc[data["_remaining_weight"] > 0].copy()
    if kept.empty:
        raise ValueError("trimming removed all positive weight")
    mean = weighted_aggregate(kept[value_col], kept["_remaining_weight"])
    return mean, data[["item_code", value_col, weight_col, "_remaining_weight"]]


def weighted_trimmed_mean(
    values: pd.DataFrame | pd.Series,
    weights: pd.Series | None = None,
    trim_lower: float = 0.10,
    trim_upper: float = 0.10,
    value_col: str = "mom_growth",
    weight_col: str = "weight",
) -> WeightedTrimmedMeanResult | tuple[float, pd.DataFrame]:
    if isinstance(values, pd.DataFrame):
        return _weighted_trimmed_mean_frame(values, trim_lower, trim_upper, value_col, weight_col)
    if weights is None:
        raise ValueError("weights are required")
    data = pd.DataFrame({"value": values, "weight": weights}).dropna().sort_values("value")
    w = validate_weights(data["weight"], "trim weights")
    total = float(w.sum())
    data["remaining_weight"] = w.astype(float)
    lower_left = trim_lower * total
    upper_left = trim_upper * total
    lower_trimmed: list[object] = []
    upper_trimmed: list[object] = []
    for idx in data.index:
        if lower_left <= 0:
            break
        cut = min(float(data.at[idx, "remaining_weight"]), lower_left)
        data.at[idx, "remaining_weight"] -= cut
        lower_left -= cut
        if data.at[idx, "remaining_weight"] == 0:
            lower_trimmed.append(idx)
    for idx in reversed(data.index):
        if upper_left <= 0:
            break
        cut = min(float(data.at[idx, "remaining_weight"]), upper_left)
        data.at[idx, "remaining_weight"] -= cut
        upper_left -= cut
        if data.at[idx, "remaining_weight"] == 0:
            upper_trimmed.append(idx)
    kept = data.loc[data["remaining_weight"] > 0]
    value = weighted_aggregate(kept["value"], kept["remaining_weight"])
    return WeightedTrimmedMeanResult(
        value=value,
        included_weight_sum=float(kept["remaining_weight"].sum()),
        lower_trimmed=tuple(lower_trimmed),
        upper_trimmed=tuple(upper_trimmed),
    )


def calculate_monthly_indicators(month: pd.DataFrame, exclude_codes: set[int], trim_lower: float, trim_upper: float) -> dict[str, float | int | None]:
    core, coverage = exclusion_core(month, exclude_codes)  # type: ignore[misc]
    trim, trim_detail = weighted_trimmed_mean(month, trim_lower=trim_lower, trim_upper=trim_upper)  # type: ignore[misc]
    median, median_item = weighted_median(month)  # type: ignore[misc]
    headline_rows = month.loc[month["item_code"].eq(1)]
    headline = float(headline_rows["mom_growth"].iloc[0]) if not headline_rows.empty else weighted_aggregate(month["mom_growth"], month["weight"])
    return {
        "headline_mom": headline,
        "exclusion_core_mom": core,
        "trimmed_mean_mom": trim,
        "weighted_median_mom": median,
        "headline_core_gap": headline - core,
        "median_item_code": median_item,
        "total_weight": coverage["total_weight"],
        "included_weight": coverage["included_weight"],
        "excluded_weight": coverage["excluded_weight"],
        "excluded_share": coverage["excluded_share"],
        "trimmed_component_count": int((trim_detail["_remaining_weight"] <= 0).sum()),
    }
