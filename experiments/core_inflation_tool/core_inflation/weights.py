from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


class WeightError(ValueError):
    pass


@dataclass(frozen=True)
class WeightSummary:
    original_sum: float
    normalized_sum: float
    n_weights: int


def validate_weights(weights: pd.Series, context: str = "weights") -> pd.Series:
    numeric = pd.to_numeric(weights, errors="coerce")
    if numeric.isna().any():
        raise WeightError(f"{context} contains NaN or non-numeric values")
    if (numeric < 0).any():
        raise WeightError(f"{context} contains negative values")
    if float(numeric.sum()) <= 0.0:
        raise WeightError(f"{context} sum must be positive")
    return numeric.astype(float)


def normalize_weights(weights: pd.Series) -> pd.Series:
    numeric = validate_weights(weights, "weights")
    return numeric / numeric.sum()


def normalize_weight_column(frame: pd.DataFrame, weight_col: str = "weight") -> pd.DataFrame:
    out = frame.copy()
    out["normalized_weight"] = normalize_weights(out[weight_col])
    return out


def normalize_weights_by_group(frame: pd.DataFrame, weight_col: str = "weight", group_cols: list[str] | None = None) -> pd.DataFrame:
    out = frame.copy()
    if group_cols is None:
        out["weight_norm"] = normalize_weights(out[weight_col])
        return out
    out["weight_norm"] = 0.0
    for key, idx in out.groupby(group_cols, dropna=False).groups.items():
        out.loc[idx, "weight_norm"] = normalize_weights(out.loc[idx, weight_col])
    return out


def summarize_weights(weights: pd.Series) -> WeightSummary:
    normalized = normalize_weights(weights)
    return WeightSummary(
        original_sum=float(pd.to_numeric(weights, errors="coerce").sum()),
        normalized_sum=float(normalized.sum()),
        n_weights=int(len(normalized)),
    )


def weight_coverage(frame: pd.DataFrame, included_mask: pd.Series, weight_col: str = "weight") -> dict[str, float]:
    weights = validate_weights(frame[weight_col], weight_col)
    included = weights.loc[included_mask]
    excluded = weights.loc[~included_mask]
    return {
        "total_weight": float(weights.sum()),
        "included_weight": float(included.sum()),
        "excluded_weight": float(excluded.sum()),
        "excluded_share": float(excluded.sum() / weights.sum()),
    }
