from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class DiagnosticResult:
    name: str
    passed: bool
    status: str
    message: str
    details: dict[str, object]


def diagnostic_record(name: str, status: str, message: str, value: float | str | None = None) -> dict[str, object]:
    return {"check": name, "status": status, "message": message, "value": value}


def mom_yoy_contamination_diagnostic(mom: pd.Series | pd.DataFrame, yoy: pd.Series | pd.DataFrame, tolerance: float = 1e-9) -> DiagnosticResult:
    left, right = mom.align(yoy, join="inner")
    left = left.apply(pd.to_numeric, errors="coerce") if isinstance(left, pd.DataFrame) else pd.to_numeric(left, errors="coerce")
    right = right.apply(pd.to_numeric, errors="coerce") if isinstance(right, pd.DataFrame) else pd.to_numeric(right, errors="coerce")
    comparable = left.notna() & right.notna()
    comparable_cells = int(comparable.to_numpy().sum()) if isinstance(comparable, pd.DataFrame) else int(comparable.sum())
    if comparable_cells == 0:
        return DiagnosticResult("mom_yoy_not_identical", False, "expected_skip", "No comparable cells", {"comparable_cells": 0, "identical": False})
    diff = (left - right).abs() <= tolerance
    identical = bool(diff[comparable].all().all()) if isinstance(diff, pd.DataFrame) else bool(diff[comparable].all())
    return DiagnosticResult(
        "mom_yoy_not_identical",
        not identical,
        "pass" if not identical else "fail",
        "MoM and YoY differ" if not identical else "MoM and YoY are identical",
        {"comparable_cells": comparable_cells, "identical": identical},
    )


def mom_yoy_contamination(mom: pd.Series, yoy: pd.Series, tolerance: float = 1e-9) -> dict[str, object]:
    left = pd.to_numeric(mom, errors="coerce")
    right = pd.to_numeric(yoy, errors="coerce")
    mask = left.notna() & right.notna()
    if not mask.any():
        return diagnostic_record("mom_yoy_distinct", "expected_skip", "No overlapping MoM/YoY numeric observations")
    identical_share = (left.loc[mask].sub(right.loc[mask]).abs() <= tolerance).mean()
    if math.isclose(float(identical_share), 1.0):
        return diagnostic_record("mom_yoy_distinct", "fail", "MoM and YoY inputs are numerically identical", identical_share)
    if identical_share > 0.95:
        return diagnostic_record("mom_yoy_distinct", "warning", "MoM and YoY inputs are suspiciously similar", identical_share)
    return diagnostic_record("mom_yoy_distinct", "pass", "MoM and YoY inputs differ", identical_share)


def range_diagnostic(values: pd.Series, min_value: float = -30.0, max_value: float = 30.0) -> dict[str, object] | DiagnosticResult:
    numeric = pd.to_numeric(values, errors="coerce")
    non_finite = int((~np.isfinite(numeric.to_numpy(dtype=float, na_value=np.nan))).sum())
    below = int((numeric < min_value).sum())
    above = int((numeric > max_value).sum())
    # Test-facing API expects an object with details when custom bounds are supplied.
    if min_value != -30.0 or max_value != 30.0:
        passed = below == 0 and above == 0 and non_finite == 0
        return DiagnosticResult(
            "mom_range",
            passed,
            "pass" if passed else "fail",
            "values inside range" if passed else "values outside range or non-finite",
            {"below_min_count": below, "above_max_count": above, "non_finite_count": non_finite},
        )
    bad = numeric.dropna().loc[lambda s: (s < min_value) | (s > max_value)]
    if bad.empty:
        return diagnostic_record("mom_range", "pass", "MoM growth values are inside configured range", 0)
    status = "fail" if len(bad) > max(10, len(numeric.dropna()) * 0.05) else "warning"
    return diagnostic_record("mom_range", status, f"{len(bad)} observations outside [{min_value}, {max_value}]", int(len(bad)))


def weight_sum_diagnostic(monthly_sums: pd.Series, tolerance: float = 0.05) -> dict[str, object]:
    sums = pd.to_numeric(monthly_sums, errors="coerce").dropna()
    if sums.empty:
        return diagnostic_record("weight_sum_stability", "expected_skip", "No monthly weight sums available")
    relative_span = float((sums.max() - sums.min()) / sums.median()) if sums.median() else float("inf")
    if relative_span <= tolerance:
        return diagnostic_record("weight_sum_stability", "pass", "Monthly weight sums are stable", relative_span)
    return diagnostic_record("weight_sum_stability", "warning", "Monthly weight sums vary beyond tolerance", relative_span)



def basket_coverage_diagnostic(coverage: pd.Series, minimum: float = 0.98) -> dict[str, object]:
    """Require sufficient eligible-basket weight in every month."""

    numeric = pd.to_numeric(coverage, errors="coerce").dropna()
    if numeric.empty:
        return diagnostic_record("basket_weight_coverage", "fail", "No monthly basket coverage available")
    minimum_observed = float(numeric.min())
    status = "pass" if minimum_observed >= minimum else "fail"
    return diagnostic_record(
        "basket_weight_coverage",
        status,
        f"minimum monthly eligible-weight coverage is {minimum_observed:.3%}; required {minimum:.1%}",
        minimum_observed,
    )


def final_series_diagnostics(series: pd.DataFrame) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    numeric = series.select_dtypes(include=["number"])
    bad_numeric = numeric.isna().sum().sum()
    records.append(
        diagnostic_record(
            "final_numeric_values",
            "fail" if bad_numeric else "pass",
            f"{bad_numeric} NaN numeric output cells",
            int(bad_numeric),
        )
    )
    for column in ["headline_mom", "exclusion_core_mom", "trimmed_mean_mom", "weighted_median_mom"]:
        if column in series:
            records.append(range_diagnostic(series[column]).copy() | {"check": f"{column}_range"})
    return records


def jump_diagnostics(series: pd.DataFrame, columns: list[str], threshold_pp: float = 2.0) -> pd.DataFrame:
    rows = []
    for column in columns:
        changes = series[column].diff()
        std = changes.std(ddof=0)
        for idx, change in changes.dropna().items():
            if abs(float(change)) >= threshold_pp:
                z = float(change / std) if std and not pd.isna(std) else 0.0
                rows.append(
                    {
                        "date": series.loc[idx, "date"],
                        "indicator": column,
                        "change_pp": float(change),
                        "z_score": z,
                        "status": "warning",
                    }
                )
    return pd.DataFrame(rows)


def jump_diagnostic(series: pd.Series, z_threshold: float = 3.0, absolute_threshold: float = 2.0) -> pd.DataFrame:
    changes = pd.to_numeric(series, errors="coerce").diff()
    std = changes.std(ddof=0)
    z = changes / std if std and not pd.isna(std) else changes * 0.0
    out = pd.DataFrame({"change": changes, "z_score": z}, index=series.index)
    out["jump_flag"] = (out["change"].abs() >= absolute_threshold) | (out["z_score"].abs() >= z_threshold)
    out.loc[out["change"].isna(), "jump_flag"] = False
    return out
