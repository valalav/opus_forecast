import math
from pathlib import Path
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core_inflation.indicators import (  # noqa: E402
    exclusion_core,
    index_to_growth,
    weighted_aggregate,
    weighted_median,
    weighted_trimmed_mean,
    weighted_winsorized_mean,
)


def test_index_to_growth_converts_index_levels_to_growth():
    index = pd.Series([100.0, 101.25, 99.5], index=["a", "b", "c"])

    growth = index_to_growth(index)

    assert growth.to_dict() == {"a": 0.0, "b": 1.25, "c": -0.5}


def test_weighted_aggregation_returns_exact_expected_value():
    values = pd.Series([1.0, 3.0, 10.0], index=["bread", "fuel", "rent"])
    weights = pd.Series([2.0, 3.0, 5.0], index=values.index)

    result = weighted_aggregate(values, weights)

    assert result == 6.1


def test_exclusion_core_renormalizes_weights_and_reports_weight_sums():
    values = pd.Series([1.0, 3.0, 10.0], index=["bread", "fuel", "rent"])
    weights = pd.Series([2.0, 3.0, 5.0], index=values.index)

    result = exclusion_core(values, weights, exclusions=["fuel"])

    assert result.value == (1.0 * 2.0 + 10.0 * 5.0) / 7.0
    assert result.original_weight_sum == 10.0
    assert result.excluded_weight_sum == 3.0
    assert result.included_weight_sum == 7.0
    assert result.excluded_components == ("fuel",)


def test_weighted_median_handles_uneven_weights():
    values = pd.Series([-3.0, 0.5, 8.0], index=["small_tail", "large_center", "upper_tail"])
    weights = pd.Series([0.10, 0.70, 0.20], index=values.index)

    result = weighted_median(values, weights)

    assert result.value == 0.5
    assert result.component == "large_center"
    assert result.component_weight == 0.70


def test_weighted_trimmed_mean_handles_tail_removal_by_weight():
    values = pd.Series([-100.0, 0.0, 10.0, 100.0], index=["low", "center_a", "center_b", "high"])
    weights = pd.Series([0.10, 0.40, 0.40, 0.10], index=values.index)

    result = weighted_trimmed_mean(values, weights, trim_lower=0.10, trim_upper=0.10)

    assert math.isclose(result.value, 5.0)
    assert math.isclose(result.included_weight_sum, 0.80)
    assert result.lower_trimmed == ("low",)
    assert result.upper_trimmed == ("high",)


def test_weighted_trimmed_mean_partially_trims_boundary_weights():
    values = pd.Series([-10.0, 0.0, 10.0], index=["low", "middle", "high"])
    weights = pd.Series([0.20, 0.60, 0.20], index=values.index)

    result = weighted_trimmed_mean(values, weights, trim_lower=0.10, trim_upper=0.10)

    assert math.isclose(result.value, 0.0, abs_tol=1e-12)
    assert math.isclose(result.included_weight_sum, 0.80)


def test_weighted_winsorized_mean_caps_tail_values_without_dropping_weight():
    values = pd.Series([-100.0, 0.0, 1.0, 100.0])
    weights = pd.Series([0.05, 0.45, 0.45, 0.05])

    result = weighted_winsorized_mean(values, weights, lower=0.05, upper=0.05)

    assert math.isclose(result, 0.5)
