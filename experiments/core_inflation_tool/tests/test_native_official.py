import math
from statistics import median

import pandas as pd
import pytest

from experiments.core_inflation_tool.core_inflation.native_official import (
    BASE_CPI_CODE,
    COMPONENT_CODES,
    FOOD_EX_FRUIT_REGSA_CODE,
    calculate_native_series,
    annualize_index,
    volatility_exclusion_index,
    weighted_trimmed_index,
)
from experiments.core_inflation_tool.core_inflation.official_workbook import OFFICIAL_COMPONENT_ROWS


def test_annualize_index_matches_monthly_compounding():
    assert annualize_index(100.0) == pytest.approx(0.0)
    assert annualize_index(101.0) == pytest.approx(((1.01**12) - 1.0) * 100.0)
    with pytest.raises(ValueError):
        annualize_index(0.0)


def test_weighted_trimmed_index_partially_trims_both_boundaries():
    values = pd.Series([0.0, 10.0, 20.0])
    weights = pd.Series([0.2, 0.6, 0.2])

    result = weighted_trimmed_index(values, weights, 0.2)

    assert result == pytest.approx(10.0)


def test_volatility_exclusion_supports_partial_and_forced_boundaries():
    current = pd.Series([90.0, 100.0, 110.0], index=[1, 2, 3])
    volatility = pd.Series([3.0, 2.0, 1.0], index=[1, 2, 3])
    weights = pd.Series([0.2, 0.5, 0.3], index=[1, 2, 3])

    partial = volatility_exclusion_index(current, volatility, weights, 0.3)
    forced = volatility_exclusion_index(
        current,
        volatility,
        weights,
        0.3,
        forced_codes=frozenset({3}),
    )

    assert partial == pytest.approx((100.0 * 0.4 + 110.0 * 0.3) / 0.7)
    assert forced == pytest.approx((90.0 * 0.2 + 100.0 * 0.5) / 0.7)


def _synthetic_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    dates = pd.date_range("2019-01-01", "2021-02-01", freq="MS")
    component_rows = []
    aggregate_rows = []
    for month_number, date in enumerate(dates):
        for code in sorted(COMPONENT_CODES):
            component_rows.append(
                {
                    "date": date,
                    "item_code": code,
                    "index_value": 100.0 + (code % 9 - 4) * 0.03 + month_number * (code % 5 - 2) * 0.001,
                }
            )
        aggregate_rows.extend(
            [
                {"date": date, "item_code": BASE_CPI_CODE, "index_value": 100.4},
                {"date": date, "item_code": FOOD_EX_FRUIT_REGSA_CODE, "index_value": 100.2},
            ]
        )
    weights = [
        {"year": 2021, "item_code": code, "weight": 1.0 / len(COMPONENT_CODES)}
        for code in sorted(COMPONENT_CODES)
    ]
    weights.extend(
        [
            {"year": 2021, "item_code": BASE_CPI_CODE, "weight": 0.75},
            {"year": 2021, "item_code": 8, "weight": 0.34},
        ]
    )
    component_panel = pd.DataFrame(component_rows)
    regsa = pd.concat([component_panel, pd.DataFrame(aggregate_rows)], ignore_index=True)
    return component_panel, pd.DataFrame(weights), regsa


def test_native_series_calculates_16_estimates_and_uses_no_future_values():
    component_panel, weights, regsa = _synthetic_inputs()

    full_series, full_components = calculate_native_series(component_panel, weights, regsa)
    cutoff = pd.Timestamp("2021-01-01")
    truncated_panel = component_panel.loc[component_panel["date"] <= cutoff]
    truncated_regsa = regsa.loc[regsa["date"] <= cutoff]
    truncated_series, _ = calculate_native_series(truncated_panel, weights, truncated_regsa)

    january_components = full_components.loc[full_components["date"].eq("2021-01-01")]
    assert len(january_components) == len(OFFICIAL_COMPONENT_ROWS) == 16
    assert set(january_components["component"]) == set(OFFICIAL_COMPONENT_ROWS.values())
    assert full_series.loc[0, "native_stable_inflation_saar"] == pytest.approx(
        median(january_components["value_saar"])
    )
    assert full_series.loc[0, "native_stable_inflation_saar"] == pytest.approx(
        truncated_series.loc[0, "native_stable_inflation_saar"]
    )
    assert math.isnan(full_series.loc[0, "native_stable_inflation_3mma_saar"])
