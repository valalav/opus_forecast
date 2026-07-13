import pandas as pd

from experiments.core_inflation_tool.core_inflation.contributions import (
    build_contribution_table,
    build_series_from_contributions,
    top_contributors,
)


def test_build_contribution_table_normalizes_headline_and_core_weights():
    components = pd.DataFrame(
        [
            {"date": "2026-01", "component": "A", "mom": 1.0, "weight": 50, "excluded": False},
            {"date": "2026-01", "component": "B", "mom": 3.0, "weight": 30, "excluded": False},
            {"date": "2026-01", "component": "C", "mom": 10.0, "weight": 20, "excluded": True},
        ]
    )

    contributions = build_contribution_table(components)
    series = build_series_from_contributions(contributions)

    assert round(series.loc[0, "headline_mom"], 6) == 3.4
    assert round(series.loc[0, "exclusion_core_mom"], 6) == 1.75
    assert round(series.loc[0, "headline_core_gap"], 6) == 1.65
    assert round(contributions["headline_weight_share"].sum(), 6) == 1.0
    assert round(contributions["core_weight_share"].sum(), 6) == 1.0


def test_top_contributors_returns_positive_and_negative_drivers():
    contributions = build_contribution_table(
        pd.DataFrame(
            [
                {"date": "2026-01", "component": "A", "mom": -2.0, "weight": 1, "excluded": False},
                {"date": "2026-01", "component": "B", "mom": 5.0, "weight": 1, "excluded": False},
                {"date": "2026-01", "component": "C", "mom": 1.0, "weight": 1, "excluded": False},
            ]
        )
    )

    positive, negative = top_contributors(contributions, "2026-01", n=1)

    assert positive.iloc[0]["component"] == "B"
    assert negative.iloc[0]["component"] == "A"
