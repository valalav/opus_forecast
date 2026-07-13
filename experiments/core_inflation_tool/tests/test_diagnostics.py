from pathlib import Path
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core_inflation.diagnostics import (  # noqa: E402
    basket_coverage_diagnostic,
    jump_diagnostic,
    mom_yoy_contamination_diagnostic,
    range_diagnostic,
)


def test_mom_yoy_contamination_diagnostic_fails_identical_matrices():
    mom = pd.DataFrame(
        [[0.1, 0.2], [0.3, 0.4]],
        index=["2026-01", "2026-02"],
        columns=["food", "services"],
    )
    yoy = mom.copy()

    result = mom_yoy_contamination_diagnostic(mom, yoy)

    assert result.passed is False
    assert result.details["identical"] is True
    assert result.details["comparable_cells"] == 4


def test_mom_yoy_contamination_diagnostic_passes_different_matrices():
    mom = pd.DataFrame([[0.1, 0.2], [0.3, 0.4]])
    yoy = pd.DataFrame([[5.1, 6.2], [7.3, 8.4]])

    result = mom_yoy_contamination_diagnostic(mom, yoy)

    assert result.passed is True
    assert result.details["identical"] is False


def test_range_diagnostic_flags_non_finite_and_out_of_range_values():
    values = pd.Series([0.2, 5.0, -100.0, float("inf")])

    result = range_diagnostic(values, min_value=-20.0, max_value=20.0)

    assert result.passed is False
    assert result.details["below_min_count"] == 1
    assert result.details["non_finite_count"] == 1



def test_basket_coverage_diagnostic_blocks_incomplete_month():
    result = basket_coverage_diagnostic(pd.Series([1.0, 0.97]), minimum=0.98)

    assert result["status"] == "fail"
    assert result["value"] == 0.97


def test_jump_diagnostic_flags_large_absolute_jump():
    series = pd.Series([0.1, 0.2, 0.3, 4.0], index=["m1", "m2", "m3", "m4"])

    result = jump_diagnostic(series, z_threshold=10.0, absolute_threshold=2.0)

    assert result.loc["m4", "change"] == 3.7
    assert result.loc["m4", "jump_flag"] == True
    assert result.loc["m1", "jump_flag"] == False
