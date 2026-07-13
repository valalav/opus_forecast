from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_ROOT))

from core_inflation.weights import (  # noqa: E402
    WeightError,
    normalize_weight_column,
    normalize_weights,
    summarize_weights,
)


def test_normalize_weights_scales_to_one() -> None:
    normalized = normalize_weights(pd.Series([2.0, 3.0, 5.0]))

    assert normalized.tolist() == pytest.approx([0.2, 0.3, 0.5])
    assert normalized.sum() == pytest.approx(1.0)


def test_normalize_weight_column_preserves_input_columns() -> None:
    frame = pd.DataFrame({"item_code": [1, 2], "weight": [10.0, 30.0]})

    result = normalize_weight_column(frame)

    assert list(result.columns) == ["item_code", "weight", "normalized_weight"]
    assert result["normalized_weight"].tolist() == pytest.approx([0.25, 0.75])
    assert "normalized_weight" not in frame.columns


def test_normalize_weights_rejects_zero_sum() -> None:
    with pytest.raises(WeightError, match="sum must be positive"):
        normalize_weights(pd.Series([0.0, 0.0]))


def test_normalize_weights_rejects_nan() -> None:
    with pytest.raises(WeightError, match="contains NaN"):
        normalize_weights(pd.Series([1.0, float("nan")]))


def test_normalize_weights_rejects_negative_values() -> None:
    with pytest.raises(WeightError, match="negative"):
        normalize_weights(pd.Series([1.0, -0.1]))


def test_summarize_weights_reports_original_and_normalized_sums() -> None:
    summary = summarize_weights(pd.Series([4.0, 6.0]))

    assert summary.original_sum == pytest.approx(10.0)
    assert summary.normalized_sum == pytest.approx(1.0)
    assert summary.n_weights == 2
