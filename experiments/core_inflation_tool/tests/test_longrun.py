from pathlib import Path
import sys

import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core_inflation.longrun import (  # noqa: E402
    add_stable_rate_metrics,
    build_longrun_metrics,
    causal_robust_filter,
    future_average,
    render_dynamics_report,
)


def test_future_average_uses_next_observations_only():
    series = pd.Series([1.0, 2.0, 3.0, 4.0])

    result = future_average(series, horizon=2)

    assert result.iloc[0] == 2.5
    assert result.iloc[1] == 3.5
    assert pd.isna(result.iloc[2])
    assert pd.isna(result.iloc[3])


def test_causal_robust_filter_limits_one_month_adjustment_without_lookahead():
    signal = pd.Series([0.2, 5.0, 0.4, -4.0])

    result = causal_robust_filter(signal, alpha=0.35, max_innovation_pp=1.0)

    assert result.tolist() == pytest.approx([0.2, 0.55, 0.4975, 0.1475])
    assert result.diff().abs().max() <= 0.35 + 1e-12
    changed_future = causal_robust_filter(pd.Series([0.2, 5.0, 0.4, 9.0]), alpha=0.35, max_innovation_pp=1.0)
    assert changed_future.iloc[:3].tolist() == pytest.approx(result.iloc[:3].tolist())


def test_build_longrun_metrics_includes_stable_core_usefulness():
    dates = pd.date_range("2020-01-01", periods=15, freq="MS")
    frame = pd.DataFrame(
        {
            "date": dates,
            "headline_mom": [0.2, 0.3, 1.5, 0.2, 0.3, 0.4, 0.2, 0.3, 0.2, 0.3, 0.4, 0.2, 0.3, 0.2, 0.3],
            "stable_core_mom": [0.25] * 15,
        }
    )

    metrics = build_longrun_metrics(frame, horizon=3)

    stable = metrics.loc[metrics["indicator"].eq("stable_core_mom")].iloc[0]
    assert stable["n_months"] == 15
    assert stable["std_mom"] == 0.0
    assert "mae_vs_next_3m_avg_headline" in metrics.columns


def test_add_stable_rate_metrics_adds_3mma_and_12m_rate():
    frame = pd.DataFrame(
        {
            "date": pd.date_range("2025-01-01", periods=12, freq="MS"),
            "stable_core_mom": [1.0] * 12,
            "stable_core_sa_mom": [0.5] * 12,
        }
    )

    enriched = add_stable_rate_metrics(frame)

    assert enriched.loc[1, "stable_core_3mma"] != enriched.loc[1, "stable_core_3mma"]
    assert enriched.loc[2, "stable_core_3mma"] == 1.0
    assert enriched.loc[2, "stable_core_3mma_annualized"] > 12.0
    assert abs(enriched.loc[11, "stable_core_12m"] - ((1.01**12) - 1.0) * 100.0) < 1e-12
    assert enriched.loc[2, "stable_core_sa_3mma"] == 0.5
    assert abs(enriched.loc[11, "stable_core_sa_12m"] - ((1.005**12) - 1.0) * 100.0) < 1e-12


def test_render_dynamics_report_is_russian_markdown():
    frame = pd.DataFrame(
        {
            "date": pd.date_range("2025-01-01", periods=3, freq="MS"),
            "headline_mom": [0.5, 1.0, 0.2],
            "stable_core_mom": [0.4, 0.4, 0.4],
            "stable_core_sa_mom": [0.3, 0.3, 0.3],
        }
    )
    metrics = build_longrun_metrics(frame, horizon=1)
    diagnostics = pd.DataFrame([{"check": "x", "status": "pass", "message": "ok"}])

    report = render_dynamics_report(frame, metrics, diagnostics)

    assert "# Динамика устойчивой инфляции КБР" in report
    assert "Последние 12 месяцев" in report
    assert "3MMA" in report
