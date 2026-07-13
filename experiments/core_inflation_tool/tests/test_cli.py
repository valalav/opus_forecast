from pathlib import Path

import pandas as pd
import pytest

from experiments.core_inflation_tool.core_inflation import cli


def _write_fixture(tmp_path: Path) -> Path:
    pd.DataFrame(
        [
            {"date": "2026-01", "component": "A", "mom": 0.2, "weight": 50, "excluded": False},
            {"date": "2026-01", "component": "B", "mom": 1.0, "weight": 30, "excluded": False},
            {"date": "2026-01", "component": "Fuel", "mom": 5.0, "weight": 20, "excluded": True},
            {"date": "2026-02", "component": "A", "mom": 0.3, "weight": 50, "excluded": False},
            {"date": "2026-02", "component": "B", "mom": 1.1, "weight": 30, "excluded": False},
            {"date": "2026-02", "component": "Fuel", "mom": 6.0, "weight": 20, "excluded": True},
        ]
    ).to_csv(tmp_path / "components.csv", index=False)
    pd.DataFrame(
        [
            {"check": "mom_yoy_not_identical", "status": "fail", "message": "fixture forces failed diagnostic"},
            {"check": "weights_present", "status": "pass", "message": "weights are present"},
        ]
    ).to_csv(tmp_path / "diagnostics.csv", index=False)
    config = tmp_path / "config.yaml"
    config.write_text(
        """
input:
  components_csv: components.csv
  diagnostics_csv: diagnostics.csv
report:
  jump_threshold: 0.1
""".strip()
        + "\n",
        encoding="utf-8",
    )
    return config


def test_run_writes_required_outputs_and_marks_failed_diagnostics(tmp_path):
    config = _write_fixture(tmp_path)
    output = tmp_path / "out"

    paths = cli.run(config, output, allow_output_outside_experiment=True)

    expected = {
        "series",
        "diagnostics",
        "contributions",
        "sa_contributions",
        "longrun_metrics",
        "jump_report",
        "dynamics_report",
        "config",
    }
    assert set(paths) == expected
    for path in paths.values():
        assert path.exists()

    series = pd.read_csv(paths["series"])
    contributions = pd.read_csv(paths["contributions"])
    report = paths["jump_report"].read_text(encoding="utf-8")
    dynamics = paths["dynamics_report"].read_text(encoding="utf-8")
    snapshot = paths["config"].read_text(encoding="utf-8")

    assert {
        "headline_mom",
        "exclusion_core_mom",
        "trimmed_mean_mom",
        "weighted_median_mom",
        "stable_core_signal_mom",
        "stable_core_mom",
        "stable_core_3mma",
        "stable_core_3mma_annualized",
        "stable_core_12m",
    }.issubset(series.columns)
    assert "diagnostic_status" in series.columns
    assert set(series["diagnostic_status"]) == {"fail"}
    assert series["stable_core_mom"].diff().abs().max() <= 0.35
    assert {"headline_contribution_pp", "core_contribution_pp"}.issubset(contributions.columns)
    assert "FAILED DIAGNOSTICS" in report
    assert "FAIL `mom_yoy_not_identical`" in report
    assert "Динамика устойчивой инфляции" in dynamics
    assert "components.csv" in snapshot


def test_run_refuses_output_outside_experiment_without_fixture_override(tmp_path):
    config = _write_fixture(tmp_path)

    with pytest.raises(ValueError, match="output path must be under"):
        cli.run(config, tmp_path / "outside")
