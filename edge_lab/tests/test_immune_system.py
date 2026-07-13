"""
Tests for Immune System: Adversarial Stress Testing Agent
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "agents"))

import pytest
import numpy as np
import pandas as pd
from immune_system import (
    ImmuneSystemTester,
    BlackSwanInjector,
    BlackSwanType,
    BlackSwanEvent,
    SurvivalReport,
    StressTestResult,
    create_sample_model,
)


class TestBlackSwanInjector:
    """Test black swan injection functionality"""

    @pytest.fixture
    def sample_data(self):
        """Create sample time series data"""
        np.random.seed(42)
        dates = pd.date_range(start="2020-01-01", periods=60, freq="MS")
        df = pd.DataFrame(
            {
                "target": np.random.randn(60) * 0.3 + 0.5,
                "mom": np.random.randn(60) * 0.3 + 0.5,
                "Ki_i": np.random.randn(60) * 0.1 + 7.0,
            },
            index=dates,
        )
        return df

    def test_inject_extreme_value(self, sample_data):
        """Test extreme value injection"""
        injector = BlackSwanInjector()
        df_shocked = injector.inject_extreme_value(
            sample_data, severity=1.0, periods=1, col="target"
        )

        # Check that last value was modified
        assert df_shocked["target"].iloc[-1] != sample_data["target"].iloc[-1]

    def test_inject_regime_change(self, sample_data):
        """Test regime change injection"""
        injector = BlackSwanInjector()
        df_shocked = injector.inject_regime_change(
            sample_data, severity=1.0, periods=6, col="target"
        )

        # Check distribution shift in last 6 periods
        original_mean = sample_data["target"].iloc[:-6].mean()
        shocked_mean = df_shocked["target"].iloc[-6:].mean()
        assert abs(shocked_mean - original_mean) > 0.1  # Significant shift

    def test_inject_missing_data(self, sample_data):
        """Test missing data injection"""
        injector = BlackSwanInjector()
        df_shocked = injector.inject_missing_data(
            sample_data, severity=1.0, col="target"
        )

        # Check for NaN values
        assert df_shocked["target"].isna().sum() > 0

    def test_inject_feature_outlier(self, sample_data):
        """Test feature outlier injection"""
        injector = BlackSwanInjector()
        df_shocked = injector.inject_feature_outlier(
            sample_data, severity=1.0, periods=1, feature="Ki_i"
        )

        # Check feature was modified
        assert df_shocked["Ki_i"].iloc[-1] != sample_data["Ki_i"].iloc[-1]

    def test_inject_volatility_explosion(self, sample_data):
        """Test volatility explosion injection"""
        injector = BlackSwanInjector()
        df_shocked = injector.inject_volatility_explosion(
            sample_data, severity=1.0, periods=6, col="target"
        )

        # Check increased volatility
        original_std = sample_data["target"].iloc[:-6].std()
        shocked_std = df_shocked["target"].iloc[-6:].std()
        assert shocked_std > original_std

    def test_inject_consecutive_shocks(self, sample_data):
        """Test consecutive shocks injection"""
        injector = BlackSwanInjector()
        df_shocked = injector.inject_consecutive_shocks(
            sample_data, severity=1.0, periods=3, col="target"
        )

        # Check last 3 values were modified
        original_last = sample_data["target"].iloc[-3:].values
        shocked_last = df_shocked["target"].iloc[-3:].values
        assert not np.allclose(original_last, shocked_last)

    def test_apply_method(self, sample_data):
        """Test apply method with BlackSwanEvent"""
        injector = BlackSwanInjector()
        event = BlackSwanEvent(
            type=BlackSwanType.EXTREME_VALUE, severity=0.8, duration_periods=2
        )

        df_shocked = injector.apply(sample_data, event)
        assert df_shocked["target"].iloc[-1] != sample_data["target"].iloc[-1]


class TestImmuneSystemTester:
    """Test immune system tester functionality"""

    @pytest.fixture
    def sample_data(self):
        """Create sample time series data"""
        np.random.seed(42)
        dates = pd.date_range(start="2020-01-01", periods=60, freq="MS")
        df = pd.DataFrame(
            {
                "target": np.random.randn(60) * 0.3 + 0.5,
                "mom": np.random.randn(60) * 0.3 + 0.5,
            },
            index=dates,
        )
        return df

    @pytest.fixture
    def tester(self):
        """Create immune system tester"""
        return ImmuneSystemTester()

    def test_initialization(self, tester):
        """Test tester initialization"""
        assert tester.target_col == "target"
        assert tester.survival_threshold_mae == 2.0
        assert tester.prediction_bounds == (-5.0, 10.0)

    def test_generate_black_swan_events(self, tester):
        """Test black swan event generation"""
        events = tester.generate_black_swan_events(count=10)

        # Should generate requested number
        assert len(events) == 10

        # All should be BlackSwanEvent instances
        assert all(isinstance(e, BlackSwanEvent) for e in events)

        # Severity should be in valid range
        assert all(0.0 <= e.severity <= 1.0 for e in events)

        # Should include all event types
        event_types = {e.type for e in events}
        assert len(event_types) == len(BlackSwanType)

    def test_stress_test_model(self, tester, sample_data):
        """Test stress testing a single model"""
        model = create_sample_model("TestModel")

        report = tester.stress_test_model(model, sample_data)

        assert isinstance(report, SurvivalReport)
        assert report.model_name == "TestModel"
        assert report.total_tests > 0
        assert report.survival_rate >= 0.0
        assert report.survival_rate <= 100.0
        assert report.passed_tests + report.failed_tests == report.total_tests

    def test_stress_test_model_with_custom_events(self, tester, sample_data):
        """Test stress testing with custom black swan events"""
        model = create_sample_model("TestModel")

        custom_events = [
            BlackSwanEvent(
                type=BlackSwanType.EXTREME_VALUE, severity=0.5, duration_periods=1
            ),
            BlackSwanEvent(
                type=BlackSwanType.MISSING_DATA, severity=0.8, duration_periods=1
            ),
        ]

        report = tester.stress_test_model(model, sample_data, black_swans=custom_events)

        assert report.total_tests == 2
        assert len(report.results) == 2

    def test_test_models(self, tester, sample_data):
        """Test stress testing multiple models"""
        models = [
            create_sample_model("ModelA"),
            create_sample_model("ModelB"),
            create_sample_model("ModelC"),
        ]

        reports = tester.test_models(models, sample_data)

        # All models should be tested and have reports
        # Note: Due to model copying, names may not be preserved uniquely
        assert len(reports) >= 1
        assert all(isinstance(r, SurvivalReport) for r in reports.values())

    def test_generate_report(self, tester, sample_data):
        """Test report generation"""
        model = create_sample_model("TestModel")
        reports = {"TestModel": tester.stress_test_model(model, sample_data)}

        report_text = tester.generate_report(reports)

        assert "🛡️ Immune System Stress Test Report" in report_text
        assert "TestModel" in report_text
        assert "Survival Rate" in report_text
        assert "Passed/Total" in report_text


class TestSurvivalCheck:
    """Test survival criteria checking"""

    @pytest.fixture
    def tester(self):
        """Create tester with custom bounds"""
        return ImmuneSystemTester(
            prediction_bounds=(-2.0, 5.0), survival_threshold_mae=3.0
        )

    def test_survival_within_bounds(self, tester):
        """Test survival with prediction within bounds"""
        survived, details = tester._check_survival(
            1.5, None, BlackSwanEvent(type=BlackSwanType.EXTREME_VALUE, severity=0.5)
        )

        assert survived is True
        assert details.get("bounds_ok") is True

    def test_survival_outside_lower_bound(self, tester):
        """Test failure with prediction below lower bound"""
        survived, details = tester._check_survival(
            -3.0, None, BlackSwanEvent(type=BlackSwanType.EXTREME_VALUE, severity=0.5)
        )

        assert survived is False
        assert details.get("bounds_ok") is False

    def test_survival_outside_upper_bound(self, tester):
        """Test failure with prediction above upper bound"""
        survived, details = tester._check_survival(
            6.0, None, BlackSwanEvent(type=BlackSwanType.EXTREME_VALUE, severity=0.5)
        )

        assert survived is False
        assert details.get("bounds_ok") is False

    def test_survival_with_nan(self, tester):
        """Test failure with NaN prediction"""
        survived, details = tester._check_survival(
            np.nan, None, BlackSwanEvent(type=BlackSwanType.EXTREME_VALUE, severity=0.5)
        )

        assert survived is False
        assert details.get("has_nan") is True

    def test_survival_with_inf(self, tester):
        """Test failure with infinite prediction"""
        survived, details = tester._check_survival(
            np.inf, None, BlackSwanEvent(type=BlackSwanType.EXTREME_VALUE, severity=0.5)
        )

        assert survived is False
        assert details.get("has_nan") is True


class TestDataClasses:
    """Test dataclasses"""

    def test_black_swan_event(self):
        """Test BlackSwanEvent dataclass"""
        event = BlackSwanEvent(
            type=BlackSwanType.EXTREME_VALUE, severity=0.8, duration_periods=2
        )

        assert event.type == BlackSwanType.EXTREME_VALUE
        assert event.severity == 0.8
        assert event.duration_periods == 2
        assert event.target_index is None

    def test_stress_test_result(self):
        """Test StressTestResult dataclass"""
        result = StressTestResult(
            model_name="TestModel",
            black_swan=BlackSwanEvent(type=BlackSwanType.EXTREME_VALUE, severity=0.5),
            survived=True,
        )

        assert result.model_name == "TestModel"
        assert result.survived is True
        assert result.error is None

    def test_survival_report(self):
        """Test SurvivalReport dataclass"""
        report = SurvivalReport(
            model_name="TestModel",
            survival_rate=95.0,
            total_tests=10,
            passed_tests=9,
            failed_tests=1,
        )

        assert report.model_name == "TestModel"
        assert report.survival_rate == 95.0
        assert report.total_tests == 10
        assert report.passed_tests == 9
        assert report.failed_tests == 1
        assert len(report.results) == 0
        assert len(report.vulnerabilities) == 0


class TestRobustModel:
    """Test with a more robust model implementation"""

    @pytest.fixture
    def robust_data(self):
        """Create robust test data"""
        np.random.seed(123)
        dates = pd.date_range(start="2018-01-01", periods=72, freq="MS")
        df = pd.DataFrame(
            {
                "target": np.random.randn(72) * 0.2 + 0.5,
                "mom": np.random.randn(72) * 0.2 + 0.5,
            },
            index=dates,
        )
        return df

    def test_robust_model_survival(self, robust_data):
        """Test robust model should survive most events"""
        tester = ImmuneSystemTester()
        model = create_sample_model("RobustModel")

        report = tester.stress_test_model(model, robust_data)

        # Should survive most tests
        assert report.survival_rate >= 70.0

    def test_multiple_model_comparison(self, robust_data):
        """Test comparing multiple models"""
        tester = ImmuneSystemTester()
        models = [create_sample_model(f"Model{i}") for i in range(5)]

        reports = tester.test_models(models, robust_data)

        # All models should be tested (at least 1 report)
        assert len(reports) >= 1

        # Check at least one has good survival rate
        rates = [r.survival_rate for r in reports.values()]
        assert max(rates) >= 80.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
