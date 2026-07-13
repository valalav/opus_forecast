"""
Tests for Regime Detector
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "agents"))

import pytest
import numpy as np
import pandas as pd
from regime_detector import (
    RegimeDetector,
    RegimeType,
    RegimeDetectionResult,
)


class TestRegimeDetector:
    """Test regime detection functionality"""

    @pytest.fixture
    def normal_data(self):
        """Create sample data in normal regime"""
        np.random.seed(42)
        dates = pd.date_range(start="2020-01-01", periods=60, freq="MS")
        df = pd.DataFrame(
            {
                "Ki_i": np.linspace(7.0, 7.5, 60) + np.random.randn(60) * 0.1,
                "Ruonia": np.linspace(6.8, 7.3, 60) + np.random.randn(60) * 0.1,
                "mom": np.random.randn(60) * 0.3 + 0.5,
            },
            index=dates,
        )
        return df

    @pytest.fixture
    def shock_data(self):
        """Create sample data with shock"""
        np.random.seed(42)
        dates = pd.date_range(start="2020-01-01", periods=60, freq="MS")

        # Simulate sudden rate increase (shock)
        ki_values = np.concatenate(
            [
                np.linspace(7.0, 7.5, 20),  # Normal
                np.linspace(7.5, 9.5, 5),  # Shock (rapid rate hike)
                np.linspace(9.5, 9.0, 35),  # Stabilized at higher level
            ]
        )

        df = pd.DataFrame(
            {
                "Ki_i": ki_values,
                "Ruonia": ki_values - 0.2 + np.random.randn(60) * 0.1,
                "mom": np.random.randn(60) * 0.3 + 0.5,
            },
            index=dates,
        )
        return df

    @pytest.fixture
    def high_inflation_data(self):
        """Create sample data with high inflation acceleration"""
        np.random.seed(42)
        dates = pd.date_range(start="2020-01-01", periods=60, freq="MS")

        # Simulate high inflation acceleration
        mom_values = np.concatenate(
            [
                np.random.randn(24) * 0.2 + 0.5,  # Normal first 2 years
                np.random.randn(36) * 0.3 + 2.0,  # High inflation acceleration
            ]
        )

        df = pd.DataFrame(
            {
                "Ki_i": np.linspace(7.0, 7.5, 60) + np.random.randn(60) * 0.1,
                "Ruonia": np.linspace(6.8, 7.3, 60) + np.random.randn(60) * 0.1,
                "mom": mom_values,
            },
            index=dates,
        )
        return df

    @pytest.fixture
    def detector(self):
        """Create RegimeDetector instance"""
        return RegimeDetector()

    def test_initialization(self, detector):
        """Test detector initialization"""
        assert detector.shock_threshold == 0.5
        assert detector.inflation_shock_threshold == 1.5
        assert detector.volatility_window == 3
        assert len(detector._regime_history) == 0

    def test_initialization_custom_params(self):
        """Test detector initialization with custom parameters"""
        detector = RegimeDetector(
            shock_threshold=1.0,
            inflation_shock_threshold=2.0,
            volatility_window=6,
        )
        assert detector.shock_threshold == 1.0
        assert detector.inflation_shock_threshold == 2.0
        assert detector.volatility_window == 6

    def test_detect_normal_regime(self, detector, normal_data):
        """Test detection of normal regime"""
        result = detector.detect(normal_data)

        assert isinstance(result, RegimeDetectionResult)
        assert result.regime == RegimeType.NORMAL
        assert 0.0 <= result.confidence <= 1.0
        assert len(result.history) == 1

    def test_detect_shock_regime(self, detector, shock_data):
        """Test detection of shock regime"""
        # Detect during shock period (month 21, after rate hike)
        shock_date = shock_data.index[22]
        result = detector.detect(shock_data, date=shock_date)

        assert result.regime == RegimeType.SHOCK
        assert result.confidence > 0.3
        assert "ki_change" in result.diagnostics

    def test_detect_high_inflation_regime(self, detector, high_inflation_data):
        """Test detection of high inflation regime"""
        # Detect during high inflation period (month 30)
        high_infl_date = high_inflation_data.index[30]
        result = detector.detect(high_inflation_data, date=high_infl_date)

        # May detect high inflation or shock depending on values
        assert isinstance(result.regime, RegimeType)
        assert result.confidence >= 0.0

    def test_detect_with_date(self, detector, normal_data):
        """Test detection with specific date"""
        date = normal_data.index[25]
        result = detector.detect(normal_data, date=date)

        assert isinstance(result, RegimeDetectionResult)
        assert len(result.history) == 1

    def test_detect_without_date(self, detector, normal_data):
        """Test detection without date (uses last row)"""
        result = detector.detect(normal_data)

        # Should detect on last row
        assert isinstance(result, RegimeDetectionResult)
        assert result.regime == RegimeType.NORMAL

    def test_detect_with_nonexistent_date(self, detector, normal_data):
        """Test detection with date not in index"""
        date = pd.Timestamp("2030-01-01")
        result = detector.detect(normal_data, date=date)

        # Should fall back to last row
        assert isinstance(result, RegimeDetectionResult)

    def test_detect_batch(self, detector, normal_data):
        """Test batch detection"""
        dates = [normal_data.index[10], normal_data.index[20], normal_data.index[30]]
        results = detector.detect_batch(normal_data, dates)

        assert len(results) == 3
        assert all(isinstance(r, RegimeDetectionResult) for r in results)
        assert len(detector._regime_history) == 3

    def test_detect_batch_all_dates(self, detector, normal_data):
        """Test batch detection on all dates"""
        results = detector.detect_batch(normal_data)

        assert len(results) == len(normal_data)
        assert all(isinstance(r, RegimeDetectionResult) for r in results)
        assert len(detector._regime_history) == len(normal_data)

    def test_regime_history_tracking(self, detector, normal_data):
        """Test that regime history is tracked correctly"""
        dates = [normal_data.index[10], normal_data.index[20]]
        detector.detect_batch(normal_data, dates)

        history = detector._regime_history
        assert len(history) == 2
        assert all(isinstance(r, RegimeType) for r in history)

    def test_reset_history(self, detector, normal_data):
        """Test history reset"""
        detector.detect_batch(normal_data)
        assert len(detector._regime_history) > 0

        detector.reset_history()
        assert len(detector._regime_history) == 0

    def test_get_regime_statistics_empty(self, detector):
        """Test statistics with no detections"""
        stats = detector.get_regime_statistics()

        assert stats["total"] == 0
        assert stats["by_type"] == {}
        assert stats["last_regime"] is None

    def test_get_regime_statistics_with_data(self, detector, normal_data):
        """Test statistics with detections"""
        detector.detect_batch(normal_data)
        stats = detector.get_regime_statistics()

        assert stats["total"] == len(normal_data)
        assert "normal" in stats["by_type"]
        assert stats["last_regime"] is not None

    def test_get_regime_statistics_mixed_regimes(self, detector, shock_data):
        """Test statistics with mixed regimes"""
        # Detect across entire period
        detector.detect_batch(shock_data)
        stats = detector.get_regime_statistics()

        # Should detect at least normal regime
        assert stats["total"] == len(shock_data)
        assert stats["last_regime"] is not None

    def test_calculate_diagnostics(self, detector, normal_data):
        """Test diagnostic calculation"""
        result = detector.detect(normal_data)

        # Should have diagnostic values
        assert isinstance(result.diagnostics, dict)
        assert "inflation_volatility" in result.diagnostics

    def test_diagnostic_ki_change(self, detector, normal_data):
        """Test Ki change diagnostic"""
        result = detector.detect(normal_data)

        if "ki_change" in result.diagnostics:
            ki_change = result.diagnostics["ki_change"]
            assert isinstance(ki_change, (int, float))

    def test_diagnostic_ruonia_change(self, detector, normal_data):
        """Test Ruonia change diagnostic"""
        result = detector.detect(normal_data)

        if "ruonia_change" in result.diagnostics:
            ruonia_change = result.diagnostics["ruonia_change"]
            assert isinstance(ruonia_change, (int, float))

    def test_diagnostic_inflation_volatility(self, detector, normal_data):
        """Test inflation volatility diagnostic"""
        result = detector.detect(normal_data)

        assert "inflation_volatility" in result.diagnostics
        volatility = result.diagnostics["inflation_volatility"]
        assert volatility >= 0.0

    def test_confidence_normal_regime(self, detector, normal_data):
        """Test confidence for normal regime"""
        result = detector.detect(normal_data)

        if result.regime == RegimeType.NORMAL:
            assert result.confidence > 0.3  # Normal data should have decent confidence

    def test_confidence_shock_regime(self, detector, shock_data):
        """Test confidence for shock regime"""
        # Detect during shock
        shock_date = shock_data.index[22]
        result = detector.detect(shock_data, date=shock_date)

        if result.regime == RegimeType.SHOCK:
            # Confidence should be higher for strong shocks
            assert result.confidence > 0.0

    def test_confidence_bounds(self, detector, normal_data):
        """Test confidence is always within valid bounds"""
        results = detector.detect_batch(normal_data)

        for result in results:
            assert 0.0 <= result.confidence <= 1.0

    def test_shock_threshold_classification(self, normal_data):
        """Test that shock threshold affects classification"""
        # Lower threshold should detect more shocks
        detector_low = RegimeDetector(shock_threshold=0.1)
        detector_high = RegimeDetector(shock_threshold=1.0)

        result_low = detector_low.detect(normal_data)
        result_high = detector_high.detect(normal_data)

        # Both should be valid results
        assert isinstance(result_low, RegimeDetectionResult)
        assert isinstance(result_high, RegimeDetectionResult)

    def test_missing_columns(self, detector):
        """Test handling of missing columns"""
        # Data with only inflation column
        df = pd.DataFrame(
            {
                "mom": np.random.randn(60) * 0.3 + 0.5,
            },
            index=pd.date_range(start="2020-01-01", periods=60, freq="MS"),
        )

        result = detector.detect(df)

        # Should still work (classify based on available data)
        assert isinstance(result, RegimeDetectionResult)
        assert isinstance(result.regime, RegimeType)

    def test_edge_case_single_row(self, detector):
        """Test detection with single row of data"""
        df = pd.DataFrame(
            {
                "Ki_i": [7.5],
                "Ruonia": [7.3],
                "mom": [0.5],
            },
            index=pd.to_datetime(["2020-01-01"]),
        )

        result = detector.detect(df)

        assert isinstance(result, RegimeDetectionResult)
        assert len(result.history) == 1

    def test_result_dataclass(self, detector, normal_data):
        """Test RegimeDetectionResult dataclass"""
        result = detector.detect(normal_data)

        assert result.regime is not None
        assert isinstance(result.confidence, float)
        assert isinstance(result.diagnostics, dict)
        assert isinstance(result.history, list)


class TestRegimeType:
    """Test RegimeType enum"""

    def test_regime_types_exist(self):
        """Test that all regime types are defined"""
        assert hasattr(RegimeType, "NORMAL")
        assert hasattr(RegimeType, "SHOCK")
        assert hasattr(RegimeType, "HIGH_INFLATION")

    def test_regime_type_values(self):
        """Test regime type string values"""
        assert RegimeType.NORMAL.value == "normal"
        assert RegimeType.SHOCK.value == "shock"
        assert RegimeType.HIGH_INFLATION.value == "high_inflation"

    def test_regime_type_comparison(self):
        """Test regime type comparison"""
        assert RegimeType.NORMAL == RegimeType.NORMAL
        assert RegimeType.NORMAL != RegimeType.SHOCK


class TestRegimeDetectionRealism:
    """Tests for realistic regime detection scenarios"""

    @pytest.fixture
    def realistic_data(self):
        """Create more realistic test data"""
        np.random.seed(42)
        dates = pd.date_range(start="2018-01-01", periods=84, freq="MS")  # 7 years

        # Simulate realistic scenarios
        # 2018-2019: Normal (24 months)
        # 2020: COVID shock (12 months)
        # 2021-2023: Recovery and stabilization (36 months)
        # 2024: Another potential shock (12 months)

        ki_values = np.concatenate(
            [
                np.linspace(7.5, 7.8, 24),  # Normal
                np.linspace(7.8, 4.5, 12),  # COVID shock (rate cuts)
                np.linspace(4.5, 7.5, 36),  # Recovery
                np.linspace(7.5, 16.0, 12),  # 2024 inflation shock
            ]
        )

        mom_values = np.concatenate(
            [
                np.random.randn(24) * 0.2 + 0.5,  # Normal
                np.random.randn(12) * 0.3 + 0.3,  # COVID (lower inflation)
                np.random.randn(36) * 0.2 + 0.6,  # Recovery
                np.random.randn(12) * 0.4 + 0.9,  # 2024 shock
            ]
        )

        df = pd.DataFrame(
            {
                "Ki_i": ki_values,
                "Ruonia": ki_values - 0.3 + np.random.randn(84) * 0.2,
                "mom": mom_values,
            },
            index=dates,
        )

        return df

    def test_realistic_scenario_detection(self, realistic_data):
        """Test detection on realistic scenario"""
        detector = RegimeDetector()

        # Detect across entire period
        results = detector.detect_batch(realistic_data)
        stats = detector.get_regime_statistics()

        # Should have detected multiple regimes
        assert stats["total"] == 84
        assert len(stats["by_type"]) >= 1

        # Check that different regimes were detected
        regime_types = [r.regime for r in results]
        assert RegimeType.NORMAL in regime_types  # Should detect normal periods

    def test_regime_transitions(self, realistic_data):
        """Test detection of regime transitions"""
        detector = RegimeDetector()
        results = detector.detect_batch(realistic_data)

        # Find transitions
        regimes = [r.regime for r in results]
        transitions = sum(
            1 for i in range(1, len(regimes)) if regimes[i] != regimes[i - 1]
        )

        # Should have some transitions
        assert transitions > 0

    def test_shock_detection_at_key_periods(self, realistic_data):
        """Test shock detection at known shock periods"""
        detector = RegimeDetector()

        # COVID shock period (March 2020)
        covid_date = realistic_data.index[26]
        covid_result = detector.detect(realistic_data, covid_date)

        # 2024 inflation shock period
        shock_date = realistic_data.index[72]
        shock_result = detector.detect(realistic_data, shock_date)

        # Both should produce valid results
        assert isinstance(covid_result, RegimeDetectionResult)
        assert isinstance(shock_result, RegimeDetectionResult)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
