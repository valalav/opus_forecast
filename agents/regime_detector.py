"""
Regime Detector: Detects market regimes and shock states in time series data.

Detects three main regimes:
1. Normal: Standard market conditions
2. Shock: Significant rate changes (|ΔKi| > 0.5 or |ΔRuonia| > 0.5)
3. High Inflation: High inflation acceleration (ΔИнфляция > 1.5pp)

Based on methodology from SIRENA-КБR forecasting system.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Literal
from dataclasses import dataclass, field
from enum import Enum


class RegimeType(Enum):
    """Types of market regimes"""

    NORMAL = "normal"
    SHOCK = "shock"
    HIGH_INFLATION = "high_inflation"


@dataclass
class RegimeDetectionResult:
    """Result of regime detection"""

    regime: RegimeType
    confidence: float  # 0.0 to 1.0
    diagnostics: Dict[str, float] = field(default_factory=dict)
    history: List[RegimeType] = field(default_factory=list)


class RegimeDetector:
    """
    Detects market regimes based on key indicators.

    Uses threshold-based detection for:
    - Interest rate shocks (Ki, Ruonia)
    - Inflation acceleration
    - Volatility changes
    """

    def __init__(
        self,
        shock_threshold: float = 0.5,  # Threshold for rate change shocks
        inflation_shock_threshold: float = 1.5,  # Threshold for inflation change (pp)
        volatility_window: int = 3,  # Window for volatility calculation
    ):
        self.shock_threshold = shock_threshold
        self.inflation_shock_threshold = inflation_shock_threshold
        self.volatility_window = volatility_window
        self._regime_history: List[RegimeType] = []

    def detect(
        self,
        df: pd.DataFrame,
        date: Optional[pd.Timestamp] = None,
        ki_col: str = "Ki_i",
        ruonia_col: str = "Ruonia",
        inflation_col: str = "mom",
    ) -> RegimeDetectionResult:
        """
        Detect the current regime from data.

        Args:
            df: DataFrame with time series data
            date: Target date (uses last row if None)
            ki_col: Column name for key rate
            ruonia_col: Column name for Ruonia rate
            inflation_col: Column name for inflation (MoM)

        Returns:
            RegimeDetectionResult with regime and diagnostics
        """
        if date is None:
            date = df.index[-1]

        # Get row for target date
        if date in df.index:
            row_idx = df.index.get_loc(date)
        else:
            # Use last row if date not found
            row_idx = len(df) - 1
            date = df.index[row_idx]

        # Calculate diagnostics
        diagnostics = self._calculate_diagnostics(
            df, row_idx, ki_col, ruonia_col, inflation_col
        )

        # Determine regime
        regime = self._classify_regime(diagnostics)

        # Update history
        self._regime_history.append(regime)

        # Calculate confidence based on diagnostic values
        confidence = self._calculate_confidence(diagnostics, regime)

        return RegimeDetectionResult(
            regime=regime,
            confidence=confidence,
            diagnostics=diagnostics,
            history=self._regime_history.copy(),
        )

    def _calculate_diagnostics(
        self,
        df: pd.DataFrame,
        row_idx: int,
        ki_col: str,
        ruonia_col: str,
        inflation_col: str,
    ) -> Dict[str, float]:
        """Calculate diagnostic metrics"""
        diagnostics = {}

        # Rate changes (differences)
        if ki_col in df.columns and row_idx >= 1:
            ki_curr = df[ki_col].iloc[row_idx]
            ki_prev = df[ki_col].iloc[row_idx - 1]
            diagnostics["ki_change"] = ki_curr - ki_prev

        if ruonia_col in df.columns and row_idx >= 1:
            ruonia_curr = df[ruonia_col].iloc[row_idx]
            ruonia_prev = df[ruonia_col].iloc[row_idx - 1]
            diagnostics["ruonia_change"] = ruonia_curr - ruonia_prev

        # Inflation change (YoY acceleration)
        if inflation_col in df.columns and row_idx >= 12:
            infl_curr = df[inflation_col].iloc[row_idx]
            infl_prev = df[inflation_col].iloc[row_idx - 12]
            diagnostics["inflation_change_yoy"] = infl_curr - infl_prev

        # Inflation volatility
        if inflation_col in df.columns:
            start_idx = max(0, row_idx - self.volatility_window)
            inflation_slice = df[inflation_col].iloc[start_idx : row_idx + 1]
            diagnostics["inflation_volatility"] = inflation_slice.std()

        return diagnostics

    def _classify_regime(self, diagnostics: Dict[str, float]) -> RegimeType:
        """Classify regime based on diagnostics"""
        # Check for shock regime (rate changes)
        ki_change = diagnostics.get("ki_change", 0.0)
        ruonia_change = diagnostics.get("ruonia_change", 0.0)

        if (
            abs(ki_change) > self.shock_threshold
            or abs(ruonia_change) > self.shock_threshold
        ):
            return RegimeType.SHOCK

        # Check for high inflation regime
        inflation_change = diagnostics.get("inflation_change_yoy", 0.0)
        if abs(inflation_change) > self.inflation_shock_threshold:
            return RegimeType.HIGH_INFLATION

        # Default to normal
        return RegimeType.NORMAL

    def _calculate_confidence(
        self, diagnostics: Dict[str, float], regime: RegimeType
    ) -> float:
        """Calculate confidence score for regime detection"""
        if regime == RegimeType.SHOCK:
            ki_change = diagnostics.get("ki_change", 0.0)
            ruonia_change = diagnostics.get("ruonia_change", 0.0)
            max_change = max(abs(ki_change), abs(ruonia_change))
            # Confidence increases with magnitude of shock above threshold
            confidence = min(1.0, max_change / (self.shock_threshold * 2))
            return confidence

        elif regime == RegimeType.HIGH_INFLATION:
            infl_change = diagnostics.get("inflation_change_yoy", 0.0)
            # Confidence increases with magnitude of inflation change
            confidence = min(
                1.0, abs(infl_change) / (self.inflation_shock_threshold * 2)
            )
            return confidence

        else:  # NORMAL
            # Confidence based on stability (low volatility, small changes)
            volatility = diagnostics.get("inflation_volatility", 0.5)
            ki_change = abs(diagnostics.get("ki_change", 0.0))
            ruonia_change = abs(diagnostics.get("ruonia_change", 0.0))
            infl_change = abs(diagnostics.get("inflation_change_yoy", 0.0))

            # Higher confidence when changes are small
            stability_score = 1.0 - min(1.0, volatility * 2)
            rate_stability = 1.0 - min(1.0, max(ki_change, ruonia_change) * 2)
            infl_stability = 1.0 - min(1.0, infl_change * 0.5)

            confidence = (stability_score + rate_stability + infl_stability) / 3
            return confidence

    def detect_batch(
        self, df: pd.DataFrame, dates: Optional[List[pd.Timestamp]] = None
    ) -> List[RegimeDetectionResult]:
        """
        Detect regimes for multiple dates.

        Args:
            df: DataFrame with time series data
            dates: List of dates to detect (uses all dates if None)

        Returns:
            List of RegimeDetectionResult
        """
        if dates is None:
            dates = df.index.tolist()

        results = []
        for date in dates:
            result = self.detect(df, date)
            results.append(result)

        return results

    def get_regime_statistics(self) -> Dict[str, any]:
        """Get statistics on detected regimes"""
        regime_counts = {}
        for regime in self._regime_history:
            regime_name = regime.value
            regime_counts[regime_name] = regime_counts.get(regime_name, 0) + 1

        return {
            "total": len(self._regime_history),
            "by_type": regime_counts,
            "last_regime": self._regime_history[-1].value
            if self._regime_history
            else None,
        }

    def reset_history(self):
        """Reset regime detection history"""
        self._regime_history.clear()


if __name__ == "__main__":
    # Quick test
    print("🔍 Regime Detector")
    print("=" * 60)

    # Create sample data with shocks
    np.random.seed(42)
    dates = pd.date_range(start="2020-01-01", periods=60, freq="MS")

    # Simulate normal period, then shock, then recovery
    ki_values = np.concatenate(
        [
            np.linspace(7.0, 7.5, 20),  # Normal
            np.linspace(7.5, 9.0, 5),  # Shock (rate hike)
            np.linspace(9.0, 8.5, 5),  # Stabilizing
            np.linspace(8.5, 8.0, 30),  # Normal
        ]
    )

    ruonia_values = ki_values + np.random.randn(60) * 0.2
    inflation_values = np.random.randn(60) * 0.3 + 0.5

    df = pd.DataFrame(
        {
            "Ki_i": ki_values,
            "Ruonia": ruonia_values,
            "mom": inflation_values,
        },
        index=dates,
    )

    # Detect regime at different points
    detector = RegimeDetector()

    test_dates = [
        pd.Timestamp("2020-06-01"),  # Normal
        pd.Timestamp("2020-11-01"),  # Shock (rate hike)
        pd.Timestamp("2024-12-01"),  # Normal
    ]

    for date in test_dates:
        if date in df.index:
            result = detector.detect(df, date)
            print(f"\n📅 Date: {date.strftime('%Y-%m-%d')}")
            print(f"  Regime: {result.regime.value}")
            print(f"  Confidence: {result.confidence:.2f}")
            if result.diagnostics:
                print(f"  Diagnostics:")
                for key, val in result.diagnostics.items():
                    print(f"    {key}: {val:.3f}")

    # Get statistics
    stats = detector.get_regime_statistics()
    print(f"\n📊 Regime Statistics:")
    print(f"  Total detections: {stats['total']}")
    print(f"  By type: {stats['by_type']}")
    print(f"  Last regime: {stats['last_regime']}")
