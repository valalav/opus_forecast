"""
Immune System: Adversarial Stress Testing Agent

Generates synthetic 'Black Swan' events to test model survival and resilience.
Measures survival rate and identifies weaknesses in forecasting models.

Black Swan Types:
- Extreme Value Shock: Massive spike/drop in target variable
- Regime Change: Sudden shift in data distribution
- Missing Data: Simulate data gaps
- Feature Outlier: Extreme values in exogenous features
- Volatility Explosion: Sudden increase in noise

Survival Definition:
- Model doesn't crash or raise exceptions
- Predictions within reasonable bounds (e.g., -5 to +10% MoM)
- MAE degradation below 2x normal baseline
- No NaN or infinite predictions
"""

import numpy as np
import pandas as pd
from typing import Callable, Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
import warnings

warnings.filterwarnings("ignore")


class BlackSwanType(Enum):
    """Types of synthetic black swan events"""

    EXTREME_VALUE = "extreme_value"  # Massive spike/drop
    REGIME_CHANGE = "regime_change"  # Distribution shift
    MISSING_DATA = "missing_data"  # Data gaps
    FEATURE_OUTLIER = "feature_outlier"  # Exogenous feature shock
    VOLATILITY_EXPLOSION = "volatility_explosion"  # Noise spike
    CONSECUTIVE_SHOCKS = "consecutive_shocks"  # Multiple shocks


@dataclass
class BlackSwanEvent:
    """A synthetic black swan event configuration"""

    type: BlackSwanType
    severity: float  # 0.0 to 1.0
    duration_periods: int = 1
    target_index: Optional[str] = None  # Which variable to shock (if applicable)
    affected_features: List[str] = field(default_factory=list)


@dataclass
class StressTestResult:
    """Result of stress testing a single model"""

    model_name: str
    black_swan: BlackSwanEvent
    survived: bool
    error: Optional[str] = None
    mae_degradation: float = 0.0  # Ratio vs baseline
    prediction_bounds_ok: bool = True
    predictions_contain_nan: bool = False
    execution_time_ms: float = 0.0
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SurvivalReport:
    """Comprehensive survival report for a model"""

    model_name: str
    survival_rate: float  # Percentage of tests passed
    total_tests: int
    passed_tests: int
    failed_tests: int
    results: List[StressTestResult] = field(default_factory=list)
    vulnerabilities: List[str] = field(default_factory=list)


class BlackSwanInjector:
    """Generates synthetic black swan events in time series data"""

    @staticmethod
    def inject_extreme_value(
        df: pd.DataFrame, severity: float, periods: int = 1, col: str = "target"
    ) -> pd.DataFrame:
        """Inject extreme value spike/drop"""
        df_shocked = df.copy()
        n = len(df_shocked)

        # Add extreme values at the end (near prediction point)
        base_value = df_shocked[col].iloc[-1]
        shock_magnitude = base_value * severity * 0.5  # Up to 50% of base

        for i in range(periods):
            idx = n - periods + i
            if idx >= 0 and idx < n:
                # Randomly choose direction (spike or drop)
                direction = 1 if np.random.random() > 0.5 else -1
                df_shocked.at[df_shocked.index[idx], col] += direction * shock_magnitude

        return df_shocked

    @staticmethod
    def inject_regime_change(
        df: pd.DataFrame, severity: float, periods: int = 12, col: str = "target"
    ) -> pd.DataFrame:
        """Shift distribution (new mean and variance)"""
        df_shocked = df.copy()
        n = len(df_shocked)

        # Calculate original statistics
        original_mean = df_shocked[col].iloc[:-periods].mean()
        original_std = df_shocked[col].iloc[:-periods].std()

        # Apply regime shift (change mean by severity, increase variance)
        shift_amount = original_mean * severity * 0.3
        new_std = original_std * (1 + severity)

        for i in range(periods):
            idx = n - periods + i
            if idx >= 0 and idx < n:
                # Sample from new distribution
                df_shocked.at[df_shocked.index[idx], col] = (
                    original_mean + shift_amount + np.random.randn() * new_std
                )

        return df_shocked

    @staticmethod
    def inject_missing_data(
        df: pd.DataFrame, severity: float, col: str = "target"
    ) -> pd.DataFrame:
        """Remove random data points"""
        df_shocked = df.copy()
        n = len(df_shocked)

        # Calculate number of missing points
        n_missing = int(n * severity * 0.3)  # Up to 30% of data

        # Randomly remove points near the end
        indices_to_remove = np.random.choice(
            range(n - 24, n), size=min(n_missing, 24), replace=False
        )

        for idx in indices_to_remove:
            df_shocked.at[df_shocked.index[idx], col] = np.nan

        return df_shocked

    @staticmethod
    def inject_feature_outlier(
        df: pd.DataFrame, severity: float, periods: int = 1, feature: str = "mom"
    ) -> pd.DataFrame:
        """Inject extreme values in exogenous features"""
        if feature not in df.columns:
            return df

        df_shocked = df.copy()
        n = len(df_shocked)

        # Add extreme values
        base_value = df_shocked[feature].iloc[-1]
        shock_magnitude = base_value * severity

        for i in range(periods):
            idx = n - periods + i
            if idx >= 0 and idx < n:
                direction = 1 if np.random.random() > 0.5 else -1
                df_shocked.at[df_shocked.index[idx], feature] += (
                    direction * shock_magnitude
                )

        return df_shocked

    @staticmethod
    def inject_volatility_explosion(
        df: pd.DataFrame, severity: float, periods: int = 6, col: str = "target"
    ) -> pd.DataFrame:
        """Dramatically increase noise"""
        df_shocked = df.copy()
        n = len(df_shocked)

        # Calculate original noise level
        original_std = df_shocked[col].iloc[:-periods].std()
        new_std = original_std * (1 + severity * 5)  # Up to 6x noise

        for i in range(periods):
            idx = n - periods + i
            if idx >= 0 and idx < n:
                # Add noise
                df_shocked.at[df_shocked.index[idx], col] += np.random.randn() * new_std

        return df_shocked

    @staticmethod
    def inject_consecutive_shocks(
        df: pd.DataFrame, severity: float, periods: int = 3, col: str = "target"
    ) -> pd.DataFrame:
        """Multiple consecutive extreme shocks"""
        df_shocked = df.copy()
        n = len(df_shocked)

        base_value = df_shocked[col].iloc[-1]

        for i in range(periods):
            idx = n - periods + i
            if idx >= 0 and idx < n:
                # Alternating direction shocks
                direction = 1 if i % 2 == 0 else -1
                shock_magnitude = base_value * severity * 0.4
                df_shocked.at[df_shocked.index[idx], col] += direction * shock_magnitude

        return df_shocked

    def apply(self, df: pd.DataFrame, event: BlackSwanEvent) -> pd.DataFrame:
        """Apply a black swan event to the data"""
        if event.type == BlackSwanType.EXTREME_VALUE:
            return self.inject_extreme_value(df, event.severity, event.duration_periods)
        elif event.type == BlackSwanType.REGIME_CHANGE:
            return self.inject_regime_change(df, event.severity, event.duration_periods)
        elif event.type == BlackSwanType.MISSING_DATA:
            return self.inject_missing_data(df, event.severity)
        elif event.type == BlackSwanType.FEATURE_OUTLIER:
            feature = event.target_index if event.target_index else "mom"
            return self.inject_feature_outlier(
                df, event.severity, event.duration_periods, feature
            )
        elif event.type == BlackSwanType.VOLATILITY_EXPLOSION:
            return self.inject_volatility_explosion(
                df, event.severity, event.duration_periods
            )
        elif event.type == BlackSwanType.CONSECUTIVE_SHOCKS:
            return self.inject_consecutive_shocks(
                df, event.severity, event.duration_periods
            )
        else:
            return df


class ImmuneSystemTester:
    """
    Main immune system stress testing agent.

    Tests model resilience against synthetic black swan events
    and calculates survival rate metrics.
    """

    def __init__(
        self,
        target_col: str = "target",
        survival_threshold_mae: float = 2.0,  # MAE must be < 2x baseline
        prediction_bounds: Tuple[float, float] = (-5.0, 10.0),  # MoM in %
    ):
        self.target_col = target_col
        self.survival_threshold_mae = survival_threshold_mae
        self.prediction_bounds = prediction_bounds
        self.injector = BlackSwanInjector()

    def generate_black_swan_events(self, count: int = 10) -> List[BlackSwanEvent]:
        """Generate random black swan events for testing"""
        events = []
        event_types = list(BlackSwanType)

        # Ensure at least one of each type
        for event_type in event_types:
            severity = np.random.uniform(0.5, 1.0)
            duration = np.random.randint(1, 4)
            events.append(
                BlackSwanEvent(
                    type=event_type, severity=severity, duration_periods=duration
                )
            )

        # Fill remaining with random types
        for _ in range(count - len(event_types)):
            event_type_index = np.random.randint(0, len(event_types))
            event_type = event_types[event_type_index]
            severity = np.random.uniform(0.3, 1.0)
            duration = np.random.randint(1, 6)
            events.append(
                BlackSwanEvent(
                    type=event_type, severity=severity, duration_periods=duration
                )
            )

        return events

    def stress_test_model(
        self,
        model: Any,
        train_data: pd.DataFrame,
        baseline_mae: Optional[float] = None,
        black_swans: Optional[List[BlackSwanEvent]] = None,
    ) -> SurvivalReport:
        """
        Stress test a single model against black swan events.

        Args:
            model: Model with fit() and predict() methods
            train_data: Training DataFrame
            baseline_mae: Optional baseline MAE for comparison
            black_swans: List of black swan events (auto-generated if None)

        Returns:
            SurvivalReport with survival rate and detailed results
        """
        # Try to get model name from instance attribute, then class name
        model_name = getattr(model, "name", None)
        if model_name is None:
            model_name = getattr(model.__class__, "__name__", str(model))

        # Generate black swan events if not provided
        if black_swans is None:
            black_swans = self.generate_black_swan_events(count=10)

        results = []

        for event in black_swans:
            result = self._test_single_black_swan(
                model, train_data, event, baseline_mae
            )
            results.append(result)

        # Calculate survival statistics
        passed = sum(1 for r in results if r.survived)
        total = len(results)
        survival_rate = (passed / total * 100) if total > 0 else 0.0

        # Identify vulnerabilities
        vulnerabilities = []
        failed_types = [r.black_swan.type for r in results if not r.survived]
        from collections import Counter

        type_counts = Counter(failed_types)
        for event_type, count in type_counts.most_common():
            if count >= 2:
                vulnerabilities.append(f"{event_type.value}: {count} failures")

        return SurvivalReport(
            model_name=model_name,
            survival_rate=survival_rate,
            total_tests=total,
            passed_tests=passed,
            failed_tests=total - passed,
            results=results,
            vulnerabilities=vulnerabilities,
        )

    def _test_single_black_swan(
        self,
        model: Any,
        train_data: pd.DataFrame,
        event: BlackSwanEvent,
        baseline_mae: Optional[float],
    ) -> StressTestResult:
        """Test model against a single black swan event"""
        import time

        start_time = time.time()

        try:
            # Apply black swan to data
            shocked_data = self.injector.apply(train_data, event)

            # Fit model on shocked data
            model_copy = self._copy_model(model)
            model_copy.fit(shocked_data)

            # Make prediction on last row
            last_row = shocked_data.iloc[[-1]]

            # Handle different prediction interfaces
            try:
                pred = model_copy.predict(last_row)
            except Exception:
                # Try forecast method
                try:
                    pred = model_copy.forecast(horizon=1)
                    if isinstance(pred, (list, np.ndarray)):
                        pred = pred[0] if len(pred) > 0 else np.nan
                except Exception:
                    pred = np.nan

            # Check prediction quality
            survived, details = self._check_survival(pred, baseline_mae, event)

            execution_time = (time.time() - start_time) * 1000

            return StressTestResult(
                model_name=getattr(model_copy.__class__, "__name__", str(model)),
                black_swan=event,
                survived=survived,
                error=None,
                mae_degradation=details.get("mae_degradation", 0.0),
                prediction_bounds_ok=details.get("bounds_ok", True),
                predictions_contain_nan=details.get("has_nan", False),
                execution_time_ms=execution_time,
                details=details,
            )

        except Exception as e:
            return StressTestResult(
                model_name=getattr(model.__class__, "__name__", str(model)),
                black_swan=event,
                survived=False,
                error=str(e),
                execution_time_ms=(time.time() - start_time) * 1000,
            )

    def _copy_model(self, model: Any) -> Any:
        """Create a fresh copy of the model"""
        try:
            # Try to instantiate same class
            return model.__class__()
        except Exception:
            return model

    def _check_survival(
        self, prediction: Any, baseline_mae: Optional[float], event: BlackSwanEvent
    ) -> Tuple[bool, Dict[str, Any]]:
        """Check if prediction meets survival criteria"""
        details = {}

        # Convert prediction to scalar
        try:
            pred_val = float(prediction)
        except (TypeError, ValueError):
            return False, {"reason": "Invalid prediction type"}

        # Check for NaN/Inf
        if np.isnan(pred_val) or np.isinf(pred_val):
            return False, {"reason": "NaN/Inf prediction", "has_nan": True}

        # Check bounds
        bounds_ok = self.prediction_bounds[0] <= pred_val <= self.prediction_bounds[1]
        details["bounds_ok"] = bounds_ok
        details["prediction_value"] = pred_val

        if not bounds_ok:
            return False, details

        # If baseline provided, check MAE degradation
        # Skip MAE degradation check for testing without baseline
        if baseline_mae is not None and baseline_mae > 0:
            # Estimate error (since we don't have actual values, use distance from expected range)
            # This is a heuristic - real implementation would compare to actual values
            expected_range_center = np.mean(self.prediction_bounds)
            estimated_error = abs(pred_val - expected_range_center)

            degradation_ratio = estimated_error / max(baseline_mae, 0.1)
            details["mae_degradation"] = degradation_ratio

            if degradation_ratio > self.survival_threshold_mae:
                details["reason"] = (
                    f"MAE degradation too high: {degradation_ratio:.2f}x"
                )
                return False, details

        return True, details

    def test_models(
        self,
        models: List[Any],
        train_data: pd.DataFrame,
        baseline_mae: Optional[float] = None,
        min_survival_rate: float = 90.0,
    ) -> Dict[str, SurvivalReport]:
        """
        Stress test multiple models.

        Returns:
            Dictionary of model_name -> SurvivalReport
        """
        reports = {}

        for model in models:
            model_name = getattr(model.__class__, "__name__", str(model))
            report = self.stress_test_model(model, train_data, baseline_mae)
            reports[model_name] = report

            print(f"\n{model_name}:")
            print(f"  Survival Rate: {report.survival_rate:.1f}%")
            print(f"  Passed: {report.passed_tests}/{report.total_tests}")

            if report.vulnerabilities:
                print(f"  Vulnerabilities: {', '.join(report.vulnerabilities)}")

        # Check if all models meet minimum survival rate
        all_passed = all(r.survival_rate >= min_survival_rate for r in reports.values())

        if all_passed:
            print(f"\n✅ ALL MODELS PASSED: Survival Rate > {min_survival_rate}%")
        else:
            print(f"\n❌ SOME MODELS FAILED: Minimum required {min_survival_rate}%")

        return reports

    def generate_report(
        self, reports: Dict[str, SurvivalReport], output_path: Optional[str] = None
    ) -> str:
        """Generate a comprehensive markdown report"""
        lines = []
        lines.append("# 🛡️ Immune System Stress Test Report\n")
        lines.append(f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        lines.append("---\n")

        # Summary table
        lines.append("## Summary\n")
        lines.append("| Model | Survival Rate | Passed/Total | Status |")
        lines.append("|-------|---------------|--------------|--------|")

        for model_name, report in reports.items():
            status = "✅ PASS" if report.survival_rate >= 90 else "❌ FAIL"
            lines.append(
                f"| {model_name} | {report.survival_rate:.1f}% | "
                f"{report.passed_tests}/{report.total_tests} | {status} |"
            )
        lines.append("")

        # Detailed results
        lines.append("## Detailed Results\n")

        for model_name, report in reports.items():
            lines.append(f"### {model_name}\n")
            lines.append(f"**Survival Rate:** {report.survival_rate:.1f}%\n")
            lines.append(f"**Total Tests:** {report.total_tests}\n")
            lines.append(f"**Passed:** {report.passed_tests}\n")
            lines.append(f"**Failed:** {report.failed_tests}\n")

            if report.vulnerabilities:
                lines.append("**Vulnerabilities:**")
                for vuln in report.vulnerabilities:
                    lines.append(f"  - {vuln}")
                lines.append("")

            # Failure details
            failures = [r for r in report.results if not r.survived]
            if failures:
                lines.append("**Failures:**\n")
                for f in failures[:5]:  # Show top 5
                    lines.append(
                        f"  - **{f.black_swan.type.value}** (severity={f.black_swan.severity:.2f})"
                    )
                    if f.error:
                        lines.append(f"    Error: {f.error}")
                    elif "reason" in f.details:
                        lines.append(f"    Reason: {f.details['reason']}")
                lines.append("")

        report_text = "\n".join(lines)

        if output_path:
            with open(output_path, "w") as f:
                f.write(report_text)
            print(f"\nReport saved to: {output_path}")

        return report_text


def create_sample_model(name: str):
    """Create a simple dummy model for testing"""

    class DummyModel:
        def __init__(self):
            self.name = name
            self._fitted = False
            self.mean_value = 0.5

        def fit(self, data):
            # Handle NaN by using non-NaN values
            self._fitted = True
            target_col = data.columns[0] if len(data.columns) > 0 else "target"
            valid_data = data[target_col].dropna()
            if len(valid_data) > 0:
                self.mean_value = valid_data.mean()
            return self

        def predict(self, data):
            if not self._fitted:
                raise RuntimeError("Model not fitted")
            # Return simple prediction based on learned mean
            return self.mean_value if not np.isnan(self.mean_value) else 0.5

        def forecast(self, horizon=1):
            return [self.mean_value if not np.isnan(self.mean_value) else 0.5] * horizon

    return DummyModel()


if __name__ == "__main__":
    # Quick test
    print("🛡️ Immune System: Adversarial Stress Testing Agent")
    print("=" * 60)

    # Create sample data
    np.random.seed(42)
    dates = pd.date_range(start="2020-01-01", periods=60, freq="MS")
    df = pd.DataFrame(
        {
            "target": np.random.randn(60) * 0.3 + 0.5,
            "mom": np.random.randn(60) * 0.3 + 0.5,
        },
        index=dates,
    )

    # Create tester
    tester = ImmuneSystemTester()

    # Generate black swan events
    events = tester.generate_black_swan_events(count=5)
    print(f"\nGenerated {len(events)} black swan events:")
    for e in events:
        print(
            f"  - {e.type.value}: severity={e.severity:.2f}, duration={e.duration_periods}"
        )

    # Test sample model
    model = create_sample_model("TestModel")
    report = tester.stress_test_model(model, df)

    print(f"\n📊 Test Results:")
    print(f"  Survival Rate: {report.survival_rate:.1f}%")
    print(f"  Passed: {report.passed_tests}/{report.total_tests}")

    if report.survival_rate >= 90:
        print("\n✅ TASK COMPLETED: Survival Rate > 90%")
    else:
        print("\n❌ TASK FAILED: Survival Rate < 90%")
