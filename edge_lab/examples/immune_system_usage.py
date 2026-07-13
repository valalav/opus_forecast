"""
Example: Using the Immune System with Real Forecasting Models

This example demonstrates how to use the ImmuneSystemTester to stress test
actual forecasting models from the SIRENA system.

To run this example:
    python3 examples/immune_system_usage.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "agents"))

import numpy as np
import pandas as pd
from immune_system import ImmuneSystemTester


def create_forecaster_model(name):
    """
    Example of how to wrap a real forecasting model for immune system testing.

    In production, you would replace this with actual models like:
    - RidgeExtendedForecaster
    - NGBoostForecaster
    - SubcomponentForecaster
    """

    class ForecastingModel:
        """Wrapper for SIRENA forecasting model"""

        def __init__(self):
            self.name = name
            self._fitted = False

        def fit(self, data):
            """Fit model to training data"""
            # In production, this would call model.fit(df, 'Все товары и услуги')
            self._fitted = True
            self.mean_value = data.iloc[:, 0].mean()
            return self

        def predict(self, data):
            """Make prediction for next period"""
            if not self._fitted:
                raise RuntimeError("Model not fitted")
            # In production, this would call model.predict(df_ext, target_date)
            return self.mean_value

        def forecast(self, horizon=1):
            """Make multi-step forecast"""
            return [self.mean_value] * horizon

    return ForecastingModel()


def main():
    """Demonstrate immune system with forecasting models"""

    print("=" * 70)
    print("🛡️ IMMUNE SYSTEM: STRESS TESTING REAL FORECASTERS")
    print("=" * 70)
    print()

    # Load historical inflation data
    # In production, load from data/inflation_data.csv
    np.random.seed(42)
    dates = pd.date_range(start="2018-01-01", periods=72, freq="MS")

    # Simulate inflation data (in production: load from CSV)
    t = np.arange(len(dates))
    seasonal = 0.3 * np.sin(2 * np.pi * t / 12)
    noise = np.random.randn(len(dates)) * 0.2

    mom = 0.5 + seasonal + noise

    df = pd.DataFrame(
        {
            "target": mom,
            "mom": mom,
        },
        index=dates,
    )

    print(f"📊 Loaded Data: {len(df)} months ({dates[0].year}-{dates[-1].year})")
    print()

    # Create immune system tester
    tester = ImmuneSystemTester(
        target_col="target",
        survival_threshold_mae=2.0,
        prediction_bounds=(-5.0, 10.0),  # MoM in %
    )

    # Create models to test
    print("🤖 Creating Forecasting Models:")
    models = [
        create_forecaster_model("RidgeExtendedForecaster"),
        create_forecaster_model("NGBoostForecaster"),
        create_forecaster_model("SubcomponentForecaster"),
        create_forecaster_model("HuberForecaster"),
    ]

    for model in models:
        print(f"   - {model.name}")
    print()

    # Run stress tests
    print("=" * 70)
    print("⚡ RUNNING STRESS TESTS")
    print("=" * 70)
    print()

    reports = tester.test_models(models=models, train_data=df, min_survival_rate=90.0)

    # Calculate summary
    survival_rates = [r.survival_rate for r in reports.values()]
    avg_survival = np.mean(survival_rates)

    print()
    print("=" * 70)
    print("📊 STRESS TEST SUMMARY")
    print("=" * 70)
    print()

    for model_name, report in reports.items():
        status = "✅" if report.survival_rate >= 90 else "❌"
        print(f"{status} {model_name:25s}: {report.survival_rate:5.1f}% survival")
        if report.vulnerabilities:
            print(f"    Vulnerabilities: {', '.join(report.vulnerabilities)}")
        print()

    print(f"Average Survival Rate: {avg_survival:.1f}%")
    print()

    # Generate detailed report
    report_path = "examples/stress_test_report.md"
    Path(report_path).parent.mkdir(exist_ok=True)
    tester.generate_report(reports, output_path=report_path)

    print(f"✅ Detailed report saved to: {report_path}")
    print()

    # Recommendations
    print("=" * 70)
    print("💡 RECOMMENDATIONS")
    print("=" * 70)
    print()

    if avg_survival >= 90:
        print("✅ Models are robust against black swan events")
        print("   - Current ensemble weights are acceptable")
        print("   - Consider adding more diverse models for further resilience")
    else:
        print("⚠️  Models need improvement:")
        print("   - Add robust loss functions (Huber, Quantile)")
        print("   - Implement ensemble methods to average out outliers")
        print("   - Add regime-switching to handle structural breaks")
        print("   - Consider conformal prediction for uncertainty quantification")

    print()
    print("=" * 70)


if __name__ == "__main__":
    main()
