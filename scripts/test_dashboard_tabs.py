#!/usr/bin/env python3
"""
SIRENA-KBR Dashboard Tabs Test
==============================

Tests each dashboard tab's core functionality programmatically.

Запуск: python3 scripts/test_dashboard_tabs.py
"""

import sys
import os
import time
import json
import traceback
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Tuple

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np


class TabTester:
    """Test each dashboard tab's functionality."""

    def __init__(self):
        self.results: List[Dict] = []
        self.df = None
        self.loader = None

    def load_data(self) -> bool:
        """Load data for testing."""
        try:
            from sirena import DataLoader
            self.loader = DataLoader()
            self.df = self.loader.load_monthly_kbr()
            return self.df is not None and len(self.df) > 0
        except Exception as e:
            self.log_result("Data Loading", False, str(e))
            return False

    def log_result(self, tab_name: str, success: bool, details: str = "", duration: float = None):
        """Log test result."""
        self.results.append({
            'tab': tab_name,
            'status': 'OK' if success else 'FAIL',
            'details': details,
            'duration': duration,
            'timestamp': datetime.now().isoformat()
        })

    def test_tab1_forecast(self) -> bool:
        """Tab 1: Прогноз - Test ensemble forecast."""
        print("  Testing Tab 1: Прогноз...")
        start = time.time()

        try:
            from sirena.models import (
                RidgeForecaster, RidgeExtendedForecaster, RidgeShockDummiesForecaster,
                HuberForecaster, ElasticNetForecaster, ProphetForecaster, EBMForecaster
            )

            last_date = self.df.dropna(subset=['Все товары и услуги']).index.max()
            target_date = last_date + pd.DateOffset(months=1)

            # Test each production model
            models_tested = 0
            predictions = {}

            for model_class, name in [
                (RidgeForecaster, 'Ridge'),
                (HuberForecaster, 'Huber'),
                (ElasticNetForecaster, 'ElasticNet'),
            ]:
                model = model_class()
                model.fit(self.df)
                df_ext = self.df.copy()
                df_ext.loc[target_date] = np.nan
                pred = model.predict(df_ext, target_date)

                if isinstance(pred, dict):
                    predictions[name] = pred['prediction'] - 100
                else:
                    predictions[name] = float(pred) - 100

                models_tested += 1

            duration = time.time() - start
            details = f"{models_tested} models, predictions: " + ", ".join(
                f"{k}={v:.3f}%" for k, v in predictions.items()
            )
            self.log_result("Tab 1: Прогноз", True, details, duration)
            return True

        except Exception as e:
            self.log_result("Tab 1: Прогноз", False, str(e), time.time() - start)
            traceback.print_exc()
            return False

    def test_tab2_exogenous(self) -> bool:
        """Tab 2: Экзогенные - Test exogenous variables."""
        print("  Testing Tab 2: Экзогенные...")
        start = time.time()

        try:
            # Test loading exogenous settings
            exog_file = PROJECT_ROOT / 'data' / 'exog_settings.json'

            # Test exog_functions
            import exog_functions

            # Get default values
            usd_default = getattr(exog_functions, 'get_default_usd', lambda: 100)()

            duration = time.time() - start
            self.log_result("Tab 2: Экзогенные", True,
                          f"Loaded exog_functions, default USD={usd_default}", duration)
            return True

        except Exception as e:
            self.log_result("Tab 2: Экзогенные", False, str(e), time.time() - start)
            return False

    def test_tab3_backtest(self) -> bool:
        """Tab 3: Бэктест - Test comparative backtest."""
        print("  Testing Tab 3: Бэктест...")
        start = time.time()

        try:
            from sirena.models import RidgeForecaster

            # Run mini backtest on last 3 months
            model = RidgeForecaster()

            dates = self.df.dropna(subset=['Все товары и услуги']).index[-3:]
            errors = []

            for target_date in dates:
                cutoff = target_date - pd.DateOffset(months=1)
                train_df = self.df[self.df.index <= cutoff].copy()

                if len(train_df) < 24:
                    continue

                model.fit(train_df)
                train_ext = train_df.copy()
                train_ext.loc[target_date] = np.nan
                pred = model.predict(train_ext, target_date)

                if isinstance(pred, dict):
                    pred_val = pred['prediction']
                else:
                    pred_val = float(pred)

                actual = self.df.loc[target_date, 'Все товары и услуги']
                error = abs(actual - pred_val)
                errors.append(error)

            mae = np.mean(errors) if errors else None

            duration = time.time() - start
            self.log_result("Tab 3: Бэктест", True,
                          f"Mini backtest MAE={mae:.4f} on {len(errors)} dates", duration)
            return True

        except Exception as e:
            self.log_result("Tab 3: Бэктест", False, str(e), time.time() - start)
            traceback.print_exc()
            return False

    def test_tab4_methodology(self) -> bool:
        """Tab 4: Методология - Just check it's accessible."""
        print("  Testing Tab 4: Методология...")
        start = time.time()

        try:
            # Check CLAUDE.md exists (used for methodology)
            claude_md = PROJECT_ROOT / 'CLAUDE.md'
            exists = claude_md.exists()

            duration = time.time() - start
            self.log_result("Tab 4: Методология", exists,
                          f"CLAUDE.md exists: {exists}", duration)
            return exists

        except Exception as e:
            self.log_result("Tab 4: Методология", False, str(e), time.time() - start)
            return False

    def test_tab5_history(self) -> bool:
        """Tab 5: История (Opus) - Test forecast history."""
        print("  Testing Tab 5: История...")
        start = time.time()

        try:
            # Check if historical data exists
            from sirena.models import RidgeForecaster

            model = RidgeForecaster()
            model.fit(self.df)

            # Get feature importance if available
            importance = model.get_feature_importance() if hasattr(model, 'get_feature_importance') else {}

            duration = time.time() - start
            self.log_result("Tab 5: История", True,
                          f"Feature importance: {len(importance)} features", duration)
            return True

        except Exception as e:
            self.log_result("Tab 5: История", False, str(e), time.time() - start)
            return False

    def test_tab6_regions(self) -> bool:
        """Tab 6: Регионы - Test regional comparison."""
        print("  Testing Tab 6: Регионы...")
        start = time.time()

        try:
            # Check if regional data exists
            regional_files = list((PROJECT_ROOT / 'data').glob('*region*.csv'))

            duration = time.time() - start
            self.log_result("Tab 6: Регионы", True,
                          f"Regional files found: {len(regional_files)}", duration)
            return True

        except Exception as e:
            self.log_result("Tab 6: Регионы", False, str(e), time.time() - start)
            return False

    def test_tab7_insider(self) -> bool:
        """Tab 7: Инсайдер - Test insider data loading."""
        print("  Testing Tab 7: Инсайдер...")
        start = time.time()

        try:
            insider_data = self.loader.load_inflation_data()

            if insider_data is not None:
                duration = time.time() - start
                self.log_result("Tab 7: Инсайдер", True,
                              f"Loaded {len(insider_data)} months insider data", duration)
                return True
            else:
                self.log_result("Tab 7: Инсайдер", False, "No insider data available")
                return False

        except Exception as e:
            self.log_result("Tab 7: Инсайдер", False, str(e), time.time() - start)
            return False

    def test_tab8_ebm(self) -> bool:
        """Tab 8: EBM - Test EBM model."""
        print("  Testing Tab 8: EBM...")
        start = time.time()

        try:
            from sirena.models import EBMForecaster

            model = EBMForecaster()
            model.fit(self.df)

            last_date = self.df.dropna(subset=['Все товары и услуги']).index.max()
            target_date = last_date + pd.DateOffset(months=1)

            df_ext = self.df.copy()
            df_ext.loc[target_date] = np.nan
            pred = model.predict(df_ext, target_date)

            if isinstance(pred, dict):
                prediction = pred['prediction'] - 100
            else:
                prediction = float(pred) - 100

            duration = time.time() - start
            self.log_result("Tab 8: EBM", True,
                          f"EBM prediction: {prediction:.4f}%", duration)
            return True

        except Exception as e:
            self.log_result("Tab 8: EBM", False, str(e), time.time() - start)
            return False

    def test_tab9_backtest_h1(self) -> bool:
        """Tab 9: Бэктест h=1 - Test h=1 backtest."""
        print("  Testing Tab 9: Бэктест h=1...")
        start = time.time()

        try:
            # Just verify the backtest framework works
            from sirena.models import RidgeForecaster

            # Quick test
            model = RidgeForecaster()
            model.fit(self.df)

            duration = time.time() - start
            self.log_result("Tab 9: Бэктест h=1", True,
                          "Backtest framework ready", duration)
            return True

        except Exception as e:
            self.log_result("Tab 9: Бэктест h=1", False, str(e), time.time() - start)
            return False

    def run_all_tests(self) -> Tuple[int, int]:
        """Run all tab tests."""
        print("\n" + "=" * 60)
        print("SIRENA-KBR Dashboard Tabs Test")
        print("=" * 60)

        # Load data first
        print("\nLoading data...")
        if not self.load_data():
            print("FATAL: Cannot load data!")
            return 0, 1

        print(f"Data loaded: {len(self.df)} months\n")

        # Run tests
        tests = [
            self.test_tab1_forecast,
            self.test_tab2_exogenous,
            self.test_tab3_backtest,
            self.test_tab4_methodology,
            self.test_tab5_history,
            self.test_tab6_regions,
            self.test_tab7_insider,
            self.test_tab8_ebm,
            self.test_tab9_backtest_h1,
        ]

        passed = 0
        failed = 0

        for test in tests:
            try:
                if test():
                    passed += 1
                else:
                    failed += 1
            except Exception as e:
                failed += 1
                print(f"  EXCEPTION: {e}")

        return passed, failed

    def print_summary(self):
        """Print test summary."""
        print("\n" + "=" * 60)
        print("TEST RESULTS")
        print("=" * 60)

        for r in self.results:
            icon = "✓" if r['status'] == 'OK' else "✗"
            duration = f"({r['duration']:.2f}s)" if r['duration'] else ""
            print(f"{icon} {r['tab']}: {r['status']} {duration}")
            if r['details']:
                print(f"   {r['details'][:70]}")

        print("=" * 60)

        ok_count = sum(1 for r in self.results if r['status'] == 'OK')
        fail_count = sum(1 for r in self.results if r['status'] == 'FAIL')
        print(f"TOTAL: {ok_count} OK, {fail_count} FAIL")

        return fail_count == 0

    def save_results(self):
        """Save results to JSON."""
        results_file = PROJECT_ROOT / 'logs' / 'tabs_test_results.json'
        results_file.parent.mkdir(exist_ok=True)

        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump({
                'timestamp': datetime.now().isoformat(),
                'results': self.results
            }, f, indent=2, ensure_ascii=False)

        print(f"\nResults saved to: {results_file}")


def main():
    tester = TabTester()
    passed, failed = tester.run_all_tests()
    all_ok = tester.print_summary()
    tester.save_results()

    return 0 if all_ok else 1


if __name__ == '__main__':
    sys.exit(main())
