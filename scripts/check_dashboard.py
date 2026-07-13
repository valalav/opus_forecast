#!/usr/bin/env python3
"""
SIRENA-KBR Dashboard Health Check
=================================

Проверяет работоспособность всех компонентов dashboard.

Запуск: python3 scripts/check_dashboard.py
"""

import sys
import os
import time
import json
import subprocess
import socket
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import warnings
warnings.filterwarnings('ignore')


def check_port(host: str, port: int) -> bool:
    """Check if a port is open."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(2)
    try:
        result = sock.connect_ex((host, port))
        return result == 0
    finally:
        sock.close()


def check_dashboard_running() -> Tuple[bool, str]:
    """Check if dashboard is running on port 8503."""
    if check_port('localhost', 8503):
        return True, "Dashboard is running on http://localhost:8503"
    return False, "Dashboard is NOT running on port 8503"


def check_data_files() -> List[Dict]:
    """Check required data files."""
    data_dir = PROJECT_ROOT / 'data'
    required_files = [
        ('infl_kbr.csv', 'Monthly KBR inflation data'),
        ('inflation_data.csv', 'Extended inflation with macro'),
        ('weekly_prices.csv', 'Weekly prices (optional)'),
    ]

    results = []
    for filename, description in required_files:
        path = data_dir / filename
        if path.exists():
            size = path.stat().st_size
            results.append({
                'file': filename,
                'status': 'OK',
                'size': f"{size/1024:.1f} KB",
                'description': description
            })
        else:
            results.append({
                'file': filename,
                'status': 'MISSING',
                'size': '-',
                'description': description
            })

    return results


def check_models() -> List[Dict]:
    """Check all ensemble models can be imported."""
    results = []

    # Production models
    production_models = [
        ('sirena.models.ridge', 'RidgeForecaster'),
        ('sirena.models.ridge_extended', 'RidgeExtendedForecaster'),
        ('sirena.models.ridge_shock_dummies', 'RidgeShockDummiesForecaster'),
        ('sirena.models.huber', 'HuberForecaster'),
        ('sirena.models.elasticnet', 'ElasticNetForecaster'),
        ('sirena.models.prophet', 'ProphetForecaster'),
        ('sirena.models.ebm', 'EBMForecaster'),
    ]

    # NGBoost (optional)
    optional_models = [
        ('sirena.models.ngboost_model', 'NGBoostForecaster'),
        ('sirena.models.ngboost_shock', 'NGBoostShockForecaster'),
    ]

    # Auxiliary models
    auxiliary_models = [
        ('sirena.models.bvar', 'BVARForecaster'),
        ('sirena.models.ets', 'ETSForecaster'),
        ('sirena.models.arima', 'SARIMAForecaster'),
        ('sirena.models.lightgbm', 'LightGBMForecaster'),
    ]

    for module_path, class_name in production_models + optional_models + auxiliary_models:
        try:
            module = __import__(module_path, fromlist=[class_name])
            cls = getattr(module, class_name)
            results.append({
                'model': class_name,
                'status': 'OK',
                'type': 'production' if (module_path, class_name) in production_models else 'auxiliary'
            })
        except ImportError as e:
            results.append({
                'model': class_name,
                'status': 'OPTIONAL' if 'optional' in module_path else 'ERROR',
                'error': str(e),
                'type': 'optional'
            })
        except Exception as e:
            results.append({
                'model': class_name,
                'status': 'ERROR',
                'error': str(e),
                'type': 'unknown'
            })

    return results


def check_dashboard_status() -> Dict:
    """Check dashboard status from log file."""
    status_file = PROJECT_ROOT / 'logs' / 'dashboard_status.json'

    if not status_file.exists():
        return {'status': 'no_status_file'}

    try:
        with open(status_file) as f:
            return json.load(f)
    except Exception as e:
        return {'status': 'error', 'error': str(e)}


def run_quick_model_test() -> List[Dict]:
    """Run a quick test of key models."""
    results = []

    try:
        from sirena import DataLoader
        loader = DataLoader()
        df = loader.load_monthly_kbr()

        if df is None:
            return [{'model': 'DataLoader', 'status': 'ERROR', 'error': 'Failed to load data'}]

        import pandas as pd
        import numpy as np

        last_date = df.dropna(subset=['Все товары и услуги']).index.max()
        target_date = last_date + pd.DateOffset(months=1)

        # Test key models
        test_models = [
            ('sirena.models.ridge', 'RidgeForecaster'),
            ('sirena.models.huber', 'HuberForecaster'),
        ]

        for module_path, class_name in test_models:
            try:
                start = time.time()
                module = __import__(module_path, fromlist=[class_name])
                cls = getattr(module, class_name)
                model = cls()
                model.fit(df)

                df_ext = df.copy()
                df_ext.loc[target_date] = np.nan
                pred = model.predict(df_ext, target_date)

                duration = time.time() - start

                if isinstance(pred, dict):
                    prediction = pred.get('prediction', 0) - 100
                else:
                    prediction = float(pred) - 100

                results.append({
                    'model': class_name,
                    'status': 'OK',
                    'prediction': f"{prediction:.3f}%",
                    'duration': f"{duration:.2f}s"
                })

            except Exception as e:
                results.append({
                    'model': class_name,
                    'status': 'ERROR',
                    'error': str(e)
                })

    except Exception as e:
        results.append({
            'model': 'Test Setup',
            'status': 'ERROR',
            'error': str(e)
        })

    return results


def main():
    print("=" * 60)
    print("SIRENA-KBR v4.8 Dashboard Health Check")
    print("=" * 60)
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    all_ok = True

    # 1. Check dashboard running
    print("1. DASHBOARD STATUS")
    print("-" * 40)
    running, msg = check_dashboard_running()
    print(f"   {'✓' if running else '✗'} {msg}")
    if not running:
        all_ok = False
    print()

    # 2. Check data files
    print("2. DATA FILES")
    print("-" * 40)
    data_results = check_data_files()
    for r in data_results:
        icon = '✓' if r['status'] == 'OK' else '✗'
        print(f"   {icon} {r['file']}: {r['status']} ({r['size']})")
        if r['status'] == 'MISSING' and 'optional' not in r['description'].lower():
            all_ok = False
    print()

    # 3. Check model imports
    print("3. MODEL IMPORTS")
    print("-" * 40)
    model_results = check_models()
    for r in model_results:
        icon = '✓' if r['status'] == 'OK' else ('○' if r['status'] == 'OPTIONAL' else '✗')
        print(f"   {icon} {r['model']}: {r['status']}")
        if r['status'] == 'ERROR':
            all_ok = False
            print(f"      Error: {r.get('error', 'Unknown')[:50]}")
    print()

    # 4. Quick model test
    print("4. QUICK MODEL TEST")
    print("-" * 40)
    test_results = run_quick_model_test()
    for r in test_results:
        icon = '✓' if r['status'] == 'OK' else '✗'
        if r['status'] == 'OK':
            print(f"   {icon} {r['model']}: {r['prediction']} ({r['duration']})")
        else:
            print(f"   {icon} {r['model']}: {r['status']} - {r.get('error', '')[:40]}")
            all_ok = False
    print()

    # 5. Dashboard status from logs
    print("5. DASHBOARD LOGS STATUS")
    print("-" * 40)
    status = check_dashboard_status()
    if status.get('status') == 'no_status_file':
        print("   ○ No status file (dashboard hasn't written logs yet)")
    elif 'tabs' in status:
        tabs = status.get('tabs', {})
        for tab_name, tab_info in tabs.items():
            icon = '✓' if tab_info.get('status') == 'ok' else '✗'
            print(f"   {icon} {tab_name}: {tab_info.get('status', 'unknown')}")
        errors = status.get('errors', [])
        if errors:
            print(f"\n   Recent errors ({len(errors)}):")
            for err in errors[-3:]:
                print(f"      - {err.get('context', 'Unknown')}: {err.get('error_message', '')[:40]}")
    print()

    # Summary
    print("=" * 60)
    if all_ok:
        print("✓ ALL CHECKS PASSED")
    else:
        print("✗ SOME CHECKS FAILED - Review issues above")
    print("=" * 60)

    # Save results
    results_file = PROJECT_ROOT / 'logs' / 'health_check.json'
    results_file.parent.mkdir(exist_ok=True)

    with open(results_file, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'all_ok': all_ok,
            'dashboard_running': running,
            'data_files': data_results,
            'models': model_results,
            'quick_tests': test_results
        }, f, indent=2, ensure_ascii=False)

    print(f"\nResults saved to: {results_file}")

    return 0 if all_ok else 1


if __name__ == '__main__':
    sys.exit(main())
