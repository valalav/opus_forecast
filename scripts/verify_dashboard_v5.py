"""
Verification script for Dashboard v5.0
Checks all 10 tabs and takes screenshots
"""

import os
import sys
import pandas as pd

# Colors for terminal output
GREEN = '\033[92m'
RED = '\033[91m'
RESET = '\033[0m'

def check_mark(condition, message):
    if condition:
        print(f"{GREEN}[OK]{RESET} {message}")
        return True
    else:
        print(f"{RED}[FAIL]{RESET} {message}")
        return False


def verify_backtest_files():
    """Verify all backtest CSV files exist and have Micro."""
    print("\n=== 1. Backtest Files ===")
    all_ok = True

    for h in [1, 2, 3, 6, 12]:
        filepath = f'archive/results/backtest_h{h}_predictions.csv'
        exists = os.path.exists(filepath)

        if exists:
            df = pd.read_csv(filepath)
            has_micro = 'Micro' in df.columns
            has_actual = 'Actual' in df.columns
            num_models = len([c for c in df.columns if c not in ['Date', 'Actual']])

            ok = exists and has_micro and has_actual
            all_ok = all_ok and ok

            check_mark(ok, f"h={h}: {num_models} models, Micro={'yes' if has_micro else 'NO!'}")
        else:
            all_ok = False
            check_mark(False, f"h={h}: FILE NOT FOUND")

    return all_ok


def verify_dashboard_code():
    """Verify dashboard.py has correct structure."""
    print("\n=== 2. Dashboard Code ===")

    with open('dashboard.py', 'r') as f:
        content = f.read()

    checks = [
        ('ALL_MODELS' in content, "ALL_MODELS constant defined"),
        ('MODEL_COLORS' in content, "MODEL_COLORS constant defined"),
        ('tab_f1, tab_f2, tab_f3, tab_f6, tab_f12' in content, "5 forecast tabs defined"),
        ('tab_b1, tab_b2, tab_b3, tab_b6, tab_b12' in content, "5 backtest tabs defined"),
        ('render_forecast_tab' in content, "render_forecast_tab function"),
        ('render_backtest_tab' in content, "render_backtest_tab function"),
        ('Micro' in content, "Micro model referenced"),
    ]

    all_ok = True
    for condition, message in checks:
        ok = check_mark(condition, message)
        all_ok = all_ok and ok

    return all_ok


def verify_models_available():
    """Verify critical models can be imported."""
    print("\n=== 3. Model Imports ===")

    models_to_check = [
        ('sirena.models.huber', 'HuberForecaster'),
        ('sirena.models.ngboost_shock', 'NGBoostShockForecaster'),
        ('sirena.models.microcomponent', 'MicrocomponentForecaster'),
        ('sirena.models.prophet', 'ProphetForecaster'),
        ('sirena.models.ridge_extended', 'RidgeExtendedForecaster'),
    ]

    all_ok = True
    for module, classname in models_to_check:
        try:
            mod = __import__(module, fromlist=[classname])
            cls = getattr(mod, classname)
            check_mark(True, f"{classname}")
        except Exception as e:
            check_mark(False, f"{classname}: {e}")
            all_ok = False

    return all_ok


def take_screenshots():
    """Take screenshots of all 10 tabs using Playwright."""
    print("\n=== 4. Screenshots ===")

    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        print(f"{RED}Playwright not installed. Run: pip install playwright && playwright install{RESET}")
        return False

    tab_names = [
        "🎯 Прогноз h=1", "🎯 Прогноз h=2", "🎯 Прогноз h=3", "🎯 Прогноз h=6", "📈 Прогноз h=12",
        "📊 Бэктест h=1", "📊 Бэктест h=2", "📊 Бэктест h=3", "📊 Бэктест h=6", "📊 Бэктест h=12"
    ]

    os.makedirs('assets/screenshots', exist_ok=True)

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(viewport={'width': 1920, 'height': 1080})

        try:
            page.goto('http://localhost:8503', timeout=30000)
            page.wait_for_load_state('networkidle', timeout=30000)

            for i, tab_name in enumerate(tab_names):
                try:
                    # Find and click tab
                    tabs = page.locator('[data-baseweb="tab"]')
                    tabs.nth(i).click()
                    page.wait_for_timeout(2000)  # Wait for tab to load

                    # Take screenshot
                    filename = f"assets/screenshots/tab_{i+1:02d}_{tab_name.replace(' ', '_').replace('=', '')}.png"
                    page.screenshot(path=filename)
                    check_mark(True, f"Tab {i+1}: {tab_name}")

                except Exception as e:
                    check_mark(False, f"Tab {i+1}: {e}")

            browser.close()
            return True

        except Exception as e:
            browser.close()
            print(f"{RED}Could not connect to dashboard: {e}{RESET}")
            return False


def main():
    print("=" * 60)
    print("СИРЕНА-КБР v5.0 - Dashboard Verification")
    print("=" * 60)

    results = []

    results.append(("Backtest Files", verify_backtest_files()))
    results.append(("Dashboard Code", verify_dashboard_code()))
    results.append(("Model Imports", verify_models_available()))
    results.append(("Screenshots", take_screenshots()))

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    all_passed = True
    for name, passed in results:
        status = f"{GREEN}PASS{RESET}" if passed else f"{RED}FAIL{RESET}"
        print(f"  {name}: {status}")
        all_passed = all_passed and passed

    if all_passed:
        print(f"\n{GREEN}All checks passed!{RESET}")
    else:
        print(f"\n{RED}Some checks failed. Review above.{RESET}")

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
