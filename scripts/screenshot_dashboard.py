#!/usr/bin/env python3
"""
Screenshot Dashboard - визуальная верификация всех вкладок
==========================================================
Делает скриншоты всех вкладок dashboard для проверки отображения.

Использование:
    python3 scripts/screenshot_dashboard.py

Результаты:
    assets/screenshots/tab1_forecast.png
    assets/screenshots/tab3_backtest.png
    assets/screenshots/tab9_backtest_h1.png
    ...
"""

import asyncio
import os
from pathlib import Path

# Create screenshots directory
SCREENSHOTS_DIR = Path(__file__).parent.parent / 'assets' / 'screenshots'
SCREENSHOTS_DIR.mkdir(parents=True, exist_ok=True)

DASHBOARD_URL = "http://localhost:8503"

# Tab names as they appear in the current dashboard UI.
TABS = [
    ("tab_f1", "🎯 Прогноз h=1"),
    ("tab_f2", "🎯 Прогноз h=2"),
    ("tab_f3", "🎯 Прогноз h=3"),
    ("tab_f6", "🎯 Прогноз h=6"),
    ("tab_f12", "📈 Прогноз h=12"),
    ("tab_seasonality", "📊 Сезонность"),
    ("tab_macro", "🔍 Макро"),
    ("tab_scenarios", "🎚️ Сценарии Ki"),
    ("tab_exog", "📉 Экзогенные"),
    ("tab_weekly", "📈 Weekly"),
    ("tab_nowcast", "📊 Nowcast"),
    ("tab_compare", "🔍 Сравнение"),
    ("tab_b1", "📊 Бэктест h=1"),
    ("tab_b2", "📊 Бэктест h=2"),
    ("tab_b3", "📊 Бэктест h=3"),
    ("tab_b6", "📊 Бэктест h=6"),
    ("tab_b12", "📊 Бэктест h=12"),
]


def screenshot_name(tab_id, tab_name):
    """Build a stable screenshot file name for a dashboard tab."""
    cleaned = tab_name
    for token in ["📈", "🔧", "✅", "🛠", "🧠", "🗺", "🔒", "🔍", "📊", "🎯", "🎚️", "📉"]:
        cleaned = cleaned.replace(token, "")
    cleaned = cleaned.strip().replace(" ", "_")
    return f"{tab_id}_{cleaned}.png"


async def take_screenshots():
    from playwright.async_api import async_playwright

    print("=" * 60)
    print("SCREENSHOT DASHBOARD")
    print("=" * 60)

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page(viewport={'width': 1920, 'height': 1080})

        print(f"\nОткрываю {DASHBOARD_URL}...")
        try:
            await page.goto(DASHBOARD_URL, timeout=60000)
            await page.wait_for_load_state('networkidle', timeout=60000)
        except Exception as e:
            print(f"ERROR: Не удалось открыть dashboard: {e}")
            await browser.close()
            return False

        print("Dashboard загружен.\n")

        # Take screenshot of initial state
        initial_path = SCREENSHOTS_DIR / "00_initial.png"
        await page.screenshot(path=str(initial_path), full_page=True)
        print(f"[0] Initial state → {initial_path}")

        # Click each tab and take screenshot
        for i, (tab_id, tab_name) in enumerate(TABS, 1):
            print(f"\n[{i}/{len(TABS)}] {tab_name}...")

            try:
                # Find and click the tab
                tab_button = page.locator(f'button:has-text("{tab_name}")')

                if await tab_button.count() == 0:
                    print(f"  WARNING: Tab '{tab_name}' not found!")
                    continue

                await tab_button.first.click()

                # Wait for content to load
                await page.wait_for_timeout(3000)  # 3 seconds for content to render

                # Check if there's a spinner (loading indicator)
                spinner = page.locator('.stSpinner')
                if await spinner.count() > 0:
                    print(f"  Loading... (waiting up to 120s)")
                    try:
                        await spinner.wait_for(state='hidden', timeout=120000)
                    except:
                        print(f"  WARNING: Still loading after 120s")

                # Take screenshot
                screenshot_path = SCREENSHOTS_DIR / screenshot_name(tab_id, tab_name)
                await page.screenshot(path=str(screenshot_path), full_page=True)
                print(f"  OK → {screenshot_path.name}")

                # Check for error messages
                error_elements = await page.locator('.stAlert').all()
                for err in error_elements:
                    text = await err.text_content()
                    if text:
                        print(f"  ALERT: {text[:100]}...")

            except Exception as e:
                print(f"  ERROR: {e}")
                # Still try to take screenshot
                error_path = SCREENSHOTS_DIR / f"{tab_id}_ERROR.png"
                await page.screenshot(path=str(error_path))

        await browser.close()

    print("\n" + "=" * 60)
    print(f"Скриншоты сохранены в: {SCREENSHOTS_DIR}")
    print("=" * 60)

    # List all screenshots
    screenshots = sorted(SCREENSHOTS_DIR.glob("*.png"))
    print(f"\nВсего скриншотов: {len(screenshots)}")
    for s in screenshots:
        size_kb = s.stat().st_size / 1024
        print(f"  {s.name} ({size_kb:.1f} KB)")

    return True


if __name__ == '__main__':
    success = asyncio.run(take_screenshots())
    exit(0 if success else 1)
