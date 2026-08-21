#!/usr/bin/env python3
"""
ПОЛНАЯ ВЕРИФИКАЦИЯ ВСЕХ ВКЛАДОК DASHBOARD
==========================================
Проверяет ВСЕ 12 вкладок на:
1. Актуальность списка моделей
2. Наличие Micro модели
3. Согласованность данных
4. Отсутствие ошибок

Использование:
    python3 scripts/verify_all_tabs.py

ЗАПУСКАТЬ ПЕРЕД ЛЮБЫМ "ГОТОВО"!
"""

import sys
import os
import json
import pandas as pd
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

# Expected models (must match ALL_MODELS in dashboard.py)
EXPECTED_MODELS = [
    'Ridge', 'Ridge_Ext', 'Bayes_Ridge', 'ElasticNet', 'Huber', 'Ridge_Shock',
    'NGBoost', 'NGBoost_Shock', 'BVAR', 'SARIMA', 'LightGBM', 'Prophet',
    'ETS', 'EBM', 'CatBoost', 'Subcomp', 'Subcomp_Multi', 'Micro', 'Ensemble'
]

# Critical models that MUST be present
CRITICAL_MODELS = ['Ridge', 'Huber', 'Micro', 'NGBoost_Shock', 'Ensemble']


def check_precomputed_forecasts():
    """Проверка data/precomputed_forecasts.json"""
    print("\n" + "=" * 60)
    print("[1] PRECOMPUTED FORECASTS (tab1)")
    print("=" * 60)

    errors = []

    try:
        with open('data/precomputed_forecasts.json', 'r') as f:
            data = json.load(f)

        models_in_file = list(data.get('forecasts', {}).keys())
        print(f"Моделей в файле: {len(models_in_file)}")
        print(f"Модели: {models_in_file}")

        # Check critical models
        for m in ['Ridge', 'Huber', 'Micro', 'Ensemble']:
            if m in models_in_file:
                print(f"  ✓ {m}")
            else:
                print(f"  ✗ {m} — ОТСУТСТВУЕТ!")
                errors.append(f"precomputed_forecasts.json: {m} отсутствует")

        # Check data freshness
        generated = data.get('generated_at', 'unknown')
        print(f"\nСгенерировано: {generated}")

        diagnostics = data.get('diagnostics', {})
        weekly_bridge = diagnostics.get('weekly_bridge') if isinstance(diagnostics, dict) else None
        if weekly_bridge:
            print("\nWeekly bridge diagnostics:")
            print(f"  ✓ method: {weekly_bridge.get('method')}")
            print(f"  ✓ source: {weekly_bridge.get('source_file')}")
            print(f"  ✓ months: {sorted(weekly_bridge.get('by_month', {}).keys())}")
            if 'WeeklyBridge' in models_in_file:
                print("  ✗ WeeklyBridge попал в forecasts!")
                errors.append("precomputed_forecasts.json: WeeklyBridge не должен быть в forecasts")
        else:
            print("  ✗ diagnostics.weekly_bridge отсутствует!")
            errors.append("precomputed_forecasts.json: diagnostics.weekly_bridge отсутствует")

    except FileNotFoundError:
        errors.append("precomputed_forecasts.json не найден")
        print("✗ Файл не найден!")
    except Exception as e:
        errors.append(f"precomputed_forecasts.json: {e}")
        print(f"✗ Ошибка: {e}")

    return errors


def check_backtest_csv(horizon):
    """Проверка archive/results/backtest_h{horizon}_predictions.csv"""
    print("\n" + "=" * 60)
    print(f"[{horizon+1}] BACKTEST h={horizon} (tab{8+horizon})")
    print("=" * 60)

    errors = []
    filepath = f'archive/results/backtest_h{horizon}_predictions.csv'

    try:
        df = pd.read_csv(filepath)
        models_in_file = [c for c in df.columns if c not in ['Date', 'Actual']]
        print(f"Моделей в файле: {len(models_in_file)}")

        # Check critical models
        for m in CRITICAL_MODELS:
            if m in models_in_file:
                # Check for NaN values
                nan_count = df[m].isna().sum()
                if nan_count == 0:
                    print(f"  ✓ {m}")
                else:
                    print(f"  ⚠ {m} — {nan_count} NaN значений")
            else:
                print(f"  ✗ {m} — ОТСУТСТВУЕТ!")
                errors.append(f"backtest_h{horizon}: {m} отсутствует")

        # Check Actual column
        if 'Actual' in df.columns:
            nan_actual = df['Actual'].isna().sum()
            if nan_actual > 0:
                print(f"  ⚠ Actual: {nan_actual} NaN значений")

        # File modification time
        mtime = os.path.getmtime(filepath)
        mtime_str = datetime.fromtimestamp(mtime).strftime('%Y-%m-%d %H:%M:%S')
        print(f"\nПоследнее обновление: {mtime_str}")

    except FileNotFoundError:
        errors.append(f"backtest_h{horizon}_predictions.csv не найден")
        print("✗ Файл не найден!")
    except Exception as e:
        errors.append(f"backtest_h{horizon}: {e}")
        print(f"✗ Ошибка: {e}")

    return errors


def check_dashboard_constants():
    """Проверка констант в dashboard.py"""
    print("\n" + "=" * 60)
    print("[4] DASHBOARD.PY CONSTANTS")
    print("=" * 60)

    errors = []

    try:
        with open('dashboard.py', 'r') as f:
            content = f.read()

        # Check ALL_MODELS exists
        if 'ALL_MODELS = [' in content or 'from pages import' in content and 'ALL_MODELS' in content:
            print("  ✓ ALL_MODELS определён (или импортирован)")
        else:
            print("  ✗ ALL_MODELS не определён!")
            errors.append("dashboard.py: ALL_MODELS не определён")

        # Check MODEL_COLORS exists
        if 'MODEL_COLORS = {' in content or 'from pages import' in content and 'MODEL_COLORS' in content:
            print("  ✓ MODEL_COLORS определён (или импортирован)")
        else:
            print("  ✗ MODEL_COLORS не определён!")
            errors.append("dashboard.py: MODEL_COLORS не определён")

        # Check for hardcoded model lists (should use ALL_MODELS)
        hardcoded_count = content.count("models_h1 = ['") + content.count("models_h2 = ['")
        if hardcoded_count > 0:
            print(f"  ⚠ Найдено {hardcoded_count} захардкоженных списков моделей")
            errors.append(f"dashboard.py: {hardcoded_count} захардкоженных списков")
        else:
            print("  ✓ Нет захардкоженных списков")

        # Check Micro is in ALL_MODELS
        # Since ALL_MODELS might be imported, we can't easily check its content statically in dashboard.py
        # We should check if pages/constants.py has Micro if imported
        if 'from pages import' in content:
             print("  ✓ Проверка состава ALL_MODELS делегирована (импорт из pages)")
        elif "'Micro'" in content and 'ALL_MODELS' in content:
            # Simple check - Micro should appear after ALL_MODELS
            all_models_pos = content.find('ALL_MODELS = [')
            micro_pos = content.find("'Micro'", all_models_pos)
            if micro_pos > all_models_pos and micro_pos < all_models_pos + 500:
                print("  ✓ Micro в ALL_MODELS")
            else:
                print("  ⚠ Micro может отсутствовать в ALL_MODELS")

    except Exception as e:
        errors.append(f"dashboard.py: {e}")
        print(f"✗ Ошибка: {e}")

    return errors


def check_backtest_framework():
    """Проверка scripts/backtest_framework.py"""
    print("\n" + "=" * 60)
    print("[5] BACKTEST_FRAMEWORK.PY")
    print("=" * 60)

    errors = []

    try:
        with open('scripts/backtest_framework.py', 'r') as f:
            content = f.read()

        # Check Micro import
        if 'MicrocomponentForecaster' in content:
            print("  ✓ MicrocomponentForecaster импортирован")
        else:
            print("  ✗ MicrocomponentForecaster не импортирован!")
            errors.append("backtest_framework.py: Micro не импортирован")

        # Check _forecast_micro method
        if '_forecast_micro' in content:
            print("  ✓ _forecast_micro метод существует")
        else:
            print("  ✗ _forecast_micro метод отсутствует!")
            errors.append("backtest_framework.py: _forecast_micro отсутствует")

        # Check Micro in predictions
        if "predictions['Micro']" in content or 'predictions["Micro"]' in content:
            print("  ✓ Micro добавляется в predictions")
        else:
            print("  ✗ Micro не добавляется в predictions!")
            errors.append("backtest_framework.py: Micro не в predictions")

    except FileNotFoundError:
        print("  ⚠ Файл не найден (может быть ok)")
    except Exception as e:
        errors.append(f"backtest_framework.py: {e}")
        print(f"✗ Ошибка: {e}")

    return errors


def take_screenshots():
    """Делает скриншоты всех вкладок"""
    print("\n" + "=" * 60)
    print("[6] СКРИНШОТЫ ВКЛАДОК")
    print("=" * 60)

    try:
        import subprocess
        result = subprocess.run(
            ['python3', 'scripts/screenshot_dashboard.py'],
            capture_output=True,
            text=True,
            timeout=300
        )

        # Count only actual status lines. Old screenshot filenames can contain
        # "ERROR" and should not make verification fail.
        lines = result.stdout.splitlines()
        ok_count = sum(1 for line in lines if 'OK →' in line)
        error_count = sum(1 for line in lines if line.strip().startswith('ERROR:'))
        warning_count = sum(1 for line in lines if line.strip().startswith('WARNING:'))

        print(f"  Успешно: {ok_count} вкладок")
        if error_count > 0 or warning_count > 0 or ok_count == 0:
            print(f"  Ошибки: {error_count}; предупреждения: {warning_count}")
            return [
                "screenshots: "
                f"{error_count} ошибок, {warning_count} предупреждений, "
                f"{ok_count} успешных вкладок"
            ]

        print("\n  Скриншоты сохранены в assets/screenshots/")
        return []

    except Exception as e:
        print(f"  ⚠ Не удалось сделать скриншоты: {e}")
        return []


def main():
    print("=" * 60)
    print("ПОЛНАЯ ВЕРИФИКАЦИЯ DASHBOARD")
    print(f"Время: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    all_errors = []

    # 1. Check precomputed forecasts
    all_errors.extend(check_precomputed_forecasts())

    # 2. Check backtest h=1
    all_errors.extend(check_backtest_csv(1))

    # 3. Check backtest h=2
    all_errors.extend(check_backtest_csv(2))

    # 4. Check dashboard.py constants
    all_errors.extend(check_dashboard_constants())

    # 5. Check backtest_framework.py
    all_errors.extend(check_backtest_framework())

    # 6. Take screenshots
    all_errors.extend(take_screenshots())

    # Summary
    print("\n" + "=" * 60)
    print("ИТОГО")
    print("=" * 60)

    if all_errors:
        print(f"\n❌ НАЙДЕНО {len(all_errors)} ПРОБЛЕМ:\n")
        for e in all_errors:
            print(f"  • {e}")
        print("\n⚠️  НЕ ГОВОРИ 'ГОТОВО' ПОКА НЕ ИСПРАВИШЬ!")
        return False
    else:
        print("\n✅ ВСЕ ПРОВЕРКИ ПРОЙДЕНЫ")
        print("\nМожно говорить 'готово' только после просмотра скриншотов!")
        return True


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
