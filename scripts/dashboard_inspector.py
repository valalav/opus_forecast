#!/usr/bin/env python3
"""
SIRENA-KBR Dashboard Inspector
==============================

Запускает код dashboard.py и показывает:
1. Какие модели реально используются
2. Какие переменные определены
3. Какие ошибки возникают

Запуск: python3 scripts/dashboard_inspector.py
"""

import sys
import re
from pathlib import Path
from collections import defaultdict

PROJECT_ROOT = Path(__file__).parent.parent
DASHBOARD_FILE = PROJECT_ROOT / 'dashboard.py'


def extract_model_info():
    """Extract all model-related info from dashboard.py"""

    with open(DASHBOARD_FILE, 'r', encoding='utf-8') as f:
        content = f.read()
        lines = content.split('\n')

    results = {
        'models_in_weights': [],
        'models_on_chart': [],
        'model_imports': [],
        'model_df_vars': set(),
        'backtest_models': [],
        'undefined_refs': [],
        'lmmr_refs': [],
        'bvar_refs': [],
    }

    # 1. Find model_weights definition
    in_weights = False
    weights_content = []
    for i, line in enumerate(lines, 1):
        if 'model_weights = {' in line:
            in_weights = True
            continue
        if in_weights:
            if line.strip().startswith('}'):
                in_weights = False
            else:
                # Extract model name
                match = re.search(r"'([^']+)':\s*\(", line)
                if match:
                    results['models_in_weights'].append((match.group(1), i))

    # 2. Find add_trace calls (models on chart)
    for i, line in enumerate(lines, 1):
        if 'add_trace' in line and 'name=' in line:
            match = re.search(r"name=['\"]([^'\"]+)['\"]", line)
            if match:
                results['models_on_chart'].append((match.group(1), i))

    # 3. Find all _df variable definitions and uses
    df_defined = {}
    df_used = []

    for i, line in enumerate(lines, 1):
        # Definitions: xxx_df = or xxx_df, yyy = func()
        for match in re.finditer(r'\b(\w+_df)\s*=\s*(?!None)', line):
            var = match.group(1)
            if var not in df_defined:
                df_defined[var] = i

        # Tuple unpacking: xxx_df, yyy = func()
        match = re.match(r'\s*(\w+_df)\s*,\s*\w+\s*=', line)
        if match:
            var = match.group(1)
            if var not in df_defined:
                df_defined[var] = i

        # Uses in if statements
        for match in re.finditer(r'if\s+(\w+_df)\s+is\s+not\s+None', line):
            df_used.append((match.group(1), i))

    # Check for undefined
    for var, line_num in df_used:
        if var not in df_defined:
            results['undefined_refs'].append((var, line_num))

    results['model_df_vars'] = set(df_defined.keys())

    # 4. Find LMMR references
    for i, line in enumerate(lines, 1):
        if 'lmmr' in line.lower() and not line.strip().startswith('#'):
            results['lmmr_refs'].append((i, line.strip()[:70]))

    # 5. Find BVAR references
    for i, line in enumerate(lines, 1):
        if 'bvar' in line.lower() and not line.strip().startswith('#'):
            # Skip bvar_df_full which is data loading
            if 'bvar_df_full' not in line.lower():
                results['bvar_refs'].append((i, line.strip()[:70]))

    # 6. Find backtest models (all_models list)
    for i, line in enumerate(lines, 1):
        if 'all_models' in line and '=' in line and '[' in line:
            match = re.search(r'\[([^\]]+)\]', line)
            if match:
                models = [m.strip().strip("'\"") for m in match.group(1).split(',')]
                results['backtest_models'] = [(m, i) for m in models if m]

    return results


def print_report(results):
    """Print inspection report."""

    print("=" * 70)
    print("SIRENA-KBR Dashboard Inspector")
    print("=" * 70)
    print()

    # 1. Models in ensemble weights
    print("1. МОДЕЛИ В АНСАМБЛЕ (model_weights):")
    print("-" * 50)
    if results['models_in_weights']:
        for name, line in results['models_in_weights']:
            print(f"   ✓ {name} (line {line})")
        print(f"   ИТОГО: {len(results['models_in_weights'])} моделей")
    else:
        print("   ✗ Не найдено")
    print()

    # 2. Models on chart
    print("2. МОДЕЛИ НА ГРАФИКЕ (add_trace):")
    print("-" * 50)
    seen = set()
    for name, line in results['models_on_chart']:
        if name not in seen and 'Факт' not in name and 'Цель' not in name:
            print(f"   • {name} (line {line})")
            seen.add(name)
    print(f"   ИТОГО: {len(seen)} уникальных моделей на графике")
    print()

    # 3. LMMR references
    print("3. ССЫЛКИ НА LMMR:")
    print("-" * 50)
    if results['lmmr_refs']:
        print(f"   ⚠ НАЙДЕНО {len(results['lmmr_refs'])} ссылок!")
        for line_num, text in results['lmmr_refs'][:10]:
            print(f"   Line {line_num}: {text}")
        if len(results['lmmr_refs']) > 10:
            print(f"   ... и ещё {len(results['lmmr_refs']) - 10}")
    else:
        print("   ✓ LMMR полностью удалена")
    print()

    # 4. BVAR references
    print("4. ССЫЛКИ НА BVAR (в ансамбле/на графике):")
    print("-" * 50)
    if results['bvar_refs']:
        print(f"   ⚠ НАЙДЕНО {len(results['bvar_refs'])} ссылок!")
        for line_num, text in results['bvar_refs'][:10]:
            print(f"   Line {line_num}: {text}")
    else:
        print("   ✓ BVAR не используется")
    print()

    # 5. Undefined variables
    print("5. НЕОПРЕДЕЛЁННЫЕ ПЕРЕМЕННЫЕ:")
    print("-" * 50)
    if results['undefined_refs']:
        print(f"   ✗ НАЙДЕНО {len(results['undefined_refs'])} проблем!")
        for var, line in results['undefined_refs']:
            print(f"   Line {line}: {var}")
    else:
        print("   ✓ Все переменные определены")
    print()

    # 6. Backtest models
    print("6. МОДЕЛИ В БЭКТЕСТЕ (all_models):")
    print("-" * 50)
    if results['backtest_models']:
        for name, line in results['backtest_models']:
            print(f"   • {name}")
        print(f"   ИТОГО: {len(results['backtest_models'])} моделей")
    else:
        print("   ✗ Не найдено")
    print()

    # Summary
    print("=" * 70)
    issues = []
    if results['lmmr_refs']:
        issues.append(f"LMMR всё ещё используется ({len(results['lmmr_refs'])} мест)")
    if results['undefined_refs']:
        issues.append(f"Неопределённые переменные ({len(results['undefined_refs'])})")
    if len(results['bvar_refs']) > 5:
        issues.append(f"BVAR активно используется ({len(results['bvar_refs'])} мест)")

    if issues:
        print("✗ ПРОБЛЕМЫ:")
        for issue in issues:
            print(f"   • {issue}")
    else:
        print("✓ ВСЁ В ПОРЯДКЕ")
    print("=" * 70)

    return len(issues) == 0


def main():
    results = extract_model_info()
    ok = print_report(results)
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
