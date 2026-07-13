#!/usr/bin/env python3
"""
СТАТУС ВСЕХ МОДЕЛЕЙ СИСТЕМЫ СИРЕНА-КБР
======================================

Показывает:
- Все модели в системе
- Метрики по всем горизонтам бэктеста (h=1, h=2, h=12)
- Ранжирование моделей
- Текущие прогнозы

Запуск:
    python3 scripts/models_status.py           # Полный отчёт
    python3 scripts/models_status.py --list    # Только список моделей
    python3 scripts/models_status.py --metrics # Только метрики
    python3 scripts/models_status.py --rank    # Ранжирование по всем горизонтам
    python3 scripts/models_status.py --forecast # Прогнозы моделей

Автор: Claude Code
"""

import sys
import os
import re
from pathlib import Path
import pandas as pd
import numpy as np

# Цвета для вывода
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
BOLD = '\033[1m'
RESET = '\033[0m'


def get_all_models():
    """Получить список всех моделей из dashboard.py"""
    dashboard_file = Path('dashboard.py')
    if not dashboard_file.exists():
        return []

    content = dashboard_file.read_text()
    match = re.search(r'ALL_MODELS\s*=\s*\[(.*?)\]', content, re.DOTALL)
    if match:
        models_str = match.group(1)
        models = re.findall(r"'(\w+)'", models_str)
        return models
    return []


def get_model_colors():
    """Получить цвета моделей из dashboard.py"""
    dashboard_file = Path('dashboard.py')
    if not dashboard_file.exists():
        return {}

    content = dashboard_file.read_text()
    match = re.search(r'MODEL_COLORS\s*=\s*\{(.*?)\}', content, re.DOTALL)
    if match:
        colors_str = match.group(1)
        colors = dict(re.findall(r"'(\w+)':\s*'([^']+)'", colors_str))
        return colors
    return {}


def load_backtest_metrics():
    """Загрузить метрики бэктестов для всех горизонтов"""
    metrics = {}
    for h in [1, 2, 12]:
        csv_file = Path(f'archive/results/backtest_h{h}_metrics.csv')
        if csv_file.exists():
            df = pd.read_csv(csv_file)
            metrics[h] = df
    return metrics


def load_backtest_predictions():
    """Загрузить прогнозы бэктестов"""
    predictions = {}
    for h in [1, 2, 12]:
        csv_file = Path(f'archive/results/backtest_h{h}_predictions.csv')
        if csv_file.exists():
            df = pd.read_csv(csv_file)
            predictions[h] = df
    return predictions


def print_models_list():
    """Вывести список всех моделей"""
    print(f"\n{BOLD}{'='*60}{RESET}")
    print(f"{BOLD}МОДЕЛИ В СИСТЕМЕ СИРЕНА-КБР{RESET}")
    print(f"{'='*60}\n")

    models = get_all_models()
    colors = get_model_colors()

    print(f"Всего моделей: {BLUE}{len(models)}{RESET}\n")

    for i, model in enumerate(models, 1):
        color = colors.get(model, '#000000')
        print(f"  {i:2}. {model:<20} цвет: {color}")

    # Проверить файлы моделей
    print(f"\n{BOLD}Файлы моделей:{RESET}")
    models_dir = Path('sirena/models')
    if models_dir.exists():
        py_files = sorted(models_dir.glob('*.py'))
        for f in py_files:
            if f.name not in ['__init__.py', 'base.py', 'registry.py']:
                print(f"  - sirena/models/{f.name}")


def print_metrics():
    """Вывести метрики всех моделей"""
    print(f"\n{BOLD}{'='*60}{RESET}")
    print(f"{BOLD}МЕТРИКИ БЭКТЕСТОВ{RESET}")
    print(f"{'='*60}\n")

    metrics = load_backtest_metrics()

    for h, df in metrics.items():
        print(f"\n{YELLOW}--- Горизонт h={h} ---{RESET}\n")
        print(f"{'Модель':<20} {'MAE':>8} {'KPI Viol':>10} {'Coverage':>10}")
        print("-" * 50)

        for _, row in df.iterrows():
            model = row['Model']
            mae = row['MAE']
            kpi = int(row['KPI_Violations'])
            cov = row['Coverage_50pct']
            print(f"{model:<20} {mae:>8.3f} {kpi:>10} {cov:>9.1f}%")


def print_ranking():
    """Вывести ранжирование моделей по всем горизонтам"""
    print(f"\n{BOLD}{'='*70}{RESET}")
    print(f"{BOLD}РАНЖИРОВАНИЕ МОДЕЛЕЙ ПО ВСЕМ ГОРИЗОНТАМ{RESET}")
    print(f"{'='*70}\n")

    metrics = load_backtest_metrics()

    # Собрать все модели и их ранги
    all_models = set()
    ranks = {}

    for h, df in metrics.items():
        df_sorted = df.sort_values('MAE')
        for rank, (_, row) in enumerate(df_sorted.iterrows(), 1):
            model = row['Model']
            all_models.add(model)
            if model not in ranks:
                ranks[model] = {}
            ranks[model][f'h{h}_mae'] = row['MAE']
            ranks[model][f'h{h}_rank'] = rank

    # Вычислить средний ранг
    for model in ranks:
        model_ranks = [ranks[model].get(f'h{h}_rank', 999) for h in [1, 2, 12]]
        ranks[model]['avg_rank'] = np.mean(model_ranks)

    # Сортировать по среднему рангу
    sorted_models = sorted(ranks.keys(), key=lambda m: ranks[m]['avg_rank'])

    # Вывести таблицу
    print(f"{'Модель':<20} {'h=1':>12} {'h=2':>12} {'h=12':>12} {'Ср.ранг':>10}")
    print(f"{'':20} {'MAE(#)':>12} {'MAE(#)':>12} {'MAE(#)':>12}")
    print("-" * 70)

    for model in sorted_models:
        r = ranks[model]
        h1 = f"{r.get('h1_mae', 0):.3f}(#{r.get('h1_rank', '-')})" if 'h1_mae' in r else '-'
        h2 = f"{r.get('h2_mae', 0):.3f}(#{r.get('h2_rank', '-')})" if 'h2_mae' in r else '-'
        h12 = f"{r.get('h12_mae', 0):.3f}(#{r.get('h12_rank', '-')})" if 'h12_mae' in r else '-'
        avg = r['avg_rank']

        # Выделить топ-3
        color = GREEN if avg <= 3 else (YELLOW if avg <= 6 else RESET)
        print(f"{color}{model:<20} {h1:>12} {h2:>12} {h12:>12} {avg:>10.1f}{RESET}")


def print_forecasts():
    """Вывести последние прогнозы моделей"""
    print(f"\n{BOLD}{'='*60}{RESET}")
    print(f"{BOLD}ПОСЛЕДНИЕ ПРОГНОЗЫ МОДЕЛЕЙ{RESET}")
    print(f"{'='*60}\n")

    predictions = load_backtest_predictions()

    if not predictions:
        print(f"{RED}Нет данных о прогнозах{RESET}")
        return

    # Взять последний прогноз из h=1
    if 1 in predictions:
        df = predictions[1]
        last_row = df.iloc[-1]
        date = last_row.get('Date', 'N/A')
        actual = last_row.get('Actual', np.nan)

        print(f"Последняя дата: {BLUE}{date}{RESET}")
        print(f"Факт: {BLUE}{actual:.3f}{RESET}\n")

        print(f"{'Модель':<20} {'Прогноз':>10} {'Ошибка':>10}")
        print("-" * 42)

        models = get_all_models()
        for model in models:
            if model in last_row.index:
                pred = last_row[model]
                if pd.notna(pred) and pd.notna(actual):
                    error = pred - actual
                    color = GREEN if abs(error) <= 0.5 else RED
                    print(f"{model:<20} {pred:>10.3f} {color}{error:>+10.3f}{RESET}")
                elif pd.notna(pred):
                    print(f"{model:<20} {pred:>10.3f} {'N/A':>10}")


def print_full_report():
    """Полный отчёт"""
    print_models_list()
    print_ranking()
    print_forecasts()


def check_model(model_name: str):
    """Проверить что модель добавлена везде (динамически)"""
    print(f"\n{BOLD}{'='*60}{RESET}")
    print(f"{BOLD}ПРОВЕРКА МОДЕЛИ: {model_name}{RESET}")
    print(f"{'='*60}\n")

    # Нормализация имени
    file_name = model_name.lower()
    class_name = ''.join(word.capitalize() for word in model_name.split('_')) + 'Forecaster'

    checks = []

    # 1. Файл модели
    model_file = Path(f'sirena/models/{file_name}.py')
    checks.append(('Файл модели', f'sirena/models/{file_name}.py', model_file.exists()))

    # 2. Импорт в __init__.py
    init_file = Path('sirena/models/__init__.py')
    if init_file.exists():
        content = init_file.read_text()
        found = f'from .{file_name} import' in content or class_name in content
        checks.append(('Импорт в __init__.py', 'sirena/models/__init__.py', found))

    # 3-4. Dashboard
    dashboard_file = Path('dashboard.py')
    if dashboard_file.exists():
        content = dashboard_file.read_text()
        checks.append(('ALL_MODELS', 'dashboard.py', f"'{model_name}'" in content))
        checks.append(('MODEL_COLORS', 'dashboard.py', f"'{model_name}':" in content))

    # 5-8. Backtest framework
    backtest_file = Path('scripts/backtest_framework.py')
    if backtest_file.exists():
        content = backtest_file.read_text()
        checks.append(('Backtest импорт', 'scripts/backtest_framework.py',
                      f'from sirena.models.{file_name} import' in content or class_name in content))
        checks.append(('Backtest _forecast_', 'scripts/backtest_framework.py',
                      f'def _forecast_{file_name}' in content))
        checks.append(('Backtest _run_rolling', 'scripts/backtest_framework.py',
                      f"predictions['{model_name}']" in content))
        checks.append(('Backtest _run_h12', 'scripts/backtest_framework.py',
                      f"{file_name}_model" in content or f"predictions['{model_name}']" in content))

    # 9-11. CSV файлы
    for h in [1, 2, 12]:
        csv_file = Path(f'archive/results/backtest_h{h}_predictions.csv')
        if csv_file.exists():
            with open(csv_file, 'r') as f:
                header = f.readline()
            checks.append((f'CSV h={h}', f'backtest_h{h}_predictions.csv', model_name in header))

    # Вывести результаты
    passed = 0
    for name, location, ok in checks:
        status = f"{GREEN}[OK]{RESET}" if ok else f"{RED}[MISSING]{RESET}"
        print(f"{status} {name}: {location}")
        if ok:
            passed += 1

    total = len(checks)
    print(f"\n{'='*60}")
    if passed == total:
        print(f"{GREEN}ВСЕ ПРОВЕРКИ ПРОЙДЕНЫ: {passed}/{total}{RESET}")
    else:
        print(f"{RED}ЕСТЬ ПРОБЛЕМЫ: {passed}/{total}{RESET}")
        print(f"\n{YELLOW}Для добавления модели используй: /add-model {model_name}{RESET}")

    return passed == total


if __name__ == '__main__':
    os.chdir(Path(__file__).parent.parent)  # Перейти в корень проекта

    if len(sys.argv) < 2:
        print_full_report()
    elif sys.argv[1] == '--list':
        print_models_list()
    elif sys.argv[1] == '--metrics':
        print_metrics()
    elif sys.argv[1] == '--rank':
        print_ranking()
    elif sys.argv[1] == '--forecast':
        print_forecasts()
    elif sys.argv[1] == '--check':
        if len(sys.argv) < 3:
            print("Использование: python3 scripts/models_status.py --check ModelName")
        else:
            check_model(sys.argv[2])
    else:
        # Предположить что это имя модели для проверки
        check_model(sys.argv[1])
