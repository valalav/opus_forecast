"""
Скрипт синхронизации результатов в папку sync/
=============================================

Копирует актуальные результаты, графики и отчёты из:
- archive/results/ → sync/csv/ и sync/reports/
- assets/charts/ → sync/charts/ и sync/html/
- experiments/*/results/ → sync/experiments/

Запуск:
    python3 scripts/sync_to_share.py

Автор: Claude Code
Дата: 2026-02-02
"""

import os
import shutil
import glob
from datetime import datetime
from pathlib import Path


# Корень проекта
PROJECT_ROOT = Path(__file__).parent.parent
SYNC_DIR = PROJECT_ROOT / "sync"

# Источники и назначения
SYNC_MAP = {
    # CSV-файлы с результатами
    "csv": {
        "sources": [
            PROJECT_ROOT / "archive" / "results" / "backtest_h1_predictions.csv",
            PROJECT_ROOT / "archive" / "results" / "backtest_h1_metrics.csv",
            PROJECT_ROOT / "archive" / "results" / "backtest_h1_summary.md",
            PROJECT_ROOT / "archive" / "results" / "backtest_h2_predictions.csv",
            PROJECT_ROOT / "archive" / "results" / "backtest_h2_metrics.csv",
            PROJECT_ROOT / "archive" / "results" / "backtest_h2_summary.md",
            PROJECT_ROOT / "archive" / "results" / "backtest_h12_predictions.csv",
            PROJECT_ROOT / "archive" / "results" / "backtest_h12_metrics.csv",
            PROJECT_ROOT / "archive" / "results" / "backtest_h12_summary.md",
            PROJECT_ROOT / "archive" / "results" / "model_comparison.csv",
            PROJECT_ROOT / "archive" / "results" / "forecasts_current.csv",
        ],
        "dest": SYNC_DIR / "csv"
    },
    
    # PNG графики
    "charts": {
        "sources": [
            PROJECT_ROOT / "assets" / "charts" / "*.png",
        ],
        "dest": SYNC_DIR / "charts"
    },
    
    # HTML визуализации
    "html": {
        "sources": [
            PROJECT_ROOT / "assets" / "charts" / "*.html",
        ],
        "dest": SYNC_DIR / "html"
    },
    
    # Отчёты
    "reports": {
        "sources": [
            PROJECT_ROOT / "archive" / "results" / "*.md",
        ],
        "dest": SYNC_DIR / "reports"
    }
}


def copy_file(src, dst):
    """Копирование файла с созданием директорий."""
    try:
        if not src.exists():
            return False, f"File not found: {src}"
        
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        return True, f"Copied: {src.name}"
    except Exception as e:
        return False, f"Error copying {src}: {e}"


def sync_files():
    """Основная функция синхронизации."""
    print("=" * 70)
    print("СИНХРОНИЗАЦИЯ РЕЗУЛЬТАТОВ В sync/")
    print("=" * 70)
    print(f"Время: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    stats = {"copied": 0, "skipped": 0, "errors": 0}
    
    # 1. Синхронизация CSV, отчётов
    for category in ["csv", "reports"]:
        config = SYNC_MAP[category]
        dest_dir = config["dest"]
        
        print(f"📁 {category.upper()}/")
        
        for src_path in config["sources"]:
            if "*" in str(src_path):
                # Glob pattern
                files = glob.glob(str(src_path))
                for file in files:
                    src = Path(file)
                    dst = dest_dir / src.name
                    success, msg = copy_file(src, dst)
                    
                    if success:
                        print(f"  ✓ {msg}")
                        stats["copied"] += 1
                    else:
                        print(f"  ⚠ {msg}")
                        stats["errors"] += 1
            else:
                # Single file
                dst = dest_dir / src_path.name
                success, msg = copy_file(src_path, dst)
                
                if success:
                    print(f"  ✓ {msg}")
                    stats["copied"] += 1
                elif "not found" in msg.lower():
                    print(f"  ⚠ {msg}")
                    stats["skipped"] += 1
                else:
                    print(f"  ✗ {msg}")
                    stats["errors"] += 1
        
        print()
    
    # 2. Синхронизация графиков (PNG)
    print("📁 CHARTS/ (PNG)")
    charts_dir = PROJECT_ROOT / "assets" / "charts"
    if charts_dir.exists():
        png_files = list(charts_dir.glob("*.png"))
        for src in png_files:
            dst = SYNC_MAP["charts"]["dest"] / src.name
            success, msg = copy_file(src, dst)
            if success:
                print(f"  ✓ {msg}")
                stats["copied"] += 1
            else:
                print(f"  ⚠ {msg}")
                stats["errors"] += 1
    else:
        print(f"  ⚠ Directory not found: {charts_dir}")
    print()
    
    # 3. Синхронизация HTML
    print("📁 HTML/")
    html_dir = PROJECT_ROOT / "assets" / "charts"
    if html_dir.exists():
        html_files = list(html_dir.glob("*.html"))
        for src in html_files:
            dst = SYNC_MAP["html"]["dest"] / src.name
            success, msg = copy_file(src, dst)
            if success:
                print(f"  ✓ {msg}")
                stats["copied"] += 1
            else:
                print(f"  ⚠ {msg}")
                stats["errors"] += 1
    else:
        print(f"  ⚠ Directory not found: {html_dir}")
    print()
    
    # 4. Синхронизация экспериментов
    print("📁 EXPERIMENTS/")
    experiments_dir = PROJECT_ROOT / "experiments"
    if experiments_dir.exists():
        for exp_dir in experiments_dir.iterdir():
            if exp_dir.is_dir():
                exp_results = exp_dir / "results"
                if exp_results.exists():
                    exp_sync_dir = SYNC_DIR / "experiments" / exp_dir.name
                    exp_sync_dir.mkdir(parents=True, exist_ok=True)
                    
                    # Копируем все файлы из results/
                    for src in exp_results.iterdir():
                        if src.is_file():
                            dst = exp_sync_dir / src.name
                            success, msg = copy_file(src, dst)
                            if success:
                                print(f"  ✓ {exp_dir.name}/{msg}")
                                stats["copied"] += 1
                            else:
                                print(f"  ⚠ {exp_dir.name}/{msg}")
                                stats["errors"] += 1
    print()
    
    # Итоги
    print("=" * 70)
    print("ИТОГИ")
    print("=" * 70)
    print(f"✓ Скопировано: {stats['copied']} файлов")
    print(f"⚠ Пропущено: {stats['skipped']} файлов")
    print(f"✗ Ошибок: {stats['errors']}")
    print()
    print("Папка sync/ готова к синхронизации через Syncthing!")
    print()
    
    # Список файлов в sync/
    print("Содержимое sync/:")
    for root, dirs, files in os.walk(SYNC_DIR):
        level = root.replace(str(SYNC_DIR), '').count(os.sep)
        indent = '  ' * level
        print(f"{indent}{os.path.basename(root)}/")
        sub_indent = '  ' * (level + 1)
        for file in files[:5]:  # Показываем первые 5 файлов
            print(f"{sub_indent}{file}")
        if len(files) > 5:
            print(f"{sub_indent}... и ещё {len(files) - 5} файлов")
    
    return stats


def main():
    """Главная функция."""
    # Создаём папки
    for subdir in ["charts", "csv", "html", "experiments", "reports"]:
        (SYNC_DIR / subdir).mkdir(parents=True, exist_ok=True)
    
    # Запускаем синхронизацию
    stats = sync_files()
    
    return stats["errors"] == 0


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
