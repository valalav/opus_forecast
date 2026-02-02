"""
Backtest h=12: Прогноз на 12 месяцев вперед (годовая траектория)

Фиксированный cutoff (ноябрь 2024):
- train на данных до ноября 2024
- прогноз траектории: декабрь 2024 → ноябрь 2025 (12 точек)
- сравнение каждой точки с фактом

Метрики: MAE по 12 точкам, стабильность траектории

Автор: Claude Code
Дата: 2025-12-25
"""

import sys
import os

# Add project root to path
sys.path.append(os.getcwd())

from backtest_framework import BacktestRunner


if __name__ == '__main__':
    # Create runner
    runner = BacktestRunner(
        horizon=12,
        test_months=1,  # ОСОБЕННОСТЬ: 1 окно, 12 дат в траектории
        output_dir='archive/results'
    )

    # Run backtest
    results = runner.run()

    # Calculate metrics
    metrics = runner.calculate_metrics(results)

    # Save results
    runner.save_results(results, metrics)

    # Print summary
    print("\n" + "="*70)
    print("BACKTEST h=12 ЗАВЕРШЕН")
    print("="*70)
    print(f"\nTop 3 модели:")
    for i, row in metrics.head(3).iterrows():
        print(f"  {i+1}. {row['Model']}: MAE {row['MAE']:.3f}")

    print(f"\nРезультаты сохранены в archive/results/")
    print(f"  - backtest_h12_predictions.csv")
    print(f"  - backtest_h12_metrics.csv")
    print(f"  - backtest_h12_summary.md\n")
    
    # Автоматическая синхронизация в sync/
    print("🔄 Синхронизация результатов в sync/...")
    try:
        import subprocess
        result = subprocess.run(
            ['python3', 'scripts/sync_to_share.py'],
            capture_output=True,
            text=True,
            timeout=60
        )
        if result.returncode == 0:
            print("✅ Результаты синхронизированы в sync/")
        else:
            print(f"⚠️ Ошибка синхронизации: {result.stderr[:100]}")
    except Exception as e:
        print(f"⚠️ Не удалось синхронизировать: {e}")
        print("   Запустите вручную: python3 scripts/sync_to_share.py")
