"""
Backtest h=1: Прогноз на 1 месяц вперед (САМЫЙ ВАЖНЫЙ КПЭ)

Rolling window за последние 12 месяцев:
- Октябрь 2025: train до сентября → прогноз на октябрь
- Сентябрь 2025: train до августа → прогноз на сентябрь
- ... и так далее

Метрики: MAE, KPI violations (|error| > 0.5), Coverage 50%

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
        horizon=1,
        test_months=12,
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
    print("BACKTEST h=1 ЗАВЕРШЕН")
    print("="*70)
    print(f"\nTop 3 модели:")
    for i, row in metrics.head(3).iterrows():
        print(f"  {i+1}. {row['Model']}: MAE {row['MAE']:.3f}")

    print(f"\nРезультаты сохранены в archive/results/")
    print(f"  - backtest_h1_predictions.csv")
    print(f"  - backtest_h1_metrics.csv")
    print(f"  - backtest_h1_summary.md\n")
    
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
