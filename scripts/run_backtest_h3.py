"""
Backtest h=3: Прогноз на 3 месяца вперед

Rolling window за последние 12 месяцев:
- Ноябрь 2025: train до августа → прогноз [сен, окт, ноя] → берем ноябрь
- Октябрь 2025: train до июля → прогноз [авг, сен, окт] → берем октябрь
- ... и так далее

Метрики: MAE, KPI violations (|error| > 0.5), Coverage 50%
"""

import sys
import os

sys.path.append(os.getcwd())

from backtest_framework import BacktestRunner


if __name__ == '__main__':
    runner = BacktestRunner(
        horizon=3,
        test_months=12,
        output_dir='archive/results'
    )

    results = runner.run()
    metrics = runner.calculate_metrics(results)
    runner.save_results(results, metrics)

    print("\n" + "="*70)
    print("BACKTEST h=3 ЗАВЕРШЕН")
    print("="*70)
    print(f"\nTop 3 модели:")
    for i, row in metrics.head(3).iterrows():
        print(f"  {i+1}. {row['Model']}: MAE {row['MAE']:.3f}")

    print(f"\nРезультаты сохранены в archive/results/")
    print(f"  - backtest_h3_predictions.csv")
    print(f"  - backtest_h3_metrics.csv")
    print(f"  - backtest_h3_summary.md\n")
    
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
