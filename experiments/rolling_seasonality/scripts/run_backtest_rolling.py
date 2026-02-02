"""
Бэктест Rolling Seasonality Ridge
=================================

Скрипт для тестирования модели с разными значениями seasonality_window:
- 24 месяца (2 года)
- 36 месяцев (3 года)  
- 48 месяцев (4 года)

Сравнение с baseline Ridge и другими моделями.

Метрики: MAE, RMSE, KPI violations

Автор: Claude Code
Дата: 2026-02-02
"""

import sys
import os

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd
import numpy as np
from datetime import datetime

# Импортируем нашу модель
from models.rolling_seasonality_ridge import RollingSeasonalityRidge

# Импортируем baseline для сравнения
from sirena.models import RidgeForecaster, HuberForecaster


def load_data():
    """Загрузка данных."""
    # Определяем путь к данным относительно этого скрипта
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, '..', '..', '..', 'data')
    
    data_paths = [
        os.path.join(data_dir, 'inflation_data.csv'),
        os.path.join(data_dir, 'infl_kbr.csv'),
    ]
    
    for path in data_paths:
        if os.path.exists(path):
            print(f"Загружаем данные из: {os.path.abspath(path)}")
            
            if 'inflation_data' in path:
                df = pd.read_csv(path, sep=';', decimal=',', parse_dates=['Date'], index_col='Date')
                # Переименовываем колонки для совместимости
                df = df.rename(columns={
                    'mom': 'Все товары и услуги',
                    'Prod': 'Продовольственные товары',
                    'Nonprod': 'Непродовольственные товары',
                    'Serv': 'Услуги',
                    'Ki': 'Ki',
                    'Ruonia': 'Ruonia'
                })
            else:
                # infl_kbr.csv
                df = pd.read_csv(path, sep=';', decimal=',')
                df['Date'] = pd.to_datetime(df['Day'], format='%d.%m.%Y', errors='coerce')
                df = df.set_index('Date')
                # Pivot если нужно
                if 'Товар' in df.columns:
                    df = df.pivot(columns='Товар', values='MoM')
            
            return df
    
    raise FileNotFoundError(f"Не найдены данные. Искали в: {data_dir}")


def run_backtest_h1(model, df, test_months=12, target_col='Все товары и услуги'):
    """
    Бэктест h=1 (прогноз на 1 месяц вперед).
    
    Rolling window за последние test_months месяцев.
    """
    # Определяем даты для тестирования
    valid_dates = df.dropna(subset=[target_col]).index
    end_date = valid_dates.max()
    start_test = end_date - pd.DateOffset(months=test_months-1)
    test_dates = valid_dates[valid_dates >= start_test]
    
    print(f"Бэктест h=1: {len(test_dates)} месяцев ({test_dates[0].strftime('%Y-%m')} - {test_dates[-1].strftime('%Y-%m')})")
    
    results = []
    
    for target_date in test_dates:
        # Обучаем на всех данных до target_date
        train_df = df[df.index < target_date].copy()
        
        if len(train_df.dropna(subset=[target_col])) < 36:
            continue
        
        try:
            # Создаём свежую модель
            if hasattr(model, 'seasonality_window'):
                m = RollingSeasonalityRidge(
                    seasonality_window=model.seasonality_window,
                    alpha=model.alpha if hasattr(model, 'alpha') else 0.3,
                    use_macro=model.use_macro if hasattr(model, 'use_macro') else True
                )
            else:
                m = model.__class__(
                    alpha=model.alpha if hasattr(model, 'alpha') else 0.3,
                    use_macro=model.use_macro if hasattr(model, 'use_macro') else True
                )
            
            m.fit(train_df, target_col)
            
            test_df = df[df.index <= target_date].copy()
            pred_result = m.predict(test_df, target_date)
            actual = df.loc[target_date, target_col]
            
            results.append({
                'date': target_date,
                'actual': actual,
                'prediction': pred_result['prediction'],
                'error': actual - pred_result['prediction']
            })
        except Exception as e:
            print(f"  Error at {target_date}: {e}")
            continue
    
    return pd.DataFrame(results)


def calculate_metrics(results_df):
    """Расчёт метрик качества."""
    if results_df.empty:
        return {'MAE': np.nan, 'RMSE': np.nan, 'KPI': np.nan, 'KPI_violations': np.nan}
    
    errors = results_df['error'].abs()
    mae = errors.mean()
    rmse = np.sqrt((results_df['error'] ** 2).mean())
    kpi_rate = (errors <= 0.5).sum() / len(results_df) * 100
    kpi_violations = (errors > 0.5).sum()
    
    return {
        'MAE': mae,
        'RMSE': rmse,
        'KPI': kpi_rate,
        'KPI_violations': kpi_violations,
        'n_obs': len(results_df)
    }


def main():
    """Главная функция."""
    print("="*70)
    print("БЭКТЕСТ ROLLING SEASONALITY RIDGE")
    print("="*70)
    print(f"Дата запуска: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Загружаем данные
    df = load_data()
    print(f"Данные: {len(df)} записей с {df.index.min().strftime('%Y-%m')} по {df.index.max().strftime('%Y-%m')}")
    print()
    
    # Модели для тестирования
    models_to_test = [
        ('Ridge (baseline)', RidgeForecaster()),
        ('Huber (best)', HuberForecaster()),
        ('Rolling_24m', RollingSeasonalityRidge(seasonality_window=24)),
        ('Rolling_36m', RollingSeasonalityRidge(seasonality_window=36)),
        ('Rolling_48m', RollingSeasonalityRidge(seasonality_window=48)),
    ]
    
    results_summary = []
    all_predictions = {}
    
    # Запускаем бэктесты
    for model_name, model in models_to_test:
        print(f"\n{'='*70}")
        print(f"Модель: {model_name}")
        print(f"{'='*70}")
        
        results = run_backtest_h1(model, df, test_months=12)
        
        if not results.empty:
            metrics = calculate_metrics(results)
            
            print(f"  MAE: {metrics['MAE']:.4f}")
            print(f"  RMSE: {metrics['RMSE']:.4f}")
            print(f"  KPI Rate: {metrics['KPI']:.1f}%")
            print(f"  KPI Violations: {metrics['KPI_violations']}/{metrics['n_obs']}")
            print()
            print("  Детализация по месяцам:")
            for _, row in results.iterrows():
                marker = "❌" if abs(row['error']) > 0.5 else "✓"
                print(f"    {marker} {row['date'].strftime('%Y-%m')}: "
                      f"pred={row['prediction']:.2f}, "
                      f"actual={row['actual']:.2f}, "
                      f"err={row['error']:+.2f}")
            
            results_summary.append({
                'Model': model_name,
                **metrics
            })
            
            all_predictions[model_name] = results
        else:
            print("  ❌ Нет результатов")
    
    # Сводная таблица
    print(f"\n{'='*70}")
    print("СВОДНЫЕ РЕЗУЛЬТАТЫ")
    print(f"{'='*70}")
    
    summary_df = pd.DataFrame(results_summary)
    summary_df = summary_df.sort_values('MAE')
    
    print("\n" + summary_df.to_string(index=False))
    
    # Сравнение с baseline
    if len(results_summary) >= 2:
        baseline_mae = None
        for r in results_summary:
            if 'Ridge (baseline)' in r['Model']:
                baseline_mae = r['MAE']
                break
        
        if baseline_mae:
            print(f"\n{'='*70}")
            print("СРАВНЕНИЕ С BASELINE (Ridge)")
            print(f"{'='*70}")
            
            for r in results_summary:
                if 'Ridge' not in r['Model']:
                    improvement = (baseline_mae - r['MAE']) / baseline_mae * 100
                    direction = "лучше" if improvement > 0 else "хуже"
                    print(f"  {r['Model']}: {abs(improvement):.1f}% {direction} "
                          f"({r['MAE']:.4f} vs {baseline_mae:.4f})")
    
    # Сохраняем результаты
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Сохраняем сводку
    summary_path = os.path.join(output_dir, f'backtest_summary_{timestamp}.csv')
    summary_df.to_csv(summary_path, index=False)
    print(f"\n💾 Результаты сохранены в: {summary_path}")
    
    # Сохраняем предсказания
    for model_name, preds in all_predictions.items():
        safe_name = model_name.replace(' ', '_').replace('(', '').replace(')', '')
        preds_path = os.path.join(output_dir, f'predictions_{safe_name}_{timestamp}.csv')
        preds.to_csv(preds_path, index=False)
    
    print(f"💾 Предсказания сохранены для {len(all_predictions)} моделей")
    
    # Автоматическая синхронизация в sync/
    print("\n🔄 Синхронизация результатов в sync/...")
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
            print(f"⚠️ Ошибка синхронизации")
    except Exception as e:
        print(f"⚠️ Не удалось синхронизировать: {e}")
    
    return summary_df, all_predictions


if __name__ == '__main__':
    summary, predictions = main()
