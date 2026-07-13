"""
Сравнение Deep Learning моделей с лучшими традиционными моделями
===============================================================

Сравниваем:
- LSTM (новая Deep Learning модель)
- NGBoost Shock (лучшая модель v4.6, MAE 0.2981)
- Ridge (baseline)
"""

import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


def load_data():
    """Загрузка данных из CSV."""
    df = pd.read_csv(
        'data/infl_kbr.csv',
        sep=';',
        decimal=','
    )
    df['Day'] = pd.to_datetime(df['Day'], format='%d.%m.%Y')

    pivot = df.pivot(index='Day', columns='Товар', values='MoM')
    pivot.index = pd.to_datetime(pivot.index)
    pivot = pivot.sort_index()

    for col in pivot.columns:
        pivot[col] = pd.to_numeric(pivot[col], errors='coerce')

    return pivot


def get_metrics(results: pd.DataFrame) -> dict:
    """Расчёт метрик качества."""
    if results.empty:
        return {'MAE': float('inf'), 'RMSE': float('inf'), 'KPI': 0, 'n_points': 0}

    errors = results['error'].abs()
    mae = errors.mean()
    rmse = np.sqrt((results['error'] ** 2).mean())
    kpi = (errors <= 0.5).sum() / len(results) * 100

    return {'MAE': mae, 'RMSE': rmse, 'KPI': kpi, 'n_points': len(results)}


def run_comparison():
    """Запуск сравнения Deep Learning vs традиционных моделей."""
    print("=" * 80)
    print("SIRENA-KBR v4.6: Deep Learning Comparison")
    print("=" * 80)

    df = load_data()
    print(f"\nДанные: {df.index.min().strftime('%Y-%m')} — {df.index.max().strftime('%Y-%m')}")
    print(f"Количество точек: {len(df)}")

    start_date = '2023-01-01'
    target_col = 'Все товары и услуги'

    results = {}

    # === LSTM ===
    print("\n" + "=" * 40)
    print("LSTM (Deep Learning)")
    print("=" * 40)

    try:
        from sirena.models import LSTMForecaster, TORCH_AVAILABLE
        if TORCH_AVAILABLE:
            print("[LSTM] Запуск бэктеста...")
            lstm = LSTMForecaster(
                hidden_size=32,
                num_layers=2,
                dropout=0.2,
                sequence_length=12,
                epochs=100,
                patience=10
            )
            bt = lstm.backtest(df, start_date=start_date, target_col=target_col)
            metrics = get_metrics(bt)
            results['LSTM'] = metrics
            print(f"   MAE: {metrics['MAE']:.4f}")
            print(f"   RMSE: {metrics['RMSE']:.4f}")
            print(f"   KPI (<=0.5): {metrics['KPI']:.1f}%")
            print(f"   Точек: {metrics['n_points']}")
        else:
            print("   ОШИБКА: PyTorch не установлен")
    except Exception as e:
        print(f"   ОШИБКА: {e}")

    # === NGBoost Shock ===
    print("\n" + "=" * 40)
    print("NGBoost Shock (лучшая модель v4.6)")
    print("=" * 40)

    try:
        from sirena.models.ngboost_shock import NGBoostShockForecaster
        print("[NGBoost Shock] Запуск бэктеста...")
        ngb = NGBoostShockForecaster()
        bt = ngb.backtest(df, start_date=start_date, target_col=target_col)
        metrics = get_metrics(bt)
        results['NGBoost Shock'] = metrics
        print(f"   MAE: {metrics['MAE']:.4f}")
        print(f"   RMSE: {metrics['RMSE']:.4f}")
        print(f"   KPI (<=0.5): {metrics['KPI']:.1f}%")
        print(f"   Точек: {metrics['n_points']}")
    except Exception as e:
        print(f"   ОШИБКА: {e}")

    # === Ridge (baseline) ===
    print("\n" + "=" * 40)
    print("Ridge (baseline)")
    print("=" * 40)

    try:
        from sirena.models import RidgeForecaster
        print("[Ridge] Запуск бэктеста...")
        ridge = RidgeForecaster()
        bt = ridge.backtest(df, start_date=start_date, target_col=target_col)
        metrics = get_metrics(bt)
        results['Ridge'] = metrics
        print(f"   MAE: {metrics['MAE']:.4f}")
        print(f"   RMSE: {metrics['RMSE']:.4f}")
        print(f"   KPI (<=0.5): {metrics['KPI']:.1f}%")
        print(f"   Точек: {metrics['n_points']}")
    except Exception as e:
        print(f"   ОШИБКА: {e}")

    # === Итоговые результаты ===
    print("\n" + "=" * 80)
    print("ИТОГОВЫЕ РЕЗУЛЬТАТЫ")
    print("=" * 80)

    if not results:
        print("Нет результатов для отображения")
        return

    df_results = pd.DataFrame(results).T
    df_results = df_results.sort_values('MAE')

    baseline_mae = results.get('Ridge', {}).get('MAE', df_results['MAE'].iloc[0])

    print("\n{:<20} {:>8} {:>8} {:>8} {:>12}".format(
        'Model', 'MAE', 'RMSE', 'KPI%', 'vs Ridge%'))
    print("-" * 70)

    for model_name, row in df_results.iterrows():
        vs = (row['MAE'] - baseline_mae) / baseline_mae * 100 if baseline_mae > 0 else 0
        marker = " ***" if model_name == 'LSTM' else ""
        print("{:<20} {:>8.4f} {:>8.4f} {:>8.1f} {:>+12.2f}%{}".format(
            model_name, row['MAE'], row['RMSE'], row['KPI'], vs, marker))

    # === Выводы ===
    print("\n" + "=" * 80)
    print("ВЫВОДЫ")
    print("=" * 80)

    best_model = df_results['MAE'].idxmin()
    best_mae = df_results['MAE'].min()

    if 'LSTM' in results:
        lstm_mae = results['LSTM']['MAE']
        ngb_mae = results.get('NGBoost Shock', {}).get('MAE', float('inf'))
        ridge_mae = results.get('Ridge', {}).get('MAE', float('inf'))

        if lstm_mae < ngb_mae:
            print(f"✅ LSTM ПРЕВОСХОДИТ NGBoost Shock!")
            print(f"   LSTM MAE: {lstm_mae:.4f}")
            print(f"   NGBoost Shock MAE: {ngb_mae:.4f}")
            print(f"   Улучшение: {(ngb_mae - lstm_mae) / ngb_mae * 100:.2f}%")
        else:
            print(f"❌ LSTM не превзошла NGBoost Shock")
            print(f"   LSTM MAE: {lstm_mae:.4f}")
            print(f"   NGBoost Shock MAE: {ngb_mae:.4f}")
            print(f"   Разница: {(lstm_mae - ngb_mae) / ngb_mae * 100:+.2f}%")

        print(f"\nВывод относительно Ridge:")
        vs_ridge = (lstm_mae - ridge_mae) / ridge_mae * 100
        if vs_ridge < 0:
            print(f"   LSTM лучше Ridge на {-vs_ridge:.2f}%")
        else:
            print(f"   LSTM хуже Ridge на {vs_ridge:.2f}%")

    print(f"\n{'='*80}")
    print(f"ЛУЧШАЯ МОДЕЛЬ: {best_model} (MAE: {best_mae:.4f})")
    print(f"{'='*80}")

    # Сохранение
    df_results.to_csv('deep_learning_comparison.csv')
    print(f"\nРезультаты сохранены в deep_learning_comparison.csv")

    return df_results


if __name__ == '__main__':
    run_comparison()
