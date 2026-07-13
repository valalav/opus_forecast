"""
Сравнительный бэктест всех ЛММР моделей с графиками.

Модели:
- LMMR (Gemini) - sirena/models/lmmr.py
- LMMR Claude - sirena/models/lmmr_claude.py
- LMMR Hybrid - sirena/models/lmmr_hybrid.py
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
import warnings
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

warnings.filterwarnings("ignore")

def load_data():
    """Загрузка и подготовка данных."""
    # Инфляция
    df_raw = pd.read_csv('data/infl_kbr.csv', sep=';', decimal=',')
    df_raw['MoM'] = pd.to_numeric(df_raw['MoM'].astype(str).str.replace(',', '.'), errors='coerce')
    df = df_raw.pivot(index='Date', columns='Товар', values='MoM')
    df.index = pd.to_datetime(df.index)
    df.index = df.index.to_period('M').to_timestamp()
    df = df.sort_index()

    # Экзогенные переменные
    try:
        exog = pd.read_csv('data/inflation_data.csv', sep=';', decimal=',')
        exog['Date'] = pd.to_datetime(exog['Date'], format='%d.%m.%Y', errors='coerce')

        for col in ['usd_nom_i', 'Ki_i', 'Ruonia', 'fl_potrb_zad', 'all_real']:
            if col in exog.columns:
                exog[col] = pd.to_numeric(exog[col], errors='coerce')

        exog['period_date'] = exog['Date'].dt.to_period('M').dt.to_timestamp()
        exog.set_index('period_date', inplace=True)

        for col in ['usd_nom_i', 'Ki_i', 'Ruonia', 'fl_potrb_zad', 'all_real', 'brent']:
            if col in exog.columns:
                df = df.join(pd.DataFrame(exog[col]), how='left')
    except Exception as e:
        print(f"Warning: Could not load exogenous data: {e}")

    return df

def run_backtest_model(model_class, df, model_name, start_date='2023-01-01', target_col='Все товары и услуги'):
    """Запуск бэктеста для одной модели."""
    start = pd.Timestamp(start_date)
    test_dates = df.index[df.index >= start]

    results = []

    for target_date in test_dates:
        train_df = df[df.index < target_date].copy()

        # Минимум данных для обучения
        if len(train_df.dropna(subset=[target_col])) < 36:
            continue

        try:
            model = model_class()
            model.fit(train_df, target_col)

            test_df = df[df.index <= target_date].copy()
            pred = model.predict(test_df, target_date)
            prediction = pred['prediction']
            actual = df.loc[target_date, target_col]

            results.append({
                'date': target_date,
                'actual': actual,
                'prediction': prediction,
                'error': actual - prediction,
                'abs_error': abs(actual - prediction)
            })
        except Exception as e:
            continue

    return pd.DataFrame(results)

def main():
    print("=" * 60)
    print("СРАВНИТЕЛЬНЫЙ БЭКТЕСТ МОДЕЛЕЙ ЛММР")
    print("=" * 60)
    print()

    # Загрузка данных
    print("Загрузка данных...")
    df = load_data()
    target_col = 'Все товары и услуги'

    print(f"Период данных: {df.index.min().date()} — {df.index.max().date()}")
    print(f"Всего наблюдений: {len(df)}")
    print()

    # Импорт моделей
    results_all = {}

    # 1. LMMR Gemini
    print("1. Бэктест LMMR (Gemini)...")
    try:
        from sirena.models.lmmr import LMMRForecaster
        results_gemini = run_backtest_model(LMMRForecaster, df, "LMMR Gemini")
        if len(results_gemini) > 0:
            results_all['LMMR Gemini'] = results_gemini
            mae = results_gemini['abs_error'].mean()
            print(f"   MAE: {mae:.3f}, точек: {len(results_gemini)}")
        else:
            print("   Нет результатов")
    except Exception as e:
        print(f"   Ошибка: {e}")

    # 2. LMMR Claude
    print("2. Бэктест LMMR Claude...")
    try:
        from sirena.models.lmmr_claude import LMMRForecasterClaude
        results_claude = run_backtest_model(LMMRForecasterClaude, df, "LMMR Claude")
        if len(results_claude) > 0:
            results_all['LMMR Claude'] = results_claude
            mae = results_claude['abs_error'].mean()
            print(f"   MAE: {mae:.3f}, точек: {len(results_claude)}")
        else:
            print("   Нет результатов")
    except Exception as e:
        print(f"   Ошибка: {e}")

    # 3. LMMR Hybrid
    print("3. Бэктест LMMR Hybrid...")
    try:
        from sirena.models.lmmr_hybrid import LMMRHybridForecaster
        results_hybrid = run_backtest_model(LMMRHybridForecaster, df, "LMMR Hybrid")
        if len(results_hybrid) > 0:
            results_all['LMMR Hybrid'] = results_hybrid
            mae = results_hybrid['abs_error'].mean()
            print(f"   MAE: {mae:.3f}, точек: {len(results_hybrid)}")
        else:
            print("   Нет результатов")
    except Exception as e:
        print(f"   Ошибка: {e}")

    # 4. Baseline Ridge (для сравнения)
    print("4. Бэктест Ridge (baseline)...")
    try:
        from sirena.models.ridge import RidgeForecaster
        results_ridge = run_backtest_model(RidgeForecaster, df, "Ridge")
        if len(results_ridge) > 0:
            results_all['Ridge (baseline)'] = results_ridge
            mae = results_ridge['abs_error'].mean()
            print(f"   MAE: {mae:.3f}, точек: {len(results_ridge)}")
        else:
            print("   Нет результатов")
    except Exception as e:
        print(f"   Ошибка: {e}")

    if len(results_all) == 0:
        print("\nНет результатов для визуализации!")
        return

    print()
    print("=" * 60)
    print("СВОДНАЯ ТАБЛИЦА МЕТРИК")
    print("=" * 60)

    metrics_table = []
    for name, res in results_all.items():
        mae = res['abs_error'].mean()
        rmse = np.sqrt((res['error'] ** 2).mean())
        direction_acc = np.mean(
            np.sign(res['actual'] - 100) == np.sign(res['prediction'] - 100)
        ) * 100
        metrics_table.append({
            'Модель': name,
            'MAE': f"{mae:.3f}",
            'RMSE': f"{rmse:.3f}",
            'Direction Acc': f"{direction_acc:.1f}%",
            'N': len(res)
        })

    metrics_df = pd.DataFrame(metrics_table)
    print(metrics_df.to_string(index=False))
    print()

    # Построение графиков
    print("Построение графиков...")

    # Цвета для моделей
    colors = {
        'LMMR Gemini': '#2ecc71',      # зелёный
        'LMMR Claude': '#3498db',       # синий
        'LMMR Hybrid': '#e74c3c',       # красный
        'Ridge (baseline)': '#9b59b6',  # фиолетовый
        'Actual': '#2c3e50'             # тёмно-серый
    }

    # Создаём фигуру с тремя subplot-ами
    fig, axes = plt.subplots(3, 1, figsize=(14, 12))
    fig.suptitle('Сравнительный бэктест моделей ЛММР (2023-2025)', fontsize=14, fontweight='bold')

    # 1. График прогнозов vs факт
    ax1 = axes[0]

    # Определяем общий диапазон дат
    all_dates = set()
    for res in results_all.values():
        all_dates.update(res['date'].tolist())
    all_dates = sorted(all_dates)

    # Факт (берём из первого результата)
    first_results = list(results_all.values())[0]
    ax1.plot(first_results['date'], first_results['actual'],
             color=colors['Actual'], linewidth=2.5, label='Факт', marker='o', markersize=4)

    # Прогнозы каждой модели
    for name, res in results_all.items():
        ax1.plot(res['date'], res['prediction'],
                 color=colors.get(name, '#777777'), linewidth=1.5,
                 label=name, linestyle='--', alpha=0.8)

    ax1.set_ylabel('ИПЦ MoM (%)')
    ax1.set_title('Прогнозы vs Факт')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')

    # 2. График ошибок
    ax2 = axes[1]

    for name, res in results_all.items():
        ax2.plot(res['date'], res['error'],
                 color=colors.get(name, '#777777'), linewidth=1.5,
                 label=name, marker='o', markersize=3, alpha=0.8)

    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax2.set_ylabel('Ошибка (Факт - Прогноз)')
    ax2.set_title('Ошибки прогноза')
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')

    # 3. Кумулятивная абсолютная ошибка
    ax3 = axes[2]

    for name, res in results_all.items():
        cumulative_mae = res['abs_error'].cumsum() / (np.arange(len(res)) + 1)
        ax3.plot(res['date'], cumulative_mae,
                 color=colors.get(name, '#777777'), linewidth=2,
                 label=name)

    ax3.set_ylabel('Кумулятивный MAE')
    ax3.set_xlabel('Дата')
    ax3.set_title('Динамика MAE')
    ax3.legend(loc='upper right')
    ax3.grid(True, alpha=0.3)
    ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax3.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45, ha='right')

    plt.tight_layout()

    # Сохраняем график
    output_path = 'assets/images/lmmr_backtest_comparison.png'
    os.makedirs('assets/images', exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"График сохранён: {output_path}")

    print()
    print("=" * 60)
    print("ВЫВОДЫ")
    print("=" * 60)

    # Находим лучшую модель
    best_model = min(results_all.items(), key=lambda x: x[1]['abs_error'].mean())
    worst_model = max(results_all.items(), key=lambda x: x[1]['abs_error'].mean())

    print(f"Лучшая модель: {best_model[0]} (MAE: {best_model[1]['abs_error'].mean():.3f})")
    print(f"Худшая модель: {worst_model[0]} (MAE: {worst_model[1]['abs_error'].mean():.3f})")

    if 'Ridge (baseline)' in results_all:
        ridge_mae = results_all['Ridge (baseline)']['abs_error'].mean()
        print()
        print("Сравнение с Ridge baseline:")
        for name, res in results_all.items():
            if name != 'Ridge (baseline)':
                mae = res['abs_error'].mean()
                diff = (mae - ridge_mae) / ridge_mae * 100
                sign = '+' if diff > 0 else ''
                print(f"  {name}: {sign}{diff:.1f}%")

if __name__ == "__main__":
    main()
