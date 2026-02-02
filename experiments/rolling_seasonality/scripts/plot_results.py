"""
Визуализация результатов Rolling Seasonality Ridge
=================================================

Графики:
1. Сравнение прогнозов всех моделей vs факт
2. Ошибки по месяцам
3. MAE сравнение
4. Сезонные нормы (Rolling vs Global)

Автор: Claude Code
Дата: 2026-02-02
"""

import sys
import os
import glob

# Add paths
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime

# Настройка стиля
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (14, 10)
plt.rcParams['font.size'] = 10


def load_predictions(results_dir):
    """Загрузка всех предсказаний."""
    prediction_files = glob.glob(os.path.join(results_dir, 'predictions_*.csv'))
    
    all_data = {}
    for file in prediction_files:
        # Извлекаем имя модели из имени файла
        basename = os.path.basename(file)
        model_name = basename.replace('predictions_', '').replace('.csv', '')
        # Убираем timestamp
        if '_' in model_name and model_name.split('_')[-1].isdigit():
            model_name = '_'.join(model_name.split('_')[:-1])
        
        df = pd.read_csv(file, parse_dates=['date'])
        all_data[model_name] = df
        
    return all_data


def plot_forecast_comparison(all_data, output_dir):
    """График 1: Сравнение прогнозов vs факт."""
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    # Цвета для моделей
    colors = {
        'Ridge_baseline': '#1f77b4',
        'Huber_best': '#2ca02c',
        'Rolling_24m': '#d62728',
        'Rolling_36m': '#ff7f0e',
        'Rolling_48m': '#9467bd',
    }
    
    # Получаем фактические данные (из любой модели)
    first_model = list(all_data.values())[0]
    actual = first_model[['date', 'actual']].copy()
    
    # === График 1: Уровень (прогнозы vs факт) ===
    ax1 = axes[0]
    
    # Факт
    ax1.plot(actual['date'], actual['actual'], 'ko-', linewidth=2, 
             markersize=8, label='Actual', zorder=10)
    
    # Модели
    for model_name, df in all_data.items():
        color = colors.get(model_name, '#7f7f7f')
        label = model_name.replace('_', ' ')
        ax1.plot(df['date'], df['prediction'], 's--', color=color, 
                linewidth=1.5, markersize=6, label=label, alpha=0.8)
    
    ax1.set_title('Forecast Comparison: All Models vs Actual (2025)', fontsize=14, fontweight='bold')
    ax1.set_ylabel('MoM Index (%)', fontsize=12)
    ax1.legend(loc='best', fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax1.xaxis.set_major_locator(mdates.MonthLocator())
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)
    
    # === График 2: Ошибки ===
    ax2 = axes[1]
    
    x = np.arange(len(actual))
    width = 0.15
    
    models_list = list(all_data.keys())
    for i, model_name in enumerate(models_list):
        df = all_data[model_name]
        errors = df['error'].values
        color = colors.get(model_name, '#7f7f7f')
        offset = (i - len(models_list)/2) * width
        ax2.bar(x + offset, errors, width, label=model_name.replace('_', ' '), 
                color=color, alpha=0.7)
    
    # Линия нуля
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    # KPI линии
    ax2.axhline(y=0.5, color='red', linestyle='--', linewidth=1, alpha=0.5, label='KPI ±0.5')
    ax2.axhline(y=-0.5, color='red', linestyle='--', linewidth=1, alpha=0.5)
    
    ax2.set_title('Forecast Errors by Month', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Error (Actual - Predicted)', fontsize=12)
    ax2.set_xlabel('Month', fontsize=12)
    ax2.set_xticks(x)
    ax2.set_xticklabels([d.strftime('%Y-%m') for d in actual['date']], rotation=45)
    ax2.legend(loc='best', fontsize=9)
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'forecast_comparison.png'), dpi=150, bbox_inches='tight')
    print(f"💾 Сохранено: {os.path.join(output_dir, 'forecast_comparison.png')}")
    plt.close()


def plot_mae_comparison(summary_df, output_dir):
    """График 2: Сравнение MAE."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Сортируем по MAE
    summary_df = summary_df.sort_values('MAE')
    
    colors_list = []
    for model in summary_df['Model']:
        if 'Ridge' in model and 'Rolling' not in model:
            colors_list.append('#1f77b4')  # Ridge baseline
        elif 'Huber' in model:
            colors_list.append('#2ca02c')  # Huber (green)
        elif '24m' in model:
            colors_list.append('#d62728')  # Rolling 24m (red, best)
        elif '36m' in model:
            colors_list.append('#ff7f0e')  # Rolling 36m (orange)
        elif '48m' in model:
            colors_list.append('#9467bd')  # Rolling 48m (purple)
        else:
            colors_list.append('#7f7f7f')
    
    bars = ax.barh(summary_df['Model'], summary_df['MAE'], color=colors_list, alpha=0.8, edgecolor='black')
    
    # Подписи значений
    for bar, mae in zip(bars, summary_df['MAE']):
        ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height()/2, 
                f'{mae:.4f}', va='center', fontsize=10, fontweight='bold')
    
    # Линия Ridge baseline
    ridge_mae = summary_df[summary_df['Model'].str.contains('Ridge') & 
                           ~summary_df['Model'].str.contains('Rolling')]['MAE'].values
    if len(ridge_mae) > 0:
        ax.axvline(x=ridge_mae[0], color='#1f77b4', linestyle='--', linewidth=2, 
                   alpha=0.7, label=f'Ridge Baseline ({ridge_mae[0]:.4f})')
    
    ax.set_xlabel('MAE (Mean Absolute Error)', fontsize=12)
    ax.set_title('Model Comparison: MAE (Lower is Better)', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'mae_comparison.png'), dpi=150, bbox_inches='tight')
    print(f"💾 Сохранено: {os.path.join(output_dir, 'mae_comparison.png')}")
    plt.close()


def plot_cumulative_errors(all_data, output_dir):
    """График 3: Кумулятивные ошибки."""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    colors = {
        'Ridge_baseline': '#1f77b4',
        'Huber_best': '#2ca02c',
        'Rolling_24m': '#d62728',
        'Rolling_36m': '#ff7f0e',
        'Rolling_48m': '#9467bd',
    }
    
    first_model = list(all_data.values())[0]
    dates = first_model['date']
    
    for model_name, df in all_data.items():
        color = colors.get(model_name, '#7f7f7f')
        label = model_name.replace('_', ' ')
        
        # Кумулятивная абсолютная ошибка
        cum_errors = np.abs(df['error']).cumsum()
        ax.plot(dates, cum_errors, 'o-', color=color, linewidth=2, 
                markersize=6, label=label, alpha=0.8)
    
    ax.set_xlabel('Month', fontsize=12)
    ax.set_ylabel('Cumulative Absolute Error', fontsize=12)
    ax.set_title('Cumulative Absolute Errors Over Time', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'cumulative_errors.png'), dpi=150, bbox_inches='tight')
    print(f"💾 Сохранено: {os.path.join(output_dir, 'cumulative_errors.png')}")
    plt.close()


def plot_error_distribution(all_data, output_dir):
    """График 4: Распределение ошибок."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    colors = {
        'Ridge_baseline': '#1f77b4',
        'Huber_best': '#2ca02c',
        'Rolling_24m': '#d62728',
        'Rolling_36m': '#ff7f0e',
        'Rolling_48m': '#9467bd',
    }
    
    for i, (model_name, df) in enumerate(all_data.items()):
        if i >= 5:
            break
            
        ax = axes[i]
        errors = df['error']
        color = colors.get(model_name, '#7f7f7f')
        
        # Гистограмма
        ax.hist(errors, bins=8, color=color, alpha=0.6, edgecolor='black')
        
        # Статистики
        mae = np.abs(errors).mean()
        mean_err = errors.mean()
        std_err = errors.std()
        
        # Нормальное распределение для сравнения
        x = np.linspace(errors.min(), errors.max(), 100)
        normal = (1/(std_err * np.sqrt(2*np.pi))) * np.exp(-0.5*((x-mean_err)/std_err)**2)
        ax.plot(x, normal * len(errors) * (errors.max()-errors.min())/8, 
                'r--', linewidth=2, label='Normal fit')
        
        ax.axvline(x=0, color='black', linestyle='-', linewidth=1)
        ax.axvline(x=0.5, color='red', linestyle='--', linewidth=1, alpha=0.5)
        ax.axvline(x=-0.5, color='red', linestyle='--', linewidth=1, alpha=0.5)
        
        ax.set_title(f'{model_name.replace("_", " ")}\nMAE={mae:.3f}, Mean={mean_err:.3f}', 
                     fontsize=11, fontweight='bold')
        ax.set_xlabel('Error', fontsize=10)
        ax.set_ylabel('Frequency', fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    
    # Убираем лишний subplot
    if len(all_data) < 6:
        fig.delaxes(axes[5])
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'error_distribution.png'), dpi=150, bbox_inches='tight')
    print(f"💾 Сохранено: {os.path.join(output_dir, 'error_distribution.png')}")
    plt.close()


def plot_seasonal_norms_comparison(output_dir):
    """График 5: Сравнение сезонных норм (Rolling vs Global)."""
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    
    from models.rolling_seasonality_ridge import RollingSeasonalityRidge
    from sirena.models import RidgeForecaster
    
    # Загружаем данные
    data_path = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'data', 'inflation_data.csv')
    df = pd.read_csv(data_path, sep=';', decimal=',', parse_dates=['Date'], index_col='Date')
    df = df.rename(columns={'mom': 'Все товары и услуги', 'year': 'year'})
    df['year'] = df.index.year
    df['month'] = df.index.month
    
    # Считаем сезонные нормы
    # Global (Ridge baseline)
    clean_df = df[~df['year'].isin([2022, 2010])]
    global_norm = clean_df.groupby('month')['Все товары и услуги'].mean()
    
    # Rolling 24m
    recent_24m = df.last('24M')
    rolling_24m_norm = recent_24m.groupby('month')['Все товары и услуги'].mean()
    
    # Rolling 36m
    recent_36m = df.last('36M')
    rolling_36m_norm = recent_36m.groupby('month')['Все товары и услуги'].mean()
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    months = range(1, 13)
    month_labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                    'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    
    ax.plot(months, global_norm.values, 'o-', linewidth=2, markersize=8, 
            label='Global (excl. 2022, 2010)', color='#1f77b4')
    ax.plot(months, rolling_24m_norm.values, 's-', linewidth=2, markersize=8, 
            label='Rolling 24m (last 2 years)', color='#d62728')
    ax.plot(months, rolling_36m_norm.values, '^-', linewidth=2, markersize=8, 
            label='Rolling 36m (last 3 years)', color='#ff7f0e')
    
    ax.set_xticks(months)
    ax.set_xticklabels(month_labels)
    ax.set_xlabel('Month', fontsize=12)
    ax.set_ylabel('Seasonal Norm (MoM %)', fontsize=12)
    ax.set_title('Seasonal Norms Comparison: Global vs Rolling Window', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=11)
    ax.grid(True, alpha=0.3)
    
    # Аннотации различий
    for m in months:
        diff = rolling_24m_norm.loc[m] - global_norm.loc[m]
        if abs(diff) > 0.3:
            ax.annotate(f'{diff:+.2f}', 
                       xy=(m, rolling_24m_norm.loc[m]), 
                       xytext=(m, rolling_24m_norm.loc[m] + 0.15),
                       fontsize=8, ha='center', color='#d62728')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'seasonal_norms_comparison.png'), dpi=150, bbox_inches='tight')
    print(f"💾 Сохранено: {os.path.join(output_dir, 'seasonal_norms_comparison.png')}")
    plt.close()


def create_summary_table(all_data, output_dir):
    """Таблица с детальными результатами."""
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.axis('off')
    
    # Собираем данные
    rows = []
    for model_name, df in all_data.items():
        errors = df['error']
        rows.append({
            'Model': model_name.replace('_', ' '),
            'MAE': f"{np.abs(errors).mean():.4f}",
            'RMSE': f"{np.sqrt((errors**2).mean()):.4f}",
            'Mean Error': f"{errors.mean():.4f}",
            'Std Error': f"{errors.std():.4f}",
            'Max |Error|': f"{np.abs(errors).max():.4f}",
            'KPI Rate': f"{(np.abs(errors) <= 0.5).sum() / len(errors) * 100:.1f}%"
        })
    
    # Сортируем по MAE
    rows = sorted(rows, key=lambda x: float(x['MAE']))
    
    # Создаём таблицу
    table_data = [[r[k] for k in r.keys()] for r in rows]
    columns = list(rows[0].keys())
    
    table = ax.table(cellText=table_data, colLabels=columns, 
                    cellLoc='center', loc='center',
                    colWidths=[0.2, 0.1, 0.1, 0.12, 0.12, 0.12, 0.12])
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # Стиль заголовка
    for i in range(len(columns)):
        table[(0, i)].set_facecolor('#40466e')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Цвета строк
    for i in range(1, len(rows) + 1):
        if i == 1:  # Лучшая модель
            for j in range(len(columns)):
                table[(i, j)].set_facecolor('#d4edda')
        elif 'Ridge' in table_data[i-1][0] and 'Rolling' not in table_data[i-1][0]:
            for j in range(len(columns)):
                table[(i, j)].set_facecolor('#cce5ff')
    
    ax.set_title('Detailed Results Summary (Sorted by MAE)', fontsize=16, fontweight='bold', pad=20)
    
    plt.savefig(os.path.join(output_dir, 'summary_table.png'), dpi=150, bbox_inches='tight')
    print(f"💾 Сохранено: {os.path.join(output_dir, 'summary_table.png')}")
    plt.close()


def main():
    """Главная функция."""
    print("="*70)
    print("ГЕНЕРАЦИЯ ГРАФИКОВ")
    print("="*70)
    print(f"Дата: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Пути
    results_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
    output_dir = results_dir
    
    print(f"Загрузка данных из: {results_dir}")
    
    # Загружаем предсказания
    all_data = load_predictions(results_dir)
    
    if not all_data:
        print("❌ Не найдены файлы с предсказаниями!")
        print(f"   Искали в: {results_dir}")
        return
    
    print(f"✓ Загружены данные для {len(all_data)} моделей:")
    for name in all_data.keys():
        print(f"   - {name}")
    print()
    
    # Загружаем сводку
    summary_files = glob.glob(os.path.join(results_dir, 'backtest_summary_*.csv'))
    if summary_files:
        summary_df = pd.read_csv(summary_files[0])
    else:
        print("❌ Не найден файл сводки!")
        summary_df = None
    
    # Генерируем графики
    print("Генерация графиков...")
    print()
    
    print("1. Сравнение прогнозов vs факт...")
    plot_forecast_comparison(all_data, output_dir)
    
    print("2. Сравнение MAE...")
    if summary_df is not None:
        plot_mae_comparison(summary_df, output_dir)
    
    print("3. Кумулятивные ошибки...")
    plot_cumulative_errors(all_data, output_dir)
    
    print("4. Распределение ошибок...")
    plot_error_distribution(all_data, output_dir)
    
    print("5. Сравнение сезонных норм...")
    try:
        plot_seasonal_norms_comparison(output_dir)
    except Exception as e:
        print(f"   ⚠️ Ошибка при построении сезонных норм: {e}")
    
    print("6. Таблица результатов...")
    create_summary_table(all_data, output_dir)
    
    print()
    print("="*70)
    print("ГОТОВО!")
    print("="*70)
    print()
    print("Сгенерированные файлы:")
    for f in ['forecast_comparison.png', 'mae_comparison.png', 
              'cumulative_errors.png', 'error_distribution.png',
              'seasonal_norms_comparison.png', 'summary_table.png']:
        path = os.path.join(output_dir, f)
        if os.path.exists(path):
            print(f"  ✓ {f}")
    print()


if __name__ == '__main__':
    main()
