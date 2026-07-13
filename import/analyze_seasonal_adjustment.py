# scripts/analyze_seasonal_adjustment.py

import sys
from pathlib import Path
import argparse
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf

# Настройка путей и импорты
ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))
from utils.data_loader import load_and_prepare_data
from utils.seasonal_adjustment_x13 import perform_seasonal_adjustment

def analyze_seasonal_adjustment(series_name: str):
    print(f"--- Анализ качества сезонного сглаживания для '{series_name}' ---")
    
    df_orig, df_sa = load_and_prepare_data(
        ROOT_DIR,
        seasonal_adjuster_func=perform_seasonal_adjustment
    )

    if series_name not in df_sa or df_sa[series_name].dropna().empty:
        print(f"\nАнализ невозможен: не удалось сгенерировать сглаженный ряд для '{series_name}'.")
        return
        
    original_series = df_orig[series_name].dropna()
    adjusted_series = df_sa[series_name].dropna()
    seasonal_component = (original_series - adjusted_series).dropna()
    
    print("\n[1] Построение диагностических графиков...")
    fig, axes = plt.subplots(4, 1, figsize=(15, 20), sharex=True)
    fig.suptitle(f"Анализ сезонного сглаживания для '{series_name}'", fontsize=16)
    
    axes[0].plot(original_series, label='Исходный ряд', color='blue', alpha=0.7)
    axes[0].plot(adjusted_series, label='Сглаженный ряд', color='red', linewidth=2)
    axes[0].set_title('Исходный vs. Сглаженный ряд'); axes[0].legend(); axes[0].grid(True)
    
    axes[1].bar(seasonal_component.index, seasonal_component, width=20)
    axes[1].set_title('Извлеченная сезонная компонента'); axes[1].grid(True)

    plot_acf(original_series, lags=36, ax=axes[2], title='ACF Исходного ряда')
    plot_acf(adjusted_series, lags=36, ax=axes[3], title='ACF Сглаженного ряда')
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    plot_path = (ROOT_DIR / "results/plots/analysis").resolve()
    plot_path.mkdir(parents=True, exist_ok=True)
    filename = plot_path / f"sa_analysis_{series_name}.png"
    plt.savefig(filename, dpi=150)
    print(f"Диагностический график сохранен: {filename}")
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Анализ сезонного сглаживания.")
    parser.add_argument("series_name", default="mom", nargs="?", help="Имя переменной (по умолч. 'mom').")
    args = parser.parse_args()
    analyze_seasonal_adjustment(args.series_name)