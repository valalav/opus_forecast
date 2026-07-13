#!/usr/bin/env python3
"""
Добавление сезонности к SA-прогнозам (ПРАВИЛЬНАЯ версия)
==========================================================
SA-прогнозы + сезонность из исходных (не-SA) данных
"""

import json
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path

def compute_seasonal_pattern():
    """Вычислить сезонность из исходных (не-SA) данных"""
    df = pd.read_csv('data/inflation_data.csv', sep=';', decimal=',', encoding='utf-8-sig')
    df['Date'] = pd.to_datetime(df['Date'], format='%d.%m.%Y')
    df['mom'] = pd.to_numeric(df['mom'].astype(str).str.replace(',', '.'), errors='coerce') - 100
    df['month'] = df['Date'].dt.month
    
    # За последние 4 года (исключая аномальный 2022)
    recent = df[df['Date'].dt.year.isin([2023, 2024, 2025])]
    seasonal = recent.groupby('month')['mom'].mean()
    
    # Центрируем
    seasonal = seasonal - seasonal.mean()
    
    return seasonal.to_dict()

def add_seasonality_correct():
    """Добавить сезонность правильно"""
    
    with open('data/precomputed_forecasts.json', 'r') as f:
        data = json.load(f)
    
    seasonal = compute_seasonal_pattern()
    forecast_dates = [datetime.strptime(d, '%Y-%m-%d') for d in data['forecast_dates']]
    
    months_name = ['', 'Янв', 'Фев', 'Мар', 'Апр', 'Май', 'Июн', 
                   'Июл', 'Авг', 'Сен', 'Окт', 'Ноя', 'Дек']
    
    print("Сезонные корректировки (из исходных данных):")
    for m, adj in sorted(seasonal.items()):
        print(f"  {months_name[m]}: {adj:+.2f}%")
    
    print("\n" + "="*60)
    print("SA-ПРОГНОЗЫ + СЕЗОННОСТЬ = ИТОГОВЫЕ ПРОГНОЗЫ")
    print("="*60)
    
    final_forecasts = {}
    
    # Только ключевые модели для примера
    for model in ['Ridge', 'Huber', 'Ensemble']:
        sa_values = data['forecasts'][model]
        adjusted = []
        
        print(f"\n{model}:")
        print(f"{'Месяц':<8} {'SA':>6} {'Сезон':>7} {'Итого':>7}")
        print("-" * 32)
        
        for i, (date, sa_val) in enumerate(zip(forecast_dates, sa_values)):
            month = date.month
            seasonal_adj = seasonal.get(month, 0)
            final = sa_val + seasonal_adj
            adjusted.append(final)
            
            if i < 6:  # Показываем первые 6 месяцев
                print(f"{months_name[month]:<8} {sa_val:>6.2f} {seasonal_adj:>+7.2f} {final:>7.2f}")
        
        final_forecasts[model] = adjusted
    
    # Сохраняем
    data['forecasts_with_seasonality'] = final_forecasts
    data['seasonal_adjustments'] = seasonal
    
    with open('data/precomputed_forecasts_v2.json', 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"\n✅ Сохранено: data/precomputed_forecasts_v2.json")
    
    # Создаём итоговую таблицу
    create_final_table(data, final_forecasts, forecast_dates)
    
    return data, final_forecasts

def create_final_table(data, final_forecasts, forecast_dates):
    """Создать итоговую таблицу сравнения"""
    
    rows = []
    months_name = ['', 'Янв', 'Фев', 'Мар', 'Апр', 'Май', 'Июн', 
                   'Июл', 'Авг', 'Сен', 'Окт', 'Ноя', 'Дек']
    
    for i, date in enumerate(forecast_dates):
        row = {
            'Месяц': months_name[date.month],
            'Ensemble_SA': f"{data['forecasts']['Ensemble'][i]:.2f}",
            'Ensemble_Итог': f"{final_forecasts['Ensemble'][i]:.2f}",
            'Сезонность': f"{data['seasonal_adjustments'].get(date.month, 0):+.2f}"
        }
        rows.append(row)
    
    df = pd.DataFrame(rows)
    print("\n📋 Итоговая таблица:")
    print(df.to_string(index=False))
    df.to_csv('data/forecasts_final.csv', index=False)
    print("\n✅ CSV сохранён: data/forecasts_final.csv")

if __name__ == '__main__':
    add_seasonality_correct()
