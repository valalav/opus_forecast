#!/usr/bin/env python3
"""
Добавление сезонности к прогнозам
================================
Сглаженные ML-прогнозы корректируются историческими сезонными паттернами.
"""

import json
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path

def load_seasonal_pattern():
    """Загрузить сезонные нормы из SA данных"""
    df = pd.read_csv('data/sa_fl.csv', sep=';', decimal=',', encoding='utf-8-sig')
    df = df[df['Товар'] == 'Все товары и услуги'].copy()
    df['Дата'] = pd.to_datetime(df['Дата'], format='%d.%m.%Y')
    df['month'] = df['Дата'].dt.month
    df['mom'] = df['Значение'] - 100
    
    # За последние 24 месяца (исключая выбросы)
    recent = df[df['Дата'] >= '2023-01-01']
    seasonal = recent.groupby('month')['mom'].mean()
    
    # Центрируем (среднее = 0)
    seasonal = seasonal - seasonal.mean()
    
    return seasonal.to_dict()

def add_seasonality_to_forecasts():
    """Добавить сезонность к прогнозам"""
    
    # Загружаем текущие прогнозы
    with open('data/precomputed_forecasts.json', 'r') as f:
        data = json.load(f)
    
    seasonal = load_seasonal_pattern()
    forecast_dates = [datetime.strptime(d, '%Y-%m-%d') for d in data['forecast_dates']]
    
    print("Сезонные корректировки по месяцам:")
    for month, adj in sorted(seasonal.items()):
        month_name = ['', 'Янв', 'Фев', 'Мар', 'Апр', 'Май', 'Июн', 
                      'Июл', 'Авг', 'Сен', 'Окт', 'Ноя', 'Дек'][month]
        print(f"  {month_name}: {adj:+.2f}%")
    
    print("\n" + "="*60)
    print("ПРОГНОЗЫ С СЕЗОННОСТЬЮ")
    print("="*60)
    
    # Создаём новые прогнозы с сезонностью
    seasonal_forecasts = {}
    
    for model, values in data['forecasts'].items():
        if values is None:
            continue
            
        adjusted = []
        for i, (date, val) in enumerate(zip(forecast_dates, values)):
            month = date.month
            seasonal_adj = seasonal.get(month, 0)
            
            # Блендинг: 70% исходный прогноз + 30% сезонная корректировка
            # Но с затуханием сезонности для дальних горизонтов
            decay = max(0.3, 1 - i * 0.05)  # От 1.0 до 0.3
            adjusted_val = val + seasonal_adj * decay * 0.5
            
            adjusted.append(adjusted_val)
        
        seasonal_forecasts[f"{model}_SA"] = adjusted
        
        # Показываем сравнение
        print(f"\n{model}:")
        print(f"  Было:  {[f'{v:.2f}' for v in values[:6]]}")
        print(f"  Стало: {[f'{v:.2f}' for v in adjusted[:6]]}")
    
    # Сохраняем расширенные прогнозы
    data['forecasts_seasonal'] = seasonal_forecasts
    data['seasonal_adjustments'] = seasonal
    
    output_path = 'data/precomputed_forecasts_seasonal.json'
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"\n\n✅ Сезонные прогнозы сохранены: {output_path}")
    
    # Создаём CSV для удобства
    create_seasonal_csv(data, seasonal_forecasts, forecast_dates)
    
    return data

def create_seasonal_csv(data, seasonal_forecasts, forecast_dates):
    """Создать CSV с сезонными прогнозами"""
    
    rows = []
    for i, date in enumerate(forecast_dates):
        row = {'Date': date.strftime('%Y-%m-%d'), 'Month': date.month}
        
        # Оригинальные прогнозы
        for model, values in data['forecasts'].items():
            if values:
                row[model] = values[i]
        
        # Сезонные прогнозы
        for model, values in seasonal_forecasts.items():
            row[model] = values[i]
        
        rows.append(row)
    
    df = pd.DataFrame(rows)
    csv_path = 'data/forecasts_seasonal.csv'
    df.to_csv(csv_path, index=False)
    print(f"✅ CSV сохранён: {csv_path}")

if __name__ == '__main__':
    add_seasonality_to_forecasts()
