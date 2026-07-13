#!/usr/bin/env python3
"""
Сравнение гладких vs сезонных прогнозов
"""

import json
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
from pathlib import Path

def create_comparison_chart():
    """Создать сравнительный график"""
    
    # Загружаем данные
    with open('data/precomputed_forecasts_seasonal.json', 'r') as f:
        data = json.load(f)
    
    dates = [datetime.strptime(d, '%Y-%m-%d') for d in data['forecast_dates']]
    months = [d.strftime('%b') for d in dates]
    
    # Берём ключевые модели
    models = ['Ridge', 'Huber', 'Ensemble']
    colors = {'Ridge': '#1f77b4', 'Huber': '#2ca02c', 'Ensemble': '#d62728'}
    
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=['📊 Гладкие ML-прогнозы (исходные)', '📈 Сезонно-корректированные прогнозы'],
        vertical_spacing=0.15
    )
    
    # График 1: Гладкие прогнозы
    for model in models:
        values = data['forecasts'][model]
        fig.add_trace(
            go.Scatter(
                x=months, y=values,
                name=f'{model} (гладкий)',
                line=dict(color=colors[model], width=2, dash='dash'),
                mode='lines+markers'
            ),
            row=1, col=1
        )
    
    # График 2: Сезонные прогнозы
    for model in models:
        sa_values = data['forecasts_seasonal'][f'{model}_SA']
        fig.add_trace(
            go.Scatter(
                x=months, y=sa_values,
                name=f'{model} (сезонный)',
                line=dict(color=colors[model], width=3),
                mode='lines+markers'
            ),
            row=2, col=1
        )
    
    # Добавляем аннотации с сезонными корректировками
    seasonal = data['seasonal_adjustments']
    for i, month_num in enumerate([d.month for d in dates[:6]]):  # Первые 6 месяцев
        adj = seasonal.get(str(month_num), 0)
        if abs(adj) > 0.3:
            fig.add_annotation(
                x=months[i], y=max(sa_values[i] for sa_values in 
                                   [data['forecasts_seasonal'][f'{m}_SA'] for m in models]),
                text=f'{adj:+.2f}%',
                showarrow=True,
                arrowhead=2,
                row=2, col=1
            )
    
    fig.update_layout(
        title='Сравнение прогнозов: гладкие vs сезонно-корректированные',
        height=800,
        showlegend=True,
        legend=dict(orientation='h', yanchor='bottom', y=-0.15, xanchor='center', x=0.5),
        hovermode='x unified'
    )
    
    fig.update_yaxes(title_text='MoM %', row=1, col=1)
    fig.update_yaxes(title_text='MoM %', row=2, col=1)
    
    # Сохраняем
    output_path = 'assets/charts/seasonal_comparison.html'
    fig.write_html(output_path)
    print(f'✅ График сохранён: {output_path}')
    
    # Создаём таблицу сравнения
    create_comparison_table(data, months)
    
    return output_path

def create_comparison_table(data, months):
    """Создать таблицу сравнения"""
    
    rows = []
    for i, month in enumerate(months):
        row = {'Месяц': month}
        
        # Ensemble
        smooth = data['forecasts']['Ensemble'][i]
        seasonal = data['forecasts_seasonal']['Ensemble_SA'][i]
        row['Ensemble_гладкий'] = f'{smooth:.2f}'
        row['Ensemble_сезонный'] = f'{seasonal:.2f}'
        row['Разница'] = f'{seasonal-smooth:+.2f}'
        
        rows.append(row)
    
    df = pd.DataFrame(rows)
    print('\n📋 Сравнительная таблица (Ensemble):')
    print(df.to_string(index=False))
    
    df.to_csv('data/ensemble_comparison.csv', index=False)
    print('\n✅ Таблица сохранена: data/ensemble_comparison.csv')

if __name__ == '__main__':
    create_comparison_chart()
