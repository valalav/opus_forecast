#!/usr/bin/env python3
"""
Итоговые прогнозы с сезонностью
"""

import json
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime

def create_final_chart():
    with open('data/precomputed_forecasts_v2.json', 'r') as f:
        data = json.load(f)
    
    dates = [datetime.strptime(d, '%Y-%m-%d') for d in data['forecast_dates']]
    months = [d.strftime('%b') for d in dates]
    
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=['SA прогнозы (без сезонности)', 'Итоговые прогнозы (с сезонностью)'],
        vertical_spacing=0.12
    )
    
    colors = {'Ridge': '#1f77b4', 'Huber': '#2ca02c', 'Ensemble': '#d62728'}
    
    # SA прогнозы
    for model in ['Ridge', 'Huber', 'Ensemble']:
        values = data['forecasts'][model]
        fig.add_trace(
            go.Scatter(x=months, y=values, name=f'{model} SA',
                      line=dict(color=colors[model], width=2, dash='dash'),
                      mode='lines+markers'),
            row=1, col=1
        )
    
    # Сезонные прогнозы
    for model in ['Ridge', 'Huber', 'Ensemble']:
        values = data['forecasts_with_seasonality'][model]
        fig.add_trace(
            go.Scatter(x=months, y=values, name=f'{model} (итог)',
                      line=dict(color=colors[model], width=3),
                      mode='lines+markers'),
            row=2, col=1
        )
    
    # Аннотации для сезонных пиков
    seasonal = data['seasonal_adjustments']
    ensemble_final = data['forecasts_with_seasonality']['Ensemble']
    
    for i, (month, date) in enumerate(zip(months, dates)):
        adj = seasonal.get(str(date.month), 0)
        if abs(adj) > 0.3:  # Только значимые отклонения
            direction = "📉" if adj < 0 else "📈"
            fig.add_annotation(
                x=month, y=ensemble_final[i],
                text=f'{direction} {adj:+.2f}%',
                showarrow=True, arrowhead=2, arrowsize=1,
                row=2, col=1
            )
    
    fig.update_layout(
        title='Прогнозы инфляции КБР: SA + сезонность',
        height=800,
        hovermode='x unified',
        legend=dict(orientation='h', yanchor='bottom', y=-0.2, xanchor='center', x=0.5)
    )
    fig.update_yaxes(title_text='MoM %', row=1, col=1)
    fig.update_yaxes(title_text='MoM %', row=2, col=1)
    
    output = 'assets/charts/final_forecasts_seasonal.html'
    fig.write_html(output)
    print(f'✅ График сохранён: {output}')
    
    # Показать ключевые точки
    print("\n📊 Ключевые моменты итогового прогноза (Ensemble):")
    for i, date in enumerate(dates):
        month = date.strftime('%b')
        val = ensemble_final[i]
        sa = data['forecasts']['Ensemble'][i]
        adj = seasonal.get(str(date.month), 0)
        if abs(adj) > 0.2 or i < 3:
            note = ""
            if adj < -0.3: note = "← плодоовощная дефляция"
            elif adj > 0.3: note = "← высокий сезон"
            print(f"  {month}: {val:.2f}% (SA {sa:.2f} + сезон {adj:+.2f}) {note}")

if __name__ == '__main__':
    create_final_chart()
