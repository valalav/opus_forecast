"""
Экзогенные переменные: загрузка фактов и AR(1) прогноз.
Работает с абсолютными значениями (USD в рублях, ставки в %).
"""
import pandas as pd
import json
import os
from datetime import datetime

MANUAL_EXOG_FILE = 'manual_exog_forecast.json'
EXOG_SETTINGS_FILE = 'exog_settings.json'

def load_last_exog_values():
    """Загрузить последние фактические значения."""
    try:
        with open(EXOG_SETTINGS_FILE, 'r') as f:
            settings = json.load(f)
            return {
                'usd': settings.get('USD_RUB_CURRENT', 77.0),
                'key_rate': settings.get('KEY_RATE_CURRENT', 16.5),
                'ruonia': settings.get('RUONIA_CURRENT', 16.25)
            }
    except:
        return {'usd': 77.0, 'key_rate': 16.5, 'ruonia': 16.25}

def generate_auto_exog_forecast(last_date, horizon, current_usd=None, current_keyrate=None, current_ruonia=None):
    """AR(1) прогноз экзогенных от фактических значений."""
    
    actual = load_last_exog_values()
    
    if current_usd is None:
        current_usd = actual['usd']
    if current_keyrate is None:
        current_keyrate = actual['key_rate']
    if current_ruonia is None:
        current_ruonia = actual['ruonia']
    
    dates = pd.date_range(start=last_date + pd.DateOffset(months=1), periods=horizon, freq='MS')
    
    # AR(1) с возвратом к среднему
    # USD: стабильный около текущего уровня
    usd_target = 75.0
    usd_speed = 0.02
    
    # Ключевая ставка: постепенное снижение к нейтральной
    kr_target = 10.0  # Долгосрочный ориентир
    kr_speed = 0.04   # ~0.3 п.п. в месяц при текущем уровне
    
    usd_forecast = []
    keyrate_forecast = []
    ruonia_forecast = []
    
    usd_prev = current_usd
    kr_prev = current_keyrate
    
    for i in range(horizon):
        usd_new = usd_prev + usd_speed * (usd_target - usd_prev)
        usd_forecast.append(round(usd_new, 2))
        usd_prev = usd_new
        
        kr_new = kr_prev + kr_speed * (kr_target - kr_prev)
        keyrate_forecast.append(round(kr_new, 2))
        kr_prev = kr_new
        
        # RUONIA обычно чуть ниже ключевой
        ruonia_new = kr_new - 0.25
        ruonia_forecast.append(round(ruonia_new, 2))
    
    return {
        'dates': [d.strftime('%Y-%m-%d') for d in dates],
        'USD': usd_forecast,
        'KeyRate': keyrate_forecast,
        'RUONIA': ruonia_forecast,
        'source': 'auto',
        'base_values': {
            'USD': current_usd,
            'KeyRate': current_keyrate,
            'RUONIA': current_ruonia
        }
    }

def calc_index_from_levels(levels):
    """Пересчёт абсолютных значений в индексы м/м (100 = без изменений)."""
    indices = [100.0]  # Первый месяц - база
    for i in range(1, len(levels)):
        idx = (levels[i] / levels[i-1]) * 100
        indices.append(round(idx, 2))
    return indices

def load_manual_exog():
    if os.path.exists(MANUAL_EXOG_FILE):
        try:
            with open(MANUAL_EXOG_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except:
            pass
    return None

def save_manual_exog(data):
    data['updated_at'] = datetime.now().isoformat()
    data['source'] = 'manual'
    with open(MANUAL_EXOG_FILE, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2, default=str)

if __name__ == '__main__':
    actual = load_last_exog_values()
    print("=== Фактические значения ===")
    print(f"USD: {actual['usd']} руб")
    print(f"Ключевая ставка: {actual['key_rate']}%")
    print(f"RUONIA: {actual['ruonia']}%")
    
    print("\n=== Авто-прогноз AR(1) на 12 мес ===")
    forecast = generate_auto_exog_forecast(pd.Timestamp('2025-11-01'), 12)
    for i, d in enumerate(forecast['dates']):
        print(f"{d}: USD={forecast['USD'][i]:5.1f}, КС={forecast['KeyRate'][i]:5.2f}%, RUONIA={forecast['RUONIA'][i]:5.2f}%")
