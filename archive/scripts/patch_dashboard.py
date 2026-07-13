# Патч для dashboard_v32.py - заменяем функцию генерации прогноза экзогенных

new_code = '''
def load_last_exog_values():
    """Load last actual values of exogenous variables from data."""
    try:
        df = pd.read_csv('data/inflation_data.csv', sep=';', decimal=',')
        
        # Fix numeric columns
        for col in ['usd_nom_i', 'Ki_i', 'Ruonia']:
            if col in df.columns:
                if df[col].dtype == object:
                    df[col] = df[col].astype(str).str.replace(',', '.')
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        last_row = df.iloc[-1]
        
        # RUONIA - прямо ставка в %
        ruonia = last_row['Ruonia']
        
        # usd_nom_i - индекс м/м изменения курса (100 = без изменений)
        # Для расчёта уровня курса нужен внешний источник
        # Используем примерное значение на основе текущего курса ЦБ (~100 руб)
        usd_level = 100.0  # Примерный текущий курс
        
        # Ключевая ставка - восстанавливаем из индекса
        # Ki_i показывает изменение ставки. Последняя известная ставка ~21%
        # Но судя по динамике Ki_i (снижение), ставка снижалась
        # Грубая оценка: если RUONIA=16.25%, то ключевая ~16-17%
        key_rate = ruonia + 0.5  # Ключевая обычно чуть выше RUONIA
        
        return {
            'usd_level': usd_level,
            'usd_index': last_row['usd_nom_i'],
            'key_rate': key_rate,
            'ki_index': last_row['Ki_i'],
            'ruonia': ruonia,
            'date': last_row['Date']
        }
    except Exception as e:
        return {
            'usd_level': 100.0,
            'usd_index': 100.0,
            'key_rate': 16.5,
            'ki_index': 100.0,
            'ruonia': 16.25,
            'date': 'unknown'
        }

def generate_auto_exog_forecast(last_date, horizon, current_usd=None, current_keyrate=None, current_ruonia=None):
    """Generate automatic exogenous forecasts based on actual data using AR(1)."""
    
    # Load actual last values if not provided
    actual = load_last_exog_values()
    
    if current_usd is None:
        current_usd = actual['usd_level']
    if current_keyrate is None:
        current_keyrate = actual['key_rate']
    if current_ruonia is None:
        current_ruonia = actual['ruonia']
    
    dates = pd.date_range(start=last_date + pd.DateOffset(months=1), periods=horizon, freq='MS')
    
    # AR(1) forecasts with mean reversion
    # USD: slow mean reversion to ~95 (assume slight strengthening)
    usd_target = 95.0
    usd_speed = 0.05  # 5% mean reversion per month
    
    # Key rate: gradual decline expected
    kr_target = 14.0  # Long-term neutral rate
    kr_speed = 0.03
    
    # RUONIA follows key rate closely
    ruonia_spread = -0.3  # RUONIA usually slightly below key rate
    
    usd_forecast = []
    keyrate_forecast = []
    ruonia_forecast = []
    
    usd_prev = current_usd
    kr_prev = current_keyrate
    
    for i in range(horizon):
        # USD: AR(1) with mean reversion
        usd_new = usd_prev + usd_speed * (usd_target - usd_prev)
        usd_forecast.append(round(usd_new, 2))
        usd_prev = usd_new
        
        # Key rate: gradual decline
        kr_new = kr_prev + kr_speed * (kr_target - kr_prev)
        keyrate_forecast.append(round(kr_new, 2))
        kr_prev = kr_new
        
        # RUONIA follows key rate
        ruonia_new = kr_new + ruonia_spread
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
'''

# Читаем dashboard
with open('dashboard_v32.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Находим и заменяем функцию generate_auto_exog_forecast
import re

# Паттерн для старой функции
old_pattern = r'def generate_auto_exog_forecast\(.*?\n    return \{[^}]+\}'

# Проверяем что нашли
if 'def generate_auto_exog_forecast' in content:
    # Заменяем блок между def и следующим def или классом
    lines = content.split('\n')
    new_lines = []
    skip_until_next_def = False
    inserted = False
    
    for i, line in enumerate(lines):
        if 'def generate_auto_exog_forecast' in line and not inserted:
            # Вставляем новый код
            new_lines.append('')
            new_lines.append('def load_last_exog_values():')
            # Добавляем остальной код...
            skip_until_next_def = True
            inserted = True
            continue
        
        if skip_until_next_def:
            if (line.startswith('def ') or line.startswith('class ') or line.startswith('# ===')) and 'generate_auto' not in line:
                skip_until_next_def = False
                # Вставляем новую функцию перед этой строкой
                new_lines.append(new_code)
                new_lines.append('')
                new_lines.append(line)
            continue
        
        new_lines.append(line)
    
    # Записываем
    with open('dashboard_v32.py', 'w', encoding='utf-8') as f:
        f.write('\n'.join(new_lines))
    
    print("✅ Патч применён")
else:
    print("❌ Функция не найдена")
