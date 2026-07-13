import pandas as pd
import numpy as np

# Загружаем данные
df = pd.read_csv('data/inflation_data.csv', sep=';', decimal=',')

# Смотрим последние значения
print("=== Последние 5 строк ===")
print(df[['Date', 'usd_nom_i', 'Ki_i', 'Ruonia']].tail(10))

# RUONIA - это прямо ставка в %
print(f"\nПоследняя RUONIA: {df['Ruonia'].iloc[-1]}%")

# usd_nom_i - индекс (100 = без изменений)
# Нужно восстановить уровень курса
# Предположим базовый курс на начало периода ~30 руб
base_usd = 30.0  # руб/$ на январь 2010

# Восстанавливаем курс кумулятивно
df['usd_level'] = base_usd * (df['usd_nom_i'] / 100).cumprod()
print(f"\nРасчётный курс USD на последнюю дату: {df['usd_level'].iloc[-1]:.2f} руб")

# Смотрим последние курсы
print("\n=== Расчётный курс USD (последние 12 мес) ===")
print(df[['Date', 'usd_nom_i', 'usd_level']].tail(12))
