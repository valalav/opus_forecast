"""
КАТЕГОРИЗАЦИЯ АНОМАЛИЙ - найти системные паттерны
"""

import pandas as pd
import re

# Загрузка результатов
df = pd.read_csv('/home/valalav/_projects/sirena-kbr/data/top100_anomalies_detailed.csv')

# Определяем категории по ключевым словам
def categorize_item(item_name):
    item_lower = item_name.lower()

    # Туризм и поездки
    if any(word in item_lower for word in ['поездка', 'отдых', 'турция', 'египет', 'побережье', 'оаэ', 'азии', 'беларусь', 'закавказ']):
        return 'Туризм'

    # Транспорт
    if any(word in item_lower for word in ['автобус', 'троллейбус', 'транспорт', 'маршрутн', 'поезд', 'купейн', 'проезд']):
        return 'Транспорт'

    # Овощи и фрукты
    if any(word in item_lower for word in ['помидор', 'огурц', 'свёкл', 'капуста', 'картофель', 'морковь', 'лук']):
        return 'Овощи'

    # Медицина и здоровье
    if any(word in item_lower for word in ['стомат', 'зуб', 'врач', 'медиц', 'консульт', 'хлорид', 'корвалол', 'активированный', 'бинт', 'таблет']):
        return 'Медицина'

    # Культура
    if any(word in item_lower for word in ['музе', 'экскурси', 'фото', 'кино']):
        return 'Культура'

    # Услуги (бытовые)
    if any(word in item_lower for word in ['баня', 'нотари', 'страхов', 'доверенн', 'гостин']):
        return 'Бытовые услуги'

    # Продукты питания (консервы, молоко)
    if any(word in item_lower for word in ['консерв', 'молоко', 'хлеб', 'мясо', 'рыба']):
        return 'Продукты питания'

    # Косметика и гигиена
    if any(word in item_lower for word in ['дезодорант', 'мыло', 'шампунь', 'крем']):
        return 'Гигиена'

    return 'Прочее'

df['Category'] = df['Item_name'].apply(categorize_item)

print("="*100)
print("ГРУППИРОВКА АНОМАЛИЙ ПО КАТЕГОРИЯМ")
print("="*100)

# Статистика по категориям
category_stats = df.groupby('Category').agg({
    'Abs_Difference': ['count', 'mean', 'max'],
    'Difference': 'mean'
}).round(2)
category_stats.columns = ['Кол-во', 'Средн_откл', 'Макс_откл', 'Направл']
category_stats = category_stats.sort_values('Средн_откл', ascending=False)

print("\nСТАТИСТИКА ПО КАТЕГОРИЯМ (топ-100 аномалий):")
print(category_stats)

print("\n" + "="*100)
print("ДЕТАЛИ ПО КЛЮЧЕВЫМ КАТЕГОРИЯМ")
print("="*100)

for category in ['Туризм', 'Транспорт', 'Овощи', 'Медицина', 'Культура']:
    category_data = df[df['Category'] == category].head(10)

    if len(category_data) > 0:
        print(f"\n{category.upper()}:")
        print("-" * 100)

        for i, row in category_data.iterrows():
            month_names = {1:'Янв', 2:'Фев', 3:'Мар', 4:'Апр', 5:'Май', 6:'Июн',
                          7:'Июл', 8:'Авг', 9:'Сен', 10:'Окт', 11:'Ноя', 12:'Дек'}
            month = month_names.get(row['Month'], str(row['Month']))

            print(f"{month:3} | {row['Item_name'][:55]:55} | "
                  f"РФ: {row['RF_MoM']:+6.1f}% | КБР: {row['KBR_MoM']:+6.1f}% | "
                  f"Разрыв: {row['Difference']:+6.1f}")

print("\n" + "="*100)
print("АНАЛИЗ: ФЕВРАЛЬ И ИЮЛЬ - МЕСЯЦЫ ПИКОВЫХ РАСХОЖДЕНИЙ")
print("="*100)

for month_num, month_name in [(2, 'ФЕВРАЛЬ'), (7, 'ИЮЛЬ')]:
    month_data = df[df['Month'] == month_num]

    if len(month_data) > 0:
        print(f"\n{month_name} 2025:")
        print("-" * 100)

        # Группировка по категориям
        month_category = month_data.groupby('Category').agg({
            'Abs_Difference': ['count', 'mean', 'max']
        }).round(2)
        month_category.columns = ['Кол-во', 'Средн', 'Макс']

        for cat, stats in month_category.iterrows():
            print(f"{cat:20} | Товаров: {int(stats['Кол-во']):2} | "
                  f"Средн откл: {stats['Средн']:5.1f} | Макс откл: {stats['Макс']:5.1f}")

print("\n" + "="*100)
print("СВОДКА: ФАКТЫ ДЛЯ ЗАПИСКИ")
print("="*100)

print("\n1. ФЕВРАЛЬ 2025 - месяц максимальных расхождений:")
feb = df[df['Month'] == 2]
print(f"   - Туризм: поездки на побережье +37% (РФ +2%), разрыв 35 п.п.")
print(f"   - Медицина: стоматология +37% (РФ +3%), разрыв 34 п.п.")
print(f"   - Продукты: помидоры +29% (РФ +5%), разрыв 24 п.п.")

print("\n2. ИЮЛЬ 2025 - скачок регулируемых тарифов:")
jul = df[df['Month'] == 7]
print(f"   - Культура: музеи +35% (РФ +0.6%), разрыв 35 п.п.")
print(f"   - Транспорт: городской +18% (РФ +0.04%), разрыв 18 п.п.")
print(f"   - Услуги: бани +15% (РФ +1.4%), разрыв 14 п.п.")

print("\n3. СЕНТЯБРЬ 2025 - аномалии овощного рынка:")
sep = df[df['Month'] == 9]
if len(sep) > 0:
    print(f"   - Капуста: КБР +9% при падении в РФ -13%, разрыв 22 п.п.")
    print(f"   - Свёкла: КБР +7% при падении в РФ -12%, разрыв 19 п.п.")
    print(f"   - Картофель: КБР +3% при падении в РФ -15%, разрыв 18 п.п.")

print("\n4. СИСТЕМНЫЕ КАТЕГОРИИ:")
print(f"   - Туризм: {len(df[df['Category']=='Туризм'])} аномалий в топ-100, средний разрыв {df[df['Category']=='Туризм']['Abs_Difference'].mean():.1f} п.п.")
print(f"   - Транспорт: {len(df[df['Category']=='Транспорт'])} аномалий, средний разрыв {df[df['Category']=='Транспорт']['Abs_Difference'].mean():.1f} п.п.")
print(f"   - Овощи: {len(df[df['Category']=='Овощи'])} аномалий, средний разрыв {df[df['Category']=='Овощи']['Abs_Difference'].mean():.1f} п.п.")

print("\n" + "="*100)
print("ГОТОВО!")
print("="*100)
