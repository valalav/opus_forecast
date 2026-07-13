import pandas as pd
import os

def convert_access_dump_to_model_format():
    print("Преобразование данных из Access...")
    
    # 1. Загружаем индексы КБР (Region_code = 7)
    try:
        df_kbr = pd.read_csv('data/kbr_indices.csv')
        items = pd.read_csv('data/items_names.csv')
    except FileNotFoundError:
        print("ОШИБКА: Файлы kbr_indices.csv или items_names.csv не найдены.")
        return

    # 2. Мэппинг кодов товаров
    # Нам нужны:
    # 1: Все товары и услуги
    # 3: Продовольственные товары (из Access ID может отличаться, проверяем items_names)
    # 2: Непродовольственные товары
    # 4: Услуги
    
    # Создаем словарь {Item_code: Item_name}
    item_map = dict(zip(items['Item_code'], items['Item_name']))
    
    # Добавляем названия товаров в индексы
    df_kbr['Товар'] = df_kbr['Item_code'].map(item_map)
    
    # Фильтруем только нужные агрегаты
    required_items = [
        'Все товары и услуги',
        'Продовольственные товары',
        'Непродовольственные товары',
        'Услуги'
    ]
    
    df_filtered = df_kbr[df_kbr['Товар'].isin(required_items)].copy()
    
    if df_filtered.empty:
        print("Внимание: Не найдены основные агрегаты по названиям. Проверьте items_names.csv")
        # Fallback по кодам (если названия отличаются)
        # 1: Все, 3: Прод, 2: Непрод, 4: Услуги (по дампу выше)
        code_map = {
            1: 'Все товары и услуги',
            3: 'Продовольственные товары',
            2: 'Непродовольственные товары',
            4: 'Услуги'
        }
        df_kbr['Товар'] = df_kbr['Item_code'].map(code_map)
        df_filtered = df_kbr[df_kbr['Item_code'].isin(code_map.keys())].copy()
    
    # 3. Форматирование даты
    # В Access формат: 01/01/10 00:00:00 (MM/DD/YY)
    df_filtered['Date'] = pd.to_datetime(df_filtered['Day'], format='%m/%d/%y %H:%M:%S')
    
    # 4. Формируем финальную таблицу
    # Нужные колонки: Day (dd.mm.yyyy), Товар, MoM (число с запятой или точкой)
    
    df_output = df_filtered[['Date', 'Товар', 'MoM']].copy()
    df_output['Day'] = df_output['Date'].dt.strftime('%d.%m.%Y')
    
    # Access дает точку, модель ест и то и то, но для совместимости со старым форматом:
    # Оставим точку, dashboard.py умеет читать точку.
    
    # Сохраняем
    output_path = 'data/infl_kbr_new.csv'
    df_output.to_csv(output_path, sep=';', index=False, decimal='.')
    
    print(f"Данные сохранены в {output_path}")
    print(f"Строк: {len(df_output)}")
    print(f"Период: {df_output['Date'].min()} - {df_output['Date'].max()}")
    
    # Проверка
    pivot = df_output.pivot_table(index='Date', columns='Товар', values='MoM', aggfunc='first')
    print("\nПоследние данные:")
    print(pivot.tail())

if __name__ == "__main__":
    convert_access_dump_to_model_format()
