import pandas as pd
import os

def extract_detailed_components():
    print("Извлечение детальных компонентов для модели v3.0...")
    
    # 1. Загружаем данные индексов для КБР (Region 7)
    # Мы предполагаем, что kbr_indices.csv уже создан на предыдущем шаге.
    # Если нет, можно пересоздать из data_indices_dump.csv
    if not os.path.exists('data/kbr_indices.csv'):
        print("Загрузка из дампа...")
        try:
             df = pd.read_csv('data/data_indices_dump.csv')
             df_kbr = df[df['Region_code'] == 7].copy()
        except:
             print("Ошибка: нет дампа данных.")
             return
    else:
        df_kbr = pd.read_csv('data/kbr_indices.csv')
        
    # 2. Загружаем веса (data_weights) из базы Access, если возможно
    # Если нет, используем приблизительные веса Росстата 2025:
    # Прод: ~40% (из них Плодоовощи ~5%), Непрод: ~35% (Топливо ~5%), Услуги: ~25% (ЖКХ ~10%)
    
    # 3. Определяем коды нужных компонентов
    # 1: Все товары и услуги
    # 3: Продовольственные товары
    # 2: Непродовольственные товары
    # 4: Услуги
    # 33: Плодоовощная продукция
    # 14: Жилищно-коммунальные услуги
    # 42: Топливо моторное
    
    target_codes = {
        1: 'Все товары и услуги',
        3: 'Продовольственные товары',
        2: 'Непродовольственные товары',
        4: 'Услуги',
        33: 'Плодоовощная продукция',
        14: 'ЖКУ',
        42: 'Топливо'
    }
    
    df_filtered = df_kbr[df_kbr['Item_code'].isin(target_codes.keys())].copy()
    df_filtered['Товар'] = df_filtered['Item_code'].map(target_codes)
    
    # 4. Форматирование
    df_filtered['Date'] = pd.to_datetime(df_filtered['Day'], format='%m/%d/%y %H:%M:%S', errors='coerce')
    # Fallback для формата DD/MM/YY if needed
    if df_filtered['Date'].isna().any():
         df_filtered['Date'] = pd.to_datetime(df_filtered['Day'], format='%d/%m/%y %H:%M:%S', errors='coerce')

    df_output = df_filtered[['Date', 'Товар', 'MoM']].copy()
    df_output['Day'] = df_output['Date'].dt.strftime('%d.%m.%Y')
    
    # 5. Pivot table
    pivot = df_output.pivot_table(index='Date', columns='Товар', values='MoM', aggfunc='first')
    
    # 6. Расчет "Базовой инфляции" (приблизительно)
    # Базовая = Все - Плодоовощи - Топливо - ЖКУ
    # Но корректнее считать: Базовый Прод + Базовый Непрод + Базовые Услуги
    
    # Проверим наличие данных
    print("Доступные столбцы:", pivot.columns.tolist())
    
    # Сохраняем
    pivot.to_csv('data/infl_kbr_detailed.csv', sep=';', decimal='.')
    print(f"Сохранено в data/infl_kbr_detailed.csv. Строк: {len(pivot)}")
    print(pivot.tail())

if __name__ == "__main__":
    extract_detailed_components()
