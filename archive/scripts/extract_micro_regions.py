import pandas as pd
import subprocess
import os

def extract_micro_regions():
    print("1. Извлечение микро-данных по всем регионам...")
    
    # Item Codes for Micro-Analysis (Top by weight/volatility)
    # 10: Мясопродукты
    # 1100: Молоко и молочная продукция
    # 21: Плодоовощная продукция (повторно)
    # 42: Топливо моторное
    # 1700: Кондитерские изделия
    # 4700: Обувь
    # 9400: ЖКУ
    # 7400: Стройматериалы
    
    target_items = {10, 1100, 21, 42, 1700, 4700, 9400, 7400}
    
    cmd = ["mdb-export", "data/db_cpi_store.accdb", "data_indices"]
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    
    chunk_size = 100000
    chunks = []
    
    try:
        for chunk in pd.read_csv(process.stdout, sep=',', chunksize=chunk_size):
            filtered = chunk[chunk['Item_code'].isin(target_items)].copy()
            if not filtered.empty:
                subset = filtered[['Day', 'Region_code', 'Item_code', 'MoM']]
                chunks.append(subset)
                
    except Exception as e:
        print(f"Ошибка чтения: {e}")
        
    print("2. Объединение...")
    if not chunks:
        print("Нет данных.")
        return

    full_df = pd.concat(chunks)
    
    # Clean Dates
    full_df['Date'] = pd.to_datetime(full_df['Day'], format='%m/%d/%y %H:%M:%S', errors='coerce')
    full_df = full_df.dropna(subset=['Date'])
    
    # Save
    full_df.to_csv('data/all_regions_micro.csv', index=False)
    print(f"   Сохранено {len(full_df)} строк в data/all_regions_micro.csv")

if __name__ == "__main__":
    extract_micro_regions()
