import pandas as pd
import subprocess
import os
import io

def extract_all_regions_data():
    print("1. Экспорт названий регионов...")
    # Export regions names
    subprocess.run("mdb-export data/db_cpi_store.accdb regions_names > data/regions_names.csv", shell=True, check=True)
    regions = pd.read_csv('data/regions_names.csv')
    # Filter out "Russia" (code 0) if needed, usually code 0 is total
    print(f"   Найдено {len(regions)} регионов.")

    print("2. Экспорт индексов (это может занять время)...")
    # We use mdb-export and pipe to python to filter on the fly to save disk/ram
    # Item codes for aggregates (from previous analysis):
    # 1: Все товары и услуги
    # 2: Непродовольственные товары
    # 3: Продовольственные товары
    # 4: Услуги
    # 33: Плодоовощная (для интереса)
    target_items = {1, 2, 3, 4, 33}
    
    cmd = ["mdb-export", "data/db_cpi_store.accdb", "data_indices"]
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    
    chunk_size = 100000
    chunks = []
    
    # Read using pandas in chunks
    # Columns in Access: Code, Day, Region_code, Item_code, MoM, YoY, Calc
    # We need: Day, Region_code, Item_code, MoM
    
    try:
        for chunk in pd.read_csv(process.stdout, sep=',', chunksize=chunk_size):
            # Filter
            filtered = chunk[chunk['Item_code'].isin(target_items)].copy()
            if not filtered.empty:
                # Keep only necessary columns
                subset = filtered[['Day', 'Region_code', 'Item_code', 'MoM']]
                chunks.append(subset)
                
    except Exception as e:
        print(f"Ошибка чтения: {e}")
        
    print("3. Объединение и сохранение...")
    if not chunks:
        print("Нет данных.")
        return

    full_df = pd.concat(chunks)
    
    # Clean Dates
    full_df['Date'] = pd.to_datetime(full_df['Day'], format='%m/%d/%y %H:%M:%S', errors='coerce')
    full_df = full_df.dropna(subset=['Date'])
    
    # Save raw long format
    full_df.to_csv('data/all_regions_indices.csv', index=False)
    print(f"   Сохранено {len(full_df)} строк в data/all_regions_indices.csv")
    
    return full_df

if __name__ == "__main__":
    extract_all_regions_data()
