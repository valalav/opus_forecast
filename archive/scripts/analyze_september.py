import pandas as pd

def analyze_september():
    try:
        # Загружаем данные
        df = pd.read_csv('data/inflation_data.csv', sep=';', decimal=',')
        
        # Исправляем формат дат
        df['Date'] = pd.to_datetime(df['Date'], format='%d.%m.%Y', errors='coerce')
        if df['Date'].isna().any():
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce') # Fallback
            
        df = df.set_index('Date').sort_index()
        
        # Смотрим последние данные за август-сентябрь-октябрь
        target_dates = df.index[df.index.month.isin([8, 9, 10]) & (df.index.year == 2025)]
        
        if len(target_dates) == 0:
            # Если 2025 нет, смотрим 2024 (вдруг речь о прошлом годе)
            target_dates = df.index[df.index.month.isin([8, 9, 10]) & (df.index.year == 2024)]
            print("Данные за 2025 год не найдены, анализируем 2024:")
        else:
            print("Анализ данных за 2025 год:")

        subset = df.loc[target_dates, ['mom', 'Prod', 'Nonprod', 'Serv', 'usd_nom_i']]
        
        # Переводим индексы в % изменения (если они > 50, значит это индексы типа 100.5)
        for col in subset.columns:
            if subset[col].mean() > 50:
                subset[col] = subset[col] - 100
                
        print(subset)
        
        # Считаем вклад (грубо, если весов нет, просто смотрим на величину скачка)
        print("\nОтклонения от среднего за год:")
        yearly_mean = df.loc[df.index.year == target_dates[0].year, ['mom', 'Prod', 'Nonprod', 'Serv']].mean() - 100
        print(yearly_mean)

    except Exception as e:
        print(f"Ошибка анализа: {e}")

analyze_september()
