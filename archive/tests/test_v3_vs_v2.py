import pandas as pd
import numpy as np
from sirena_kbr_v2_4_auto import SirenaKBR_v24
from sirena_kbr_v3 import SirenaKBR_v3
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings

warnings.filterwarnings('ignore')

def run_comparison():
    print("="*60)
    print("СРАВНЕНИЕ МОДЕЛЕЙ: v2.4 (Aggregate) vs v3.0 (Component-wise)")
    print("="*60)
    
    # 1. Загрузка данных
    # Для v2.4 нужен агрегированный формат
    try:
        df_agg = pd.read_csv('data/infl_kbr.csv', sep=';', decimal='.')
        if 'Day' in df_agg.columns:
             df_agg['Date'] = pd.to_datetime(df_agg['Day'], format='%d.%m.%Y')
        elif 'Date' in df_agg.columns:
             df_agg['Date'] = pd.to_datetime(df_agg['Date'])
             
        # Если файл в длинном формате (после обновления из Access), пивотим
        if 'Товар' in df_agg.columns and 'MoM' in df_agg.columns:
             df_agg = df_agg.pivot_table(index='Date', columns='Товар', values='MoM', aggfunc='first')
        
        df_agg = df_agg.sort_index()
        # Для v2.4 нужны колонки: ['Все товары и услуги', 'Продовольственные товары', 'Непродовольственные товары', 'Услуги']
        # Убедимся, что они есть
        required_v2 = ['Все товары и услуги', 'Продовольственные товары', 'Непродовольственные товары', 'Услуги']
        if not all(c in df_agg.columns for c in required_v2):
            print(f"Ошибка: v2.4 требует колонки {required_v2}")
            return

    except Exception as e:
        print(f"Ошибка загрузки infl_kbr.csv: {e}")
        return

    # Для v3.0 нужен детальный формат
    try:
        df_det = pd.read_csv('data/infl_kbr_detailed.csv', sep=';', decimal='.')
        df_det['Date'] = pd.to_datetime(df_det['Date'])
        df_det = df_det.set_index('Date').sort_index()
    except Exception as e:
        print(f"Ошибка загрузки infl_kbr_detailed.csv: {e}")
        return
        
    # 2. Настройка бэктеста
    start_date = '2019-01-01'
    # Общий период (пересечение индексов)
    common_index = df_agg.index.intersection(df_det.index)
    test_dates = [d for d in common_index if d >= pd.Timestamp(start_date)]
    
    if not test_dates:
        print("Нет общих дат для тестирования.")
        return
        
    print(f"Период тестирования: {test_dates[0].strftime('%Y-%m')} — {test_dates[-1].strftime('%Y-%m')}")
    print(f"Количество точек: {len(test_dates)}")
    
    results_v2 = []
    results_v3 = []
    
    model_v2 = SirenaKBR_v24()
    model_v3 = SirenaKBR_v3()
    
    for i, target_date in enumerate(test_dates):
        if (i+1) % 12 == 0: print(f"  Обработано {i+1}...")
        
        cutoff = target_date - pd.DateOffset(months=1)
        
        # --- Тест v2.4 ---
        train_v2 = df_agg[df_agg.index <= cutoff].copy()
        if len(train_v2) >= 36:
            try:
                model_v2.fit(train_v2)
                # Для прогноза передаем "будущий" фрейм (в v2.4 predict берет лаги из переданного df)
                # Нам нужно, чтобы в df были данные до cutoff, а predict сам найдет лаги для target_date
                # Но v2.4 predict(df, date) ищет строку target_date в df.
                # В бэктесте мы не должны видеть факт target_date.
                # SirenaKBR_v24.predict использует лаги. Лаги для target_date находятся в cutoff.
                # Мы можем передать df_agg[index <= cutoff], но нужно добавить пустую строку для target_date?
                # Посмотрим код v2.4.
                # v2.4: df['y_lag1'] = df['...'].shift(1).
                # Если мы передадим df с данными по cutoff, то последняя строка - cutoff.
                # shift(1) для target_date (которого нет в индексе) не сработает.
                # Нам нужно создать временный df с индексом target_date.
                
                train_v2_ext = train_v2.copy()
                train_v2_ext.loc[target_date] = np.nan # Placeholder
                
                res_v2 = model_v2.predict(train_v2_ext, target_date)
                results_v2.append({
                    'date': target_date,
                    'actual': df_agg.loc[target_date, 'Все товары и услуги'],
                    'pred': res_v2['prediction']
                })
            except Exception as e:
                # print(f"Err v2: {e}")
                pass

        # --- Тест v3.0 ---
        train_v3 = df_det[df_det.index <= cutoff].copy()
        if len(train_v3) >= 36:
            try:
                model_v3.fit(train_v3)
                # v3 predict_next: прогнозирует на 1 шаг вперед от конца переданного df
                # Если мы передали train_v3 (до cutoff), то следующий месяц - target_date.
                # Идеально.
                res_v3_df = model_v3.predict_next(train_v3, horizon=1)
                
                # res_v3_df содержит 1 строку
                pred_val = res_v3_df.iloc[0]['MoM_Index']
                
                results_v3.append({
                    'date': target_date,
                    'actual': df_agg.loc[target_date, 'Все товары и услуги'],
                    'pred': pred_val
                })
            except Exception as e:
                 # print(f"Err v3: {e}")
                 pass

    # 3. Сравнение метрик
    res_v2_df = pd.DataFrame(results_v2)
    res_v3_df = pd.DataFrame(results_v3)
    
    # Синхронизируем (берем только даты, где обе модели дали прогноз)
    common_dates = set(res_v2_df['date']).intersection(set(res_v3_df['date']))
    
    v2_final = res_v2_df[res_v2_df['date'].isin(common_dates)].sort_values('date')
    v3_final = res_v3_df[res_v3_df['date'].isin(common_dates)].sort_values('date')
    
    print("\nРЕЗУЛЬТАТЫ:")
    print("-" * 40)
    
    for name, df_res in [("v2.4 (Agg)", v2_final), ("v3.0 (Comp)", v3_final)]:
        error = df_res['actual'] - df_res['pred']
        mae = error.abs().mean()
        rmse = np.sqrt((error**2).mean())
        kpi = (error.abs() <= 0.5).sum()
        print(f"{name}: MAE={mae:.4f}, RMSE={rmse:.4f}, KPI={kpi}/{len(df_res)} ({kpi/len(df_res)*100:.1f}%)")
        
    print("-" * 40)
    
    # Анализ по годам (2022, 2024, 2025)
    df_res = v2_final[['date', 'actual']].copy()
    df_res['err_v2'] = (v2_final['actual'] - v2_final['pred']).abs()
    df_res['err_v3'] = (v3_final['actual'] - v3_final['pred']).abs()
    df_res['year'] = df_res['date'].dt.year
    
    print("\nMAE по годам:")
    print(f"{ 'Год':<6} {'v2.4':<10} {'v3.0':<10} {'Победитель'}")
    for y in sorted(df_res['year'].unique()):
        sub = df_res[df_res['year'] == y]
        m2 = sub['err_v2'].mean()
        m3 = sub['err_v3'].mean()
        win = "v3.0" if m3 < m2 else "v2.4"
        print(f"{y:<6} {m2:.4f}     {m3:.4f}     {win}")

if __name__ == "__main__":
    run_comparison()
