# utils/seasonal_adjustment_x13.py

import pandas as pd
from pathlib import Path
import sys
import hashlib
import warnings

# --- 1. Настройка путей для корректного импорта ---
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

# Мы по-прежнему используем низкоуровневый драйвер
from utils.x13 import x13_arima_analysis

def get_file_hash(filepath: Path) -> str:
    """Рассчитывает хэш-сумму SHA256 для файла."""
    h = hashlib.sha256()
    with open(filepath, 'rb') as file:
        while True:
            chunk = file.read(h.block_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()

def perform_seasonal_adjustment(df_raw: pd.DataFrame, roles_to_adjust: list) -> pd.DataFrame:
    """
    Выполняет сезонную корректировку для указанных колонок DataFrame.
    [v2.0] Использует интеллектуальное кэширование и округление до 2 знаков.
    """
    print("\n--- [Seasonal Adjustment v2.0] Запуск сезонного сглаживания с кэшированием ---")
    
    # Определяем пути
    data_source_path = ROOT_DIR / "data" / "raw" / "inflation_data.csv"
    cache_dir = ROOT_DIR / "data" / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Проверяем кэш
    try:
        source_hash = get_file_hash(data_source_path)
        cache_file_path = cache_dir / f"sa_cache_{source_hash}.csv"

        if cache_file_path.exists():
            print(f"[OK] Найден актуальный кэш сглаженных данных. Загрузка из файла: {cache_file_path.name}")
            df_sa = pd.read_csv(cache_file_path, index_col=0, parse_dates=True)
            # Возвращаем только те колонки, которые есть в исходном df_raw
            return df_sa[df_raw.columns.intersection(df_sa.columns)]
    except Exception as e:
        print(f"[WARNING] Ошибка при работе с кэшем: {e}. Выполняется полное сглаживание.")
        source_hash = None
    
    # 2. Если кэша нет, выполняем полное сглаживание
    print("Актуальный кэш не найден. Выполняется полное сезонное сглаживание...")
    df_sa = df_raw.copy()
    
    for role in roles_to_adjust:
        if role not in df_sa.columns:
            continue
        
        print(f"  Обработка ряда '{role}'...")
        series_to_adjust = df_sa[role].dropna()
        if len(series_to_adjust) < 36:
            warnings.warn(f"Ряд '{role}' слишком короткий ({len(series_to_adjust)} < 36). Сглаживание пропущено.")
            continue
        if series_to_adjust.nunique() <= 1:
            warnings.warn(f"Ряд '{role}' является константой. Сглаживание пропущено.")
            continue
        
        # Данные для X-13 должны быть положительными, поэтому работаем с индексами
        series_index = (series_to_adjust / 100 + 1).cumprod() * 100

        try:
            # Вызываем X-13
            result = x13_arima_analysis(series_index, series_name=role, log=True)
            
            if result and not result.seasadj.empty and not result.seasadj.isna().all():
                # Возвращаемся к процентным пунктам
                sa_series_pp = result.seasadj.pct_change(fill_method=None) * 100
                # [ОКРУГЛЕНИЕ] Округляем до 2 знаков
                df_sa[role] = sa_series_pp.round(2)
                print(f"  [OK] Ряд '{role}' успешно сглажен и заменен в df_sa.")
            else:
                print(f"  - [WARNING] Не удалось сгладить ряд '{role}'. В итоговом df_sa останется исходный ряд.")
        except Exception as e:
            print(f"  - [ERROR] КРИТИЧЕСКАЯ ОШИБКА при сглаживании '{role}': {e}")
    
    # 3. Сохраняем результат в кэш для будущих запусков
    if source_hash:
        try:
            # Удаляем старые файлы кэша, чтобы не засорять диск
            for old_cache in cache_dir.glob("sa_cache_*.csv"):
                if old_cache.name != cache_file_path.name:
                    old_cache.unlink()
            
            # Сохраняем новый кэш
            df_sa.to_csv(cache_file_path)
            print(f"[OK] Результаты сглаживания сохранены в кэш: {cache_file_path.name}")
        except Exception as e:
            print(f"[WARNING] Не удалось сохранить результаты в кэш: {e}")
            
    return df_sa