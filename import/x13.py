# utils/x13.py
# ФИНАЛЬНАЯ ВЕРСИЯ - 16.06.2025

import os
import sys
import shutil
import tempfile
import subprocess
from pathlib import Path
from warnings import warn
import pandas as pd

class X13Error(Exception): pass

class X13ArimaAnalysisResult:
    """Класс для хранения результатов анализа X-13."""
    def __init__(self, **kwargs):
        self.observed = pd.Series(dtype=float)
        self.seasadj = pd.Series(dtype=float)
        self.trend = pd.Series(dtype=float)
        self.irregular = pd.Series(dtype=float)
        self.sf = pd.Series(dtype=float)
        self.results = ""
        for key, value in kwargs.items(): setattr(self, key, value)

def _find_x12():
    """Находит исполняемый файл X-13 в папке /bin проекта."""
    root_dir = Path(__file__).resolve().parents[1]
    bin_dir = root_dir / 'bin'
    platform_dir = 'windows' if sys.platform.startswith('win') else 'linux'
    exe_name = 'x13as_ascii.exe' if sys.platform.startswith('win') else 'x13as_ascii'
    exe_path = bin_dir / platform_dir / exe_name
    if not exe_path.is_file(): raise FileNotFoundError(f"Не найден исполняемый файл X-13: {exe_path}")
    return str(exe_path)

def _open_and_read(fname):
    try:
        with open(fname, 'r', encoding="latin-1") as fin: return fin.read()
    except FileNotFoundError: return ""

def _convert_out_to_series(file_content, name, original_series):
    """[v9] Железобетонный парсер с проверкой на пустой DataFrame."""
    from io import StringIO
    if not file_content.strip():
        return pd.Series(index=original_series.index, dtype=float, name=name)
    try:
        df = pd.read_csv(StringIO(file_content), sep=r'\s+', header=None, skiprows=1, engine='python')
        
        # --- ГЛАВНОЕ ИСПРАВЛЕНИЕ ---
        if df.empty:
            warn(f"Файл для '{name}' пуст или нечитаем, результат будет пустым.")
            return pd.Series(index=original_series.index, dtype=float, name=name)
        # -----------------------------

        values = pd.to_numeric(df.iloc[:, -1], errors='coerce').dropna().values
        
        if len(values) == 0:
            return pd.Series(index=original_series.index, dtype=float, name=name)
        
        if len(values) != len(original_series):
            temp_index = original_series.index[:len(values)]
            temp_series = pd.Series(values, index=temp_index, name=name)
            return temp_series.reindex(original_series.index)
        else:
            return pd.Series(values, index=original_series.index, name=name, dtype=float)
    except Exception as e:
        warn(f"Критическая ошибка парсинга для '{name}': {e}.")
        return pd.Series(index=original_series.index, dtype=float, name=name)

def x13_arima_analysis(endog, log=True, outlier=True, seats=True, endog_name='series'):
    """Финальная надежная версия с планом 'А' (automdl) и планом 'Б' (airline)."""
    
    def _run_and_check(spec_text, temp_dir):
        x12_path = _find_x12()
        spec_path = os.path.join(temp_dir, 'spec.spc')
        out_path_base = os.path.join(temp_dir, 'spec')
        with open(spec_path, 'w', encoding='utf8') as f: f.write(spec_text)

        proc = subprocess.Popen([x12_path, spec_path[:-4], out_path_base],
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=False)
        proc.wait()
        
        err_file = out_path_base + '.err'
        if os.path.exists(err_file) and os.path.getsize(err_file) > 5:
            errors = _open_and_read(err_file)
            if 'ERROR' in errors or 'FATAL' in errors: raise X13Error(errors)
        
        key_output_file = out_path_base + ('.s11' if seats else '.d11')
        if not os.path.exists(key_output_file):
            raise X13Error(f"Ключевой выходной файл {os.path.basename(key_output_file)} не был создан.")

    endog.name = endog_name
    freq_str = endog.index.freqstr or pd.infer_freq(endog.index)
    period = {'M': 12, 'Q': 4}.get(freq_str[0], 12)
    start_date = endog.index[0]
    data = "({0})".format("\n".join(map(str, endog.values.tolist())))
    
    base_spec = (f"series{{\n"
                 f"name=\"{endog_name}\"\ndata={data}\n"
                 f"start={start_date.year}.{start_date.month}\nperiod={period}\n"
                 f"}}\n"
                 f"transform{{function={'log' if log else 'none'}}}\n")
    if outlier: base_spec += "outlier{}\n"
    if seats: base_spec += "seats{save=(s11 s12 s13 s18)}\n"

    temp_dir = tempfile.mkdtemp()
    
    try:
        try:
            automdl_spec = "automdl{maxorder=(2,1)\nmaxdiff=(2,1)}\n"
            _run_and_check(base_spec + automdl_spec, temp_dir)
        except X13Error as e:
            print(f"  - ⚠️ automdl не сработал. Переход к плану 'Б'.")
            airline_spec = "arima{model=(0 1 1)(0 1 1)}\n"
            _run_and_check(base_spec + airline_spec, temp_dir)

        out_path_base = os.path.join(temp_dir, 'spec')
        results = _open_and_read(out_path_base + '.out')
        seasadj = _convert_out_to_series(_open_and_read(out_path_base + '.s11'), 'seasadj', endog)
        trend = _convert_out_to_series(_open_and_read(out_path_base + '.s12'), 'trend', endog)
        irregular = _convert_out_to_series(_open_and_read(out_path_base + '.s13'), 'irregular', endog)
        sf = _convert_out_to_series(_open_and_read(out_path_base + '.s18'), 'sf', endog)
        
        return X13ArimaAnalysisResult(observed=endog, results=results, sf=sf, seasadj=seasadj, trend=trend, irregular=irregular)

    except X13Error as e:
        print(f"  - ❌ КРИТИЧЕСКАЯ ОШИБКА: Оба плана не сработали для '{endog.name}'.")
        return X13ArimaAnalysisResult(observed=endog, results=str(e))
    finally:
        shutil.rmtree(temp_dir)