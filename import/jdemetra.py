# import/jdemetra.py
# Python wrapper for JDemetra+ v3 seasonal adjustment
# Uses bin/linux/jdemetra_sa.jar via subprocess
# Last Updated: 2026-03-10

import os
import sys
import json
import tempfile
import subprocess
from pathlib import Path
from warnings import warn

import pandas as pd
import numpy as np

# Путь к JAR
ROOT_DIR = Path(__file__).resolve().parents[1]
JAR_PATH = ROOT_DIR / 'bin' / 'linux' / 'jdemetra_sa.jar'


class JDemetraResult:
    """Results of JDemetra+ seasonal adjustment."""
    def __init__(self, observed=None, seasadj=None, trend=None,
                 seasonal=None, irregular=None, method=None, status=None):
        self.observed = observed if observed is not None else pd.Series(dtype=float)
        self.seasadj = seasadj if seasadj is not None else pd.Series(dtype=float)
        self.trend = trend if trend is not None else pd.Series(dtype=float)
        self.seasonal = seasonal if seasonal is not None else pd.Series(dtype=float)
        self.irregular = irregular if irregular is not None else pd.Series(dtype=float)
        self.method = method or ''
        self.status = status or ''


def _find_java():
    """Find Java executable."""
    java = 'java'
    try:
        proc = subprocess.run([java, '-version'], capture_output=True, text=True,
                              timeout=10)
        if proc.returncode == 0:
            return java
    except Exception:
        pass
    raise FileNotFoundError("Java not found. Install JDK 21+: sudo apt install openjdk-21-jdk-headless")


def jdemetra_seasonal_adjustment(endog, method='tramo-seats', freq=12, endog_name='series'):
    """
    Perform seasonal adjustment using JDemetra+ v3 (TRAMO-SEATS or X-13).
    
    Parameters
    ----------
    endog : pd.Series
        Time series data (cumulative index, NOT rates/pp).
        Must be positive and have DatetimeIndex with monthly/quarterly frequency.
    method : str
        'tramo-seats' (default, Eurostat standard) or 'x13'
    freq : int
        12 for monthly (default), 4 for quarterly
    endog_name : str
        Name for the series (for logging)
    
    Returns
    -------
    JDemetraResult with observed, seasadj, trend, seasonal, irregular
    """
    if not JAR_PATH.is_file():
        raise FileNotFoundError(
            f"JDemetra+ JAR not found: {JAR_PATH}\n"
            f"Build: cd tools/jdemetra && mvn package"
        )
    
    java = _find_java()
    
    # Validate input
    if len(endog) < 36:
        warn(f"Series '{endog_name}' too short ({len(endog)} < 36). Skipping.")
        return JDemetraResult(observed=endog, status='TOO_SHORT')
    
    if endog.isna().any():
        endog = endog.dropna()
    
    if (endog <= 0).any():
        warn(f"Series '{endog_name}' has non-positive values. Log transform may fail.")
    
    # Create temp CSV
    temp_dir = tempfile.mkdtemp(prefix='jdemetra_')
    input_csv = os.path.join(temp_dir, 'input.csv')
    output_json = os.path.join(temp_dir, 'output.json')
    
    try:
        # Write CSV with dates
        with open(input_csv, 'w') as f:
            f.write("date,value\n")
            for idx, val in endog.items():
                date_str = idx.strftime('%Y-%m-%d')
                f.write(f"{date_str},{val:.6f}\n")
        
        # Run JDemetra+
        cmd = [
            java, '-jar', str(JAR_PATH),
            '--input', input_csv,
            '--output', output_json,
            '--method', method,
            '--freq', str(freq)
        ]
        
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        
        if proc.returncode != 0:
            warn(f"JDemetra+ failed for '{endog_name}': {proc.stderr}")
            return JDemetraResult(observed=endog, status=f'ERROR: {proc.stderr}')
        
        # Parse output
        if not os.path.exists(output_json):
            warn(f"JDemetra+ output file not created for '{endog_name}'")
            return JDemetraResult(observed=endog, status='NO_OUTPUT')
        
        with open(output_json, 'r') as f:
            data = json.load(f)
        
        if data.get('status') != 'OK':
            warn(f"JDemetra+ status: {data.get('status')}")
            return JDemetraResult(observed=endog, status=data.get('status', 'UNKNOWN'))
        
        # Build result Series with same index as input
        index = endog.index
        
        def _to_series(values, name):
            if values and len(values) == len(index):
                return pd.Series(values, index=index, name=name, dtype=float)
            elif values:
                # Truncate or pad
                n = min(len(values), len(index))
                return pd.Series(values[:n], index=index[:n], name=name, dtype=float)
            return pd.Series(dtype=float, name=name)
        
        result = JDemetraResult(
            observed=endog,
            seasadj=_to_series(data.get('seasadj', []), 'seasadj'),
            trend=_to_series(data.get('trend', []), 'trend'),
            seasonal=_to_series(data.get('seasonal', []), 'seasonal'),
            irregular=_to_series(data.get('irregular', []), 'irregular'),
            method=method,
            status='OK'
        )
        
        print(f"  [OK] JDemetra+ ({method}) '{endog_name}': {len(result.seasadj)} obs")
        return result
        
    except subprocess.TimeoutExpired:
        warn(f"JDemetra+ timeout for '{endog_name}'")
        return JDemetraResult(observed=endog, status='TIMEOUT')
    except Exception as e:
        warn(f"JDemetra+ error for '{endog_name}': {e}")
        return JDemetraResult(observed=endog, status=f'ERROR: {e}')
    finally:
        # Cleanup
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)


def jdemetra_tramo_seats(endog, endog_name='series'):
    """Shortcut for TRAMO-SEATS method."""
    return jdemetra_seasonal_adjustment(endog, method='tramo-seats', endog_name=endog_name)


def jdemetra_x13(endog, endog_name='series'):
    """Shortcut for X-13 method (via JDemetra+ v3)."""
    return jdemetra_seasonal_adjustment(endog, method='x13', endog_name=endog_name)


if __name__ == '__main__':
    # Quick test
    print(f"JAR path: {JAR_PATH}")
    print(f"JAR exists: {JAR_PATH.is_file()}")
    print(f"Java: {_find_java()}")
    print("Module loaded OK.")
