"""
Persistent Disk Cache Manager for Sirena Forecasts
==================================================

Provides disk-based caching for forecasting operations using joblib.Memory.
Cache invalidates based on data hash to ensure correctness when data changes.

Usage:
    from sirena.cache_manager import cached_fit, cached_predict, compute_data_hash

    class MyModel(BaseForecaster):
        @cached_fit
        def fit(self, df, target_col):
            ...

        @cached_predict
        def predict(self, df, target_date):
            ...
"""

import hashlib
import pickle
import time
from functools import wraps
from pathlib import Path
from typing import Any, Optional, Union, Callable

import joblib
import numpy as np
import pandas as pd


CACHE_DIR = Path(".cache/forecasts")
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def compute_data_hash(df: pd.DataFrame) -> str:
    """
    Compute hash of DataFrame for cache invalidation.

    Args:
        df: DataFrame to hash

    Returns:
        SHA256 hash string
    """
    df_str = df.to_csv().encode()
    return hashlib.sha256(df_str).hexdigest()


def _get_model_signature(model: Any) -> str:
    """
    Get unique signature for a model instance.

    Args:
        model: Model instance

    Returns:
        Unique signature string
    """
    class_name = model.__class__.__name__
    model_name = getattr(model, "name", class_name)
    params = getattr(model, "params", {})
    params_str = str(sorted(params.items()))
    return f"{model_name}_{params_str}"


def _get_fit_cache_key(
    model: Any, df: pd.DataFrame, target_col: str = "Все товары и услуги"
) -> str:
    """
    Generate cache key for fit() operation.

    Args:
        model: Model instance
        df: Training data
        target_col: Target column name

    Returns:
        Cache key string
    """
    model_sig = _get_model_signature(model)
    data_hash = compute_data_hash(df)
    return f"fit_{model_sig}_{data_hash}"


def _get_predict_cache_key(
    model: Any, df: pd.DataFrame, target_date: pd.Timestamp
) -> str:
    """
    Generate cache key for predict() operation.

    Args:
        model: Model instance
        df: Input data
        target_date: Prediction date

    Returns:
        Cache key string
    """
    model_sig = _get_model_signature(model)
    data_hash = compute_data_hash(df)
    date_str = target_date.strftime("%Y-%m-%d")
    return f"predict_{model_sig}_{data_hash}_{date_str}"


def _get_cache_memory() -> joblib.Memory:
    """
    Get or create joblib Memory instance.

    Returns:
        joblib.Memory instance configured with cache directory
    """
    return joblib.Memory(location=CACHE_DIR, verbose=0)


def cached_fit(func: Callable) -> Callable:
    """
    Decorator for caching fit() operations.

    Cache key includes model signature and data hash.
    Invalidates when data changes.

    Args:
        func: fit() method to decorate

    Returns:
        Decorated function
    """

    @wraps(func)
    def wrapper(
        self, df: pd.DataFrame, target_col: str = "Все товары и услуги", *args, **kwargs
    ):
        cache_key = _get_fit_cache_key(self, df, target_col)
        cache_path = CACHE_DIR / f"{cache_key}.pkl"

        if cache_path.exists():
            try:
                with open(cache_path, "rb") as f:
                    cached_result = pickle.load(f)
                cached_data_hash = cached_result.get("data_hash")
                current_data_hash = compute_data_hash(df)

                if cached_data_hash == current_data_hash:
                    state = cached_result.get("model_state")
                    if state:
                        for key, value in state.items():
                            setattr(self, key, value)
                        return self
            except Exception:
                pass

        result = func(self, df, target_col, *args, **kwargs)

        try:
            state = {"_is_fitted": self._is_fitted}
            if hasattr(self, "_last_train_date") and self._last_train_date is not None:
                state["_last_train_date"] = self._last_train_date

            state["model_obj"] = self

            cache_data = {
                "data_hash": compute_data_hash(df),
                "model_state": state,
                "timestamp": time.time(),
            }
            with open(cache_path, "wb") as f:
                pickle.dump(cache_data, f)
        except Exception:
            pass

        return result

    return wrapper


def cached_predict(func: Callable) -> Callable:
    """
    Decorator for caching predict() operations.

    Cache key includes model signature, data hash, and target date.
    Invalidates when data changes.

    Args:
        func: predict() method to decorate

    Returns:
        Decorated function
    """

    @wraps(func)
    def wrapper(self, df: pd.DataFrame, target_date: pd.Timestamp, *args, **kwargs):
        cache_key = _get_predict_cache_key(self, df, target_date)
        cache_path = CACHE_DIR / f"{cache_key}.pkl"

        if cache_path.exists():
            try:
                with open(cache_path, "rb") as f:
                    cached_result = pickle.load(f)
                cached_data_hash = cached_result.get("data_hash")
                current_data_hash = compute_data_hash(df)

                if cached_data_hash == current_data_hash:
                    return cached_result["result"]
            except Exception:
                pass

        result = func(self, df, target_date, *args, **kwargs)

        try:
            cache_data = {
                "data_hash": compute_data_hash(df),
                "result": result,
                "timestamp": time.time(),
            }
            with open(cache_path, "wb") as f:
                pickle.dump(cache_data, f)
        except Exception:
            pass

        return result

    return wrapper


def clear_cache(pattern: Optional[str] = None) -> int:
    """
    Clear cached forecast files.

    Args:
        pattern: Optional pattern to match (e.g., "fit_", "predict_").
                If None, clears all cache files.

    Returns:
        Number of files deleted
    """
    count = 0

    for cache_file in CACHE_DIR.glob("*.pkl"):
        if pattern is None or cache_file.stem.startswith(pattern):
            try:
                cache_file.unlink()
                count += 1
            except Exception:
                pass

    return count


def get_cache_stats() -> dict:
    """
    Get cache directory statistics.

    Returns:
        Dictionary with cache stats
    """
    cache_files = list(CACHE_DIR.glob("*.pkl"))
    total_size = sum(f.stat().st_size for f in cache_files)

    fit_count = sum(1 for f in cache_files if f.stem.startswith("fit_"))
    predict_count = sum(1 for f in cache_files if f.stem.startswith("predict_"))

    return {
        "cache_dir": str(CACHE_DIR),
        "total_files": len(cache_files),
        "fit_cache_count": fit_count,
        "predict_cache_count": predict_count,
        "total_size_bytes": total_size,
        "total_size_mb": round(total_size / (1024 * 1024), 2),
    }


__all__ = [
    "cached_fit",
    "cached_predict",
    "compute_data_hash",
    "clear_cache",
    "get_cache_stats",
    "CACHE_DIR",
]
