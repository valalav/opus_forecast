"""
Cached Data Loader for SIRENA-KBR
==================================

Provides data loading with Redis caching support for improved performance.
"""

import logging
from pathlib import Path
from typing import Optional
from functools import wraps

import pandas as pd
import numpy as np

from sirena.cache import ForecastCache, CacheKey

logger = logging.getLogger(__name__)


def measure_time(func):
    """Decorator to measure function execution time."""
    import time

    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start
        logger.debug(f"{func.__name__} took {elapsed:.4f}s")
        return result

    return wrapper


class CachedDataLoader:
    """
    Data loader with Redis caching support.

    Provides automatic caching of loaded data to improve performance.
    """

    def __init__(
        self,
        data_dir: str = "data",
        use_cache: bool = True,
        cache_backend: str = "redis",
        cache_ttl: int = 3600,
    ):
        """
        Initialize cached data loader.

        Args:
            data_dir: Data directory path
            use_cache: Enable caching
            cache_backend: Cache backend ('memory', 'file', 'redis')
            cache_ttl: Cache TTL in seconds
        """
        self.data_dir = Path(data_dir)
        self.use_cache = use_cache and cache_backend != "none"
        self._cache: Optional[ForecastCache] = None

        if self.use_cache:
            try:
                self._cache = ForecastCache(backend=cache_backend, ttl=cache_ttl)
                logger.info(f"Cache enabled: {cache_backend}")
            except Exception as e:
                logger.warning(
                    f"Cache initialization failed: {e}. Using uncached mode."
                )
                self.use_cache = False

        self._monthly_data: Optional[pd.DataFrame] = None
        self._weekly_data: Optional[pd.DataFrame] = None
        self._inflation_data: Optional[pd.DataFrame] = None

    def _get_file_hash(self, filepath: Path) -> str:
        """Get hash of file for cache key."""
        if not filepath.exists():
            return "no_file"
        stat = filepath.stat()
        return f"{filepath.name}_{stat.st_size}_{stat.st_mtime}"

    @measure_time
    def load_monthly_kbr(self, force_refresh: bool = False) -> Optional[pd.DataFrame]:
        """
        Load monthly KBR inflation data with caching.

        Args:
            force_refresh: Force refresh from disk

        Returns:
            DataFrame with monthly inflation data
        """
        filepath = self.data_dir / "infl_kbr.csv"
        cache_key = f"monthly_kbr_{self._get_file_hash(filepath)}"

        if self.use_cache and not force_refresh:
            cached = self._cache.get(cache_key)
            if cached is not None:
                logger.info("Monthly KBR data loaded from cache")
                self._monthly_data = cached
                return cached

        if not filepath.exists():
            logger.error(f"File not found: {filepath}")
            return None

        try:
            df_raw = pd.read_csv(filepath, sep=";", decimal=".")

            if "Day" in df_raw.columns:
                try:
                    df_raw["Date"] = pd.to_datetime(df_raw["Day"], format="%d.%m.%Y")
                except ValueError:
                    df_raw["Date"] = pd.to_datetime(
                        df_raw["Day"], format="%Y-%m-%d", errors="coerce"
                    )
                    if df_raw["Date"].isna().all():
                        df_raw["Date"] = pd.to_datetime(df_raw["Day"])

            if "Товар" in df_raw.columns and "MoM" in df_raw.columns:
                df = df_raw.pivot_table(
                    index="Date", columns="Товар", values="MoM", aggfunc="first"
                )
            else:
                df = df_raw.set_index("Date")

            required_cols = [
                "Все товары и услуги",
                "Продовольственные товары",
                "Непродовольственные товары",
                "Услуги",
            ]
            df = df[required_cols].copy()
            df = df.sort_index()

            self._monthly_data = df

            if self.use_cache:
                self._cache.set(cache_key, df)

            logger.info(f"Loaded {len(df)} months of KBR data")
            return df

        except Exception as e:
            logger.error(f"Error loading infl_kbr.csv: {e}")
            return None

    @measure_time
    def load_weekly_prices(self, force_refresh: bool = False) -> Optional[pd.DataFrame]:
        """
        Load weekly prices data with caching.

        Args:
            force_refresh: Force refresh from disk

        Returns:
            DataFrame with weekly prices
        """
        filepath = self.data_dir / "weekly_prices.csv"
        cache_key = f"weekly_prices_{self._get_file_hash(filepath)}"

        if self.use_cache and not force_refresh:
            cached = self._cache.get(cache_key)
            if cached is not None:
                logger.info("Weekly prices loaded from cache")
                self._weekly_data = cached
                return cached

        if not filepath.exists():
            logger.warning(f"Weekly data not found: {filepath}")
            return None

        try:
            w = pd.read_csv(filepath, sep=";", decimal=",")

            if "Товары" not in w.columns:
                w = pd.read_csv(filepath, sep=";", decimal=".")

            if "Сведено" in w.columns:
                w[["year", "week"]] = (
                    w["Сведено"].str.split("_", expand=True).astype(int)
                )
                w["month"] = pd.to_datetime(
                    w["year"].astype(str) + w["week"].astype(str) + "1", format="%Y%W%w"
                ).dt.month

            self._weekly_data = w

            if self.use_cache:
                self._cache.set(cache_key, w)

            logger.info(f"Loaded {len(w)} weekly price records")
            return w

        except Exception as e:
            logger.warning(f"Error loading weekly_prices.csv: {e}")
            return None

    @measure_time
    def load_inflation_data(
        self, force_refresh: bool = False
    ) -> Optional[pd.DataFrame]:
        """
        Load extended inflation data with caching.

        Args:
            force_refresh: Force refresh from disk

        Returns:
            DataFrame with inflation and macro indicators
        """
        filepath = self.data_dir / "inflation_data.csv"
        cache_key = f"inflation_data_{self._get_file_hash(filepath)}"

        if self.use_cache and not force_refresh:
            cached = self._cache.get(cache_key)
            if cached is not None:
                logger.info("Inflation data loaded from cache")
                self._inflation_data = cached
                return cached

        if not filepath.exists():
            logger.error(f"File not found: {filepath}")
            return None

        try:
            df = pd.read_csv(filepath, sep=";", decimal=",")

            cols_to_fix = ["mom", "Prod", "Nonprod", "Serv", "usd_nom_i", "Ruonia", "Ki", "Ki_i"]
            for col in cols_to_fix:
                if col in df.columns:
                    if df[col].dtype == object:
                        df[col] = df[col].astype(str).str.replace(",", ".")
                    df[col] = pd.to_numeric(df[col], errors="coerce")

            df["Date"] = pd.to_datetime(df["Date"], format="%d.%m.%Y", errors="coerce")
            if df["Date"].isna().any():
                df["Date"] = pd.to_datetime(df["Date"])

            df["Date"] = df["Date"].dt.to_period("M").dt.to_timestamp()
            df = df.set_index("Date").sort_index()

            self._inflation_data = df

            if self.use_cache:
                self._cache.set(cache_key, df)

            logger.info(f"Loaded {len(df)} months of macro data")
            return df

        except Exception as e:
            logger.error(f"Error loading inflation_data.csv: {e}")
            return None

    def load_all(self, force_refresh: bool = False) -> tuple:
        """
        Load all data with caching.

        Args:
            force_refresh: Force refresh from disk

        Returns:
            Tuple (monthly_data, weekly_data)
        """
        monthly = self.load_monthly_kbr(force_refresh=force_refresh)
        weekly = self.load_weekly_prices(force_refresh=force_refresh)
        return monthly, weekly

    @property
    def monthly_data(self) -> Optional[pd.DataFrame]:
        """Monthly data (lazy load)."""
        if self._monthly_data is None:
            self.load_monthly_kbr()
        return self._monthly_data

    @property
    def weekly_data(self) -> Optional[pd.DataFrame]:
        """Weekly data (lazy load)."""
        if self._weekly_data is None:
            self.load_weekly_prices()
        return self._weekly_data

    @property
    def inflation_data(self) -> Optional[pd.DataFrame]:
        """Inflation data (lazy load)."""
        if self._inflation_data is None:
            self.load_inflation_data()
        return self._inflation_data

    def clear_cache(self):
        """Clear all cached data."""
        if self._cache:
            self._cache.clear()
            self._monthly_data = None
            self._weekly_data = None
            self._inflation_data = None
            logger.info("Cache cleared")

    def get_cache_stats(self) -> dict:
        """Get cache statistics."""
        if self._cache:
            return self._cache.stats()
        return {"enabled": False}


# Singleton loader
_cached_loader: Optional[CachedDataLoader] = None


def get_cached_loader(
    use_cache: bool = True, cache_backend: str = "redis"
) -> CachedDataLoader:
    """
    Get global cached data loader.

    Args:
        use_cache: Enable caching
        cache_backend: Cache backend

    Returns:
        CachedDataLoader instance
    """
    global _cached_loader

    if _cached_loader is None:
        _cached_loader = CachedDataLoader(
            use_cache=use_cache, cache_backend=cache_backend
        )

    return _cached_loader


if __name__ == "__main__":
    import time

    logging.basicConfig(level=logging.INFO)

    loader = CachedDataLoader(use_cache=True, cache_backend="memory")

    print("=" * 50)
    print("First load (uncached):")
    start = time.time()
    monthly = loader.load_monthly_kbr()
    elapsed = time.time() - start
    print(f"Time: {elapsed:.4f}s")
    if monthly is not None:
        print(f"Shape: {monthly.shape}")
        print(f"Columns: {list(monthly.columns)}")

    print("\n" + "=" * 50)
    print("Second load (from cache):")
    start = time.time()
    monthly = loader.load_monthly_kbr()
    elapsed = time.time() - start
    print(f"Time: {elapsed:.4f}s")

    print("\n" + "=" * 50)
    print("Cache stats:")
    print(loader.get_cache_stats())
