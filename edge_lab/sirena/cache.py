"""
Redis Cache for Sirena Forecasting System
===========================================

Provides caching functionality for forecasting operations using Redis or in-memory backend.
Improves performance by caching expensive data loading and forecast computations.

Usage:
    from sirena.cache import cache_decorator, CachedDataLoader

    # Use decorator for caching any function
    @cache_decorator(key="my_forecast", ttl=3600)
    def my_expensive_function(param):
        return compute_expensive_result(param)

    # Or use the cached data loader
    loader = CachedDataLoader(use_cache=True, cache_backend="redis")
    data = loader.load_data()
"""

import json
import hashlib
import time
import pickle
from typing import Any, Optional, Callable, Dict, Union
from functools import wraps
from pathlib import Path

try:
    import redis

    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False


class CacheBackend:
    """Abstract cache backend."""

    def get(self, key: str) -> Optional[Any]:
        raise NotImplementedError

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        raise NotImplementedError

    def delete(self, key: str) -> bool:
        raise NotImplementedError

    def clear(self) -> bool:
        raise NotImplementedError

    def exists(self, key: str) -> bool:
        raise NotImplementedError


class MemoryCache(CacheBackend):
    """In-memory cache backend (fallback when Redis unavailable)."""

    def __init__(self):
        self._store: Dict[str, tuple] = {}
        self._timestamps: Dict[str, float] = {}

    def get(self, key: str) -> Optional[Any]:
        if key in self._store:
            data, expiry = self._store[key]
            if expiry is None or time.time() < expiry:
                return data
            else:
                del self._store[key]
                del self._timestamps[key]
        return None

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        expiry = time.time() + ttl if ttl else None
        self._store[key] = (value, expiry)
        self._timestamps[key] = time.time()
        return True

    def delete(self, key: str) -> bool:
        if key in self._store:
            del self._store[key]
            del self._timestamps[key]
            return True
        return False

    def clear(self) -> bool:
        self._store.clear()
        self._timestamps.clear()
        return True

    def exists(self, key: str) -> bool:
        return key in self._store

    def size(self) -> int:
        return len(self._store)


class RedisCache(CacheBackend):
    """Redis cache backend."""

    def __init__(
        self,
        host: str = "localhost",
        port: int = 6379,
        db: int = 0,
        password: Optional[str] = None,
    ):
        self.host = host
        self.port = port
        self.db = db
        self.password = password
        self._client: Optional[redis.Redis] = None
        self._enabled = False
        self._connect()

    def _connect(self):
        """Connect to Redis server."""
        if not REDIS_AVAILABLE:
            return

        try:
            self._client = redis.Redis(
                host=self.host,
                port=self.port,
                db=self.db,
                password=self.password,
                decode_responses=False,
                socket_connect_timeout=2,
            )
            self._client.ping()
            self._enabled = True
        except Exception:
            self._enabled = False
            self._client = None

    def get(self, key: str) -> Optional[Any]:
        if not self._enabled or self._client is None:
            return None
        try:
            data = self._client.get(key)
            if data is not None:
                return pickle.loads(data)
            return None
        except Exception:
            return None

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        if not self._enabled or self._client is None:
            return False
        try:
            serialized = pickle.dumps(value)
            if ttl:
                return self._client.setex(key, ttl, serialized)
            return self._client.set(key, serialized)
        except Exception:
            return False

    def delete(self, key: str) -> bool:
        if not self._enabled or self._client is None:
            return False
        try:
            return bool(self._client.delete(key))
        except Exception:
            return False

    def clear(self) -> bool:
        if not self._enabled or self._client is None:
            return False
        try:
            return self._client.flushdb()
        except Exception:
            return False

    def exists(self, key: str) -> bool:
        if not self._enabled or self._client is None:
            return False
        try:
            return bool(self._client.exists(key))
        except Exception:
            return False

    def size(self) -> int:
        if not self._enabled or self._client is None:
            return 0
        try:
            return self._client.dbsize()
        except Exception:
            return 0


class CacheManager:
    """Manages cache backend selection and operations."""

    def __init__(self, backend: str = "auto", **kwargs):
        """
        Initialize cache manager.

        Args:
            backend: "redis", "memory", or "auto" (tries Redis, falls back to memory)
            **kwargs: Additional arguments for Redis connection (host, port, password)
        """
        self.backend_type = backend
        self._backend: Optional[CacheBackend] = None

        if backend == "auto":
            self._backend = RedisCache(**kwargs)
            if not isinstance(self._backend, RedisCache) or not self._backend._enabled:
                self._backend = MemoryCache()
                self.backend_type = "memory"
        elif backend == "redis":
            self._backend = RedisCache(**kwargs)
        else:
            self._backend = MemoryCache()

    @property
    def backend(self) -> CacheBackend:
        if self._backend is None:
            self._backend = MemoryCache()
        return self._backend

    def get(self, key: str) -> Optional[Any]:
        return self.backend.get(key)

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        return self.backend.set(key, value, ttl)

    def delete(self, key: str) -> bool:
        return self.backend.delete(key)

    def clear(self) -> bool:
        return self.backend.clear()

    def exists(self, key: str) -> bool:
        return self.backend.exists(key)

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        stats = {"backend": self.backend_type, "enabled": True}

        if isinstance(self.backend, MemoryCache):
            stats["size"] = self.backend.size()
        elif isinstance(self.backend, RedisCache):
            stats["enabled"] = self.backend._enabled
            stats["host"] = self.backend.host
            stats["port"] = self.backend.port
            stats["size"] = self.backend.size()

        return stats


_cache_manager: Optional[CacheManager] = None


def get_cache_manager(**kwargs) -> CacheManager:
    """Get or create global cache manager."""
    global _cache_manager
    if _cache_manager is None:
        _cache_manager = CacheManager(**kwargs)
    return _cache_manager


def generate_cache_key(prefix: str, *args, **kwargs) -> str:
    """Generate a unique cache key from function arguments."""
    key_parts = [prefix]

    for arg in args:
        if isinstance(arg, (str, int, float, bool)):
            key_parts.append(str(arg))
        else:
            key_parts.append(hashlib.md5(str(arg).encode()).hexdigest()[:8])

    for k, v in sorted(kwargs.items()):
        key_parts.append(f"{k}={v}")

    return ":".join(key_parts)


def cache_decorator(key_prefix: str, ttl: Optional[int] = None, use_args: bool = True):
    """
    Decorator for caching function results.

    Args:
        key_prefix: Prefix for cache key
        ttl: Time to live in seconds (None = no expiry)
        use_args: Include function arguments in cache key

    Example:
        @cache_decorator(key_prefix="forecast", ttl=3600)
        def compute_forecast(model, horizon):
            return expensive_computation()
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            cache = get_cache_manager()

            if use_args:
                cache_key = generate_cache_key(
                    key_prefix, func.__name__, *args, **kwargs
                )
            else:
                cache_key = f"{key_prefix}:{func.__name__}"

            cached = cache.get(cache_key)
            if cached is not None:
                return cached

            result = func(*args, **kwargs)
            cache.set(cache_key, result, ttl)

            return result

        return wrapper

    return decorator


class CachedDataLoader:
    """Data loader with caching support."""

    def __init__(
        self,
        use_cache: bool = True,
        cache_backend: str = "auto",
        data_dir: Optional[Union[str, Path]] = None,
        ttl: int = 3600,
    ):
        """
        Initialize cached data loader.

        Args:
            use_cache: Enable caching
            cache_backend: "redis", "memory", or "auto"
            data_dir: Data directory path
            ttl: Cache time-to-live in seconds
        """
        self.use_cache = use_cache
        self.cache_ttl = ttl
        self.data_dir = Path(data_dir) if data_dir else Path.cwd().parent / "data"

        self._cache_manager: Optional[CacheManager] = None
        if use_cache:
            self._cache_manager = CacheManager(backend=cache_backend)

    def _get_cache_key(self, name: str, *args) -> str:
        """Generate cache key for data loading."""
        key_parts = [name, str(self.data_dir)]
        for arg in args:
            key_parts.append(str(arg))
        return ":".join(key_parts)

    def load_csv(self, filename: str, force_refresh: bool = False) -> Optional[Any]:
        """Load CSV file with caching."""
        cache_key = self._get_cache_key("csv", filename)

        if not self.use_cache:
            return self._load_csv_uncached(filename)

        if self._cache_manager and not force_refresh:
            cached = self._cache_manager.get(cache_key)
            if cached is not None:
                return cached

        data = self._load_csv_uncached(filename)
        if data is not None and self._cache_manager:
            self._cache_manager.set(cache_key, data, self.cache_ttl)

        return data

    def _load_csv_uncached(self, filename: str) -> Optional[Any]:
        """Load CSV file without caching."""
        try:
            import pandas as pd

            filepath = self.data_dir / filename

            if not filepath.exists():
                return None

            return pd.read_csv(filepath, sep=";", encoding="utf-8-sig")
        except Exception:
            return None

    def load_monthly_kbr(self, force_refresh: bool = False) -> Optional[Any]:
        """Load monthly KBR inflation data with caching."""
        return self.load_csv("inflation_data.csv", force_refresh)

    def load_all(self, force_refresh: bool = False):
        """Load all data types."""
        monthly = self.load_monthly_kbr(force_refresh)
        return monthly, None

    def clear_cache(self):
        """Clear all cached data."""
        if self._cache_manager:
            self._cache_manager.clear()

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        if not self._cache_manager:
            return {"backend": "none", "enabled": False, "size": 0}
        return self._cache_manager.get_stats()


__all__ = [
    "CacheManager",
    "MemoryCache",
    "RedisCache",
    "cache_decorator",
    "CachedDataLoader",
    "get_cache_manager",
    "generate_cache_key",
]
