"""
Кэширование результатов СИРЕНА-КБР
===================================

Поддерживает различные бэкенды:
- memory: в памяти (LRU)
- file: файловый кэш
- redis: Redis (требует redis-py)
"""

import hashlib
import json
import pickle
import time
from typing import Any, Dict, Optional, Union
from functools import lru_cache
from pathlib import Path
import numpy as np
import pandas as pd


class CacheKey:
    """Генератор ключей кэша."""

    @staticmethod
    def make_key(
        model: str, data_hash: str, horizon: int, params: Optional[Dict] = None
    ) -> str:
        """
        Создание уникального ключа кэша.

        Args:
            model: Название модели
            data_hash: Хэш данных
            horizon: Горизонт прогноза
            params: Дополнительные параметры

        Returns:
            Уникальный ключ
        """
        key_parts = [model, data_hash, str(horizon)]

        if params:
            params_str = json.dumps(params, sort_keys=True)
            key_parts.append(hashlib.md5(params_str.encode()).hexdigest()[:8])

        return ":".join(key_parts)

    @staticmethod
    def hash_dataframe(df: pd.DataFrame) -> str:
        """
        Хэш DataFrame для ключа кэша.

        Args:
            df: DataFrame

        Returns:
            MD5 хэш
        """
        # Используем последние данные и размер
        key_data = f"{df.shape}_{df.index.max()}_{df.iloc[-1].sum()}"
        return hashlib.md5(key_data.encode()).hexdigest()[:12]


class MemoryCache:
    """
    Кэш в памяти с LRU политикой.

    Example:
        cache = MemoryCache(maxsize=100, ttl=3600)
        cache.set('key', value)
        result = cache.get('key')
    """

    def __init__(self, maxsize: int = 100, ttl: int = 3600):
        """
        Инициализация.

        Args:
            maxsize: Максимальный размер кэша
            ttl: Время жизни записи в секундах
        """
        self.maxsize = maxsize
        self.ttl = ttl
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._access_order: list = []

    def get(self, key: str) -> Optional[Any]:
        """Получить значение из кэша."""
        if key not in self._cache:
            return None

        entry = self._cache[key]

        # Проверяем TTL
        if time.time() - entry["timestamp"] > self.ttl:
            self.delete(key)
            return None

        # Обновляем порядок доступа
        if key in self._access_order:
            self._access_order.remove(key)
        self._access_order.append(key)

        return entry["value"]

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Записать значение в кэш."""
        # Удаляем старые записи если превышен лимит
        while len(self._cache) >= self.maxsize and self._access_order:
            oldest_key = self._access_order.pop(0)
            if oldest_key in self._cache:
                del self._cache[oldest_key]

        self._cache[key] = {
            "value": value,
            "timestamp": time.time(),
            "ttl": ttl or self.ttl,
        }
        self._access_order.append(key)

    def delete(self, key: str) -> bool:
        """Удалить запись."""
        if key in self._cache:
            del self._cache[key]
            if key in self._access_order:
                self._access_order.remove(key)
            return True
        return False

    def clear(self) -> None:
        """Очистить кэш."""
        self._cache.clear()
        self._access_order.clear()

    def stats(self) -> Dict[str, Any]:
        """Статистика кэша."""
        return {"size": len(self._cache), "maxsize": self.maxsize, "ttl": self.ttl}


class FileCache:
    """
    Файловый кэш.

    Example:
        cache = FileCache(cache_dir='.cache')
        cache.set('key', value)
    """

    def __init__(self, cache_dir: str = ".cache", ttl: int = 86400):
        """
        Инициализация.

        Args:
            cache_dir: Директория для кэша
            ttl: Время жизни в секундах (по умолчанию 24 часа)
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.ttl = ttl

    def _get_path(self, key: str) -> Path:
        """Путь к файлу кэша."""
        safe_key = hashlib.md5(key.encode()).hexdigest()
        return self.cache_dir / f"{safe_key}.pkl"

    def get(self, key: str) -> Optional[Any]:
        """Получить значение."""
        path = self._get_path(key)

        if not path.exists():
            return None

        # Проверяем TTL
        if time.time() - path.stat().st_mtime > self.ttl:
            path.unlink()
            return None

        try:
            with open(path, "rb") as f:
                return pickle.load(f)
        except Exception:
            return None

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Записать значение."""
        path = self._get_path(key)

        try:
            with open(path, "wb") as f:
                pickle.dump(value, f)
        except Exception as e:
            print(f"Ошибка записи кэша: {e}")

    def delete(self, key: str) -> bool:
        """Удалить запись."""
        path = self._get_path(key)
        if path.exists():
            path.unlink()
            return True
        return False

    def clear(self) -> None:
        """Очистить кэш."""
        for file in self.cache_dir.glob("*.pkl"):
            file.unlink()

    def stats(self) -> Dict[str, Any]:
        """Статистика."""
        files = list(self.cache_dir.glob("*.pkl"))
        total_size = sum(f.stat().st_size for f in files)

        return {
            "count": len(files),
            "size_bytes": total_size,
            "size_mb": round(total_size / 1024 / 1024, 2),
            "cache_dir": str(self.cache_dir),
        }


class RedisCache:
    """
    Redis кэш с поддержкой TTL.

    Требует установленного redis-py и работающего Redis сервера.

    Example:
        cache = RedisCache(host='localhost', port=6379, ttl=3600)
        cache.set('key', value)
        result = cache.get('key')
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 6379,
        db: int = 0,
        password: Optional[str] = None,
        ttl: int = 3600,
    ):
        """
        Инициализация.

        Args:
            host: Redis host
            port: Redis port
            db: Redis database number
            password: Redis password (optional)
            ttl: Default TTL в секундах
        """
        self.host = host
        self.port = port
        self.db = db
        self.password = password
        self.ttl = ttl
        self._client = None
        self._enabled = True

    def connect(self) -> bool:
        """Установить соединение с Redis."""
        try:
            import redis

            self._client = redis.Redis(
                host=self.host,
                port=self.port,
                db=self.db,
                password=self.password,
                decode_responses=False,
                socket_connect_timeout=5,
                socket_timeout=5,
            )
            self._client.ping()
            print(f"Redis cache connected: {self.host}:{self.port}")
            return True
        except Exception as e:
            print(f"Redis connection failed: {e}. Cache disabled.")
            self._enabled = False
            self._client = None
            return False

    @property
    def is_enabled(self) -> bool:
        """Проверить включен ли кэш."""
        return self._enabled and self._client is not None

    def _serialize(self, value: Any) -> bytes:
        """Сериализация для хранения в Redis."""
        if isinstance(value, pd.DataFrame):
            return pickle.dumps({"type": "dataframe", "data": value})
        elif isinstance(value, pd.Series):
            return pickle.dumps({"type": "series", "data": value})
        elif isinstance(value, np.ndarray):
            return pickle.dumps({"type": "array", "data": value})
        else:
            return pickle.dumps({"type": "object", "data": value})

    def _deserialize(self, value: bytes) -> Any:
        """Десериализация из Redis."""
        try:
            obj = pickle.loads(value)
            obj_type = obj.get("type")
            data = obj.get("data")
            return data
        except Exception as e:
            print(f"Deserialization error: {e}")
            raise

    def get(self, key: str) -> Optional[Any]:
        """Получить значение из кэша."""
        if not self.is_enabled:
            return None
        try:
            value = self._client.get(key)
            if value:
                return self._deserialize(value)
            return None
        except Exception as e:
            print(f"Cache get error: {e}")
            return None

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Записать значение в кэш."""
        if not self.is_enabled:
            return False
        try:
            ttl = ttl or self.ttl
            serialized = self._serialize(value)
            self._client.setex(key, ttl, serialized)
            return True
        except Exception as e:
            print(f"Cache set error: {e}")
            return False

    def delete(self, key: str) -> bool:
        """Удалить запись."""
        if not self.is_enabled:
            return False
        try:
            self._client.delete(key)
            return True
        except Exception as e:
            print(f"Cache delete error: {e}")
            return False

    def clear(self) -> bool:
        """Очистить кэш."""
        if not self.is_enabled:
            return False
        try:
            self._client.flushdb()
            return True
        except Exception as e:
            print(f"Cache clear error: {e}")
            return False

    def stats(self) -> Dict[str, Any]:
        """Статистика кэша."""
        info = {}
        if self._client:
            try:
                info = self._client.info("memory")
            except Exception:
                pass
        return {
            "backend": "redis",
            "host": self.host,
            "port": self.port,
            "enabled": self._enabled,
            "ttl": self.ttl,
        }


class ForecastCache:
    """
    Умный кэш для прогнозов.

    Автоматически выбирает бэкенд и управляет ключами.

    Example:
        cache = ForecastCache(backend='redis')

        # Проверяем кэш
        key = cache.make_key('ridge', df, 12)
        result = cache.get(key)

        if result is None:
            result = model.forecast(12)
            cache.set(key, result)
    """

    def __init__(
        self,
        backend: str = "memory",
        maxsize: int = 100,
        ttl: int = 3600,
        cache_dir: str = ".cache",
        redis_host: str = "localhost",
        redis_port: int = 6379,
        redis_db: int = 0,
        redis_password: Optional[str] = None,
    ):
        """
        Инициализация.

        Args:
            backend: 'memory', 'file' или 'redis'
            maxsize: Размер кэша (для memory)
            ttl: TTL в секундах
            cache_dir: Директория (для file)
            redis_host: Redis host (для redis)
            redis_port: Redis port (для redis)
            redis_db: Redis database (для redis)
            redis_password: Redis password (для redis)
        """
        self.backend_name = backend

        if backend == "memory":
            self._backend = MemoryCache(maxsize=maxsize, ttl=ttl)
        elif backend == "file":
            self._backend = FileCache(cache_dir=cache_dir, ttl=ttl)
        elif backend == "redis":
            self._backend = RedisCache(
                host=redis_host,
                port=redis_port,
                db=redis_db,
                password=redis_password,
                ttl=ttl,
            )
            self._backend.connect()
        else:
            raise ValueError(f"Неизвестный бэкенд: {backend}")

    def make_key(
        self, model: str, df: pd.DataFrame, horizon: int, params: Optional[Dict] = None
    ) -> str:
        """Создать ключ для прогноза."""
        data_hash = CacheKey.hash_dataframe(df)
        return CacheKey.make_key(model, data_hash, horizon, params)

    def get(self, key: str) -> Optional[np.ndarray]:
        """Получить прогноз из кэша."""
        return self._backend.get(key)

    def set(self, key: str, value: np.ndarray, ttl: Optional[int] = None) -> None:
        """Сохранить прогноз в кэш."""
        self._backend.set(key, value, ttl)

    def get_or_compute(
        self,
        model: str,
        df: pd.DataFrame,
        horizon: int,
        compute_fn,
        params: Optional[Dict] = None,
    ) -> np.ndarray:
        """
        Получить из кэша или вычислить.

        Args:
            model: Название модели
            df: DataFrame
            horizon: Горизонт
            compute_fn: Функция вычисления
            params: Параметры

        Returns:
            Прогноз
        """
        key = self.make_key(model, df, horizon, params)
        result = self.get(key)

        if result is None:
            result = compute_fn()
            self.set(key, result)

        return result

    def delete(self, key: str) -> bool:
        """Удалить запись."""
        return self._backend.delete(key)

    def clear(self) -> None:
        """Очистить кэш."""
        self._backend.clear()

    def stats(self) -> Dict[str, Any]:
        """Статистика."""
        stats = self._backend.stats()
        stats["backend"] = self.backend_name
        return stats


# Глобальный кэш (синглтон)
_global_cache: Optional[ForecastCache] = None


def get_cache(backend: str = "memory", **kwargs) -> ForecastCache:
    """
    Получить глобальный кэш.

    Args:
        backend: Тип бэкенда
        **kwargs: Параметры кэша

    Returns:
        ForecastCache instance
    """
    global _global_cache

    if _global_cache is None:
        _global_cache = ForecastCache(backend=backend, **kwargs)

    return _global_cache


def clear_cache() -> None:
    """Очистить глобальный кэш."""
    global _global_cache

    if _global_cache is not None:
        _global_cache.clear()


if __name__ == "__main__":
    # Тестирование
    print("Тестирование кэша...")

    # Memory cache
    cache = ForecastCache(backend="memory", maxsize=10, ttl=60)

    # Тестовые данные
    df = pd.DataFrame(
        {"Все товары и услуги": [100.5, 100.3, 100.4]},
        index=pd.date_range("2024-01-01", periods=3, freq="MS"),
    )

    key = cache.make_key("ridge", df, 12)
    print(f"Ключ: {key}")

    # Записываем
    forecast = np.array([0.5, 0.4, 0.6])
    cache.set(key, forecast)
    print(f"Записано: {forecast}")

    # Читаем
    result = cache.get(key)
    print(f"Прочитано: {result}")

    # Статистика
    print(f"Статистика: {cache.stats()}")
