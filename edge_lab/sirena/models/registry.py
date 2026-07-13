"""
Реестр моделей прогнозирования СИРЕНА-КБР
==========================================

Factory pattern для управления моделями.
Позволяет регистрировать, получать и перечислять модели.
"""

from typing import Dict, Type, List, Optional, Any
from .base import BaseForecaster


class ModelRegistry:
    """
    Реестр моделей прогнозирования.

    Использование:
        # Регистрация модели
        @ModelRegistry.register("ridge")
        class RidgeModel(BaseForecaster):
            ...

        # Получение модели
        model = ModelRegistry.get("ridge")

        # Список моделей
        models = ModelRegistry.list_models()
    """

    _models: Dict[str, Type[BaseForecaster]] = {}
    _default_weights: Dict[str, float] = {
        "ridge": 0.40,
        "bvar": 0.20,
        "lightgbm": 0.15,
        "prophet": 0.10,
        "sarima": 0.05,
        "ets": 0.05,
        "lstm": 0.05,
    }

    @classmethod
    def register(cls, name: str):
        """
        Декоратор для регистрации модели.

        Args:
            name: Уникальное имя модели

        Example:
            @ModelRegistry.register("my_model")
            class MyModel(BaseForecaster):
                ...
        """

        def decorator(model_class: Type[BaseForecaster]) -> Type[BaseForecaster]:
            if not issubclass(model_class, BaseForecaster):
                raise TypeError(
                    f"{model_class.__name__} должен наследоваться от BaseForecaster"
                )

            if name in cls._models:
                raise ValueError(f"Модель '{name}' уже зарегистрирована")

            model_class.name = name
            cls._models[name] = model_class

            return model_class

        return decorator

    @classmethod
    def register_model(cls, name: str, model_class: Type[BaseForecaster]):
        """
        Программная регистрация модели (без декоратора).

        Args:
            name: Уникальное имя модели
            model_class: Класс модели
        """
        if not issubclass(model_class, BaseForecaster):
            raise TypeError(
                f"{model_class.__name__} должен наследоваться от BaseForecaster"
            )

        model_class.name = name
        cls._models[name] = model_class

    @classmethod
    def get(cls, name: str, **kwargs) -> BaseForecaster:
        """
        Получить экземпляр модели по имени.

        Args:
            name: Имя зарегистрированной модели
            **kwargs: Параметры для конструктора модели

        Returns:
            Экземпляр модели

        Raises:
            KeyError: Если модель не найдена
        """
        if name not in cls._models:
            available = ", ".join(cls._models.keys()) or "нет моделей"
            raise KeyError(f"Модель '{name}' не найдена. Доступные: {available}")

        return cls._models[name](**kwargs)

    @classmethod
    def get_class(cls, name: str) -> Type[BaseForecaster]:
        """
        Получить класс модели (не экземпляр).

        Args:
            name: Имя модели

        Returns:
            Класс модели
        """
        if name not in cls._models:
            raise KeyError(f"Модель '{name}' не найдена")

        return cls._models[name]

    @classmethod
    def list_models(cls) -> List[str]:
        """
        Список зарегистрированных моделей.

        Returns:
            Список имён моделей
        """
        return list(cls._models.keys())

    @classmethod
    def is_registered(cls, name: str) -> bool:
        """Проверка регистрации модели."""
        return name in cls._models

    @classmethod
    def unregister(cls, name: str) -> bool:
        """
        Удалить модель из реестра.

        Returns:
            True если модель была удалена
        """
        if name in cls._models:
            del cls._models[name]
            return True
        return False

    @classmethod
    def clear(cls):
        """Очистить реестр (для тестов)."""
        cls._models.clear()

    @classmethod
    def get_default_weight(cls, name: str) -> float:
        """Получить дефолтный вес модели в ансамбле."""
        return cls._default_weights.get(name, 0.0)

    @classmethod
    def set_default_weight(cls, name: str, weight: float):
        """Установить дефолтный вес модели."""
        if not 0 <= weight <= 1:
            raise ValueError("Вес должен быть от 0 до 1")
        cls._default_weights[name] = weight

    @classmethod
    def get_all_weights(cls) -> Dict[str, float]:
        """Получить все дефолтные веса."""
        return cls._default_weights.copy()

    @classmethod
    def create_ensemble(
        cls,
        models: Optional[List[str]] = None,
        weights: Optional[Dict[str, float]] = None,
        **kwargs,
    ) -> Dict[str, BaseForecaster]:
        """
        Создать набор моделей для ансамбля.

        Args:
            models: Список моделей (если None - все зарегистрированные)
            weights: Веса моделей (опционально)
            **kwargs: Параметры для всех моделей

        Returns:
            Dict с экземплярами моделей
        """
        if models is None:
            models = cls.list_models()

        ensemble = {}
        for name in models:
            if cls.is_registered(name):
                try:
                    ensemble[name] = cls.get(name, **kwargs)
                except Exception as e:
                    print(f"Ошибка создания модели {name}: {e}")

        return ensemble

    @classmethod
    def info(cls) -> Dict[str, Any]:
        """
        Информация о реестре.

        Returns:
            Dict с информацией о моделях
        """
        return {
            "registered_models": cls.list_models(),
            "model_count": len(cls._models),
            "default_weights": cls._default_weights,
            "models_info": {
                name: {
                    "class": model_class.__name__,
                    "min_train_size": getattr(model_class, "MIN_TRAIN_SIZE", 24),
                    "weight": cls._default_weights.get(name, 0),
                }
                for name, model_class in cls._models.items()
            },
        }


registry = ModelRegistry
