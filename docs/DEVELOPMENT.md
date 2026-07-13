# Руководство разработчика СИРЕНА-КБР

Инструкции по разработке и расширению системы.

## Содержание

- [Настройка окружения](#настройка-окружения)
- [Структура проекта](#структура-проекта)
- [Архитектурные принципы](#архитектурные-принципы)
- [Добавление модели](#добавление-модели)
- [Добавление API эндпоинта](#добавление-api-эндпоинта)
- [Тестирование](#тестирование)
- [Стиль кода](#стиль-кода)
- [CI/CD](#cicd)

---

## Настройка окружения

### Требования

- Python 3.10+
- pip или conda

### Установка

```bash
# Клонирование
cd /home/valalav
git clone <repo> opus_forecast
cd opus_forecast

# Виртуальное окружение (опционально)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# Зависимости
pip install -r requirements.txt

# Опциональные (для LSTM)
pip install tensorflow

# Для разработки
pip install pytest pytest-cov black isort mypy
```

### Проверка установки

```bash
# Тесты
pytest tests/ -v

# Запуск API
uvicorn api.main:app --reload

# Запуск Dashboard
streamlit run dashboard.py
```

---

## Структура проекта

```
opus_forecast/
├── api/                      # REST API
│   ├── __init__.py
│   ├── main.py               # FastAPI приложение
│   ├── routes/               # Эндпоинты
│   │   ├── __init__.py
│   │   ├── forecast.py
│   │   ├── backtest.py
│   │   └── models.py
│   └── schemas/              # Pydantic схемы
│       ├── __init__.py
│       ├── forecast.py
│       ├── backtest.py
│       └── models.py
│
├── sirena/                   # Ядро системы
│   ├── __init__.py
│   ├── forecast.py           # EnsembleForecaster
│   ├── async_runner.py       # Параллельный запуск
│   ├── cache.py              # Кэширование
│   └── models/               # Модели
│       ├── __init__.py
│       ├── base.py           # BaseForecaster
│       ├── registry.py       # ModelRegistry
│       └── ... (см. MODELS.md для полного списка)

│
├── data/                     # Данные
│   └── infl_kbr.csv
│
├── docs/                     # Документация
│   ├── API.md
│   ├── MODELS.md
│   └── DEVELOPMENT.md
│
├── tests/                    # Тесты
│   ├── __init__.py
│   ├── test_models.py
│   └── test_api.py
│
├── dashboard.py              # Streamlit UI
├── requirements.txt
└── README.md
```

---

## Архитектурные принципы

### 1. Абстракция моделей

Все модели наследуют `BaseForecaster`:

```python
class BaseForecaster(ABC):
    @abstractmethod
    def fit(self, df, target_col) -> 'BaseForecaster': ...

    @abstractmethod
    def forecast(self, horizon) -> np.ndarray: ...

    @abstractmethod
    def backtest(self, df, start_date, target_col) -> pd.DataFrame: ...
```

### 2. Factory Pattern (ModelRegistry)

```python
# Регистрация
@ModelRegistry.register("my_model")
class MyModel(BaseForecaster): ...

# Получение
model = ModelRegistry.get("my_model")
```

### 3. Dependency Injection в API

```python
# Данные загружаются через функцию
def get_data():
    return load_csv()

@router.post("/forecast")
async def forecast(data = Depends(get_data)):
    ...
```

### 4. Разделение ответственности

- `sirena/models/` — логика моделей
- `api/routes/` — HTTP обработка
- `api/schemas/` — валидация данных
- `sirena/cache.py` — кэширование
- `sirena/async_runner.py` — параллелизация

---

## Добавление модели

### Шаг 1: Создайте файл модели

```python
# sirena/models/xgboost.py

import numpy as np
import pandas as pd
from typing import Dict, Any

from .base import BaseForecaster
from .registry import ModelRegistry


@ModelRegistry.register("xgboost")
class XGBoostForecaster(BaseForecaster):
    """XGBoost модель прогнозирования."""

    name = "xgboost"
    MIN_TRAIN_SIZE = 36

    def __init__(self, n_estimators: int = 100, **kwargs):
        super().__init__(**kwargs)
        self.n_estimators = n_estimators
        self.model = None

    def fit(self, df: pd.DataFrame, target_col: str = 'Все товары и услуги') -> 'XGBoostForecaster':
        """Обучение модели."""
        # Валидация входных данных
        series = self._validate_data(df, target_col)

        # Подготовка признаков
        X, y = self._prepare_features(df, target_col)

        # Обучение
        import xgboost as xgb
        self.model = xgb.XGBRegressor(n_estimators=self.n_estimators)
        self.model.fit(X, y)

        self._is_fitted = True
        self._last_train_date = df.index.max()

        return self

    def forecast(self, horizon: int = 12) -> np.ndarray:
        """Прогноз на горизонт."""
        self._check_fitted()

        predictions = []
        # ... рекурсивный прогноз ...

        return np.array(predictions)

    def backtest(
        self,
        df: pd.DataFrame,
        start_date: str = '2019-01-01',
        target_col: str = 'Все товары и услуги'
    ) -> pd.DataFrame:
        """Бэктестирование."""
        results = []
        # ... скользящий бэктест ...

        return pd.DataFrame(results)

    def _prepare_features(self, df, target_col):
        """Подготовка признаков."""
        # ... ваша логика ...
        return X, y
```

### Шаг 2: Добавьте импорт

```python
# sirena/models/__init__.py

from .xgboost import XGBoostForecaster

__all__ = [
    ...
    'XGBoostForecaster',
]
```

### Шаг 3: Установите вес (опционально)

```python
# sirena/models/registry.py

_default_weights: Dict[str, float] = {
    ...
    'xgboost': 0.05,  # добавьте
}
```

### Шаг 4: Добавьте тесты

```python
# tests/test_models.py

class TestXGBoost:
    @pytest.fixture
    def sample_data(self):
        dates = pd.date_range('2020-01-01', periods=48, freq='MS')
        return pd.DataFrame({
            'Все товары и услуги': 100.5 + np.random.randn(48) * 0.3
        }, index=dates)

    def test_import(self):
        from sirena.models import XGBoostForecaster
        assert XGBoostForecaster is not None

    def test_fit(self, sample_data):
        from sirena.models import XGBoostForecaster
        model = XGBoostForecaster()
        model.fit(sample_data)
        assert model.is_fitted

    def test_forecast(self, sample_data):
        from sirena.models import XGBoostForecaster
        model = XGBoostForecaster()
        model.fit(sample_data)
        fc = model.forecast(horizon=6)
        assert len(fc) == 6
```

### Шаг 5: Обновите зависимости

```
# requirements.txt
xgboost>=2.0.0
```

---

## Добавление API эндпоинта

### Шаг 1: Создайте схему

```python
# api/schemas/analysis.py

from pydantic import BaseModel, Field
from typing import List, Optional


class AnalysisRequest(BaseModel):
    model: str = Field(description="Модель для анализа")
    period: str = Field(default="2024", description="Период")


class AnalysisResponse(BaseModel):
    model: str
    feature_importance: List[dict]
    diagnostics: dict
```

### Шаг 2: Создайте роут

```python
# api/routes/analysis.py

from fastapi import APIRouter, HTTPException
from ..schemas.analysis import AnalysisRequest, AnalysisResponse

router = APIRouter(prefix="/analysis", tags=["Analysis"])


@router.post("", response_model=AnalysisResponse)
async def run_analysis(request: AnalysisRequest):
    """Анализ модели."""
    try:
        from sirena.models import ModelRegistry

        if not ModelRegistry.is_registered(request.model):
            raise HTTPException(404, f"Модель не найдена")

        model = ModelRegistry.get(request.model)
        # ... анализ ...

        return AnalysisResponse(
            model=request.model,
            feature_importance=[...],
            diagnostics={...}
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, str(e))
```

### Шаг 3: Зарегистрируйте роут

```python
# api/routes/__init__.py

from .analysis import router as analysis_router

__all__ = [..., 'analysis_router']
```

```python
# api/main.py

from api.routes import analysis_router

app.include_router(analysis_router)
```

### Шаг 4: Добавьте тест

```python
# tests/test_api.py

class TestAnalysisEndpoint:
    def test_analysis(self):
        response = client.post("/analysis", json={"model": "ridge"})
        assert response.status_code == 200
```

---

## Тестирование

### Запуск тестов

```bash
# Все тесты
pytest tests/ -v

# С покрытием
pytest tests/ --cov=sirena --cov=api --cov-report=html

# Конкретный файл
pytest tests/test_models.py -v

# Конкретный тест
pytest tests/test_models.py::TestRidge::test_fit -v

# С выводом print
pytest tests/ -v -s
```

### Структура тестов

```python
import pytest
import numpy as np
import pandas as pd


class TestMyFeature:
    """Группа связанных тестов."""

    @pytest.fixture
    def sample_data(self):
        """Фикстура с тестовыми данными."""
        return pd.DataFrame({...})

    def test_basic_functionality(self, sample_data):
        """Тест базовой функциональности."""
        result = my_function(sample_data)
        assert result is not None

    def test_edge_case(self):
        """Тест граничного случая."""
        with pytest.raises(ValueError):
            my_function(empty_data)

    @pytest.mark.slow
    def test_heavy_computation(self, sample_data):
        """Долгий тест (пометка @slow)."""
        result = heavy_function(sample_data)
        assert result['status'] == 'ok'
```

### Моки

```python
from unittest.mock import Mock, patch


def test_with_mock():
    with patch('sirena.models.ridge.Ridge') as mock_ridge:
        mock_ridge.return_value.fit.return_value = Mock()
        mock_ridge.return_value.predict.return_value = [0.5]

        model = RidgeForecaster()
        model.fit(data)
        result = model.forecast(1)

        assert result[0] == 0.5
```

---

## Стиль кода

### Форматирование

```bash
# Black (форматирование)
black sirena/ api/ tests/

# isort (сортировка импортов)
isort sirena/ api/ tests/

# Проверка типов
mypy sirena/ api/
```

### Конвенции

1. **Именование:**
   - Классы: `CamelCase` (`RidgeForecaster`)
   - Функции/методы: `snake_case` (`get_forecast`)
   - Константы: `UPPER_CASE` (`MIN_TRAIN_SIZE`)

2. **Документация:**
   - Docstrings для всех публичных методов
   - Type hints для аргументов и возвращаемых значений

3. **Импорты:**
   ```python
   # Стандартная библиотека
   import os
   from typing import Dict, List

   # Сторонние пакеты
   import numpy as np
   import pandas as pd

   # Локальные модули
   from .base import BaseForecaster
   ```

---

## CI/CD

### GitHub Actions (пример)

```yaml
# .github/workflows/test.yml

name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest

    steps:
    - uses: actions/checkout@v4

    - name: Set up Python
      uses: actions/setup-python@v5
      with:
        python-version: '3.10'

    - name: Install dependencies
      run: |
        pip install -r requirements.txt
        pip install pytest pytest-cov

    - name: Run tests
      run: pytest tests/ -v --cov=sirena --cov=api

    - name: Upload coverage
      uses: codecov/codecov-action@v3
```

### Pre-commit hooks

```yaml
# .pre-commit-config.yaml

repos:
  - repo: https://github.com/psf/black
    rev: 23.12.0
    hooks:
      - id: black

  - repo: https://github.com/pycqa/isort
    rev: 5.13.0
    hooks:
      - id: isort

  - repo: local
    hooks:
      - id: pytest
        name: pytest
        entry: pytest tests/ -v
        language: system
        pass_filenames: false
```

---

## Отладка

### Логирование

```python
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MyModel:
    def fit(self, df):
        logger.info(f"Fitting model with {len(df)} rows")
        # ...
        logger.debug(f"Features: {self.features}")
```

### Профилирование

```python
import cProfile
import pstats

profiler = cProfile.Profile()
profiler.enable()

# Код для профилирования
model.fit(df)
forecast = model.forecast(12)

profiler.disable()
stats = pstats.Stats(profiler)
stats.sort_stats('cumulative')
stats.print_stats(20)
```

### Отладка API

```python
# Запуск с подробным логом
uvicorn api.main:app --reload --log-level debug
```
