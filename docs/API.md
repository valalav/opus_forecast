# СИРЕНА-КБР REST API v4.0

Документация REST API для системы прогнозирования инфляции.

> **Note**: API (v4.0) динамически загружает модели из реестра. Актуальный список моделей (v5.2) доступен через эндпоинт `GET /models`.

## Содержание

- [Обзор](#обзор)
- [Аутентификация](#аутентификация)
- [Эндпоинты](#эндпоинты)
  - [Health](#health)
  - [Models](#models)
  - [Forecast](#forecast)
  - [Backtest](#backtest)
    - [POST /backtest](#post-backtest)
    - [GET /backtest/metrics/{model_name}](#get-backtestmetricsmodel_name)
    - [GET /backtest/history](#get-backtesthistory)
  - [Metrics](#metrics)
- [Схемы данных](#схемы-данных)
- [Коды ошибок](#коды-ошибок)
- [Примеры](#примеры)

---

## Обзор

**Base URL:** `http://localhost:8000`

**Формат:** JSON

**Документация:**
- Swagger UI: `/docs`
- ReDoc: `/redoc`
- OpenAPI JSON: `/openapi.json`

---

## Аутентификация

В текущей версии API не требует аутентификации.

---

## Эндпоинты

### Health

#### GET /health

Проверка статуса сервера для мониторинга и load balancer'ов.

Возвращает API статус, версию, количество доступных моделей и статус загрузки данных.

**Response:**
```json
{
  "status": "ok",
  "version": "4.0.0",
  "models_available": 25,
  "data_loaded": true
}
```

**Response Fields:**
| Поле | Тип | Описание |
|------|-----|----------|
| status | string | Статус API ("ok" или "error") |
| version | string | Версия API |
| models_available | int | Количество зарегистрированных моделей |
| data_loaded | bool | Загружены ли данные |

#### GET /health/detailed

Расширенная проверка статуса с информацией о времени работы (uptime).

Используется мониторинговыми дашбордами для отслеживания стабильности сервиса.

**Response:**
```json
{
  "status": "ok",
  "version": "4.0.0",
  "uptime_seconds": 86400.0,
  "uptime_hours": 24.0,
  "uptime_days": 1.0,
  "models_available": 25,
  "data_loaded": true
}
```

**Response Fields:**
| Поле | Тип | Описание |
|------|-----|----------|
| status | string | Статус API ("ok" или "error") |
| version | string | Версия API |
| uptime_seconds | float | Время работы в секундах |
| uptime_hours | float | Время работы в часах |
| uptime_days | float | Время работы в днях |
| models_available | int | Количество зарегистрированных моделей |
| data_loaded | bool | Загружены ли данные |
| error | string | Ошибка (только при status="error") |

---

### Models

#### GET /models

Список всех доступных моделей с весами в ансамбле и MAE из backtest h=1.

Возвращает информацию о каждой модели из реестра, включая вес в ансамбле и MAE из последнего бэктеста (горизонт 1 месяц).

**Response:**
```json
{
  "models": [
    {
      "name": "ridge",
      "weight": 0.40,
      "min_train_size": 36,
      "description": "Ridge регрессия с ETS сезонной компонентой",
      "mae": 0.321
    },
    {
      "name": "bvar",
      "weight": 0.20,
      "min_train_size": 24,
      "description": "Байесовская VAR с Minnesota Prior",
      "mae": 0.345
    },
    {
      "name": "subcomp_multi",
      "weight": 0.12,
      "min_train_size": 24,
      "description": "Многомодельная агрегация субкомпонентов",
      "mae": 0.297
    }
  ]
}
```

**Response Fields:**
| Поле | Тип | Описание |
|------|-----|----------|
| models | array | Список доступных моделей |
| models[].name | string | Имя модели (реестра) |
| models[].weight | float | Вес модели в ансамбле |
| models[].min_train_size | int | Минимальный размер обучающей выборки |
| models[].description | string | Описание модели |
| models[].mae | float\|null | MAE из backtest h=1 (null если нет данных) |

**Note:** Список моделей динамически загружается из `ModelRegistry`. MAE берётся из `archive/results/backtest_h1_metrics.csv`.

#### GET /models/{model_name}

Информация о конкретной модели с MAE из backtest h=1.

Возвращает детальную информацию о модели, включая метрики качества из последнего бэктеста.

**Parameters:**
- `model_name` (path) — название модели (реестра)

**Response:**
```json
{
  "name": "ridge",
  "weight": 0.40,
  "min_train_size": 36,
  "description": "Ridge регрессия с ETS сезонной компонентой",
  "mae": 0.321
}
```

**Response Fields:**
| Поле | Тип | Описание |
|------|-----|----------|
| name | string | Имя модели (реестра) |
| weight | float | Вес модели в ансамбле |
| min_train_size | int | Минимальный размер обучающей выборки |
| description | string | Описание модели |
| mae | float\|null | MAE из backtest h=1 (null если нет данных) |

**Errors:**
- `404` — модель не найдена в реестре
- `500` — ошибка при загрузке метрик бэктеста

---

### Forecast

#### POST /forecast

Создать прогноз инфляции.

**Request Body:**
```json
{
  "horizon": 12,
  "models": ["ridge", "bvar", "lightgbm"],
  "weights": {
    "ridge": 0.5,
    "bvar": 0.3,
    "lightgbm": 0.2
  },
  "include_intervals": false
}
```

**Parameters:**
| Поле | Тип | По умолчанию | Описание |
|------|-----|--------------|----------|
| horizon | int | 12 | Горизонт прогноза (1-24) |
| models | array | null | Список моделей (null = все) |
| weights | object | null | Кастомные веса |
| include_intervals | bool | false | Доверительные интервалы |

**Response:**
```json
{
  "ensemble": {
    "values": [0.45, 0.38, 0.52, 0.41, 0.55, 0.48, 0.42, 0.50, 0.45, 0.38, 0.52, 0.46],
    "dates": ["2025-01", "2025-02", "2025-03", "2025-04", "2025-05", "2025-06",
              "2025-07", "2025-08", "2025-09", "2025-10", "2025-11", "2025-12"]
  },
  "models": {
    "ridge": {
      "model": "ridge",
      "values": [0.48, 0.40, 0.55, ...],
      "dates": ["2025-01", "2025-02", ...],
      "weight": 0.5
    },
    "bvar": {
      "model": "bvar",
      "values": [0.42, 0.35, 0.48, ...],
      "dates": ["2025-01", "2025-02", ...],
      "weight": 0.3
    },
    "lightgbm": {
      "model": "lightgbm",
      "values": [0.44, 0.38, 0.52, ...],
      "dates": ["2025-01", "2025-02", ...],
      "weight": 0.2
    }
  },
  "data_date": "2024-12",
  "version": "4.0"
}
```

#### GET /forecast/quick

Быстрый прогноз только Ridge моделью.

**Parameters:**
- `horizon` (query) — горизонт прогноза (по умолчанию 3)

**Response:**
```json
{
  "values": [0.48, 0.42, 0.55],
  "dates": ["2025-01", "2025-02", "2025-03"],
  "model": "ridge",
  "data_date": "2024-12"
}
```

#### POST /forecast/batch

Пакетное прогнозирование для нескольких сценариев.

Принимает список сценариев с разными параметрами (горизонт, модели, веса) и возвращает прогноз для каждого сценария.

**Use Cases:**
- Сравнить разные горизонты (h=6 vs h=12)
- Сравнить разные модели (Ridge vs LightGBM)
- Сравнить разные веса (равные vs оптимизированные)

**Request Body:**
```json
{
  "scenarios": [
    {
      "horizon": 12,
      "models": ["ridge", "bvar"],
      "weights": {"ridge": 0.5, "bvar": 0.5},
      "include_intervals": false
    },
    {
      "horizon": 6,
      "models": ["lightgbm", "prophet"],
      "weights": {"lightgbm": 0.7, "prophet": 0.3},
      "include_intervals": true
    }
  ]
}
```

**Parameters:**
| Поле | Тип | По умолчанию | Описание |
|------|-----|--------------|----------|
| scenarios | array | required | Список сценариев (1-10) |
| scenarios[].horizon | int | 12 | Горизонт прогноза (1-24) |
| scenarios[].models | array | null | Список моделей (null = все) |
| scenarios[].weights | object | null | Кастомные веса для моделей |
| scenarios[].include_intervals | bool | false | Доверительные интервалы |

**Response:**
```json
{
  "results": [
    {
      "ensemble": {
        "values": [0.45, 0.38, 0.52, 0.41, 0.55, 0.48, 0.42, 0.50, 0.45, 0.38, 0.52, 0.46],
        "dates": ["2025-01", "2025-02", "2025-03", "2025-04", "2025-05", "2025-06",
                  "2025-07", "2025-08", "2025-09", "2025-10", "2025-11", "2025-12"]
      },
      "models": {
        "ridge": {
          "model": "ridge",
          "values": [0.48, 0.40, 0.55, ...],
          "dates": ["2025-01", "2025-02", ...],
          "weight": 0.5
        },
        "bvar": {
          "model": "bvar",
          "values": [0.42, 0.35, 0.48, ...],
          "dates": ["2025-01", "2025-02", ...],
          "weight": 0.5
        }
      },
      "data_date": "2024-12",
      "version": "4.0"
    },
    {
      "ensemble": {
        "values": [0.50, 0.42, 0.55, 0.48, 0.52, 0.45],
        "dates": ["2025-01", "2025-02", "2025-03", "2025-04", "2025-05", "2025-06"]
      },
      "models": {
        "lightgbm": {
          "model": "lightgbm",
          "values": [0.48, 0.40, 0.52, ...],
          "dates": ["2025-01", "2025-02", ...],
          "weight": 0.7
        },
        "prophet": {
          "model": "prophet",
          "values": [0.52, 0.45, 0.60, ...],
          "dates": ["2025-01", "2025-02", ...],
          "weight": 0.3
        }
      },
      "data_date": "2024-12",
      "version": "4.0"
    }
  ],
  "count": 2,
  "version": "4.0"
}
```

**Errors:**
- `400` — Bad Request (некорректные параметры)
- `500` — Internal Server Error (все сценарии завершились с ошибкой)

---

### Backtest

#### POST /backtest

Запустить бэктестирование модели.

**Request Body:**
```json
{
  "model": "ridge",
  "start_date": "2023-01-01",
  "end_date": "2024-12-01"
}
```

**Parameters:**
| Поле | Тип | По умолчанию | Описание |
|------|-----|--------------|----------|
| model | string | required | Название модели |
| start_date | string | "2019-01-01" | Начало периода (YYYY-MM-DD) |
| end_date | string | null | Конец периода |

**Response:**
```json
{
  "model": "ridge",
  "start_date": "2023-01-01",
  "end_date": "2024-12-01",
  "metrics": {
    "MAE": 0.2845,
    "RMSE": 0.3521,
    "KPI": 87.5,
    "count": 24
  },
  "results": [
    {
      "date": "2023-01",
      "actual": 0.52,
      "prediction": 0.48,
      "error": 0.04
    },
    {
      "date": "2023-02",
      "actual": 0.38,
      "prediction": 0.42,
      "error": -0.04
    }
  ]
}
```

#### GET /backtest/metrics/{model_name}

Получить только метрики бэктеста.

**Parameters:**
- `model_name` (path) — название модели
- `start_date` (query) — начало периода (по умолчанию "2019-01-01")

**Response:**
```json
{
  "model": "ridge",
  "MAE": 0.2845,
  "RMSE": 0.3521,
  "KPI": 87.5,
  "count": 72
}
```

#### GET /backtest/history

Получить исторические прогнозы из бэктеста для проверки точности.

Загружает данные из CSV файлов с результатами бэктестов (`backtest_h{horizon}_predictions.csv`).

**Use Cases:**
- Проверить точность исторических прогнозов
- Сравнить фактические значения с прогнозами
- Анализировать ошибки по временному периоду

**Parameters:**
| Параметр | Тип | По умолчанию | Описание |
|-----------|------|--------------|----------|
| start_date | string | null | Начало периода (YYYY-MM-DD) |
| end_date | string | null | Конец периода (YYYY-MM-DD) |
| model | string | null | Фильтр по названию модели (null = Ensemble) |
| horizon | int | 1 | Горизонт прогноза (1, 2, или 12) |

**Response:**
```json
{
  "count": 24,
  "horizon": 1,
  "model": "ridge",
  "start_date": "2024-01-01",
  "end_date": "2025-12-01",
  "data": [
    {
      "date": "2024-01-01",
      "actual": 0.52,
      "prediction": 0.48,
      "error": 0.04
    },
    {
      "date": "2024-02-01",
      "actual": 0.38,
      "prediction": 0.42,
      "error": -0.04
    }
  ]
}
```

**Response Fields:**
| Поле | Тип | Описание |
|------|-----|----------|
| count | int | Количество записей |
| horizon | int | Горизонт прогноза (1, 2, или 12) |
| model | string\|null | Фильтр модели (null если не указан) |
| start_date | string | Начало периода |
| end_date | string | Конец периода |
| data | array | Список записей с прогнозами |
| data[].date | string | Дата (YYYY-MM-DD) |
| data[].actual | float | Фактическое значение |
| data[].prediction | float | Прогноз модели |
| data[].error | float | Ошибка (actual - prediction) |

**Errors:**
- `400` — неверный горизонт (должен быть 1, 2, или 12)
- `404` — файл бэктеста не найден или модель не существует
- `500` — ошибка сервера

---

### Metrics

#### GET /metrics

Метрики качества всех моделей.

**Response:**
```json
{
  "period": "2023-01 — present",
  "metrics": {
    "ridge": {
      "MAE": 0.2845,
      "KPI": 87.5,
      "count": 24
    },
    "bvar": {
      "MAE": 0.3521,
      "KPI": 78.2,
      "count": 24
    },
    "lightgbm": {
      "MAE": 0.3156,
      "KPI": 82.1,
      "count": 24
    }
  }
}
```

---

## Схемы данных

### HealthResponse

```json
{
  "status": "string (ok|error)",
  "version": "string",
  "models_available": "integer",
  "data_loaded": "boolean"
}
```

### HealthDetailedResponse

```json
{
  "status": "string (ok|error)",
  "version": "string",
  "uptime_seconds": "float",
  "uptime_hours": "float",
  "uptime_days": "float",
  "models_available": "integer",
  "data_loaded": "boolean",
  "error": "string | null"
}
```

### ModelInfo

```json
{
  "name": "string",
  "weight": "float",
  "min_train_size": "integer",
  "description": "string",
  "mae": "float | null"
}
```

### ModelsListResponse

```json
{
  "models": [ModelInfo],
  "total_weight": "float"
}
```

### ForecastRequest

```json
{
  "horizon": "integer (1-24)",
  "models": ["string"] | null,
  "weights": {"model": "float"} | null,
  "include_intervals": "boolean"
}
```

### ForecastResponse

```json
{
  "ensemble": {
    "values": ["float"],
    "dates": ["string"],
    "lower": ["float"] | null,
    "upper": ["float"] | null
  },
  "models": {
    "model_name": {
      "model": "string",
      "values": ["float"],
      "dates": ["string"],
      "weight": "float"
    }
  },
  "data_date": "string",
  "version": "string"
}
```

### BacktestRequest

```json
{
  "model": "string (required)",
  "start_date": "string (YYYY-MM-DD)",
  "end_date": "string (YYYY-MM-DD)" | null
}
```

### BacktestResponse

```json
{
  "model": "string",
  "start_date": "string",
  "end_date": "string",
  "metrics": {
    "MAE": "float",
    "RMSE": "float",
    "KPI": "float",
    "count": "integer"
  },
  "results": [{
    "date": "string",
    "actual": "float",
    "prediction": "float",
    "error": "float"
  }]
}
```

### BatchForecastRequest

```json
{
  "scenarios": [
    {
      "horizon": "integer (1-24)",
      "models": ["string"] | null,
      "weights": {"model": "float"} | null,
      "include_intervals": "boolean"
    }
  ],
  "min_length": 1,
  "max_length": 10
}
```

### BatchForecastResponse

```json
{
  "results": [ForecastResponse],
  "count": "integer",
  "version": "string"
}
```

### HistoryRequest

```json
{
  "start_date": "string (YYYY-MM-DD) | null",
  "end_date": "string (YYYY-MM-DD) | null",
  "model": "string | null",
  "horizon": "integer (1, 2, или 12)"
}
```

### HistoryEntry

```json
{
  "date": "string (YYYY-MM-DD)",
  "actual": "float",
  "prediction": "float",
  "error": "float"
}
```

### HistoryResponse

```json
{
  "count": "integer",
  "horizon": "integer",
  "model": "string | null",
  "start_date": "string",
  "end_date": "string",
  "data": [HistoryEntry]
}
```

---

## Коды ошибок

| Код | Описание |
|-----|----------|
| 200 | OK |
| 400 | Bad Request — неверные параметры |
| 404 | Not Found — ресурс не найден |
| 422 | Validation Error — ошибка валидации |
| 500 | Internal Server Error — ошибка сервера |

### Формат ошибки

```json
{
  "detail": "Описание ошибки"
}
```

---

## Примеры

### Python (requests)

```python
import requests

# Прогноз
response = requests.post(
    "http://localhost:8000/forecast",
    json={
        "horizon": 6,
        "models": ["ridge", "bvar"]
    }
)
data = response.json()
print(data["ensemble"]["values"])

# Пакетный прогноз (сравнение сценариев)
response = requests.post(
    "http://localhost:8000/forecast/batch",
    json={
        "scenarios": [
            {
                "horizon": 12,
                "models": ["ridge", "bvar"],
                "weights": {"ridge": 0.7, "bvar": 0.3}
            },
            {
                "horizon": 12,
                "models": ["lightgbm", "prophet"],
                "weights": {"lightgbm": 0.6, "prophet": 0.4}
            }
        ]
    }
)
data = response.json()
print(f"Обработано сценариев: {data['count']}")
for i, result in enumerate(data['results']):
    print(f"Сценарий {i}: {result['ensemble']['values']}")

# Бэктест
response = requests.post(
    "http://localhost:8000/backtest",
    json={
        "model": "ridge",
        "start_date": "2024-01-01"
    }
)
metrics = response.json()["metrics"]
print(f"MAE: {metrics['MAE']}")

# Исторические прогнозы
response = requests.get(
    "http://localhost:8000/backtest/history",
    params={
        "model": "ridge",
        "horizon": 1,
        "start_date": "2024-01-01",
        "end_date": "2024-12-01"
    }
)
history = response.json()
print(f"Записей: {history['count']}")
for entry in history['data']:
    print(f"{entry['date']}: actual={entry['actual']}, pred={entry['prediction']}, error={entry['error']}")
```

### JavaScript (fetch)

```javascript
// Прогноз
const response = await fetch('http://localhost:8000/forecast', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    horizon: 12,
    models: ['ridge', 'bvar', 'lightgbm']
  })
});
const data = await response.json();
console.log(data.ensemble.values);

// Список моделей
const models = await fetch('http://localhost:8000/models');
console.log(await models.json());

// Исторические прогнозы
const history = await fetch('http://localhost:8000/backtest/history?model=ridge&horizon=1');
const historyData = await history.json();
console.log(`Записей: ${historyData.count}`);
historyData.data.forEach(entry => {
  console.log(`${entry.date}: actual=${entry.actual}, pred=${entry.prediction}`);
});
```

### curl

```bash
# Health check (basic)
curl http://localhost:8000/health

# Health check (detailed with uptime)
curl http://localhost:8000/health/detailed

# Список всех моделей
curl http://localhost:8000/models

# Информация о конкретной модели
curl http://localhost:8000/models/ridge
curl http://localhost:8000/models/subcomp_multi

# Быстрый прогноз
curl "http://localhost:8000/forecast/quick?horizon=6"

# Полный прогноз
curl -X POST http://localhost:8000/forecast \
  -H "Content-Type: application/json" \
  -d '{"horizon": 12}'

# Пакетный прогноз (2 сценария)
curl -X POST http://localhost:8000/forecast/batch \
  -H "Content-Type: application/json" \
  -d '{
    "scenarios": [
      {
        "horizon": 12,
        "models": ["ridge", "bvar"],
        "weights": {"ridge": 0.7, "bvar": 0.3}
      },
      {
        "horizon": 6,
        "models": ["lightgbm"],
        "include_intervals": true
      }
    ]
  }'

# Бэктест
curl -X POST http://localhost:8000/backtest \
  -H "Content-Type: application/json" \
  -d '{"model": "ridge", "start_date": "2023-01-01"}'

# Исторические прогнозы (h=1, по умолчанию)
curl "http://localhost:8000/backtest/history"

# Исторические прогнозы с фильтром по модели
curl "http://localhost:8000/backtest/history?model=ridge&horizon=1"

# Исторические прогнозы с диапазоном дат
curl "http://localhost:8000/backtest/history?start_date=2024-01-01&end_date=2024-12-01&horizon=1"
```

---

## Rate Limits

В текущей версии лимиты запросов не установлены.

---

## Версионирование

API версия указана в поле `version` ответов и в OpenAPI спецификации.

Текущая версия: **4.0.0**
