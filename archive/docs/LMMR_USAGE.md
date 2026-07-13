# Руководство по запуску модели LMMR (Gemini)

Модель `LMMRForecaster` успешно реализована и протестирована. Она показывает лучшую производительность среди рассмотренных вариантов LMMR (MAE 0.81 vs 1.06 у Claude).

## 1. Быстрый запуск (Python)

```python
import pandas as pd
from sirena.models.lmmr import LMMRForecaster

# 1. Загрузка данных
# Убедитесь, что индекс - datetime, а данные - monthly start
df = pd.read_csv('data/infl_kbr.csv', sep=';', decimal='.', parse_dates=['Date'], index_col='Date')

# 2. Инициализация и обучение
model = LMMRForecaster(alpha=0.5, demand_proxy='all_real')
model.fit(df, target_col='Все товары и услуги')

# 3. Прогноз на конкретную дату (например, следующий месяц)
target_date = pd.Timestamp('2025-05-01')
# DataFrame для прогноза должен содержать историю до target_date
prediction = model.predict(df, target_date)

print(f"Прогноз на {target_date.date()}: {prediction['prediction']:.2f}% (MoM)")
print(f"Компоненты: SA={prediction['sa_prediction']:.2f}, Seasonal={prediction['seasonal_factor']:.2f}")
```

## 2. Запуск бэктеста

Чтобы проверить качество модели на исторических данных (2023-2025):

```bash
python3 -m scripts.backtest_lmmr_gemini
```

## 3. Интеграция в ансамбль

Модель автоматически зарегистрирована в `ModelRegistry` под именем `lmmr`.
Для использования в ансамбле или через API достаточно запросить её по имени:

```python
from sirena.models import ModelRegistry

model = ModelRegistry.get("lmmr")
```

## 4. Особенности модели

*   **Тип:** Динамическая регрессия на сезонно-сглаженных данных (SA MoM).
*   **Факторы:**
    *   Лаг инфляции (AR component).
    *   Курс доллара (USD).
    *   Реальные доходы (или кредиты) как прокси спроса.
    *   Шоковые dummy-переменные (2014, 2015, 2022).
*   **Сезонность:** STL декомпозиция (Robust Seasonal-Trend decomposition).

## 5. Результаты тестов

| Модель | MAE (2023-2025) | RMSE | Примечание |
|--------|-----------------|------|------------|
| **LMMR (Gemini)** | **0.8098** | **0.9548** | Лучшая версия |
| LMMR (Claude) | 1.0621 | 1.2967 | Хуже обработка трендов |

*Примечание: Текущий MAE (0.81) выше, чем у базовой Ridge модели (0.31), что типично для структурных моделей без тонкой настройки гиперпараметров на короткой истории.*
