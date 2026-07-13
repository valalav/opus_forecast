# LMMR X13 - Модель прогнозирования инфляции

## Описание

**LMMR X13** (Локальная Модель Множественной Регрессии с X13-ARIMA) - модель для прогнозирования региональной инфляции, **точно следующая** методике ЦБ РФ (Отделение Волгоград) из файла `Пример кода лмнр.R`.

**Файл:** `sirena/models/lmmr_x13.py`

---

## Ключевые особенности

### 1. Сезонное сглаживание через X13-ARIMA
- Использует X13-ARIMA (как в R-коде через RJDemetra)
- Декомпозиция временного ряда на SA Base и Seasonal Component (SC)
- Fallback на STL (statsmodels) если X13 недоступен

### 2. Обработка данных (точно по R-коду)
```
MoM индексы → Base Index (f.calc_base)
                   ↓
             X13-ARIMA → SA Base + SC
                   ↓
             SA Base → SA MoM (dt_mom_SA)
                   ↓
             Ridge Regression на SA MoM
                   ↓
             SA MoM прогноз → SA Base прогноз
                   ↓
             Base = SA_Base + SC (АДДИТИВНАЯ модель!)
                   ↓
             MoM = Base / lag(Base) * 100
```

### 3. Признаки (точно по R-коду, строка 496)

**Формула из R-кода:**
```
ipc ~ L(ipc, 1) + L(usd, 1) + prom_price_food + L(gruz_price, 1) +
      m201412_15 + m201707 + m202203 + m202204
```

- **y_sa_lag1:** `L(ipc, 1)` — лаг SA MoM
- **usd_lag1:** `L(usd, 1)` — лаг курса доллара
- **prom_price_food:** цены производителей продовольствия (без лага!)
- **gruz_price_lag1:** `L(gruz_price, 1)` — лаг цен грузовых перевозок
- **Shock dummies (4 шт):**
  - `is_shock_dec2014_jan2015` (m201412_15) - дек 2014 + янв 2015 (комбинированный)
  - `is_tariff_jul` (m201707) - июль (индексация тарифов ЖКХ)
  - `is_shock_mar2022` (m202203) - март 2022
  - `is_shock_apr2022` (m202204) - апрель 2022

---

## Производительность

### Backtest (2024-2025)
- **LMMR X13 (STL):** MAE 0.447
- **Direct Ridge + Ki_i:** MAE 0.384 (−9.0% vs baseline)
- **N:** 22 наблюдения

### Важность экзогенных признаков (эмпирический анализ)
| Признак | Корреляция | CV MAE | Δ vs baseline |
|---------|------------|--------|---------------|
| **Ki_i_lag1** (ключевая ставка) | +0.479 | 0.338 | **−4.1%** |
| Ki_i (без лага) | +0.414 | 0.353 | −0.4% |
| torg (торговля) | +0.185 | 0.351 | −0.4% |
| prom_prod (промышленность) | +0.142 | 0.353 | 0.0% |

**Вывод:** Ключевая ставка ЦБ (Ki_i) с лагом 1 месяц — лучший экзогенный признак!

### Важность признаков
| Признак | Коэффициент | Важность |
|---------|-------------|----------|
| `is_shock_mar2022` | 3.05 | Высокая |
| `is_shock_jan2015` | 0.96 | Высокая |
| `is_shock_dec2014_jan2015` | 0.77 | Средняя |
| `y_sa_lag1` | 0.16 | Средняя |

---

## Использование

### Базовый пример

```python
from sirena.models import LMMRX13Forecaster
import pandas as pd

# Загрузка данных
df = pd.read_csv('data/infl_kbr.csv', sep=';', decimal=',')
df = df.pivot(index='Date', columns='Товар', values='MoM')
df.index = pd.to_datetime(df.index)

# Обучение модели
model = LMMRX13Forecaster(alpha=0.5, use_x13=True)
model.fit(df, target_col='Все товары и услуги')

# Прогноз
target_date = pd.Timestamp('2025-01-01')
prediction = model.predict(df, target_date)

print(f"Прогноз MoM: {prediction['prediction']:.3f}")
print(f"SA MoM прогноз: {prediction['sa_mom_prediction']:.3f}")
print(f"Seasonal Component (SC): {prediction['seasonal_component']:.3f}")
print(f"Base прогноз: {prediction['base_prediction']:.3f}")
```

### Backtest

```python
# Бэктест с расширяющимся окном
results = model.backtest(df, start_date='2024-01-01')

# Метрики
mae = results['abs_error'].mean()
rmse = (results['error']**2).mean()**0.5

print(f"MAE: {mae:.3f}")
print(f"RMSE: {rmse:.3f}")
```

### Важность признаков

```python
# Получить важность признаков
importance = model.get_feature_importance()
print(importance.head(10))
```

---

## Параметры

### Инициализация

```python
LMMRX13Forecaster(alpha=0.5, use_x13=True, minimal=False)
```

**Параметры:**
- `alpha` (float): Параметр регуляризации Ridge (по умолчанию 0.5)
- `use_x13` (bool): Использовать X13-ARIMA (True) или STL (False)
- `minimal` (bool): Использовать минимальный набор признаков (y_sa_lag1 + Ki_i_lag1)
  - При `minimal=True` модель использует только 2 признака: лаг SA MoM и ключевую ставку
  - Эмпирически доказано: минимальная конфигурация даёт MAE −9% vs baseline
  - Рекомендуется если данные по ключевой ставке доступны

### Методы

#### `fit(df, target_col)`
Обучение модели.

**Параметры:**
- `df` (DataFrame): Данные с индексом DatetimeIndex
- `target_col` (str): Название целевой колонки (по умолчанию 'Все товары и услуги')

**Возвращает:** `self`

#### `predict(df, target_date)`
Точечный прогноз на конкретную дату.

**Параметры:**
- `df` (DataFrame): Данные
- `target_date` (Timestamp): Дата прогноза

**Возвращает:** `dict` с ключами:
- `date`: Дата прогноза
- `prediction`: MoM прогноз
- `sa_mom_prediction`: SA MoM прогноз
- `sa_base_prediction`: SA Base прогноз
- `seasonal_component`: Сезонная компонента (SC)
- `base_prediction`: Base прогноз (SA_Base + SC)
- `model`: Название модели

#### `backtest(df, start_date, target_col)`
Бэктест с расширяющимся окном.

**Параметры:**
- `df` (DataFrame): Данные
- `start_date` (str): Начальная дата бэктеста
- `target_col` (str): Целевая колонка

**Возвращает:** `DataFrame` с результатами

#### `get_feature_importance()`
Важность признаков из коэффициентов Ridge.

**Возвращает:** `DataFrame` с колонками `feature`, `coefficient`, `abs_importance`

---

## Технические детали

### Алгоритм (точно по R-коду)

1. **Преобразование MoM → Base Index** (функция f.calc_base)
   ```python
   base[i] = base[i-1] * mom[i] / 100
   ```

2. **X13-ARIMA декомпозиция** (RJDemetra)
   - Спецификация `RSA5c` (автоматический подбор)
   - Outlier detection: `ao=TRUE`, `ls=TRUE`
   - Результат: SA Base + Seasonal Component (SC)

3. **SA Base → SA MoM**
   ```python
   sa_mom[i] = sa_base[i] / sa_base[i-1] * 100
   ```

4. **Ridge регрессия на SA MoM**
   - Обучение на SA MoM данных
   - RobustScaler для нормализации
   - Исключение outlier years (2010)

5. **Прогноз (АДДИТИВНАЯ модель!)** (функция f.prognoz.mom)
   ```python
   # Строки 92-99 из R-кода
   sa_base_f = cumprod(sa_mom_f / 100) * sa_base[last_date]
   base_f = sa_base_f + SC  # АДДИТИВНАЯ модель!
   mom_f = base_f / lag(base_f) * 100
   ```

### Зависимости

- `pandas` - работа с данными
- `numpy` - вычисления
- `sklearn` - Ridge, RobustScaler
- `statsmodels` - STL (fallback)
- X13-ARIMA бинарник в `bin/linux/` или `bin/windows/`

---

## Сравнение с LMMR Hybrid

| Аспект | LMMR X13 | LMMR Hybrid |
|--------|----------|-------------|
| **Сезонное сглаживание** | Динамическое (X13-ARIMA) | Из файла `sa_fl.csv` |
| **Автономность** | ✅ Полная | ❌ Требует SA файл |
| **Прозрачность** | ✅ Весь процесс виден | ⚠️ SA из черного ящика |
| **Методология ЦБ** | **~95%** | ~60% |
| **Сезонная модель** | ✅ Аддитивная (как в R) | ❌ Мультипликативная |
| **MAE (2024-2025)** | TBD | 0.310 |
| **Скорость** | Медленнее (X13) | Быстрее |

---

## Требования к данным

### Минимальные требования
- Минимум 48 наблюдений (4 года) для X13-ARIMA
- Месячная частота
- DatetimeIndex

### Необходимые колонки
- Целевая переменная (MoM индексы)
- Опционально (по R-коду):
  - `usd_nom_i` — курс доллара (L(usd, 1))
  - `prom_price_food` — цены производителей продовольствия
  - `gruz_price` — цены грузовых перевозок (L(gruz_price, 1))

---

## Установка X13-ARIMA

Бинарники уже включены в проект:
- Linux: `bin/linux/x13as_ascii`
- Windows: `bin/windows/x13as_ascii.exe`

Для Linux нужны права на выполнение:
```bash
chmod +x bin/linux/x13as_ascii
```

---

## Troubleshooting

### X13-ARIMA не работает
**Симптом:** Модель автоматически переключается на STL

**Решение:**
1. Проверить наличие бинарника: `ls -la bin/linux/x13as_ascii`
2. Проверить права: `chmod +x bin/linux/x13as_ascii`
3. Использовать STL явно: `LMMRX13Forecaster(use_x13=False)`

### Недостаточно данных
**Ошибка:** `Insufficient training data`

**Решение:** Убедиться, что есть минимум 48 наблюдений

### Плохое качество прогноза
**Решение:**
1. Проверить качество входных данных
2. Настроить `alpha` параметр Ridge
3. Добавить больше экзогенных переменных

---

## Лицензия

Использует X13-ARIMA-SEATS от U.S. Census Bureau (Public Domain)

---

## Авторы

Реализация: Antigravity AI (2025)
Методология: ЦБ РФ, Отделение Волгоград
