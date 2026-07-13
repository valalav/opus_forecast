# План реализации модели ЛММР (Claude) — v2.0

**Автор:** Claude Opus 4.5
**Дата:** 2025-12-18
**Версия:** 2.0 (гибридный план с учётом идей Gemini и Qwen)

---

## 1. Анализ исходной модели ЛММР (R-код ЦБ)

### 1.1 Что делает оригинальная модель

Модель ЛММР из `metod/Пример кода лмнр.R`:

1. **Преобразование данных:**
   - MoM индексы → Базисные индексы
   - Сезонное сглаживание X13 (JDemetra+) → SA ряды
   - Извлечение сезонной компоненты (SC)

2. **Динамическая регрессия (dynlm):**
   ```r
   ipc ~ L(ipc, 1) + L(usd, 1) + prom_price_food + L(gruz_price, 1) +
         m201412_15 + m201707 + m202203 + m202204
   ```

3. **Shock dummies (5 штук):**
   - `m201412` — декабрь 2014 (валютный кризис)
   - `m201412_15` — декабрь 2014 + январь 2015
   - `m201501` — январь 2015
   - `m201707` — июль 2017 (индексация тарифов ЖКХ)
   - `m202203`, `m202204` — санкционный шок 2022

4. **Веса компонентов (федеральные):**
   - Продовольствие: 43.123%
   - Непродовольствие: 37.063%
   - Услуги: 19.814%

---

## 2. Гибридный подход (Claude + Gemini + Qwen)

### 2.1 Что берём из каждого плана

| Источник | Идея | Обоснование |
|----------|------|-------------|
| **Claude** | Все 5 shock dummies из R-кода | Точное воспроизведение методики ЦБ |
| **Claude** | Детальный код с конкретными функциями | Готовность к реализации |
| **Gemini** | Готовые SA данные из `sa_fl.csv` | Не изобретать велосипед, данные уже есть |
| **Gemini** | Использование `USDForecaster` | Прогноз USD уже реализован в v4.7 |
| **Qwen** | Возможность MixedLM для компонентов | Учёт иерархии Food/NonFood/Services |
| **Qwen** | Доверительные интервалы | Оценка неопределённости прогноза |

### 2.2 Архитектура модели

```
LMMRForecaster
├── Данные SA (из sa_fl.csv — идея Gemini)
│   ├── Загрузка через SADataLoader
│   └── Сезонная компонента = Факт / SA
├── Динамическая регрессия
│   ├── Базовая: Ridge (как в текущих моделях)
│   └── Альтернатива: MixedLM (идея Qwen)
├── Признаки
│   ├── Лаг SA: y_sa_lag1
│   ├── Экзогенные: usd_lag1 (через USDForecaster), brent_lag1
│   └── Все 5 shock dummies (Claude)
└── Прогноз
    ├── SA прогноз → + сезонность → MoM
    └── Доверительные интервалы (Qwen)
```

---

## 3. Детальный план реализации

### Этап 1: Структура класса

**Файл:** `sirena/models/lmmr.py`

```python
from typing import Optional, Dict, Tuple
import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import RobustScaler

from .base import BaseForecaster
from .registry import ModelRegistry
from ..sa_data_loader import SADataLoader


@ModelRegistry.register("lmmr")
class LMMRForecaster(BaseForecaster):
    """
    ЛММР - Локальная Мультипликативная Модель Регрессии.

    Гибридная реализация методики ЦБ РФ:
    - SA данные из sa_fl.csv (идея Gemini)
    - Все 5 shock dummies из R-кода (Claude)
    - Опциональный MixedLM для компонентов (идея Qwen)
    """

    name = "lmmr"
    MIN_TRAIN_SIZE = 36

    # НЕ исключаем 2022 — используем shock dummies
    OUTLIER_YEARS = [2010]

    # Веса компонентов КБР (из CLAUDE.md)
    COMPONENT_WEIGHTS = {
        'food': 0.3948,
        'nonfood': 0.3653,
        'services': 0.2342
    }

    # Признаки модели
    BASE_FEATURES = ['y_sa_lag1', 'y_sa_lag2']
    EXOG_FEATURES = ['usd_lag1', 'brent_lag1']
    SHOCK_DUMMIES = [
        'is_shock_dec2014',
        'is_shock_jan2015',
        'is_shock_dec2014_jan2015',
        'is_tariff_jul',
        'is_shock_mar2022',
        'is_shock_apr2022'
    ]

    def __init__(self, alpha: float = 0.5, use_mixed_model: bool = False):
        """
        Args:
            alpha: Коэффициент регуляризации Ridge
            use_mixed_model: Использовать MixedLM вместо Ridge (идея Qwen)
        """
        super().__init__()
        self.alpha = alpha
        self.use_mixed_model = use_mixed_model
        self.model = None
        self.scaler = None
        self.sa_loader = SADataLoader()
        self.seasonal_factors = None  # Месячные сезонные коэффициенты
```

### Этап 2: Загрузка SA данных (идея Gemini)

```python
def _load_sa_data(self, df: pd.DataFrame) -> pd.DataFrame:
    """
    Загрузка сезонно-скорректированных данных из sa_fl.csv.

    Вместо вычисления STL используем готовые SA данные (идея Gemini).
    """
    # Загрузка SA данных через существующий loader
    sa_df = self.sa_loader.load_sa_data()

    # Выбираем только "Все товары и услуги"
    sa_total = sa_df[sa_df['Товар'] == 'Все товары и услуги'].copy()
    sa_total = sa_total.set_index('Дата')['Значение']

    # Вычисляем сезонную компоненту: SC = Факт / SA
    merged = df[['Все товары и услуги']].join(sa_total.rename('SA'), how='inner')
    merged['seasonal_factor'] = merged['Все товары и услуги'] / merged['SA']

    # Средние сезонные коэффициенты по месяцам
    merged['month'] = merged.index.month
    self.seasonal_factors = merged.groupby('month')['seasonal_factor'].mean().to_dict()

    return sa_total
```

### Этап 3: Shock dummies (все 5 из R-кода)

```python
def _add_shock_dummies(self, df: pd.DataFrame) -> pd.DataFrame:
    """
    Добавить все shock dummies из оригинального R-кода.

    Это ключевое отличие от упрощённых планов Gemini/Qwen.
    """
    result = df.copy()

    # 1. Декабрь 2014 — валютный кризис (резкая девальвация)
    result['is_shock_dec2014'] = (
        (df.index.year == 2014) & (df.index.month == 12)
    ).astype(int)

    # 2. Январь 2015 — продолжение валютного шока
    result['is_shock_jan2015'] = (
        (df.index.year == 2015) & (df.index.month == 1)
    ).astype(int)

    # 3. Комбинированный dummy (dec2014 + jan2015) — как m201412_15 в R
    result['is_shock_dec2014_jan2015'] = (
        result['is_shock_dec2014'] | result['is_shock_jan2015']
    ).astype(int)

    # 4. Июль — ежегодная индексация тарифов ЖКХ (m201707 в R)
    result['is_tariff_jul'] = (df.index.month == 7).astype(int)

    # 5. Март 2022 — санкционный шок
    result['is_shock_mar2022'] = (
        (df.index.year == 2022) & (df.index.month == 3)
    ).astype(int)

    # 6. Апрель 2022 — продолжение санкционного шока
    result['is_shock_apr2022'] = (
        (df.index.year == 2022) & (df.index.month == 4)
    ).astype(int)

    return result
```

### Этап 4: Подготовка признаков

```python
def _prepare_features(self, df: pd.DataFrame, sa_series: pd.Series) -> pd.DataFrame:
    """Подготовка всех признаков для регрессии."""

    result = df.copy()

    # 1. Лаги SA компоненты
    result['y_sa_lag1'] = sa_series.shift(1)
    result['y_sa_lag2'] = sa_series.shift(2)

    # 2. Экзогенные переменные
    # USD — используем USDForecaster если нужен прогноз (идея Gemini)
    if 'usd_nom_i' in df.columns:
        result['usd_lag1'] = df['usd_nom_i'].shift(1)
    elif 'USD' in df.columns:
        result['usd_lag1'] = df['USD'].shift(1)

    # Brent как proxy для цен производителей и грузоперевозок
    if 'brent' in df.columns:
        result['brent_lag1'] = df['brent'].shift(1)

    # 3. Shock dummies
    result = self._add_shock_dummies(result)

    # 4. Заполняем NaN медианой для стабильности
    for col in self.EXOG_FEATURES:
        if col in result.columns:
            result[col] = result[col].fillna(result[col].median())

    return result
```

### Этап 5: Обучение модели

```python
def fit(self, df: pd.DataFrame, target_col: str = 'Все товары и услуги'):
    """
    Обучение ЛММР модели.

    1. Загрузка SA данных из sa_fl.csv (идея Gemini)
    2. Вычисление сезонных коэффициентов
    3. Подготовка признаков с shock dummies
    4. Обучение Ridge (или MixedLM — идея Qwen)
    """
    series = self._validate_data(df, target_col)

    # 1. Загрузка SA данных
    self.sa_series = self._load_sa_data(df)

    # 2. Подготовка признаков
    df_prep = self._prepare_features(df, self.sa_series)

    # 3. Формирование списка признаков
    self._features = self.BASE_FEATURES.copy()
    for f in self.EXOG_FEATURES:
        if f in df_prep.columns and df_prep[f].notna().sum() > 10:
            self._features.append(f)
    self._features.extend(self.SHOCK_DUMMIES)

    # 4. Исключение выбросных лет (только 2010, НЕ 2022!)
    df_prep['year'] = df_prep.index.year
    train_df = df_prep[~df_prep['year'].isin(self.OUTLIER_YEARS)]
    train_clean = train_df.dropna(subset=self._features)

    if len(train_clean) < self.MIN_TRAIN_SIZE:
        raise ValueError(f"Недостаточно данных: {len(train_clean)} < {self.MIN_TRAIN_SIZE}")

    # 5. Подготовка X и y
    X = train_clean[self._features].values
    y = self.sa_series.loc[train_clean.index].values

    # 6. Масштабирование
    self.scaler = RobustScaler()
    X_scaled = self.scaler.fit_transform(X)

    # 7. Обучение модели
    if self.use_mixed_model:
        # Альтернатива: MixedLM (идея Qwen)
        self._fit_mixed_model(train_clean, y)
    else:
        # Базовый вариант: Ridge
        self.model = Ridge(alpha=self.alpha)
        self.model.fit(X_scaled, y)

    self._is_fitted = True
    self._last_train_date = df.index.max()

    return self


def _fit_mixed_model(self, train_df: pd.DataFrame, y: np.ndarray):
    """
    Альтернативное обучение через MixedLM (идея Qwen).

    Позволяет учитывать случайные эффекты компонентов.
    """
    try:
        import statsmodels.formula.api as smf

        # Формула с фиксированными эффектами
        formula = "y_sa ~ " + " + ".join(self._features)

        train_df = train_df.copy()
        train_df['y_sa'] = y

        # MixedLM с случайным эффектом по месяцам
        self.model = smf.mixedlm(
            formula,
            train_df,
            groups=train_df.index.month
        ).fit()

    except ImportError:
        # Fallback к Ridge
        X_scaled = self.scaler.fit_transform(train_df[self._features].values)
        self.model = Ridge(alpha=self.alpha)
        self.model.fit(X_scaled, y)
```

### Этап 6: Прогнозирование

```python
def predict(self, df: pd.DataFrame, target_date: pd.Timestamp) -> dict:
    """
    Прогноз на конкретную дату.

    1. Прогноз SA компоненты
    2. + Сезонный коэффициент
    3. → MoM индекс
    4. + Доверительный интервал (идея Qwen)
    """
    self._check_fitted()

    # 1. Подготовка признаков
    df_prep = self._prepare_features(df, self.sa_series)

    # 2. Прогноз SA
    test_row = df_prep.loc[[target_date], self._features]

    if self.use_mixed_model and hasattr(self.model, 'predict'):
        sa_pred = self.model.predict(test_row).values[0]
        # CI из MixedLM
        ci_lower, ci_upper = None, None  # TODO: извлечь из модели
    else:
        X_scaled = self.scaler.transform(test_row.values)
        sa_pred = self.model.predict(X_scaled)[0]
        ci_lower, ci_upper = None, None

    # 3. Добавляем сезонность
    month = target_date.month
    seasonal = self.seasonal_factors.get(month, 1.0)
    mom_pred = sa_pred * seasonal

    # 4. Результат
    result = {
        'date': target_date,
        'prediction': mom_pred,
        'model': self.name,
        'sa_prediction': sa_pred,
        'seasonal_factor': seasonal
    }

    if ci_lower is not None:
        result['ci_lower'] = ci_lower * seasonal
        result['ci_upper'] = ci_upper * seasonal

    return result


def forecast(self, horizon: int = 12) -> np.ndarray:
    """Прогноз на горизонт (рекурсивный)."""
    self._check_fitted()

    predictions = []
    current_date = self._last_train_date

    for h in range(horizon):
        target_date = current_date + pd.DateOffset(months=h+1)
        # Рекурсивный прогноз: используем предыдущие прогнозы как лаги
        pred = self.predict(self._get_extended_df(predictions), target_date)
        predictions.append(pred['prediction'])

    return np.array(predictions)
```

### Этап 7: Бэктестирование

```python
def backtest(self, df: pd.DataFrame, start_date: str = '2019-01-01',
             target_col: str = 'Все товары и услуги') -> pd.DataFrame:
    """Бэктестирование с expanding window."""

    start = pd.Timestamp(start_date)
    valid_dates = df.dropna(subset=[target_col]).index
    test_dates = valid_dates[valid_dates >= start]

    results = []

    for target_date in test_dates:
        train_df = df[df.index < target_date].copy()

        if len(train_df.dropna(subset=[target_col])) < self.MIN_TRAIN_SIZE:
            continue

        try:
            # Новая модель для каждой точки (чистый бэктест)
            model = LMMRForecaster(alpha=self.alpha, use_mixed_model=self.use_mixed_model)
            model.fit(train_df, target_col)

            test_df = df[df.index <= target_date].copy()
            pred = model.predict(test_df, target_date)

            actual = df.loc[target_date, target_col]

            results.append({
                'date': target_date,
                'actual': actual,
                'prediction': pred['prediction'],
                'error': actual - pred['prediction'],
                'sa_prediction': pred.get('sa_prediction'),
                'seasonal_factor': pred.get('seasonal_factor')
            })
        except Exception as e:
            continue

    return pd.DataFrame(results)
```

### Этап 8: Важность признаков

```python
def get_feature_importance(self) -> pd.DataFrame:
    """Важность признаков (коэффициенты модели)."""
    self._check_fitted()

    if self.use_mixed_model:
        coefs = self.model.fe_params.values
    else:
        coefs = self.model.coef_

    importance = pd.DataFrame({
        'feature': self._features,
        'coefficient': coefs,
        'abs_importance': np.abs(coefs)
    }).sort_values('abs_importance', ascending=False)

    return importance
```

---

## 4. Интеграция в Dashboard

### 4.1 Tab1 (Прогноз) — после строки ~854

```python
# ЛММР модель (гибридная)
lmmr_df = None
try:
    from sirena.models.lmmr import LMMRForecaster
    model = LMMRForecaster()
    model.fit(df)
    vals = []
    for h in range(horizon):
        target_date = last_date + pd.DateOffset(months=h+1)
        df_ext = df.copy()
        df_ext.loc[target_date] = np.nan
        pred = model.predict(df_ext, target_date)['prediction'] - 100
        vals.append(pred)
    lmmr_df = pd.DataFrame({
        'Date': pd.date_range(start=last_date + pd.DateOffset(months=1),
                              periods=horizon, freq='MS'),
        'LMMR': vals
    })
except Exception as e:
    st.warning(f"ЛММР: {e}")
```

### 4.2 model_weights (~строка 953)

```python
model_weights = {
    'LMMR': (lmmr_df['LMMR'].values if lmmr_df is not None else None, 0.10),
    # ... остальные модели
}
```

### 4.3 Добавить trace на график (~строка 981)

```python
if lmmr_df is not None:
    fig_fc.add_trace(go.Scatter(
        x=lmmr_df['Date'], y=lmmr_df['LMMR'],
        name='ЛММР', line=dict(color='#9C27B0', width=2, dash='dot')
    ))
```

### 4.4 Tab3 (Бэктест)

Аналогично другим моделям — добавить импорт, прогноз в цикле, в results.append(), в all_models и model_colors.

---

## 5. Тестирование

### 5.1 Unit-тесты

**Файл:** `tests/test_lmmr.py`

```python
import pytest
import pandas as pd
import numpy as np
from sirena.models.lmmr import LMMRForecaster


class TestLMMRForecaster:

    @pytest.fixture
    def sample_data(self):
        """Тестовые данные."""
        # Загрузка реальных данных
        df = pd.read_csv('data/infl_kbr.csv', sep=';', decimal=',')
        # ... подготовка ...
        return df

    def test_fit(self, sample_data):
        """Тест обучения."""
        model = LMMRForecaster()
        model.fit(sample_data)
        assert model._is_fitted
        assert model.seasonal_factors is not None
        assert len(model.seasonal_factors) == 12

    def test_shock_dummies(self, sample_data):
        """Тест создания shock dummies."""
        model = LMMRForecaster()
        df_with_dummies = model._add_shock_dummies(sample_data)

        # Проверяем наличие всех 5 dummies
        assert 'is_shock_dec2014' in df_with_dummies.columns
        assert 'is_shock_jan2015' in df_with_dummies.columns
        assert 'is_tariff_jul' in df_with_dummies.columns
        assert 'is_shock_mar2022' in df_with_dummies.columns
        assert 'is_shock_apr2022' in df_with_dummies.columns

        # Проверяем значения
        dec2014 = df_with_dummies.loc['2014-12-01', 'is_shock_dec2014']
        assert dec2014 == 1

    def test_predict(self, sample_data):
        """Тест прогнозирования."""
        model = LMMRForecaster()
        model.fit(sample_data)

        target_date = sample_data.index[-1] + pd.DateOffset(months=1)
        pred = model.predict(sample_data, target_date)

        assert 'prediction' in pred
        assert 'sa_prediction' in pred
        assert 'seasonal_factor' in pred
        assert 95 < pred['prediction'] < 105

    def test_backtest_mae(self, sample_data):
        """Тест MAE на бэктесте."""
        model = LMMRForecaster()
        results = model.backtest(sample_data, start_date='2023-01-01')

        mae = results['error'].abs().mean()
        assert mae < 0.40  # Должен быть лучше 0.40
        print(f"ЛММР MAE: {mae:.3f}")

    def test_feature_importance(self, sample_data):
        """Тест важности признаков."""
        model = LMMRForecaster()
        model.fit(sample_data)

        importance = model.get_feature_importance()

        # Shock dummies должны быть значимыми
        shock_features = importance[importance['feature'].str.contains('shock')]
        assert len(shock_features) > 0
```

---

## 6. Сравнение подходов

| Аспект | R-оригинал | Claude v1 | Claude v2 (гибрид) |
|--------|------------|-----------|-------------------|
| SA данные | X13 (JDemetra) | STL | **sa_fl.csv** (готовые) |
| Shock dummies | 5 | 5 | **5** |
| Регрессия | dynlm | Ridge | **Ridge / MixedLM** |
| USD прогноз | Внешний | Нет | **USDForecaster** |
| CI | Нет | Нет | **Да (MixedLM)** |
| Зависимости | R + Java | statsmodels | **минимальные** |

---

## 7. Файлы для создания/изменения

| Файл | Действие | Описание |
|------|----------|----------|
| `sirena/models/lmmr.py` | **Создать** | Основной код модели |
| `sirena/models/__init__.py` | Изменить | Добавить `from .lmmr import LMMRForecaster` |
| `tests/test_lmmr.py` | **Создать** | Unit-тесты |
| `dashboard.py` | Изменить | Интеграция в tab1, tab3 |
| `CLAUDE.md` | Изменить | Документация модели |

---

## 8. Зависимости

Все уже установлены:
```
statsmodels>=0.14.0  # MixedLM (опционально)
scikit-learn>=1.3.0  # Ridge, RobustScaler
pandas>=2.0.0
numpy>=1.24.0
```

---

## 9. Порядок реализации

1. [ ] Создать `sirena/models/lmmr.py` с базовой структурой
2. [ ] Реализовать `_load_sa_data()` из `sa_fl.csv`
3. [ ] Реализовать `_add_shock_dummies()` — все 5 из R-кода
4. [ ] Реализовать `_prepare_features()`
5. [ ] Реализовать `fit()` с Ridge
6. [ ] Реализовать `predict()` с сезонностью
7. [ ] Реализовать `backtest()`
8. [ ] Добавить опцию `use_mixed_model` (MixedLM)
9. [ ] Написать unit-тесты
10. [ ] Провести бэктест и сравнить с Ridge (MAE ≤ 0.35)
11. [ ] Интегрировать в Dashboard
12. [ ] Обновить документацию

---

## 10. Критерии успеха

- [ ] MAE на бэктесте 2023-2025 ≤ 0.35
- [ ] Все 5 shock dummies реализованы и интерпретируемы
- [ ] SA данные загружаются из `sa_fl.csv`
- [ ] Все unit-тесты проходят
- [ ] Модель интегрирована в Dashboard
- [ ] Коэффициенты shock dummies показывают ожидаемый вклад:
  - `is_shock_dec2014`: +1.5–2.0 п.п.
  - `is_shock_jan2015`: +0.8–1.2 п.п.
  - `is_tariff_jul`: +0.1–0.3 п.п.

---

## 11. Преимущества гибридного подхода

1. **Простота** — используем готовые SA данные (не изобретаем X13)
2. **Точность** — все 5 shock dummies из методики ЦБ
3. **Гибкость** — опция MixedLM для продвинутого анализа
4. **Интеграция** — использование существующих компонентов (SADataLoader, USDForecaster)
5. **Интерпретируемость** — коэффициенты shock dummies показывают вклад шоков
