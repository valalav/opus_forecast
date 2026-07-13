"""
LightGBM модель для прогнозирования инфляции КБР
=================================================

Gradient Boosting для нелинейных зависимостей.
v4.0: добавлены макро-признаки Ki и Ruonia.

Вес в ансамбле: 15%
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
from typing import Dict, Any, List, Optional
import warnings

warnings.filterwarnings('ignore')

from .base import BaseForecaster
from .registry import ModelRegistry

# Проверка LightGBM
try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False


@ModelRegistry.register("lightgbm")
class LightGBMForecaster(BaseForecaster):
    """
    LightGBM модель прогнозирования.

    Использует базовые признаки + макро-признаки Ki/Ruonia (v4.0).
    """

    name = "lightgbm"
    MIN_TRAIN_SIZE = 36

    BASE_FEATURES = [
        'y_lag1', 'y_lag2', 'y_lag12', 'y_ma3',
        'month_sin', 'month_cos',
        'food_lag1', 'nonfood_lag1', 'services_lag1',
        'seasonal_norm', 'deviation_lag1'
    ]

    # Макро-признаки (v4.0)
    MACRO_FEATURES = [
        'ruonia_diff_lag1',  # r=0.477
        'spread_lag4',       # r=0.444
        'ki_diff_lag6',      # r=0.300
        'ki_vol',            # r=0.256
    ]

    OUTLIER_YEARS = [2010, 2022]

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 5,
        learning_rate: float = 0.1,
        num_leaves: int = 31,
        min_child_samples: int = 10,
        outlier_years: List[int] = None,
        use_macro: bool = False,
        **kwargs
    ):
        """
        Инициализация LightGBM.

        Args:
            n_estimators: Количество деревьев
            max_depth: Максимальная глубина
            learning_rate: Скорость обучения
            num_leaves: Количество листьев
            min_child_samples: Минимум сэмплов в листе
            outlier_years: Годы-выбросы
            use_macro: Использовать макро-признаки Ki/Ruonia (по умолчанию False,
                       т.к. ухудшает KPI на 2.4 п.п. при незначительном улучшении MAE)
        """
        super().__init__(**kwargs)

        if not LIGHTGBM_AVAILABLE:
            raise ImportError("LightGBM not installed. Run: pip install lightgbm")

        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.num_leaves = num_leaves
        self.min_child_samples = min_child_samples
        self.outlier_years = outlier_years or self.OUTLIER_YEARS
        self.use_macro = use_macro

        self.model = None
        self.scaler = RobustScaler()
        self.seasonal_norm = None
        self.last_values = None
        self._has_macro = False
        self._features = None
        self._last_macro_values = None  # Для прогноза

    @property
    def FEATURES(self) -> List[str]:
        """Динамический список признаков."""
        if self._features is not None:
            return self._features
        return self.BASE_FEATURES

    def _add_macro_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Добавить макро-признаки Ki и Ruonia."""
        df = df.copy()

        if 'Ki' not in df.columns or 'Ruonia' not in df.columns:
            return df

        # ΔRuonia с лагом 1
        df['ruonia_diff'] = df['Ruonia'].diff()
        df['ruonia_diff_lag1'] = df['ruonia_diff'].shift(1)

        # Спред Ki - Ruonia с лагом 4
        df['spread'] = df['Ki'] - df['Ruonia']
        df['spread_lag4'] = df['spread'].shift(4)

        # ΔKi с лагом 6
        df['ki_diff'] = df['Ki'].diff()
        df['ki_diff_lag6'] = df['ki_diff'].shift(6)

        # Волатильность Ki
        df['ki_vol'] = df['Ki'].rolling(6).std().shift(1)

        # Заполняем NaN
        for col in self.MACRO_FEATURES:
            if col in df.columns:
                df[col] = df[col].fillna(df[col].median())

        return df

    def _prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Подготовка признаков."""
        df = df.copy()
        df['month'] = df.index.month
        df['year'] = df.index.year

        # Лаги
        df['y_lag1'] = df['Все товары и услуги'].shift(1)
        df['y_lag2'] = df['Все товары и услуги'].shift(2)
        df['y_lag12'] = df['Все товары и услуги'].shift(12)
        df['y_ma3'] = df['Все товары и услуги'].rolling(3).mean().shift(1)

        # Сезонность
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

        # Компоненты
        if 'Продовольственные товары' in df.columns:
            df['food_lag1'] = df['Продовольственные товары'].shift(1)
        else:
            df['food_lag1'] = df['y_lag1']

        if 'Непродовольственные товары' in df.columns:
            df['nonfood_lag1'] = df['Непродовольственные товары'].shift(1)
        else:
            df['nonfood_lag1'] = df['y_lag1']

        if 'Услуги' in df.columns:
            df['services_lag1'] = df['Услуги'].shift(1)
        else:
            df['services_lag1'] = df['y_lag1']

        # Сезонная норма
        clean = df[~df['year'].isin(self.outlier_years)]
        if len(clean) < 12:
            clean = df
        self.seasonal_norm = clean.groupby('month')['Все товары и услуги'].mean().to_dict()

        for m in range(1, 13):
            if m not in self.seasonal_norm:
                self.seasonal_norm[m] = 100.5

        df['seasonal_norm'] = df['month'].map(self.seasonal_norm)
        df['deviation_lag1'] = df['y_lag1'] - df['month'].shift(1).map(self.seasonal_norm)

        return df

    def fit(self, df: pd.DataFrame, target_col: str = 'Все товары и услуги') -> 'LightGBMForecaster':
        """Обучение LightGBM."""
        self._validate_data(df, target_col)

        df_prep = self._prepare_features(df)

        # Определяем список признаков
        self._features = self.BASE_FEATURES.copy()

        # Добавляем макро-признаки
        self._has_macro = False
        if self.use_macro and 'Ki' in df.columns and 'Ruonia' in df.columns:
            df_prep = self._add_macro_features(df_prep)
            available_macro = [f for f in self.MACRO_FEATURES if f in df_prep.columns]
            if available_macro:
                self._features.extend(available_macro)
                self._has_macro = True
                # Сохраняем последние значения макро для прогноза
                self._last_macro_values = {
                    col: df_prep[col].iloc[-1] for col in available_macro
                }

        train = df_prep.dropna(subset=self._features + [target_col])
        train = train[~train['year'].isin(self.outlier_years)]

        if len(train) < self.MIN_TRAIN_SIZE:
            raise ValueError(f"Недостаточно данных: {len(train)} < {self.MIN_TRAIN_SIZE}")

        X = train[self._features].values
        y = train[target_col].values

        X_scaled = self.scaler.fit_transform(X)

        # Сохраняем последние значения
        self.last_values = {
            'y': list(df[target_col].dropna().values),
            'food': df.get('Продовольственные товары', df[target_col]).iloc[-1],
            'nonfood': df.get('Непродовольственные товары', df[target_col]).iloc[-1],
            'services': df.get('Услуги', df[target_col]).iloc[-1]
        }

        # Обучение
        self.model = lgb.LGBMRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            num_leaves=self.num_leaves,
            min_child_samples=self.min_child_samples,
            verbosity=-1,
            random_state=42
        )
        self.model.fit(X_scaled, y)

        self._is_fitted = True
        self._last_train_date = df.index.max()
        self._target_col = target_col
        self._train_df = df.copy()

        return self

    def forecast(self, horizon: int = 12) -> np.ndarray:
        """Рекурсивный прогноз LightGBM."""
        self._check_fitted()

        history = self.last_values['y'].copy()
        forecasts = []

        for i in range(horizon):
            last_month = self._last_train_date.month if self._last_train_date else 1
            t_m = ((last_month + i) % 12) + 1

            y_lag1 = history[-1]
            y_lag2 = history[-2] if len(history) > 1 else history[-1]
            y_lag12 = history[-12] if len(history) > 11 else 100.5
            y_ma3 = np.mean(history[-3:]) if len(history) > 2 else history[-1]

            seasonal = self.seasonal_norm.get(t_m, 100.5)
            prev_month = (t_m - 1) if t_m > 1 else 12
            prev_seasonal = self.seasonal_norm.get(prev_month, 100.5)
            deviation = y_lag1 - prev_seasonal

            X_feat = np.array([[
                y_lag1, y_lag2, y_lag12, y_ma3,
                np.sin(2 * np.pi * t_m / 12),
                np.cos(2 * np.pi * t_m / 12),
                self.last_values['food'],
                self.last_values['nonfood'],
                self.last_values['services'],
                seasonal,
                deviation
            ]])

            X_scaled = self.scaler.transform(X_feat)
            pred = self.model.predict(X_scaled)[0]

            forecasts.append(pred - 100)  # Конвертируем в %
            history.append(pred)

        return np.array(forecasts)

    def predict(self, df: pd.DataFrame, target_date: pd.Timestamp) -> Dict[str, Any]:
        """Точечный прогноз."""
        self._check_fitted()

        df_prep = self._prepare_features(df)

        # Добавляем макро-признаки если использовались при обучении
        if self._has_macro:
            df_prep = self._add_macro_features(df_prep)

        test_row = df_prep.loc[[target_date]]

        X = self.scaler.transform(test_row[self._features].values)
        prediction = self.model.predict(X)[0]

        return {
            'date': target_date,
            'prediction': prediction,
            'model': self.name,
            'has_macro': self._has_macro
        }

    def backtest(
        self,
        df: pd.DataFrame,
        start_date: str = '2019-01-01',
        target_col: str = 'Все товары и услуги'
    ) -> pd.DataFrame:
        """Бэктестирование LightGBM."""
        start = pd.Timestamp(start_date)

        # Используем реальные даты из данных
        valid_dates = df.dropna(subset=[target_col]).index
        test_dates = valid_dates[valid_dates >= start]

        results = []

        for target_date in test_dates:
            actual = df.loc[target_date, target_col]
            if pd.isna(actual):
                continue

            train_df = df[df.index < target_date].copy()

            if len(train_df.dropna(subset=[target_col])) < self.MIN_TRAIN_SIZE:
                continue

            try:
                model = LightGBMForecaster(
                    n_estimators=self.n_estimators,
                    max_depth=self.max_depth,
                    outlier_years=self.outlier_years,
                    use_macro=self.use_macro
                )
                model.fit(train_df, target_col)

                test_df = df[df.index <= target_date].copy()
                pred = model.predict(test_df, target_date)

                results.append({
                    'date': target_date,
                    'actual': actual,
                    'prediction': pred['prediction'],
                    'error': actual - pred['prediction'],
                    'has_macro': pred.get('has_macro', False)
                })
            except Exception:
                continue

        return pd.DataFrame(results)

    def get_feature_importance(self) -> pd.DataFrame:
        """Важность признаков."""
        self._check_fitted()

        importance = pd.DataFrame({
            'feature': self._features,
            'importance': self.model.feature_importances_
        })
        importance['is_macro'] = importance['feature'].isin(self.MACRO_FEATURES)
        return importance.sort_values('importance', ascending=False)


# Алиас для обратной совместимости
SirenaLightGBM = LightGBMForecaster
