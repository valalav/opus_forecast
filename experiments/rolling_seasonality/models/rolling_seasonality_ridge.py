"""
Rolling Seasonality Ridge — модель с адаптивной скользящей сезонностью
====================================================================

Проблема: базовая Ridge модель использует глобальную сезонную норму,
вычисленную на всей истории. После структурных сдвигов 2022-2024 
это приводит к плохим прогнозам.

Решение: сезонность вычисляется только на последних N месяцах (rolling window),
что позволяет адаптироваться к "новой реальности".

Параметры:
- seasonality_window: количество месяцев для расчёта сезонности (24, 36, 48)
- Использует все остальные признаки из Ridge (лаги, макро и т.д.)

Автор: Claude Code
Дата: 2026-02-02
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import RobustScaler
from typing import Dict, Optional, Any, List
import sys
import os

# Add project root to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from sirena.models.base import BaseForecaster
from sirena.models.registry import ModelRegistry


@ModelRegistry.register("rolling_seasonality_ridge")
class RollingSeasonalityRidge(BaseForecaster):
    """
    Ridge регрессия со скользящей сезонностью.
    
    Ключевое отличие от базовой Ridge:
    - Сезонная норма вычисляется на rolling window (последние N месяцев)
    - Вместо глобальной сезонности на всей истории
    
    Это позволяет модели адаптироваться к структурным сдвигам
    и не полагаться на устаревшие паттерны.
    """
    
    name = "rolling_seasonality_ridge"
    MIN_TRAIN_SIZE = 36
    
    # Годы-выбросы (исключаем при обучении Ridge, но НЕ при расчёте сезонности)
    # Потому что для rolling seasonality мы хотим учитывать даже шоковые периоды,
    # если они попадают в окно
    OUTLIER_YEARS = [2010]  # 2022 исключили — он важен для новой сезонности!
    
    # ETS веса по месяцам (как в базовой Ridge)
    ETS_WEIGHTS = {
        1: 0.9, 2: 0.0, 3: 0.5, 4: 0.3,
        5: 0.9, 6: 0.5, 7: 0.0, 8: 0.5,
        9: 0.9, 10: 0.9, 11: 0.0, 12: 0.0
    }
    
    # Ridge регуляризация
    ALPHA = 0.3
    
    # Базовые признаки (как в Ridge)
    BASE_FEATURES = [
        'y_lag1', 'y_lag2', 'y_lag12', 'y_ma3',
        'month_sin', 'month_cos',
        'food_lag1', 'nonfood_lag1', 'services_lag1',
        'seasonal_norm', 'deviation_lag1'
    ]
    
    # Макро-признаки
    MACRO_FEATURES = [
        'ruonia_diff_lag1',
        'spread_lag4',
        'ki_diff_lag6',
        'ki_vol',
    ]
    
    def __init__(
        self, 
        alpha: float = None, 
        use_macro: bool = True,
        ets_weights: Dict[int, float] = None,
        seasonality_window: int = 36,  # Ключевой параметр!
        **kwargs
    ):
        """
        Инициализация модели.
        
        Args:
            alpha: Ridge регуляризация (по умолчанию 0.3)
            use_macro: Использовать макро-признаки Ki/Ruonia
            ets_weights: Словарь весов сезонности {month: weight}
            seasonality_window: Количество месяцев для rolling сезонности (24, 36, 48)
        """
        super().__init__(**kwargs)
        self.alpha = alpha or self.ALPHA
        self.use_macro = use_macro
        self.ets_weights = ets_weights if ets_weights is not None else self.ETS_WEIGHTS
        self.seasonality_window = seasonality_window
        self.ridge = None
        self.scaler = None
        self.seasonal_norm = None
        self._has_macro = False
        self._features = None
        
    @property
    def FEATURES(self) -> List[str]:
        """Динамический список признаков."""
        if self._features is not None:
            return self._features
        return self.BASE_FEATURES
    
    def _prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Подготовка признаков (как в базовой Ridge)."""
        df = df.copy()
        
        df['month'] = df.index.month
        df['year'] = df.index.year
        
        # Лаги целевой переменной
        df['y_lag1'] = df['Все товары и услуги'].shift(1)
        df['y_lag2'] = df['Все товары и услуги'].shift(2)
        df['y_lag12'] = df['Все товары и услуги'].shift(12)
        
        # Скользящее среднее
        df['y_ma3'] = df['Все товары и услуги'].rolling(3).mean().shift(1)
        
        # Сезонные признаки
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        
        # Лаги компонентов
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
            
        return df
    
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
        
        # Волатильность Ki за 6 месяцев
        df['ki_vol'] = df['Ki'].rolling(6).std().shift(1)
        
        # Заполняем NaN медианой
        for col in self.MACRO_FEATURES:
            if col in df.columns:
                df[col] = df[col].fillna(df[col].median())
                
        return df
    
    def _compute_seasonal_norm(self, df: pd.DataFrame) -> pd.Series:
        """
        Вычисление скользящей сезонной нормы.
        
        КЛЮЧЕВОЕ ОТЛИЧИЕ от базовой Ridge:
        - Берём только последние seasonality_window месяцев
        - Не исключаем 2022 год (он важен для новой сезонности!)
        - Исключаем только 2010 (слишком далеко)
        """
        # Берём последние N месяцев
        cutoff_date = df.index.max() - pd.DateOffset(months=self.seasonality_window)
        recent_df = df[df.index >= cutoff_date]
        
        # Исключаем только явно выбросные годы (2010)
        clean_df = recent_df[~recent_df['year'].isin(self.OUTLIER_YEARS)]
        
        if len(clean_df) < 12:
            # Если слишком мало данных — используем всё
            clean_df = recent_df
            
        # Считаем среднее по месяцам
        seasonal_norm = clean_df.groupby('month')['Все товары и услуги'].mean()
        
        return seasonal_norm
    
    def fit(self, df: pd.DataFrame, target_col: str = 'Все товары и услуги') -> 'RollingSeasonalityRidge':
        """
        Обучение модели.
        
        Args:
            df: DataFrame с данными
            target_col: Целевая колонка
            
        Returns:
            self
        """
        # Валидация
        series = self._validate_data(df, target_col)
        
        # Подготовка признаков
        df_prep = self._prepare_features(df)
        
        # СКОЛЬЗЯЩАЯ сезонная норма (ключевое отличие!)
        self.seasonal_norm = self._compute_seasonal_norm(df_prep)
        
        # Добавляем сезонные признаки
        df_prep['seasonal_norm'] = df_prep['month'].map(self.seasonal_norm)
        df_prep['deviation_lag1'] = df_prep['y_lag1'] - df_prep['month'].shift(1).map(self.seasonal_norm)
        
        # Определяем список признаков
        self._features = self.BASE_FEATURES.copy()
        
        # Добавляем макро-признаки если включены
        self._has_macro = False
        if self.use_macro and 'Ki' in df.columns and 'Ruonia' in df.columns:
            df_prep = self._add_macro_features(df_prep)
            available_macro = [f for f in self.MACRO_FEATURES if f in df_prep.columns]
            if available_macro:
                self._features.extend(available_macro)
                self._has_macro = True
        
        # Исключаем выбросные годы для обучения Ridge (но не для сезонности!)
        train_df = df_prep[~df_prep['year'].isin(self.OUTLIER_YEARS)]
        
        # Очистка
        train_clean = train_df.dropna(subset=self._features + [target_col])
        
        if len(train_clean) < self.MIN_TRAIN_SIZE:
            raise ValueError(f"Недостаточно данных: {len(train_clean)} < {self.MIN_TRAIN_SIZE}")
        
        # Обучение
        X = train_clean[self._features].values
        y = train_clean[target_col].values
        
        self.scaler = RobustScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        self.ridge = Ridge(alpha=self.alpha)
        self.ridge.fit(X_scaled, y)
        
        self._is_fitted = True
        self._last_train_date = df.index.max()
        self._target_col = target_col
        
        # Сохраняем DataFrame для iterative_forecast
        self._train_df = df.copy()
        
        return self
    
    def forecast(self, horizon: int = 12) -> np.ndarray:
        """
        Прогноз на горизонт через итеративный predict().
        """
        self._check_fitted()
        
        # Используем iterative_forecast с сохранёнными данными
        if hasattr(self, '_train_df') and self._train_df is not None:
            target_col = getattr(self, '_target_col', 'Все товары и услуги')
            return self.iterative_forecast(self._train_df, horizon, target_col)
        
        # Fallback на сезонную норму
        if self.seasonal_norm is None:
            return np.zeros(horizon)
            
        last_month = self._last_train_date.month if self._last_train_date else 1
        predictions = []
        
        for i in range(horizon):
            month = ((last_month + i) % 12) + 1
            pred = self.seasonal_norm.get(month, 100.0) - 100  # MoM%
            predictions.append(pred)
            
        return np.array(predictions)
    
    def predict(self, df: pd.DataFrame, target_date: pd.Timestamp) -> Dict[str, Any]:
        """
        Точечный прогноз на дату.
        """
        self._check_fitted()
        
        df_prep = self._prepare_features(df)
        df_prep['seasonal_norm'] = df_prep['month'].map(self.seasonal_norm)
        df_prep['deviation_lag1'] = df_prep['y_lag1'] - df_prep['month'].shift(1).map(self.seasonal_norm)
        
        # Добавляем макро-признаки если использовались при обучении
        if self._has_macro:
            df_prep = self._add_macro_features(df_prep)
            
        test_row = df_prep.loc[[target_date]]
        
        # Ridge прогноз
        X_test = self.scaler.transform(test_row[self._features].values)
        pred_ridge = self.ridge.predict(X_test)[0]
        
        # ETS прогноз
        target_month = target_date.month
        pred_ets = self.seasonal_norm.get(target_month, 100.0)
        
        # Комбинация
        ets_weight = self.ets_weights.get(target_month, 0.3)
        pred_combined = (1 - ets_weight) * pred_ridge + ets_weight * pred_ets
        
        return {
            'date': target_date,
            'prediction': pred_combined,
            'pred_ridge': pred_ridge,
            'pred_ets': pred_ets,
            'ets_weight': ets_weight,
            'model': self.name,
            'has_macro': self._has_macro,
            'seasonality_window': self.seasonality_window
        }
    
    def backtest(
        self,
        df: pd.DataFrame,
        start_date: str = '2019-01-01',
        target_col: str = 'Все товары и услуги'
    ) -> pd.DataFrame:
        """
        Бэктестирование модели.
        """
        start = pd.Timestamp(start_date)
        
        valid_dates = df.dropna(subset=[target_col]).index
        test_dates = valid_dates[valid_dates >= start]
        
        results = []
        
        for target_date in test_dates:
            # Cutoff — все данные до текущего месяца
            cutoff = target_date - pd.DateOffset(days=1)
            train_df = df[df.index < target_date].copy()
            
            if len(train_df.dropna(subset=[target_col])) < self.MIN_TRAIN_SIZE:
                continue
                
            try:
                # Создаём новую модель для чистого бэктеста
                model = RollingSeasonalityRidge(
                    alpha=self.alpha,
                    use_macro=self.use_macro,
                    ets_weights=self.ets_weights,
                    seasonality_window=self.seasonality_window
                )
                model.fit(train_df, target_col)
                
                test_df = df[df.index <= target_date].copy()
                pred_result = model.predict(test_df, target_date)
                
                actual = df.loc[target_date, target_col]
                
                results.append({
                    'date': target_date,
                    'actual': actual,
                    'prediction': pred_result['prediction'],
                    'error': actual - pred_result['prediction'],
                    'pred_ridge': pred_result['pred_ridge'],
                    'pred_ets': pred_result['pred_ets'],
                    'has_macro': pred_result.get('has_macro', False),
                    'seasonality_window': self.seasonality_window
                })
            except Exception as e:
                print(f"RollingSeasonalityRidge Error at {target_date}: {e}")
                continue
                
        return pd.DataFrame(results)
    
    def get_feature_importance(self) -> pd.DataFrame:
        """Важность признаков."""
        self._check_fitted()
        
        importance = pd.DataFrame({
            'feature': self._features,
            'coefficient': self.ridge.coef_
        })
        importance['abs_coef'] = importance['coefficient'].abs()
        importance['is_macro'] = importance['feature'].isin(self.MACRO_FEATURES)
        return importance.sort_values('abs_coef', ascending=False)
    
    def get_metrics(self, results: pd.DataFrame) -> Dict[str, float]:
        """Расчёт метрик качества."""
        if results.empty:
            return {'MAE': 0, 'RMSE': 0, 'KPI': 0, 'KPI_violations': 0}
        
        errors = results['error'].abs()
        mae = errors.mean()
        rmse = np.sqrt((results['error'] ** 2).mean())
        kpi_rate = (errors <= 0.5).sum() / len(results) * 100
        kpi_violations = (errors > 0.5).sum()
        
        return {
            'MAE': mae,
            'RMSE': rmse,
            'KPI': kpi_rate,
            'KPI_violations': kpi_violations
        }
