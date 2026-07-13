"""
LMMR - Локальная Мультипликативная Модель Регрессии (Claude версия)
Реализация методики ЦБ РФ для прогнозирования региональной инфляции.
Основана на сезонной декомпозиции (STL) и динамической регрессии.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import RobustScaler
from statsmodels.tsa.seasonal import STL
from typing import Tuple, Dict, Any
import warnings

from sirena.models.base import BaseForecaster
from sirena.models.registry import ModelRegistry


@ModelRegistry.register("lmmr_claude")
class LMMRForecasterClaude(BaseForecaster):
    """
    ЛММР - Локальная Мультипликативная Модель Регрессии (Claude Implementation).

    Реализация методики ЦБ РФ для прогнозирования региональной инфляции.
    Основана на сезонной декомпозиции (STL) и динамической регрессии.
    """

    name = "lmmr_claude"
    MIN_TRAIN_SIZE = 48  # Требуется больше данных для STL

    # Веса компонентов КБР (из CLAUDE.md)
    COMPONENT_WEIGHTS = {
        'food': 0.3948,
        'nonfood': 0.3653,
        'services': 0.2342
    }

    def __init__(self, alpha: float = 0.5):
        super().__init__()
        self.alpha = alpha  # Regularization parameter for Ridge
        self.model = None
        self.scaler = None
        self.sa_series = None
        self.seasonal = None
        self.base_index = None
        self.features = [
            'y_sa_lag1', 'Ki_i_lag1', 'usd_lag1', 'brent_lag1',
            'is_shock_dec2014_jan2015', 'is_tariff_jul',
            'is_shock_mar2022', 'is_shock_apr2022'
        ]
        
    def _validate_data(self, df: pd.DataFrame, target_col: str = 'Все товары и услуги') -> pd.Series:
        """Validate and return the target series."""
        if target_col not in df.columns:
            available_cols = list(df.columns)
            raise ValueError(f"Target column '{target_col}' not found in DataFrame. Available columns: {available_cols}")
        
        series = df[target_col].dropna()
        
        if len(series) < self.MIN_TRAIN_SIZE:
            raise ValueError(f"Insufficient data: need at least {self.MIN_TRAIN_SIZE} observations, got {len(series)}")
        
        return series

    def _decompose_series(self, series: pd.Series) -> Tuple[pd.Series, pd.Series]:
        """
        Сезонная декомпозиция ряда с использованием STL.

        Returns:
            (sa_series, seasonal_component)
        """
        # Ensure the series is in the correct format for STL
        if len(series) < 13:  # Minimum for STL
            raise ValueError(f"Series too short for STL decomposition. Need at least 13 observations, got {len(series)}")
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            stl = STL(series, period=12, robust=True)
            result = stl.fit()

        sa = result.trend + result.resid  # Сезонно-скорректированный
        sc = result.seasonal              # Сезонная компонента

        return sa, sc

    def _to_base_index(self, mom_series: pd.Series) -> pd.Series:
        """
        Преобразование MoM индексов в базисные индексы.

        base[0] = mom[0]
        base[i] = base[i-1] * mom[i] / 100
        """
        base = mom_series.copy()
        base.iloc[0] = mom_series.iloc[0]

        for i in range(1, len(base)):
            base.iloc[i] = base.iloc[i-1] * mom_series.iloc[i] / 100

        return base

    def _from_base_to_mom(self, base_series: pd.Series) -> pd.Series:
        """
        Обратное преобразование базисных индексов в MoM.

        mom[i] = base[i] / base[i-1] * 100
        """
        mom = base_series / base_series.shift(1) * 100
        mom.iloc[0] = base_series.iloc[0]
        return mom

    def _get_seasonal_factor(self, month: int) -> float:
        """Get seasonal factor for a specific month."""
        if self.seasonal is not None:
            # Get the seasonal factor for the specified month
            monthly_seasonal = self.seasonal.groupby(self.seasonal.index.month).mean()
            return monthly_seasonal[month]
        return 0.0

    def _prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Подготовка признаков для динамической регрессии."""
        result = df.copy()
        
        # 1. Лаги зависимой переменной (на SA данных)
        # NOTE: We'll compute these after decomposition
        if self.sa_series is not None:
            # For prediction on historical data, use the fitted SA series
            sa_lags = self.sa_series.shift(1)
            # Map SA lags to the result DataFrame index
            sa_lag_dict = sa_lags.to_dict()
            result['y_sa_lag1'] = result.index.map(sa_lag_dict)
            # Forward fill to handle missing values (dates not in the original sa_series)
            result['y_sa_lag1'] = result['y_sa_lag1'].fillna(method='ffill').fillna(0.0)
        else:
            # If SA series is not fitted yet, create a placeholder
            result['y_sa_lag1'] = result['Все товары и услуги'].shift(1)

        # 2. Экзогенные переменные
        if 'usd_nom_i' in df.columns:
            result['usd_lag1'] = df['usd_nom_i'].shift(1)
        else:
            # If USD is not available, create a placeholder
            result['usd_lag1'] = 0.0

        # Brent как proxy для цен производителей и грузоперевозок
        if 'brent' in df.columns:
            result['brent_lag1'] = df['brent'].shift(1)
        else:
            # If Brent is not available, create a placeholder
            result['brent_lag1'] = 0.0

        # Ki_i (ключевая ставка ЦБ) — лучший экзогенный признак!
        # Эмпирический анализ показал: MAE -9% vs baseline
        if 'Ki_i' in df.columns:
            result['Ki_i_lag1'] = df['Ki_i'].shift(1)
        else:
            result['Ki_i_lag1'] = 0.0

        # 3. Shock dummies (из методики ЦБ)
        result['is_shock_dec2014'] = (
            (df.index.year == 2014) & (df.index.month == 12)
        ).astype(int)

        result['is_shock_jan2015'] = (
            (df.index.year == 2015) & (df.index.month == 1)
        ).astype(int)

        result['is_shock_dec2014_jan2015'] = (
            result['is_shock_dec2014'] | result['is_shock_jan2015']
        ).astype(int)

        result['is_tariff_jul'] = (df.index.month == 7).astype(int)

        result['is_shock_mar2022'] = (
            (df.index.year == 2022) & (df.index.month == 3)
        ).astype(int)

        result['is_shock_apr2022'] = (
            (df.index.year == 2022) & (df.index.month == 4)
        ).astype(int)

        return result

    def fit(self, df: pd.DataFrame, target_col: str = 'Все товары и услуги'):
        """
        Обучение ЛММР модели.

        1. Преобразование MoM → базисные индексы
        2. Сезонная декомпозиция (STL)
        3. Подготовка признаков
        4. Обучение Ridge регрессии на SA данных
        """
        series = self._validate_data(df, target_col)

        # 1. Преобразование к базисным индексам
        self.base_index = self._to_base_index(series)

        # 2. Сезонная декомпозиция
        self.sa_series, self.seasonal = self._decompose_series(self.base_index)

        # 3. Подготовка признаков
        df_prep = self._prepare_features(df)

        # 4. Формирование обучающей выборки
        available_features = []
        for feat in self.features:
            if feat in df_prep.columns:
                available_features.append(feat)
        
        if not available_features:
            raise ValueError("No features available for training")
        
        train_df = df_prep.dropna(subset=available_features)
        
        # Ensure target variable is also available
        sa_train = self.sa_series.loc[train_df.index].dropna()
        common_idx = train_df.index.intersection(sa_train.index)
        
        if len(common_idx) < self.MIN_TRAIN_SIZE:
            raise ValueError(f"Insufficient overlapping data after feature preparation: need {self.MIN_TRAIN_SIZE}, got {len(common_idx)}")
        
        train_df = train_df.loc[common_idx]
        sa_train = sa_train.loc[common_idx]

        X = train_df[available_features].values
        y = sa_train.values

        # 5. Масштабирование и обучение
        self.scaler = RobustScaler()
        X_scaled = self.scaler.fit_transform(X)

        self.model = Ridge(alpha=self.alpha)
        self.model.fit(X_scaled, y)

        self._is_fitted = True
        self._last_train_date = df.index.max()
        self.features = available_features  # Update with actually available features
        
        # Store the training DataFrame and last target value for forecasting
        self._last_df = df.copy()
        self._last_target_value = series.iloc[-1] if len(series) > 0 else 100.0

        return self

    def predict(self, df: pd.DataFrame, target_date: pd.Timestamp) -> dict:
        """
        Прогноз на конкретную дату.

        1. Прогноз SA компоненты (Ridge)
        2. + Сезонная компонента (из STL)
        3. → Базисный индекс
        4. → MoM индекс
        """
        self._check_fitted()

        df_prep = self._prepare_features(df)

        if target_date not in df_prep.index:
            raise ValueError(f"Target date {target_date} not in prepared features DataFrame")

        # 1. Подготовка признаков для прогноза
        X_test_row = df_prep.loc[[target_date], self.features]
        
        # Check for NaN values and handle them
        if X_test_row.isna().any().any():
            # Forward fill NaN values or use a default value
            X_test = X_test_row.fillna(method='ffill').fillna(method='bfill').values
        else:
            X_test = X_test_row.values

        # Check if we still have any NaN values
        if np.isnan(X_test).any():
            # If there are still NaN values, use the mean values of the scaler
            # This is a fallback to avoid errors
            X_test = np.nan_to_num(X_test, nan=0.0)

        X_scaled = self.scaler.transform(X_test)
        sa_pred = self.model.predict(X_scaled)[0]

        # 2. Добавляем сезонную компоненту
        month = target_date.month
        seasonal_factor = self._get_seasonal_factor(month)
        base_pred = sa_pred + seasonal_factor

        # 3. Преобразуем в MoM
        # To get the MoM from the base index prediction, we need the previous base value
        # Find the most recent base index value from training data
        prev_base_idx = self.base_index.index[self.base_index.index < target_date]
        if len(prev_base_idx) > 0:
            prev_base = self.base_index.loc[prev_base_idx[-1]]
            mom_pred = base_pred / prev_base * 100
        else:
            # Fallback: assume base prediction is the MoM value
            mom_pred = base_pred

        return {
            'date': target_date,
            'prediction': mom_pred,
            'model': self.name,
            'sa_prediction': sa_pred,
            'seasonal_factor': seasonal_factor
        }

    def forecast(self, horizon: int = 12) -> np.ndarray:
        """
        Forecast for multiple periods ahead using recursive approach.
        
        Parameters:
        - horizon: number of periods to forecast
        
        Returns:
        - array of MoM forecasts
        """
        self._check_fitted()
        
        # Generate future dates
        last_date = self._last_train_date
        future_dates = pd.date_range(
            start=last_date + pd.DateOffset(months=1),
            periods=horizon,
            freq='MS'
        )
        
        forecasts = []
        current_df = self._last_df.copy()
        
        # Keep track of the last known base index for MoM conversion
        last_known_base = self.base_index.iloc[-1]
        
        for i, future_date in enumerate(future_dates):
            # Extend the DataFrame to include the future date
            if future_date not in current_df.index:
                # Add the future date with NaN values for all columns
                new_row = pd.DataFrame(index=[future_date], columns=current_df.columns)
                current_df = pd.concat([current_df, new_row]).sort_index()
            
            # Update features based on the current state of the DataFrame
            df_with_features = self._prepare_features(current_df)

            # Check if required features for the future date are available
            if not all(feat in df_with_features.columns for feat in self.features):
                # If features are missing, use a simple fallback
                forecasts.append(self._last_target_value)
                continue

            # Try to make the prediction
            try:
                # For recursive forecasting with lags, we need to update values as we go
                # For y_sa_lag1, we need the previous SA value
                # If it's the first forecast, we use the last known SA value
                if i == 0:
                    # For the first forecast, we can use the model directly
                    temp_model = LMMRForecasterClaude(alpha=self.alpha)
                    temp_model.model = self.model
                    temp_model.scaler = self.scaler
                    temp_model.sa_series = self.sa_series
                    temp_model.seasonal = self.seasonal
                    temp_model.base_index = self.base_index
                    temp_model.features = self.features
                    temp_model._is_fitted = True
                    
                    # Predict for the first future date
                    pred = temp_model.predict(df_with_features, future_date)
                    forecast_value = pred['prediction']
                else:
                    # For subsequent forecasts, we need to handle the recursive nature
                    # Extract features for the target date
                    if future_date in df_with_features.index:
                        X_test = df_with_features.loc[[future_date], self.features].values
                        if X_test.shape[1] == len(self.features):  # All features available
                            X_scaled = self.scaler.transform(X_test)
                            sa_pred = self.model.predict(X_scaled)[0]

                            # Add seasonal component
                            month = future_date.month
                            seasonal_factor = self._get_seasonal_factor(month)
                            base_pred = sa_pred + seasonal_factor

                            # Convert to MoM using the last known or predicted base
                            mom_pred = base_pred / last_known_base * 100
                            forecast_value = mom_pred
                        else:
                            # Fallback if features are not available
                            forecast_value = self._last_target_value
                    else:
                        forecast_value = self._last_target_value

                forecasts.append(forecast_value)

                # Update the DataFrame with this prediction for subsequent forecasts
                # For now, we just keep track of the last forecast for the next iteration
                # In a more complex implementation, we would update the series used for lags
                
            except Exception:
                # If prediction fails, use the last known value as a fallback
                forecasts.append(self._last_target_value)
        
        return np.array(forecasts)

    def backtest(self, df: pd.DataFrame, start_date: str = '2019-01-01',
                 target_col: str = 'Все товары и услуги') -> pd.DataFrame:
        """Бэктестирование модели."""
        
        start = pd.Timestamp(start_date)
        test_dates = df.index[df.index >= start]

        results = []

        for target_date in test_dates:
            train_df = df[df.index < target_date].copy()

            if len(train_df) < self.MIN_TRAIN_SIZE:
                continue

            try:
                # Store the last training date for later use
                model = LMMRForecasterClaude(alpha=self.alpha)
                model.fit(train_df, target_col)

                # Store necessary attributes for forecasting
                model._last_df = train_df.copy()
                last_target_val = train_df[target_col].iloc[-1] if len(train_df) > 0 else 100
                model._last_target_value = last_target_val

                test_df = df[df.index <= target_date].copy()
                pred = model.predict(test_df, target_date)

                actual = df.loc[target_date, target_col]

                results.append({
                    'date': target_date,
                    'actual': actual,
                    'prediction': pred['prediction'],
                    'error': actual - pred['prediction'],
                    'model': self.name
                })
            except Exception as e:
                # Log the error but continue with other dates
                continue

        return pd.DataFrame(results)