"""
LSTM модель для прогнозирования инфляции
=========================================

Deep Learning подход с рекуррентной нейронной сетью.

Особенности:
- Sequence-to-one архитектура
- Консервативные параметры (малые данные ~190 точек)
- ETS fallback для высокой сезонности
- Early stopping для предотвращения переобучения
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Tuple
import warnings

from .base import BaseForecaster
from .registry import ModelRegistry

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class LSTMNetwork(nn.Module):
    """LSTM сеть для временных рядов."""

    def __init__(self, input_size: int, hidden_size: int, num_layers: int,
                 dropout: float, output_size: int = 1):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )

        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, output_size)
        )

    def forward(self, x):
        # x: (batch, seq_len, input_size)
        lstm_out, _ = self.lstm(x)
        # Берём только последний выход
        last_out = lstm_out[:, -1, :]  # (batch, hidden_size)
        out = self.fc(last_out)
        return out


@ModelRegistry.register("lstm")
class LSTMForecaster(BaseForecaster):
    """
    LSTM модель прогнозирования инфляции.

    Параметры:
    - hidden_size: размер скрытого состояния (default: 32)
    - num_layers: количество LSTM слоёв (default: 2)
    - dropout: вероятность dropout (default: 0.2)
    - sequence_length: длина входной последовательности (default: 12)
    - epochs: максимальное количество эпох (default: 100)
    - patience: терпение для early stopping (default: 10)
    """

    name = "lstm"
    MIN_TRAIN_SIZE = 36

    OUTLIER_YEARS = [2010, 2022]

    # ETS веса для комбинации
    ETS_WEIGHTS = {
        1: 0.9, 2: 0.0, 3: 0.5, 4: 0.3,
        5: 0.9, 6: 0.5, 7: 0.0, 8: 0.5,
        9: 0.9, 10: 0.9, 11: 0.0, 12: 0.0
    }

    # Признаки для LSTM
    FEATURE_COLS = [
        'y_scaled',
        'month_sin', 'month_cos',
        'food_scaled', 'nonfood_scaled', 'services_scaled',
        'seasonal_norm_scaled', 'deviation_scaled'
    ]

    def __init__(self,
                 hidden_size: int = 32,
                 num_layers: int = 2,
                 dropout: float = 0.2,
                 sequence_length: int = 12,
                 epochs: int = 100,
                 patience: int = 10,
                 learning_rate: float = 0.001,
                 **kwargs):
        super().__init__(**kwargs)

        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch не установлен. pip install torch")

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.sequence_length = sequence_length
        self.epochs = epochs
        self.patience = patience
        self.learning_rate = learning_rate

        self.model = None
        self.scaler_mean = None
        self.scaler_std = None
        self.seasonal_norm = None
        self.device = torch.device('cpu')

    def _prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Подготовка признаков."""
        df = df.copy()

        df['month'] = df.index.month
        df['year'] = df.index.year

        y = df['Все товары и услуги']

        # Сезонность
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

        # Компоненты
        df['food'] = df.get('Продовольственные товары', y)
        df['nonfood'] = df.get('Непродовольственные товары', y)
        df['services'] = df.get('Услуги', y)

        return df

    def _compute_seasonal_norm(self, df: pd.DataFrame) -> pd.Series:
        """Сезонная норма."""
        clean_df = df[~df['year'].isin(self.OUTLIER_YEARS)]
        return clean_df.groupby('month')['Все товары и услуги'].mean()

    def _create_sequences(self, data: np.ndarray, target: np.ndarray
                          ) -> Tuple[np.ndarray, np.ndarray]:
        """Создание последовательностей для LSTM."""
        X, y = [], []
        for i in range(self.sequence_length, len(data)):
            X.append(data[i - self.sequence_length:i])
            y.append(target[i])
        return np.array(X), np.array(y)

    def _scale_data(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """Масштабирование данных."""
        df = df.copy()

        cols_to_scale = ['Все товары и услуги', 'food', 'nonfood', 'services',
                         'seasonal_norm', 'deviation']

        if fit:
            self.scaler_mean = {}
            self.scaler_std = {}
            for col in cols_to_scale:
                if col in df.columns:
                    self.scaler_mean[col] = df[col].mean()
                    self.scaler_std[col] = df[col].std()
                    if self.scaler_std[col] == 0:
                        self.scaler_std[col] = 1

        # Масштабируем
        df['y_scaled'] = (df['Все товары и услуги'] - self.scaler_mean.get('Все товары и услуги', 100)) / self.scaler_std.get('Все товары и услуги', 1)
        df['food_scaled'] = (df['food'] - self.scaler_mean.get('food', 100)) / self.scaler_std.get('food', 1)
        df['nonfood_scaled'] = (df['nonfood'] - self.scaler_mean.get('nonfood', 100)) / self.scaler_std.get('nonfood', 1)
        df['services_scaled'] = (df['services'] - self.scaler_mean.get('services', 100)) / self.scaler_std.get('services', 1)
        df['seasonal_norm_scaled'] = (df['seasonal_norm'] - self.scaler_mean.get('seasonal_norm', 100)) / self.scaler_std.get('seasonal_norm', 1)
        df['deviation_scaled'] = (df['deviation'] - self.scaler_mean.get('deviation', 0)) / self.scaler_std.get('deviation', 1)

        return df

    def fit(self, df: pd.DataFrame, target_col: str = 'Все товары и услуги') -> 'LSTMForecaster':
        """Обучение LSTM."""
        self._validate_data(df, target_col)

        df_prep = self._prepare_features(df)
        self.seasonal_norm = self._compute_seasonal_norm(df_prep)

        df_prep['seasonal_norm'] = df_prep['month'].map(self.seasonal_norm)
        df_prep['deviation'] = df_prep['Все товары и услуги'] - df_prep['seasonal_norm']

        # Исключаем выбросные годы
        train_df = df_prep[~df_prep['year'].isin(self.OUTLIER_YEARS)].copy()
        train_df = train_df.dropna()

        if len(train_df) < self.MIN_TRAIN_SIZE + self.sequence_length:
            raise ValueError(f"Недостаточно данных: {len(train_df)}")

        # Масштабирование
        train_df = self._scale_data(train_df, fit=True)

        # Подготовка данных для LSTM
        feature_data = train_df[self.FEATURE_COLS].values
        target_data = train_df['y_scaled'].values

        X, y = self._create_sequences(feature_data, target_data)

        # Train/Val split (80/20)
        split_idx = int(len(X) * 0.8)
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]

        # Конвертируем в тензоры
        X_train_t = torch.FloatTensor(X_train).to(self.device)
        y_train_t = torch.FloatTensor(y_train).unsqueeze(1).to(self.device)
        X_val_t = torch.FloatTensor(X_val).to(self.device)
        y_val_t = torch.FloatTensor(y_val).unsqueeze(1).to(self.device)

        # Создаём модель
        input_size = len(self.FEATURE_COLS)
        self.model = LSTMNetwork(
            input_size=input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout
        ).to(self.device)

        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

        # Обучение с early stopping
        best_val_loss = float('inf')
        patience_counter = 0
        best_state = None

        self.model.train()
        for epoch in range(self.epochs):
            optimizer.zero_grad()
            outputs = self.model(X_train_t)
            loss = criterion(outputs, y_train_t)
            loss.backward()
            optimizer.step()

            # Валидация
            self.model.eval()
            with torch.no_grad():
                val_outputs = self.model(X_val_t)
                val_loss = criterion(val_outputs, y_val_t).item()
            self.model.train()

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_state = self.model.state_dict().copy()
            else:
                patience_counter += 1

            if patience_counter >= self.patience:
                break

        # Загружаем лучшую модель
        if best_state is not None:
            self.model.load_state_dict(best_state)

        self.model.eval()
        self._is_fitted = True
        self._last_train_date = df.index.max()

        return self

    def predict(self, df: pd.DataFrame, target_date: pd.Timestamp) -> Dict[str, Any]:
        """Прогноз на дату."""
        self._check_fitted()

        df_prep = self._prepare_features(df)
        df_prep['seasonal_norm'] = df_prep['month'].map(self.seasonal_norm)
        df_prep['deviation'] = df_prep['Все товары и услуги'] - df_prep['seasonal_norm']

        # Масштабирование (без fit)
        df_prep = self._scale_data(df_prep, fit=False)

        # Получаем последовательность до target_date
        end_idx = df_prep.index.get_loc(target_date)
        start_idx = end_idx - self.sequence_length

        if start_idx < 0:
            raise ValueError("Недостаточно данных для последовательности")

        sequence = df_prep.iloc[start_idx:end_idx][self.FEATURE_COLS].values

        # Прогноз
        X = torch.FloatTensor(sequence).unsqueeze(0).to(self.device)

        with torch.no_grad():
            pred_scaled = self.model(X).item()

        # Обратное масштабирование
        pred_lstm = pred_scaled * self.scaler_std.get('Все товары и услуги', 1) + self.scaler_mean.get('Все товары и услуги', 100)

        # ETS компонента
        target_month = target_date.month
        pred_ets = self.seasonal_norm.get(target_month, 100.0)

        # Комбинация
        ets_weight = self.ETS_WEIGHTS.get(target_month, 0.3)
        pred_combined = (1 - ets_weight) * pred_lstm + ets_weight * pred_ets

        return {
            'date': target_date,
            'prediction': pred_combined,
            'pred_lstm': pred_lstm,
            'pred_ets': pred_ets,
            'ets_weight': ets_weight,
            'model': self.name
        }

    def forecast(self, horizon: int = 12) -> np.ndarray:
        """Прогноз на горизонт (ETS fallback)."""
        self._check_fitted()

        last_month = self._last_train_date.month if self._last_train_date else 1
        predictions = []

        for i in range(horizon):
            month = ((last_month + i) % 12) + 1
            pred = self.seasonal_norm.get(month, 100.0)
            predictions.append(pred)

        return np.array(predictions)

    def backtest(
        self,
        df: pd.DataFrame,
        start_date: str = '2023-01-01',
        target_col: str = 'Все товары и услуги'
    ) -> pd.DataFrame:
        """Бэктестирование LSTM."""
        start = pd.Timestamp(start_date)

        valid_dates = df.dropna(subset=[target_col]).index
        test_dates = valid_dates[valid_dates >= start]

        results = []

        for target_date in test_dates:
            train_df = df[df.index < target_date].copy()

            if len(train_df.dropna(subset=[target_col])) < self.MIN_TRAIN_SIZE + self.sequence_length:
                continue

            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    model = LSTMForecaster(
                        hidden_size=self.hidden_size,
                        num_layers=self.num_layers,
                        dropout=self.dropout,
                        sequence_length=self.sequence_length,
                        epochs=self.epochs,
                        patience=self.patience
                    )
                    model.fit(train_df, target_col)

                    test_df = df[df.index <= target_date].copy()
                    pred = model.predict(test_df, target_date)

                    actual = df.loc[target_date, target_col]

                    results.append({
                        'date': target_date,
                        'actual': actual,
                        'prediction': pred['prediction'],
                        'error': actual - pred['prediction'],
                        'pred_lstm': pred['pred_lstm'],
                        'pred_ets': pred['pred_ets']
                    })
            except Exception as e:
                continue

        return pd.DataFrame(results)

    def get_model_info(self) -> Dict:
        """Информация о модели."""
        return {
            'name': self.name,
            'hidden_size': self.hidden_size,
            'num_layers': self.num_layers,
            'dropout': self.dropout,
            'sequence_length': self.sequence_length,
            'epochs': self.epochs,
            'patience': self.patience,
            'is_fitted': self._is_fitted
        }
