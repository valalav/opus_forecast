"""
LSTM модель для прогнозирования инфляции КБР
============================================

Рекуррентная нейронная сеть с LSTM слоями для захвата
долгосрочных зависимостей во временных рядах.

Автор: СИРЕНА-КБР
Версия: 1.0
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import warnings

warnings.filterwarnings('ignore')

# Проверяем доступность TensorFlow
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.callbacks import EarlyStopping
    TENSORFLOW_AVAILABLE = True

    # Отключаем лишние логи TensorFlow
    tf.get_logger().setLevel('ERROR')
except ImportError:
    TENSORFLOW_AVAILABLE = False
    print("Warning: TensorFlow not installed. Run: pip install tensorflow")


class SirenaLSTM:
    """
    LSTM модель для прогнозирования временных рядов.

    Архитектура:
    - LSTM(64) → Dropout(0.2) → LSTM(32) → Dense(1)

    Параметры:
        sequence_length: Длина входной последовательности (по умолчанию 12 месяцев)
        lstm_units: Количество нейронов в первом LSTM слое
        epochs: Количество эпох обучения
        batch_size: Размер батча
    """

    def __init__(
        self,
        sequence_length: int = 12,
        lstm_units: int = 64,
        lstm_units_2: int = 32,
        dropout: float = 0.2,
        epochs: int = 100,
        batch_size: int = 16,
        patience: int = 10
    ):
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow not installed. Run: pip install tensorflow")

        self.sequence_length = sequence_length
        self.lstm_units = lstm_units
        self.lstm_units_2 = lstm_units_2
        self.dropout = dropout
        self.epochs = epochs
        self.batch_size = batch_size
        self.patience = patience

        self.model = None
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.last_sequence = None
        self.history = None

    def _create_sequences(self, data: np.ndarray) -> tuple:
        """Создание последовательностей для LSTM."""
        X, y = [], []

        for i in range(self.sequence_length, len(data)):
            X.append(data[i - self.sequence_length:i, 0])
            y.append(data[i, 0])

        X = np.array(X)
        y = np.array(y)

        # Reshape для LSTM [samples, timesteps, features]
        X = X.reshape(X.shape[0], X.shape[1], 1)

        return X, y

    def _build_model(self):
        """Построение архитектуры модели."""
        model = Sequential([
            LSTM(
                self.lstm_units,
                return_sequences=True,
                input_shape=(self.sequence_length, 1)
            ),
            Dropout(self.dropout),
            LSTM(self.lstm_units_2, return_sequences=False),
            Dropout(self.dropout),
            Dense(1)
        ])

        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        return model

    def fit(
        self,
        df: pd.DataFrame,
        target_col: str = 'Все товары и услуги',
        verbose: int = 0
    ):
        """
        Обучение модели.

        Args:
            df: DataFrame с данными
            target_col: Целевая колонка
            verbose: Уровень вывода (0 = тихо)

        Returns:
            self
        """
        # Извлекаем данные
        if target_col in df.columns:
            series = df[target_col].dropna().values
        else:
            series = df.dropna().values

        # Конвертируем если нужно
        if np.mean(series) > 50:
            series = series - 100

        # Масштабирование
        series = series.reshape(-1, 1)
        scaled_data = self.scaler.fit_transform(series)

        # Создаём последовательности
        X, y = self._create_sequences(scaled_data)

        if len(X) < 10:
            raise ValueError(f"Недостаточно данных: {len(X)} последовательностей")

        # Сохраняем последнюю последовательность для прогноза
        self.last_sequence = scaled_data[-self.sequence_length:]

        # Строим и обучаем модель
        self.model = self._build_model()

        early_stop = EarlyStopping(
            monitor='loss',
            patience=self.patience,
            restore_best_weights=True
        )

        self.history = self.model.fit(
            X, y,
            epochs=self.epochs,
            batch_size=self.batch_size,
            callbacks=[early_stop],
            verbose=verbose
        )

        return self

    def forecast(self, horizon: int = 12) -> np.ndarray:
        """
        Рекурсивный прогноз на горизонт.

        Args:
            horizon: Количество месяцев

        Returns:
            numpy array с прогнозами (MoM в %)
        """
        if self.model is None:
            raise ValueError("Модель не обучена")

        predictions = []
        current_sequence = self.last_sequence.copy()

        for _ in range(horizon):
            # Reshape для предсказания
            X = current_sequence.reshape(1, self.sequence_length, 1)

            # Прогноз
            pred = self.model.predict(X, verbose=0)[0, 0]
            predictions.append(pred)

            # Обновляем последовательность
            current_sequence = np.append(current_sequence[1:], [[pred]], axis=0)

        # Обратное масштабирование
        predictions = np.array(predictions).reshape(-1, 1)
        predictions = self.scaler.inverse_transform(predictions).flatten()

        return predictions

    def backtest(
        self,
        df: pd.DataFrame,
        start_date: str = '2019-01-01',
        target_col: str = 'Все товары и услуги'
    ) -> pd.DataFrame:
        """
        Бэктестирование модели.

        Note: LSTM бэктест медленный из-за переобучения на каждую дату.

        Args:
            df: DataFrame с данными
            start_date: Начальная дата
            target_col: Целевая колонка

        Returns:
            DataFrame с результатами
        """
        if target_col in df.columns:
            series = df[target_col].dropna()
        else:
            series = df.dropna()

        # Конвертируем
        if series.mean() > 50:
            series = series - 100

        test_dates = series[series.index >= start_date].index
        results = []

        for target_date in test_dates:
            cutoff = target_date - pd.DateOffset(months=1)
            train_series = series[series.index <= cutoff]

            # Нужно минимум sequence_length + 10 наблюдений
            if len(train_series) < self.sequence_length + 10:
                continue

            try:
                # Создаём новую модель для чистого бэктеста
                model = SirenaLSTM(
                    sequence_length=self.sequence_length,
                    epochs=50,  # Меньше эпох для скорости
                    patience=5
                )

                train_df = pd.DataFrame({
                    target_col: train_series.values + 100
                }, index=train_series.index)

                model.fit(train_df, target_col, verbose=0)
                fc = model.forecast(horizon=1)

                prediction = fc[0]
                actual = series.loc[target_date]

                results.append({
                    'date': target_date,
                    'actual': actual,
                    'prediction': prediction,
                    'error': actual - prediction
                })
            except Exception:
                continue

        return pd.DataFrame(results)

    def get_training_history(self) -> dict:
        """Получить историю обучения."""
        if self.history is None:
            return {}

        return {
            'loss': self.history.history['loss'],
            'mae': self.history.history.get('mae', [])
        }


# Fallback модель без TensorFlow (простая авторегрессия)
class SirenaLSTMFallback:
    """
    Fallback модель если TensorFlow недоступен.
    Использует простую авторегрессию.
    """

    def __init__(self, sequence_length: int = 12):
        self.sequence_length = sequence_length
        self.coefficients = None
        self.last_values = None

    def fit(self, df: pd.DataFrame, target_col: str = 'Все товары и услуги'):
        """Простое обучение через OLS."""
        from sklearn.linear_model import Ridge

        if target_col in df.columns:
            series = df[target_col].dropna().values
        else:
            series = df.dropna().values

        if np.mean(series) > 50:
            series = series - 100

        # Создаём лаговые признаки
        X, y = [], []
        for i in range(self.sequence_length, len(series)):
            X.append(series[i - self.sequence_length:i])
            y.append(series[i])

        X = np.array(X)
        y = np.array(y)

        model = Ridge(alpha=1.0)
        model.fit(X, y)
        self.coefficients = model

        self.last_values = series[-self.sequence_length:]
        return self

    def forecast(self, horizon: int = 12) -> np.ndarray:
        """Авторегрессивный прогноз."""
        if self.coefficients is None:
            raise ValueError("Модель не обучена")

        predictions = []
        current = self.last_values.copy()

        for _ in range(horizon):
            pred = self.coefficients.predict(current.reshape(1, -1))[0]
            predictions.append(pred)
            current = np.append(current[1:], pred)

        return np.array(predictions)


# Автоматический выбор модели
def create_lstm_model(**kwargs):
    """Создаёт LSTM или fallback модель."""
    if TENSORFLOW_AVAILABLE:
        return SirenaLSTM(**kwargs)
    else:
        print("Using fallback AR model (TensorFlow not available)")
        return SirenaLSTMFallback(sequence_length=kwargs.get('sequence_length', 12))


# Тестирование
if __name__ == "__main__":
    # Загрузка данных
    df_raw = pd.read_csv('data/infl_kbr.csv', sep=';', decimal='.')

    try:
        df_raw['Date'] = pd.to_datetime(df_raw['Day'], format='%d.%m.%Y')
    except:
        df_raw['Date'] = pd.to_datetime(df_raw['Day'])

    if 'Товар' in df_raw.columns:
        df = df_raw.pivot_table(index='Date', columns='Товар', values='MoM', aggfunc='first')
    else:
        df = df_raw.set_index('Date')

    df = df.sort_index()

    print(f"TensorFlow available: {TENSORFLOW_AVAILABLE}")

    # Создаём модель
    model = create_lstm_model(epochs=50, patience=5)
    print(f"Using model: {model.__class__.__name__}")

    print("\nОбучение модели...")
    model.fit(df, 'Все товары и услуги')

    # Прогноз
    fc = model.forecast(12)
    print("\nLSTM Прогноз на 12 месяцев (MoM %):")
    print(fc)

    # История обучения (только для TensorFlow)
    if TENSORFLOW_AVAILABLE and hasattr(model, 'get_training_history'):
        history = model.get_training_history()
        if history:
            print(f"\nФинальный loss: {history['loss'][-1]:.4f}")
