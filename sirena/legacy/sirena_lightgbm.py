"""
LightGBM модель для прогнозирования инфляции КБР
=================================================

Gradient Boosting модель для захвата нелинейных зависимостей.
Использует те же признаки, что и Ridge модель.

Автор: СИРЕНА-КБР
Версия: 1.0
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
import warnings

warnings.filterwarnings('ignore')

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    print("Warning: LightGBM not installed. Run: pip install lightgbm")


class SirenaLightGBM:
    """
    LightGBM модель прогнозирования инфляции.

    Использует те же 11 признаков, что и Ridge:
    - Лаги: y_lag1, y_lag2, y_lag12, y_ma3
    - Сезонность: month_sin, month_cos
    - Компоненты: food_lag1, nonfood_lag1, services_lag1
    - Отклонения: seasonal_norm, deviation_lag1
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 5,
        learning_rate: float = 0.1,
        num_leaves: int = 31,
        min_child_samples: int = 10,
        outlier_years: list = None
    ):
        if not LIGHTGBM_AVAILABLE:
            raise ImportError("LightGBM not installed")

        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.num_leaves = num_leaves
        self.min_child_samples = min_child_samples
        self.outlier_years = outlier_years or [2010, 2022]

        self.model = None
        self.scaler = RobustScaler()
        self.seasonal_norm = None
        self.last_values = None

        self.feature_cols = [
            'y_lag1', 'y_lag2', 'y_lag12', 'y_ma3',
            'month_sin', 'month_cos',
            'food_lag1', 'nonfood_lag1', 'services_lag1',
            'seasonal_norm', 'deviation_lag1'
        ]

    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Подготовка признаков (аналогично Ridge)."""
        df = df.copy()
        df['month'] = df.index.month
        df['year'] = df.index.year

        # Лаги
        df['y_lag1'] = df['Все товары и услуги'].shift(1)
        df['y_lag2'] = df['Все товары и услуги'].shift(2)
        df['y_lag12'] = df['Все товары и услуги'].shift(12)
        df['y_ma3'] = df['Все товары и услуги'].rolling(3).mean().shift(1)

        # Сезонность (тригонометрическая)
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

        # Компоненты
        df['food_lag1'] = df['Продовольственные товары'].shift(1)
        df['nonfood_lag1'] = df['Непродовольственные товары'].shift(1)
        df['services_lag1'] = df['Услуги'].shift(1)

        # Сезонная норма
        clean = df[~df['year'].isin(self.outlier_years)]
        if len(clean) < 12:
            clean = df
        self.seasonal_norm = clean.groupby('month')['Все товары и услуги'].mean()

        for m in range(1, 13):
            if m not in self.seasonal_norm:
                self.seasonal_norm[m] = 100.5

        df['seasonal_norm'] = df['month'].map(self.seasonal_norm)
        df['deviation_lag1'] = df['y_lag1'] - df['month'].shift(1).map(self.seasonal_norm)

        return df

    def fit(self, df: pd.DataFrame):
        """
        Обучение модели.

        Args:
            df: DataFrame с колонками 'Все товары и услуги',
                'Продовольственные товары', 'Непродовольственные товары', 'Услуги'

        Returns:
            self
        """
        df = self.prepare_features(df)

        # Очистка данных
        train = df.dropna(subset=self.feature_cols + ['Все товары и услуги'])
        train = train[~train['year'].isin(self.outlier_years)]

        if len(train) < 24:
            raise ValueError(f"Недостаточно данных: {len(train)} < 24")

        X = train[self.feature_cols].values
        y = train['Все товары и услуги'].values

        # Масштабирование
        X_scaled = self.scaler.fit_transform(X)

        # Сохраняем последние значения для прогноза
        self.last_values = {
            'y': list(df['Все товары и услуги'].dropna().values),
            'food': df['Продовольственные товары'].iloc[-1],
            'nonfood': df['Непродовольственные товары'].iloc[-1],
            'services': df['Услуги'].iloc[-1]
        }

        # Обучение LightGBM
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

        return self

    def predict(self, df: pd.DataFrame, target_date: pd.Timestamp) -> dict:
        """
        Прогноз на одну дату.

        Args:
            df: DataFrame с историческими данными
            target_date: Дата прогноза

        Returns:
            dict с прогнозом
        """
        if self.model is None:
            raise ValueError("Модель не обучена")

        df = self.prepare_features(df)
        test_row = df.loc[[target_date]]

        X = self.scaler.transform(test_row[self.feature_cols].values)
        prediction = self.model.predict(X)[0]

        return {
            'date': target_date,
            'prediction': prediction,
            'month': target_date.month
        }

    def forecast(self, horizon: int = 12, start_date: pd.Timestamp = None) -> np.ndarray:
        """
        Рекурсивный прогноз на горизонт.

        Args:
            horizon: Количество месяцев
            start_date: Начальная дата (если None, используется последняя + 1 месяц)

        Returns:
            numpy array с прогнозами (MoM в %)
        """
        if self.model is None:
            raise ValueError("Модель не обучена")

        history = self.last_values['y'].copy()
        forecasts = []

        for i in range(horizon):
            # Определяем месяц
            if start_date:
                t_date = start_date + pd.DateOffset(months=i)
            else:
                t_date = pd.Timestamp.now().replace(day=1) + pd.DateOffset(months=i)

            t_m = t_date.month

            # Формируем признаки
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

    def backtest(
        self,
        df: pd.DataFrame,
        start_date: str = '2019-01-01',
        end_date: str = None
    ) -> pd.DataFrame:
        """
        Бэктестирование модели.

        Args:
            df: DataFrame с данными
            start_date: Начало бэктеста
            end_date: Конец бэктеста (None = последняя дата)

        Returns:
            DataFrame с результатами
        """
        valid = df.dropna(subset=['Все товары и услуги'])
        last_date = valid.index.max()

        if end_date is None:
            end_date = last_date
        else:
            end_date = min(pd.Timestamp(end_date), last_date)

        test_dates = pd.date_range(start_date, end_date, freq='MS')
        results = []

        for target_date in test_dates:
            if target_date not in df.index:
                continue

            actual = df.loc[target_date, 'Все товары и услуги']
            if pd.isna(actual):
                continue

            cutoff = target_date - pd.DateOffset(months=1)
            train_df = df[df.index <= cutoff].copy()
            train_df = train_df.dropna(subset=['Все товары и услуги'])

            if len(train_df) < 36:
                continue

            try:
                # Создаём новый экземпляр
                model = SirenaLightGBM(
                    n_estimators=self.n_estimators,
                    max_depth=self.max_depth,
                    outlier_years=self.outlier_years
                )
                model.fit(train_df)

                test_df = df[df.index <= target_date].copy()
                pred = model.predict(test_df, target_date)

                results.append({
                    'date': target_date,
                    'actual': actual,
                    'prediction': pred['prediction'],
                    'error': actual - pred['prediction']
                })
            except Exception:
                continue

        return pd.DataFrame(results)

    def get_feature_importance(self) -> pd.DataFrame:
        """Получить важность признаков."""
        if self.model is None:
            return pd.DataFrame()

        importance = pd.DataFrame({
            'feature': self.feature_cols,
            'importance': self.model.feature_importances_
        })
        return importance.sort_values('importance', ascending=False)


# Тестирование
if __name__ == "__main__":
    if not LIGHTGBM_AVAILABLE:
        print("LightGBM not available")
        exit()

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

    # Обучение
    model = SirenaLightGBM()
    model.fit(df)

    # Прогноз
    fc = model.forecast(12)
    print("LightGBM Прогноз на 12 месяцев (MoM %):")
    print(fc)

    # Важность признаков
    print("\nВажность признаков:")
    print(model.get_feature_importance())

    # Бэктест
    bt = model.backtest(df, start_date='2023-01-01')
    if not bt.empty:
        mae = bt['error'].abs().mean()
        print(f"\nMAE на бэктесте: {mae:.3f}")
