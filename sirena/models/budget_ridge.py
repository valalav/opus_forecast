"""
Budget-Enhanced Ridge модель для прогнозирования инфляции КБР
=============================================================

Модель использует бюджетные данные КБР как опережающие индикаторы.

По результатам исследования (январь 2026):
- Доходы КБР lag 3: r=+0.484*** (p=0.0000)
- НДФЛ КБР lag 3: r=+0.466*** (p=0.0000)
- Налог на прибыль lag 3: r=+0.427*** (p=0.0000)

Бюджетные данные опережают инфляцию на 3 месяца.
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge, HuberRegressor
from sklearn.preprocessing import StandardScaler
from typing import Dict, Optional, Any, List
from pathlib import Path

from .base import BaseForecaster
from .registry import ModelRegistry


@ModelRegistry.register("budget_ridge")
class BudgetRidgeForecaster(BaseForecaster):
    """
    Ridge регрессия с бюджетными признаками КБР.

    Особенности:
    - Использует бюджетные доходы КБР с лагом 3 месяца
    - Комбинирует с базовыми макро-признаками
    - Исключение выбросных лет (2022, 2010)

    Бюджетные признаки (lag 3):
    - budget_income_L3: Доходы всего (r=+0.484)
    - budget_ndfl_L3: НДФЛ (r=+0.466)
    - budget_profit_L3: Налог на прибыль (r=+0.427)
    """

    name = "budget_ridge"
    MIN_TRAIN_SIZE = 36

    # Годы-выбросы
    OUTLIER_YEARS = [2022, 2010]

    # Путь к данным бюджета
    BUDGET_DATA_PATH = "edge_lab/data/kbr_budget_data.csv"

    # Признаки модели
    FEATURES = [
        # Авторегрессия
        'mom_L1', 'mom_L2', 'mom_L3',
        # Бюджетные признаки с оптимальным лагом 3
        'budget_income_L3',      # Доходы КБР (r=+0.484***)
        'budget_ndfl_L3',        # НДФЛ (r=+0.466***)
        'budget_profit_L3',      # Налог на прибыль (r=+0.427***)
        # Базовые макро
        'usd_L2',                # USD lag 2
        'ki_L6',                 # Ki lag 6
        # Компоненты
        'prod_L1',
        'serv_L1',
        # Сезонность
        'month_sin', 'month_cos',
    ]

    def __init__(
        self,
        alpha: float = 1.0,
        use_huber: bool = False,
        budget_data_path: Optional[str] = None,
        **kwargs
    ):
        """
        Args:
            alpha: Ridge регуляризация
            use_huber: Использовать Huber вместо Ridge
            budget_data_path: Путь к CSV с бюджетными данными
        """
        super().__init__(**kwargs)
        self.alpha = alpha
        self.use_huber = use_huber
        self.budget_data_path = budget_data_path or self.BUDGET_DATA_PATH

        self.model = None
        self.scaler = None
        self._available_features = None
        self._train_df = None
        self._budget_df = None
        self._seasonal_norms = {}

    def _load_budget_data(self) -> Optional[pd.DataFrame]:
        """Загрузка бюджетных данных КБР."""
        path = Path(self.budget_data_path)

        if not path.exists():
            # Попробуем альтернативные пути
            alt_paths = [
                Path("edge_lab/data/kbr_budget_data.csv"),
                Path("data/kbr_budget_data.csv"),
            ]
            for alt in alt_paths:
                if alt.exists():
                    path = alt
                    break

        if not path.exists():
            print(f"Warning: Budget data not found at {path}")
            return None

        try:
            budget = pd.read_csv(path, parse_dates=['date'])
            budget = budget.set_index('date')
            budget.index = budget.index.to_period('M').to_timestamp()
            return budget
        except Exception as e:
            print(f"Warning: Failed to load budget data: {e}")
            return None

    def _compute_seasonal_norms(self, df: pd.DataFrame) -> Dict[str, Dict[int, float]]:
        """Вычисление сезонных норм."""
        norms = {}
        df_clean = df[~df.index.year.isin(self.OUTLIER_YEARS)]

        for col in ['mom', 'Prod', 'Serv', 'Nonprod']:
            if col in df_clean.columns:
                monthly = df_clean.groupby(df_clean.index.month)[col].mean()
                norms[col] = monthly.to_dict()

        return norms

    def _prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Создание признаков включая бюджетные."""
        df = df.copy()

        # Целевая переменная
        if 'mom' in df.columns:
            y = df['mom']
        elif 'Все товары и услуги' in df.columns:
            y = df['Все товары и услуги']
        else:
            raise ValueError("No target column found")

        df['mom'] = y

        # Авторегрессионные признаки
        df['mom_L1'] = y.shift(1)
        df['mom_L2'] = y.shift(2)
        df['mom_L3'] = y.shift(3)

        # ========== БЮДЖЕТНЫЕ ПРИЗНАКИ ==========
        if self._budget_df is not None:
            # Merge бюджетных данных
            df_month = df.index.to_period('M').to_timestamp()

            for col_src, col_dst in [
                ('KBR_Доходы_всего', 'budget_income'),
                ('KBR_НДФЛ', 'budget_ndfl'),
                ('KBR_Налог на прибыль', 'budget_profit'),
            ]:
                if col_src in self._budget_df.columns:
                    # Создаём Series с правильным индексом
                    budget_series = self._budget_df[col_src].copy()

                    # Добавляем в df через reindex
                    df[col_dst] = budget_series.reindex(df_month).values

                    # Лаг 3 месяца (оптимальный по исследованию)
                    df[f'{col_dst}_L3'] = df[col_dst].shift(3)

                    # Используем YoY темп роста вместо абсолютных значений
                    # (темпы роста более информативны для инфляции)
                    if df[col_dst].notna().sum() > 12:
                        # YoY изменение
                        df[f'{col_dst}_yoy'] = df[col_dst].pct_change(12) * 100
                        df[f'{col_dst}_L3'] = df[f'{col_dst}_yoy'].shift(3)
                    else:
                        # MoM изменение если мало данных
                        df[f'{col_dst}_mom'] = df[col_dst].pct_change(1) * 100
                        df[f'{col_dst}_L3'] = df[f'{col_dst}_mom'].shift(3)

        # ========== МАКРО-ПРИЗНАКИ ==========
        # Ki с лагом 6
        if 'Ki_i' in df.columns:
            df['ki_L6'] = df['Ki_i'].shift(6)
        elif 'Ki' in df.columns:
            df['ki_L6'] = df['Ki'].shift(6)

        # USD с лагом 2
        if 'usd_nom_i' in df.columns:
            df['usd_L2'] = df['usd_nom_i'].shift(2)
        elif 'usd' in df.columns:
            df['usd_L2'] = df['usd'].shift(2)

        # Компоненты
        if 'Prod' in df.columns:
            df['prod_L1'] = df['Prod'].shift(1)
        elif 'Продовольственные товары' in df.columns:
            df['prod_L1'] = df['Продовольственные товары'].shift(1)

        if 'Serv' in df.columns:
            df['serv_L1'] = df['Serv'].shift(1)
        elif 'Услуги' in df.columns:
            df['serv_L1'] = df['Услуги'].shift(1)

        # Сезонность
        df['month'] = df.index.month
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        df['year'] = df.index.year

        return df

    def fit(self, df: pd.DataFrame, target_col: str = 'Все товары и услуги') -> 'BudgetRidgeForecaster':
        """Обучение модели."""
        # Загружаем бюджетные данные
        self._budget_df = self._load_budget_data()

        if self._budget_df is None:
            print("Warning: No budget data available, model will use only macro features")

        # Сезонные нормы
        self._seasonal_norms = self._compute_seasonal_norms(df)

        # Подготовка признаков
        df_prep = self._prepare_features(df)

        # Определяем доступные признаки
        self._available_features = [f for f in self.FEATURES if f in df_prep.columns]

        # Fallback если бюджетных данных нет
        if len(self._available_features) < 5:
            # Используем базовые признаки без бюджета
            base_features = ['mom_L1', 'mom_L2', 'mom_L3', 'month_sin', 'month_cos']
            self._available_features = [f for f in base_features if f in df_prep.columns]

        if len(self._available_features) < 3:
            raise ValueError(f"Недостаточно признаков: {self._available_features}")

        # Исключаем выбросные годы
        train_df = df_prep[~df_prep['year'].isin(self.OUTLIER_YEARS)]

        # Очистка от NaN
        train_clean = train_df.dropna(subset=self._available_features + ['mom'])

        if len(train_clean) < self.MIN_TRAIN_SIZE:
            raise ValueError(f"Недостаточно данных: {len(train_clean)} < {self.MIN_TRAIN_SIZE}")

        # Обучение
        X = train_clean[self._available_features].values
        y = train_clean['mom'].values

        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        if self.use_huber:
            self.model = HuberRegressor(epsilon=1.35)
        else:
            self.model = Ridge(alpha=self.alpha)

        self.model.fit(X_scaled, y)

        self._is_fitted = True
        self._last_train_date = df.index.max()
        self._train_df = df.copy()

        return self

    def forecast(self, horizon: int = 12) -> np.ndarray:
        """Итеративный прогноз."""
        self._check_fitted()

        if self._train_df is None:
            return np.zeros(horizon)

        df_work = self._train_df.copy()
        predictions = []

        for h in range(horizon):
            target_date = self._last_train_date + pd.DateOffset(months=h+1)

            # Добавляем строку для целевой даты
            if target_date not in df_work.index:
                df_work.loc[target_date] = np.nan

            try:
                pred_result = self.predict(df_work, target_date)
                pred = pred_result['prediction']
            except Exception as e:
                # Fallback: сезонная норма
                month = target_date.month
                pred = self._seasonal_norms.get('mom', {}).get(month, 100.0)

            predictions.append(pred)

            # Обновляем mom для авторегрессии
            df_work.loc[target_date, 'mom'] = pred

        # Конвертируем из индекса в MoM%
        return np.array(predictions) - 100

    def predict(self, df: pd.DataFrame, target_date: pd.Timestamp) -> Dict[str, Any]:
        """Точечный прогноз на дату."""
        self._check_fitted()

        df_prep = self._prepare_features(df)

        if target_date not in df_prep.index:
            raise ValueError(f"target_date {target_date} not in data index")

        # Forward fill
        df_prep = df_prep.ffill()

        test_row = df_prep.loc[[target_date], self._available_features]

        if test_row.isna().any().any():
            test_row = test_row.fillna(0)

        X_test = self.scaler.transform(test_row.values)
        prediction = self.model.predict(X_test)[0]

        return {
            'date': target_date,
            'prediction': prediction,
            'model': self.name,
            'n_features': len(self._available_features),
            'features_used': self._available_features,
            'has_budget_features': any('budget' in f for f in self._available_features)
        }

    def backtest(
        self,
        df: pd.DataFrame,
        start_date: str = '2019-01-01',
        target_col: str = 'Все товары и услуги',
        horizon: int = 1
    ) -> pd.DataFrame:
        """Бэктестирование модели."""
        start = pd.Timestamp(start_date)

        y_col = 'mom' if 'mom' in df.columns else target_col
        valid_dates = df.dropna(subset=[y_col]).index
        test_dates = valid_dates[valid_dates >= start]

        results = []

        for cutoff_date in test_dates:
            train_df = df[df.index < cutoff_date].copy()

            if len(train_df.dropna(subset=[y_col])) < self.MIN_TRAIN_SIZE:
                continue

            target_date = cutoff_date
            if horizon > 1:
                target_date = cutoff_date + pd.DateOffset(months=horizon-1)
                if target_date not in df.index:
                    continue

            try:
                model = BudgetRidgeForecaster(
                    alpha=self.alpha,
                    use_huber=self.use_huber,
                    budget_data_path=self.budget_data_path
                )
                model.fit(train_df, target_col)

                fc = model.forecast(horizon)
                pred = fc[-1] + 100

                actual = df.loc[target_date, y_col]

                results.append({
                    'date': target_date,
                    'cutoff': cutoff_date,
                    'horizon': horizon,
                    'actual': actual,
                    'prediction': pred,
                    'error': actual - pred,
                    'has_budget': model._budget_df is not None
                })
            except Exception as e:
                continue

        return pd.DataFrame(results)

    def get_feature_importance(self) -> pd.DataFrame:
        """Важность признаков."""
        self._check_fitted()

        importance = pd.DataFrame({
            'feature': self._available_features,
            'coefficient': self.model.coef_
        })
        importance['abs_coef'] = importance['coefficient'].abs()
        importance['is_budget'] = importance['feature'].str.contains('budget')

        return importance.sort_values('abs_coef', ascending=False)

    def get_metrics(self, results: pd.DataFrame) -> Dict[str, float]:
        """Расчёт метрик качества."""
        if results.empty:
            return {'MAE': 0, 'RMSE': 0, 'KPI': 0}

        errors = results['error'].abs()
        mae = errors.mean()
        rmse = np.sqrt((results['error'] ** 2).mean())
        kpi = (errors <= 0.5).sum() / len(results) * 100

        return {'MAE': mae, 'RMSE': rmse, 'KPI': kpi}
