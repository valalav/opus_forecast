"""
Ridge Macro модель для прогнозирования инфляции КБР
===================================================

Модель на основе результатов исследования признаков (декабрь 2024).
Использует оптимальные лаги для макро-показателей по тестам Грейнджера.

Ключевые признаки (по исследованию):
- USD lag 2 (p=0.0020)
- Ki lag 6 (p=0.0000)
- Ruonia_D1 (r=0.492)
- Brent lag 5 (p=0.0080)
- prod, serv компоненты

ВАЖНО: Модель прогнозирует экзогенные переменные для будущих периодов:
- Ki: последнее известное значение (ЦБ объявляет заранее)
- Ruonia: Ki - исторический спред
- USD: Random Walk (последнее известное)
- Brent: Random Walk (последнее известное)
- Prod/Serv: сезонная норма
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from sklearn.linear_model import Ridge, HuberRegressor
from sklearn.preprocessing import StandardScaler
from typing import Dict, Optional, Any, List

from .base import BaseForecaster
from .registry import ModelRegistry


@ModelRegistry.register("ridge_macro")
class RidgeMacroForecaster(BaseForecaster):
    """
    Ridge регрессия с оптимальными макро-признаками и прогнозом экзогенных.

    Особенности:
    - Лаги определены по тестам Грейнджера
    - Использует USD lag 2, Ki lag 6, Brent lag 5
    - Ruonia через первую разность (D1)
    - Компоненты prod и serv
    - Исключение выбросных лет (2022, 2010)
    - Прогнозирование экзогенных переменных для будущих периодов
    """

    name = "ridge_macro"
    MIN_TRAIN_SIZE = 36

    # Годы-выбросы
    OUTLIER_YEARS = [2022, 2010]

    # Признаки на основе исследования (с динамическими лагами)
    # FEATURES = [
    #    # Авторегрессия (начинаем с L1 - известные значения)
    #    'mom_L1', 'mom_L2', 'mom_L3',
    #    # Федеральные макро с оптимальными лагами
    #    'ki_L6',        # Грейнджер: лаг 6, p=0.0000
    #    'Ruonia_D1',    # r=0.492 (разница за предыдущий месяц)
    #    'usd_L2',       # Грейнджер: лаг 2, p=0.0020
    #    'brent_L5',     # Грейнджер: лаг 5, p=0.0080
    #    'brent_STD3',   # r=0.461
    #    # Компоненты (с лагом 1 - известные)
    #    'prod_L1',      # Продовольствие
    #    'serv_L1',      # Услуги
    #    # Сезонность
    #    'month_sin', 'month_cos',
    # ]

    DEFAULT_LAGS = {"ki": 6, "usd": 2, "brent": 5}

    @staticmethod
    def _load_optimal_lags():
        """Load optimal lags from data/optimal_lags.json if available."""
        lags = RidgeMacroForecaster.DEFAULT_LAGS.copy()

        # Try edge_lab directory first
        lag_files = [
            Path("data/optimal_lags.json"),
            Path("../edge_lab/data/optimal_lags.json"),
            Path("edge_lab/data/optimal_lags.json"),
        ]

        for lag_file in lag_files:
            if lag_file.exists():
                try:
                    with open(lag_file, "r") as f:
                        data = json.load(f)
                    if "ki" in data and data["ki"].get("optimal_lag") is not None:
                        lags["ki"] = data["ki"]["optimal_lag"]
                    if "usd" in data and data["usd"].get("optimal_lag") is not None:
                        lags["usd"] = data["usd"]["optimal_lag"]
                    if "brent" in data and data["brent"].get("optimal_lag") is not None:
                        lags["brent"] = data["brent"]["optimal_lag"]
                    break
                except Exception:
                    continue

        return lags

    def __init__(self, alpha: float = 1.0, use_huber: bool = False, **kwargs):
        """
        Args:
            alpha: Ridge регуляризация
            use_huber: Использовать Huber вместо Ridge (робастность к выбросам)
        """
        super().__init__(**kwargs)
        self.alpha = alpha
        self.use_huber = use_huber
        self.model = None
        self.scaler = None
        self._available_features = None
        self._train_df = None
        self._seasonal_norms = {}  # Сезонные нормы для компонентов
        self.optimal_lags = self._load_optimal_lags()

        # Build FEATURES list dynamically based on optimal lags
        ki_lag = self.optimal_lags["ki"]
        usd_lag = self.optimal_lags["usd"]
        brent_lag = self.optimal_lags["brent"]

        self.FEATURES = [
            "mom_L1",
            "mom_L2",
            "mom_L3",
            f"ki_L{ki_lag}",
            "Ruonia_D1",
            f"usd_L{usd_lag}",
            f"brent_L{brent_lag}",
            "brent_STD3",
            "prod_L1",
            "serv_L1",
            "month_sin",
            "month_cos",
        ]

    def _compute_seasonal_norms(self, df: pd.DataFrame) -> Dict[str, Dict[int, float]]:
        """Вычисление сезонных норм для компонентов."""
        norms = {}

        # Исключаем выбросные годы
        df_clean = df[~df.index.year.isin(self.OUTLIER_YEARS)]

        # Для каждой колонки вычисляем среднее по месяцам
        for col in ["mom", "Prod", "Serv", "Nonprod"]:
            if col in df_clean.columns:
                monthly = df_clean.groupby(df_clean.index.month)[col].mean()
                norms[col] = monthly.to_dict()

        return norms

    def _forecast_exogenous(self, df: pd.DataFrame, horizon: int) -> pd.DataFrame:
        """
        Прогноз экзогенных переменных на horizon месяцев.

        Стратегии (ВАЖНО: обеспечиваем непрерывность с историей!):
        - Ki: последнее известное (ЦБ объявляет заранее)
        - Ruonia: последнее известное + AR(1) тренд (не скачок!)
        - USD: Random Walk (последнее известное)
        - Brent: Random Walk (последнее известное)
        - Prod/Serv: сезонная норма

        Returns:
            DataFrame с прогнозами экзогенных на future dates
        """
        last_date = df.index.max()
        future_dates = pd.date_range(
            start=last_date + pd.DateOffset(months=1), periods=horizon, freq="MS"
        )

        # Создаём DataFrame для будущих значений
        future_df = pd.DataFrame(index=future_dates)

        # Ki: последнее известное значение
        ki_col = (
            "Ki_i" if "Ki_i" in df.columns else "Ki" if "Ki" in df.columns else None
        )
        if ki_col:
            future_df[ki_col] = df[ki_col].iloc[-1]

        # Ruonia: продолжаем тренд (AR(1)-like)
        # ВАЖНО: не делаем скачок! Ruonia_D1 должна быть близка к 0
        if "Ruonia" in df.columns:
            last_ruonia = df["Ruonia"].iloc[-1]
            # Вычисляем средний месячный прирост Ruonia за последние 6 месяцев
            ruonia_diff = df["Ruonia"].diff().tail(6).mean()
            # Ограничиваем прирост чтобы не было резких скачков
            ruonia_diff = np.clip(ruonia_diff, -0.5, 0.5)
            # Прогноз: продолжаем тренд
            for i, date in enumerate(future_dates):
                future_df.loc[date, "Ruonia"] = last_ruonia + ruonia_diff * (i + 1)

        # USD: Random Walk (последнее известное)
        if "usd_nom_i" in df.columns:
            future_df["usd_nom_i"] = df["usd_nom_i"].iloc[-1]
        elif "usd" in df.columns:
            future_df["usd"] = df["usd"].iloc[-1]

        # Brent: Random Walk с сохранением исторической волатильности
        # ВАЖНО: brent_STD3 нуждается в волатильности, иначе = 0
        if "brent" in df.columns:
            last_brent = df["brent"].iloc[-1]
            # Вычисляем историческую волатильность
            brent_std = df["brent"].tail(12).std()
            # Добавляем небольшой шум для сохранения волатильности
            np.random.seed(42)  # Для воспроизводимости
            for i, date in enumerate(future_dates):
                # Random walk с drift=0 и исторической волатильностью
                noise = np.random.normal(
                    0, brent_std * 0.1
                )  # 10% от историч. волатильности
                future_df.loc[date, "brent"] = last_brent + noise
        elif "brent_pct" in df.columns:
            future_df["brent_pct"] = df["brent_pct"].iloc[-1]

        # Компоненты: сезонная норма
        for col in ["Prod", "Serv", "Nonprod"]:
            if col in df.columns and col in self._seasonal_norms:
                for i, date in enumerate(future_dates):
                    month = date.month
                    future_df.loc[date, col] = self._seasonal_norms[col].get(
                        month, 100.0
                    )

        # mom: будет заполняться итеративно в forecast()
        future_df["mom"] = np.nan

        return future_df

    def _prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Создание всех признаков."""
        df = df.copy()

        # Целевая переменная
        if "mom" in df.columns:
            y = df["mom"]
        elif "Все товары и услуги" in df.columns:
            y = df["Все товары и услуги"]
        else:
            raise ValueError("No target column found")

        df["mom"] = y

        # Авторегрессионные признаки
        df["mom_L1"] = y.shift(1)
        df["mom_L2"] = y.shift(2)
        df["mom_L3"] = y.shift(3)

        # Ki с оптимальным лагом (динамический)
        ki_lag = self.optimal_lags.get("ki", 6)
        if "Ki_i" in df.columns:
            df[f"ki_L{ki_lag}"] = df["Ki_i"].shift(ki_lag)
        elif "Ki" in df.columns:
            df[f"ki_L{ki_lag}"] = df["Ki"].shift(ki_lag)

        # Ruonia: первая разность
        if "Ruonia" in df.columns:
            df["Ruonia_D1"] = df["Ruonia"].diff(1)

        # USD с оптимальным лагом (динамический)
        usd_lag = self.optimal_lags.get("usd", 2)
        if "usd_nom_i" in df.columns:
            df[f"usd_L{usd_lag}"] = df["usd_nom_i"].shift(usd_lag)
        elif "usd" in df.columns:
            df[f"usd_L{usd_lag}"] = df["usd"].shift(usd_lag)

        # Brent с оптимальным лагом (динамический) и волатильность
        brent_lag = self.optimal_lags.get("brent", 5)
        if "brent" in df.columns:
            df[f"brent_L{brent_lag}"] = df["brent"].shift(brent_lag)
            df["brent_STD3"] = df["brent"].rolling(3).std()
        elif "brent_pct" in df.columns:
            df[f"brent_L{brent_lag}"] = df["brent_pct"].shift(brent_lag)
            df["brent_STD3"] = df["brent_pct"].rolling(3).std()

        # Компоненты инфляции
        if "Prod" in df.columns:
            df["prod_L1"] = df["Prod"].shift(1)
        elif "prod" in df.columns:
            df["prod_L1"] = df["prod"].shift(1)
        elif "Продовольственные товары" in df.columns:
            df["prod_L1"] = df["Продовольственные товары"].shift(1)

        if "Serv" in df.columns:
            df["serv_L1"] = df["Serv"].shift(1)
        elif "serv" in df.columns:
            df["serv_L1"] = df["serv"].shift(1)
        elif "Услуги" in df.columns:
            df["serv_L1"] = df["Услуги"].shift(1)

        # Сезонность
        df["month"] = df.index.month
        df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
        df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

        df["year"] = df.index.year

        return df

    def fit(
        self, df: pd.DataFrame, target_col: str = "Все товары и услуги"
    ) -> "RidgeMacroForecaster":
        """Обучение модели."""
        # Вычисляем сезонные нормы для прогноза экзогенных
        self._seasonal_norms = self._compute_seasonal_norms(df)

        # Подготовка признаков
        df_prep = self._prepare_features(df)

        # Определяем доступные признаки
        self._available_features = [f for f in self.FEATURES if f in df_prep.columns]

        if len(self._available_features) < 5:
            raise ValueError(
                f"Недостаточно признаков: {len(self._available_features)} ({self._available_features})"
            )

        # Исключаем выбросные годы
        train_df = df_prep[~df_prep["year"].isin(self.OUTLIER_YEARS)]

        # Очистка от NaN
        train_clean = train_df.dropna(subset=self._available_features + ["mom"])

        if len(train_clean) < self.MIN_TRAIN_SIZE:
            raise ValueError(
                f"Недостаточно данных: {len(train_clean)} < {self.MIN_TRAIN_SIZE}"
            )

        # Обучение
        X = train_clean[self._available_features].values
        y = train_clean["mom"].values

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
        """
        Итеративный прогноз на горизонт с прогнозом экзогенных переменных.

        Args:
            horizon: Количество месяцев

        Returns:
            numpy array с прогнозами (MoM%, т.е. 0.5 означает 0.5%)
        """
        self._check_fitted()

        if self._train_df is None:
            return np.zeros(horizon)

        # Прогнозируем экзогенные переменные
        future_exog = self._forecast_exogenous(self._train_df, horizon)

        # Объединяем исторические и будущие данные
        df_work = pd.concat([self._train_df, future_exog])
        df_work = df_work.sort_index()

        predictions = []

        for h in range(horizon):
            target_date = self._last_train_date + pd.DateOffset(months=h + 1)

            try:
                pred_result = self.predict(df_work, target_date)
                pred = pred_result["prediction"]
            except Exception as e:
                # Fallback: сезонная норма
                month = target_date.month
                pred = self._seasonal_norms.get("mom", {}).get(month, 100.0)

            predictions.append(pred)

            # Добавляем прогноз в данные для следующего шага (авторегрессия)
            df_work.loc[target_date, "mom"] = pred

        # Конвертируем из индекса (100.xx) в MoM% (0.xx)
        predictions = np.array(predictions) - 100
        return predictions

    def predict(self, df: pd.DataFrame, target_date: pd.Timestamp) -> Dict[str, Any]:
        """Точечный прогноз на дату."""
        self._check_fitted()

        df_prep = self._prepare_features(df)

        if target_date not in df_prep.index:
            raise ValueError(f"target_date {target_date} not in data index")

        # Forward fill для заполнения лаговых признаков
        df_prep = df_prep.ffill()

        test_row = df_prep.loc[[target_date], self._available_features]

        if test_row.isna().any().any():
            test_row = test_row.fillna(0)

        X_test = self.scaler.transform(test_row.values)
        prediction = self.model.predict(X_test)[0]

        return {
            "date": target_date,
            "prediction": prediction,
            "model": self.name,
            "n_features": len(self._available_features),
            "features_used": self._available_features,
        }

    def backtest(
        self,
        df: pd.DataFrame,
        start_date: str = "2019-01-01",
        target_col: str = "Все товары и услуги",
        horizon: int = 1,
    ) -> pd.DataFrame:
        """
        Бэктестирование модели.

        ВАЖНО: При бэктесте модель НЕ использует будущие значения экзогенных.
        Для каждой точки прогноза используются только данные до cutoff date.

        Args:
            df: DataFrame с данными
            start_date: Начало бэктеста
            target_col: Целевая колонка
            horizon: Горизонт прогноза (1 = следующий месяц)
        """
        start = pd.Timestamp(start_date)

        if "mom" in df.columns:
            y_col = "mom"
        elif target_col in df.columns:
            y_col = target_col
        else:
            raise ValueError("No target column")

        valid_dates = df.dropna(subset=[y_col]).index
        test_dates = valid_dates[valid_dates >= start]

        results = []

        for cutoff_date in test_dates:
            # Данные до cutoff (не включая cutoff)
            train_df = df[df.index < cutoff_date].copy()

            if len(train_df.dropna(subset=[y_col])) < self.MIN_TRAIN_SIZE:
                continue

            # Целевая дата = cutoff + horizon месяцев
            # Для h=1: если cutoff=Dec, target=Dec (текущий месяц)
            # Корректируем: target_date = cutoff_date (прогнозируем месяц cutoff)
            target_date = cutoff_date

            # Если horizon > 1, сдвигаем target_date
            if horizon > 1:
                # Проверяем что target_date есть в данных
                target_date = cutoff_date + pd.DateOffset(months=horizon - 1)
                if target_date not in df.index:
                    continue

            try:
                # Обучаем модель на данных ДО cutoff
                model = RidgeMacroForecaster(alpha=self.alpha, use_huber=self.use_huber)
                model.fit(train_df, target_col)

                # Прогнозируем на horizon шагов
                fc = model.forecast(horizon)
                pred = (
                    fc[-1] + 100
                )  # Берём последний прогноз, конвертируем обратно в индекс

                actual = df.loc[target_date, y_col]

                results.append(
                    {
                        "date": target_date,
                        "cutoff": cutoff_date,
                        "horizon": horizon,
                        "actual": actual,
                        "prediction": pred,
                        "error": actual - pred,
                        "n_features": model._available_features,
                    }
                )
            except Exception as e:
                continue

        return pd.DataFrame(results)

    def get_feature_importance(self) -> pd.DataFrame:
        """Важность признаков."""
        self._check_fitted()

        importance = pd.DataFrame(
            {"feature": self._available_features, "coefficient": self.model.coef_}
        )
        importance["abs_coef"] = importance["coefficient"].abs()
        return importance.sort_values("abs_coef", ascending=False)

    def get_exogenous_forecasts(self, horizon: int = 12) -> pd.DataFrame:
        """
        Получить прогнозы экзогенных переменных.

        Useful для отладки и понимания что модель использует.
        """
        self._check_fitted()
        return self._forecast_exogenous(self._train_df, horizon)

    def get_metrics(self, results: pd.DataFrame) -> Dict[str, float]:
        """Расчёт метрик качества."""
        if results.empty:
            return {"MAE": 0, "RMSE": 0, "KPI": 0}

        errors = results["error"].abs()
        mae = errors.mean()
        rmse = np.sqrt((results["error"] ** 2).mean())
        kpi = (errors <= 0.5).sum() / len(results) * 100

        return {"MAE": mae, "RMSE": rmse, "KPI": kpi}
