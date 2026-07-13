"""
BVAR модель со ставкой для сценарного анализа
=============================================

Модель для анализа влияния ключевой ставки (Ki/Ruonia) на инфляцию
по субкомпонентам CPI с расчётом IRF (Impulse Response Functions).

Сценарии:
- base: Ki без изменений
- hike: Ki +2 п.п. за 6 месяцев
- cut: Ki -2 п.п. за 6 месяцев
- custom: пользовательская траектория

Каналы трансмиссии:
- Кредитный (3-6 мес): автомобили, мебель, электроника
- Курсовой (1-3 мес): импортные товары
- Депозитный (6-12 мес): услуги
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Union, Tuple
from scipy.stats import invwishart

from .base import BaseForecaster
from .registry import ModelRegistry


@ModelRegistry.register("bvar_rate")
class BVARRateForecaster(BaseForecaster):
    """
    BVAR модель с эндогенной ключевой ставкой для сценарного анализа.

    Позволяет прогнозировать инфляцию при разных траекториях ставки.

    Переменные (по умолчанию):
    - CPI total
    - CPI food (продовольствие)
    - CPI nonfood (непродовольствие)
    - CPI services (услуги)
    - Ki (ключевая ставка)
    - Ruonia

    Сценарии ставки:
    - 'base': текущий уровень Ki
    - 'hike': повышение на 2 п.п. за 6 месяцев
    - 'cut': снижение на 2 п.п. за 6 месяцев
    - array[horizon]: пользовательская траектория Ki
    """

    name = "bvar_rate"
    MIN_TRAIN_SIZE = 36

    # Веса компонентов в общем CPI
    COMPONENT_WEIGHTS = {
        'food': 0.3948,
        'nonfood': 0.3653,
        'services': 0.2342
    }

    def __init__(
        self,
        lags: int = 2,
        lambda1: float = 0.2,
        lambda2: float = 0.5,
        lambda3: float = 1.0,
        lambda4: float = 100,
        n_draws: int = 500,
        include_ruonia: bool = True,
        include_usd: bool = False,
        **kwargs
    ):
        """
        Инициализация BVAR Rate.

        Args:
            lags: Число лагов VAR
            lambda1: Overall tightness (Minnesota prior)
            lambda2: Cross-variable tightness
            lambda3: Lag decay
            lambda4: Intercept prior variance
            n_draws: Число draws для прогноза
            include_ruonia: Включать ли Ruonia в модель
            include_usd: Включать ли USD/RUB в модель
        """
        super().__init__(**kwargs)
        self.lags = lags
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.lambda3 = lambda3
        self.lambda4 = lambda4
        self.n_draws = n_draws
        self.include_ruonia = include_ruonia
        self.include_usd = include_usd

        # Внутренние переменные
        self.var_names: List[str] = []
        self.k = 0
        self.n_params = 0
        self.raw_data = None
        self.Y = None
        self.X = None
        self.T = 0

        # Posterior
        self.B_post = None
        self.V_post = None
        self.Sigma_post = None
        self.sigma_i = None
        self.S_post = None
        self.d_post = None

        # Позиция Ki в переменных (для сценариев)
        self.ki_index = -1
        self.ruonia_index = -1

        # Последнее значение Ki для сценариев
        self.last_ki = None

    def _prepare_data(self, df: pd.DataFrame) -> None:
        """
        Подготовка данных для VAR с Ki/Ruonia.

        Args:
            df: DataFrame с данными (CPI компоненты + макро)
        """
        # Определяем переменные
        self.var_names = []

        # CPI компоненты
        cpi_cols = ['mom', 'Prod', 'Nonprod', 'Serv']
        for col in cpi_cols:
            if col in df.columns:
                self.var_names.append(col)

        # Макро-переменные
        if 'Ki_i' in df.columns:
            self.var_names.append('Ki_i')
            self.ki_index = len(self.var_names) - 1

        if self.include_ruonia and 'Ruonia' in df.columns:
            self.var_names.append('Ruonia')
            self.ruonia_index = len(self.var_names) - 1

        if self.include_usd and 'usd_nom_i' in df.columns:
            self.var_names.append('usd_nom_i')

        self.k = len(self.var_names)

        # Конвертируем данные
        data = df[self.var_names].dropna().values.astype(np.float64)

        # Преобразуем в MoM % (если индексы около 100)
        for i, col in enumerate(self.var_names):
            if col in ['mom', 'Prod', 'Nonprod', 'Serv', 'usd_nom_i', 'Ki_i']:
                if np.mean(data[:, i]) > 50:
                    data[:, i] = data[:, i] - 100

        self.raw_data = data
        self.last_ki = data[-1, self.ki_index] if self.ki_index >= 0 else 0

        T_total = len(data)
        T = T_total - self.lags

        # Y: зависимая переменная
        self.Y = data[self.lags:, :]

        # X: константа + лаги
        self.X = np.ones((T, 1 + self.k * self.lags))
        for t in range(T):
            for lag in range(1, self.lags + 1):
                start_col = 1 + (lag - 1) * self.k
                self.X[t, start_col:start_col + self.k] = data[self.lags + t - lag, :]

        self.T = T
        self.n_params = self.X.shape[1]

    def _minnesota_prior(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Minnesota prior (Litterman, 1986)."""
        n = self.n_params
        k = self.k

        beta0 = np.zeros((n, k))
        # AR(1) persistence для CPI переменных
        for i in range(min(4, k)):  # Первые 4 = CPI компоненты
            if 1 + i < n:
                beta0[1 + i, i] = 0.7  # Умеренная persistence

        V0 = np.zeros((n, k))

        sigma_i = np.array([
            max(np.std(np.diff(self.raw_data[:, i])), 1e-6)
            for i in range(k)
        ])

        V0[0, :] = self.lambda4 * sigma_i**2

        for lag in range(1, self.lags + 1):
            for j in range(k):
                row = 1 + (lag - 1) * k + j
                if row >= n:
                    continue
                for i in range(k):
                    if i == j:
                        V0[row, i] = (self.lambda1 / lag**self.lambda3)**2
                    else:
                        V0[row, i] = (
                            (self.lambda1 * self.lambda2 / lag**self.lambda3)**2
                            * (sigma_i[i] / sigma_i[j])**2
                        )

        return beta0, V0, sigma_i

    def _inverse_wishart_prior(self) -> Tuple[np.ndarray, int]:
        """Inverse-Wishart prior на ковариационную матрицу."""
        k = self.k
        d0 = k + 2
        S0 = np.diag(self.sigma_i**2) * max(1, d0 - k - 1)
        return S0, d0

    def fit(
        self,
        df: pd.DataFrame,
        target_col: str = 'mom'
    ) -> 'BVARRateForecaster':
        """
        Обучение BVAR Rate на данных с Ki/Ruonia.

        Args:
            df: DataFrame с CPI компонентами и Ki/Ruonia
            target_col: Целевая колонка (по умолчанию 'mom' = общий CPI)

        Returns:
            self
        """
        self._prepare_data(df)

        if self.T < self.MIN_TRAIN_SIZE:
            raise ValueError(f"Недостаточно данных: {self.T} < {self.MIN_TRAIN_SIZE}")

        # Minnesota Prior
        beta0, V0, sigma_i = self._minnesota_prior()
        self.sigma_i = sigma_i

        XtX = self.X.T @ self.X
        XtY = self.X.T @ self.Y

        # Posterior на коэффициенты
        beta_post = np.zeros_like(beta0)
        V_post = np.zeros_like(V0)

        for i in range(self.k):
            V0_inv = np.diag(1.0 / (V0[:, i] + 1e-10))
            V_post_inv = V0_inv + XtX / (sigma_i[i]**2)
            V_post_i = np.linalg.inv(V_post_inv)

            beta_post[:, i] = V_post_i @ (
                V0_inv @ beta0[:, i] + XtY[:, i] / sigma_i[i]**2
            )
            V_post[:, i] = np.diag(V_post_i)

        # Inverse-Wishart posterior на Σ
        S0, d0 = self._inverse_wishart_prior()
        resid = self.Y - self.X @ beta_post

        self.S_post = S0 + resid.T @ resid
        self.d_post = d0 + self.T

        self.Sigma_post = self.S_post / max(1, self.d_post - self.k - 1)

        self.B_post = beta_post
        self.V_post = V_post

        self._is_fitted = True
        self._last_train_date = df.index.max()

        return self

    def _generate_ki_scenario(
        self,
        scenario: Union[str, np.ndarray],
        horizon: int
    ) -> np.ndarray:
        """
        Генерация траектории Ki для сценария.

        Args:
            scenario: 'base', 'hike', 'cut' или массив значений
            horizon: Горизонт прогноза

        Returns:
            Массив изменений Ki (в п.п.) для каждого месяца
        """
        if isinstance(scenario, (list, np.ndarray)):
            ki_path = np.array(scenario)
            if len(ki_path) < horizon:
                # Дополняем последним значением
                ki_path = np.concatenate([
                    ki_path,
                    np.full(horizon - len(ki_path), ki_path[-1])
                ])
            return ki_path[:horizon]

        if scenario == 'base':
            # Ki остаётся без изменений
            return np.zeros(horizon)

        elif scenario == 'hike':
            # Повышение на 2 п.п. за 6 месяцев, потом стабильно
            ki_path = np.zeros(horizon)
            hike_months = min(6, horizon)
            monthly_hike = 2.0 / 6  # ~0.33 п.п. в месяц
            for i in range(hike_months):
                ki_path[i] = monthly_hike * (i + 1)
            for i in range(hike_months, horizon):
                ki_path[i] = 2.0  # Стабильно +2 п.п.
            return ki_path

        elif scenario == 'cut':
            # Снижение на 2 п.п. за 6 месяцев
            ki_path = np.zeros(horizon)
            cut_months = min(6, horizon)
            monthly_cut = -2.0 / 6  # ~-0.33 п.п. в месяц
            for i in range(cut_months):
                ki_path[i] = monthly_cut * (i + 1)
            for i in range(cut_months, horizon):
                ki_path[i] = -2.0  # Стабильно -2 п.п.
            return ki_path

        else:
            raise ValueError(f"Неизвестный сценарий: {scenario}")

    def forecast_scenario(
        self,
        horizon: int = 12,
        ki_scenario: Union[str, np.ndarray] = 'base'
    ) -> Dict[str, np.ndarray]:
        """
        Прогноз при заданном сценарии ставки.

        Args:
            horizon: Горизонт прогноза
            ki_scenario: 'base', 'hike', 'cut' или массив значений Ki

        Returns:
            Dict с прогнозами:
            - total: общий CPI
            - food: продовольствие
            - nonfood: непродовольствие
            - services: услуги
            - ki: траектория Ki
            - quantiles: доверительные интервалы
        """
        self._check_fitted()

        # Генерируем траекторию Ki
        ki_path = self._generate_ki_scenario(ki_scenario, horizon)

        # Preallocate forecasts
        forecasts = np.zeros((self.n_draws, horizon, self.k))
        Y_history = self.raw_data[-self.lags:, :].copy()

        for draw in range(self.n_draws):
            # Draw Σ из IW posterior
            try:
                Sigma_draw = invwishart.rvs(df=self.d_post, scale=self.S_post)
            except:
                Sigma_draw = self.Sigma_post

            Sigma_draw = np.atleast_2d(Sigma_draw)

            # Draw коэффициентов
            beta_draw = np.zeros_like(self.B_post)
            for i in range(self.k):
                scale_factor = Sigma_draw[i, i] / (self.sigma_i[i]**2)
                V_scaled = self.V_post[:, i] * scale_factor

                beta_draw[:, i] = np.random.normal(
                    self.B_post[:, i],
                    np.sqrt(np.maximum(V_scaled, 1e-10))
                )

            # Cholesky
            try:
                L = np.linalg.cholesky(Sigma_draw + np.eye(self.k) * 1e-6)
            except np.linalg.LinAlgError:
                L = np.eye(self.k) * np.sqrt(np.diag(Sigma_draw).mean())

            Y_curr = Y_history.copy()

            for t in range(horizon):
                X_t = np.ones(1 + self.k * self.lags)
                for lag in range(1, self.lags + 1):
                    idx = -lag
                    X_t[1 + (lag - 1) * self.k: 1 + lag * self.k] = Y_curr[idx, :]

                Y_mean = X_t @ beta_draw
                shock = L @ np.random.randn(self.k)
                Y_new = Y_mean + shock

                # Применяем сценарий Ki (заменяем Ki на заданное значение)
                if self.ki_index >= 0:
                    Y_new[self.ki_index] = ki_path[t]

                forecasts[draw, t, :] = Y_new
                Y_curr = np.vstack([Y_curr, Y_new])

        # Агрегируем результаты
        result = {
            'total': np.median(forecasts[:, :, 0], axis=0),
            'total_mean': np.mean(forecasts[:, :, 0], axis=0),
            'total_std': np.std(forecasts[:, :, 0], axis=0),
            'total_q05': np.percentile(forecasts[:, :, 0], 5, axis=0),
            'total_q95': np.percentile(forecasts[:, :, 0], 95, axis=0),
        }

        # Добавляем компоненты если они есть
        component_names = ['food', 'nonfood', 'services']
        for i, name in enumerate(component_names):
            if i + 1 < self.k:
                result[name] = np.median(forecasts[:, :, i + 1], axis=0)

        # Траектория Ki
        result['ki_path'] = ki_path
        result['ki_scenario'] = ki_scenario if isinstance(ki_scenario, str) else 'custom'

        return result

    def forecast(self, horizon: int = 12) -> np.ndarray:
        """
        Базовый прогноз (сценарий base).

        Args:
            horizon: Горизонт прогноза

        Returns:
            Массив MoM прогнозов
        """
        result = self.forecast_scenario(horizon, ki_scenario='base')
        return result['total']

    def compute_irf(
        self,
        shock_var: str = 'Ki_i',
        shock_size: float = 1.0,
        horizon: int = 24,
        n_draws: int = 1000
    ) -> Dict[str, np.ndarray]:
        """
        Impulse Response Function: реакция переменных на шок.

        Args:
            shock_var: Переменная для шока (по умолчанию Ki)
            shock_size: Размер шока (по умолчанию 1 п.п.)
            horizon: Горизонт IRF
            n_draws: Число симуляций для CI

        Returns:
            Dict с IRF для каждой переменной + confidence bands
        """
        self._check_fitted()

        # Находим индекс переменной для шока
        shock_idx = self.var_names.index(shock_var) if shock_var in self.var_names else self.ki_index

        if shock_idx < 0:
            raise ValueError(f"Переменная {shock_var} не найдена в модели")

        # IRF через симуляцию
        irf_draws = np.zeros((n_draws, horizon, self.k))

        for draw in range(n_draws):
            # Draw Σ
            try:
                Sigma_draw = invwishart.rvs(df=self.d_post, scale=self.S_post)
            except:
                Sigma_draw = self.Sigma_post

            Sigma_draw = np.atleast_2d(Sigma_draw)

            # Draw β
            beta_draw = np.zeros_like(self.B_post)
            for i in range(self.k):
                scale_factor = Sigma_draw[i, i] / (self.sigma_i[i]**2)
                V_scaled = self.V_post[:, i] * scale_factor
                beta_draw[:, i] = np.random.normal(
                    self.B_post[:, i],
                    np.sqrt(np.maximum(V_scaled, 1e-10))
                )

            # IRF: Y_t = Φ^t * shock
            # Строим companion matrix
            A = np.zeros((self.k * self.lags, self.k * self.lags))

            # Заполняем из beta_draw (коэффициенты без константы)
            for lag in range(self.lags):
                A[:self.k, lag * self.k:(lag + 1) * self.k] = beta_draw[
                    1 + lag * self.k: 1 + (lag + 1) * self.k, :
                ].T

            # Identity для сдвига лагов
            if self.lags > 1:
                A[self.k:, :self.k * (self.lags - 1)] = np.eye(self.k * (self.lags - 1))

            # Начальный шок
            shock = np.zeros(self.k * self.lags)
            shock[shock_idx] = shock_size

            # Распространение шока
            state = shock.copy()
            for t in range(horizon):
                irf_draws[draw, t, :] = state[:self.k]
                state = A @ state

        # Агрегируем
        result = {
            'horizon': np.arange(horizon),
            'median': {},
            'mean': {},
            'q05': {},
            'q95': {}
        }

        for i, var_name in enumerate(self.var_names):
            result['median'][var_name] = np.median(irf_draws[:, :, i], axis=0)
            result['mean'][var_name] = np.mean(irf_draws[:, :, i], axis=0)
            result['q05'][var_name] = np.percentile(irf_draws[:, :, i], 5, axis=0)
            result['q95'][var_name] = np.percentile(irf_draws[:, :, i], 95, axis=0)

        result['shock_var'] = shock_var
        result['shock_size'] = shock_size

        return result

    def compare_scenarios(
        self,
        horizon: int = 12,
        scenarios: List[str] = None
    ) -> pd.DataFrame:
        """
        Сравнение прогнозов для разных сценариев.

        Args:
            horizon: Горизонт прогноза
            scenarios: Список сценариев (по умолчанию ['base', 'hike', 'cut'])

        Returns:
            DataFrame с прогнозами для каждого сценария
        """
        if scenarios is None:
            scenarios = ['base', 'hike', 'cut']

        results = []
        dates = pd.date_range(
            start=self._last_train_date + pd.DateOffset(months=1),
            periods=horizon,
            freq='MS'
        )

        for scenario in scenarios:
            fc = self.forecast_scenario(horizon, ki_scenario=scenario)

            for t in range(horizon):
                results.append({
                    'date': dates[t],
                    'horizon': t + 1,
                    'scenario': scenario,
                    'total_mom': fc['total'][t],
                    'ki_change': fc['ki_path'][t],
                    'ci_lower': fc['total_q05'][t],
                    'ci_upper': fc['total_q95'][t]
                })

        return pd.DataFrame(results)

    def get_rate_sensitivity(self) -> Dict[str, float]:
        """
        Чувствительность CPI компонентов к изменению ставки.

        Рассчитывается как кумулятивный эффект IRF за 12 месяцев.

        Returns:
            Dict с чувствительностью каждого компонента
        """
        self._check_fitted()

        irf = self.compute_irf(horizon=12, n_draws=500)

        sensitivity = {}
        for var_name in self.var_names[:4]:  # CPI компоненты
            # Кумулятивный эффект за 12 месяцев
            cumulative = np.sum(irf['median'][var_name])
            sensitivity[var_name] = cumulative

        return sensitivity

    def get_model_info(self) -> Dict[str, Any]:
        """Информация о модели."""
        self._check_fitted()

        return {
            'name': self.name,
            'lags': self.lags,
            'n_variables': self.k,
            'var_names': self.var_names,
            'n_observations': self.T,
            'ki_index': self.ki_index,
            'ruonia_index': self.ruonia_index,
            'last_ki': self.last_ki,
            'include_ruonia': self.include_ruonia,
            'include_usd': self.include_usd
        }

    def backtest(
        self,
        df: pd.DataFrame,
        start_date: str = '2019-01-01',
        target_col: str = 'mom'
    ) -> pd.DataFrame:
        """Бэктестирование BVAR Rate."""
        start = pd.Timestamp(start_date)
        last_fact = df.dropna(subset=[target_col]).index.max()
        test_dates = pd.date_range(start=start, end=last_fact, freq='MS')

        results = []

        for target_date in test_dates:
            if target_date not in df.index:
                continue

            cutoff = target_date - pd.DateOffset(months=1)
            train_df = df[df.index <= cutoff].copy()

            if len(train_df.dropna(subset=[target_col])) < self.MIN_TRAIN_SIZE:
                continue

            try:
                model = BVARRateForecaster(
                    lags=self.lags,
                    lambda1=self.lambda1,
                    n_draws=200,
                    include_ruonia=self.include_ruonia,
                    include_usd=self.include_usd
                )
                model.fit(train_df, target_col)
                fc = model.forecast(horizon=1)

                actual = df.loc[target_date, target_col]
                if isinstance(actual, str):
                    actual = float(actual.replace(',', '.'))
                actual = actual - 100 if actual > 50 else actual

                prediction = fc[0]

                results.append({
                    'date': target_date,
                    'actual': actual,
                    'prediction': prediction,
                    'error': actual - prediction
                })
            except Exception:
                continue

        return pd.DataFrame(results)
