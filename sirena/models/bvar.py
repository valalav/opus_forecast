"""
BVAR модель для прогнозирования инфляции КБР
=============================================

Полная Байесовская VAR с Minnesota Prior и Inverse-Wishart prior на Σ.

Улучшения v2.0:
- Inverse-Wishart prior на ковариационную матрицу
- Empirical Bayes для автоматического выбора гиперпараметров
- Поддержка нескольких лагов с автовыбором по BIC

Вес в ансамбле: 20%
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
from scipy.stats import invwishart
from scipy.optimize import minimize_scalar

from .base import BaseForecaster
from .registry import ModelRegistry


@ModelRegistry.register("bvar")
class BVARForecaster(BaseForecaster):
    """
    Полная Байесовская VAR с Minnesota Prior и Inverse-Wishart на Σ.

    Гиперпараметры Minnesota Prior:
    - lambda1: Overall tightness (по умолчанию 0.2, или auto)
    - lambda2: Cross-variable tightness (по умолчанию 0.5)
    - lambda3: Lag decay (по умолчанию 1.0)
    - lambda4: Intercept prior variance (по умолчанию 100)

    Гиперпараметры Inverse-Wishart:
    - sigma_prior_df: Степени свободы (по умолчанию k+2)
    - sigma_prior_scale: Масштаб (по умолчанию 1.0)

    Автовыбор:
    - auto_lambda: Empirical Bayes для lambda1
    - auto_lags: Выбор лагов по BIC
    """

    name = "bvar"
    MIN_TRAIN_SIZE = 24

    def __init__(
        self,
        lags: int = 1,
        lambda1: float = 0.2,
        lambda2: float = 0.5,
        lambda3: float = 1.0,
        lambda4: float = 100,
        n_draws: int = 1000,
        # Inverse-Wishart параметры
        sigma_prior_df: Optional[int] = None,
        sigma_prior_scale: float = 1.0,
        # Автоматический выбор
        auto_lambda: bool = False,
        auto_lags: bool = False,
        max_lags: int = 4,
        # Gibbs sampler
        use_gibbs: bool = False,
        gibbs_iter: int = 1000,
        gibbs_burnin: int = 200,
        var_names: List[str] = None,
        **kwargs
    ):
        """
        Инициализация BVAR.

        Args:
            lags: Количество лагов (игнорируется если auto_lags=True)
            lambda1: Overall tightness (игнорируется если auto_lambda=True)
            lambda2: Cross-variable tightness
            lambda3: Lag decay
            lambda4: Intercept prior variance
            n_draws: Количество draws для прогноза
            sigma_prior_df: Степени свободы IW prior (по умолчанию k+2)
            sigma_prior_scale: Масштаб IW prior
            auto_lambda: Автоматический выбор lambda1 через Empirical Bayes
            auto_lags: Автоматический выбор лагов по BIC
            max_lags: Максимальное число лагов для auto_lags
            use_gibbs: Использовать Gibbs sampler для полного joint posterior
            gibbs_iter: Количество итераций Gibbs после burn-in
            gibbs_burnin: Burn-in период для Gibbs
            var_names: Список переменных (если None, используются дефолтные)
        """
        super().__init__(**kwargs)
        self.lags = lags
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.lambda3 = lambda3
        self.lambda4 = lambda4
        self.n_draws = n_draws

        # Inverse-Wishart
        self.sigma_prior_df = sigma_prior_df
        self.sigma_prior_scale = sigma_prior_scale

        # Auto-tuning
        self.auto_lambda = auto_lambda
        self.auto_lags = auto_lags
        self.max_lags = max_lags

        # Gibbs
        self.use_gibbs = use_gibbs
        self.gibbs_iter = gibbs_iter
        self.gibbs_burnin = gibbs_burnin

        # Внутренние переменные
        self.var_names: List[str] = var_names if var_names is not None else []
        self.k = 0
        self.n_params = 0
        self.raw_data = None
        self.Y = None
        self.X = None
        self.T = 0

        # Posterior параметры
        self.B_post = None
        self.V_post = None
        self.Sigma_post = None
        self.sigma_i = None

        # Inverse-Wishart posterior
        self.S_post = None
        self.d_post = None

        # Gibbs samples
        self.B_samples = None
        self.Sigma_samples = None

        # Оптимальные параметры (после auto-tuning)
        self.optimal_lambda1 = None
        self.optimal_lags = None

    def _prepare_var_data(self, df: pd.DataFrame, lags: int = None) -> None:
        """Подготовка данных для VAR."""
        if lags is None:
            lags = self.lags

        # Выбираем переменные
        if not self.var_names:
            cols = ['Все товары и услуги']
            for col in ['Продовольственные товары', 'Непродовольственные товары', 'Услуги']:
                if col in df.columns:
                    cols.append(col)
            self.var_names = cols
        else:
            cols = self.var_names

        self.k = len(cols)

        # Конвертируем данные
        data = df[cols].dropna().values.astype(np.float64)

        # Если данные в индексном формате (около 100)
        if np.mean(data[:, 0]) > 50:
            data = data - 100

        self.raw_data = data
        T_total = len(data)
        T = T_total - lags

        # Y: зависимая переменная
        self.Y = data[lags:, :]

        # X: константа + лаги
        self.X = np.ones((T, 1 + self.k * lags))
        for t in range(T):
            for lag in range(1, lags + 1):
                start_col = 1 + (lag - 1) * self.k
                self.X[t, start_col:start_col + self.k] = data[lags + t - lag, :]

        self.T = T
        self.n_params = self.X.shape[1]
        self.lags = lags

    def _minnesota_prior(self, lambda1: float = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Minnesota prior (Litterman, 1986).

        Returns:
            beta0: Prior mean (n_params × k)
            V0: Prior variance (n_params × k)
            sigma_i: OLS standard deviations
        """
        if lambda1 is None:
            lambda1 = self.lambda1

        n = self.n_params
        k = self.k

        beta0 = np.zeros((n, k))
        for i in range(k):
            if 1 + i < n:
                beta0[1 + i, i] = 0.8  # AR(1) persistence

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
                        V0[row, i] = (lambda1 / lag**self.lambda3)**2
                    else:
                        V0[row, i] = (
                            (lambda1 * self.lambda2 / lag**self.lambda3)**2
                            * (sigma_i[i] / sigma_i[j])**2
                        )

        return beta0, V0, sigma_i

    def _inverse_wishart_prior(self) -> Tuple[np.ndarray, int]:
        """
        Inverse-Wishart prior на ковариационную матрицу.

        Теория:
            Σ ~ IW(S0, d0)
            E[Σ] = S0 / (d0 - k - 1)  для d0 > k + 1

        Returns:
            S0: Scale matrix (k × k)
            d0: Degrees of freedom
        """
        k = self.k

        # Степени свободы: минимально информативный prior
        d0 = self.sigma_prior_df if self.sigma_prior_df else k + 2

        # Scale matrix: диагональная, основана на OLS дисперсиях
        S0 = np.diag(self.sigma_i**2) * self.sigma_prior_scale * max(1, d0 - k - 1)

        return S0, d0

    def _compute_marginal_likelihood(self, lambda1: float) -> float:
        """
        Вычисление (аппроксимации) log marginal likelihood для Empirical Bayes.

        Используем BIC-style аппроксимацию:
            log p(Y|λ) ≈ log p(Y|β_post, Σ_post) - (k_eff/2) * log(T)

        Args:
            lambda1: Overall tightness parameter

        Returns:
            Negative log marginal likelihood (для минимизации)
        """
        beta0, V0, sigma_i = self._minnesota_prior(lambda1)

        XtX = self.X.T @ self.X
        XtY = self.X.T @ self.Y

        beta_post = np.zeros_like(beta0)

        for i in range(self.k):
            V0_inv = np.diag(1.0 / (V0[:, i] + 1e-10))
            V_post_inv = V0_inv + XtX / (sigma_i[i]**2)
            V_post_i = np.linalg.inv(V_post_inv)

            beta_post[:, i] = V_post_i @ (
                V0_inv @ beta0[:, i] + XtY[:, i] / sigma_i[i]**2
            )

        resid = self.Y - self.X @ beta_post
        SSR = np.sum(resid**2)

        # Effective number of parameters (shrinkage reduces this)
        # Приближение: k_eff = trace(X @ V_post @ X.T @ Σ^{-1})
        k_eff = self.n_params * self.k * (1 - lambda1)  # упрощенная формула

        # BIC-style criterion
        log_lik = -0.5 * self.T * self.k * np.log(2 * np.pi) - 0.5 * SSR
        bic = -2 * log_lik + k_eff * np.log(self.T)

        return bic

    def _select_optimal_lambda(self) -> float:
        """
        Empirical Bayes: выбор оптимального lambda1.

        Returns:
            Оптимальное значение lambda1
        """
        result = minimize_scalar(
            self._compute_marginal_likelihood,
            bounds=(0.01, 1.0),
            method='bounded'
        )
        return result.x

    def _compute_bic(self, lags: int) -> float:
        """
        Вычисление BIC для заданного числа лагов.

        Args:
            lags: Число лагов

        Returns:
            BIC value
        """
        # Временно сохраняем текущие данные
        old_lags = self.lags
        old_Y = self.Y
        old_X = self.X
        old_T = self.T
        old_n_params = self.n_params

        # Подготавливаем данные с новым числом лагов
        self._prepare_var_data(
            pd.DataFrame(self.raw_data, columns=self.var_names),
            lags=lags
        )

        if self.T < self.MIN_TRAIN_SIZE:
            # Восстанавливаем
            self.lags = old_lags
            self.Y = old_Y
            self.X = old_X
            self.T = old_T
            self.n_params = old_n_params
            return np.inf

        beta0, V0, sigma_i = self._minnesota_prior()

        XtX = self.X.T @ self.X
        XtY = self.X.T @ self.Y

        beta_post = np.zeros_like(beta0)

        for i in range(self.k):
            V0_inv = np.diag(1.0 / (V0[:, i] + 1e-10))
            V_post_inv = V0_inv + XtX / (sigma_i[i]**2)
            V_post_i = np.linalg.inv(V_post_inv)

            beta_post[:, i] = V_post_i @ (
                V0_inv @ beta0[:, i] + XtY[:, i] / sigma_i[i]**2
            )

        resid = self.Y - self.X @ beta_post
        SSR = np.sum(resid**2)

        n_params_total = self.n_params * self.k
        bic = self.T * np.log(SSR / self.T) + n_params_total * np.log(self.T)

        # Восстанавливаем
        self.lags = old_lags
        self.Y = old_Y
        self.X = old_X
        self.T = old_T
        self.n_params = old_n_params

        return bic

    def _select_optimal_lags(self, df: pd.DataFrame) -> int:
        """
        Выбор оптимального числа лагов по BIC.

        Args:
            df: DataFrame с данными

        Returns:
            Оптимальное число лагов
        """
        # Инициализируем данные для вычисления BIC
        self._prepare_var_data(df, lags=1)

        bic_values = {}
        for lags in range(1, self.max_lags + 1):
            bic_values[lags] = self._compute_bic(lags)

        optimal_lags = min(bic_values, key=bic_values.get)
        return optimal_lags

    def _gibbs_sampler(self) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Gibbs sampler для joint posterior P(B, Σ | Y).

        Алгоритм:
            1. Draw B | Σ, Y ~ N(...)
            2. Draw Σ | B, Y ~ IW(...)
            3. Повторить

        Returns:
            B_samples: List of coefficient matrices
            Sigma_samples: List of covariance matrices
        """
        B_samples = []
        Sigma_samples = []

        # Инициализация
        B_curr = self.B_post.copy()
        Sigma_curr = self.Sigma_post.copy()

        # Prior parameters
        beta0, V0, _ = self._minnesota_prior()
        S0, d0 = self._inverse_wishart_prior()

        XtX = self.X.T @ self.X
        XtY = self.X.T @ self.Y

        for iteration in range(self.gibbs_iter + self.gibbs_burnin):
            # Step 1: Draw B | Σ, Y
            for i in range(self.k):
                V0_inv = np.diag(1.0 / (V0[:, i] + 1e-10))
                V_post_inv = V0_inv + XtX / max(Sigma_curr[i, i], 1e-10)
                V_post_i = np.linalg.inv(V_post_inv)

                beta_post_i = V_post_i @ (
                    V0_inv @ beta0[:, i] + XtY[:, i] / max(Sigma_curr[i, i], 1e-10)
                )

                try:
                    B_curr[:, i] = np.random.multivariate_normal(
                        beta_post_i, V_post_i
                    )
                except:
                    B_curr[:, i] = np.random.normal(
                        beta_post_i, np.sqrt(np.maximum(np.diag(V_post_i), 1e-10))
                    )

            # Step 2: Draw Σ | B, Y
            resid = self.Y - self.X @ B_curr
            S_post = S0 + resid.T @ resid
            d_post = d0 + self.T

            try:
                Sigma_curr = invwishart.rvs(df=d_post, scale=S_post)
            except:
                Sigma_curr = S_post / max(1, d_post - self.k - 1)

            # Сохраняем после burn-in
            if iteration >= self.gibbs_burnin:
                B_samples.append(B_curr.copy())
                Sigma_samples.append(Sigma_curr.copy())

        return B_samples, Sigma_samples

    def fit(self, df: pd.DataFrame, target_col: str = 'Все товары и услуги') -> 'BVARForecaster':
        """
        Обучение BVAR с полным Bayesian inference.

        Args:
            df: DataFrame с данными
            target_col: Целевая колонка

        Returns:
            self
        """
        self._validate_data(df, target_col)

        # Автовыбор лагов
        if self.auto_lags:
            self._prepare_var_data(df, lags=1)  # Инициализация для auto_lags
            self.optimal_lags = self._select_optimal_lags(df)
            self.lags = self.optimal_lags

        # Подготовка данных
        self._prepare_var_data(df)

        # Автовыбор lambda1
        if self.auto_lambda:
            self.optimal_lambda1 = self._select_optimal_lambda()
            self.lambda1 = self.optimal_lambda1

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

        # Posterior mean для Σ
        self.Sigma_post = self.S_post / max(1, self.d_post - self.k - 1)

        self.B_post = beta_post
        self.V_post = V_post

        # Gibbs sampler (опционально)
        if self.use_gibbs:
            self.B_samples, self.Sigma_samples = self._gibbs_sampler()

        self._is_fitted = True
        self._last_train_date = df.index.max()

        return self

    def forecast(self, horizon: int = 12) -> np.ndarray:
        """
        Прогноз BVAR с sampling из IW posterior.

        Args:
            horizon: Горизонт прогноза

        Returns:
            numpy array с прогнозами (первая переменная = CPI)
        """
        self._check_fitted()

        # Если есть Gibbs samples, используем их
        if self.use_gibbs and self.B_samples is not None:
            return self._forecast_gibbs(horizon)

        forecasts = np.zeros((self.n_draws, horizon, self.k))
        Y_history = self.raw_data[-self.lags:, :].copy()

        for draw in range(self.n_draws):
            # Draw Σ из IW posterior
            try:
                Sigma_draw = invwishart.rvs(df=self.d_post, scale=self.S_post)
            except:
                Sigma_draw = self.Sigma_post

            # Handle scalar case (k=1)
            Sigma_draw = np.atleast_2d(Sigma_draw)

            # Draw коэффициентов (условно на Σ)
            beta_draw = np.zeros_like(self.B_post)
            for i in range(self.k):
                scale_factor = Sigma_draw[i, i] / (self.sigma_i[i]**2)
                V_scaled = self.V_post[:, i] * scale_factor

                beta_draw[:, i] = np.random.normal(
                    self.B_post[:, i],
                    np.sqrt(np.maximum(V_scaled, 1e-10))
                )

            # Cholesky для structural shocks
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

                forecasts[draw, t, :] = Y_new
                Y_curr = np.vstack([Y_curr, Y_new])

        return np.median(forecasts[:, :, 0], axis=0)

    def _forecast_gibbs(self, horizon: int) -> np.ndarray:
        """
        Прогноз с использованием Gibbs samples.

        Args:
            horizon: Горизонт прогноза

        Returns:
            numpy array с прогнозами
        """
        n_samples = len(self.B_samples)
        forecasts = np.zeros((n_samples, horizon, self.k))
        Y_history = self.raw_data[-self.lags:, :].copy()

        for s, (B_s, Sigma_s) in enumerate(zip(self.B_samples, self.Sigma_samples)):
            try:
                L = np.linalg.cholesky(Sigma_s + np.eye(self.k) * 1e-6)
            except:
                L = np.eye(self.k) * np.sqrt(np.diag(Sigma_s).mean())

            Y_curr = Y_history.copy()

            for t in range(horizon):
                X_t = np.ones(1 + self.k * self.lags)
                for lag in range(1, self.lags + 1):
                    X_t[1 + (lag - 1) * self.k: 1 + lag * self.k] = Y_curr[-lag, :]

                Y_mean = X_t @ B_s
                shock = L @ np.random.randn(self.k)
                Y_new = Y_mean + shock

                forecasts[s, t, :] = Y_new
                Y_curr = np.vstack([Y_curr, Y_new])

        return np.median(forecasts[:, :, 0], axis=0)

    def forecast_full(self, horizon: int = 12) -> Dict[str, np.ndarray]:
        """
        Полный прогноз со всей статистикой.

        Returns:
            Dict с mean, median, std, quantiles
        """
        self._check_fitted()

        # Если есть Gibbs samples, используем их
        if self.use_gibbs and self.B_samples is not None:
            n_samples = len(self.B_samples)
            forecasts = np.zeros((n_samples, horizon, self.k))
            Y_history = self.raw_data[-self.lags:, :].copy()

            for s, (B_s, Sigma_s) in enumerate(zip(self.B_samples, self.Sigma_samples)):
                try:
                    L = np.linalg.cholesky(Sigma_s + np.eye(self.k) * 1e-6)
                except:
                    L = np.eye(self.k) * np.sqrt(np.diag(Sigma_s).mean())

                Y_curr = Y_history.copy()

                for t in range(horizon):
                    X_t = np.ones(1 + self.k * self.lags)
                    for lag in range(1, self.lags + 1):
                        X_t[1 + (lag - 1) * self.k: 1 + lag * self.k] = Y_curr[-lag, :]

                    Y_mean = X_t @ B_s
                    shock = L @ np.random.randn(self.k)
                    Y_new = Y_mean + shock

                    forecasts[s, t, :] = Y_new
                    Y_curr = np.vstack([Y_curr, Y_new])
        else:
            forecasts = np.zeros((self.n_draws, horizon, self.k))
            Y_history = self.raw_data[-self.lags:, :].copy()

            for draw in range(self.n_draws):
                try:
                    Sigma_draw = invwishart.rvs(df=self.d_post, scale=self.S_post)
                except:
                    Sigma_draw = self.Sigma_post

                # Handle scalar case (k=1)
                Sigma_draw = np.atleast_2d(Sigma_draw)

                beta_draw = np.zeros_like(self.B_post)
                for i in range(self.k):
                    scale_factor = Sigma_draw[i, i] / (self.sigma_i[i]**2)
                    V_scaled = self.V_post[:, i] * scale_factor

                    beta_draw[:, i] = np.random.normal(
                        self.B_post[:, i],
                        np.sqrt(np.maximum(V_scaled, 1e-10))
                    )

                try:
                    L = np.linalg.cholesky(Sigma_draw + np.eye(self.k) * 1e-6)
                except np.linalg.LinAlgError:
                    L = np.eye(self.k) * np.sqrt(np.diag(Sigma_draw).mean())

                Y_curr = Y_history.copy()

                for t in range(horizon):
                    X_t = np.ones(1 + self.k * self.lags)
                    for lag in range(1, self.lags + 1):
                        X_t[1 + (lag - 1) * self.k: 1 + lag * self.k] = Y_curr[-lag, :]

                    Y_mean = X_t @ beta_draw
                    shock = L @ np.random.randn(self.k)
                    Y_new = Y_mean + shock

                    forecasts[draw, t, :] = Y_new
                    Y_curr = np.vstack([Y_curr, Y_new])

        return {
            'mean': np.mean(forecasts, axis=0),
            'median': np.median(forecasts, axis=0),
            'std': np.std(forecasts, axis=0),
            'q05': np.percentile(forecasts, 5, axis=0),
            'q16': np.percentile(forecasts, 16, axis=0),
            'q84': np.percentile(forecasts, 84, axis=0),
            'q95': np.percentile(forecasts, 95, axis=0),
            'draws': forecasts,
            'var_names': self.var_names
        }

    def backtest(
        self,
        df: pd.DataFrame,
        start_date: str = '2019-01-01',
        target_col: str = 'Все товары и услуги'
    ) -> pd.DataFrame:
        """Бэктестирование BVAR."""
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
                model = BVARForecaster(
                    lags=self.lags,
                    lambda1=self.lambda1,
                    lambda2=self.lambda2,
                    lambda3=self.lambda3,
                    sigma_prior_df=self.sigma_prior_df,
                    sigma_prior_scale=self.sigma_prior_scale,
                    auto_lambda=self.auto_lambda,
                    auto_lags=False,  # Не пересчитываем лаги в бэктесте
                    n_draws=500,
                    use_gibbs=False  # Быстрее для бэктеста
                )
                model.fit(train_df, target_col)
                fc = model.forecast(horizon=1)

                actual_raw = df.loc[target_date, target_col]
                # Конвертируем в float (может быть строкой)
                if isinstance(actual_raw, str):
                    actual_raw = float(actual_raw.replace(',', '.'))
                else:
                    actual_raw = float(actual_raw)
                actual = actual_raw - 100 if actual_raw > 50 else actual_raw

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

    def get_coefficients(self) -> pd.DataFrame:
        """Коэффициенты модели."""
        self._check_fitted()

        rows = ['const'] + [
            f'{var}_lag{lag}'
            for lag in range(1, self.lags + 1)
            for var in self.var_names
        ]

        return pd.DataFrame(
            self.B_post,
            index=rows,
            columns=self.var_names
        )

    def get_model_info(self) -> Dict[str, Any]:
        """
        Информация о модели после обучения.

        Returns:
            Dict с параметрами и диагностикой
        """
        self._check_fitted()

        info = {
            'lags': self.lags,
            'lambda1': self.lambda1,
            'lambda2': self.lambda2,
            'lambda3': self.lambda3,
            'lambda4': self.lambda4,
            'n_variables': self.k,
            'n_observations': self.T,
            'n_params_per_eq': self.n_params,
            'total_params': self.n_params * self.k,
            'var_names': self.var_names,
            'sigma_prior_df': self.sigma_prior_df or (self.k + 2),
            'sigma_prior_scale': self.sigma_prior_scale,
            'd_post': self.d_post,
            'use_gibbs': self.use_gibbs,
        }

        if self.auto_lambda:
            info['optimal_lambda1'] = self.optimal_lambda1

        if self.auto_lags:
            info['optimal_lags'] = self.optimal_lags

        return info


# Алиас для обратной совместимости
BayesianVAR = BVARForecaster
