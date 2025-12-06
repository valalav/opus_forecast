"""
BVAR модель для прогнозирования инфляции КБР
============================================

Байесовская векторная авторегрессия с Minnesota (Litterman) Prior

Автор: СИРЕНА-КБР
Версия: 1.0
Дата: Декабрь 2025

Методология:
- Normal-Inverse-Wishart сопряжённый prior
- Minnesota prior для регуляризации (shrinkage к random walk)
- Аналитическое апостериорное распределение (без MCMC)

Референсы:
- Litterman (1986) "Forecasting with Bayesian Vector Autoregressions"
- Doan, Litterman, Sims (1984)
- Koop & Korobilis (2010) "Bayesian Multivariate Time Series Methods"
"""

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


class BayesianVAR:
    """
    Bayesian VAR с Minnesota (Litterman) Prior
    
    Parameters
    ----------
    data : DataFrame
        Данные с временными рядами
    var_names : list
        Список названий переменных для VAR
    lags : int
        Число лагов (по умолчанию 1)
    
    Attributes
    ----------
    lambda1 : float
        Overall tightness (по умолчанию 0.2)
    lambda2 : float  
        Cross-variable tightness (по умолчанию 0.5)
    lambda3 : float
        Lag decay (по умолчанию 1.0)
    lambda4 : float
        Intercept prior variance (по умолчанию 100)
    
    Examples
    --------
    >>> model = BayesianVAR(data, ['ipc', 'prod', 'nonprod', 'serv'], lags=1)
    >>> model.fit(lambda1=0.2)
    >>> forecasts = model.forecast(h=12, n_draws=5000)
    """
    
    def __init__(self, data, var_names, lags=1):
        """Инициализация BVAR модели"""
        self.raw_data = data[var_names].values
        self.var_names = var_names
        self.p = lags
        self.k = len(var_names)
        
        # Подготовка данных
        self._prepare_data()
        
        # Гиперпараметры Minnesota prior (значения по умолчанию)
        self.lambda1 = 0.2   # overall tightness
        self.lambda2 = 0.5   # cross-variable tightness  
        self.lambda3 = 1.0   # lag decay
        self.lambda4 = 100   # intercept prior variance
        
        # Результаты оценки
        self.beta_post = None
        self.V_post = None
        self.Sigma_post = None
        self.fitted = False
        
    def _prepare_data(self):
        """Подготовка матриц Y и X для VAR"""
        T_total = len(self.raw_data)
        T = T_total - self.p
        
        # Y: зависимая переменная (T x k)
        Y = self.raw_data[self.p:, :]
        
        # X: регрессоры (T x (1 + k*p)) — константа + лаги
        X = np.ones((T, 1 + self.k * self.p))
        
        for t in range(T):
            for lag in range(1, self.p + 1):
                start_col = 1 + (lag - 1) * self.k
                X[t, start_col:start_col + self.k] = self.raw_data[self.p + t - lag, :]
        
        self.Y = Y
        self.X = X
        self.T = T
        self.n_params = X.shape[1]
        
    def _minnesota_prior(self):
        """
        Формирование Minnesota prior
        
        Minnesota prior (Litterman, 1986):
        - Prior mean: random walk для собственных лагов, 0 для остальных
        - Prior variance: убывает с номером лага, меньше для кросс-переменных
        
        Returns
        -------
        beta0 : ndarray
            Prior mean для коэффициентов (n_params x k)
        V0 : ndarray
            Prior variance для коэффициентов (n_params x k)
        sigma_i : ndarray
            Оценки дисперсий для масштабирования (k,)
        """
        n = self.n_params
        k = self.k
        
        # Prior mean: random walk для первого лага собственной переменной
        beta0 = np.zeros((n, k))
        for i in range(k):
            # Для стационарных рядов (MoM) лучше 0 или AR(1) < 1
            # Если данные growth rates, то RW (1.0) не подходит. 
            # Лучше 0.0 для MoM deviation или AR(1) estimate
            # Здесь используем 0.8 как компромисс для инфляции (persistence)
            beta0[1 + i, i] = 0.8 
            
        # Prior variance
        V0 = np.zeros((n, k))
        
        # Оценка дисперсий из AR(1) для масштабирования
        sigma_i = np.array([
            max(np.std(np.diff(self.raw_data[:, i])), 1e-6) 
            for i in range(k)
        ])
        
        # Константа: большая дисперсия (uninformative)
        V0[0, :] = self.lambda4 * sigma_i**2
        
        # Коэффициенты при лагах
        for lag in range(1, self.p + 1):
            for j in range(k):  # переменная в X
                row = 1 + (lag - 1) * k + j
                for i in range(k):  # уравнение
                    if i == j:  # own lag
                        V0[row, i] = (self.lambda1 / lag**self.lambda3)**2
                    else:  # cross lag
                        V0[row, i] = (
                            (self.lambda1 * self.lambda2 / lag**self.lambda3)**2 
                            * (sigma_i[i] / sigma_i[j])**2
                        )
        
        return beta0, V0, sigma_i
    
    def fit(self, lambda1=None, lambda2=None, lambda3=None, lambda4=None):
        """
        Оценка BVAR с Minnesota prior
        
        Parameters
        ----------
        lambda1 : float, optional
            Overall tightness
        lambda2 : float, optional
            Cross-variable tightness
        lambda3 : float, optional
            Lag decay
        lambda4 : float, optional
            Intercept prior variance
            
        Returns
        -------
        self
        """
        # Обновление гиперпараметров
        if lambda1 is not None: self.lambda1 = lambda1
        if lambda2 is not None: self.lambda2 = lambda2
        if lambda3 is not None: self.lambda3 = lambda3
        if lambda4 is not None: self.lambda4 = lambda4
            
        # Получаем prior
        beta0, V0, sigma_i = self._minnesota_prior()
        
        # OLS оценка для сравнения
        XtX = self.X.T @ self.X
        XtY = self.X.T @ self.Y
        # Ridge correction for OLS stability
        self.beta_ols = np.linalg.solve(XtX + np.eye(self.n_params)*1e-4, XtY)
        
        # Байесовская оценка (equation-by-equation)
        beta_post = np.zeros_like(beta0)
        V_post = np.zeros_like(V0)
        
        for i in range(self.k):
            # Prior precision для уравнения i
            V0_inv = np.diag(1.0 / (V0[:, i] + 1e-10))
            
            # Posterior precision = prior precision + data precision
            V_post_inv = V0_inv + XtX / (sigma_i[i]**2)
            V_post_i = np.linalg.inv(V_post_inv)
            
            # Posterior mean
            beta_post[:, i] = V_post_i @ (
                V0_inv @ beta0[:, i] + XtY[:, i] / sigma_i[i]**2
            )
            V_post[:, i] = np.diag(V_post_i)
        
        # Ковариация остатков
        resid = self.Y - self.X @ beta_post
        Sigma_post = resid.T @ resid / max(1, self.T - self.n_params)
        
        # Сохраняем результаты
        self.beta_post = beta_post
        self.V_post = V_post
        self.Sigma_post = Sigma_post
        self.sigma_i = sigma_i
        self.resid = resid
        self.fitted = True
        
        return self
    
    def forecast(self, h=12, n_draws=1000):
        """
        Прогнозирование с учётом неопределённости параметров
        """
        if not self.fitted:
            raise ValueError("Модель не оценена. Вызовите fit() сначала.")
            
        forecasts = np.zeros((n_draws, h, self.k))
        
        # Последние p наблюдений для инициализации
        Y_history = self.raw_data[-self.p:, :].copy()
        
        for draw in range(n_draws):
            # Draw коэффициентов из апостериорного распределения
            beta_draw = np.zeros_like(self.beta_post)
            for i in range(self.k):
                beta_draw[:, i] = np.random.normal(
                    self.beta_post[:, i], 
                    np.sqrt(np.maximum(self.V_post[:, i], 1e-10))
                )
            
            # Cholesky decomposition для генерации шоков
            try:
                L = np.linalg.cholesky(self.Sigma_post + np.eye(self.k) * 1e-6)
            except:
                L = np.eye(self.k) * np.sqrt(np.diag(self.Sigma_post).mean())
            
            # Рекурсивный прогноз
            Y_curr = Y_history.copy()
            
            for t in range(h):
                # Формируем X_t = [1, Y_{t-1}, Y_{t-2}, ...]
                X_t = np.ones(1 + self.k * self.p)
                for lag in range(1, self.p + 1):
                    idx = -lag
                    X_t[1 + (lag-1)*self.k : 1 + lag*self.k] = Y_curr[idx, :]
                
                # Прогноз: Y_t = X_t @ beta + shock
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
            'draws': forecasts
        }
    
    def irf(self, shock_var=0, h=12, orthogonalized=True):
        """Импульсные функции отклика (IRF)"""
        if not self.fitted:
            raise ValueError("Модель не оценена. Вызовите fit() сначала.")
            
        if orthogonalized:
            try:
                P = np.linalg.cholesky(self.Sigma_post)
            except:
                P = np.eye(self.k)
        else:
            P = np.eye(self.k)
        
        # Матрица коэффициентов A (для VAR(1))
        # beta: [const, Y_t-1, Y_t-2 ...]
        # A should be (k x k*p)
        # For VAR(1): A is (k x k)
        
        # Simplified for VAR(1)
        if self.p == 1:
            A = self.beta_post[1:self.k+1, :].T  # (k x k)
            
            irf = np.zeros((h, self.k))
            shock = np.zeros(self.k)
            shock[shock_var] = 1.0
            
            response = P @ shock
            irf[0, :] = response
            
            for t in range(1, h):
                response = A @ response
                irf[t, :] = response
            return irf
        else:
            # TODO: General IRF for VAR(p)
            return np.zeros((h, self.k))

if __name__ == "__main__":
    print("BayesianVAR class ready.")