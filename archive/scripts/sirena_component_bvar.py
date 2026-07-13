import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sirena.models.bvar import BayesianVAR

class SirenaComponentBVAR:
    def __init__(self):
        self.models = {}
        self.weights = {}
        self.components = ['Prod', 'Nonprod', 'Serv']
        self.exog_map = {
            'Prod': ['Prod', 'usd_nom_i'],
            'Nonprod': ['Nonprod', 'usd_nom_i'],
            'Serv': ['Serv']
        }
        # Optimized parameters (Opus Grid Search)
        self.best_params = {
            'Prod': {'lags': 6, 'lambda1': 0.5, 'lambda2': 1.0},
            'Nonprod': {'lags': 2, 'lambda1': 0.5, 'lambda2': 1.0},
            'Serv': {'lags': 2, 'lambda1': 1.0, 'lambda2': 0.5}
        }
    
    def fit(self, df):
        """
        Обучает 3 отдельные модели с оптимизированными параметрами.
        """
        # 1. Оценка весов
        recent_df = df.tail(60).dropna(subset=self.components + ['mom'])
        
        if len(recent_df) < 12:
            self.weights = {'Prod': 0.4, 'Nonprod': 0.35, 'Serv': 0.25}
        else:
            X = recent_df[self.components]
            y = recent_df['mom']
            lr = LinearRegression(fit_intercept=False, positive=True)
            lr.fit(X, y)
            self.weights = dict(zip(self.components, lr.coef_))
            
            # Normalize
            total = sum(self.weights.values())
            if total > 0:
                for k in self.weights:
                    self.weights[k] /= total
            else:
                self.weights = {'Prod': 0.4, 'Nonprod': 0.35, 'Serv': 0.25}
        
        # 2. Обучение моделей
        for comp in self.components:
            cols = self.exog_map[comp]
            valid_cols = [c for c in cols if c in df.columns]
            if not valid_cols:
                raise ValueError(f"Нет данных для компонента {comp}")
                
            train_data = df[valid_cols].copy()
            for c in train_data.columns:
                if train_data[c].mean() > 50:
                    train_data[c] = train_data[c] - 100
            
            # Используем оптимизированные параметры
            params = self.best_params.get(comp, {'lags': 4, 'lambda1': 0.2, 'lambda2': 0.5})
            
            model = BayesianVAR(
                lags=params['lags'], 
                lambda1=params['lambda1'],
                lambda2=params.get('lambda2', 0.5),
                var_names=valid_cols
            )
            model.fit(train_data, target_col=valid_cols[0])
            self.models[comp] = model
            
        return self

    def predict(self, horizon=12):
        """
        Прогноз на горизонт.
        """
        results = {}
        dates = None
        
        for comp, model in self.models.items():
            fc_dict = model.forecast_full(horizon=horizon)
            median = fc_dict['median'] # shape (h, k)
            
            if median.ndim == 2:
                vals = median[:, 0]
            else:
                vals = median
                
            vals = np.atleast_1d(vals)
            results[comp] = vals + 100
            
            if dates is None:
                last_date = model._last_train_date
                dates = pd.date_range(start=last_date + pd.DateOffset(months=1), periods=horizon, freq='MS')
        
        results['Date'] = dates
        
        # Собираем CPI
        cpi_pred = np.zeros(horizon)
        for comp in self.components:
            cpi_pred += (results[comp] - 100) * self.weights[comp]
            
        results['CPI'] = cpi_pred + 100
        
        return pd.DataFrame(results)
    
    def get_weights(self):
        return self.weights
