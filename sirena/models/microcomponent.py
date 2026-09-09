#!/usr/bin/env python3
"""Dated bottom-up RAW CPI forecasts with explicit parent treatment of missing mass."""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import warnings

warnings.filterwarnings("ignore")

from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import VotingRegressor
from sklearn.preprocessing import StandardScaler


class MicrocomponentForecaster:
    """Dated leaf Ridge/Voting models with explicit weighted parent forecasts.

    The parent policy is selected on the development h=1 window. Observed
    coverage and modeled coverage are distinct. No official data are imputed.
    """

    name = "microcomponent"

    # Volatile items (vegetables) - need extended features
    VOLATILE_ITEMS = {
        435,
        382,
        506,
        279,
        305,
        333,
        342,
        755,
    }  # Огурцы, морковь, помидоры и др.

    # Historical seasonal adjustment (2019-2024, excluding 2022)
    # Calculated from aggregate microcomponent data
    SEASONAL_ADJ = {
        1: +0.30,  # Январь - высокая инфляция после НГ
        2: +0.13,  # Февраль
        3: +0.07,  # Март
        4: +0.14,  # Апрель
        5: -0.31,  # Май - начало сезона низкой инфляции
        6: -0.55,  # Июнь - низкая (свежие овощи)
        7: -0.20,  # Июль - тарифы ЖКХ, но овощи дешевеют
        8: -0.59,  # Август - минимум инфляции
        9: +0.36,  # Сентябрь - рост (конец сезона)
        10: +0.19,  # Октябрь
        11: +0.43,  # Ноябрь - высокая инфляция
        12: +0.12,  # Декабрь
    }

    def __init__(
        self,
        horizon=1,
        train_start="2016-01-01",
        random_state=42,
        top_n=100,
        use_extended_for_volatile=True,
        use_seasonal_adj=False,
        fallback_policy="parent_seasonal",
        data_dir=None,
    ):
        """
        Parameters
        ----------
        horizon : int
            Forecast horizon (1, 2, or 12)
        train_start : str
            Start date for training data
        random_state : int
            Random seed
        top_n : int
            Number of top microcomponents by weight to use individual Ridge models
        use_extended_for_volatile : bool
            Use extended features for volatile items
        use_seasonal_adj : bool
            Apply legacy manual adjustment (default False; not used in production)
        """
        self.horizon = horizon
        self.train_start = train_start
        self.random_state = random_state
        self.top_n = top_n
        self.use_extended_for_volatile = use_extended_for_volatile
        self.use_seasonal_adj = use_seasonal_adj

        if fallback_policy not in {"parent_ridge", "parent_seasonal"}:
            raise ValueError("Unknown parent forecast policy")
        self.fallback_policy = fallback_policy
        self.data_dir = Path(data_dir) if data_dir else None
        self._is_fitted = False
        self.micro_models = {}  # {item_code: {'model': model, 'scaler': scaler, ...}}
        self.weights = {}
        self.top_items = set()

    def _create_features(self, series, extended=False):
        """Create features for ML models."""
        df = pd.DataFrame({"y": series})

        # Basic lags
        for lag in [1, 2, 3, 6, 12]:
            df[f"L{lag}"] = df["y"].shift(lag)

        # Momentum
        df["D1"] = df["y"].diff(1)
        df["MA3"] = df["y"].rolling(3).mean()

        # Seasonality
        df["month_sin"] = np.sin(2 * np.pi * df.index.month / 12)
        df["month_cos"] = np.cos(2 * np.pi * df.index.month / 12)

        if extended:
            # Extended features for volatile items
            df["MA6"] = df["y"].rolling(6).mean()
            df["STD3"] = df["y"].rolling(3).std()
            df["STD6"] = df["y"].rolling(6).std()
            df["MAX3"] = df["y"].rolling(3).max()
            df["MIN3"] = df["y"].rolling(3).min()
            df["RANGE3"] = df["MAX3"] - df["MIN3"]

        return df

    def _fit_ridge(self, series, item_code):
        """Fit Ridge model for top items."""
        extended = self.use_extended_for_volatile and item_code in self.VOLATILE_ITEMS
        df = self._create_features(series, extended=extended)
        df["target"] = df["y"].shift(-1)
        df = df.dropna()

        if self.train_start:
            df = df[df.index >= pd.to_datetime(self.train_start)]

        if len(df) < 24:
            return None

        feature_cols = [c for c in df.columns if c not in ["target", "y"]]
        X = df[feature_cols].values
        y = df["target"].values

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Higher regularization for volatile items
        alpha = 200.0 if item_code in self.VOLATILE_ITEMS else 100.0
        model = Ridge(alpha=alpha, random_state=self.random_state)
        model.fit(X_scaled, y)

        return {
            "type": "ridge",
            "model": model,
            "scaler": scaler,
            "feature_cols": feature_cols,
            "extended": extended,
            "last_data": series.copy(),
        }

    def _fit_voting(self, series, item_code):
        """Fit VotingRegressor for other items."""
        df = self._create_features(series, extended=False)
        df["target"] = df["y"].shift(-1)
        df = df.dropna()

        if self.train_start:
            df = df[df.index >= pd.to_datetime(self.train_start)]

        if len(df) < 24:
            return None

        feature_cols = [c for c in df.columns if c not in ["target", "y"]]
        X = df[feature_cols].values
        y = df["target"].values

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        model = VotingRegressor(
            [
                ("ridge", Ridge(alpha=100.0, random_state=self.random_state)),
                (
                    "lasso",
                    Lasso(alpha=0.1, random_state=self.random_state, max_iter=5000),
                ),
            ]
        )
        model.fit(X_scaled, y)

        return {
            "type": "voting",
            "model": model,
            "scaler": scaler,
            "feature_cols": feature_cols,
            "extended": False,
            "last_data": series.copy(),
        }

    def _load_data(self, data_dir):
        from sirena.data.micro_basket import load_micro_basket
        self.basket = load_micro_basket(data_dir, self.cutoff)
        self.weights = self.basket['leaves'].Weight_vertical.to_dict()
        self.item_names = self.basket['names']
        self.item_subcomp = self.basket['leaves'].Subcomponent.astype(int).to_dict()
        self.top_items = set(sorted(self.weights, key=self.weights.get, reverse=True)[:self.top_n])
        return self.basket['history'].reindex(columns=list(self.weights))

    def fit(self, df, target_col='Все товары и услуги'):
        from sirena.data_loader import DataFreshnessError
        if df.attrs.get('input_contract', {}).get('representation') == 'sa':
            raise DataFreshnessError('Micro requires homogeneous RAW inputs')
        self._is_fitted = False
        self.micro_models, self.parent_models, self.fallbacks = {}, {}, {}
        self.cutoff = pd.Timestamp(df.index.max()).to_period('M').to_timestamp()
        self.macro_df = df.loc[:self.cutoff].copy()
        data_dir = self.data_dir or Path(__file__).parent.parent.parent / 'data'
        self.basket = None
        micro = self._load_data(data_dir)
        if self.basket is None:
            raise DataFreshnessError(f'Micro requires a validated basket and parent history through {self.cutoff:%Y-%m}')
        history = self.basket['history']
        for parent in self.basket['groups'].index:
            series = history[parent].loc[:self.cutoff]
            series = series.loc[series.first_valid_index():]
            result = self._fit_ridge(series, int(parent))
            self.parent_models[int(parent)] = {'fit':result, 'history':series.copy()}
        for code in self.weights:
            series = micro[code].loc[:self.cutoff]
            first = series.first_valid_index()
            reason = None
            if first is None:
                reason = 'no_item_history'
            elif not np.isfinite(series.iloc[-1]):
                reason = 'missing_current_observation'
            elif series.notna().sum() < 36:
                reason = 'short_history'
            elif not np.isfinite(series.tail(13)).all():
                reason = 'calendar_gap_in_required_features'
            result = None
            if reason is None:
                # Trim leading absence, preserving the complete monthly calendar.
                series = series.loc[first:]
                result = (self._fit_ridge(series, code) if code in self.top_items
                          else self._fit_voting(series, code))
                if result is None:
                    reason = 'insufficient_training_rows'
            if result is not None:
                self.micro_models[code] = result
            else:
                parent = self.item_subcomp[code]
                # Short CURRENT series retain their recent relative signal.
                # Missing-current series use only the parent's current trajectory.
                offset = 0.
                if np.isfinite(series.iloc[-1]):
                    residual = (series - history[parent]).tail(6).dropna()
                    offset = float(residual.mean() * len(residual)/(len(residual)+12)) if len(residual) else 0.
                last = series.last_valid_index()
                self.fallbacks[code] = {'parent':parent, 'reason':reason, 'offset':offset,
                    'last_observation':str(last.date()) if last is not None else None}
        self.coverage = {
            'cutoff':str(self.cutoff.date()),
            'weight_vintage':str(self.basket['weight_vintage'].date()),
            'weight_policy':'latest vintage at cutoff, frozen through forecast horizon',
            'representation':'raw', 'parent_method':self.fallback_policy,
            'basket_items':len(self.weights),
            'observed_weight':float(sum(w for c,w in self.weights.items() if np.isfinite(micro[c].iloc[-1]))),
            'native_model_weight':float(sum(self.weights[c] for c in self.micro_models)),
            'fallback_item_weight':float(sum(self.weights[c] for c in self.fallbacks)),
            'residual_parent_weight':float(self.basket['residual'].sum()),
            'forecast_weight':float(sum(self.weights.values())+self.basket['residual'].sum()),
            'fallback_items':[{'item_code':int(c), 'name':self.item_names.get(c,str(c)),
                'weight':float(self.weights[c]), **v} for c,v in self.fallbacks.items()],
            'residual_buckets':{str(c):float(w) for c,w in self.basket['residual'].items() if w>1e-10},
            'historical_classification':'revised source hierarchy, not real-time publication vintages',
        }
        if not np.isclose(self.coverage['forecast_weight'],1.,atol=1e-8,rtol=0):
            raise DataFreshnessError('Micro forecast weight is not one')
        self._is_fitted = True
        return self

    @staticmethod
    def _feature_row(values, date, columns):
        """Exact equivalent of the last _create_features row, without rebuilding a frame."""
        a = np.asarray(values,dtype=float)
        if len(a)<13:
            raise ValueError('At least 13 calendar months required for a native forecast')
        row = {f'L{lag}':a[-lag-1] for lag in [1,2,3,6,12]}
        row.update(D1=a[-1]-a[-2], MA3=np.mean(a[-3:]),
                   month_sin=np.sin(2*np.pi*date.month/12), month_cos=np.cos(2*np.pi*date.month/12),
                   MA6=np.mean(a[-6:]), STD3=np.std(a[-3:],ddof=1), STD6=np.std(a[-6:],ddof=1),
                   MAX3=np.max(a[-3:]), MIN3=np.min(a[-3:]), RANGE3=np.ptp(a[-3:]))
        out=np.array([[row[c] for c in columns]])
        if not np.isfinite(out).all():
            raise ValueError('Non-finite native features; cannot replace them by zero')
        return out

    def _model_step(self, fitted, values, date):
        features=self._feature_row(values,date,fitted['feature_cols'])
        pred=float(fitted['model'].predict(fitted['scaler'].transform(features))[0])
        if not np.isfinite(pred):
            raise ValueError('Non-finite native forecast')
        return pred

    def forecast(self, horizon=None):
        if not self._is_fitted:
            raise ValueError('Model not fitted')
        h=int(horizon if horizon is not None else self.horizon)
        if h<1:
            raise ValueError('Positive forecast horizon required')
        histories={c:list(v['last_data'].tail(13).to_numpy()) for c,v in self.micro_models.items()}
        parent_histories={c:list(v['history'].tail(13).to_numpy()) for c,v in self.parent_models.items()}
        output, details, rows = [], [], []
        for step in range(1,h+1):
            target=self.cutoff+pd.DateOffset(months=step)
            origin=target-pd.DateOffset(months=1)
            parents, parent_fallbacks = {}, []
            for c,v in self.parent_models.items():
                if self.fallback_policy=='parent_ridge' and v['fit'] is not None:
                    try:
                        pred=self._model_step(v['fit'],parent_histories[c],origin)
                    except ValueError as exc:
                        pred=float(parent_histories[c][-12])
                        parent_fallbacks.append({'parent':c,'reason':str(exc)})
                else:
                    pred=float(parent_histories[c][-12])
                if not np.isfinite(pred):
                    raise ValueError(f'No valid parent forecast: {c}')
                parents[c]=pred
                parent_histories[c].append(pred)
            total, native_weight, dynamic_fallbacks = 0., 0., []
            for c,w in self.weights.items():
                parent=self.item_subcomp[c]
                reason=None
                if c in self.micro_models:
                    try:
                        pred=self._model_step(self.micro_models[c],histories[c],origin)
                        native_weight+=w
                    except ValueError as exc:
                        pred=parents[parent]
                        reason=str(exc)
                        dynamic_fallbacks.append({'item_code':int(c),'weight':float(w),'reason':reason})
                    histories[c].append(pred)
                else:
                    fallback=self.fallbacks[c]
                    pred=parents[parent]+fallback['offset']
                    reason=fallback['reason']
                total+=w*pred
                rows.append({'date':str(target.date()),'Item_code':int(c),
                             'Name':self.item_names.get(c,str(c)),'Weight':float(w),
                             'Prediction':float(pred),'Contribution':float(w*pred),
                             'Method':reason or 'native','Parent':parent})
            for c,w in self.basket['residual'].items():
                if w>1e-10:
                    total+=w*parents[int(c)]
                    rows.append({'date':str(target.date()),'Item_code':f'residual:{c}',
                                 'Name':self.item_names.get(c,str(c)),'Weight':float(w),
                                 'Prediction':float(parents[int(c)]),'Contribution':float(w*parents[int(c)]),
                                 'Method':'residual_parent_weight','Parent':int(c)})
            if self.use_seasonal_adj:
                # Explicit legacy experiment only; disabled in production/comparison.
                total+=self.SEASONAL_ADJ.get(target.month,0.)
            output.append(float(total))
            details.append({'date':str(target.date()),'native_weight':float(native_weight),
                            'parent_weight':float(1-native_weight),'dynamic_fallbacks':dynamic_fallbacks,
                            'parent_model_fallbacks':parent_fallbacks})
        self.forecast_details=details
        self.last_item_forecasts=pd.DataFrame(rows)
        return np.asarray(output)

    def predict(self, df, target_date):
        target=pd.Timestamp(target_date).to_period('M').to_timestamp()
        if pd.Timestamp(df.index.max()).to_period('M').to_timestamp()<self.cutoff:
            raise ValueError('Prediction context predates fitted cutoff')
        h=(target.year-self.cutoff.year)*12+target.month-self.cutoff.month
        if h<1:
            raise ValueError('Forecast target must follow fitted cutoff')
        value=float(self.forecast(h)[h-1])
        return {'prediction':100+value,'model':self.name,'coverage':self.coverage}

    def get_stats(self):
        return {'n_models':len(self.micro_models),'n_top_models':len(set(self.micro_models)&self.top_items),
                'total_weight':self.coverage['forecast_weight']*100, **self.coverage}

    def get_top_predictions(self,target_date,n=20):
        self.predict(self.macro_df,target_date)
        return self.last_item_forecasts[self.last_item_forecasts.date.eq(str(pd.Timestamp(target_date).date()))].nlargest(n,'Weight')
