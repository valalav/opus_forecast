#!/usr/bin/env python3
"""
СИРЕНА-КБР v2.5 — Система Интеллектуального Регионального Анализа
Кабардино-Балкарская Республика

Версия 2.5: Модель с макроэкономическими данными
Улучшение: -14.9% MAE по сравнению с v2.0

Новое в v2.5:
- Добавлены макро-признаки: курс USD, ставка RUONIA
- Использует inflation_data.csv (содержит все макропоказатели)
- КПЭ улучшен до 81.7%

Требуемые данные: inflation_data.csv с колонками:
- Date, mom, Nonprod, Prod, Serv, usd_nom_i, Ki_i, Ruonia, ...
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import RobustScaler
from typing import Dict, Optional
import warnings
warnings.filterwarnings('ignore')


class SirenaKBR_v25:
    """СИРЕНА-КБР v2.5: Модель с макроэкономическими данными."""
    
    # === ПАРАМЕТРЫ МОДЕЛИ ===
    OUTLIER_YEARS = [2022, 2010]
    ALPHA = 0.3
    MIN_TRAIN_SIZE = 36
    
    ETS_WEIGHTS = {
        1: 0.9, 2: 0.0, 3: 0.5, 4: 0.3, 5: 0.9, 6: 0.5,
        7: 0.0, 8: 0.5, 9: 0.9, 10: 0.9, 11: 0.0, 12: 0.0,
    }
    
    # Признаки v2.5 (добавлены usd_lag1, ruonia_lag1)
    FEATURES = [
        'y_lag1', 'y_lag2', 'y_lag12', 'y_ma3',
        'month_sin', 'month_cos',
        'food_lag1', 'nonfood_lag1', 'services_lag1',
        'seasonal_norm', 'deviation_lag1',
        'usd_lag1', 'ruonia_lag1',  # НОВЫЕ в v2.5
    ]
    
    def __init__(self):
        self.model = None
        self.scaler = None
        self.seasonal_norm = None
        self.is_fitted = False
        self.last_fact_date = None
        
    def _prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Подготовка признаков."""
        df = df.copy()
        
        df['month'] = df.index.month
        df['year'] = df.index.year
        
        # Целевая (если есть mom как индекс 100+x)
        if 'target' not in df.columns:
            if 'mom' in df.columns:
                df['target'] = df['mom'] - 100
            elif 'Все товары и услуги' in df.columns:
                df['target'] = df['Все товары и услуги']
        
        # Лаги целевой
        df['y_lag1'] = df['target'].shift(1)
        df['y_lag2'] = df['target'].shift(2)
        df['y_lag12'] = df['target'].shift(12)
        df['y_ma3'] = df['target'].rolling(3).mean().shift(1)
        
        # Сезонные
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        
        # Компоненты
        if 'Prod' in df.columns:
            df['food_lag1'] = (df['Prod'] - 100).shift(1)
            df['nonfood_lag1'] = (df['Nonprod'] - 100).shift(1)
            df['services_lag1'] = (df['Serv'] - 100).shift(1)
        elif 'Продовольственные товары' in df.columns:
            df['food_lag1'] = df['Продовольственные товары'].shift(1)
            df['nonfood_lag1'] = df['Непродовольственные товары'].shift(1)
            df['services_lag1'] = df['Услуги'].shift(1)
        
        # Макро
        if 'usd_nom_i' in df.columns:
            df['usd_lag1'] = df['usd_nom_i'].shift(1)
        if 'Ruonia' in df.columns:
            df['ruonia_lag1'] = df['Ruonia'].shift(1)
        
        return df
    
    def _compute_seasonal_norm(self, df: pd.DataFrame) -> pd.Series:
        """Сезонная норма без выбросных лет."""
        clean_df = df[~df['year'].isin(self.OUTLIER_YEARS)]
        return clean_df.groupby('month')['target'].mean()
    
    def _get_last_fact_date(self, df: pd.DataFrame) -> pd.Timestamp:
        """Последняя дата с фактом."""
        return df.dropna(subset=['target']).index.max()
    
    def fit(self, df: pd.DataFrame) -> 'SirenaKBR_v25':
        """Обучение модели."""
        df = self._prepare_features(df)
        self.last_fact_date = self._get_last_fact_date(df)
        self.seasonal_norm = self._compute_seasonal_norm(df)
        
        df['seasonal_norm'] = df['month'].map(self.seasonal_norm)
        df['deviation_lag1'] = df['y_lag1'] - df['month'].shift(1).map(self.seasonal_norm)
        
        train_df = df[~df['year'].isin(self.OUTLIER_YEARS)]
        train_clean = train_df.dropna(subset=self.FEATURES + ['target'])
        
        if len(train_clean) < self.MIN_TRAIN_SIZE:
            raise ValueError(f"Недостаточно данных: {len(train_clean)} < {self.MIN_TRAIN_SIZE}")
        
        X = train_clean[self.FEATURES].values
        y = train_clean['target'].values
        
        self.scaler = RobustScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        self.model = Ridge(alpha=self.ALPHA)
        self.model.fit(X_scaled, y)
        
        self.is_fitted = True
        return self
    
    def predict(self, df: pd.DataFrame, target_date: pd.Timestamp) -> Dict:
        """Прогноз на дату."""
        if not self.is_fitted:
            raise ValueError("Модель не обучена")
        
        df = self._prepare_features(df)
        df['seasonal_norm'] = df['month'].map(self.seasonal_norm)
        df['deviation_lag1'] = df['y_lag1'] - df['month'].shift(1).map(self.seasonal_norm)
        
        test_row = df.loc[[target_date]]
        X_test = self.scaler.transform(test_row[self.FEATURES].values)
        pred_ridge = self.model.predict(X_test)[0]
        
        m = target_date.month
        pred_ets = self.seasonal_norm.get(m, 0)
        ets_w = self.ETS_WEIGHTS.get(m, 0.3)
        pred = (1 - ets_w) * pred_ridge + ets_w * pred_ets
        
        return {
            'date': target_date,
            'prediction': pred,
            'pred_ridge': pred_ridge,
            'pred_ets': pred_ets,
            'ets_weight': ets_w,
        }
    
    def backtest(self, df: pd.DataFrame, 
                 start_date: str = '2019-01-01',
                 end_date: str = None,
                 verbose: bool = False) -> pd.DataFrame:
        """Скользящий бэктест."""
        last_fact = self._get_last_fact_date(self._prepare_features(df))
        if end_date is None:
            end_date = last_fact
        
        test_dates = pd.date_range(start=start_date, end=end_date, freq='MS')
        results = []
        
        for target_date in test_dates:
            if target_date not in df.index:
                continue
            
            cutoff = target_date - pd.DateOffset(months=1)
            train_df = df[df.index <= cutoff].copy()
            
            try:
                self.fit(train_df)
                test_df = df[df.index <= target_date].copy()
                pred = self.predict(test_df, target_date)
                
                actual = self._prepare_features(df).loc[target_date, 'target']
                
                results.append({
                    'date': target_date,
                    'year': target_date.year,
                    'month': target_date.month,
                    'actual': actual,
                    'prediction': pred['prediction'],
                    'error': actual - pred['prediction'],
                })
            except Exception as e:
                if verbose:
                    print(f"Ошибка для {target_date}: {e}")
                continue
        
        return pd.DataFrame(results)


def load_data(filepath: str) -> pd.DataFrame:
    """Загрузка inflation_data.csv."""
    df = pd.read_csv(filepath, sep=';', encoding='utf-8-sig')
    
    # Конвертация запятых в точки
    for col in df.columns:
        if col != 'Date':
            df[col] = df[col].astype(str).str.replace(',', '.').astype(float)
    
    df['Date'] = pd.to_datetime(df['Date'], format='%d.%m.%Y')
    df['Date'] = df['Date'].dt.to_period('M').dt.to_timestamp()
    df = df.set_index('Date').sort_index()
    
    return df


def main():
    """Основная функция."""
    print("="*60)
    print("СИРЕНА-КБР v2.5 (с макроданными)")
    print("="*60)
    
    # Загрузка
    try:
        df = load_data('/mnt/user-data/uploads/inflation_data.csv')
    except:
        df = load_data('data/inflation_data.csv')
    
    last_fact = df.dropna(subset=['mom']).index.max()
    print(f"\nДанные: {df.index.min().strftime('%Y-%m')} — {last_fact.strftime('%Y-%m')}")
    
    # Модель
    model = SirenaKBR_v25()
    
    # Бэктест
    print("\nБэктест...")
    results = model.backtest(df, verbose=True)
    
    # Метрики
    mae = results['error'].abs().mean()
    kpi = (results['error'].abs() <= 0.5).sum()
    
    print(f"\nМетрики v2.5:")
    print(f"  MAE:  {mae:.4f}")
    print(f"  КПЭ:  {kpi}/{len(results)} ({kpi/len(results)*100:.1f}%)")
    
    print(f"\nУлучшение vs v2.0: +{(0.3787-mae)/0.3787*100:.1f}%")


if __name__ == "__main__":
    main()
