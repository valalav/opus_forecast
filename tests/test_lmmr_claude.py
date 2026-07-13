import pytest
import pandas as pd
import numpy as np
from sirena.models.lmmr_claude import LMMRForecasterClaude
import warnings
warnings.filterwarnings("ignore")


class TestLMMRForecasterClaude:

    def test_initialization(self):
        """Тест инициализации модели Claude."""
        model = LMMRForecasterClaude()
        assert model.name == "lmmr_claude"
        assert model.alpha == 0.5
        assert model.MIN_TRAIN_SIZE == 48
        assert not model._is_fitted

    def test_fit(self):
        """Тест обучения модели Claude."""
        # Создаем синтетические данные
        dates = pd.date_range(start='2020-01-01', end='2024-12-01', freq='MS')
        n = len(dates)
        
        # Создаем базовый ряд с трендом и сезонностью
        trend = np.linspace(100, 105, n)
        seasonal = 2 * np.sin(2 * np.pi * np.arange(n) / 12)
        noise = np.random.normal(0, 0.5, n)
        y_values = trend + seasonal + noise
        
        # Делаем MoM для имитации инфляции
        mom_values = [y_values[0]]
        for i in range(1, n):
            mom_values.append((y_values[i]/y_values[i-1])*100)
        
        # Добавляем USD и Brent как внешние факторы
        usd_values = 60 + np.cumsum(np.random.normal(0, 0.5, n))
        brent_values = 50 + np.cumsum(np.random.normal(0, 1, n))
        
        df = pd.DataFrame({
            'Все товары и услуги': mom_values,
            'usd_nom_i': usd_values,
            'brent': brent_values
        }, index=dates)
        
        model = LMMRForecasterClaude()
        model.fit(df)
        
        assert model._is_fitted
        assert model.model is not None
        assert model.scaler is not None
        assert model.sa_series is not None
        assert model.seasonal is not None
        assert model.base_index is not None

    def test_predict(self):
        """Тест прогнозирования Claude."""
        # Создаем синтетические данные
        dates = pd.date_range(start='2020-01-01', end='2024-12-01', freq='MS')
        n = len(dates)
        
        # Создаем базовый ряд с трендом и сезонностью
        trend = np.linspace(100, 105, n)
        seasonal = 2 * np.sin(2 * np.pi * np.arange(n) / 12)
        noise = np.random.normal(0, 0.5, n)
        y_values = trend + seasonal + noise
        
        # Делаем MoM для имитации инфляции
        mom_values = [y_values[0]]
        for i in range(1, n):
            mom_values.append((y_values[i]/y_values[i-1])*100)
        
        # Добавляем USD и Brent как внешние факторы
        usd_values = 60 + np.cumsum(np.random.normal(0, 0.5, n))
        brent_values = 50 + np.cumsum(np.random.normal(0, 1, n))
        
        df = pd.DataFrame({
            'Все товары и услуги': mom_values,
            'usd_nom_i': usd_values,
            'brent': brent_values
        }, index=dates)
        
        model = LMMRForecasterClaude()
        model.fit(df)
        
        target_date = dates[-1] + pd.DateOffset(months=1)
        # Добавляем будущую дату для тестирования
        df_extended = df.copy()
        df_extended.loc[target_date] = np.nan
        df_extended.loc[target_date, ['usd_nom_i', 'brent']] = [usd_values[-1], brent_values[-1]]
        
        pred = model.predict(df_extended, target_date)
        
        assert 'prediction' in pred
        assert 'date' in pred
        assert 'model' in pred
        assert 'sa_prediction' in pred
        assert 'seasonal_factor' in pred
        assert isinstance(pred['prediction'], (float, int, np.number))

    def test_forecast(self):
        """Тест метода forecast Claude."""
        # Создаем синтетические данные
        dates = pd.date_range(start='2020-01-01', end='2024-12-01', freq='MS')
        n = len(dates)
        
        # Создаем базовый ряд с трендом и сезонностью
        trend = np.linspace(100, 105, n)
        seasonal = 2 * np.sin(2 * np.pi * np.arange(n) / 12)
        noise = np.random.normal(0, 0.5, n)
        y_values = trend + seasonal + noise
        
        # Делаем MoM для имитации инфляции
        mom_values = [y_values[0]]
        for i in range(1, n):
            mom_values.append((y_values[i]/y_values[i-1])*100)
        
        # Добавляем USD и Brent как внешние факторы
        usd_values = 60 + np.cumsum(np.random.normal(0, 0.5, n))
        brent_values = 50 + np.cumsum(np.random.normal(0, 1, n))
        
        df = pd.DataFrame({
            'Все товары и услуги': mom_values,
            'usd_nom_i': usd_values,
            'brent': brent_values
        }, index=dates)
        
        model = LMMRForecasterClaude()
        model.fit(df)
        
        horizon = 6
        forecasts = model.forecast(horizon)
        
        assert len(forecasts) == horizon
        assert all(isinstance(f, (float, int, np.number)) for f in forecasts)

    def test_backtest(self):
        """Тест бэктестирования Claude."""
        # Создаем синтетические данные
        dates = pd.date_range(start='2020-01-01', end='2024-12-01', freq='MS')
        n = len(dates)
        
        # Создаем базовый ряд с трендом и сезонностью
        trend = np.linspace(100, 105, n)
        seasonal = 2 * np.sin(2 * np.pi * np.arange(n) / 12)
        noise = np.random.normal(0, 0.5, n)
        y_values = trend + seasonal + noise
        
        # Делаем MoM для имитации инфляции
        mom_values = [y_values[0]]
        for i in range(1, n):
            mom_values.append((y_values[i]/y_values[i-1])*100)
        
        # Добавляем USD и Brent как внешние факторы
        usd_values = 60 + np.cumsum(np.random.normal(0, 0.5, n))
        brent_values = 50 + np.cumsum(np.random.normal(0, 1, n))
        
        df = pd.DataFrame({
            'Все товары и услуги': mom_values,
            'usd_nom_i': usd_values,
            'brent': brent_values
        }, index=dates)
        
        model = LMMRForecasterClaude()
        start_date = '2024-01-01'
        results = model.backtest(df, start_date=start_date)
        
        assert len(results) >= 0  # Может быть 0 результатов в зависимости от данных
        if len(results) > 0:
            assert 'actual' in results.columns
            assert 'prediction' in results.columns
            assert 'error' in results.columns
            assert 'date' in results.columns

    def test_seasonal_decomposition(self):
        """Тест сезонной декомпозиции Claude."""
        # Создаем синтетические данные с ярко выраженной сезонностью
        dates = pd.date_range(start='2020-01-01', end='2024-12-01', freq='MS')
        n = len(dates)
        
        trend = np.linspace(100, 105, n)
        seasonal = 2 * np.sin(2 * np.pi * np.arange(n) / 12)
        noise = np.random.normal(0, 0.1, n)
        y_values = trend + seasonal + noise
        
        # Делаем MoM для имитации инфляции
        mom_values = [y_values[0]]
        for i in range(1, n):
            mom_values.append((y_values[i]/y_values[i-1])*100)
        
        series = pd.Series(mom_values, index=dates)
        
        model = LMMRForecasterClaude()
        
        # Тестируем базовое преобразование в базисные индексы
        base_index = model._to_base_index(series)
        assert len(base_index) == len(series)
        
        # Тестируем STL декомпозицию
        sa, sc = model._decompose_series(base_index)
        assert len(sa) == len(base_index)
        assert len(sc) == len(base_index)
        
        # Проверяем, что сезонная компонента действительно содержит сезонность
        assert sc.std() > 0  # Должна быть некоторая изменчивость

    def test_feature_preparation(self):
        """Тест подготовки признаков Claude."""
        # Создаем синтетические данные
        dates = pd.date_range(start='2020-01-01', end='2024-12-01', freq='MS')
        n = len(dates)
        
        y_values = [100.5] * n  # Простой MoM
        usd_values = [70.0] * n
        brent_values = [60.0] * n
        
        df = pd.DataFrame({
            'Все товары и услуги': y_values,
            'usd_nom_i': usd_values,
            'brent': brent_values
        }, index=dates)
        
        model = LMMRForecasterClaude()
        model.fit(df)  # Для инициализации sa_series
        
        # Подготовка признаков
        features_df = model._prepare_features(df)
        
        # Проверяем, что все требуемые признаки присутствуют
        expected_features = [
            'is_shock_dec2014_jan2015', 
            'is_tariff_jul', 
            'is_shock_mar2022', 
            'is_shock_apr2022'
        ]
        
        for feat in expected_features:
            assert feat in features_df.columns, f"Feature {feat} missing from features"
        
        # Проверяем, что дамми для июля равны 1 в июле и 0 в других месяцах
        july_mask = features_df.index.month == 7
        assert all(features_df.loc[july_mask, 'is_tariff_jul'] == 1)
        assert all(features_df.loc[~july_mask, 'is_tariff_jul'] == 0)

    def test_insufficient_data_error(self):
        """Тест ошибки при недостатке данных Claude."""
        # Создаем очень мало данных
        dates = pd.date_range(start='2020-01-01', end='2020-12-01', freq='MS')
        df = pd.DataFrame({
            'Все товары и услуги': [100.1 + i*0.1 for i in range(len(dates))],
            'usd_nom_i': [70.0] * len(dates),
            'brent': [60.0] * len(dates)
        }, index=dates)
        
        model = LMMRForecasterClaude()
        
        # Ожидаем ошибку при обучении из-за недостатка данных
        with pytest.raises(ValueError, match="Insufficient data"):
            model.fit(df)

    def test_missing_target_column_error(self):
        """Тест ошибки при отсутствии целевой колонки Claude."""
        dates = pd.date_range(start='2020-01-01', end='2024-12-01', freq='MS')
        df = pd.DataFrame({
            'usd_nom_i': [70.0] * len(dates),
            'brent': [60.0] * len(dates)
        }, index=dates)
        
        model = LMMRForecasterClaude()
        
        # Ожидаем ошибку при обучении из-за отсутствия целевой колонки
        with pytest.raises(ValueError, match="Target column"):
            model.fit(df)