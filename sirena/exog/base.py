"""
Base class for Exogenous Variable Forecasters.
"""
from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union

class BaseExogForecaster(ABC):
    """
    Abstract base class for exogenous variable forecasting.
    
    Exogenous forecasters predict future values of macro indicators
    (USD, Brent, Ruonia, etc.) to be used by the main inflation models.
    """
    
    def __init__(self, name: str):
        self.name = name
        self._is_fitted = False
        
    @abstractmethod
    def fit(self, series: pd.Series, **kwargs) -> 'BaseExogForecaster':
        """
        Fit the model to the historical series.
        
        Args:
            series: pandas Series with DatetimeIndex
        """
        pass
        
    @abstractmethod
    def forecast(self, horizon: int) -> pd.Series:
        """
        Forecast future values.
        
        Args:
            horizon: Number of steps to forecast
            
        Returns:
            pd.Series with forecasted values and DatetimeIndex
        """
        pass
        
    def _check_fitted(self):
        if not self._is_fitted:
            raise ValueError(f"Model {self.name} is not fitted yet.")
