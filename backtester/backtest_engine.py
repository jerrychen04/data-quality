from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from typing import List, Dict, Any

class BacktestEngine(ABC):

    def __init__(self, initial_cash: float):
        self.initial_cash = initial_cash

    @abstractmethod
    def run_backtest(self, weights_df: pd.DataFrame, price_df: pd.DataFrame) -> Dict[str, Any]:
        pass

class EquityBacktestEngine(BacktestEngine):

    def run_backtest(self, weights_df: pd.DataFrame, price_df: pd.DataFrame) -> Dict[str, Any]:
        aligned_prices, aligned_weights = price_df.align(weights_df, join='right', axis=0, copy=False) 
        aligned_prices = aligned_prices.ffill() 
        daily_returns = aligned_prices.pct_change()
        lagged_weights = aligned_weights.shift(1)

        portfolio_daily_returns = (lagged_weights * daily_returns).sum(axis=1)
        portfolio_daily_returns = portfolio_daily_returns.iloc[1:] 
        portfolio_cumulative_returns = (1 + portfolio_daily_returns).cumprod()
        portfolio_values = self.initial_cash * portfolio_cumulative_returns
        portfolio_values.iloc[0] = self.initial_cash 
        
        start_date = lagged_weights.index[0] 
        portfolio_values = pd.concat([pd.Series([self.initial_cash], index=[start_date]), portfolio_values])
        portfolio_values.name = "Portfolio Value"
        print(f"Backtest complete. Final Portfolio Value: {portfolio_values.iloc[-1]:.2f}")
        return {"portfolio_values": portfolio_values, "portfolio_returns": portfolio_daily_returns}