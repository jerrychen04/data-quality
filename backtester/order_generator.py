from abc import ABC, abstractmethod
import pandas as pd
from typing import List, Dict, Any

class OrderGenerator(ABC):
    """Interface for generating trade orders based on a strategy."""
    
    @abstractmethod
    def generate_orders(self, data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Generate orders given historical price data."""
        pass

class BettingAgainstBetaOrderGenerator(OrderGenerator):
    """Betting Against Beta (BAB) strategy implementation."""
    
    def __init__(self, lookback_period: int = 60, rebalance_frequency: str = 'ME', starting_portfolio_value: float = 100000):
        self.lookback_period = lookback_period
        self.rebalance_frequency = rebalance_frequency
        self.starting_portfolio_value = starting_portfolio_value
    
    def calculate_beta(self, stock_returns: pd.Series, market_returns: pd.Series) -> float:
        """
        Calculate beta of a stock relative to the market.
        """
        covariance = stock_returns.cov(market_returns)
        market_variance = market_returns.var()
        beta = covariance / market_variance
        return beta

    def calculate_betas(self, data, spy_returns, date):
        beta_values = {}
        for ticker, df in data.items():
            if ticker == 'SPY':
                continue
            stock_returns = df['Adj Close'].pct_change(fill_method=None).dropna()

            combined_returns = pd.concat([stock_returns, spy_returns], axis=1, join='inner').loc[:date]
            combined_returns = combined_returns.iloc[-self.lookback_period:]

            if len(combined_returns) < self.lookback_period:
                continue

            recent_stock_returns = combined_returns.iloc[:, 0]
            recent_spy_returns = combined_returns.iloc[:, 1]
            beta = self.calculate_beta(recent_stock_returns, recent_spy_returns)
            beta_values[ticker] = beta

        return beta_values

    def generate_orders_for_date(self, beta_values, date):
        beta_series = pd.Series(beta_values)
        beta_series = beta_series.dropna()
        sorted_beta = beta_series.sort_values()

        num_stocks = len(sorted_beta)
        decile_size = max(int(num_stocks * 0.1), 1)
        low_beta_tickers = sorted_beta.head(decile_size).index.tolist()
        high_beta_tickers = sorted_beta.tail(decile_size).index.tolist()

        avg_low_beta = beta_series[low_beta_tickers].mean()
        avg_high_beta = beta_series[high_beta_tickers].mean()

        # ensure beta neutrality with equal weights
        low_beta_weight = avg_high_beta / (avg_low_beta + avg_high_beta)
        high_beta_weight = avg_low_beta / (avg_low_beta + avg_high_beta)

        orders = []
        for ticker in low_beta_tickers:
            quantity = int(self.starting_portfolio_value * low_beta_weight / decile_size)
            orders.append({
                "date": date,
                "type": "BUY",
                "ticker": ticker,
                "quantity": quantity  
            })
            print(f"Buying {ticker} on {date}")

        for ticker in high_beta_tickers:
            quantity = int(self.starting_portfolio_value * high_beta_weight / decile_size)
            orders.append({
                "date": date,
                "type": "SELL",
                "ticker": ticker,
                "quantity": quantity
            })
            print(f"Selling {ticker} on {date}")

        return orders

    def generate_orders(self, data: Dict[str, pd.DataFrame]) -> List[Dict[str, Any]]:
        spy_data = data.get('SPY')
        if spy_data is None:
            raise ValueError("SPY data is required for beta calculation.")
        
        spy_returns = spy_data['Adj Close'].pct_change()
        spy_returns = spy_returns.dropna()

        start_date = spy_returns.index[self.lookback_period]
        end_date = spy_returns.index[-1]
        rebalance_dates = pd.date_range(start=start_date, end=end_date, freq=self.rebalance_frequency)

        all_orders = []

        for date in rebalance_dates:
            beta_values = self.calculate_betas(data, spy_returns, date)
            if len(beta_values) < 20:
                continue
            orders = self.generate_orders_for_date(beta_values, date)
            all_orders.extend(orders)

        return all_orders