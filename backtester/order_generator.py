from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from typing import List, Dict, Any

class OrderGenerator(ABC):
    """Interface for generating trade orders based on a strategy."""
    
    @abstractmethod
    def generate_orders(self, data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Generate orders given historical price data."""
        pass

class MeanReversionOrderGenerator(OrderGenerator):
    """Mean reversion strategy implementation with 100-day rolling window."""
    def generate_orders(self, data: pd.DataFrame) -> List[Dict[str, Any]]:
        orders = []
        tickers = data.columns

        for ticker in tickers:
            ticker_data = data[ticker].to_frame(name='Adj Close')
            ticker_data['100_day_avg'] = ticker_data['Adj Close'].rolling(window=100).mean()

            for date, row in ticker_data.iterrows():
                if pd.isna(row['100_day_avg']):
                    continue
                if row['Adj Close'] < row['100_day_avg']:
                    orders.append({"date": date, "type": "BUY", "ticker": ticker, "quantity": 100})
                else:
                    orders.append({"date": date, "type": "SELL", "ticker": ticker, "quantity": 100})

        return orders


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

        # TODO: Implement position sizing based on portfolio value / parameterization for leverage to control MAX drawdown metric (sharpe should be the same)
        # long bottom decile, short top decile of beta stocks
        for ticker in low_beta_tickers:
            quantity = int(self.starting_portfolio_value * low_beta_weight / decile_size)
            max_allocation = self.starting_portfolio_value * 0.02
            quantity = min(quantity, max_allocation)
            orders.append({
                "date": date,
                "type": "BUY",
                "ticker": ticker,
                "quantity": quantity  
            })
            print(f"Buying {ticker} on {date}")

        for ticker in high_beta_tickers:
            quantity = int(self.starting_portfolio_value * high_beta_weight / decile_size)
            max_allocation = self.starting_portfolio_value * 0.02
            quantity = min(quantity, max_allocation)
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
    
class BettingAgainstIVOLOrderGenerator(OrderGenerator):
    """Betting Against Idiosyncratic Volatility (BAI) strategy implementation."""
   
    def __init__(self, lookback_period: int = 60, rebalance_frequency: str = 'ME', starting_portfolio_value: float = 100000):
        self.lookback_period = lookback_period
        self.rebalance_frequency = rebalance_frequency
        self.starting_portfolio_value = starting_portfolio_value
   
    def calculate_ivol(self, stock_returns: pd.Series, market_returns: pd.Series) -> float:
        """
        Calculate Idiosyncratic Volatility (IVOL) of a stock.
        """
        if len(stock_returns) < self.lookback_period or len(market_returns) < self.lookback_period:
            return np.nan
       
        # Run a regression
        X = market_returns[-self.lookback_period:].values.reshape(-1, 1)
        y = stock_returns[-self.lookback_period:].values.reshape(-1, 1)


        if np.isnan(X).any() or np.isnan(y).any():
            return np.nan
       
        # Compute residual standard deviation (IVOL)
        beta = np.linalg.lstsq(X, y, rcond=None)[0][0]
        residuals = y.flatten() - (beta * X.flatten())
        ivol = np.std(residuals)
       
        return ivol


    def calculate_ivols(self, data, market_returns, date):
        ivol_values = {}
        for ticker, df in data.items():
            if ticker == 'SPY':  # Market proxy
                continue
           
            stock_returns = df['Adj Close'].pct_change(fill_method=None).dropna()
            combined_returns = pd.concat([stock_returns, market_returns], axis=1, join='inner').loc[:date]
            combined_returns = combined_returns.iloc[-self.lookback_period:]


            if len(combined_returns) < self.lookback_period:
                continue


            recent_stock_returns = combined_returns.iloc[:, 0]
            recent_market_returns = combined_returns.iloc[:, 1]


            ivol = self.calculate_ivol(recent_stock_returns, recent_market_returns)
            ivol_values[ticker] = ivol


        return ivol_values


    def generate_orders_for_date(self, ivol_values, date):
        ivol_series = pd.Series(ivol_values)
        ivol_series = ivol_series.dropna()
        sorted_ivol = ivol_series.sort_values()


        num_stocks = len(sorted_ivol)
        decile_size = max(int(num_stocks * 0.1), 1)
        low_ivol_tickers = sorted_ivol.head(decile_size).index.tolist()
        high_ivol_tickers = sorted_ivol.tail(decile_size).index.tolist()


        avg_low_ivol = ivol_series[low_ivol_tickers].mean()
        avg_high_ivol = ivol_series[high_ivol_tickers].mean()


        # Ensure IVOL neutrality with equal weighting
        scaling_factor = 0.5  # Can use to reduce leverage
        total_low_ivol_weight = scaling_factor / (avg_low_ivol + avg_high_ivol)
        total_high_ivol_weight = -scaling_factor / (avg_low_ivol + avg_high_ivol)
        orders = []


        # Long low IVOL, short high IVOL stocks
        for ticker in low_ivol_tickers:
            quantity = int(100 * total_low_ivol_weight / decile_size)
            max_allocation = self.starting_portfolio_value * 0.01 
            quantity = min(quantity, max_allocation)
            orders.append({
                "date": date,
                "type": "BUY",
                "ticker": ticker,
                "quantity": quantity  
            })
            print(f"Buying {ticker} on {date}")


        for ticker in high_ivol_tickers:
            quantity = int(100 * total_high_ivol_weight / decile_size)
            max_allocation = self.starting_portfolio_value * 0.01
            quantity = min(quantity, max_allocation)
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
            raise ValueError("SPY data is required for IVOL calculation.")
        spy_returns = spy_data['Adj Close'].pct_change()
        spy_returns = spy_returns.dropna()


        start_date = spy_returns.index[self.lookback_period]
        end_date = spy_returns.index[-1]
        rebalance_dates = pd.date_range(start=start_date, end=end_date, freq=self.rebalance_frequency)


        all_orders = []


        for date in rebalance_dates:
            ivol_values = self.calculate_ivols(data, spy_returns, date)
            if len(ivol_values) < 20:
                continue
            orders = self.generate_orders_for_date(ivol_values, date)
            all_orders.extend(orders)


        return all_orders
