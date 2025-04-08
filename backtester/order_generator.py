from abc import ABC, abstractmethod
import pandas as pd
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
    

class StableMinusRiskyOrderGenerator(OrderGenerator):
    """Strategy that goes long on low volatility stocks and short on high volatility stocks."""
    
    def __init__(self, lookback_period: int = 60, rebalance_frequency: str = 'ME', starting_portfolio_value: float = 100000):
        self.lookback_period = lookback_period
        self.rebalance_frequency = rebalance_frequency
        self.starting_portfolio_value = starting_portfolio_value
    
    def calculate_volatility(self, returns: pd.Series) -> float:
        """
        Calculate volatility (standard deviation) of returns.
        """
        return returns.std()

    def calculate_volatilities(self, data, date):
        """
        Calculate volatility for each stock up to the given date.
        """
        volatility_values = {}
        for ticker, df in data.items():
            if ticker == 'SPY':
                continue

            if 'Adj Close' not in df.columns:
                continue
                
            stock_returns = df['Adj Close'].pct_change(fill_method=None).dropna()
            
            # Filter returns up to the current date
            if date not in stock_returns.index:
                continue
                
            mask = stock_returns.index <= date
            recent_returns = stock_returns.loc[mask]
            
            if len(recent_returns) < self.lookback_period:
                continue
                
            recent_returns = recent_returns.iloc[-self.lookback_period:]
            volatility = self.calculate_volatility(recent_returns)
            
            # Only add non-NA volatility values
            if not pd.isna(volatility):
                volatility_values[ticker] = volatility
            
        return volatility_values

    def generate_orders_for_date(self, volatility_values, date):
        """
        Generate orders for a specific date based on volatility values.
        """
        if not volatility_values:
            return []
            
        volatility_series = pd.Series(volatility_values)
        volatility_series = volatility_series.dropna()
        
        if len(volatility_series) == 0:
            return []
            
        sorted_volatility = volatility_series.sort_values()
        
        num_stocks = len(sorted_volatility)
        decile_size = max(int(num_stocks * 0.1), 1)
        
        low_vol_tickers = sorted_volatility.head(decile_size).index.tolist()
        high_vol_tickers = sorted_volatility.tail(decile_size).index.tolist()
        
        if not low_vol_tickers or not high_vol_tickers:
            return []
        
        avg_low_vol = float(volatility_series[low_vol_tickers].mean())
        avg_high_vol = float(volatility_series[high_vol_tickers].mean())

        if avg_low_vol == 0 or avg_high_vol == 0 or (avg_low_vol + avg_high_vol) == 0:
            return []
        
        # find weights to make it beta neutral
        low_vol_weight = avg_high_vol / (avg_low_vol + avg_high_vol)
        high_vol_weight = avg_low_vol / (avg_low_vol + avg_high_vol)
        
        orders = []
        
        # Long bottom decile (stable stocks)
        for ticker in low_vol_tickers:
            quantity = int(self.starting_portfolio_value * low_vol_weight / len(low_vol_tickers))
            orders.append({
                "date": date,
                "type": "BUY",
                "ticker": ticker,
                "quantity": quantity
            })
            print(f"Buying {ticker} on {date}")
        
        # Short top decile (risky stocks)
        for ticker in high_vol_tickers:
            quantity = int(self.starting_portfolio_value * high_vol_weight / len(high_vol_tickers))
            orders.append({
                "date": date,
                "type": "SELL",
                "ticker": ticker,
                "quantity": quantity
            })
            print(f"Selling {ticker} on {date}")
            
        return orders

    def generate_orders(self, data: Dict[str, pd.DataFrame]) -> List[Dict[str, Any]]:
        """
        Generate orders based on the stable-minus-risky strategy.
        """
        # Choose a reference ticker to determine date range
        reference_ticker = next(iter(data))
        while reference_ticker == 'SPY' and len(data) > 1:
            reference_ticker = next(iter([t for t in data.keys() if t != 'SPY']))
            
        reference_data = data[reference_ticker]['Adj Close']
        
        # Calculate returns for date range determination
        returns = reference_data.pct_change().dropna()
        
        if len(returns) <= self.lookback_period:
            return []
            
        start_date = returns.index[self.lookback_period]
        end_date = returns.index[-1]
        
        # Define rebalance dates
        rebalance_dates = pd.date_range(start=start_date, end=end_date, freq=self.rebalance_frequency)
        
        # Filter to only dates that exist in our data
        valid_dates = [date for date in rebalance_dates if date in returns.index]
        
        all_orders = []
        
        for date in valid_dates:
            volatility_values = self.calculate_volatilities(data, date)
            
            # Ensure we have enough stocks to perform the strategy
            if len(volatility_values) < 20:
                continue
                
            date_orders = self.generate_orders_for_date(volatility_values, date)
            all_orders.extend(date_orders)
            
        return all_orders