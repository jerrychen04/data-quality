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
    
class StableMinusRiskyOrderGenerator:
    def __init__(self, lookback_period = 60, rebalance_frequency = 'ME', starting_portfolio_value = 100000):
        self.lookback_period = lookback_period
        self.rebalance_frequency = rebalance_frequency
        self.starting_portfolio_value = starting_portfolio_value

    def calculate_predicted_returns(self, data, date):
        predicted_returns = {}
        for ticker, df in data.items():
            if len(df) < self.lookback_period:
                continue
            if df.index[-1] < date:
                continue
            df_up_to_date = df.loc[:date]
            if len(df_up_to_date) < self.lookback_period:
                continue
            recent = df_up_to_date.iloc[-self.lookback_period:]
            daily_returns = recent['Adj Close'].pct_change(fill_method=None).dropna()
            if len(daily_returns) > 0:
                avg_daily_return = daily_returns.mean()
                if isinstance(avg_daily_return, pd.Series):
                    avg_daily_return = avg_daily_return.iloc[0]

                ann_return = avg_daily_return * 252
                predicted_returns[ticker] = ann_return
        return predicted_returns

    def generate_orders_for_date(self, predicted_returns, date):
        pr_series = pd.Series(predicted_returns).dropna().astype(float)
        if pr_series.empty:
            return []
        sorted_by_pr = pr_series.sort_values()
        num_stocks = len(sorted_by_pr)
        decile_size = max(int(num_stocks * 0.1), 1)
        risky_tickers = sorted_by_pr.head(decile_size).index.tolist()
        stable_tickers = sorted_by_pr.tail(decile_size).index.tolist()
        half_capital = self.starting_portfolio_value / 2.0
        stable_allocation_per_ticker = half_capital / decile_size
        risky_allocation_per_ticker = half_capital / decile_size
        orders = []
        for ticker in stable_tickers:
            quantity = int(stable_allocation_per_ticker)
            if quantity > 0:
                orders.append({
                    "date": date,
                    "type": "BUY",
                    "ticker": ticker,
                    "quantity": quantity
                })
                print(f"Buying {ticker} on {date}")
        for ticker in risky_tickers:
            quantity = int(risky_allocation_per_ticker)
            if quantity > 0:
                orders.append({
                    "date": date,
                    "type": "SELL",
                    "ticker": ticker,
                    "quantity": quantity
                })
                print(f"Selling {ticker} on {date}")
        return orders

    def generate_orders(self, data: Dict[str, pd.DataFrame]) -> List[Dict[str, Any]]:
        if 'SPY' not in data:
            raise ValueError("SPY data is required.")
        spy_data = data['SPY']
        spy_dates = spy_data.index
        if len(spy_dates) <= self.lookback_period:
            raise ValueError("Not enough SPY data.")
        start_date = spy_dates[self.lookback_period]
        end_date = spy_dates[-1]
        rebalance_dates = pd.date_range(start=start_date, end=end_date, freq=self.rebalance_frequency)
        all_orders = []
        for date in rebalance_dates:
            pr_values = self.calculate_predicted_returns(data, date)
            if len(pr_values) < 20:
                continue
            orders = self.generate_orders_for_date(pr_values, date)
            all_orders.extend(orders)
        return all_orders