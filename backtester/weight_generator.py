import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from scipy import stats

class WeightsGenerator(ABC):
    @abstractmethod
    def generate_weights(self, data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        pass

class BettingAgainstBetaWeightsGenerator(WeightsGenerator):
    """Betting Against Beta (BAB) strategy implementation that generates portfolio weights."""
    
    def __init__(self, lookback_period: int = 60, rebalance_frequency: str = 'ME'):
        self.lookback_period = lookback_period
        self.rebalance_frequency = rebalance_frequency
    
    def calculate_beta(self, stock_returns: pd.Series, market_returns: pd.Series) -> float:
        """
        Calculate beta using scipy's more numerically stable least squares implementation.
        """
        try:          
            valid_data = pd.concat([stock_returns, market_returns], axis=1).dropna()
            
            if len(valid_data) < 10:
                return np.nan
                
            stock_returns = valid_data.iloc[:, 0].values
            market_returns = valid_data.iloc[:, 1].values
            
            X = np.column_stack([np.ones(len(market_returns)), market_returns])
            result = stats.linregress(market_returns, stock_returns)
            beta = result.slope
            if abs(beta) > 5:
                return np.nan
                
            return beta
            
        except Exception as e:
            print(f"Error calculating beta: {e}")
            return np.nan

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

    def generate_weights_for_date(self, beta_values, date):
        """Generate portfolio weights based on beta values for a specific date."""
        beta_series = pd.Series(beta_values)
        beta_series = beta_series.dropna()
        sorted_beta = beta_series.sort_values()

        num_stocks = len(sorted_beta)
        decile_size = max(int(num_stocks * 0.1), 1)
        low_beta_tickers = sorted_beta.head(decile_size).index.tolist()
        high_beta_tickers = sorted_beta.tail(decile_size).index.tolist()

        weights = {ticker: 0 for ticker in beta_values.keys()}
        
        avg_low_beta = np.mean([beta_values[ticker] for ticker in low_beta_tickers])
        avg_high_beta = np.mean([beta_values[ticker] for ticker in high_beta_tickers])
        
        # this is dollar neutral
        long_dollar_allocation = 1
        short_dollar_allocation = -1
        
        # The goal is to have: long_dollar_allocation * avg_low_beta + short_dollar_allocation * avg_high_beta = 0
        # adjust the weights within each group while maintaining the 50/50 split
        
        long_beta_scaling = abs(avg_high_beta) / (abs(avg_low_beta) + abs(avg_high_beta))
        short_beta_scaling = abs(avg_low_beta) / (abs(avg_low_beta) + abs(avg_high_beta))
        
        # beta AND dollar neutral here
        for ticker in low_beta_tickers:
            weights[ticker] = long_dollar_allocation * long_beta_scaling / len(low_beta_tickers)
            
        for ticker in high_beta_tickers:
            weights[ticker] = short_dollar_allocation * short_beta_scaling / len(high_beta_tickers)
        
        # weights for observability
        long_weight = sum(w for w in weights.values() if w > 0)
        short_weight = sum(w for w in weights.values() if w < 0)
        long_beta = sum(weights[t] * beta_values[t] for t in low_beta_tickers if weights[t] > 0)
        short_beta = sum(weights[t] * beta_values[t] for t in high_beta_tickers if weights[t] < 0)
        portfolio_beta = long_beta + short_beta
        
        print(f"Rebalancing weights on {date}:")
        print(f"  Long: {long_weight:.2f}, Short: {abs(short_weight):.2f}")
        print(f"  Long beta: {long_beta:.2f}, Short beta: {short_beta:.2f}, Net beta: {portfolio_beta:.4f}")
        return pd.Series(weights, name=date)

    def generate_weights(self, data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Generate portfolio weights dataframe for all rebalance dates."""
        spy_data = data.get('SPY')
        if spy_data is None:
            raise ValueError("SPY data is required for beta calculation.")
        
        spy_returns = spy_data['Adj Close'].pct_change()
        spy_returns = spy_returns.dropna()

        start_date = spy_returns.index[self.lookback_period]
        end_date = spy_returns.index[-1]
        rebalance_dates = pd.date_range(start=start_date, end=end_date, freq=self.rebalance_frequency)

        all_weights = []

        for date in rebalance_dates:
            if date not in spy_returns.index:
                continue
                
            beta_values = self.calculate_betas(data, spy_returns, date)
            if len(beta_values) < 20:
                continue
                
            weights = self.generate_weights_for_date(beta_values, date)
            all_weights.append(weights)

        if not all_weights:
            return pd.DataFrame()
            
        weights_df = pd.concat(all_weights, axis=1).T
        
        # fill forward weights for all trading days because not always rebalancing
        all_trading_days = pd.date_range(start=weights_df.index.min(), end=weights_df.index.max())
        trading_days_idx = all_trading_days[all_trading_days.isin(spy_returns.index)]
        
        full_weights_df = weights_df.reindex(trading_days_idx).ffill()
        return full_weights_df
    

class AlphaWeightGenerator(WeightsGenerator):
    
    def __init__(self, lookback_period: int = 60, rebalance_frequency: str = 'ME', 
                 skew_formation_window: int = 5, skew_weight: float = 0.0):
        """
        Initialize the strategy with parameters.
        
        Args:
            lookback_period: Lookback period for beta calculation in days
            rebalance_frequency: Frequency to rebalance the paortfolio
            skew_formation_window: Rolling window size for skewness calculation
            skew_weight: Weight allocation for skewness strategy (0.0 to 1.0)
                         0.0 = 100% BAB strategy, 1.0 = 100% skewness strategy
        """
        self.lookback_period = lookback_period
        self.rebalance_frequency = rebalance_frequency
        self.skew_formation_window = skew_formation_window
        self.skew_weight = max(0.0, min(1.0, skew_weight))  # Ensure between 0 and 1
        
    def calculate_beta(self, stock_returns: pd.Series, market_returns: pd.Series) -> float:
        """
        Calculate beta using scipy's more numerically stable least squares implementation.
        """
        try:          
            valid_data = pd.concat([stock_returns, market_returns], axis=1).dropna()
            
            if len(valid_data) < 10:
                return np.nan
                
            stock_returns = valid_data.iloc[:, 0].values
            market_returns = valid_data.iloc[:, 1].values
            
            # Add constant term for intercept
            X = np.column_stack([np.ones(len(market_returns)), market_returns])
            result = stats.linregress(market_returns, stock_returns)
            beta = result.slope
            if abs(beta) > 5:
                return np.nan
                
            return beta
            
        except Exception as e:
            print(f"Error calculating beta: {e}")
            return np.nan

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
    
    def calculate_skewness_weights(self, data, date):
        """Generate weights based on rolling skewness."""
        returns_dict = {}
        for ticker, df in data.items():
            if ticker == 'SPY':
                continue
            returns = df['Adj Close'].pct_change(fill_method=None).dropna()
            if len(returns) > self.skew_formation_window:
                returns_dict[ticker] = returns
        
        returns_df = pd.DataFrame(returns_dict)
        returns_df = returns_df.loc[:date]
        
        shifted_returns = returns_df.shift(1)  # use previous day returns to avoid look-ahead bias
        rolling_skew = shifted_returns.rolling(window=self.skew_formation_window).skew()
        
        skew_date = rolling_skew.index[rolling_skew.index <= date][-1]
        current_skew = rolling_skew.loc[skew_date].dropna()
        
        n = len(current_skew)
        group_size = max(1, int(0.05 * n))  # 5% in each extreme
        
        ranked_skew = current_skew.sort_values()
        long_skew = ranked_skew.index[-group_size:].tolist()
        short_skew = ranked_skew.index[:group_size].tolist()
        
        weights = pd.Series(0, index=current_skew.index)
        weights.loc[long_skew] = 1.0 / group_size
        weights.loc[short_skew] = -1.0 / group_size
        all_weights = pd.Series(0, index=list(data.keys()))
        all_weights.update(weights)
        
        return all_weights
    
    def generate_weights_for_date(self, beta_values, date, data=None):
        """Generate portfolio weights based on beta values for a specific date,
        optionally combined with skewness-based weights."""
        beta_series = pd.Series(beta_values)
        beta_series = beta_series.dropna()
        sorted_beta = beta_series.sort_values()

        num_stocks = len(sorted_beta)
        decile_size = max(int(num_stocks * 0.1), 1)
        low_beta_tickers = sorted_beta.head(decile_size).index.tolist()
        high_beta_tickers = sorted_beta.tail(decile_size).index.tolist()

        weights = {ticker: 0 for ticker in beta_values.keys()}
        
        avg_low_beta = np.mean([beta_values[ticker] for ticker in low_beta_tickers])
        avg_high_beta = np.mean([beta_values[ticker] for ticker in high_beta_tickers])
        
        long_dollar_allocation = 1
        short_dollar_allocation = -1
        
        long_beta_scaling = abs(avg_high_beta) / (abs(avg_low_beta) + abs(avg_high_beta))
        short_beta_scaling = abs(avg_low_beta) / (abs(avg_low_beta) + abs(avg_high_beta))
        
        bab_weights = pd.Series(weights)
        for ticker in low_beta_tickers:
            bab_weights[ticker] = long_dollar_allocation * long_beta_scaling / len(low_beta_tickers)
            
        for ticker in high_beta_tickers:
            bab_weights[ticker] = short_dollar_allocation * short_beta_scaling / len(high_beta_tickers)
        
        # If skewness weight is 0, just return BAB weights
        if self.skew_weight == 0 or data is None:
            print("no skewness weight, using BAB weights")
            final_weights = bab_weights
        else:
            skew_weights = self.calculate_skewness_weights(data, date)
        
            combined_weights = bab_weights * (1 - self.skew_weight) + skew_weights * self.skew_weight
            
            long_positions = combined_weights > 0
            short_positions = combined_weights < 0
            
            if sum(long_positions) == 0 or sum(short_positions) == 0:
                print(f"Warning: Missing {'long' if sum(long_positions) == 0 else 'short'} positions after combining strategies")
                return bab_weights  
            
            final_weights = pd.Series(0, index=combined_weights.index)

            if sum(long_positions) > 0:
                long_sum = combined_weights[long_positions].sum()
                if long_sum > 0:
                    final_weights[long_positions] = combined_weights[long_positions] / long_sum
            
            if sum(short_positions) > 0:
                short_sum = abs(combined_weights[short_positions].sum())
                if short_sum > 0:
                    final_weights[short_positions] = -1.0 * abs(combined_weights[short_positions]) / short_sum
            
            valid_long_betas = [beta_values.get(t, 0) for t in final_weights.index if final_weights[t] > 0 and t in beta_values]
            valid_short_betas = [beta_values.get(t, 0) for t in final_weights.index if final_weights[t] < 0 and t in beta_values]
            
            if valid_long_betas and valid_short_betas:
                avg_long_beta = np.mean(valid_long_betas)
                avg_short_beta = np.mean(valid_short_betas)
                
                if avg_long_beta != 0 and avg_short_beta != 0:
                    beta_scale_long = abs(avg_short_beta) / (abs(avg_long_beta) + abs(avg_short_beta))
                    beta_scale_short = abs(avg_long_beta) / (abs(avg_long_beta) + abs(avg_short_beta))
                    final_weights[long_positions] *= beta_scale_long
                    final_weights[short_positions] *= beta_scale_short
        

        long_weight = sum(w for w in final_weights if w > 0)
        short_weight = sum(w for w in final_weights if w < 0)
        
        valid_indices = [t for t in final_weights.index if t in beta_values]
        long_beta_indices = [t for t in valid_indices if final_weights[t] > 0]
        short_beta_indices = [t for t in valid_indices if final_weights[t] < 0]
        
        long_beta = sum(final_weights[t] * beta_values[t] for t in long_beta_indices)
        short_beta = sum(final_weights[t] * beta_values[t] for t in short_beta_indices)
        portfolio_beta = long_beta + short_beta
        
        print(f"Rebalancing weights on {date}:")
        print(f"  Long: {long_weight:.2f}, Short: {abs(short_weight):.2f}")
        print(f"  Long beta: {long_beta:.2f}, Short beta: {short_beta:.2f}, Net beta: {portfolio_beta:.4f}")
        print(f"  Strategy mix: BAB {(1-self.skew_weight)*100:.0f}% / Skew {self.skew_weight*100:.0f}%")
        
        return pd.Series(final_weights, name=date)

    def generate_weights(self, data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Generate portfolio weights dataframe for all rebalance dates."""
        spy_data = data.get('SPY')
        if spy_data is None:
            raise ValueError("SPY data is required for beta calculation.")
        
        spy_returns = spy_data['Adj Close'].pct_change()
        spy_returns = spy_returns.dropna()

        start_date = spy_returns.index[self.lookback_period]
        end_date = spy_returns.index[-1]
        rebalance_dates = pd.date_range(start=start_date, end=end_date, freq=self.rebalance_frequency)

        all_weights = []

        for date in rebalance_dates:
            if date not in spy_returns.index:
                continue
                
            beta_values = self.calculate_betas(data, spy_returns, date)
            if len(beta_values) < 20:
                continue
                
            weights = self.generate_weights_for_date(beta_values, date, data)
            all_weights.append(weights)

        if not all_weights:
            return pd.DataFrame()
            
        weights_df = pd.concat(all_weights, axis=1).T
        
        # fill forward weights for all trading days because not always rebalancing
        all_trading_days = pd.date_range(start=weights_df.index.min(), end=weights_df.index.max())
        trading_days_idx = all_trading_days[all_trading_days.isin(spy_returns.index)]
        
        full_weights_df = weights_df.reindex(trading_days_idx).ffill()
        return full_weights_df