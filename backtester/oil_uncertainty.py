"""
Strategy that uses oil price uncertainty index to predict SPY movements.

1. Uses the Oil Price Uncertainty (OPU) index to identify potential market stress periods
2. Applies different thresholds to find high, moderate, and low uncertainty periods
3. Takes long/short positions in SPY based on these uncertainty levels and recent price trends
"""

from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
from datetime import datetime
from order_generator import OrderGenerator

class OilUncertaintyOrderGenerator(OrderGenerator):
    
    def __init__(
        self, 
        lookback_period: int = 30, 
        rebalance_frequency: str = 'MS', 
        starting_portfolio_value: float = 100000,
        high_uncertainty_threshold: float = 90,
        low_uncertainty_threshold: float = 20,
        trend_period: int = 10,
        position_hold_days: int = 20
    ):
        """
        Args:
            lookback_period: Number of days to look back for analyzing OPU index trends
            rebalance_frequency: Frequency to rebalance the portfolio ('MS' for month start)
            starting_portfolio_value: Initial portfolio value
            high_uncertainty_threshold: Percentile threshold for high uncertainty (0-100)
            low_uncertainty_threshold: Percentile threshold for low uncertainty (0-100)
            trend_period: Days to analyze for price trend confirmation
            position_hold_days: Number of days to hold positions
        """
        self.lookback_period = lookback_period
        self.rebalance_frequency = rebalance_frequency
        self.starting_portfolio_value = starting_portfolio_value
        self.high_uncertainty_threshold = high_uncertainty_threshold
        self.low_uncertainty_threshold = low_uncertainty_threshold
        self.trend_period = trend_period
        self.position_hold_days = position_hold_days
        
    def _load_oil_uncertainty_data(self, oil_csv_path: str) -> pd.DataFrame:
        """ 
        Returns preprocessed dataframe with datetime index (change to date and sort)
        """
        oil_data = pd.read_csv(oil_csv_path)
        oil_data['date'] = pd.to_datetime(oil_data['date'], format='%m/%d/%Y')
        oil_data.set_index('date', inplace=True)
        oil_data.sort_index(inplace=True)
        
        oil_data['opu_rolling_mean'] = oil_data['OPU_index'].rolling(window=self.lookback_period).mean()
        oil_data['opu_rolling_std'] = oil_data['OPU_index'].rolling(window=self.lookback_period).std()
        
        # percentile ranks
        oil_data['opu_percentile'] = oil_data['OPU_index'].rolling(window=365).rank(pct=True) * 100
        
        # rate of change
        oil_data['opu_1m_change'] = oil_data['OPU_index'].pct_change(periods=30)
        oil_data['opu_3m_change'] = oil_data['OPU_index'].pct_change(periods=90)
        
        return oil_data
    
    def _align_data_dates(self, spy_data: pd.DataFrame, oil_data: pd.DataFrame) -> tuple:
        if not isinstance(spy_data.index, pd.DatetimeIndex):
            spy_data.index = pd.to_datetime(spy_data.index)
        
        common_dates = spy_data.index.intersection(oil_data.index)
        
        return spy_data.loc[common_dates], oil_data.loc[common_dates]
    
    def _get_trading_signal(
        self, 
        current_date: pd.Timestamp, 
        spy_data: pd.DataFrame, 
        oil_data: pd.DataFrame
    ) -> Optional[str]:
        """
        Determine trading signal based on oil uncertainty and SPY trend.
        
        Args:
            current_date: Current trading date
            spy_data: SPY price dataframe
            oil_data: Oil uncertainty dataframe
            
        Returns:
            Signal 'BUY', 'SELL', or None for no action
        """
        try:
            oil_up_to_date = oil_data.loc[:current_date]
            spy_up_to_date = spy_data.loc[:current_date]
            
            if len(oil_up_to_date) < self.lookback_period or len(spy_up_to_date) < self.trend_period:
                return None
            
            current_opu = float(oil_up_to_date['OPU_index'].iloc[-1])
            current_percentile = float(oil_up_to_date['opu_percentile'].iloc[-1])
            opu_1m_change = float(oil_up_to_date['opu_1m_change'].iloc[-1])
            
            spy_price = float(spy_up_to_date['Adj Close'].iloc[-1])
            spy_ma = float(spy_up_to_date['Adj Close'].rolling(window=self.trend_period).mean().iloc[-1])
            
            spy_trend = float(spy_up_to_date['Adj Close'].pct_change(periods=self.trend_period).iloc[-1])
            
            if pd.isna(current_percentile) or pd.isna(opu_1m_change):
                return None
                
            # High uncertainty conditions (likely market stress)
            if current_percentile > self.high_uncertainty_threshold:
                # Sell if one of these conditions is true:
                # 1. Oil uncertainty is rising sharply
                # 2. SPY is below its moving average (confirming downtrend)
                if opu_1m_change > 0.15 or spy_price < spy_ma:
                    return "SELL"
                    
            # Low uncertainty conditions (likely stable markets)
            elif current_percentile < self.low_uncertainty_threshold:
                # Buy if one of these conditions is true:
                # 1. Oil uncertainty is decreasing
                # 2. SPY is above its moving average (confirming uptrend)
                if opu_1m_change < 0 or spy_price > spy_ma:
                    return "BUY"
            
            # Moderate uncertainty - look for clear directional signals
            else:
                # Check if SPY trend is strong in either direction
                if spy_trend > 0.03 and spy_price > spy_ma:  # Strong uptrend
                    return "BUY"
                elif spy_trend < -0.03 and spy_price < spy_ma:  # Strong downtrend
                    return "SELL"
                    
            return None
            
        except (IndexError, KeyError) as e:
            print(f"Error generating signal for {current_date}: {e}")
            return None
        except Exception as e:
            print(f"Unexpected error for {current_date}: {e}")
            return None
    
    def _calculate_position_size(self, spy_price: float, signal: str) -> int:
        """
        Calculate position size based on portfolio value and SPY price.
        
        Args:
            spy_price: Current SPY price
            signal: Trading signal ("BUY" or "SELL")
            
        Returns:
            Quantity of shares to trade
        """
        risk_percentage = 0.8 if signal == "BUY" else 0.5
        position_value = self.starting_portfolio_value * risk_percentage
        quantity = int(position_value / spy_price)
        
        return quantity
        
    def generate_orders(self, data: Dict[str, pd.DataFrame], oil_csv_path: str = 'oil.csv') -> List[Dict[str, Any]]:
        """
        Generate orders based on oil uncertainty and SPY price data.
        
        Args:
            data: Dictionary of dataframes with ticker symbols as keys
            oil_csv_path: Path to oil uncertainty CSV file
            
        Returns:
            List of order dictionaries
        """
        spy_data = data.get('SPY')
        if spy_data is None:
            raise ValueError("SPY data is required for this strategy.")
        
        oil_data = self._load_oil_uncertainty_data(oil_csv_path)
        spy_data, oil_data = self._align_data_dates(spy_data, oil_data)
        
        start_date = oil_data.index[self.lookback_period]
        end_date = oil_data.index[-1]
        
        trading_dates = pd.date_range(start=start_date, end=end_date, freq=self.rebalance_frequency)
        trading_dates = trading_dates.intersection(spy_data.index)
        
        all_orders = []
        current_position = None
        
        for date in trading_dates:
            if date not in spy_data.index or date not in oil_data.index:
                continue
                
            signal = self._get_trading_signal(date, spy_data, oil_data)
            if signal is None or (current_position == signal):
                continue
                
            if current_position is not None and current_position != signal:
                current_position = None
                
            if signal:
                current_position = signal
                spy_price = float(spy_data.loc[date, 'Adj Close'])
                quantity = self._calculate_position_size(spy_price, signal)
                
                order = {
                    "date": date,
                    "type": signal,
                    "ticker": "SPY",
                    "quantity": quantity,
                    "price": spy_price,
                    "reason": f"Oil Uncertainty: {float(oil_data.loc[date, 'OPU_index']):.2f}, Percentile: {float(oil_data.loc[date, 'opu_percentile']):.2f}%"
                }
                
                all_orders.append(order)
                
        return all_orders