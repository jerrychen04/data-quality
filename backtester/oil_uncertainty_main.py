"""
Oil Uncertainty Strategy Backtest

Backtest for Oil Uncertainty Index Strategy
"""

import pickle
import pandas as pd
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import os

from oil_uncertainty import OilUncertaintyOrderGenerator
from backtest_engine import EquityBacktestEngine
from metrics import ExtendedMetrics

# OPU index data
oil_data = pd.read_csv('data/oil.csv')
oil_data['date'] = pd.to_datetime(oil_data['date'], format='%m/%d/%Y')

sp500_data = {}
with open('data/sp500_data.pkl', 'rb') as f:
    sp500_data = pickle.load(f)
    
start_date = oil_data['date'].min().strftime('%Y-%m-%d')
end_date = oil_data['date'].max().strftime('%Y-%m-%d')

# load SPY data
spy_df = yf.download("SPY", start=start_date, end=end_date, auto_adjust=False)
# keep only the needed columns
spy_df = spy_df[['Adj Close', 'Volume']].copy()
sp500_data['SPY'] = spy_df
spy_data = sp500_data['SPY'].copy()
spy_data.index = pd.to_datetime(spy_data.index)
spy_data = spy_data.sort_index()

# VISUALIZTIONS
oil_plot_data = oil_data.copy()
oil_plot_data.set_index('date', inplace=True)

plt.figure(figsize=(16, 8))
plt.plot(oil_plot_data.index, oil_plot_data['OPU_index'], linewidth=2)
plt.title('Oil Price Uncertainty (OPU) Index Over Time', fontsize=16)
plt.xlabel('Date', fontsize=14)
plt.ylabel('OPU Index', fontsize=14)
plt.grid(True, alpha=0.3)

# major oil-related events
events = {
    '1973-10-15': 'OPEC Oil Embargo',
    '1979-01-15': 'Iranian Revolution',
    '1990-08-02': 'Gulf War',
    '2008-07-11': 'Oil Price Peak',
    '2014-11-01': 'Oil Price Collapse',
    '2020-03-09': 'COVID-19 Oil Crash'
}

for date, event in events.items():
    date_obj = pd.to_datetime(date)
    if date_obj >= oil_plot_data.index.min() and date_obj <= oil_plot_data.index.max():
        closest_date = oil_plot_data.index[oil_plot_data.index.get_indexer([date_obj], method='nearest')[0]]
        value = oil_plot_data.loc[closest_date, 'OPU_index']
        plt.annotate(event, 
                    xy=(closest_date, value),
                    xytext=(closest_date, value + 50),
                    arrowprops=dict(facecolor='black', shrink=0.05, width=1.5),
                    fontsize=12)

plt.tight_layout()
plt.savefig('opu_index_over_time.png')
plt.close()

oil_plot_data['12M_rolling_mean'] = oil_plot_data['OPU_index'].rolling(window=12).mean()
oil_plot_data['12M_rolling_std'] = oil_plot_data['OPU_index'].rolling(window=12).std()

plt.figure(figsize=(16, 10))

plt.subplot(2, 1, 1)
plt.plot(oil_plot_data.index, oil_plot_data['OPU_index'], label='OPU Index', linewidth=1, alpha=0.7)
plt.plot(oil_plot_data.index, oil_plot_data['12M_rolling_mean'], label='12-Month Rolling Mean', linewidth=2)
plt.title('OPU Index with 12-Month Rolling Mean', fontsize=16)
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(2, 1, 2)
plt.plot(oil_plot_data.index, oil_plot_data['12M_rolling_std'], label='12-Month Rolling Volatility', linewidth=2, color='orange')
plt.title('OPU Index Volatility (12-Month Rolling Standard Deviation)', fontsize=16)
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('opu_rolling_statistics.png')
plt.close()

# TRADING STRATEGY
# Set up the strategy with parameters
oil_strategy = OilUncertaintyOrderGenerator(
    lookback_period=30,              # 30 days lookback for analyzing OPU trends
    rebalance_frequency='MS',        # Monthly rebalance on start of month
    starting_portfolio_value=100000, # Initial portfolio value
    high_uncertainty_threshold=80,   # 80th percentile threshold for high uncertainty
    low_uncertainty_threshold=20,    # 20th percentile threshold for low uncertainty
    trend_period=10,                 # 10-day period for trend confirmation
    position_hold_days=20            # Hold positions for 20 days
)

orders = oil_strategy.generate_orders(sp500_data, oil_csv_path='data/oil.csv')
print(f"Generated {len(orders)} orders.")
orders_df = pd.DataFrame(orders)

buy_orders = orders_df[orders_df['type'] == 'BUY']
sell_orders = orders_df[orders_df['type'] == 'SELL']

# print(f"\nBuy orders: {len(buy_orders)} ({len(buy_orders) / len(orders_df) * 100:.1f}%)")
# print(f"Sell orders: {len(sell_orders)} ({len(sell_orders) / len(orders_df) * 100:.1f}%)")

# VISUALIZE TRADES
plt.figure(figsize=(16, 8))

# OPU index as background
plt.plot(oil_plot_data.index, oil_plot_data['OPU_index'], label='OPU Index', alpha=0.3, color='gray')

for idx, order in orders_df.iterrows():
    color = 'green' if order['type'] == 'BUY' else 'red'
    plt.scatter(order['date'], order['price'], color=color, s=100, alpha=0.7)
    
from matplotlib.lines import Line2D
custom_lines = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=10),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10),
    Line2D([0], [0], color='gray', alpha=0.3)
]
plt.legend(custom_lines, ['Buy Signal', 'Sell Signal', 'OPU Index'], loc='upper left')

plt.title('Trading Signals Generated by Oil Uncertainty Strategy', fontsize=16)
plt.xlabel('Date', fontsize=14)
plt.ylabel('SPY Price / OPU Index (Scaled)', fontsize=14)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('trading_signals.png')
plt.close()

# BACKTEST
price_df = pd.DataFrame(spy_data['Adj Close'])
price_df.columns = ['SPY']

initial_cash = 100000
engine = EquityBacktestEngine(initial_cash=initial_cash)

backtest_result = engine.run_backtest(orders, price_df)
portfolio_values_df = backtest_result['portfolio_values']

portfolio_returns = portfolio_values_df['Portfolio Value'].pct_change().dropna()

spy_prices = spy_data['Adj Close']
spy_returns = spy_prices.pct_change().dropna()

common_dates = portfolio_returns.index.intersection(spy_returns.index)
portfolio_returns = portfolio_returns.loc[common_dates]
spy_returns = spy_returns.loc[common_dates]

print(f"Initial portfolio value: ${initial_cash:,.2f}")
print(f"Final portfolio value: ${portfolio_values_df['Portfolio Value'].iloc[-1]:,.2f}")
print(f"Total return: {portfolio_values_df['Portfolio Value'].iloc[-1] / initial_cash - 1:.2%}")

# plot equity curve
plt.figure(figsize=(16, 8))

# plot portfolio value
plt.plot(portfolio_values_df.index, portfolio_values_df['Portfolio Value'], 
         label='Oil Uncertainty Strategy', linewidth=2)

# create a benchmark series normalized to same starting value
start_date = portfolio_values_df.index[0]
benchmark_series = spy_prices.loc[spy_prices.index >= start_date].copy()
benchmark_series = benchmark_series / benchmark_series.iloc[0] * initial_cash

plt.plot(benchmark_series.index, benchmark_series, 
         label='SPY (Benchmark)', linewidth=2, alpha=0.7)

plt.title('Oil Uncertainty Strategy vs SPY Benchmark', fontsize=16)
plt.xlabel('Date', fontsize=14)
plt.ylabel('Portfolio Value ($)', fontsize=14)
plt.grid(True, alpha=0.3)
plt.legend(fontsize=12)
plt.tight_layout()
plt.savefig('equity_curve.png')
plt.close()

metrics_calculator = ExtendedMetrics()
metrics = metrics_calculator.calculate(
    portfolio_values_df['Portfolio Value'],
    portfolio_returns,
    benchmark_returns=spy_returns
)

print("\nPerformance Metrics:")
for key, value in metrics.items():
    print(f"{key}: {value}")