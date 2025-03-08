import pandas as pd
from data_source import YahooFinanceDataSource
from order_generator import MeanReversionOrderGenerator, BettingAgainstCorrelationOrderGenerator, BettingAgainstBetaOrderGenerator
from backtest_engine import EquityBacktestEngine
from metrics import ExtendedMetrics

# TODO: refactor into python notebooks, this is a MEAN REV demo of the backtester as a .py file
def main():
    """
    Example of using the backtester to backtest a mean reversion strategy on a portfolio of equities.
    """
    data_source = YahooFinanceDataSource()
    order_generator = BettingAgainstBetaOrderGenerator()
    backtest_engine = EquityBacktestEngine(initial_cash=100000)
    metrics_calculator = ExtendedMetrics()

    tickers = ["AAPL", "MSFT", "GOOGL", "SPY"]  
    data = data_source.get_historical_data(tickers, "2011-01-01", "2024-01-01")
    orders = order_generator.generate_orders(data)

    backtest_results = backtest_engine.run_backtest(orders, data)
    portfolio_values = backtest_results["portfolio_values"]["Portfolio Value"]
    returns = portfolio_values.pct_change().dropna()

    benchmark_data = data_source.get_historical_data("SPY", "2011-01-01", "2024-01-01")
    benchmark_returns = benchmark_data.pct_change().dropna()

    metrics = metrics_calculator.calculate(portfolio_values, returns, benchmark_returns)
    # Note: all values are annualized and assume 252 trading days in a year
    # Note: all returns are in fractional format. For example, 0.01 is 1% return
    print("Backtest Metrics:", metrics)

if __name__ == "__main__":
    main()