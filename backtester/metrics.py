from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from typing import Dict
import matplotlib.pyplot as plt

class Metrics(ABC):

    @abstractmethod
    def calculate(self, portfolio_values: pd.Series, returns: pd.Series, weights: pd.DataFrame = None, benchmark_returns: pd.Series = None) -> Dict[str, float]:
        pass


class ExtendedMetrics(Metrics):

    def calculate(self, portfolio_values: pd.Series, returns: pd.Series, weights: pd.DataFrame = None, benchmark_returns: pd.Series = None) -> Dict[str, float]:
        metrics = {}
        if returns.empty:
            print("Warning: Empty returns series, cannot calculate metrics.")
            return {}

        metrics['Cumulative Return (%)'] = ((portfolio_values.iloc[-1] / portfolio_values.iloc[0]) - 1) * 100
        metrics['Annualized Return (%)'] = ((1 + returns.mean())**252 - 1) * 100
        metrics['Annualized Volatility (%)'] = returns.std() * np.sqrt(252) * 100

        risk_free_rate = 0.0045 # Example annual risk-free rate
        daily_risk_free = risk_free_rate / 252
        excess_returns = returns - daily_risk_free

        # Handle potential division by zero if std dev is zero
        if excess_returns.std() > 1e-8:
             metrics['Sharpe Ratio'] = (excess_returns.mean() / excess_returns.std()) * np.sqrt(252)
        else:
             metrics['Sharpe Ratio'] = np.nan # Or 0, depending on desired handling


        running_max = portfolio_values.cummax()
        drawdown = (portfolio_values / running_max) - 1
        metrics['Max Drawdown (%)'] = drawdown.min() * 100

        metrics['VaR 5% Daily (%)'] = returns.quantile(0.05) * 100
        metrics['CVaR 5% Daily (%)'] = returns[returns <= returns.quantile(0.05)].mean() * 100

        if weights is not None:
             weight_diff = abs(weights - weights.shift(1))
             daily_turnover = 0.5 * weight_diff.sum(axis=1)
             metrics['Average Daily Turnover (%)'] = daily_turnover.mean() * 100

        return metrics

    def plot_returns(self, portfolio_values: pd.Series, title: str = "Portfolio Value Over Time"):
        plt.figure(figsize=(12, 7))
        portfolio_values.plot()
        plt.title(title)
        plt.xlabel("Date")
        plt.ylabel("Portfolio Value ($)")
        # plt.yscale('log') # Often useful for long periods
        plt.grid(True, which="both", ls="--", linewidth=0.5)
        plt.show()