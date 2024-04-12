from collections import deque, namedtuple
import yfinance as yf
import pandas as pd
import torch
import random
import matplotlib.pyplot as plt
import numpy as np

def return_risk_metric(daily_portfolio) -> float:
    """
    Calculates the Sharpe ratio for a given portfolio, assuming a risk-free rate of zero.

    Args:
        daily_portfolio_values (pandas.Series or numpy.ndarray): Daily portfolio values.

    Returns:
        tuple: (annualized_return, annualized_std)
    """

    num_trading_days = 252 # Assume 252 trading days in a year

    # Convert numpy array to pandas series (making use of pandas' pct_change method)
    if isinstance(daily_portfolio, np.ndarray):
        daily_portfolio = pd.Series(daily_portfolio)

    # Calculate daily returns
    daily_returns = daily_portfolio.pct_change().dropna() 

    # Calculate annualized return
    holding_period_return = (daily_portfolio.iloc[-1] / daily_portfolio.iloc[0]) - 1
    annualized_return = (1 + holding_period_return) ** (num_trading_days / len(daily_portfolio)) - 1

    # Calculate annualized standard deviation (risk)
    annualized_std = daily_returns.std() * np.sqrt(num_trading_days)

    return annualized_return / annualized_std


def maximum_drawdown(portfolio: list) -> float:
    """
    Calculates the maximum drawdown for a given portfolio.

    Args:
    - portfolio (list): List containing the portfolio values.

    Returns:
    - float: Maximum drawdown for the given portfolio (as a decimal)
    """
    max_drawdown = 0
    peak = portfolio[0] # Initial peak value

    # Iterate through the portfolio values
    for balance in portfolio:

        # Update peak value if balance is greater than current peak
        if balance > peak:
            peak = balance 
        
        # Calculate drawdown
        dd = (balance - peak) / peak
        
        # Update maximum drawdown if current drawdown is less than max drawdown
        if dd < max_drawdown:
            max_drawdown = dd
    
    return -max_drawdown