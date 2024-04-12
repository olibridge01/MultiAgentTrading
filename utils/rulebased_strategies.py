from collections import deque, namedtuple
import yfinance as yf
import pandas as pd
import torch
import random
import matplotlib.pyplot as plt
import numpy as np

def macd_trader(close_prices: pd.Series, 
                short_window: int = 12, 
                long_window: int = 26, 
                signal_window: int = 9, 
                initial_balance: int = 1000
    ) -> list:
    """
    MACD trading strategy. Generates a list of buy/sell signals based on close prices of a stock.

    Args:
    - close_prices (pd.Series): Series containing the close prices.
    - short_window (int): Short window for MACD calculation.
    - long_window (int): Long window for MACD calculation.
    - signal_window (int): Signal window for MACD calculation.
    - initial_balance (int): Initial balance for the portfolio.

    Returns:
    - pd.Series: Series containing the buy/sell signals.
    """

    # Compute MACD and signal line
    short_ema = close_prices.ewm(span=short_window, adjust=False).mean()
    long_ema = close_prices.ewm(span=long_window, adjust=False).mean()
    macd = short_ema - long_ema
    signal = macd.ewm(span=signal_window, adjust=False).mean()

    # Generate buy/sell signals
    buy_signal = (macd > signal) & (macd.shift(1) < signal.shift(1))
    sell_signal = (macd < signal) & (macd.shift(1) > signal.shift(1))

    # Create series of signals
    signals = pd.Series(0, index=close_prices.index)
    signals[buy_signal] = 0
    signals[sell_signal] = 2

    return signals


def buy_and_hold(close_prices: pd.Series) -> list:
    """
    Buy and hold strategy. Generates a list of buy/sell signals based on close prices of a stock.

    Args:
    - close_prices (pd.Series): Series containing the close prices.

    Returns:
    - list: List containing the buy/sell signals.
    """

    return [0] * len(close_prices)