"""
Eight performance metrics:
1. Average cumulative return
2. Average annual return
3. Max cumulative return
4. Min cumulative return
5. Win rate
6. Sharpe ratio
7. Coefficient of variation
8. Average of max drawdowns 
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class EvalMetrics:

    def __init__(self, balance_history):
        self._balance_history = balance_history


    def average_cumulative_return(self):
        return (self._balance_history[-1] - self._balance_history[0]) / self._balance_history[0]


    def average_annual_return(self):
        # return self.average_cumulative_return() / len(self.balance_history) # chnage according to data frequency
        pass


    def max_cumulative_return(self):
        return max(self._balance_history)
    

    def min_cumulative_return(self):
        return min(self._balance_history)


    def win_rate(self):
        wins = 0
        for i in range(len(self._balance_history) - 1):
            if self._balance_history[i] < self._balance_history[i + 1]: # succesful trade if balance increases
                wins += 1
        return wins / len(self._balance_history)


    def sharpe_ratio(self):
        # mean of returns / std of returns
        returns = []
        for i in range(len(self._balance_history) - 1):
            returns.append((self._balance_history[i + 1] - self._balance_history[i]) / self._balance_history[i])
        return np.mean(returns) / np.std(returns)


    def coefficient_of_variation(self):
        # std of returns / mean of returns
        returns = []
        for i in range(len(self._balance_history) - 1):
            returns.append((self._balance_history[i + 1] - self._balance_history[i]) / self._balance_history[i])
        return np.std(returns) / np.mean(returns) 


    def max_drawdown(self):
        max_drawdown = 0
        peak = self._balance_history[0]
        for balance in self._balance_history:
            if balance > peak:
                peak = balance
            dd = (balance - peak) / peak
            if dd < max_drawdown:
                max_drawdown = dd
        return max_drawdown

    def average_max_drawdown(self):
        drawdowns = []
        max_drawdown = 0
        peak = self._balance_history[0]
        for balance in self._balance_history:
            if balance > peak:
                peak = balance
            dd = (balance - peak) / peak
            if dd < max_drawdown:
                max_drawdown = dd
            if balance == self._balance_history[-1]:
                drawdowns.append(max_drawdown)
                max_drawdown = 0
                peak = balance
        return np.mean(drawdowns)
