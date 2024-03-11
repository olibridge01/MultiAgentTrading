# class of metrics to evaluate the performance of a portfolio allocation strategy.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from typing import List

class PortfolioPerformance:
    """
    Final portfolio value, annualized return, annualized standard error and the Sharpe ratio
    """
    def __init__(self, portolfio_value_history: List):
        self.portolfio_value_history = portolfio_value_history
        self.n_years = len(portolfio_value_history) / 252

    def final_portfolio_value(self):
        return self.portolfio_value_history[-1]
    
    def annualized_return(self):
        return (self.portolfio_value_history[-1] / self.portolfio_value_history[0]) ** (1/self.n_years) - 1
    
    def annualized_std(self):
        returns = []
        for i in range(len(self.portolfio_value_history) - 1):
            returns.append((self.portolfio_value_history[i + 1] - self.portolfio_value_history[i]) / self.portolfio_value_history[i])
        return np.std(returns) * np.sqrt(252)
    
    def sharpe_ratio(self):
        return self.annualized_return() / self.annualized_std()
    
