import numpy as np
import matplotlib.pyplot as plt

class BuyAndHold():

    """
    This strategy buys and holds the asset for the entire period.
    """

    def __init__(self, df, initial_balance=1000000):
        self.df = df
        self.initial_balance = initial_balance
        self._balance_history = None
        self.num_shares = 0

    def implement_buy_and_hold_strategy(self):
        # implement long buy and hold strategy on usd-eur
        self.df['Return'] = self.df['Close'].pct_change()
        self.df['Return'] = self.df['Return'].fillna(0)
        self._balance_history = [self.initial_balance] + ((1 + self.df['Return']).cumprod() * self.initial_balance).tolist()

    def plot_balance(self):
        plt.plot(self._balance_history)
        plt.show()

    def run_strategy(self, plot=False):
        self.implement_buy_and_hold_strategy()
        if plot:
            self.plot_balance()