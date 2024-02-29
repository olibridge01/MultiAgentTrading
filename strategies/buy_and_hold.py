import numpy as np
import matplotlib.pyplot as plt

class BuyAndHold():

    """
    This strategy buys and holds the asset for the entire period.
    """

    def __init__(self, df):
        self.df = df
        self.initial_balance = 1000
        self.current_balance = 1000
        self.balance_history = []
        self.num_shares = 0

    def implement_buy_and_hold_strategy(self):

        self.num_shares = self.current_balance // self.df['Close'].iloc[0] # buy as many shares as possible (must be integer)
        self.current_balance -= self.num_shares * self.df['Close'].iloc[0] # update balance (zero)
        
        for i in range(len(self.df)):
            self.balance_history.append(self.current_balance + self.num_shares * self.df['Close'].iloc[i])


    def plot_balance(self):
        plt.scatter(y=self.balance_history, x=self.df.index, c='blue', s=2)
        plt.show()


    def run_strategy(self):
        self.implement_buy_and_hold_strategy()
        self.plot_balance()