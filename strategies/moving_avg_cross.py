import numpy as np
import matplotlib.pyplot as plt

class MovingAverageCross():
    """
    This strategy uses two moving averages to determine when to buy and sell an asset.
    """

    def __init__(self, df, window=9):
        self.df = df
        self.window = window
        self.initial_balance = 1000
        self.current_balance = 1000
        self._balance_history = [self.initial_balance]
        self.num_shares = 0


    def get_moving_averages(self):
        # Compute the moving average
        self.df['moving_avg'] = self.df['Close'].rolling(window=self.window).mean() 

    def implement_moving_average_cross_strategy(self):
        # Initial position is 0
        position = 0

        # Iterate through the data
        for i in range(len(self.df)):
            if self.df['Close'].iloc[i] > self.df['moving_avg'].iloc[i]: # buy when close price is greater than moving average
                if position != 1:
                    self.num_shares = self.current_balance // self.df['Close'].iloc[i]
                    self.current_balance -= self.num_shares * self.df['Close'].iloc[i]
                    position = 1
                else:
                    # Hold
                    pass
            elif self.df['Close'].iloc[i] < self.df['moving_avg'].iloc[i]: # sell when close price is less than moving average
                if position != -1:
                    self.current_balance += self.num_shares * self.df['Close'].iloc[i]
                    self.num_shares = 0
                    position = -1
                else:
                    # Hold
                    pass

            self._balance_history.append(self.current_balance + self.num_shares * self.df['Close'].iloc[i])


    def plot_balance(self):
        plt.scatter(y=self._balance_history, x=np.arange(len(self._balance_history)), c='blue', s=2)
        plt.show()


    def run_strategy(self):
        self.get_moving_averages()
        self.implement_moving_average_cross_strategy()
        self.plot_balance()