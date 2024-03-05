import numpy as np
import matplotlib.pyplot as plt

class RSI():
    """
    This strategy uses the Relative Strength Index (RSI) to determine when to buy and sell an asset.
    """

    def __init__(self, df, window=14):
        self.df = df
        self.window = window
        self.initial_balance = 1000
        self.current_balance = 1000
        self._balance_history = [self.initial_balance]
        self.num_shares = 0

    def get_rsi(self):
        delta = self.df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=self.window).mean() # Gain is average of positive changes
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.window).mean() # loss is average of negative changes
        rs = gain / loss # relative strength factor
        self.df['rsi'] = 100 - (100 / (1 + rs)) # RSI

    def implement_rsi_strategy(self):
        # Initial position is 0
        position = 0

        # Iterate through the data
        for i in range(len(self.df)):
            if self.df['rsi'].iloc[i] < 30: # buy when RSI is less than 30
                if position != 1:
                    self.num_shares = self.current_balance // self.df['Close'].iloc[i]
                    self.current_balance -= self.num_shares * self.df['Close'].iloc[i]
                    position = 1
                else:
                    # Hold
                    pass
            elif self.df['rsi'].iloc[i] > 70: # sell when RSI is greater than 70
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
        self.get_rsi()
        self.implement_rsi_strategy()
        self.plot_balance()