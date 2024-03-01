# These strategies consist of MACD, RSI, MovingAvg Cross, and the Buy and Hold (B&H) with default settings

import numpy as np
import matplotlib.pyplot as plt

class MACD():

    """
    The moving average convergence divergence (MACD) is an oscillator that combines two exponential moving averages (EMA).
    Calculated by subtracting two Exponential Moving Averages.
    Deafult parameters use the 26-period and the 12-period EMA.
    Indicates the momentum of a bullish or bearish trend.
    """

    def __init__(self, df, short_window=12, long_window=26, signal_window=9):
        self.df = df
        self.short_window = short_window
        self.long_window = long_window
        self.signal_window = signal_window
        self.initial_balance = 1000
        self.current_balance = 1000
        self.balance_history = []
        self.num_shares = 0


    def get_macd_line(self):
        # Compute the MACD line; difference between EMAs
        self.df['short_mavg'] = self.df['Close'].ewm(span=self.short_window, adjust=False).mean()
        self.df['long_mavg'] = self.df['Close'].ewm(span=self.long_window, adjust=False).mean()
        self.df['MACD'] = self.df['short_mavg'] - self.df['long_mavg']


    def get_signal_line(self):
        # Compute the signal line; EMA of the MACD line with a default period of 9
        self.df['signal_line'] = self.df['MACD'].ewm(span=self.signal_window,adjust = False).mean()


    def get_signal(self):
        # Create a signal for when to buy and sell; buy when MACD crosses above signal line, sell when MACD crosses below signal line
        self.df['signal'] = 0.0
        self.df['signal'] = np.where(self.df['MACD'] > self.df['signal_line'], 1.0, 0.0)
        self.df['signal'] = np.where(self.df['MACD'] < self.df['signal_line'], -1.0, self.df['signal'])


    def implement_macd_strategy(self):    
        # initial position is 0
        position = 0

        # iterate through the data
        for i in range(len(self.df)):
            # if position is 1, buy; if 
            if self.df['signal'].iloc[i] == 1:
                if position != 1:
                    self.num_shares = self.current_balance // self.df['Close'].iloc[i] # buy as many shares as possible (must be integer)
                    self.current_balance -= self.num_shares * self.df['Close'].iloc[i] # update balance
                    position = 1
                else:
                    # hold signal
                    pass

            elif self.df['signal'].iloc[i] == -1:
                if position != -1:
                    self.current_balance += self.num_shares * self.df['Close'].iloc[i] # sell all shares
                    self.num_shares = 0 # update number of shares
                    position = -1 
                else:
                    # hold signal
                    pass

            else:
                # no action
                pass
            
            # balance is the value of the portfolio (else drops to 0)
            self.balance_history.append(self.current_balance + self.num_shares * self.df['Close'].iloc[i])


    def plot_balance(self):
        plt.scatter(y=self.balance_history, x=self.df.index, c='blue', s=2)
        plt.show()


    def run_strategy(self):

        self.get_macd_line()
        self.get_signal_line()
        self.get_signal()
        self.implement_macd_strategy()
        self.plot_balance()

        return self.balance_history


    
