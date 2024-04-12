from collections import deque, namedtuple
import yfinance as yf
import pandas as pd
import torch
import random
import matplotlib.pyplot as plt
import numpy as np

class ExperienceReplay(object):
    def __init__(self, buffer_size: int = 10000, device: str = None):
        self.buffer = deque([], maxlen=buffer_size)
        self.transition = namedtuple('Experience', ('state', 'action', 'next_state', 'reward'))

        if device is not None:
            self.device = torch.device(device)
        else:
            self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    def add_experience(self, *args):
        """Adds an experience to the replay buffer."""
        self.buffer.append(self.transition(*args))

    def sample(self, batch_size: int = 32, separate: bool = True) -> tuple:
        """Samples a batch of experiences from the replay buffer."""
        samples = random.sample(self.buffer, k=batch_size)
        if separate:
            return self.transition(*zip(*samples))
        else:
            return samples

    def __len__(self):
        """Gets length of the replay buffer."""
        return len(self.buffer)


class Config(object):
    """
    Configuration object for running experiments. 
    
    Edit to add useful features.
    """
    def __init__(self):
        self.environment = None
        self.GPU = False
        self.hyperparameters = None


class DataDownloader:
    """Class for downloading data from Yahoo Finance."""
    def __init__(
            self,
            start_date: str,
            end_date: str,
            tickers: list
    ):
        """
        Args:
        - start_date (str): Start date for data download.
        - end_date (str): End date for data download.
        - tickers (list): List of tickers to download data for.
        """
        self.start_date = start_date
        self.end_date = end_date
        self.tickers = tickers

    def download_data(self) -> pd.DataFrame:
        """
        Downloads data from Yahoo Finance.

        Returns:
            pd.DataFrame: DataFrame containing the close prices for the specified tickers.
        """
        data = yf.download(self.tickers, start=self.start_date, end=self.end_date)
        return data
    

def split_data(data: pd.DataFrame, split_date: str) -> tuple:
    """
    Splits the data into training and testing sets.

    Args:
    - data (pd.DataFrame): DataFrame containing the data to split.
    - split_date (str): Date to split the data on.

    Returns:
    - tuple: Training and testing sets.
    """
    data.set_index('Date', inplace=True)

    train = data[data.index < split_date]
    test = data[data.index >= split_date]

    return train, test


# def ohlc_converter(data: np.ndarray, index: int, window_size: int) -> np.ndarray:
#     """
#     Applies operations to the last window_size number of data points in a DataFrame.

#     Parameters:
#     data (np.ndarray): The original DataFrame converted to a NumPy array.
#     index (int): The index to start the window from.
#     window_size (int): The size of the window.

#     Returns:
#     np.ndarray: A NumPy array where each element is the result of applying an operation to the window.
#     """
#     #print(data.shape)
#     #print(index-window_size+1, index+1)
#     window_0 = data[index-window_size+1:index+1, 0]
#     window_1 = data[index-window_size+1:index+1, 1]
#     window_2 = data[index-window_size+1:index+1, 2]
#     window_3 = data[index-window_size+1:index+1, 3]

#     return np.array([window_0[0], window_1.max(), window_2.min(), window_3[-1]])


def agent_counter(count: int, num_agents: int) -> int:
    """
    Increments the agent counter - For use in multi-agent environments.

    Args:
    - count (int): Current count.
    - num_agents (int): Number of agents.

    Returns:
    - int: Incremented count.
    """
    return count % num_agents


def plot_portfolio(data: pd.DataFrame, actions: list, initial_balance: int = 1000, start_timestep: int = 0):
    """
    Plots the portfolio balance over time.

    Args:
    - data (pd.DataFrame): DataFrame containing the data to plot.
    - actions (list): List of actions taken by the agent.
    - initial_balance (int): Initial balance for the portfolio.
    - start_timestep (int): Timestep to start plotting from.
    """
    
    portfolio_value = [initial_balance]
    own_share = False
    num_shares = 0

    close_prices = data['Close'].values[start_timestep:]

    # assert len(actions) == len(close_prices), 'Length of actions and close prices must be the same.'
    # print(len(actions), len(close_prices))

    # Iterate through list of actions and update portfolio value
    for i, action in enumerate(actions):

        # If action == buy and no shares owned, buy shares
        if action == 0 and num_shares == 0:
            own_share = True
            num_shares = portfolio_value[-1] / close_prices[i]
            
            if i < len(actions) - 1:
                portfolio_value.append(num_shares * close_prices[i+1])

        # If action == sell and shares owned, sell shares
        elif action == 2 and num_shares > 0:
            own_share = False
            portfolio_value.append(num_shares * close_prices[i])
            num_shares = 0

        # If action in [buy, hold] and shares owned, update portfolio value
        elif (action == 0 or action == 1) and num_shares > 0:
            if i < len(actions) - 1:
                portfolio_value.append(num_shares * close_prices[i+1])
        
        # If action in [sell, hold] and no shares owned, hold cash
        elif (action == 2 or action == 1) and num_shares == 0:
            portfolio_value.append(portfolio_value[-1])

    plt.figure(figsize=(10, 6))
    plt.plot(portfolio_value, color='b', label='Portfolio Value')
    plt.xlim(0, len(portfolio_value))
    plt.xlabel('Timesteps / days')
    plt.ylabel('Portfolio Value / $')
    plt.title('Portfolio Value for trained agent')
    plt.legend(frameon=False)
    plt.show()

    return portfolio_value


def maximum_return(data: pd.DataFrame, initial_balance: int = 1000):
    """
    Calculates the maximum return possible for the given data.

    Args:
    - data (pd.DataFrame): DataFrame containing the data to calculate the maximum return for.
    - initial_balance (int): Initial balance for the portfolio.

    Returns:
    - float: Maximum return possible for the given data.
    """
    close_prices = data['Close'].values
    own_shares = False

    balance = initial_balance

    for i in range(len(close_prices) - 1):
        
        if close_prices[i] < close_prices[i+1]:
            own_shares = True
        else:
            own_shares = False

        if own_shares:
            balance *= (close_prices[i+1] / close_prices[i])

    return balance


def data_preprocessing(df: pd.DataFrame, timeframe: int = 3) -> pd.DataFrame:
    """
    Preprocesses the data to get OHLC values for a given timeframe.

    Args:
    - df (pd.DataFrame): DataFrame containing the data to preprocess.
    - timeframe (int): Timeframe to get OHLC values for.

    Returns:
    - pd.DataFrame: Preprocessed data.
    """
    data = df.copy()
    data['Open'] = data['Open'].shift(timeframe - 1)
    data['High'] = data['High'].rolling(window=timeframe).max()
    data['Low'] = data['Low'].rolling(window=timeframe).min()
    # data.dropna(subset=['Open'], inplace=True)  # Drop rows with missing open a week ago
    return data