from typing import List, Tuple

import numpy as np
import pandas as pd
import gymnasium as gym
import matplotlib.pyplot as plt

class SingleStockEnvironment(gym.Env):
    """
    OpenAI gym environment for single stock trading.

    Currently supports buy, sell, and hold actions.
    """
    metadata = {'render_modes': ['None']}

    def __init__(
            self,
            data: pd.DataFrame,
            initial_balance: int = 1000,
            time_window: int = 1,
            n_step: int = 10
    ):
        """
        Args:
        - data (pd.DataFrame): DataFrame containing the stock data.
        - initial_balance (int): Initial balance for the trading account.
        - time_window (int): Number of time steps to consider in the state.
        - n_step (int): Number of time steps to consider for the reward horizon.
        """
        super(SingleStockEnvironment, self).__init__()

        # Environment parameters
        self.df = data
        self.initial_balance = initial_balance
        self.time_window = time_window

        # Reward horizon
        self.n_step = n_step

        # Get array of (OHLC) states over the trading period
        self.states = self.df[['Open', 'High', 'Low', 'Close']].values

        # Get array of close prices over the trading period
        self.close_prices = self.df['Close'].values

        self._num_timesteps = len(self.df)
        self._end_timestep = self._num_timesteps - self.n_step - 1

        # Environment has 3 actions: sell (0), hold (1), buy (2)
        self.action_space = gym.spaces.Discrete(3)

        # Observation space contains `time_window` number of (OHLC) observations
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(4 * self.time_window,),
            dtype=np.float64
        )

    def reset(self, seed: int = None, options: dict = None) -> np.array:
        """
        Reset the environment to its initial state.
        
        Args:
        - seed (int): Random seed for environment.
        - options (dict): Additional options for environment reset.
        """
        super().reset(seed=seed, options=options)

        self.balance = self.initial_balance
        self.balance_history = [self.balance]
        
        self.own_share = False
        self.current_step = 0

        state = self.get_current_state()
        info = {
            'balance': self.balance,
            'current_step': self.current_step
        }

        return state, info
    
    def get_current_state(self) -> np.array:
        """
        Get the current state of the environment.
        
        Returns:
        - np.array: Current state of the environment.
        """
        return self.states[self.current_step]
    
    def step(self, action: int) -> Tuple[np.array, float, bool, bool, dict]:
        """
        Take an action in the environment.
        
        Args:
        - action (int): Action to take in the environment (0: buy, 1: hold, 2: sell).

        Returns:
        - tuple: Next state, reward, done flag, truncated flag, info dictionary.
        """
        
        done = False
        next_state = None

        if self.current_step < self._end_timestep:
            next_state = self.states[self.current_step + self.n_step]
        else:
            done = True

        if action == 0:
            self.own_share = True
        elif action == 2:
            self.own_share = False

        if self.own_share:
            self.balance *= (self.close_prices[self.current_step + 1] / self.close_prices[self.current_step])

        reward = 0 if done else self._compute_reward(action)
        info = {
            'balance': self.balance,
            'current_step': self.current_step
        }

        self.current_step += 1

        return next_state, reward, done, False, info
    
    def _compute_reward(self, action: int) -> float:
        """
        Compute the reward for the current step.
        
        Args:
        - action (int): Action taken in the current step.

        Returns:
        - float: Reward for the current step.
        """
        
        p1 = self.close_prices[self.current_step]
        p2 = self.close_prices[self.current_step + self.n_step]

        # Implement reward function as in the single-stock trading paper
        reward = 0
        if action == 0 or (action == 1 and self.own_share):
            reward = ((p2 / p1) - 1) * 100
        elif action == 2 or (action == 1 and not self.own_share):
            reward = ((p1 / p2) - 1) * 100

        return reward        