from typing import List

import numpy as np
import pandas as pd
import gymnasium as gym
import matplotlib.pyplot as plt


class TradingEnvironment(gym.Env):
    """
    A general trading environment for stock/forex trading using OpenAI Gym API.

    Work in progress - implementing basic trading environment to replicate MARL paper

    To Do:
    - Deal with rendering (if necessary)
    - Check how shorting is handled with balance updates etc.

    """

    metadata = {'render_modes': ['None']}

    def __init__(
        self,
        data: pd.DataFrame,
        initial_balance: int = 1000,
        lookback_window: int = 60,
        hold_window: int = 5,
        render_mode: str = None
    ):
        super(TradingEnvironment, self).__init__()
        assert render_mode is None or render_mode in self.metadata['render.modes']

        # Environment parameters
        self.df = data
        self.initial_balance = initial_balance
        self.lookback_window = lookback_window
        self.hold_window = hold_window
        self.render_mode = render_mode

        self._num_timesteps = len(self.df)
        self._end_timestep = self._num_timesteps - self.hold_window - 1

        # Environment has 3 actions: long, do nothing, short
        self.action_space = gym.spaces.Discrete(3)

        # Observation space contains `lookback_window` number of (OHLCV) observations
        self.observation_space = gym.spaces.Box(
            low=0.0,
            high=np.inf,
            shape=(lookback_window * 5,),
            dtype=np.float32
        )

    def reset(self, seed: int = None, options: dict = None) -> tuple:
        """Reset the environment to its initial state."""
        super().reset(seed=seed, options=options)

        self.balance = self.initial_balance
        self._balance_history = [self.balance]
        self._timestep = self.lookback_window
        self._terminated = False

        self._position = 0

        # Get first observation and corresponding info
        obs = self._get_obs()
        info = self._get_info()

        return obs, info
    
    def step(self, action: int) -> tuple:
        """Take a step in the environment given an action."""
        if self._timestep >= self._end_timestep:
            self._terminated = True
            return self._get_obs(), 0, self._terminated, False, self._get_info()

        # Update position based on action
        self._update_position(action)

        # Update balance and compute reward
        self._update_balance()
        reward = self._compute_reward()

        # Increase timestep by hold window (action is held for `hold_window` timesteps)
        self._timestep += self.hold_window

        # Get observation and info
        obs = self._get_obs()
        info = self._get_info()

        return obs, reward, self._terminated, False, info
    
    def _get_obs(self) -> np.ndarray:
        """Get the current observation."""
        return self.df.iloc[self._timestep - self.lookback_window:self._timestep, 1:].values.flatten()
    
    def _get_info(self) -> dict:
        """Get the current state information."""
        return {
            'balance': self.balance,
            'position': self._position
        }
    
    def _update_position(self, action: int):
        """Update the position based on the action."""
        if action == 0:
            self._position = 1
        elif action == 2:
            self._position = -1
        else:
            self._position = 0

    def _update_balance(self):
        """Update the balance based on the current position."""
        start_price = self.df.iloc[self._timestep, 4]
        start_balance = self.balance
        
        # Update balance for each timestep in the hold window
        for t in range(1, self.hold_window + 1, 1):
            current_price = self.df.iloc[self._timestep + t, 4]
            
            if self._position == 1:
                self.balance = start_balance * (current_price / start_price)
            elif self._position == -1:
                self.balance = 2 * start_balance - start_balance * (current_price / start_price)

            self._balance_history.append(self.balance)

    def _compute_reward(self) -> float:
        """Compute the reward based on the current balance."""
        start_price = self.df.iloc[self._timestep, 4]
        end_price = self.df.iloc[self._timestep + self.hold_window, 4]

        return self._position * (end_price - start_price) / start_price

    def render(self, mode: str = None):
        """Render the environment."""
        pass
    
    def close(self):
        """Close the environment."""
        pass