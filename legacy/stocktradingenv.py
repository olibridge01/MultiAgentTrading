from typing import List

import numpy as np
import pandas as pd
import gymnasium as gym
import matplotlib.pyplot as plt

class StockTradingEnvironment(gym.Env):

    metadata = {'render_modes': ['None']}


    def __init__(
            self,
            data: pd.DataFrame,
            initial_balance: int = 10000,
            hmax: int = 5,
            n_stocks: int = 30,
    ):
        super(StockTradingEnvironment, self).__init__()

        self.df = data
        self.initial_balance = initial_balance
        self.hmax = hmax
        self.n_stocks = n_stocks
        self.state_space = 2 * n_stocks + 1

        self.current_day = 0
        self.portfolio = np.zeros(self.n_stocks)
        self.balance = self.initial_balance

        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.state_space,),
            dtype=np.float64
        )
        self.action_space = gym.spaces.Box(
            low=-1,
            high=1,
            shape=(n_stocks,),
        )
        
        self.actions_memory = []
        self.asset_memory = [self.initial_balance]
        self.done = False

    def reset(self, seed: int = None, options: dict = None) -> tuple:
        """Reset the environment to its initial state."""
        super().reset(seed=seed, options=options)

        self.current_day = 0
        self.portfolio = np.zeros(self.n_stocks)
        self.balance = self.initial_balance
        
        self.state = self._update_state()
        info = {
            'current_day': self.current_day,
            'portfolio': self.portfolio,
            'balance': self.balance,
        }
        return self.state, info
    
    def _update_state(self) -> np.ndarray:
        """Initialize the state of the environment."""
        
        # Get the current price of each stock
        current_prices = self.df.iloc[self.current_day].values

        # Form state vector
        state = np.concatenate(
            (np.array([self.balance]), self.portfolio, current_prices)
        )
        return state
    
    def _calculate_portfolio_value(self) -> float:
        """Calculate the value of the portfolio."""
        return self.balance + np.sum(self.portfolio * self.df.iloc[self.current_day].values)

    def step(self, actions: np.ndarray) -> tuple:
        """Take a step in the environment given an action."""            

        actions = actions * self.hmax # Rescale actions to maximum number of shares to buy or sell
        actions = actions.astype(int) # Convert continuous actions to discrete values

        start_assets = self._calculate_portfolio_value() # Compute initial portfolio value

        argsort_actions = np.argsort(actions)
        sell_index = argsort_actions[:np.where(actions < 0)[0].shape[0]]
        buy_index = argsort_actions[::-1][:np.where(actions > 0)[0].shape[0]]

        # Execute sell and buy orders
        for index in sell_index:
            if self.portfolio[index] > 0:
                n_sold = min(abs(actions[index]), self.portfolio[index])
                self.balance += self.df.iloc[self.current_day].values[index] * n_sold
                self.portfolio[index] -= n_sold

        for index in buy_index:
            if self.balance > 0:
                n_buy = min(self.balance // self.df.iloc[self.current_day].values[index], actions[index])
                self.balance -= self.df.iloc[self.current_day].values[index] * n_buy
                self.portfolio[index] += n_buy
        
        self.actions_memory.append(actions) 

        # Transition to next state
        self.current_day += 1
        self.state = self._update_state()

        # Calculate reward
        end_assets = self._calculate_portfolio_value()
        self.asset_memory.append(end_assets)
        reward = end_assets - start_assets

        # Check if the episode is over
        self.done = (self.current_day == len(self.df) - 1)

        # Provide information
        info = {
            'current_day': self.current_day,
            'portfolio': self.portfolio,
            'balance': self.balance,
        }

        return self.state, reward, self.done, False, info
    
    def save_asset_memory(self) -> None:
        """Save asset memory to a file."""
        return self.asset_memory