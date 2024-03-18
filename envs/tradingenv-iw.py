from typing import List

import numpy as np
import pandas as pd
import gymnasium as gym
import matplotlib.pyplot as plt


class TradingEnvironment(gym.Env):

    metadata = {'render_modes': ['None']}

    def __init__(
            self,
            data: pd.DataFrame,
            initial_balance: float = 100000,
            render_mode: str = None,
            agent: str = None
            ):
        super(TradingEnvironment, self).__init__()
        assert render_mode is None or render_mode in self.metadata['render.modes']

        self.agent = agent

        # Environment parameters
        self.df = data
        self.stock_holdings = 0
        self.initial_balance = initial_balance
        self.balance = self.initial_balance
        self.value = self.balance
        self.render_mode = render_mode

        # episode parameters
        self._num_timesteps = len(self.df)
        self._end_timestep = self._num_timesteps - 1
        self.history = [self.initial_balance]
        self._cum_returns = []
        self._timestep = 0
        self._terminated = False
        self._total_reward = 0.
        self._holdings_history = []

        self.price = self.df.iloc[self._timestep]['Close']

        # Environment has k actions: -k, ..., -1, 0, 1, ..., k; initialise to -1, 0, 1
        self.k = 5
        if self.agent in ['A2C', 'PPO', 'DQN']:
            self.action_space = gym.spaces.Discrete(2*self.k+1)
        elif self.agent in ['DDPG', 'TD3', 'SAC']:
            self.action_space = gym.spaces.box.Box(
                low=-1,
                high=1,
                shape=(1,),
                dtype=np.float32
            )

        # Observation space contains balance, stock holdings, close price, MACD, RSI, CCI, and ADX
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(7,),
            dtype=np.float64
        )

    def reset(self, seed: int = None, options: dict = None) -> tuple:
        """Reset the environment to its initial state."""
        super().reset(seed=seed, options=options)

        self.history = [self.initial_balance]
        self._total_profit = 0
        self._timestep = 0
        self._terminated = False
        self.stock_holdings = 0
        self._total_reward = 0
        self._holdings_history = []
        self.balance = self.initial_balance
        self._cum_returns = []

        # Get first observation and corresponding info
        obs = self._get_obs()
        info = self._get_info()

        return obs, info
    
    def step(self, action: int) -> tuple:
        # if type(action) == np.ndarray:
        #     action = action[0]
        if self._timestep == self._end_timestep - 1:
            self._terminated = True
            return self._get_obs(), 0, self._terminated, False, self._get_info()

        # Execute trade
        self._exceute_trade(action)
        self._update_value()
        step_reward = self._calculate_reward()
        self._total_reward += step_reward

        # Move to the next timestep
        self._timestep += 1
        
        # Get next observation and corresponding info
        obs = self._get_obs()
        info = self._get_info()

        # print(f'Timestep: {self._timestep}, Balance: {self.balance}, Stock holdings: {self.stock_holdings}, Value: {self.value}, Reward: {step_reward}')

        # if self.agent in ['A2C', 'PPO']:
        #     self.action_space = gym.spaces.Discrete(2*self.stock_holdings+1)
        # elif self.agent in ['DQN', 'DDPG', 'TD3', 'SAC']:
        #     if self.stock_holdings == 0:
        #         self.action_space = gym.spaces.box.Box(
        #             low=-1,
        #             high=1,
        #             shape=(1,),
        #             dtype=np.float64
        #         )
        #     else:
        #         self.action_space = gym.spaces.box.Box(
        #             low=-self.stock_holdings,
        #             high=self.stock_holdings,
        #             shape=(1,),
        #             dtype=np.float64
        #         )
        
        return obs, step_reward, self._terminated, False, info

    def _get_obs(self) -> np.ndarray:
        """Get the current observation."""
        # print(type(self.stock_holdings))
        return np.array([
            self.balance,
            self.stock_holdings,
            self.df.iloc[self._timestep]['Close'],
            self.df.iloc[self._timestep]['MACD'],
            self.df.iloc[self._timestep]['RSI'],
            self.df.iloc[self._timestep]['CCI'],
            self.df.iloc[self._timestep]['ADX']
        ])
    
    def _get_info(self) -> dict:
        return dict(
            total_reward=self._total_reward,
            total_value=self.value,
            stock_holdings=self.stock_holdings
        )

    def _update_value(self):
        self.value = self.balance + self.stock_holdings*self.df.iloc[self._timestep]['Close']
        self.history.append(self.value)
        returns = (self.value - self.history[-2]) / self.history[-2]
        compound_returns = 1 + returns
        self._cum_returns.append(compound_returns)

    def _exceute_trade(self, action: int):
        if self.agent in ['A2C', 'PPO']:
            tmp_action = action - (self.k+1)
            # print(tmp_action)
            trade_amount = tmp_action * self.df.iloc[self._timestep]['Close']
            # Check if the trade is possible
            # if (self.balance - trade_amount < 0):
            if (self.stock_holdings + tmp_action < 0) or (self.balance - trade_amount < 0):                
                # Prevent the trade from occurring
                return
            self.balance -= trade_amount
            self.stock_holdings += tmp_action
            # print(f'Action: {action}, Trade amount: {trade_amount}, Balance: {self.balance}, Stock holdings: {self.stock_holdings}')
        
        elif self.agent in ['DQN', 'DDPG', 'TD3', 'SAC']:
            int_action = int(action * self.k)
            # print(int_action)
            trade_amount = int_action * self.df.iloc[self._timestep]['Close']
            # Check if the trade is possible
            # if (self.balance - trade_amount < 0):
            if (self.stock_holdings + int_action < 0) or (self.balance - trade_amount < 0):
                # Prevent the trade from occurring
                return
            self.balance -= trade_amount
            self.stock_holdings += int_action
            # print(f'Action: {action}, Trade amount: {trade_amount}, Balance: {self.balance}, Stock holdings: {self.stock_holdings}')

    def _calculate_reward(self):
        if len(self.history) == 1:
            return 0
        reward = self.value - self.history[-2]
        # print(reward)
        return reward
    
    def render_all(self):
        # compound returns from the initial balance
        cum_returns = np.array(self._cum_returns).cumprod()
        balance = self.initial_balance * cum_returns
        plt.plot(cum_returns)
        plt.title('Portfolio growth: ' + self.agent)
        plt.xlabel('Timestep')
        plt.ylabel('Portfolio value')
        plt.show()

        