from typing import List, Tuple
from utils.utils import agent_counter, data_preprocessing
import numpy as np
import pandas as pd
import gymnasium as gym
import matplotlib.pyplot as plt

class MultiAgentSingleStockEnvironment(gym.Env):
    """
    OpenAI gym environment for single stock trading with our multi-agent framework.

    Currently supports buy, sell, and hold actions.
    """
    metadata = {'render_modes': ['None']}

    def __init__(
            self,
            data: pd.DataFrame,
            initial_balance: int = 1000,
            num_agents: int = 3,
            trading_windows: List[int] = [5, 3, 1],
            time_window: int = 1,
            n_step: int = 10
    ):
        """
        Args:
        - data (pd.DataFrame): DataFrame containing the stock data.
        - initial_balance (int): Initial balance for the trading account.
        - num_agents (int): Number of agents in the environment.
        - trading_windows (List[int]): List of trading windows (timeframes) for each agent.
        - time_window (int): Number of time steps to consider in the state.
        - n_step (int): Number of time steps to consider for the reward horizon.
        """
        super(MultiAgentSingleStockEnvironment, self).__init__()

        # Environment parameters
        self.df = data
        self.initial_balance = initial_balance
        self.time_window = time_window

        # Number of agents in the environment and timeframes
        self.num_agents = num_agents
        self.trading_windows = trading_windows
        self.actions_taken = np.full(self.num_agents - 1, 1) # Initialise actions taken by longer timeframe agents to hold

        # Reward horizon
        self.n_step = n_step

        # Get array of (OHLC) states over the trading period for each agent
        self.states = {f'agent_{i}': data_preprocessing(self.df, timeframe=trading_windows[i])[['Open', 'High', 'Low', 'Close']].values for i in range(self.num_agents)}

        # Get array of close prices over the trading period
        self.close_prices = self.df['Close'].values

        self._num_timesteps = len(self.df)
        self._end_timestep = self._num_timesteps - self.n_step - 1

        # Environment has 3 actions: sell (0), hold (1), buy (2)
        self.action_space = gym.spaces.Discrete(3)

        # Observation space contains `time_window` number of (OHLC) observations and the actions of other agents 
        self.observation_space = gym.spaces.Box(
        low=-np.inf,
        high=np.inf,
        shape=(4 * self.time_window + self.num_agents - 1,),
        dtype=np.float64
        )

    def reset(self, seed: int = None, options: dict = None) -> Tuple[np.array]:
        """
        Reset the environment to its initial states.
        
        Args:
        - seed (int): Random seed for environment.
        - options (dict): Additional options for environment reset.

        Returns:
        - Tuple[np.array]: Initial states of all agents.
        """
        super().reset(seed=seed, options=options)

        # Initialise the balance
        self.balance = self.initial_balance
        self.balance_history = [self.balance]
        
        # Initialise the ownership status and done flag for each agent
        self.own_share = [False] * self.num_agents
        self.done = [False] * self.num_agents

        # Start from the longest trading window to ensure all agents have a state
        self.current_step = max(self.trading_windows) - 1
        self.counter = 0

        # Generate the initial state for each agent
        states = []
        for i in range(self.num_agents):
            states.append(np.concatenate([self.states[f'agent_{i}'][self.current_step], self.actions_taken]))
            
        info = {
            'balance': self.balance,
            'current_step': self.current_step
        }
        
        return np.array(states), info
    
    def get_current_state(self) -> np.array:
        """
        Get the current state of the environment.
        
        Returns:
        - np.array: Current state of the environment.
        """
        agent = agent_counter(self.counter, self.num_agents)
        state = np.concatenate([self.states[f'agent_{agent}'][self.current_step], self.actions_taken])

        return state 
    
    def step(self, action: int) -> Tuple[np.array, float, bool, dict]:
        """
        Take an action in the environment.

        Args:
        - action (int): Action to take in the environment (0: sell, 1: hold, 2: buy).

        Returns:
        - tuple: Next state, reward, done flag, info dictionary.
        """
        agent = agent_counter(self.counter, self.num_agents)
        self.done[agent] = False
        next_state = None

        # Update the ownership status based on the action of the current agent
        if action == 0: # sell 
            self.own_share[agent] = False
        elif action == 2: # buy
            self.own_share[agent] = True

        # Update action_taken by the current agent
        if agent < self.num_agents - 1:
            self.actions_taken[agent] = action
            # print that the agent has updated actions taken to ...
            #print (agent, 'has updated actions taken to', self.actions_taken)

        # Get the next state if the current step is not the last step
        if self.current_step < self._end_timestep:
            next_state = np.concatenate([self.states[f'agent_{agent}'][self.current_step + self.n_step], self.actions_taken])
        else:
            self.done[agent] = True

        # Compute reward
        reward = 0 if self.done[agent] else self._compute_reward(action)

        # Only the shortest timeframe agent affects the balance
        if self.own_share[-1]:
            self.balance *= (self.close_prices[self.current_step + 1] / self.close_prices[self.current_step])

        # Print the balance if the last agent has finished
        # if self.done[-1]:
        #     print('Balance:', self.balance)

        info = {
            'balance': self.balance,
            'current_step': self.current_step
        }

        terminate = self.done[agent]
        self.counter += 1 # Increment the agent counter

        if agent == self.num_agents - 1:
            self.current_step += 1 # Increment the current step if the last agent has taken an action

        return next_state, reward, terminate, False, info
    
    def _compute_reward(self, action: int) -> float:
        """
        Compute the reward for the current step.
        
        Args:
        - action (int): Action taken in the current step.

        Returns:
        - float: Reward for the current step.
        """
        
        agent = agent_counter(self.counter, self.num_agents)

        p1 = self.close_prices[self.current_step]
        p2 = self.close_prices[self.current_step + self.n_step]

        # Implement reward function as in the single-stock trading paper
        reward = 0
        if action == 0 or (action == 1 and self.own_share[agent]):
            reward = ((p2 / p1) - 1) * 100
        elif action == 2 or (action == 1 and not self.own_share[agent]):
            reward = ((p1 / p2) - 1) * 100

        return reward        