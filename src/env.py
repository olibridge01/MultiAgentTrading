import numpy as np
import matplotlib.pyplot as plt

import gymnasium as gym

class MultiAgentTradingEnv(gym.Env):
    def __init__(self, num_agents=2, initial_balance=1000, max_steps=100):
        super(MultiAgentTradingEnv, self).__init__()

        self.num_agents = num_agents
        self.initial_balance = initial_balance
        self.max_steps = max_steps
        self.current_step = 0

        self.action_space = gym.spaces.Discrete(3)
        self.observation_space = gym.spaces.Box(low=0, high=np.inf, shape=(num_agents, 2), dtype=np.float32)

        self.agent_balances = np.full((num_agents,), initial_balance, dtype=np.float32)

    def reset(self):
        self.current_step = 0
        self.agent_balances = np.full((self.num_agents,), self.initial_balance, dtype=np.float32)
        return self._get_observation()

    def step(self, actions):
        assert len(actions) == self.num_agents, "Number of actions must match the number of agents"

        # Execute actions and update balances
        for agent_idx, action in enumerate(actions):
            if action == 0:  # Buy
                self.agent_balances[agent_idx] -= 10
            elif action == 1:  # Sell
                self.agent_balances[agent_idx] += 10

        self.current_step += 1

        rewards = -np.abs(np.random.normal(size=self.num_agents))

        done = self.current_step >= self.max_steps

        return self._get_observation(), rewards, done

    def _get_observation(self):
        return np.vstack([self.agent_balances, np.random.uniform(0, 1, size=(self.num_agents,))]).T

    def render(self):
        print("Agent Balances:", self.agent_balances)
