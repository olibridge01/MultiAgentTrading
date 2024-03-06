import gymnasium as gym
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim

from utils.utils import Config
from agents.dqn import DQN
import envs

"""
To-Do:
    - Implement random seed for reproducibility?
    - Debug and test
"""

test_df = pd.read_csv('./datasets/eur-usd-h1.csv')

dqn_h1_config = Config()
dqn_h1_config.hyperparameters = {
    'buffer_size': 10000,
    'batch_size': 64,
    'hidden_dims': [300, 150],
    'learning_rate': 0.00025,
    'gamma': 0.45,
    'epsilon': 0.9,
    'epsilon_decay': 0.999,
    'target_update': 1000,
    'activation_last_layer': None,
    'gradient_clipping': 100,
    'lookback_window': 60,
    'hold_window': 5,
    'initial_balance': 1000
}
dqn_h1_config.environment = gym.make(
    'trading-v0',
    data=test_df,
    initial_balance=dqn_h1_config.hyperparameters['initial_balance'],
    lookback_window=dqn_h1_config.hyperparameters['lookback_window'],
    hold_window=dqn_h1_config.hyperparameters['hold_window']
)

n_exp = 5
balances = []

for i in range(n_exp):
    print(f'Running experiment {i+1}/{n_exp}')
    dqn_h1_agent = DQN(config=dqn_h1_config)
    balance_history, rewards = dqn_h1_agent.train()
    balances.append(balance_history)
    print(f'Final balance: {balance_history[-1]}')
    plt.plot(balance_history)


balances = np.array(balances)
plt.plot(balances.mean(axis=0), color='black', linewidth=2)

plt.show()