import gymnasium as gym
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim

from utils.utils import Config, plot_portfolio, split_data
from agents.dqn import DQN
import envs

"""
To-Do:
    - Implement Command Line input for running experiments.
"""

# Change to whichever stock you want to train/test on
df = pd.read_csv('./datasets/IBM.csv')
split_date = '2016-01-01'

train_df, test_df = split_data(df, split_date)

hyperparams = {
    'buffer_size': 20,
    'batch_size': 10,
    'hidden_dims': [128, 256],
    'learning_rate': 0.001,
    'gamma': 0.7,
    'epsilon': 0.9,
    'epsilon_decay': 0.99,
    'target_update': 5,
    'activation_last_layer': None,
    'gradient_clipping': 100,
    'lookback_window': 60,
    'hold_window': 5,
    'n_steps': 10,
    'initial_balance': 1000
}

# Config for training env
dqn_train_config = Config()
dqn_train_config.hyperparameters = hyperparams
dqn_train_config.environment = gym.make(
    'trading-v0',
    data=train_df,
    initial_balance=dqn_train_config.hyperparameters['initial_balance'],
)
dqn_agent = DQN(config=dqn_train_config)

# UNCOMMENT TO TRAIN
# rewards = dqn_agent.train(n_episodes=10)


# Config for testing env
dqn_test_config = Config()
dqn_test_config.hyperparameters = hyperparams
dqn_test_config.environment = gym.make(
    'trading-v0',
    data=test_df,
    initial_balance=dqn_test_config.hyperparameters['initial_balance'],
)

test_agent = DQN(config=dqn_test_config)

# Get OHLC prices for test_df
test_df_edited = test_df[['Open', 'High', 'Low', 'Close']]

actions = test_agent.test('models/policy_net.pkl', test_df_edited)

portfolio = plot_portfolio(test_df, actions, initial_balance=dqn_test_config.hyperparameters['initial_balance'])

print(actions)
np.savetxt('models/actions_IBM.csv', np.array(actions), delimiter=',')
# print(portfolio)