import sys
# import os
# sys.path.append('/Users/isaacwatson/Documents/MSc CSML/COMP0124/research-project/MultiAgentTrading/')

import gymnasium as gym
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim

from utils.utils import Config
# from agents.dqn import DQN
from stable_baselines3 import A2C, PPO, DDPG, TD3, SAC, DQN
import envs

from stable_baselines3.common.env_checker import check_env

# train_data = pd.read_csv('datasets/000001.SS-train.csv')
# test_data = pd.read_csv('datasets/000001.SS-test.csv')
train_data = pd.read_csv('datasets/train_data3.csv')
test_data = pd.read_csv('datasets/trade_data3.csv')
plt.plot(test_data['Close'])
plt.show()
plt.close()

agent = DQN
print(str(agent.__name__))

train_env = gym.make(
    'trading-v0',
    data=train_data,
    initial_balance=1000,
    agent=str(agent.__name__)
    )

test_env = gym.make(
    'trading-v0',
    data=test_data,
    initial_balance=1000,
    agent=str(agent.__name__)
    )

# check_env(train_env)

# From papaer: 
# "The optimizer is Adam, the learning rate is 1e-6, the internal layer activation function is tanh, and the output layer activation function is SoftMax"

if agent in [DDPG, TD3, SAC]:
    from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
    n_actions = train_env.action_space.shape[-1]
    action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(n_actions), sigma=np.ones(n_actions)) # increase sigma to prevent getting stuck in local optima
    # action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.5*np.ones(n_actions))
    model = agent(
        policy='MlpPolicy', 
        env=train_env, 
        learning_rate=1e-6,
        action_noise=action_noise, 
        learning_starts=1e2, 
        tau=1e-2,
        # gradient_steps=100,
        gamma=0.99
        )
elif agent in [A2C, PPO, DQN]:
    model = agent(
        policy='MlpPolicy', 
        env=train_env, 
        learning_rate=1e-4
        ) # increase ent_coef to prevent getting stuck in local optima
    
print('Learning model...')
model.learn(total_timesteps=1e6, log_interval=1e3)
print('Model learned!')

# vec_env = model.get_env()
obs, info = test_env.reset()

while True:
    action, _states = model.predict(obs)
    obs, reward, done, _, info = test_env.step(action)
    if done:
        print("info", info)
        break

plt.figure(figsize=(16,9))
test_env.unwrapped.render_all()
plt.show()
print(test_env.unwrapped.action_history)

from utils.eval_metrics import EvalMetrics

eval_metrics = EvalMetrics(test_env.balance_history)
print(eval_metrics.cumulative_return())
print(eval_metrics.annualise_rets())
print(eval_metrics.max_drawdown())  
print(eval_metrics.sharpe_ratio())

# dqn_config = Config()
# dqn_config.hyperparameters = {
#     'buffer_size': 100000,
#     'batch_size': 64,
#     'hidden_dims': [1024, 512],
#     'learning_rate': 0.00025,
#     'gamma': 0.95,
#     'epsilon': 0.9,
#     'epsilon_decay': 0.999,
#     'target_update': 1000,
#     'activation_last_layer': None,
# }

# dqn_config.environment = gym.make(
#     'trading-v1',
#     data=data,
#     initial_balance=100000,
# )

# n_exp = 5
# exp_histories = []

# for i in range(n_exp):
#     print(f'Running experiment {i+1}/{n_exp}')
#     dqn_agent = DQN(config=dqn_config)
#     info = dqn_agent.train()
#     print("info", info)

# plt.figure(figsize=(16,9))
# dqn_config.environment.render_all()
# plt.show()
