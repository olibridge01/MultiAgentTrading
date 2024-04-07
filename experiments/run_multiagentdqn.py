import argparse
import gymnasium as gym
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim

from utils.utils import Config, DataDownloader, plot_portfolio, split_data, maximum_return
from agents.multiagentdqn import MultiAgentDQN
import envs

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run DQN agent on stock data.')
    parser.add_argument('--stock', type=str, default='IBM', help='Stock ticker to run agent on.')
    parser.add_argument('--train', action='store_true', help='Train agent.')
    parser.add_argument('--test', action='store_true', help='Test agent.')
    parser.add_argument('--modeldir', type=str, default='models', help='Directory to save/load models.')
    parser.add_argument('--save', type=bool, default=False, help='Save actions to CSV.')
    parser.add_argument('-v', '--verbose', action='store_true', help='Print out more information.')
    args = parser.parse_args()

    # Assert that one of train or test is selected
    if (args.train and args.test) or (not args.train and not args.test):
        raise ValueError('Either train or test must be selected.')

    return args

def main():
    args = parse_args()
    stock = args.stock # Stock ticker to run agent on

    # Check if stock data is in datasets folder
    try:
        df = pd.read_csv(f'./datasets/{stock}.csv')
    except FileNotFoundError:
        # Download stock data if not found
        print(f'Downloading {stock} data...')
        
        # Download stock data from Yahoo Finance
        df = DataDownloader(
            start_date='2006-01-01',
            end_date='2018-01-01',
            tickers=[stock]
        ).download_data()

    split_date = '2016-01-01'
    train_df, test_df = split_data(df, split_date) # Split data into training and testing sets

    hyperparams = {
        'buffer_size': 20,
        'batch_size': 10,
        'hidden_dims': [512, 512],
        'learning_rate': 0.0001,
        'gamma': 0.7,
        'epsilon_start': 0.9,
        'epsilon_end': 0.05,
        'epsilon_decay': 500,
        'target_update': 5,
        'activation_last_layer': None,
        'gradient_clipping': 100,
        'lookback_window': 60,
        'hold_window': 5,
        'n_steps': 10,
        'initial_balance': 1000,
        'num_agents': 3,  # # of agents
        'trading_windows': [5, 3, 1],  # Trading window for each agent, always start with the largest
    }   
    
    # Config for training env
    dqn_train_config = Config()
    dqn_train_config.hyperparameters = hyperparams
    dqn_train_config.environment = gym.make(
        'multiagenttrading-v0',
        data=train_df,
        initial_balance=dqn_train_config.hyperparameters['initial_balance'],
        num_agents=dqn_train_config.hyperparameters['num_agents'],
        trading_windows=dqn_train_config.hyperparameters['trading_windows'],
    )
    dqn_agent = MultiAgentDQN(config=dqn_train_config)

    if args.train:
        rewards = dqn_agent.train(n_episodes=10, model_dir=args.modeldir)

    # Config for testing env
    dqn_test_config = Config()
    dqn_test_config.hyperparameters = hyperparams
    dqn_test_config.environment = gym.make(
        'multiagenttrading-v0',
        data=test_df,
        initial_balance=dqn_test_config.hyperparameters['initial_balance'],
        num_agents=dqn_test_config.hyperparameters['num_agents'],
        trading_windows=dqn_test_config.hyperparameters['trading_windows'],
    )

    test_agent = MultiAgentDQN(config=dqn_test_config)

    # Get just OHLC prices for test_df
    test_df_edited = test_df[['Open', 'High', 'Low', 'Close']]

    actions = test_agent.test(test_df_edited, args.modeldir) # Test agent on test data with trained policy network
    portfolio = plot_portfolio(
        test_df, 
        actions, 
        initial_balance=dqn_test_config.hyperparameters['initial_balance'], 
        start_timestep=max(dqn_test_config.hyperparameters['trading_windows']) - 1
    )

    if args.save:
        np.savetxt(f'models/multiagent_actions_{stock}.csv', np.array(actions), delimiter=',')

    if args.verbose:
        print(actions)
        print(f'Maximum possible return: ${maximum_return(test_df, initial_balance=1000):.2f}')
        print(f'Final portfolio value: ${portfolio[-1]:.2f}')

if __name__ == '__main__':
    main()