import argparse
import gymnasium as gym
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim

from utils.utils import Config, DataDownloader, compute_portfolio, split_data, maximum_return, data_preprocessing
from utils.metrics import return_risk_metric, maximum_drawdown
from utils.rulebased_strategies import macd_trader, buy_and_hold
from agents.dqn import DQN
import envs

def parse_args():
    """Parse command line arguments."""

    def list_of_strings(arg):
        return arg.split(',')

    parser = argparse.ArgumentParser(description='Run DQN agent on stock data.')
    parser.add_argument('--stock', type=str, default='IBM', help='Stock ticker to evaluate.')
    parser.add_argument('-tf', '--timeframes', type=list_of_strings, default='1,3,5', help='Timeframes for stock data in days.')
    parser.add_argument('--modeldir', type=str, default='models', help='Model/actions directory.')
    parser.add_argument('--save', action='store_true', help='Save figures.')
    parser.add_argument('-v', '--verbose', action='store_true', help='Print out more information.')
    args = parser.parse_args()

    return args

def main():
    # Latex fonts
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    pd.options.mode.chained_assignment = None
    args = parse_args()
    timeframes = [int(tf) for tf in args.timeframes]

    # Get stock data
    stock = args.stock
    try:
        df = pd.read_csv(f'./datasets/{stock}.csv')
    except FileNotFoundError:
        print(f'Downloading {stock} data...')
        df = DataDownloader(
            start_date='2006-01-01',
            end_date='2018-01-01',
            tickers=[stock]
        ).download_data()

    split_date = '2016-01-01'
    _, test_df = split_data(df, split_date) # Get test set


    # Load actions from csv
    single_actions = []
    single_portfolios = []
    for tf in timeframes:
        actions = np.loadtxt(f'{args.modeldir}/actions_{stock}_{tf}.csv', delimiter=',')
        single_actions.append(actions)
        portfolio = compute_portfolio(test_df, actions.tolist(), initial_balance=1000)

        if len(portfolio) > len(test_df):
            portfolio = portfolio[:len(test_df)]

        # portfolio = [1000] * (tf - 1) + portfolio
        test_df[f'portfolio_{tf}'] = portfolio


    # Get multi-agent portfolio
    multi_actions = np.loadtxt(f'{args.modeldir}/multiagent_actions_{stock}.csv', delimiter=',')
    multi_portfolio = compute_portfolio(test_df, multi_actions.tolist(), initial_balance=1000, start_timestep=max(timeframes)-1)
    multi_portfolio = [1000] * (max(timeframes)-1) + multi_portfolio
    multi_portfolio = multi_portfolio + [multi_portfolio[-1]] * (len(test_df) - len(multi_portfolio))

    test_df['portfolio_multi'] = multi_portfolio

    # Get close prices for rule-based strategies
    close = pd.Series(test_df['Close'])

    # Buy and hold
    buy_and_hold_actions = buy_and_hold(close)
    buy_and_hold_portfolio = compute_portfolio(test_df, buy_and_hold_actions, initial_balance=1000)
    test_df['portfolio_buy_and_hold'] = buy_and_hold_portfolio

    # Get MACD trader portfolio
    macd_actions = macd_trader(close, initial_balance=1000)
    print(len(macd_actions))
    print(len(test_df))
    macd_portfolio = compute_portfolio(test_df, macd_actions, initial_balance=1000)
    test_df['portfolio_macd'] = macd_portfolio

    # Plot results
    fig, ax = plt.subplots(figsize=(4.5, 3.5))
    for tf in timeframes:
        ax.plot(test_df['Date'], test_df[f'portfolio_{tf}'], label=f'{tf}-day DQN', linewidth=1)
    ax.plot(test_df['Date'], test_df['portfolio_multi'], label='Multi-agent DQN', linewidth=1)
    ax.plot(test_df['Date'], test_df['portfolio_buy_and_hold'], label='Buy-and-Hold', linewidth=1)
    plt.plot(test_df['portfolio_macd'], label='MACD', linewidth=1)
    ax.set_xticks(test_df['Date'][::65], [a[:-3] for a in test_df['Date'][::65]])
    plt.xlim(test_df.index[0], test_df.index[-1])
    plt.xticks(rotation=45)
    plt.xlabel('Date')
    plt.ylabel('Portfolio Value / \$')
    plt.legend(frameon=False, loc='upper left', ncols=2)
    plt.tight_layout()
    plt.show()

    if args.save:
        fig.savefig(f'./figures/{stock}_portfolio.pdf', bbox_inches='tight', dpi=600)
    
    # Compute metrics
    print('Metrics:')
    for tf in timeframes:
        print(f'{tf}-day DQN:')
        print(f'Cumulative return: {(test_df[f"portfolio_{tf}"].iloc[-1] - 1000) / 1000 * 100:.2f}%')
        print(f'Return-risk metric: {return_risk_metric(test_df[f"portfolio_{tf}"]):.2f}')
        print(f'Maximum drawdown: {maximum_drawdown(test_df[f"portfolio_{tf}"]) * 100:.2f}%')
        print('')
    
    print('Multi-agent DQN:')
    print(f'Cumulative return: {(test_df["portfolio_multi"].iloc[-1] - 1000) / 1000 * 100:.2f}%')
    print(f'Return-risk metric: {return_risk_metric(test_df["portfolio_multi"]):.2f}')
    print(f'Maximum drawdown: {maximum_drawdown(test_df["portfolio_multi"]) * 100:.2f}%')
    print('')

    print('Buy and Hold:')
    print(f'Cumulative return: {(test_df["portfolio_buy_and_hold"].iloc[-1] - 1000) / 1000 * 100:.2f}%')
    print(f'Return-risk metric: {return_risk_metric(test_df["portfolio_buy_and_hold"]):.2f}')
    print(f'Maximum drawdown: {maximum_drawdown(test_df["portfolio_buy_and_hold"]) * 100:.2f}%')
    print('')

    print('MACD Trader:')
    print(f'Cumulative return: {(test_df["portfolio_macd"].iloc[-1] - 1000) / 1000 * 100:.2f}%')
    print(f'Return-risk metric: {return_risk_metric(test_df["portfolio_macd"]):.2f}')
    print(f'Maximum drawdown: {maximum_drawdown(test_df["portfolio_macd"]) * 100:.2f}%')
    print('')

if __name__ == '__main__':
    main()