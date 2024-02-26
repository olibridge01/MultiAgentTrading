from gymnasium.envs.registration import register
from copy import deepcopy

from envs.tradingenv import TradingEnvironment

# Register custom environments with OpenAI Gym
register(
    id='trading-v1',
    entry_point='envs.tradingenv:TradingEnvironment',
    kwargs={
        'initial_balance': 1000,
        'lookback_window': 60,
        'hold_window': 5,
        'render_mode': None
    }
)