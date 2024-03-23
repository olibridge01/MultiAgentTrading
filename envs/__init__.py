from gymnasium.envs.registration import register
from copy import deepcopy

from legacy.tradingenv import TradingEnvironment

# Register custom environments with OpenAI Gym
register(
    id='trading-v0',
    entry_point='envs.singlestockenv:SingleStockEnvironment',
    kwargs={
        'initial_balance': 1000,
    }
)