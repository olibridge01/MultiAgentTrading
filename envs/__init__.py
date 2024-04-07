from gymnasium.envs.registration import register
from copy import deepcopy

# Register custom environments with OpenAI Gym
register(
    id='trading-v0',
    entry_point='envs.singlestockenv:SingleStockEnvironment',
    kwargs={
        'initial_balance': 1000,
    }
)

register(
    id='multiagenttrading-v0',
    entry_point='envs.multiagentsinglestockenv:MultiAgentSingleStockEnvironment',
    kwargs={
        'initial_balance': 1000,
    }
)