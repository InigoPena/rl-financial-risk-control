from gymnasium.envs.registration import register
from copy import deepcopy

from .data_processing.data_loading import (
    GOLD_DATA,
    HEDGE_DATA,
    FOREX_EURUSD_1H_ASK,
    STOCKS_GOOGL
)


register(
    id='forex-v0',
    entry_point='gym_anytrading.envs:ForexEnv',
    kwargs={
        'df': deepcopy(FOREX_EURUSD_1H_ASK),
        'window_size': 24,
        'frame_bound': (24, len(FOREX_EURUSD_1H_ASK))
    }
)

register(
    id='stocks-v0',
    entry_point='gym_anytrading.envs:StocksEnv',
    kwargs={
        'df': deepcopy(STOCKS_GOOGL),
        'window_size': 30,
        'frame_bound': (30, len(STOCKS_GOOGL))
    }
)

register(
    id='gold-hedge-v0',
    entry_point='Codigo.src.envs.gold_hedge_env:GoldHedgeEnv',
    kwargs={
        'gold_df': deepcopy(GOLD_DATA),
        'hedge_df': deepcopy(HEDGE_DATA),
        'window_size': 50,
        'frame_bound': (50, len(GOLD_DATA))
    }
)
