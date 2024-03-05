import sys
# import os
sys.path.append('/Users/isaacwatson/Documents/MSc CSML/COMP0124/research-project/MultiAgentTrading/')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from strategies.rsi import RSI

test_df = pd.read_csv('./datasets/eur-usd-testdata.csv')

rsi = RSI(df=test_df)
rsi.run_strategy()

# test eval metrics
from utils.eval_metrics import EvalMetrics

eval_metrics = EvalMetrics(balance_history=rsi._balance_history)
print(eval_metrics.average_cumulative_return())
