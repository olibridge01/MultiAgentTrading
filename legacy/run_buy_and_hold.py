import sys
# import os
sys.path.append('/Users/isaacwatson/Documents/MSc CSML/COMP0124/research-project/MultiAgentTrading/')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from strategies.buy_and_hold import BuyAndHold

test_df = pd.read_csv('datasets/000001.SS-test.csv')

buy_and_hold = BuyAndHold(df=test_df)
buy_and_hold.run_strategy(plot=True)

from utils.eval_metrics import EvalMetrics

eval_metrics = EvalMetrics(buy_and_hold._balance_history)
print(eval_metrics.cumulative_return())
print(eval_metrics.annualise_rets())
print(eval_metrics.max_drawdown())  
