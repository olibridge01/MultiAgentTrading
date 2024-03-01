import sys
# import os
sys.path.append('/Users/isaacwatson/Documents/MSc CSML/COMP0124/research-project/MultiAgentTrading/')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from strategies.buy_and_hold import BuyAndHold

test_df = pd.read_csv('./datasets/eur-usd-testdata.csv')

buy_and_hold = BuyAndHold(df=test_df)
buy_and_hold.run_strategy()
