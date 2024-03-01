import sys
# import os
sys.path.append('/Users/isaacwatson/Documents/MSc CSML/COMP0124/research-project/MultiAgentTrading/')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from strategies.macd import MACD

test_df = pd.read_csv('./datasets/eur-usd-testdata.csv')

macd = MACD(df=test_df)
macd.run_strategy()
