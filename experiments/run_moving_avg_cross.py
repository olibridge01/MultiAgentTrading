import sys
# import os
sys.path.append('/Users/isaacwatson/Documents/MSc CSML/COMP0124/research-project/MultiAgentTrading/')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from strategies.moving_avg_cross import MovingAverageCross

test_df = pd.read_csv('./datasets/eur-usd-testdata.csv')

# moving_avg_cross = MovingAverageCross(df=test_df)
# moving_avg_cross.run_strategy()