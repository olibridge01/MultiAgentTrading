# Preprocess csv files to our desired format

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import re
import argparse

# Load data
processed_data = []

with open('./datasets/eur-usd-testdata-2.csv', 'r') as f:
    data = f.readlines()
    for line in data:
        line = line.strip()
        line = re.split('\t', line)
        line[1:] = [float(x) for x in line[1:]]
        processed_data.append(line)

# Export to csv
processed_data = np.array(processed_data, dtype=object)
df = pd.DataFrame(processed_data, columns=['Date', 'Open', 'High', 'Low', 'Close', 'Volume'])
df.to_csv('./datasets/eur-usd-testdata-2-processed.csv', index=False)
