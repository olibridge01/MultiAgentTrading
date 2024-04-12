
![Python3.10](https://img.shields.io/pypi/pyversions/msmhelper)  

# Hierarchical Multi-Agent Reinforcement Learning Approaches to Algorithmic Trading
This repository contains the code for the Multi-Agent Artificial Intelligence project for COMP0124 at UCL. The project investigates hierarchical multi-agent reinforcement learning approaches to algorithmic trading. 

## Installation and Setup
To clone the repository, run the following commands in the terminal:
```
git clone https://github.com/olibridge01/MultiAgentTrading.git
cd MultiAgentTrading
```

To set up a conda environment with the required packages, run the following command in the terminal:
```
conda env create -n MultiAgentTrading python=3.10 -y
conda activate MultiAgentTrading

# Install requirements
pip install -r requirements.txt
```

## Running the Code
To run a single- or multi-agent experiment, navigate to the main directory and run the following command:
```
python run_dqn.py --stock [stock] --train --n_episodes [n_episodes] -tf [tf] --model [model_path] --save -v
```
```
python run_multiagentdqn.py --stock [stock] --train --n_episodes [n_episodes] --modeldir [model_dir] --save -v
```
- `[stock]` is the stock to trade (e.g. 'GOOGL')
- `[n_episodes]` is the number of episodes to train for
- `[tf]` is the time frame of the data (e.g. '1d')
- `[model_path]` is the path to save the model to
- `-v` is a verbose flag to print extra information
- `[model_dir]` is the directory containing the saved models

To evaluate a saved model, run the following command:
```
python evaluate.py --stock [stock] -tf [tf] --modeldir [model_dir] --save -v
```
Hyperparameters can be adjusted in the `run_dqn.py` and `run_multiagentdqn.py` files.