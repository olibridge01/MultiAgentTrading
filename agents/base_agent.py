import gymnasium as gym
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import count

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from utils.utils import Config

class BaseAgent(object):
    """
    Base class for reinforcement learning algorithm implementations.

    To-Do:
        - Add more base methods as code develops/implement more than just DQN.
    """

    def __init__(self, config: Config):
        self.env = config.environment
        self.hyperparameters = config.hyperparameters
        self.device = 'cuda:0' if config.GPU else 'cpu'
        self.n_actions = self.env.action_space.n
    
    def get_action(self, state: torch.Tensor) -> int:
        """Select an action based on agent's policy."""
        raise NotImplementedError
    
    def _update(self):
        """Takes an optimization step."""
        raise NotImplementedError
    
    def train(self):
        """Train the agent with given algorithm."""
        raise NotImplementedError