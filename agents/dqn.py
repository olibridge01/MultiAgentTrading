import gymnasium as gym
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import count

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from utils.utils import Config, ExperienceReplay
from agents.base_agent import BaseAgent
from utils.networks import QNetworkMLP

class DQN(BaseAgent):
    """
    Deep Q-Network (DQN) agent implementation.
    """
    
    def __init__(self, config: Config):
        super(DQN, self).__init__(config)
        self.memory = ExperienceReplay(buffer_size=self.hyperparameters['buffer_size'])
        
        self.policy_net = QNetworkMLP(
            input_dims=self.env.observation_space.shape[0],
            n_actions=self.n_actions,
            hidden_dims=self.hyperparameters['hidden_dims'],
            activation=nn.ReLU(),
            activation_last_layer=self.hyperparameters['activation_last_layer'],
            device=self.device
        )
        
        self.target_net = QNetworkMLP(
            input_dims=self.env.observation_space.shape[0],
            n_actions=self.n_actions,
            hidden_dims=self.hyperparameters['hidden_dims'],
            activation=nn.ReLU(),
            activation_last_layer=self.hyperparameters['activation_last_layer'],
            device=self.device
        )
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.hyperparameters['learning_rate'])
        self.loss = nn.MSELoss()

    def get_action(self, state: torch.Tensor) -> int:
        """Select an action based on epsilon-greedy policy."""
        if np.random.rand() < self.epsilon:
            return torch.Tensor([[np.random.randint(self.n_actions)]], device=self.device, dtype=torch.long)
        else:
            with torch.no_grad():
                return self.policy_net(state).argmax().item()
            
    def _update(self):
        """Takes an optimization step."""
        if len(self.memory) < self.hyperparameters['batch_size']:
            return
        
        experiences = self.memory.sample(batch_size=self.hyperparameters['batch_size'])
        loss = self._compute_loss(experiences)
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), self.hyperparameters['gradient_clipping'])
        self.optimizer.step()

    def _compute_loss(self, experiences: tuple) -> torch.Tensor:
        pass
        
    def train(self) -> list:
        pass