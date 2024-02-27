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
    """Single DQN agent."""
    def __init__(self, config: Config):
        super(DQN, self).__init__(config)
        self.memory = ExperienceReplay(buffer_size=self.hyperparameters['buffer_size'])
        
        self.policy_net = QNetworkMLP(
            input_dims=self.env.observation_space.shape[0],
            n_actions=self.n_actions,
            hidden_dims=self.hyperparameters['hidden_dims'],
            activation=nn.Tanh(),
            activation_last_layer=self.hyperparameters['activation_last_layer'],
            device=self.device
        )
        
        self.target_net = QNetworkMLP(
            input_dims=self.env.observation_space.shape[0],
            n_actions=self.n_actions,
            hidden_dims=self.hyperparameters['hidden_dims'],
            activation=nn.Tanh(),
            activation_last_layer=self.hyperparameters['activation_last_layer'],
            device=self.device
        )
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.hyperparameters['learning_rate'])
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
        
        # Sample batch of experiences from memory and compute loss
        experiences = self.memory.sample(batch_size=self.hyperparameters['batch_size'])
        loss = self._compute_loss(experiences)
        
        # Perform optimization step
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.hyperparameters['gradient_clipping'])
        self.optimizer.step()

    def _compute_loss(self, experiences: tuple) -> torch.Tensor:
        """Computes the loss for a batch of experiences."""

        # Unpack batch of experiences
        states = torch.cat(experiences.state)
        actions = torch.cat(experiences.action)
        rewards = torch.cat(experiences.reward)

        non_final_mask = torch.tensor(
            tuple(map(lambda s: s is not None, experiences.next_state)),
            device=self.device,
            dtype=torch.bool
        )
        non_final_next_states = torch.cat([s for s in experiences.next_state if s is not None])

        # Compute Q-values for the target network, accounting for terminal states
        Q_next = torch.zeros(self.hyperparameters['batch_size'], device=self.device)
        Q_next[non_final_mask] = self.target_net(non_final_next_states).max(1).values.detach()
        Q_target = (rewards + self.hyperparameters['gamma'] * Q_next).unsqueeze(1)

        # Compute Q-values for the policy network
        Q_policy = self.policy_net(states).gather(1, actions)

        # Compute loss
        loss = self.loss(Q_policy, Q_target)

        return loss

    def train(self) -> list:
        """Train the agent with DQN algorithm."""

        # Initialise training hyperparameters
        self.epsilon = self.hyperparameters['epsilon']
        self.epsilon_decay = self.hyperparameters['epsilon_decay']
        self.target_update = self.hyperparameters['target_update']

        rewards = []

        # Reset environment and get initial state
        obs, info = self.env.reset()

        # Convert observation to state torch.Tensor
        state = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(self.device)

        # Train non-episodic task
        for t in count():
            # Select and perform an action
            action = self.get_action(state)
            obs, reward, terminated, _, info = self.env.step(action.item())
            rewards.append(reward)
            reward = torch.tensor([reward], device=self.device, dtype=torch.float32)

            if terminated:
                next_state = None
            else:
                next_state = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(self.device)
            
            # Store experience in memory
            self.memory.add_experience(state, action, next_state, reward)

            # Move to the next state
            state = next_state

            # Update the policy network
            self._update()

            # Update the target network every `target_update` timesteps
            if t % self.target_update == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())

            if terminated:
                # Get the final balance history if the environment has terminated
                balance_history = self.env.balance_history
                break

        return balance_history, rewards