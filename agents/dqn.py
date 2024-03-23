import gymnasium as gym
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import count
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from utils.utils import Config, ExperienceReplay
from agents.base_agent import BaseAgent
from utils.networks import QNetworkMLP, DQN_Net, DQN_Net2


class DQN(BaseAgent):
    """
    Single DQN agent.
    
    THIS IS THE AGENT CURRENTLY BEING USED
    """
    def __init__(self, config: Config):
        super(DQN, self).__init__(config)
        self.memory = ExperienceReplay(buffer_size=self.hyperparameters['buffer_size'])
        
        self.policy_net = DQN_Net(
            input_dims=self.env.observation_space.shape[0],
            n_actions=self.n_actions,
            hidden_dims=self.hyperparameters['hidden_dims'],
            device=self.device
        )
        
        self.target_net = DQN_Net(
            input_dims=self.env.observation_space.shape[0],
            n_actions=self.n_actions,
            hidden_dims=self.hyperparameters['hidden_dims'],
            device=self.device
        )
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        self.epsilon = self.hyperparameters['epsilon']
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.hyperparameters['learning_rate'])
        self.loss = nn.SmoothL1Loss()

        # HARD-CODED EPSILON DECAY - REPLICATING PAPER
        self.EPS_START = 0.9
        self.EPS_END = 0.05
        self.EPS_DECAY = 500

        self.steps_done = 0

    def get_action(self, state: torch.Tensor) -> int:
        """
        Select an action based on epsilon-greedy policy.
        
        Args:
        - state (torch.Tensor): Current state of the environment.

        Returns:
        - int: Action to take in the environment.
        """

        eps_threshold = self.EPS_END + (self.EPS_START - self.EPS_END) * \
                        np.exp(-1. * self.steps_done / self.EPS_DECAY)
        self.steps_done += 1


        if np.random.rand() < eps_threshold:
            self.current_action = torch.tensor([[np.random.randint(self.n_actions)]], device=self.device, dtype=torch.long)

        else:
            with torch.no_grad():
                self.policy_net.eval()
                self.current_action = self.policy_net(state).max(1).indices.view(1, 1)
                self.policy_net.train()
        
        return self.current_action
        
            
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

        # Clamp gradients to prevent exploding gradients
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)

        self.optimizer.step()

    def _compute_loss(self, experiences: tuple) -> torch.Tensor:
        """
        Computes the loss for a batch of experiences.
        
        Args:
        - experiences (tuple): Tuple of experiences to compute the loss for.

        Returns:
        - torch.Tensor: Loss for the batch of experiences.
        """

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
        Q_target = (rewards + self.hyperparameters['gamma'] ** self.hyperparameters['n_steps'] * Q_next).unsqueeze(1)

        # Compute Q-values for the policy network
        Q_policy = self.policy_net(states).gather(1, actions)

        # Compute loss
        loss = self.loss(Q_policy, Q_target)

        return loss

    def train(self, n_episodes: int) -> list:
        """
        Train the agent with DQN algorithm.
        
        Args:
        - n_episodes (int): Number of episodes to train the agent for.

        Returns:
        - list: List of rewards obtained during training.
        """

        # Initialise training hyperparameters
        self.epsilon_decay = self.hyperparameters['epsilon_decay']
        self.target_update = self.hyperparameters['target_update']

        rewards = []

        print(self.policy_net)

        for i in range(n_episodes):
            print(f'Running episode {i+1}/{n_episodes}')

            # Reset environment and get initial state
            obs, info = self.env.reset()

            # Convert observation to state torch.Tensor
            state = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(self.device)

            # Run through episode
            for t in count():

                # Decay epsilon
                # Currently redundant whilst using hard-coded epsilon decay from paper
                # self.epsilon *= self.epsilon_decay

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
                state = self.env.unwrapped.get_current_state()
                state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)

                # Update the policy network
                self._update()

                if terminated:
                    break

            # Update the target network every `target_update` timesteps
            if i % self.target_update == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())
        
        # Save policy network
        torch.save(self.policy_net.state_dict(), 'models/policy_net.pkl')
        return rewards
    
    def test(self, model_path: str, data: pd.DataFrame) -> list:
        """
        Test the agent with DQN algorithm.
        
        Args:
        - model_path (str): Path to the model to test.
        - data (pd.DataFrame): Data to test the model on.

        Returns:
        - list: List of actions taken by the agent during the testing period.
        """

        self.test_net = DQN_Net(
            input_dims=self.env.observation_space.shape[0],
            n_actions=self.n_actions,
            hidden_dims=self.hyperparameters['hidden_dims'],
            device=self.device
        )
        self.test_net.load_state_dict(torch.load(model_path))

        # obs, info = self.env.reset()
        # state = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(self.device)
        actions = []

        # for t in count():
        #     with torch.no_grad():
        #         self.test_net.eval()
        #         action = self.test_net(state).max(1)[1].view(1, 1)
        #         self.test_net.train()

        #     actions.append(action.item())
        #     obs, reward, terminated, _, info = self.env.step(action.item())
        #     if terminated:
        #         break

        #     state = self.env.unwrapped.get_current_state()
        #     state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)

        # Split data into batches of 10
        for i in range(0, len(data), 10):
            state = torch.tensor(data.iloc[i:i+10].values, dtype=torch.float32).to(self.device)
            try:
                action_batch = self.test_net(state).max(1)[1]
                actions += list(action_batch.cpu().numpy())

            except ValueError:
                actions += [1]

        return actions