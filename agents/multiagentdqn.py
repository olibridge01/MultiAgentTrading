import gymnasium as gym
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import count
from typing import List, Tuple, Dict

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from utils.utils import Config, ExperienceReplay, data_preprocessing
from agents.base_agent import BaseAgent
from utils.networks import DQN_Net


class MultiAgentDQN(BaseAgent):
    """
    Multi-agent DQN framework for single stock trading.

    Agent features:
    - Epsilon-greedy policy.
    - Experience replay for each agent.
    - Target network for each agent to stabilise learning.
    - Episode-based training.
    - Hierarchical flow of information from higher timeframe agents to lower timeframe agents.
    """
    def __init__(self, config: Config):
        super(MultiAgentDQN, self).__init__(config)
        self.num_agents = self.hyperparameters['num_agents']
        self.trading_windows = self.hyperparameters['trading_windows']

        # Initialize current actions dictionary
        self.current_actions = {f'agent_{i}': None for i in range(self.num_agents)}

        # Initialize current states dictionary
        self.current_states = {f'agent_{i}': None for i in range(self.num_agents)}

        # Initialize current states dictionary
        self.current_rewards = {f'agent_{i}': None for i in range(self.num_agents)}

        # Create a memory buffer for each agent
        self.memories = {f'agent_{i}': ExperienceReplay(buffer_size=self.hyperparameters['buffer_size']) for i in range(self.num_agents)}

        # Create a policy and target network for each agent
        self.policy_nets = {f'agent_{i}': DQN_Net(
            input_dims=self.env.observation_space.shape[0] - self.num_agents + i + 1,
            n_actions=self.n_actions,
            hidden_dims=self.hyperparameters['hidden_dims'],
            device=self.device
        ) for i in range(self.num_agents)}

        self.target_nets = {f'agent_{i}': DQN_Net(
            input_dims=self.env.observation_space.shape[0] - self.num_agents + i + 1,
            n_actions=self.n_actions,
            hidden_dims=self.hyperparameters['hidden_dims'],
            device=self.device
        ) for i in range(self.num_agents)}

        # Copy the weights from the policy networks to the target networks
        for agent in self.policy_nets:
            self.target_nets[agent].load_state_dict(self.policy_nets[agent].state_dict())
        
        # Create an optimizer for each policy network
        self.optimizers = {f'agent_{i}': torch.optim.Adam(self.policy_nets[f'agent_{i}'].parameters(), lr=self.hyperparameters['learning_rate']) for i in range(self.num_agents)}

        # Use the same loss function for each agent
        self.losses = {f'agent_{i}': nn.SmoothL1Loss() for i in range(self.num_agents)}

        # Epsilon decay for epsilon-greedy policy
        self.eps_start = self.hyperparameters['epsilon_start']
        self.eps_end = self.hyperparameters['epsilon_end']
        self.eps_decay = self.hyperparameters['epsilon_decay']
        self.steps_done = 0

    def get_action(self, state: torch.Tensor, agent: str) -> int:
        """
        Select an action for a specific agent based on epsilon-greedy policy.
        
        Args:
        - state (torch.Tensor): Current state of the environment.
        - agent (str): Name of the agent.

        Returns:
        - int: Action to take in the environment.
        """

        policy_net = self.policy_nets[agent]

        # Decay epsilon based on number of steps taken
        eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * np.exp(-self.steps_done / self.eps_decay)
        self.steps_done += 1

        if np.random.rand() < eps_threshold:
            action = torch.tensor([[np.random.randint(self.n_actions)]], device=self.device, dtype=torch.long)
        else:
            with torch.no_grad():
                policy_net.eval()
                #print(state.shape)
                action = policy_net(state).max(1).indices.view(1, 1)
                # last_layer_params = list(policy_net.parameters())[-1]
                #print(agent, 's policy net final layer parameters when getting an action:', last_layer_params)
                policy_net.train()

        # Update the current action for the agent
        self.current_actions[agent] = action
        #print(agent, 's action:', action.item())

        return action
        
    def _update(self, agent: str):
        """Takes an optimization step for a specific agent."""

        memory = self.memories[agent]
        policy_net = self.policy_nets[agent]
        optimizer = self.optimizers[agent]

        if len(memory) < self.hyperparameters['batch_size']:
            return

        # Sample batch of experiences from memory and compute loss
        experiences = memory.sample(batch_size=self.hyperparameters['batch_size'])
        loss = self._compute_loss(experiences, agent)

        # Perform optimization step
        optimizer.zero_grad()
        loss.backward()

        # Print agent's last layer parameters before optimization step
        last_layer_params = list(policy_net.parameters())[-1]
        #print(agent, 's policy net final layer parameters before optimization step:', last_layer_params)  

        # Clamp gradients to prevent exploding gradients
        for param in policy_net.parameters():
            param.grad.data.clamp_(-1, 1)

        optimizer.step()

        # Print agent's last layer parameters after optimization step
        last_layer_params = list(policy_net.parameters())[-1]
        #print(agent, 's policy net final layer parameters after optimization step:', last_layer_params)

    def _compute_loss(self, experiences: tuple, agent: str) -> torch.Tensor:
        """
        Compute the loss for a specific agent.

        Args:
        - experiences (tuple): Tuple of experiences to compute the loss for.
        - agent (str): Name of the agent.

        Returns:
        - torch.Tensor: Loss.
        """

        # Get the policy network, target network, and loss function for the agent
        policy_net = self.policy_nets[agent]
        # print the last layer parameters of the policy net of the agent during the computation of the loss
        last_layer_params = list(policy_net.parameters())[-1]   
        #print(agent, 's policy net final layer parameters during the computation of the loss:', last_layer_params)
        target_net = self.target_nets[agent]
        # print the last layer parameters of the target net of the agent during the computation of the loss
        last_layer_params = list(target_net.parameters())[-1]
        #print(agent, 's target net final layer parameters during the computation of the loss:', last_layer_params)

        loss = self.losses[agent]

        # Unpack batch of experiences
        states = torch.cat(experiences.state)
        actions = torch.cat(experiences.action)
        rewards = torch.cat(experiences.reward)

        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, experiences.next_state)), device=self.device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in experiences.next_state if s is not None])

        # Compute Q values for current states and actions
        q_values = policy_net(states).gather(1, actions)

        # Compute Q values for next states
        next_q_values = torch.zeros(self.hyperparameters['batch_size'], device=self.device)
        next_q_values[non_final_mask] = target_net(non_final_next_states).max(1).values.detach()

        # Compute target Q values
        target_q_values = torch.unsqueeze(rewards + (self.hyperparameters['gamma'] ** self.hyperparameters['n_steps'] * next_q_values),1)

        # Compute loss
        loss_val = loss(q_values, target_q_values)

        return loss_val

    def train(self, n_episodes: int, model_dir: str) -> list:
        """
        Train the agent with DQN algorithm.
        
        Args:
        - n_episodes (int): Number of episodes to train the agent for.
        - model_dir (str): Directory to save the trained models.

        Returns:
        - list: List of rewards obtained during training.
        """
        # Initialise training hyperparameters
        self.target_update = self.hyperparameters['target_update']
        
        rewards = [] # List to store rewards obtained during training

        # print the final agent's policy network
        # final_agent = list(self.policy_nets.keys())[-1]
        #print(self.policy_nets[final_agent])

        print(f'Training multi-agent framework for {n_episodes} episodes...')
        
        for j in range(n_episodes):
            print(f'Running episode {j+1}/{n_episodes}')

            # Reset environment and get initial state
            states, info = self.env.reset()

            # Convert states to PyTorch tensors and update them in the current_states dictionary
            for i, agent in enumerate(self.current_states.keys()):
                state = torch.tensor(states[i], dtype=torch.float32).unsqueeze(0).to(self.device)
                self.current_states[agent] = state

            # Run through episode
            for t in count():

                # Select and perform an action for each agent
                for i, agent in enumerate(self.current_actions.keys()):

                    memory = self.memories[agent]

                    x = self.current_states[agent][:, :self.env.observation_space.shape[0] - self.num_agents + i + 1]

                    # print(x)
                    action = self.get_action(x, agent)
                    next_state, reward, terminated, _, _ = self.env.step(action.item())
                    self.current_rewards[agent] = reward
                    reward = torch.tensor([reward], device=self.device, dtype=torch.float32)

                    if terminated:
                        next_state = None
                    else:
                        next_state = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0).to(self.device)
                        next_state = next_state[:, :self.env.observation_space.shape[0] - self.num_agents + i + 1]

                    # print the next state produced by the environment for the agent 
                    #print(agent, 's next state:', next_state)

                    # Add reward to list of rewards if final agent
                    if i == self.num_agents - 1:
                        rewards.append(reward.item())

                    memory.add_experience(x, action, next_state, reward)

                    # Update the policy network
                    self._update(agent)
                
                # if terminated is True, break
                if terminated and i == self.num_agents - 1:
                    break
                
                # Loop to make all agents interact with the environment and get the next state
                current_step = self.env.unwrapped.current_step
                counter = self.env.unwrapped.counter

                # Get the new current state for each agent
                for k, agent in enumerate(self.current_states.keys()):
                    self.env.counter = k
                    state = self.env.unwrapped.get_current_state()
                    self.current_states[agent] = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)

                # Restore the original values of current_step and counter
                self.env.unwrapped.current_step = current_step
                self.env.unwrapped.counter = counter

            # Update all agents the target network every `target_update` episodes
            if j % self.target_update == 0:
                for agent in self.policy_nets:
                    self.target_nets[agent].load_state_dict(self.policy_nets[agent].state_dict())

        # Save policy networks
        for i, agent in enumerate(self.policy_nets.keys()):
            torch.save(self.policy_nets[agent].state_dict(), f'{model_dir}/multi_agent_policy_net_{i}.pkl')
        
        return rewards
    
    def generate_modelpaths(self, model_dir: str) -> List[str]:
        """
        Generate paths to the models for each agent.

        Args:
        - model_dir (str): Directory containing the trained models.
        
        Returns:
        - list: List of paths to the models for each agent.
        """

        model_paths = [f'{model_dir}/multi_agent_policy_net_{i}.pkl' for i in range(self.num_agents)]
        return model_paths

    def test(self, data: pd.DataFrame, model_dir: str) -> list:
        """
        Test the agent with DQN algorithm.
        
        Args:
        - data (pd.DataFrame): Data to test the model on.
        - model_dir (str): Directory containing the trained models.

        Returns:
        - list: List of actions taken by the agent during the testing period.
        """
        model_paths = self.generate_modelpaths(model_dir)

        self.test_nets = {f'agent_{i}': DQN_Net(
            input_dims=self.env.observation_space.shape[0] - self.num_agents + i + 1,
            n_actions=self.n_actions,
            hidden_dims=self.hyperparameters['hidden_dims'],
            device=self.device
        ) for i in range(self.num_agents)}

        # Load the models for each agent
        for i, agent in enumerate(self.test_nets.keys()):
            self.test_nets[agent].load_state_dict(torch.load(model_paths[i]))

        actions = [] # List to store actions taken by the final agent throughout testing
        test_batch = 5

        # Initialize current actions as hold for all agents
        agent_data = {f'agent_{i}': data_preprocessing(data, self.trading_windows[i]) for i in range(self.num_agents)}

        # Initialize current states dictionary
        self.current_states = {f'agent_{i}': None for i in range(self.num_agents)}

        for i in range(np.max(self.trading_windows) - 1, len(data), test_batch):
            if i + test_batch > len(data):
                break
            for j, agent in enumerate(self.current_states.keys()):
                self.current_states[agent] = torch.tensor(agent_data[agent].iloc[i:i+test_batch].values, dtype=torch.float32).to(self.device)

            actions_batches = torch.Tensor([])

            for j, agent in enumerate(self.current_states.keys()):
                try:
                    x = torch.cat([self.current_states[agent], actions_batches], dim=1)
                    action = self.test_nets[agent](x).max(1)[1]
                    actions_batches = torch.cat([actions_batches, action.unsqueeze(1)], dim=1)

                except ValueError:
                    self.actions_taken[j] = 1

                if j == self.num_agents - 1:
                    actions += list(action.cpu().numpy()) # Append the action of the final agent to the actions list

        return actions        