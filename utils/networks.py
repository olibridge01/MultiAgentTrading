import torch
import torch.nn as nn

class DQN_Net(nn.Module):
    def __init__(
        self,
        input_dims: int,
        n_actions: int,
        hidden_dims: list,
        device: str = 'cpu'
    ):
        super(DQN_Net, self).__init__()
        self.input_dims = input_dims
        self.n_actions = n_actions
        self.hidden_dims = hidden_dims
        
        self.device = device
        
        network = []
        dims = [self.input_dims] + self.hidden_dims + [self.n_actions]
        for i in range(len(dims) - 1):
            network.append(nn.Linear(dims[i], dims[i+1]))
            if i < len(dims) - 2:
                network.append(nn.BatchNorm1d(dims[i+1]))

        self.network = nn.Sequential(*network)
        self.to(self.device)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.network(state)
    