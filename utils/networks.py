import torch
import torch.nn as nn

class QNetworkMLP(nn.Module):
    def __init__(
        self,
        input_dims: int,
        n_actions: int,
        hidden_dims: list = [256, 256],
        activation: nn.Module = nn.ReLU(),
        activation_last_layer: bool = False,
        device: str = 'cpu'
    ):
        super(QNetworkMLP, self).__init__()
        self.input_dims = input_dims
        self.n_actions = n_actions
        self.hidden_dims = hidden_dims
        self.activation = activation
        
        self.device = device
        
        network = []
        dims = [self.input_dims] + self.hidden_dims + [self.n_actions]
        for i in range(len(dims) - 1):
            network.append(nn.Linear(dims[i], dims[i+1]))
            if i < len(dims) - 2 or activation_last_layer:
                network.append(self.activation)

        self.network = nn.Sequential(*network)
        self.to(self.device)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.network(state)