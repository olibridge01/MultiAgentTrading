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
    

class DQN_Paper(nn.Module):
    """
    DQN MLP used in the single-stock trading paper https://arxiv.org/abs/2010.14194

    For use when testing with model weights from their code
    """

    def __init__(self, state_length, action_length):
        super(DQN_Paper, self).__init__()
        self.policy_network = nn.Sequential(
            nn.Linear(state_length, 128),
            nn.BatchNorm1d(128),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.Linear(256, action_length))

        # self.layer1 = nn.Linear(state_length, 128)
        # self.bn1 = nn.BatchNorm1d(128)
        # self.layer2 = nn.Linear(128, 256)
        # self.bn2 = nn.BatchNorm1d(256)
        # self.out = nn.Linear(256, action_length)

    def forward(self, x):
        # if x.shape[0] > 1:
        #     x = F.relu(self.bn1(self.layer1(x)))
        #     x = F.relu(self.bn2(self.layer2(x)))
        # else:
        #     x = F.relu(self.layer1(x))
        #     x = F.relu(self.layer2(x))
        # return self.out(x)
        return self.policy_network(x)