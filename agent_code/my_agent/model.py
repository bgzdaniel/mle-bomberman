import torch
import torch.nn.functional as F
import torch.nn as nn

class Model(nn.Module):
    """ Simple placeholder neural network to use until we finalize architecture. """

    def __init__(self, n_features,  n_actions, hidden_size=128):
        super(Model, self).__init__()
        self.fc1 = nn.Linear(n_features, hidden_size)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.relu3 = nn.ReLU()
        self.fc4 = nn.Linear(hidden_size, n_actions)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        x = self.relu3(x)
        x = self.fc4(x)
        return x

