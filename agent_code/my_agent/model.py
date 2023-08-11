import torch
import torch.nn.functional as F
import torch.nn as nn

class Model(nn.Module):

    def __init__(self, n_features, n_actions):
        super(Model, self).__init__()
        self.layer1 = nn.Linear(n_features, 256)
        self.layer2 = nn.Linear(256, 256)
        self.layer3 = nn.Linear(256, n_actions)

    def forward(self, input):
        input = F.relu(self.layer1(input))
        input = F.relu(self.layer2(input))
        return self.layer3(input)
