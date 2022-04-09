import torch.nn as nn
import torch.nn.functional as F

class FraudModel(nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.linear_1 = nn.Linear(n_features, 256)
        self.linear_2 = nn.Linear(256, 256)
        self.linear_3 = nn.Linear(256, 256)
        self.output = nn.Linear(256,1)

    def forward(self, features):
        x = F.relu(self.linear_1(features))
        x = F.relu(self.linear_2(x))
        x = F.relu(self.linear_3(x))
        return self.output(x)
