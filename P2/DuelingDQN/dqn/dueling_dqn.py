import torch
import torch.nn as nn
import torch.nn.functional as F


class DuelingNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(DuelingNet, self).__init__()
        # Shared layers
        self.shared_layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
        )
        # Value stream
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)  # Outputs V(s)
        )
        # Advantage stream
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)  # Outputs A(s, a)
        )

    def forward(self, x):
        x = self.shared_layers(x)
        value = self.value_stream(x)
        advantage = self.advantage_stream(x)

        # Combine the value and advantage streams
        # Q(s, a) = V(s) + (A(s, a) - mean(A(s, a')))
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        return q_values
