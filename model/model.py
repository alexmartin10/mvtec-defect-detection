from torch import nn
import torch

class NeuralNetwork(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.flatten = nn.Flatten(start_dim=2)
        self.seq = nn.Sequential(
            nn.Linear(900*900, 512),
            nn.ReLU(),
            nn.Linear(512, 64),
            nn.ReLU(),
            nn.Linear(64, 512),
            nn.ReLU(),
            nn.Linear(512, 900*900)
        )

    def forward(self, x : torch.Tensor):
        x = self.flatten(x)
        x = self.seq(x)
        return x.view(-1, 3, 900, 900)
