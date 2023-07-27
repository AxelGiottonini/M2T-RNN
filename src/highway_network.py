import torch
import torch.nn as nn
import torch.nn.functional as F

class HighwayCell(nn.Module):
    def __init__(self, in_features, bias=True):
        super().__init__()
        
        self.non_linear = nn.Linear(in_features, in_features, bias)
        self.gate = nn.Linear(in_features, in_features, bias)
        self.activation = nn.GELU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        non_linear = self.activation(self.non_linear(input))
        gate = self.sigmoid(self.gate(input))
        return gate * non_linear + (1-gate) * input


