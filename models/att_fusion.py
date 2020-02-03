import torch
import torch.nn as nn
import torch.nn.functional as F

from .rnn import GRU


class AttFusion(nn.Module):
    def __init__(self, input_dim=512, hidden_dim=128):
        super(AttFusion, self).__init__()
        self.scorer_a = GRU(input_dim, hidden_dim, 1, 1, 1)
        self.scorer_v = GRU(input_dim, hidden_dim, 1, 1, 1)

    def forward(self, x_a, x_v):
        h_v = torch.sigmoid(self.scorer_v(x_v))
        h_a = torch.sigmoid(self.scorer_a(x_a))
        h = torch.cat((h_v, h_a), dim=-1) # (B, T, 2)
        h = F.softmax(h, dim=-1)
        f = h[..., 0].unsqueeze(-1) * x_v + h[..., 1].unsqueeze(-1) * x_a
        
        return f
