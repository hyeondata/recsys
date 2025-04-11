import torch.nn as nn
import torch.nn.functional as F
from .attention import SelfAttention

class GatingNetwork(nn.Module):
    def __init__(self, input_dim, num_experts):
        super().__init__()
        self.attn = SelfAttention(input_dim)
        self.fc = nn.Linear(input_dim, num_experts)

    def forward(self, x):
        x = self.attn(x.unsqueeze(1)).squeeze(1)
        return F.softmax(self.fc(x), dim=-1)
