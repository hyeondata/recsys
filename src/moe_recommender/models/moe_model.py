import torch.nn as nn
import torch
from .gating import GatingNetwork
from .expert import Expert

class MoEModel(nn.Module):
    def __init__(self, num_users, num_items, movie_feat_dim, embed_dim, num_experts):
        super().__init__()
        self.user_emb = nn.Embedding(num_users, embed_dim)
        self.item_emb = nn.Embedding(num_items, embed_dim)

        input_dim = embed_dim * 2 + movie_feat_dim
        self.experts = nn.ModuleList([Expert(input_dim) for _ in range(num_experts)])
        self.gating = GatingNetwork(input_dim, num_experts)

    def forward(self, user_ids, item_ids, movie_feats):
        u = self.user_emb(user_ids)
        i = self.item_emb(item_ids)
        x = torch.cat([u, i, movie_feats], dim=1)
        gate_weights = self.gating(x)
        expert_outs = torch.stack([expert(x).squeeze(1) for expert in self.experts], dim=1)
        return torch.sum(gate_weights * expert_outs, dim=1)