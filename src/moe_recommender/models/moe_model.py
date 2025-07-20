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

        gate_logits = self.gating(x)
        expert_idx = torch.argmax(gate_logits, dim=1)  # [batch_size]

        outputs = torch.zeros(x.size(0), device=x.device)

        for idx, expert in enumerate(self.experts):
            mask = (expert_idx == idx)
            if mask.any():
                x_selected = x[mask]
                out = expert(x_selected).squeeze()
                outputs[mask] = out

        return outputs