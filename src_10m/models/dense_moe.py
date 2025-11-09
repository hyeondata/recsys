"""
Dense MoE Model for 10M dataset (scaled up architecture).
"""
import torch
import torch.nn as nn
from .base_moe import BaseMoE
from .expert_network import ExpertNetwork


class DenseMoE(BaseMoE):
    """
    Dense MoE with FC Layer Gating (scaled up for 10M dataset).

    All experts are used with weighted averaging.
    """

    def __init__(
        self,
        num_users: int,
        num_movies: int,
        num_genres: int = 0,
        num_experts: int = 16,
        embedding_dim: int = 128,
        dropout: float = 0.2
    ):
        """
        Args:
            num_users: Number of users
            num_movies: Number of movies
            num_genres: Number of genre features
            num_experts: Number of expert networks
            embedding_dim: Embedding dimension
            dropout: Dropout probability
        """
        super().__init__(num_users, num_movies, num_genres, num_experts, embedding_dim, dropout)

        # Expert networks
        self.experts = nn.ModuleList([
            ExpertNetwork(self.state_dim, num_genres, dropout)
            for _ in range(num_experts)
        ])

        # Gating network
        gate_input_dim = self.state_dim
        if num_genres > 0:
            gate_input_dim += embedding_dim  # Include genre projection

        self.gate = nn.Sequential(
            nn.Linear(gate_input_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_experts)
        )

    def forward(
        self,
        user_ids: torch.Tensor,
        movie_ids: torch.Tensor,
        genres: torch.Tensor = None
    ) -> tuple:
        """
        Forward pass.

        Args:
            user_ids: User IDs [batch_size]
            movie_ids: Movie IDs [batch_size]
            genres: Genre features [batch_size, num_genres]

        Returns:
            predictions: Rating predictions [batch_size]
            gate_probs: Gate probabilities [batch_size, num_experts]
        """
        batch_size = user_ids.size(0)

        # Get state
        state, genre_features = self.get_state(user_ids, movie_ids, genres)

        # Gating
        if genre_features is not None:
            gate_input = torch.cat([state, genre_features], dim=1)
        else:
            gate_input = state

        gate_logits = self.gate(gate_input)
        gate_probs = torch.softmax(gate_logits, dim=1)  # [batch_size, num_experts]

        # Expert outputs
        expert_outputs = []
        for expert in self.experts:
            output = expert(state, genres)  # [batch_size, 1]
            expert_outputs.append(output)

        expert_outputs = torch.cat(expert_outputs, dim=1)  # [batch_size, num_experts]

        # Weighted average
        predictions = torch.sum(gate_probs * expert_outputs, dim=1)  # [batch_size]

        # Clip predictions to [1, 5]
        predictions = torch.clamp(predictions, 1.0, 5.0)

        return predictions, gate_probs
