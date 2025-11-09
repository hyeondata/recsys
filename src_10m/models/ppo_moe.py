"""
PPO-MoE Model for 10M dataset (scaled up architecture).
"""
import torch
import torch.nn as nn
from .base_moe import BaseMoE
from .expert_network import ExpertNetwork


class PPOMoE(BaseMoE):
    """
    PPO-based MoE with RL Expert Selection (scaled up for 10M dataset).
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

        # Policy network (expert selection)
        policy_input_dim = self.state_dim
        if num_genres > 0:
            policy_input_dim += embedding_dim

        self.policy = nn.Sequential(
            nn.Linear(policy_input_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_experts)
        )

        # Value network (for advantage estimation)
        self.value = nn.Sequential(
            nn.Linear(policy_input_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1)
        )

    def forward(
        self,
        user_ids: torch.Tensor,
        movie_ids: torch.Tensor,
        genres: torch.Tensor = None,
        expert_indices: torch.Tensor = None
    ) -> dict:
        """
        Forward pass.

        Args:
            user_ids: User IDs [batch_size]
            movie_ids: Movie IDs [batch_size]
            genres: Genre features [batch_size, num_genres]
            expert_indices: Pre-selected expert indices [batch_size] (optional)

        Returns:
            Dictionary with:
                - predictions: Rating predictions [batch_size]
                - log_probs: Log probabilities of selected experts [batch_size]
                - values: Value estimates [batch_size]
                - expert_indices: Selected expert indices [batch_size]
        """
        batch_size = user_ids.size(0)

        # Get state
        state, genre_features = self.get_state(user_ids, movie_ids, genres)

        # Policy input
        if genre_features is not None:
            policy_input = torch.cat([state, genre_features], dim=1)
        else:
            policy_input = state

        # Policy network
        policy_logits = self.policy(policy_input)  # [batch_size, num_experts]
        policy_dist = torch.distributions.Categorical(logits=policy_logits)

        # Select experts
        if expert_indices is None:
            expert_indices = policy_dist.sample()  # [batch_size]

        log_probs = policy_dist.log_prob(expert_indices)  # [batch_size]

        # Value network
        values = self.value(policy_input).squeeze(-1)  # [batch_size]

        # Get expert outputs (벡터화 방식 - expert별 그룹화)
        predictions = torch.zeros(batch_size, device=state.device)

        # Expert별로 그룹화하여 처리 (GPU 병렬화)
        for expert_idx in range(self.num_experts):
            # 이 expert를 선택한 샘플들의 마스크
            mask = (expert_indices == expert_idx)

            if mask.any():
                # 선택된 샘플들만 해당 expert에 통과
                selected_genres = genres[mask] if genres is not None else None
                expert_output = self.experts[expert_idx](state[mask], selected_genres)
                predictions[mask] = expert_output.squeeze(-1)

        # Clip predictions
        predictions = torch.clamp(predictions, 1.0, 5.0)

        return {
            'predictions': predictions,
            'log_probs': log_probs,
            'values': values,
            'expert_indices': expert_indices
        }
