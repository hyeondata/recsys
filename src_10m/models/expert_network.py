"""
Expert Network for 10M dataset (scaled up architecture).
"""
import torch
import torch.nn as nn


class ExpertNetwork(nn.Module):
    """
    Single Expert Network (scaled up for 10M dataset).

    Architecture: 256 + num_genres → 512 → 256 → 128 → 1
    """

    def __init__(self, state_dim: int = 256, num_genres: int = 0, dropout: float = 0.2):
        """
        Args:
            state_dim: Dimension of state (user_emb + movie_emb)
            num_genres: Number of genre features
            dropout: Dropout probability
        """
        super().__init__()

        input_dim = state_dim + num_genres

        self.network = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1)
        )

    def forward(self, state: torch.Tensor, genres: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass.

        Args:
            state: State tensor [batch_size, state_dim]
            genres: Genre features [batch_size, num_genres] (optional)

        Returns:
            Output tensor [batch_size, 1]
        """
        if genres is not None:
            x = torch.cat([state, genres], dim=1)
        else:
            x = state

        return self.network(x)
