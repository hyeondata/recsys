"""
Base MoE Model for 10M dataset (scaled up architecture).
"""
import torch
import torch.nn as nn
from typing import Tuple


class BaseMoE(nn.Module):
    """
    Base Mixture of Experts model (scaled up for 10M dataset).

    Features:
    - User embedding: 128-dim (vs 64-dim in 100k)
    - Movie embedding: 128-dim (vs 64-dim in 100k)
    - Genre features: Multi-hot encoding
    - State: User + Movie = 256-dim (vs 128-dim in 100k)
    - Experts: 16 (vs 8 in 100k)
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
            embedding_dim: Embedding dimension (default: 128)
            dropout: Dropout probability
        """
        super().__init__()

        self.num_users = num_users
        self.num_movies = num_movies
        self.num_genres = num_genres
        self.num_experts = num_experts
        self.embedding_dim = embedding_dim
        self.state_dim = embedding_dim * 2  # user + movie

        # Embeddings
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.movie_embedding = nn.Embedding(num_movies, embedding_dim)

        # Genre projection (if used)
        if num_genres > 0:
            self.genre_projection = nn.Sequential(
                nn.Linear(num_genres, embedding_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            )
        else:
            self.genre_projection = None

        # Initialize embeddings
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.movie_embedding.weight)

    def get_state(
        self,
        user_ids: torch.Tensor,
        movie_ids: torch.Tensor,
        genres: torch.Tensor = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get state representation.

        Args:
            user_ids: User IDs [batch_size]
            movie_ids: Movie IDs [batch_size]
            genres: Genre features [batch_size, num_genres] (optional)

        Returns:
            state: State tensor [batch_size, state_dim]
            genre_features: Genre features (if provided)
        """
        # Get embeddings
        user_emb = self.user_embedding(user_ids)  # [batch_size, embedding_dim]
        movie_emb = self.movie_embedding(movie_ids)  # [batch_size, embedding_dim]

        # Combine embeddings
        state = torch.cat([user_emb, movie_emb], dim=1)  # [batch_size, state_dim]

        # Process genres if provided
        genre_features = None
        if genres is not None and self.genre_projection is not None:
            genre_features = self.genre_projection(genres)  # [batch_size, embedding_dim]

        return state, genre_features

    def forward(self, user_ids: torch.Tensor, movie_ids: torch.Tensor, genres: torch.Tensor = None):
        """To be implemented by subclasses."""
        raise NotImplementedError
