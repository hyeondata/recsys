"""
Base MoE 모델 (ml-20m 버전)
사용자 demographic 정보 없음 - user_id와 movie_id만 사용
더 큰 임베딩 차원 (128) 및 더 많은 Expert (16)
"""

import torch
import torch.nn as nn
from .expert_network import ExpertEnsemble


class BaseMoEModel(nn.Module):
    """
    MoE 기반 추천 모델의 기본 클래스 (ml-20m 버전)

    Args:
        num_users (int): 사용자 수
        num_movies (int): 영화 수
        embedding_dim (int): 임베딩 차원 (기본값: 128, ml-20m은 더 큼)
        num_experts (int): Expert 개수 (기본값: 16, ml-20m은 더 많음)
        expert_hidden_dim (int): Expert 은닉층 차원 (기본값: 512)
        dropout (float): 드롭아웃 비율 (기본값: 0.2)
    """
    def __init__(
        self,
        num_users,
        num_movies,
        embedding_dim=128,
        num_experts=16,
        expert_hidden_dim=512,
        dropout=0.2
    ):
        super(BaseMoEModel, self).__init__()

        self.num_experts = num_experts
        self.embedding_dim = embedding_dim

        # User 임베딩 (demographic 정보 없음)
        self.user_embedding = nn.Embedding(num_users + 1, embedding_dim, padding_idx=0)

        # Movie 임베딩
        self.movie_embedding = nn.Embedding(num_movies + 1, embedding_dim, padding_idx=0)

        # 결합된 임베딩 차원 (user + movie)
        combined_dim = embedding_dim * 2

        # Expert 앙상블
        self.experts = ExpertEnsemble(
            num_experts=num_experts,
            input_dim=combined_dim,
            hidden_dim=expert_hidden_dim,
            output_dim=1,
            dropout=dropout
        )

    def get_user_embedding(self, user_id):
        """
        User 임베딩 생성 (demographic 정보 없음)

        Args:
            user_id (torch.Tensor): [batch_size]

        Returns:
            torch.Tensor: User 임베딩 [batch_size, embedding_dim]
        """
        return self.user_embedding(user_id)

    def get_movie_embedding(self, movie_id):
        """
        Movie 임베딩 생성

        Args:
            movie_id (torch.Tensor): [batch_size]

        Returns:
            torch.Tensor: Movie 임베딩 [batch_size, embedding_dim]
        """
        return self.movie_embedding(movie_id)

    def get_state(self, user_id, movie_id):
        """
        강화학습의 State 생성 (User 임베딩 + Movie 임베딩)

        Args:
            user_id (torch.Tensor): [batch_size]
            movie_id (torch.Tensor): [batch_size]

        Returns:
            torch.Tensor: State 벡터 [batch_size, embedding_dim * 2]
        """
        user_emb = self.get_user_embedding(user_id)
        movie_emb = self.get_movie_embedding(movie_id)

        # User + Movie 임베딩 결합
        state = torch.cat([user_emb, movie_emb], dim=-1)

        return state

    def forward(self, user_id, movie_id):
        """
        순전파 (하위 클래스에서 구현)

        Args:
            user_id (torch.Tensor): [batch_size]
            movie_id (torch.Tensor): [batch_size]

        Returns:
            dict: 예측 평점 및 추가 정보
        """
        raise NotImplementedError("하위 클래스에서 구현해야 합니다.")
