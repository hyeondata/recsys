"""
Base MoE 모델
User 임베딩과 Item 임베딩을 결합하는 기본 구조
"""

import torch
import torch.nn as nn
from .expert_network import ExpertEnsemble


class BaseMoEModel(nn.Module):
    """
    MoE 기반 추천 모델의 기본 클래스

    Args:
        num_users (int): 사용자 수
        num_movies (int): 영화 수
        num_age_groups (int): 나이 그룹 수 (기본값: 8)
        num_occupations (int): 직업 수
        embedding_dim (int): 임베딩 차원 (기본값: 64)
        num_experts (int): Expert 개수 (기본값: 8)
        expert_hidden_dim (int): Expert 은닉층 차원 (기본값: 256)
        dropout (float): 드롭아웃 비율 (기본값: 0.1)
    """
    def __init__(
        self,
        num_users,
        num_movies,
        num_age_groups=8,
        num_occupations=21,
        embedding_dim=64,
        num_experts=8,
        expert_hidden_dim=256,
        dropout=0.1
    ):
        super(BaseMoEModel, self).__init__()

        self.num_experts = num_experts
        self.embedding_dim = embedding_dim

        # User 관련 임베딩
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.age_embedding = nn.Embedding(num_age_groups, embedding_dim // 4)
        self.gender_embedding = nn.Embedding(2, embedding_dim // 4)  # M:0, F:1
        self.occupation_embedding = nn.Embedding(num_occupations, embedding_dim // 4)

        # Movie 임베딩
        self.movie_embedding = nn.Embedding(num_movies, embedding_dim)

        # User 특성을 결합하는 레이어
        user_feature_dim = embedding_dim + (embedding_dim // 4) * 3
        self.user_projection = nn.Sequential(
            nn.Linear(user_feature_dim, embedding_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

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

    def get_user_embedding(self, user_id, age_group, gender, occupation):
        """
        User 임베딩 생성

        Args:
            user_id (torch.Tensor): [batch_size]
            age_group (torch.Tensor): [batch_size]
            gender (torch.Tensor): [batch_size]
            occupation (torch.Tensor): [batch_size]

        Returns:
            torch.Tensor: User 임베딩 [batch_size, embedding_dim]
        """
        user_emb = self.user_embedding(user_id)
        age_emb = self.age_embedding(age_group)
        gender_emb = self.gender_embedding(gender)
        occupation_emb = self.occupation_embedding(occupation)

        # 모든 특성 결합
        user_features = torch.cat([user_emb, age_emb, gender_emb, occupation_emb], dim=-1)

        # Projection
        user_emb_final = self.user_projection(user_features)

        return user_emb_final

    def get_movie_embedding(self, movie_id):
        """
        Movie 임베딩 생성

        Args:
            movie_id (torch.Tensor): [batch_size]

        Returns:
            torch.Tensor: Movie 임베딩 [batch_size, embedding_dim]
        """
        return self.movie_embedding(movie_id)

    def get_state(self, user_id, movie_id, age_group, gender, occupation):
        """
        강화학습의 State 생성 (User 임베딩 + Movie 임베딩)

        Args:
            user_id (torch.Tensor): [batch_size]
            movie_id (torch.Tensor): [batch_size]
            age_group (torch.Tensor): [batch_size]
            gender (torch.Tensor): [batch_size]
            occupation (torch.Tensor): [batch_size]

        Returns:
            torch.Tensor: State 벡터 [batch_size, embedding_dim * 2]
        """
        user_emb = self.get_user_embedding(user_id, age_group, gender, occupation)
        movie_emb = self.get_movie_embedding(movie_id)

        # User + Movie 임베딩 결합
        state = torch.cat([user_emb, movie_emb], dim=-1)

        return state

    def forward(self, user_id, movie_id, age_group, gender, occupation):
        """
        순전파 (하위 클래스에서 구현)

        Args:
            user_id (torch.Tensor): [batch_size]
            movie_id (torch.Tensor): [batch_size]
            age_group (torch.Tensor): [batch_size]
            gender (torch.Tensor): [batch_size]
            occupation (torch.Tensor): [batch_size]

        Returns:
            dict: 예측 평점 및 추가 정보
        """
        raise NotImplementedError("하위 클래스에서 구현해야 합니다.")
