"""
Dense MoE 모델
Fully Connected Layer 기반 Gating Network 사용
비강화학습 방식의 베이스라인 모델
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .base_moe import BaseMoEModel


class DenseMoE(BaseMoEModel):
    """
    Dense Gating Network를 사용하는 MoE 모델

    Gating Network가 입력(state)을 받아서 각 Expert의 가중치를 계산하고,
    모든 Expert의 출력을 가중합하여 최종 예측을 생성합니다.

    Args:
        num_users (int): 사용자 수
        num_movies (int): 영화 수
        num_age_groups (int): 나이 그룹 수 (기본값: 8)
        num_occupations (int): 직업 수
        embedding_dim (int): 임베딩 차원 (기본값: 64)
        num_experts (int): Expert 개수 (기본값: 8)
        expert_hidden_dim (int): Expert 은닉층 차원 (기본값: 256)
        gating_hidden_dim (int): Gating Network 은닉층 차원 (기본값: 128)
        dropout (float): 드롭아웃 비율 (기본값: 0.1)
    """
    def __init__(
        self,
        num_users,
        num_movies,
        embedding_dim=64,
        num_experts=8,
        expert_hidden_dim=256,
        gating_hidden_dim=128,
        dropout=0.1
    ):
        super(DenseMoE, self).__init__(
            num_users=num_users,
            num_movies=num_movies,
            embedding_dim=embedding_dim,
            num_experts=num_experts,
            expert_hidden_dim=expert_hidden_dim,
            dropout=dropout
        )

        # Gating Network (FC Layer 기반)
        input_dim = embedding_dim * 2  # user + movie 임베딩
        self.gating_network = nn.Sequential(
            nn.Linear(input_dim, gating_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(gating_hidden_dim, gating_hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(gating_hidden_dim // 2, num_experts)
        )

    def forward(self, user_id, movie_id):
        """
        순전파

        Args:
            user_id (torch.Tensor): [batch_size]
            movie_id (torch.Tensor): [batch_size]
            age_group (torch.Tensor): [batch_size]
            gender (torch.Tensor): [batch_size]
            occupation (torch.Tensor): [batch_size]

        Returns:
            dict: {
                'rating': 예측 평점 [batch_size],
                'gate_logits': Gating logits [batch_size, num_experts],
                'gate_probs': Gating 확률 [batch_size, num_experts],
                'expert_outputs': 각 Expert의 출력 [batch_size, num_experts, 1]
            }
        """
        # State 생성 (User + Movie 임베딩)
        state = self.get_state(user_id, movie_id)

        # Gating Network로 Expert 가중치 계산
        gate_logits = self.gating_network(state)  # [batch_size, num_experts]
        gate_probs = F.softmax(gate_logits, dim=-1)  # [batch_size, num_experts]

        # 모든 Expert의 출력 계산
        expert_outputs = self.experts(state, expert_indices=None)  # [batch_size, num_experts, 1]

        # Gating 가중치와 Expert 출력의 가중합
        # gate_probs: [batch_size, num_experts] -> [batch_size, num_experts, 1]
        gate_probs_expanded = gate_probs.unsqueeze(-1)

        # 가중합 계산
        weighted_outputs = expert_outputs * gate_probs_expanded  # [batch_size, num_experts, 1]
        final_output = weighted_outputs.sum(dim=1).squeeze(-1)  # [batch_size]

        # 평점 범위로 조정 (sigmoid로 0-1 범위로 변환)
        rating = torch.sigmoid(final_output)

        return {
            'rating': rating,
            'gate_logits': gate_logits,
            'gate_probs': gate_probs,
            'expert_outputs': expert_outputs
        }

    def compute_loss(self, predictions, targets):
        """
        손실 함수 계산 (MSE Loss)

        Args:
            predictions (dict): forward() 출력
            targets (torch.Tensor): 정규화된 실제 평점 [batch_size]

        Returns:
            torch.Tensor: 손실 값
        """
        predicted_ratings = predictions['rating']
        mse_loss = F.mse_loss(predicted_ratings, targets)

        return mse_loss
