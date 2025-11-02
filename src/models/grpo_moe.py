"""
GRPO-MoE 모델
GRPO(Group Relative Policy Optimization) 강화학습 기반 Gating Network
그룹 단위 상대적 보상 차이를 정규화하여 더 정교한 Expert 선택
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from .base_moe import BaseMoEModel


class GRPOMoE(BaseMoEModel):
    """
    GRPO 기반 Gating Network를 사용하는 MoE 모델

    GRPO는 그룹 내에서 상대적인 보상 차이를 활용하여 정책을 업데이트합니다.
    같은 배치 내의 샘플들을 그룹으로 보고, 그룹 평균 대비 상대적 성능을 계산합니다.

    Args:
        num_users (int): 사용자 수
        num_movies (int): 영화 수
        num_age_groups (int): 나이 그룹 수 (기본값: 8)
        num_occupations (int): 직업 수
        embedding_dim (int): 임베딩 차원 (기본값: 64)
        num_experts (int): Expert 개수 (기본값: 8)
        expert_hidden_dim (int): Expert 은닉층 차원 (기본값: 256)
        policy_hidden_dim (int): Policy Network 은닉층 차원 (기본값: 128)
        clip_epsilon (float): Clipping 파라미터 (기본값: 0.2)
        temperature (float): 소프트맥스 온도 (기본값: 1.0)
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
        policy_hidden_dim=128,
        clip_epsilon=0.2,
        temperature=1.0,
        dropout=0.1
    ):
        super(GRPOMoE, self).__init__(
            num_users=num_users,
            num_movies=num_movies,
            num_age_groups=num_age_groups,
            num_occupations=num_occupations,
            embedding_dim=embedding_dim,
            num_experts=num_experts,
            expert_hidden_dim=expert_hidden_dim,
            dropout=dropout
        )

        self.clip_epsilon = clip_epsilon
        self.temperature = temperature
        input_dim = embedding_dim * 2  # user + movie 임베딩

        # Policy Network: Expert 선택 확률 분포 생성
        self.policy_network = nn.Sequential(
            nn.Linear(input_dim, policy_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(policy_hidden_dim, policy_hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(policy_hidden_dim // 2, num_experts)
        )

        # Baseline Network: 그룹 평균 성능 예측
        self.baseline_network = nn.Sequential(
            nn.Linear(input_dim, policy_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(policy_hidden_dim, policy_hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(policy_hidden_dim // 2, 1)
        )

    def forward(self, user_id, movie_id, age_group, gender, occupation, deterministic=False):
        """
        순전파

        Args:
            user_id (torch.Tensor): [batch_size]
            movie_id (torch.Tensor): [batch_size]
            age_group (torch.Tensor): [batch_size]
            gender (torch.Tensor): [batch_size]
            occupation (torch.Tensor): [batch_size]
            deterministic (bool): True면 가장 높은 확률의 Expert 선택,
                                  False면 확률에 따라 샘플링

        Returns:
            dict: {
                'rating': 예측 평점 [batch_size],
                'action': 선택된 Expert 인덱스 [batch_size],
                'log_prob': 선택된 Action의 로그 확률 [batch_size],
                'entropy': 정책 엔트로피 [batch_size],
                'baseline': Baseline 예측 값 [batch_size],
                'action_probs': 각 Expert 선택 확률 [batch_size, num_experts]
            }
        """
        # State 생성 (User + Movie 임베딩)
        state = self.get_state(user_id, movie_id, age_group, gender, occupation)

        # Policy Network로 Expert 선택 확률 계산
        policy_logits = self.policy_network(state) / self.temperature  # [batch_size, num_experts]
        action_probs = F.softmax(policy_logits, dim=-1)  # [batch_size, num_experts]

        # Action 선택
        dist = Categorical(action_probs)

        if deterministic:
            action = action_probs.argmax(dim=-1)  # [batch_size]
        else:
            action = dist.sample()  # [batch_size]

        log_prob = dist.log_prob(action)  # [batch_size]
        entropy = dist.entropy()  # [batch_size]

        # Baseline Network로 그룹 평균 성능 예측
        baseline = self.baseline_network(state).squeeze(-1)  # [batch_size]

        # 선택된 Expert의 출력으로 평점 예측
        expert_output = self.experts(state, expert_indices=action)  # [batch_size, 1]
        rating = torch.sigmoid(expert_output.squeeze(-1))  # [batch_size]

        return {
            'rating': rating,
            'action': action,
            'log_prob': log_prob,
            'entropy': entropy,
            'baseline': baseline,
            'action_probs': action_probs
        }

    def compute_loss(
        self,
        predictions,
        targets,
        old_log_probs,
        group_rewards,
        entropy_coef=0.01,
        baseline_coef=0.5
    ):
        """
        GRPO 손실 함수 계산

        Args:
            predictions (dict): forward() 출력
            targets (torch.Tensor): 정규화된 실제 평점 [batch_size]
            old_log_probs (torch.Tensor): 이전 정책의 로그 확률 [batch_size]
            group_rewards (torch.Tensor): 그룹 상대 보상 [batch_size]
            entropy_coef (float): 엔트로피 계수
            baseline_coef (float): Baseline 손실 계수

        Returns:
            dict: {
                'total_loss': 전체 손실,
                'policy_loss': Policy 손실,
                'baseline_loss': Baseline 손실,
                'entropy_loss': 엔트로피 손실,
                'rating_loss': 평점 예측 손실 (MSE)
            }
        """
        # 그룹 상대 Advantage 계산
        # group_rewards는 이미 정규화된 상대적 보상
        advantages = group_rewards - predictions['baseline'].detach()

        # Policy Loss (GRPO Objective with Clipping)
        ratio = torch.exp(predictions['log_prob'] - old_log_probs)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()

        # Baseline Loss (MSE)
        baseline_loss = F.mse_loss(predictions['baseline'], group_rewards)

        # Entropy Loss (탐색 촉진)
        entropy_loss = -predictions['entropy'].mean()

        # Rating Prediction Loss (MSE)
        rating_loss = F.mse_loss(predictions['rating'], targets)

        # 전체 손실
        total_loss = (
            policy_loss +
            baseline_coef * baseline_loss +
            entropy_coef * entropy_loss +
            rating_loss
        )

        return {
            'total_loss': total_loss,
            'policy_loss': policy_loss,
            'baseline_loss': baseline_loss,
            'entropy_loss': entropy_loss,
            'rating_loss': rating_loss
        }

    def compute_group_relative_rewards(self, rewards):
        """
        그룹 상대 보상 계산

        배치 내에서 평균과 표준편차를 계산하고,
        각 샘플의 보상을 정규화하여 상대적 성능을 반영

        Args:
            rewards (torch.Tensor): 원본 보상 [batch_size]

        Returns:
            torch.Tensor: 정규화된 그룹 상대 보상 [batch_size]
        """
        # 그룹(배치) 평균 및 표준편차 계산
        mean_reward = rewards.mean()
        std_reward = rewards.std() + 1e-8  # 0으로 나누기 방지

        # Z-score 정규화
        normalized_rewards = (rewards - mean_reward) / std_reward

        return normalized_rewards

    def compute_rewards_from_errors(self, predictions, targets):
        """
        예측 오차로부터 보상 계산

        오차가 작을수록 높은 보상
        - 보상 = -|predicted - actual|

        Args:
            predictions (torch.Tensor): 예측 평점 [batch_size]
            targets (torch.Tensor): 실제 평점 [batch_size]

        Returns:
            torch.Tensor: 보상 [batch_size]
        """
        errors = torch.abs(predictions - targets)
        rewards = -errors  # 오차가 작을수록 높은 보상

        return rewards
