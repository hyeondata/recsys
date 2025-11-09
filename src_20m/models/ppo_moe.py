"""
PPO-MoE 모델
PPO(Proximal Policy Optimization) 강화학습 기반 Gating Network
Clipped Surrogate Objective로 안정적 학습
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from .base_moe import BaseMoEModel


class PPOMoE(BaseMoEModel):
    """
    PPO 기반 Gating Network를 사용하는 MoE 모델

    강화학습을 통해 Expert를 선택:
    - State: User + Movie 임베딩
    - Action: Expert 선택 (0~7)
    - Reward: 예측 평점과 실제 평점의 차이 기반

    Args:
        num_users (int): 사용자 수
        num_movies (int): 영화 수
        num_age_groups (int): 나이 그룹 수 (기본값: 8)
        num_occupations (int): 직업 수
        embedding_dim (int): 임베딩 차원 (기본값: 64)
        num_experts (int): Expert 개수 (기본값: 8)
        expert_hidden_dim (int): Expert 은닉층 차원 (기본값: 256)
        policy_hidden_dim (int): Policy Network 은닉층 차원 (기본값: 128)
        value_hidden_dim (int): Value Network 은닉층 차원 (기본값: 128)
        clip_epsilon (float): PPO clipping 파라미터 (기본값: 0.2)
        dropout (float): 드롭아웃 비율 (기본값: 0.1)
    """
    def __init__(
        self,
        num_users,
        num_movies,
        embedding_dim=64,
        num_experts=8,
        expert_hidden_dim=256,
        policy_hidden_dim=128,
        value_hidden_dim=128,
        clip_epsilon=0.2,
        dropout=0.1
    ):
        super(PPOMoE, self).__init__(
            num_users=num_users,
            num_movies=num_movies,
            embedding_dim=embedding_dim,
            num_experts=num_experts,
            expert_hidden_dim=expert_hidden_dim,
            dropout=dropout
        )

        self.clip_epsilon = clip_epsilon
        input_dim = embedding_dim * 2  # user + movie 임베딩

        # Policy Network (Actor): Expert 선택 확률 분포 생성
        self.policy_network = nn.Sequential(
            nn.Linear(input_dim, policy_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(policy_hidden_dim, policy_hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(policy_hidden_dim // 2, num_experts)
        )

        # Value Network (Critic): State의 가치 추정
        self.value_network = nn.Sequential(
            nn.Linear(input_dim, value_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(value_hidden_dim, value_hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(value_hidden_dim // 2, 1)
        )

    def forward(self, user_id, movie_id, deterministic=False):
        """
        순전파

        Args:
            user_id (torch.Tensor): [batch_size]
            movie_id (torch.Tensor): [batch_size]
            deterministic (bool): True면 가장 높은 확률의 Expert 선택,
                                  False면 확률에 따라 샘플링

        Returns:
            dict: {
                'rating': 예측 평점 [batch_size],
                'action': 선택된 Expert 인덱스 [batch_size],
                'log_prob': 선택된 Action의 로그 확률 [batch_size],
                'entropy': 정책 엔트로피 [batch_size],
                'value': State 가치 추정 [batch_size],
                'action_probs': 각 Expert 선택 확률 [batch_size, num_experts]
            }
        """
        # State 생성 (User + Movie 임베딩)
        state = self.get_state(user_id, movie_id)

        # Policy Network로 Expert 선택 확률 계산
        policy_logits = self.policy_network(state)  # [batch_size, num_experts]
        action_probs = F.softmax(policy_logits, dim=-1)  # [batch_size, num_experts]

        # Action 선택
        dist = Categorical(action_probs)

        if deterministic:
            action = action_probs.argmax(dim=-1)  # [batch_size]
        else:
            action = dist.sample()  # [batch_size]

        log_prob = dist.log_prob(action)  # [batch_size]
        entropy = dist.entropy()  # [batch_size]

        # Value Network로 State 가치 추정
        value = self.value_network(state).squeeze(-1)  # [batch_size]

        # 선택된 Expert의 출력으로 평점 예측
        expert_output = self.experts(state, expert_indices=action)  # [batch_size, 1]
        rating = torch.sigmoid(expert_output.squeeze(-1))  # [batch_size]

        return {
            'rating': rating,
            'action': action,
            'log_prob': log_prob,
            'entropy': entropy,
            'value': value,
            'action_probs': action_probs
        }

    def compute_loss(self, predictions, targets, old_log_probs, advantages, returns, entropy_coef=0.01, value_coef=0.5):
        """
        PPO 손실 함수 계산

        Args:
            predictions (dict): forward() 출력
            targets (torch.Tensor): 정규화된 실제 평점 [batch_size]
            old_log_probs (torch.Tensor): 이전 정책의 로그 확률 [batch_size]
            advantages (torch.Tensor): Advantage 값 [batch_size]
            returns (torch.Tensor): Return 값 [batch_size]
            entropy_coef (float): 엔트로피 계수
            value_coef (float): Value 손실 계수

        Returns:
            dict: {
                'total_loss': 전체 손실,
                'policy_loss': Policy 손실,
                'value_loss': Value 손실,
                'entropy_loss': 엔트로피 손실,
                'rating_loss': 평점 예측 손실 (MSE)
            }
        """
        # Policy Loss (PPO Clipped Objective)
        ratio = torch.exp(predictions['log_prob'] - old_log_probs)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()

        # Value Loss (MSE)
        value_loss = F.mse_loss(predictions['value'], returns)

        # Entropy Loss (탐색 촉진)
        entropy_loss = -predictions['entropy'].mean()

        # Rating Prediction Loss (MSE)
        rating_loss = F.mse_loss(predictions['rating'], targets)

        # 전체 손실
        total_loss = policy_loss + value_coef * value_loss + entropy_coef * entropy_loss + rating_loss

        return {
            'total_loss': total_loss,
            'policy_loss': policy_loss,
            'value_loss': value_loss,
            'entropy_loss': entropy_loss,
            'rating_loss': rating_loss
        }

    def compute_advantages(self, rewards, values, next_values, dones, gamma=0.99, lam=0.95):
        """
        GAE(Generalized Advantage Estimation) 계산

        Args:
            rewards (torch.Tensor): 보상 [batch_size]
            values (torch.Tensor): 현재 State 가치 [batch_size]
            next_values (torch.Tensor): 다음 State 가치 [batch_size]
            dones (torch.Tensor): 종료 플래그 [batch_size]
            gamma (float): 할인율
            lam (float): GAE lambda

        Returns:
            tuple: (advantages, returns)
        """
        deltas = rewards + gamma * next_values * (1 - dones) - values
        advantages = torch.zeros_like(rewards)
        advantage = 0

        for t in reversed(range(len(rewards))):
            advantage = deltas[t] + gamma * lam * (1 - dones[t]) * advantage
            advantages[t] = advantage

        returns = advantages + values

        return advantages, returns
