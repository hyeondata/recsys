"""
MoE 기반 추천 모델 패키지

이 패키지는 세 가지 MoE 모델을 제공합니다:
1. DenseMoE: Fully Connected Layer 기반 Gating Network (베이스라인)
2. PPOMoE: PPO 강화학습 기반 Expert 선택
3. GRPOMoE: GRPO 강화학습 기반 Expert 선택 (그룹 상대 보상)
"""

from .expert_network import ExpertNetwork, ExpertEnsemble
from .base_moe import BaseMoEModel
from .dense_moe import DenseMoE
from .ppo_moe import PPOMoE
from .grpo_moe import GRPOMoE

__all__ = [
    'ExpertNetwork',
    'ExpertEnsemble',
    'BaseMoEModel',
    'DenseMoE',
    'PPOMoE',
    'GRPOMoE'
]

__version__ = '1.0.0'
