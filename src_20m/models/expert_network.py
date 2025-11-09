"""
Expert Network 모듈
각 Expert는 독립적인 Feed Forward Network(FFN)으로 구성됩니다.
"""

import torch
import torch.nn as nn


class ExpertNetwork(nn.Module):
    """
    단일 Expert Network

    Args:
        input_dim (int): 입력 차원
        hidden_dim (int): 은닉층 차원
        output_dim (int): 출력 차원
        dropout (float): 드롭아웃 비율
    """
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.1):
        super(ExpertNetwork, self).__init__()

        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim)
        )

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): 입력 텐서 [batch_size, input_dim]

        Returns:
            torch.Tensor: 출력 텐서 [batch_size, output_dim]
        """
        return self.network(x)


class ExpertEnsemble(nn.Module):
    """
    여러 Expert Network를 관리하는 앙상블 클래스

    Args:
        num_experts (int): Expert 개수 (기본값: 8)
        input_dim (int): 입력 차원
        hidden_dim (int): 은닉층 차원
        output_dim (int): 출력 차원
        dropout (float): 드롭아웃 비율
    """
    def __init__(self, num_experts=8, input_dim=128, hidden_dim=256, output_dim=1, dropout=0.1):
        super(ExpertEnsemble, self).__init__()

        self.num_experts = num_experts
        self.experts = nn.ModuleList([
            ExpertNetwork(input_dim, hidden_dim, output_dim, dropout)
            for _ in range(num_experts)
        ])

    def forward(self, x, expert_indices=None):
        """
        Args:
            x (torch.Tensor): 입력 텐서 [batch_size, input_dim]
            expert_indices (torch.Tensor, optional): 선택할 Expert 인덱스 [batch_size]
                                                      None인 경우 모든 Expert 출력 반환

        Returns:
            torch.Tensor:
                - expert_indices가 None이면: [batch_size, num_experts, output_dim]
                - expert_indices가 주어지면: [batch_size, output_dim]
        """
        if expert_indices is None:
            # 모든 Expert의 출력 반환
            outputs = []
            for expert in self.experts:
                outputs.append(expert(x))
            return torch.stack(outputs, dim=1)  # [batch_size, num_experts, output_dim]
        else:
            # 선택된 Expert의 출력만 반환 (벡터화 방식)
            batch_size = x.size(0)
            output_dim = self.experts[0].network[-1].out_features
            outputs = torch.zeros(batch_size, output_dim, device=x.device)

            # Expert별로 그룹화하여 처리 (GPU 병렬화)
            for expert_idx in range(self.num_experts):
                # 이 expert를 선택한 샘플들의 마스크
                mask = (expert_indices == expert_idx)

                if mask.any():
                    # 선택된 샘플들을 한번에 처리
                    outputs[mask] = self.experts[expert_idx](x[mask])

            return outputs
