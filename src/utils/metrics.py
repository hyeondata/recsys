"""
평가 지표 계산 유틸리티
"""

import torch
import numpy as np


def denormalize_rating(normalized_rating):
    """
    정규화된 평점을 원본 범위로 변환

    Args:
        normalized_rating: 0-1 범위의 평점

    Returns:
        1-5 범위의 평점
    """
    return normalized_rating * 4.0 + 1.0


def compute_mse(predictions, targets, denormalize=False):
    """
    MSE (Mean Squared Error) 계산

    Args:
        predictions (torch.Tensor or np.ndarray): 예측 평점
        targets (torch.Tensor or np.ndarray): 실제 평점
        denormalize (bool): True면 원본 스케일로 변환 후 계산

    Returns:
        float: MSE 값
    """
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.detach().cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.detach().cpu().numpy()

    if denormalize:
        predictions = denormalize_rating(predictions)
        targets = denormalize_rating(targets)

    mse = np.mean((predictions - targets) ** 2)
    return float(mse)


def compute_rmse(predictions, targets, denormalize=False):
    """
    RMSE (Root Mean Squared Error) 계산

    Args:
        predictions (torch.Tensor or np.ndarray): 예측 평점
        targets (torch.Tensor or np.ndarray): 실제 평점
        denormalize (bool): True면 원본 스케일로 변환 후 계산

    Returns:
        float: RMSE 값
    """
    mse = compute_mse(predictions, targets, denormalize)
    return np.sqrt(mse)


def compute_mae(predictions, targets, denormalize=False):
    """
    MAE (Mean Absolute Error) 계산

    Args:
        predictions (torch.Tensor or np.ndarray): 예측 평점
        targets (torch.Tensor or np.ndarray): 실제 평점
        denormalize (bool): True면 원본 스케일로 변환 후 계산

    Returns:
        float: MAE 값
    """
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.detach().cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.detach().cpu().numpy()

    if denormalize:
        predictions = denormalize_rating(predictions)
        targets = denormalize_rating(targets)

    mae = np.mean(np.abs(predictions - targets))
    return float(mae)


def compute_all_metrics(predictions, targets, denormalize=True):
    """
    모든 평가 지표를 한번에 계산

    Args:
        predictions (torch.Tensor or np.ndarray): 예측 평점
        targets (torch.Tensor or np.ndarray): 실제 평점
        denormalize (bool): True면 원본 스케일로 변환 후 계산

    Returns:
        dict: {'mse': float, 'rmse': float, 'mae': float}
    """
    mse = compute_mse(predictions, targets, denormalize)
    rmse = compute_rmse(predictions, targets, denormalize)
    mae = compute_mae(predictions, targets, denormalize)

    return {
        'mse': mse,
        'rmse': rmse,
        'mae': mae
    }


def compute_expert_distribution(actions):
    """
    Expert 선택 분포 계산 (강화학습 모델용)

    Args:
        actions (torch.Tensor or np.ndarray): Expert 인덱스들 [batch_size]

    Returns:
        dict: {expert_idx: count} 형태의 분포
    """
    if isinstance(actions, torch.Tensor):
        actions = actions.detach().cpu().numpy()

    unique, counts = np.unique(actions, return_counts=True)
    distribution = {int(expert_idx): int(count) for expert_idx, count in zip(unique, counts)}

    return distribution


def compute_expert_performance(actions, errors, num_experts=8):
    """
    각 Expert의 평균 성능 계산

    Args:
        actions (torch.Tensor or np.ndarray): Expert 인덱스들
        errors (torch.Tensor or np.ndarray): 예측 오차들 (절댓값)
        num_experts (int): 전체 Expert 개수

    Returns:
        dict: {expert_idx: avg_error} 형태의 성능
    """
    if isinstance(actions, torch.Tensor):
        actions = actions.detach().cpu().numpy()
    if isinstance(errors, torch.Tensor):
        errors = errors.detach().cpu().numpy()

    performance = {}
    for expert_idx in range(num_experts):
        mask = actions == expert_idx
        if mask.sum() > 0:
            avg_error = errors[mask].mean()
            performance[expert_idx] = float(avg_error)
        else:
            performance[expert_idx] = None

    return performance
