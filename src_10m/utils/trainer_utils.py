"""
학습 관련 유틸리티 함수들
"""

import os
import torch
import numpy as np
from pathlib import Path


class EarlyStopping:
    """
    Early Stopping 헬퍼 클래스

    Args:
        patience (int): 개선이 없을 때 기다리는 epoch 수
        min_delta (float): 개선으로 간주할 최소 변화량
        mode (str): 'min' (낮을수록 좋음) 또는 'max' (높을수록 좋음)
    """
    def __init__(self, patience=10, min_delta=0.0, mode='min'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, score):
        """
        점수를 확인하고 early stopping 여부 판단

        Args:
            score (float): 현재 점수 (loss 또는 metric)

        Returns:
            bool: True면 개선됨, False면 개선 안됨
        """
        if self.best_score is None:
            self.best_score = score
            return True

        if self.mode == 'min':
            improved = score < (self.best_score - self.min_delta)
        else:
            improved = score > (self.best_score + self.min_delta)

        if improved:
            self.best_score = score
            self.counter = 0
            return True
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
            return False


class CheckpointManager:
    """
    모델 체크포인트 관리 클래스

    Args:
        checkpoint_dir (str): 체크포인트 저장 디렉토리
        model_name (str): 모델 이름
        max_keep (int): 최대 유지할 체크포인트 개수 (0이면 무제한)
    """
    def __init__(self, checkpoint_dir, model_name, max_keep=3):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.model_name = model_name
        self.max_keep = max_keep
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.saved_checkpoints = []

    def save_checkpoint(self, model, optimizer, epoch, metrics, is_best=False, scheduler=None, early_stopping=None):
        """
        체크포인트 저장

        Args:
            model: PyTorch 모델
            optimizer: Optimizer
            epoch (int): 현재 epoch
            metrics (dict): 평가 지표들
            is_best (bool): 최고 성능 모델인지 여부
            scheduler: Learning rate scheduler (선택)
            early_stopping: EarlyStopping instance (선택)
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'metrics': metrics
        }

        if scheduler is not None:
            checkpoint['scheduler_state_dict'] = scheduler.state_dict()

        if early_stopping is not None:
            checkpoint['early_stopping'] = {
                'counter': early_stopping.counter,
                'best_score': early_stopping.best_score,
                'early_stop': early_stopping.early_stop
            }

        # 일반 체크포인트 저장
        filename = f"{self.model_name}_epoch_{epoch}.pt"
        filepath = self.checkpoint_dir / filename
        torch.save(checkpoint, filepath)
        self.saved_checkpoints.append(filepath)

        # 최고 성능 모델 따로 저장
        if is_best:
            best_filename = f"{self.model_name}_best.pt"
            best_filepath = self.checkpoint_dir / best_filename
            torch.save(checkpoint, best_filepath)
            print(f"Saved best model to {best_filepath}")

        # 최신 체크포인트 링크 저장 (resume용)
        latest_filename = f"{self.model_name}_latest.pt"
        latest_filepath = self.checkpoint_dir / latest_filename
        torch.save(checkpoint, latest_filepath)

        # 오래된 체크포인트 삭제 (max_keep 초과 시)
        if self.max_keep > 0 and len(self.saved_checkpoints) > self.max_keep:
            old_checkpoint = self.saved_checkpoints.pop(0)
            if old_checkpoint.exists():
                old_checkpoint.unlink()

    def get_latest_checkpoint(self):
        """
        최신 체크포인트 파일명 반환

        Returns:
            str: 최신 체크포인트 파일명 또는 None
        """
        latest_filename = f"{self.model_name}_latest.pt"
        latest_filepath = self.checkpoint_dir / latest_filename

        if latest_filepath.exists():
            return latest_filename
        return None

    def load_checkpoint(self, model, optimizer=None, filename=None, scheduler=None, early_stopping=None):
        """
        체크포인트 로드

        Args:
            model: PyTorch 모델
            optimizer: Optimizer (선택)
            filename (str): 로드할 파일명 (None이면 best 모델)
            scheduler: Learning rate scheduler (선택)
            early_stopping: EarlyStopping instance (선택)

        Returns:
            dict: 체크포인트 정보 (epoch, metrics)
        """
        if filename is None:
            filename = f"{self.model_name}_best.pt"

        filepath = self.checkpoint_dir / filename

        if not filepath.exists():
            raise FileNotFoundError(f"Checkpoint not found: {filepath}")

        checkpoint = torch.load(filepath, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])

        if optimizer is not None:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if scheduler is not None and 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        if early_stopping is not None and 'early_stopping' in checkpoint:
            es_state = checkpoint['early_stopping']
            early_stopping.counter = es_state['counter']
            early_stopping.best_score = es_state['best_score']
            early_stopping.early_stop = es_state['early_stop']

        print(f"Loaded checkpoint from {filepath}")
        print(f"Resuming from epoch {checkpoint['epoch']}")
        return {
            'epoch': checkpoint['epoch'],
            'metrics': checkpoint['metrics']
        }


def set_seed(seed=42):
    """
    재현성을 위한 시드 설정

    Args:
        seed (int): 시드 값
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    # Python random
    import random
    random.seed(seed)

    # CuDNN 설정
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def count_parameters(model):
    """
    모델의 학습 가능한 파라미터 수 계산

    Args:
        model: PyTorch 모델

    Returns:
        int: 파라미터 수
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_device():
    """
    사용 가능한 디바이스 반환

    Returns:
        torch.device: cuda 또는 cpu
    """
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class AverageMeter:
    """
    평균 계산을 위한 헬퍼 클래스
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def print_metrics(metrics, prefix=""):
    """
    지표를 보기 좋게 출력

    Args:
        metrics (dict): 지표 딕셔너리
        prefix (str): 접두사 (예: "Train", "Val")
    """
    metric_str = " | ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
    if prefix:
        print(f"{prefix} - {metric_str}")
    else:
        print(metric_str)
