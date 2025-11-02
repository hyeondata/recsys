"""
PPO-MoE 학습 스크립트
PPO(Proximal Policy Optimization) 강화학습 기반
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse

from src.models import PPOMoE
from src.data.movie_preprocessor import MoviePreprocessor
from src.data.user_preprocessor import UserPreprocessor
from src.data.enhanced_dataset import EnhancedMovieRatingDataset
from src.utils import (
    compute_all_metrics,
    compute_expert_distribution,
    compute_expert_performance,
    EarlyStopping,
    CheckpointManager,
    set_seed,
    count_parameters,
    get_device,
    AverageMeter,
    print_metrics
)


def collect_episode(model, data_loader, device, deterministic=False):
    """
    에피소드 데이터 수집

    Args:
        model: PPOMoE 모델
        data_loader: 데이터 로더
        device: 디바이스
        deterministic: 결정론적 선택 여부

    Returns:
        dict: 수집된 에피소드 데이터
    """
    model.eval()

    states = []
    actions = []
    log_probs = []
    values = []
    rewards = []
    predictions = []
    targets = []
    user_ids = []
    movie_ids = []
    age_groups = []
    genders = []
    occupations = []

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Collecting episodes", leave=False):
            user_id = batch['user_id'].to(device)
            movie_id = batch['movie_id'].to(device)
            age_group = batch['age_group'].to(device)
            gender = batch['gender'].to(device)
            occupation = batch['occupation'].to(device)
            rating = batch['rating'].to(device)

            # Forward
            outputs = model(user_id, movie_id, age_group, gender, occupation, deterministic=deterministic)

            # 보상 계산: -|predicted - actual|
            reward = -torch.abs(outputs['rating'] - rating)

            # 수집
            actions.append(outputs['action'])
            log_probs.append(outputs['log_prob'])
            values.append(outputs['value'])
            rewards.append(reward)
            predictions.append(outputs['rating'])
            targets.append(rating)
            user_ids.append(user_id)
            movie_ids.append(movie_id)
            age_groups.append(age_group)
            genders.append(gender)
            occupations.append(occupation)

    return {
        'actions': torch.cat(actions),
        'log_probs': torch.cat(log_probs),
        'values': torch.cat(values),
        'rewards': torch.cat(rewards),
        'predictions': torch.cat(predictions),
        'targets': torch.cat(targets),
        'user_ids': torch.cat(user_ids),
        'movie_ids': torch.cat(movie_ids),
        'age_groups': torch.cat(age_groups),
        'genders': torch.cat(genders),
        'occupations': torch.cat(occupations)
    }


def train_epoch(model, train_loader, optimizer, device, args):
    """
    한 epoch 학습 (PPO 방식)

    Args:
        model: PPOMoE 모델
        train_loader: 학습 데이터 로더
        optimizer: Optimizer
        device: 디바이스
        args: 하이퍼파라미터

    Returns:
        dict: 평균 손실 및 지표
    """
    # 에피소드 수집
    episode_data = collect_episode(model, train_loader, device, deterministic=False)

    # Advantages와 Returns 계산
    rewards = episode_data['rewards']
    values = episode_data['values']

    # 다음 값은 마지막 값과 동일하게 설정 (에피소드 끝)
    next_values = torch.cat([values[1:], values[-1:]])
    dones = torch.zeros_like(rewards)  # 에피소드가 끝나지 않음

    advantages, returns = model.compute_advantages(
        rewards, values, next_values, dones,
        gamma=args.gamma, lam=args.lam
    )

    # Advantages 정규화
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    # PPO 업데이트
    model.train()
    dataset_size = len(episode_data['actions'])
    indices = torch.randperm(dataset_size)

    total_loss_meter = AverageMeter()
    policy_loss_meter = AverageMeter()
    value_loss_meter = AverageMeter()
    rating_loss_meter = AverageMeter()

    for ppo_epoch in tqdm(range(args.ppo_epochs), desc="PPO epochs", leave=False):
        num_batches = (dataset_size + args.batch_size - 1) // args.batch_size
        for start_idx in tqdm(range(0, dataset_size, args.batch_size), desc=f"PPO epoch {ppo_epoch+1}/{args.ppo_epochs}", leave=False, total=num_batches):
            end_idx = min(start_idx + args.batch_size, dataset_size)
            batch_indices = indices[start_idx:end_idx]

            # 배치 데이터 (에피소드에서 수집한 데이터 사용)
            batch_user_id = episode_data['user_ids'][batch_indices]
            batch_movie_id = episode_data['movie_ids'][batch_indices]
            batch_age_group = episode_data['age_groups'][batch_indices]
            batch_gender = episode_data['genders'][batch_indices]
            batch_occupation = episode_data['occupations'][batch_indices]

            batch_old_log_probs = episode_data['log_probs'][batch_indices]
            batch_advantages = advantages[batch_indices]
            batch_returns = returns[batch_indices]
            batch_targets = episode_data['targets'][batch_indices]

            # Forward
            outputs = model(batch_user_id, batch_movie_id, batch_age_group, batch_gender, batch_occupation)

            # Loss 계산
            losses = model.compute_loss(
                outputs, batch_targets,
                batch_old_log_probs, batch_advantages, batch_returns,
                entropy_coef=args.entropy_coef,
                value_coef=args.value_coef
            )

            # Backward
            optimizer.zero_grad()
            losses['total_loss'].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()

            # Metrics
            batch_size = len(batch_indices)
            total_loss_meter.update(losses['total_loss'].item(), batch_size)
            policy_loss_meter.update(losses['policy_loss'].item(), batch_size)
            value_loss_meter.update(losses['value_loss'].item(), batch_size)
            rating_loss_meter.update(losses['rating_loss'].item(), batch_size)

    return {
        'total_loss': total_loss_meter.avg,
        'policy_loss': policy_loss_meter.avg,
        'value_loss': value_loss_meter.avg,
        'rating_loss': rating_loss_meter.avg
    }


def validate_epoch(model, val_loader, device):
    """
    검증

    Args:
        model: PPOMoE 모델
        val_loader: 검증 데이터 로더
        device: 디바이스

    Returns:
        dict: 평가 지표
    """
    # 결정론적으로 에피소드 수집
    episode_data = collect_episode(model, val_loader, device, deterministic=True)

    # 메트릭 계산
    predictions = episode_data['predictions']
    targets = episode_data['targets']
    metrics = compute_all_metrics(predictions, targets, denormalize=True)

    # Expert 분포
    actions = episode_data['actions']
    expert_dist = compute_expert_distribution(actions)
    print(f"Expert distribution: {expert_dist}")

    # Expert 성능
    errors = torch.abs(predictions - targets)
    expert_perf = compute_expert_performance(actions, errors, num_experts=8)
    print(f"Expert performance (avg error): {expert_perf}")

    return metrics


def main(args):
    # 시드 설정
    set_seed(args.seed)

    # 디바이스 설정
    device = get_device()
    print(f"Using device: {device}")

    # 데이터 전처리
    print("Loading and preprocessing data...")
    movie_preprocessor = MoviePreprocessor(args.data_dir)
    movie_df, vocab, encoded_titles = movie_preprocessor.process_movie_data()

    user_preprocessor = UserPreprocessor(args.data_dir)
    user_df, user_metadata = user_preprocessor.process_user_data()

    # 데이터셋 생성
    train_dataset = EnhancedMovieRatingDataset(
        movie_data=movie_df,
        user_data=user_df,
        rating_path=args.train_rating_path,
        vocab=vocab,
        max_length=args.max_length
    )

    val_dataset = EnhancedMovieRatingDataset(
        movie_data=movie_df,
        user_data=user_df,
        rating_path=args.val_rating_path,
        vocab=vocab,
        max_length=args.max_length
    )

    # 데이터 로더
    train_loader = DataLoader(
        train_dataset,
        batch_size=len(train_dataset),  # 전체 데이터를 한번에 로드
        shuffle=False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=len(val_dataset),
        shuffle=False
    )

    print(f"Train size: {len(train_dataset)}, Val size: {len(val_dataset)}")

    # 모델 생성
    model = PPOMoE(
        num_users=user_metadata['num_users'],
        num_movies=len(movie_df),
        num_age_groups=user_metadata['num_age_groups'],
        num_occupations=user_metadata['num_occupations'],
        embedding_dim=args.embedding_dim,
        num_experts=args.num_experts,
        expert_hidden_dim=args.expert_hidden_dim,
        policy_hidden_dim=args.policy_hidden_dim,
        value_hidden_dim=args.value_hidden_dim,
        clip_epsilon=args.clip_epsilon,
        dropout=args.dropout
    ).to(device)

    print(f"Model parameters: {count_parameters(model):,}")

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Early stopping & Checkpoint
    early_stopping = EarlyStopping(patience=args.patience, min_delta=0.001, mode='min')
    checkpoint_manager = CheckpointManager(
        checkpoint_dir=args.checkpoint_dir,
        model_name="ppo_moe",
        max_keep=3
    )

    # 학습 루프
    best_val_rmse = float('inf')

    for epoch in range(1, args.epochs + 1):
        print(f"\n{'='*50}")
        print(f"Epoch {epoch}/{args.epochs}")
        print(f"{'='*50}")

        # Train
        train_metrics = train_epoch(model, train_loader, optimizer, device, args)
        print_metrics(train_metrics, prefix="Train")

        # Validate
        val_metrics = validate_epoch(model, val_loader, device)
        print_metrics(val_metrics, prefix="Val")

        # Checkpoint
        is_best = val_metrics['rmse'] < best_val_rmse
        if is_best:
            best_val_rmse = val_metrics['rmse']

        checkpoint_manager.save_checkpoint(
            model, optimizer, epoch, val_metrics, is_best=is_best
        )

        # Early stopping
        early_stopping(val_metrics['rmse'])
        if early_stopping.early_stop:
            print(f"\nEarly stopping triggered at epoch {epoch}")
            break

    print("\nTraining completed!")
    print(f"Best validation RMSE: {best_val_rmse:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train PPO-MoE model")

    # Data
    parser.add_argument("--data_dir", type=str, default="ml-100k", help="Data directory")
    parser.add_argument("--train_rating_path", type=str, default="ml-100k/u1.base", help="Training rating file")
    parser.add_argument("--val_rating_path", type=str, default="ml-100k/u1.test", help="Validation rating file")
    parser.add_argument("--max_length", type=int, default=10, help="Max title length")

    # Model
    parser.add_argument("--embedding_dim", type=int, default=64, help="Embedding dimension")
    parser.add_argument("--num_experts", type=int, default=8, help="Number of experts")
    parser.add_argument("--expert_hidden_dim", type=int, default=256, help="Expert hidden dimension")
    parser.add_argument("--policy_hidden_dim", type=int, default=128, help="Policy hidden dimension")
    parser.add_argument("--value_hidden_dim", type=int, default=128, help="Value hidden dimension")
    parser.add_argument("--clip_epsilon", type=float, default=0.2, help="PPO clip epsilon")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")

    # PPO Training
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size for PPO updates")
    parser.add_argument("--ppo_epochs", type=int, default=4, help="PPO update epochs per iteration")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--lam", type=float, default=0.95, help="GAE lambda")
    parser.add_argument("--entropy_coef", type=float, default=0.01, help="Entropy coefficient")
    parser.add_argument("--value_coef", type=float, default=0.5, help="Value loss coefficient")
    parser.add_argument("--max_grad_norm", type=float, default=0.5, help="Max gradient norm")

    # Training
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=0.0003, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-5, help="Weight decay")
    parser.add_argument("--patience", type=int, default=15, help="Early stopping patience")

    # Misc
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints", help="Checkpoint directory")

    args = parser.parse_args()
    main(args)
