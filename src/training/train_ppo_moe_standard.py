"""
PPO-MoE 학습 스크립트 (표준 RL 방식)
표준적인 PPO 구현: 전체 epoch 데이터 수집 → 여러 번 업데이트
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
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


def collect_full_epoch_episodes(model, data_loader, device):
    """
    전체 epoch에 대한 에피소드 수집 (표준 RL 방식)

    Args:
        model: PPOMoE 모델
        data_loader: 데이터 로더
        device: 디바이스

    Returns:
        dict: 수집된 전체 에피소드 데이터
    """
    model.eval()

    all_user_ids = []
    all_movie_ids = []
    all_age_groups = []
    all_genders = []
    all_occupations = []
    all_ratings = []
    all_actions = []
    all_log_probs = []
    all_values = []
    all_predictions = []

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Collecting episodes", leave=False):
            user_id = batch['user_id'].to(device)
            movie_id = batch['movie_id'].to(device)
            age_group = batch['age_group'].to(device)
            gender = batch['gender'].to(device)
            occupation = batch['occupation'].to(device)
            rating = batch['rating'].to(device)

            # Forward (stochastic)
            outputs = model(user_id, movie_id, age_group, gender, occupation, deterministic=False)

            # 수집
            all_user_ids.append(user_id)
            all_movie_ids.append(movie_id)
            all_age_groups.append(age_group)
            all_genders.append(gender)
            all_occupations.append(occupation)
            all_ratings.append(rating)
            all_actions.append(outputs['action'])
            all_log_probs.append(outputs['log_prob'])
            all_values.append(outputs['value'])
            all_predictions.append(outputs['rating'])

    # Concatenate all
    episodes = {
        'user_ids': torch.cat(all_user_ids),
        'movie_ids': torch.cat(all_movie_ids),
        'age_groups': torch.cat(all_age_groups),
        'genders': torch.cat(all_genders),
        'occupations': torch.cat(all_occupations),
        'ratings': torch.cat(all_ratings),
        'actions': torch.cat(all_actions),
        'old_log_probs': torch.cat(all_log_probs),
        'old_values': torch.cat(all_values),
        'predictions': torch.cat(all_predictions)
    }

    return episodes


def compute_advantages_and_returns(episodes, gamma=0.99, lam=0.95):
    """
    GAE (Generalized Advantage Estimation) 계산

    Args:
        episodes: 에피소드 데이터
        gamma: Discount factor
        lam: GAE lambda

    Returns:
        advantages, returns
    """
    predictions = episodes['predictions']
    ratings = episodes['ratings']
    values = episodes['old_values']

    # 보상 계산: -|predicted - actual|
    rewards = -torch.abs(predictions - ratings)

    # 단순화된 방식 (single-step task이므로)
    # advantages = reward - baseline
    advantages = rewards - values
    returns = rewards  # Single step이므로 returns = rewards

    # Advantage 정규화
    if advantages.std() > 1e-8:
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    return advantages, returns


def create_mini_batches(episodes, advantages, returns, batch_size, shuffle=True):
    """
    전체 에피소드를 mini-batch로 나누기

    Args:
        episodes: 전체 에피소드 데이터
        advantages: Advantage values
        returns: Return values
        batch_size: Mini-batch 크기
        shuffle: 셔플 여부

    Yields:
        mini_batch dict
    """
    dataset_size = len(episodes['user_ids'])
    indices = torch.randperm(dataset_size) if shuffle else torch.arange(dataset_size)

    for start_idx in range(0, dataset_size, batch_size):
        end_idx = min(start_idx + batch_size, dataset_size)
        batch_indices = indices[start_idx:end_idx]

        mini_batch = {
            'user_ids': episodes['user_ids'][batch_indices],
            'movie_ids': episodes['movie_ids'][batch_indices],
            'age_groups': episodes['age_groups'][batch_indices],
            'genders': episodes['genders'][batch_indices],
            'occupations': episodes['occupations'][batch_indices],
            'ratings': episodes['ratings'][batch_indices],
            'old_log_probs': episodes['old_log_probs'][batch_indices],
            'advantages': advantages[batch_indices],
            'returns': returns[batch_indices]
        }

        yield mini_batch


def train_epoch(model, train_loader, optimizer, device, args):
    """
    한 epoch 학습 (표준 PPO 방식)

    Args:
        model: PPOMoE 모델
        train_loader: 학습 데이터 로더
        optimizer: Optimizer
        device: 디바이스
        args: 하이퍼파라미터

    Returns:
        dict: 평균 손실 및 지표
    """
    # Phase 1: 전체 epoch에 대한 에피소드 수집
    print("Phase 1: Collecting episodes for full epoch...")
    episodes = collect_full_epoch_episodes(model, train_loader, device)

    total_samples = len(episodes['user_ids'])
    print(f"Collected {total_samples} samples")

    # Phase 2: Advantage 및 Return 계산
    print("Phase 2: Computing advantages and returns...")
    advantages, returns = compute_advantages_and_returns(
        episodes, gamma=args.gamma, lam=args.lam
    )

    # Phase 3: 수집된 데이터로 여러 epoch 학습
    print(f"Phase 3: Training for {args.ppo_epochs} epochs...")
    model.train()

    total_loss_meter = AverageMeter()
    policy_loss_meter = AverageMeter()
    value_loss_meter = AverageMeter()
    rating_loss_meter = AverageMeter()

    for ppo_epoch in range(args.ppo_epochs):
        epoch_loss_meter = AverageMeter()

        # Mini-batch로 나눠서 학습
        mini_batches = create_mini_batches(
            episodes, advantages, returns,
            batch_size=args.mini_batch_size,
            shuffle=True
        )

        for mini_batch in tqdm(mini_batches, desc=f"PPO Epoch {ppo_epoch+1}/{args.ppo_epochs}", leave=False):
            user_ids = mini_batch['user_ids']
            movie_ids = mini_batch['movie_ids']
            age_groups = mini_batch['age_groups']
            genders = mini_batch['genders']
            occupations = mini_batch['occupations']
            ratings = mini_batch['ratings']
            old_log_probs = mini_batch['old_log_probs']
            batch_advantages = mini_batch['advantages']
            batch_returns = mini_batch['returns']

            batch_size = len(user_ids)

            # Forward
            outputs = model(user_ids, movie_ids, age_groups, genders, occupations)

            # Loss 계산
            losses = model.compute_loss(
                outputs, ratings,
                old_log_probs, batch_advantages, batch_returns,
                entropy_coef=args.entropy_coef,
                value_coef=args.value_coef
            )

            # Backward
            optimizer.zero_grad()
            losses['total_loss'].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()

            # Metrics
            epoch_loss_meter.update(losses['total_loss'].item(), batch_size)

            if ppo_epoch == args.ppo_epochs - 1:  # 마지막 epoch만 기록
                total_loss_meter.update(losses['total_loss'].item(), batch_size)
                policy_loss_meter.update(losses['policy_loss'].item(), batch_size)
                value_loss_meter.update(losses['value_loss'].item(), batch_size)
                rating_loss_meter.update(losses['rating_loss'].item(), batch_size)

        print(f"  PPO Epoch {ppo_epoch+1} - Avg Loss: {epoch_loss_meter.avg:.4f}")

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
    model.eval()

    all_predictions = []
    all_targets = []
    all_actions = []

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validation"):
            user_id = batch['user_id'].to(device)
            movie_id = batch['movie_id'].to(device)
            age_group = batch['age_group'].to(device)
            gender = batch['gender'].to(device)
            occupation = batch['occupation'].to(device)
            rating = batch['rating'].to(device)

            # Forward (deterministic)
            outputs = model(user_id, movie_id, age_group, gender, occupation, deterministic=True)

            # Collect
            all_predictions.append(outputs['rating'].cpu())
            all_targets.append(rating.cpu())
            all_actions.append(outputs['action'].cpu())

    # Concatenate
    predictions = torch.cat(all_predictions)
    targets = torch.cat(all_targets)
    actions = torch.cat(all_actions)

    # 메트릭 계산
    metrics = compute_all_metrics(predictions, targets, denormalize=True)

    # Expert 분포
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
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers
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
        model_name="ppo_moe_standard",
        max_keep=3
    )

    # 학습 루프
    best_val_rmse = float('inf')

    for epoch in range(1, args.epochs + 1):
        print(f"\n{'='*70}")
        print(f"Epoch {epoch}/{args.epochs} - Standard PPO Training")
        print(f"{'='*70}")

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
    parser = argparse.ArgumentParser(description="Train PPO-MoE model (Standard RL)")

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

    # PPO Training (Standard RL)
    parser.add_argument("--batch_size", type=int, default=512, help="Batch size for episode collection")
    parser.add_argument("--mini_batch_size", type=int, default=256, help="Mini-batch size for PPO updates")
    parser.add_argument("--ppo_epochs", type=int, default=4, help="Number of PPO epochs per iteration")
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
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints/ppo_moe_standard", help="Checkpoint directory")

    args = parser.parse_args()
    main(args)
