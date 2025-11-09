"""
PPO-MoE 학습 스크립트 (ml-20m 버전)
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

from src_20m.models import PPOMoE
from src_20m.data.movie_preprocessor import MoviePreprocessor
from src_20m.data.dataset import MovieRatingDataset, load_and_split_ratings
from src_20m.utils import (
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
    model.train()

    total_loss_meter = AverageMeter()
    policy_loss_meter = AverageMeter()
    value_loss_meter = AverageMeter()
    rating_loss_meter = AverageMeter()

    # 배치별로 학습
    for batch in tqdm(train_loader, desc="Training"):
        user_id = batch['user_id'].to(device)
        movie_id = batch['movie_id'].to(device)
        rating = batch['rating'].to(device)

        batch_size = len(user_id)

        # 1. 에피소드 수집 (현재 배치에 대해)
        with torch.no_grad():
            model.eval()
            outputs_old = model(user_id, movie_id, deterministic=False)
            old_log_probs = outputs_old['log_prob']
            old_values = outputs_old['value']
            predictions = outputs_old['rating']

        # 2. 보상 및 Advantage 계산
        rewards = -torch.abs(predictions - rating)

        # 간단한 방식: advantage = reward - baseline (value)
        advantages = rewards - old_values
        returns = rewards  # 단일 스텝이므로 returns = rewards

        # Advantages 정규화
        if advantages.std() > 1e-8:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # 3. PPO 업데이트 (배치를 ppo_epochs번 반복 학습)
        model.train()
        for _ in range(args.ppo_epochs):
            # Forward
            outputs = model(user_id, movie_id)

            # Loss 계산
            losses = model.compute_loss(
                outputs, rating,
                old_log_probs, advantages, returns,
                entropy_coef=args.entropy_coef,
                value_coef=args.value_coef
            )

            # Backward
            optimizer.zero_grad()
            losses['total_loss'].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()

        # Metrics (마지막 업데이트 기준)
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


def validate_epoch(model, val_loader, device, num_experts=16):
    """
    검증

    Args:
        model: PPOMoE 모델
        val_loader: 검증 데이터 로더
        device: 디바이스
        num_experts: Expert 개수

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
            rating = batch['rating'].to(device)

            # Forward (deterministic)
            outputs = model(user_id, movie_id, deterministic=True)

            # Collect
            all_predictions.append(outputs['rating'].cpu())
            all_targets.append(rating.cpu())
            all_actions.append(outputs['action'].cpu())

    # Concatenate
    predictions = torch.cat(all_predictions)
    targets = torch.cat(all_targets)
    actions = torch.cat(all_actions)

    # 메트릭 계산
    metrics = compute_all_metrics(predictions, targets, denormalize=False)

    # Expert 분포
    expert_dist = compute_expert_distribution(actions, num_experts=num_experts)
    print(f"Expert distribution: {expert_dist}")

    # Expert 성능
    errors = torch.abs(predictions - targets)
    expert_perf = compute_expert_performance(actions, errors, num_experts=num_experts)
    print(f"Expert performance (avg error): {expert_perf}")

    return metrics


def main(args):
    # 시드 설정
    set_seed(args.seed)

    # 디바이스 설정
    device = get_device()
    print(f"Using device: {device}")

    # 데이터 전처리
    print("\nLoading and preprocessing data (ml-20m)...")
    movie_preprocessor = MoviePreprocessor(args.data_dir)
    movie_df, vocab, encoded_titles = movie_preprocessor.process_movie_data()

    # Train/Test split
    print("\nSplitting data...")
    train_ratings, val_ratings = load_and_split_ratings(
        args.data_dir,
        train_ratio=0.8,
        random_seed=args.seed
    )

    # 데이터셋 생성
    print("\nCreating datasets...")
    train_dataset = MovieRatingDataset(
        movie_data=movie_df,
        rating_data=train_ratings,
        vocab=vocab,
        max_length=args.max_length
    )

    val_dataset = MovieRatingDataset(
        movie_data=movie_df,
        rating_data=val_ratings,
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

    # 모델 초기화
    print("\nInitializing model...")
    num_users = train_ratings['userId'].max()
    num_movies = movie_df['movieId'].max()

    model = PPOMoE(
        num_users=num_users,
        num_movies=num_movies,
        embedding_dim=args.embedding_dim,
        num_experts=args.num_experts,
        expert_hidden_dim=args.expert_hidden_dim,
        policy_hidden_dim=args.policy_hidden_dim,
        value_hidden_dim=args.value_hidden_dim,
        clip_epsilon=args.clip_epsilon,
        dropout=args.dropout
    ).to(device)

    print(f"Model parameters: {count_parameters(model):,}")
    print(f"Users: {num_users:,}, Movies: {num_movies:,}")

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Early stopping & Checkpoint
    early_stopping = EarlyStopping(patience=args.patience)
    checkpoint_manager = CheckpointManager(
        checkpoint_dir=args.checkpoint_dir,
        model_name="ppo_moe",
        max_keep=3
    )

    # 학습 루프
    best_val_rmse = float('inf')

    print(f"\nStarting training for {args.epochs} epochs...")
    for epoch in range(1, args.epochs + 1):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch}/{args.epochs}")
        print(f"{'='*60}")

        # Train
        train_metrics = train_epoch(model, train_loader, optimizer, device, args)
        print_metrics(train_metrics, prefix="Train")

        # Validate
        val_metrics = validate_epoch(model, val_loader, device, num_experts=args.num_experts)
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
    parser = argparse.ArgumentParser(description="Train PPO-MoE model (ml-20m)")

    # Data
    parser.add_argument("--data_dir", type=str, default="ml-20m", help="Data directory")
    parser.add_argument("--max_length", type=int, default=10, help="Max title length")

    # Model
    parser.add_argument("--embedding_dim", type=int, default=128, help="Embedding dimension")
    parser.add_argument("--num_experts", type=int, default=16, help="Number of experts")
    parser.add_argument("--expert_hidden_dim", type=int, default=512, help="Expert hidden dimension")
    parser.add_argument("--policy_hidden_dim", type=int, default=128, help="Policy hidden dimension")
    parser.add_argument("--value_hidden_dim", type=int, default=128, help="Value hidden dimension")
    parser.add_argument("--clip_epsilon", type=float, default=0.2, help="PPO clip epsilon")
    parser.add_argument("--dropout", type=float, default=0.2, help="Dropout rate")

    # PPO Training
    parser.add_argument("--batch_size", type=int, default=1024, help="Batch size for PPO updates")
    parser.add_argument("--ppo_epochs", type=int, default=4, help="PPO update epochs per iteration")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--lam", type=float, default=0.95, help="GAE lambda")
    parser.add_argument("--entropy_coef", type=float, default=0.01, help="Entropy coefficient")
    parser.add_argument("--value_coef", type=float, default=0.5, help="Value loss coefficient")
    parser.add_argument("--max_grad_norm", type=float, default=0.5, help="Max gradient norm")

    # Training
    parser.add_argument("--epochs", type=int, default=20, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=0.0003, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-5, help="Weight decay")
    parser.add_argument("--patience", type=int, default=10, help="Early stopping patience")

    # Misc
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints_20m/ppo_moe", help="Checkpoint directory")

    args = parser.parse_args()
    main(args)
