"""
GRPO-MoE 학습 스크립트
GRPO(Group Relative Policy Optimization) 강화학습 기반
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse

from src.models import GRPOMoE
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
        model: GRPOMoE 모델
        data_loader: 데이터 로더
        device: 디바이스
        deterministic: 결정론적 선택 여부

    Returns:
        dict: 수집된 에피소드 데이터
    """
    model.eval()

    actions = []
    log_probs = []
    baselines = []
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

            # 수집
            actions.append(outputs['action'])
            log_probs.append(outputs['log_prob'])
            baselines.append(outputs['baseline'])
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
        'baselines': torch.cat(baselines),
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
    한 epoch 학습 (GRPO 방식)

    Args:
        model: GRPOMoE 모델
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
    baseline_loss_meter = AverageMeter()
    rating_loss_meter = AverageMeter()

    # 배치별로 학습 (Dense MoE와 동일한 방식)
    for batch in tqdm(train_loader, desc="Training"):
        user_id = batch['user_id'].to(device)
        movie_id = batch['movie_id'].to(device)
        age_group = batch['age_group'].to(device)
        gender = batch['gender'].to(device)
        occupation = batch['occupation'].to(device)
        rating = batch['rating'].to(device)

        batch_size = len(user_id)

        # 1. 에피소드 수집 (현재 배치에 대해)
        with torch.no_grad():
            model.eval()
            outputs_old = model(user_id, movie_id, age_group, gender, occupation, deterministic=False)
            old_log_probs = outputs_old['log_prob']
            predictions = outputs_old['rating']

        # 2. 보상 계산
        raw_rewards = model.compute_rewards_from_errors(predictions, rating)
        group_rewards = model.compute_group_relative_rewards(raw_rewards)

        # 3. GRPO 업데이트 (배치를 grpo_epochs번 반복 학습)
        model.train()
        for _ in range(args.grpo_epochs):
            # Forward
            outputs = model(user_id, movie_id, age_group, gender, occupation)

            # Loss 계산
            losses = model.compute_loss(
                outputs, rating,
                old_log_probs, group_rewards,
                entropy_coef=args.entropy_coef,
                baseline_coef=args.baseline_coef
            )

            # Backward
            optimizer.zero_grad()
            losses['total_loss'].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()

        # Metrics (마지막 업데이트 기준)
        total_loss_meter.update(losses['total_loss'].item(), batch_size)
        policy_loss_meter.update(losses['policy_loss'].item(), batch_size)
        baseline_loss_meter.update(losses['baseline_loss'].item(), batch_size)
        rating_loss_meter.update(losses['rating_loss'].item(), batch_size)

    return {
        'total_loss': total_loss_meter.avg,
        'policy_loss': policy_loss_meter.avg,
        'baseline_loss': baseline_loss_meter.avg,
        'rating_loss': rating_loss_meter.avg
    }


def validate_epoch(model, val_loader, device):
    """
    검증

    Args:
        model: GRPOMoE 모델
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

    # 데이터 로더 (Dense MoE와 동일한 방식)
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
    model = GRPOMoE(
        num_users=user_metadata['num_users'],
        num_movies=len(movie_df),
        num_age_groups=user_metadata['num_age_groups'],
        num_occupations=user_metadata['num_occupations'],
        embedding_dim=args.embedding_dim,
        num_experts=args.num_experts,
        expert_hidden_dim=args.expert_hidden_dim,
        policy_hidden_dim=args.policy_hidden_dim,
        clip_epsilon=args.clip_epsilon,
        temperature=args.temperature,
        dropout=args.dropout
    ).to(device)

    print(f"Model parameters: {count_parameters(model):,}")

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Early stopping & Checkpoint
    early_stopping = EarlyStopping(patience=args.patience, min_delta=0.001, mode='min')
    checkpoint_manager = CheckpointManager(
        checkpoint_dir=args.checkpoint_dir,
        model_name="grpo_moe",
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
    parser = argparse.ArgumentParser(description="Train GRPO-MoE model")

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
    parser.add_argument("--clip_epsilon", type=float, default=0.2, help="GRPO clip epsilon")
    parser.add_argument("--temperature", type=float, default=1.0, help="Softmax temperature")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")

    # GRPO Training
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size for GRPO updates")
    parser.add_argument("--grpo_epochs", type=int, default=4, help="GRPO update epochs per iteration")
    parser.add_argument("--entropy_coef", type=float, default=0.01, help="Entropy coefficient")
    parser.add_argument("--baseline_coef", type=float, default=0.5, help="Baseline loss coefficient")
    parser.add_argument("--max_grad_norm", type=float, default=0.5, help="Max gradient norm")

    # Training
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=0.0003, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-5, help="Weight decay")
    parser.add_argument("--patience", type=int, default=15, help="Early stopping patience")

    # Misc
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints", help="Checkpoint directory")

    args = parser.parse_args()
    main(args)
