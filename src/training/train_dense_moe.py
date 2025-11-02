"""
Dense MoE 학습 스크립트
Fully Connected Layer 기반 Gating Network (베이스라인)
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse

from src.models import DenseMoE
from src.data.movie_preprocessor import MoviePreprocessor
from src.data.user_preprocessor import UserPreprocessor
from src.data.enhanced_dataset import EnhancedMovieRatingDataset
from src.utils import (
    compute_all_metrics,
    EarlyStopping,
    CheckpointManager,
    set_seed,
    count_parameters,
    get_device,
    AverageMeter,
    print_metrics
)


def train_epoch(model, train_loader, optimizer, device):
    """
    한 epoch 학습

    Args:
        model: DenseMoE 모델
        train_loader: 학습 데이터 로더
        optimizer: Optimizer
        device: 디바이스

    Returns:
        dict: 평균 손실 및 지표
    """
    model.train()
    loss_meter = AverageMeter()
    mse_meter = AverageMeter()

    for batch in tqdm(train_loader, desc="Training"):
        user_id = batch['user_id'].to(device)
        movie_id = batch['movie_id'].to(device)
        age_group = batch['age_group'].to(device)
        gender = batch['gender'].to(device)
        occupation = batch['occupation'].to(device)
        rating = batch['rating'].to(device)

        # Forward
        outputs = model(user_id, movie_id, age_group, gender, occupation)
        loss = model.compute_loss(outputs, rating)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Metrics
        loss_meter.update(loss.item(), user_id.size(0))
        mse_meter.update(
            torch.mean((outputs['rating'] - rating) ** 2).item(),
            user_id.size(0)
        )

    return {
        'loss': loss_meter.avg,
        'mse': mse_meter.avg
    }


def validate_epoch(model, val_loader, device):
    """
    검증

    Args:
        model: DenseMoE 모델
        val_loader: 검증 데이터 로더
        device: 디바이스

    Returns:
        dict: 평균 손실 및 지표
    """
    model.eval()
    loss_meter = AverageMeter()
    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validation"):
            user_id = batch['user_id'].to(device)
            movie_id = batch['movie_id'].to(device)
            age_group = batch['age_group'].to(device)
            gender = batch['gender'].to(device)
            occupation = batch['occupation'].to(device)
            rating = batch['rating'].to(device)

            # Forward
            outputs = model(user_id, movie_id, age_group, gender, occupation)
            loss = model.compute_loss(outputs, rating)

            # Collect predictions
            loss_meter.update(loss.item(), user_id.size(0))
            all_predictions.append(outputs['rating'].cpu())
            all_targets.append(rating.cpu())

    # Compute metrics
    all_predictions = torch.cat(all_predictions)
    all_targets = torch.cat(all_targets)
    metrics = compute_all_metrics(all_predictions, all_targets, denormalize=True)
    metrics['loss'] = loss_meter.avg

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
    model = DenseMoE(
        num_users=user_metadata['num_users'],
        num_movies=len(movie_df),
        num_age_groups=user_metadata['num_age_groups'],
        num_occupations=user_metadata['num_occupations'],
        embedding_dim=args.embedding_dim,
        num_experts=args.num_experts,
        expert_hidden_dim=args.expert_hidden_dim,
        gating_hidden_dim=args.gating_hidden_dim,
        dropout=args.dropout
    ).to(device)

    print(f"Model parameters: {count_parameters(model):,}")

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )

    # Early stopping & Checkpoint
    early_stopping = EarlyStopping(patience=args.patience, min_delta=0.001, mode='min')
    checkpoint_manager = CheckpointManager(
        checkpoint_dir=args.checkpoint_dir,
        model_name="dense_moe",
        max_keep=3
    )

    # 학습 루프
    best_val_loss = float('inf')

    for epoch in range(1, args.epochs + 1):
        print(f"\n{'='*50}")
        print(f"Epoch {epoch}/{args.epochs}")
        print(f"{'='*50}")

        # Train
        train_metrics = train_epoch(model, train_loader, optimizer, device)
        print_metrics(train_metrics, prefix="Train")

        # Validate
        val_metrics = validate_epoch(model, val_loader, device)
        print_metrics(val_metrics, prefix="Val")

        # Learning rate scheduling
        scheduler.step(val_metrics['loss'])

        # Checkpoint
        is_best = val_metrics['loss'] < best_val_loss
        if is_best:
            best_val_loss = val_metrics['loss']

        checkpoint_manager.save_checkpoint(
            model, optimizer, epoch, val_metrics, is_best=is_best
        )

        # Early stopping
        early_stopping(val_metrics['loss'])
        if early_stopping.early_stop:
            print(f"\nEarly stopping triggered at epoch {epoch}")
            break

    print("\nTraining completed!")
    print(f"Best validation loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Dense MoE model")

    # Data
    parser.add_argument("--data_dir", type=str, default="ml-100k", help="Data directory")
    parser.add_argument("--train_rating_path", type=str, default="ml-100k/u1.base", help="Training rating file")
    parser.add_argument("--val_rating_path", type=str, default="ml-100k/u1.test", help="Validation rating file")
    parser.add_argument("--max_length", type=int, default=10, help="Max title length")

    # Model
    parser.add_argument("--embedding_dim", type=int, default=64, help="Embedding dimension")
    parser.add_argument("--num_experts", type=int, default=8, help="Number of experts")
    parser.add_argument("--expert_hidden_dim", type=int, default=256, help="Expert hidden dimension")
    parser.add_argument("--gating_hidden_dim", type=int, default=128, help="Gating hidden dimension")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")

    # Training
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-5, help="Weight decay")
    parser.add_argument("--patience", type=int, default=15, help="Early stopping patience")

    # Misc
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints", help="Checkpoint directory")

    args = parser.parse_args()
    main(args)
