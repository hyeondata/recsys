"""
Dense MoE 학습 스크립트 (ml-20m 버전)
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

from src_20m.models import DenseMoE
from src_20m.data.movie_preprocessor import MoviePreprocessor
from src_20m.data.dataset import MovieRatingDataset, load_and_split_ratings
from src_20m.utils import (
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
    """한 epoch 학습"""
    model.train()
    loss_meter = AverageMeter()
    mse_meter = AverageMeter()

    for batch in tqdm(train_loader, desc="Training"):
        user_id = batch['user_id'].to(device)
        movie_id = batch['movie_id'].to(device)
        rating = batch['rating'].to(device)

        # Forward
        outputs = model(user_id, movie_id)
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
    """검증"""
    model.eval()
    loss_meter = AverageMeter()
    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validation"):
            user_id = batch['user_id'].to(device)
            movie_id = batch['movie_id'].to(device)
            rating = batch['rating'].to(device)

            # Forward
            outputs = model(user_id, movie_id)
            loss = model.compute_loss(outputs, rating)

            # Collect predictions
            loss_meter.update(loss.item(), user_id.size(0))
            all_predictions.append(outputs['rating'].cpu())
            all_targets.append(rating.cpu())

    # Compute metrics
    all_predictions = torch.cat(all_predictions)
    all_targets = torch.cat(all_targets)
    metrics = compute_all_metrics(all_predictions, all_targets, denormalize=False)
    metrics['loss'] = loss_meter.avg

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
    
    model = DenseMoE(
        num_users=num_users,
        num_movies=num_movies,
        embedding_dim=args.embedding_dim,
        num_experts=args.num_experts,
        expert_hidden_dim=args.expert_hidden_dim,
        dropout=args.dropout
    ).to(device)

    print(f"Model parameters: {count_parameters(model):,}")
    print(f"Users: {num_users:,}, Movies: {num_movies:,}")

    # Optimizer & Scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )

    # Training utilities
    early_stopping = EarlyStopping(patience=args.patience)
    checkpoint_manager = CheckpointManager(args.checkpoint_dir, model_name='dense_moe', max_keep=3)

    # Training loop
    print(f"\nStarting training for {args.epochs} epochs...")
    for epoch in range(1, args.epochs + 1):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch}/{args.epochs}")
        print('='*60)

        # Train
        train_metrics = train_epoch(model, train_loader, optimizer, device)
        print(f"Train Loss: {train_metrics['loss']:.4f} | Train MSE: {train_metrics['mse']:.4f}")

        # Validate
        val_metrics = validate_epoch(model, val_loader, device)
        print(f"Val Loss: {val_metrics['loss']:.4f}")
        print_metrics(val_metrics, prefix="Val")

        # Scheduler step
        scheduler.step(val_metrics['loss'])

        # Save checkpoint
        checkpoint_manager.save_checkpoint(
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            metrics=val_metrics,
            config=vars(args)
        )

        # Early stopping
        early_stopping(val_metrics['rmse'])
        if early_stopping.early_stop:
            print("Early stopping triggered!")
            break

    print("\nTraining complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    # Data
    parser.add_argument('--data_dir', type=str, default='ml-20m')
    parser.add_argument('--max_length', type=int, default=10)
    
    # Model
    parser.add_argument('--embedding_dim', type=int, default=128)
    parser.add_argument('--num_experts', type=int, default=16)
    parser.add_argument('--expert_hidden_dim', type=int, default=512)
    parser.add_argument('--dropout', type=float, default=0.2)
    
    # Training
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=42)
    
    # Checkpoint
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints_20m/dense_moe')
    
    args = parser.parse_args()
    main(args)
