"""
Training script for Dense MoE on 10M dataset.
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.movie_preprocessor import MoviePreprocessor
from data.dataset import load_ratings, train_test_split_temporal, MovieRatingDataset
from models.dense_moe import DenseMoE
from utils.trainer_utils import (
    EarlyStopping, CheckpointManager, set_seed, count_parameters, get_device
)
from utils.metrics import compute_all_metrics


def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    num_batches = 0

    with tqdm(dataloader, desc="Training") as pbar:
        for batch in pbar:
            if len(batch) == 4:
                user_ids, movie_ids, ratings, genres = batch
                genres = genres.to(device)
            else:
                user_ids, movie_ids, ratings = batch
                genres = None

            user_ids = user_ids.to(device)
            movie_ids = movie_ids.to(device)
            ratings = ratings.to(device)

            optimizer.zero_grad()

            # Forward pass
            predictions, gate_probs = model(user_ids, movie_ids, genres)

            # Compute loss
            loss = criterion(predictions, ratings)

            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

            pbar.set_postfix({'loss': loss.item()})

    avg_loss = total_loss / num_batches
    return avg_loss


def evaluate(model, dataloader, device):
    """Evaluate model."""
    model.eval()
    all_predictions = []
    all_targets = []

    with torch.no_grad():
        with tqdm(dataloader, desc="Evaluating") as pbar:
            for batch in pbar:
                if len(batch) == 4:
                    user_ids, movie_ids, ratings, genres = batch
                    genres = genres.to(device)
                else:
                    user_ids, movie_ids, ratings = batch
                    genres = None

                user_ids = user_ids.to(device)
                movie_ids = movie_ids.to(device)
                ratings = ratings.to(device)

                # Forward pass
                predictions, gate_probs = model(user_ids, movie_ids, genres)

                all_predictions.append(predictions.cpu())
                all_targets.append(ratings.cpu())

    all_predictions = torch.cat(all_predictions)
    all_targets = torch.cat(all_targets)

    metrics = compute_all_metrics(all_predictions, all_targets, denormalize=False)
    return metrics


def main(args):
    # Set seed
    set_seed(args.seed)

    # Device
    device = get_device()
    print(f"Using device: {device}")

    # Load data
    print("\n=== Loading Data ===")
    movie_preprocessor = MoviePreprocessor()
    movie_preprocessor.fit(os.path.join(args.data_dir, 'movies.dat'))

    # Load ratings
    all_ratings = load_ratings(os.path.join(args.data_dir, 'ratings.dat'))

    # Train/test split
    train_ratings, test_ratings = train_test_split_temporal(all_ratings, test_ratio=args.test_ratio)

    # Create datasets
    train_dataset = MovieRatingDataset(train_ratings, movie_preprocessor, use_genres=args.use_genres)
    test_dataset = MovieRatingDataset(test_ratings, movie_preprocessor, use_genres=args.use_genres)

    # Dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size * 2,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )

    print(f"\nDataset info:")
    print(f"  Train: {len(train_dataset)} ratings")
    print(f"  Test: {len(test_dataset)} ratings")

    # Model
    print("\n=== Building Model ===")
    num_genres = movie_preprocessor.num_genres if args.use_genres else 0
    model = DenseMoE(
        num_users=train_dataset.num_users,
        num_movies=movie_preprocessor.num_movies,
        num_genres=num_genres,
        num_experts=args.num_experts,
        embedding_dim=args.embedding_dim,
        dropout=args.dropout
    ).to(device)

    print(f"Model parameters: {count_parameters(model):,}")

    # Optimizer and criterion
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.MSELoss()

    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )

    # Early stopping and checkpoint manager
    early_stopping = EarlyStopping(patience=args.patience, mode='min')
    checkpoint_manager = CheckpointManager(args.checkpoint_dir, 'dense_moe', max_keep=3)

    # Resume functionality
    start_epoch = 1
    best_rmse = float('inf')

    if args.resume:
        latest_checkpoint = checkpoint_manager.get_latest_checkpoint()
        if latest_checkpoint:
            try:
                checkpoint_info = checkpoint_manager.load_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    filename=latest_checkpoint,
                    scheduler=scheduler,
                    early_stopping=early_stopping
                )
                start_epoch = checkpoint_info['epoch'] + 1
                best_rmse = checkpoint_info['metrics'].get('rmse', float('inf'))

                print(f"\n{'='*50}")
                print(f"Resumed from epoch {checkpoint_info['epoch']}")
                print(f"Best RMSE: {best_rmse:.4f}")
                print(f"{'='*50}\n")
            except Exception as e:
                print(f"Warning: Failed to load checkpoint: {e}")
                print("Starting from scratch...")
        else:
            print("No checkpoint found. Starting from scratch...")

    # Training loop
    print("\n=== Training ===")

    for epoch in range(start_epoch, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")

        # Train
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        print(f"Train Loss: {train_loss:.4f}")

        # Evaluate
        test_metrics = evaluate(model, test_loader, device)
        print(f"Test - RMSE: {test_metrics['rmse']:.4f}, MAE: {test_metrics['mae']:.4f}")

        # Scheduler step
        scheduler.step(test_metrics['rmse'])

        # Check for improvement
        is_best = test_metrics['rmse'] < best_rmse
        if is_best:
            best_rmse = test_metrics['rmse']
            print(f"New best RMSE: {best_rmse:.4f}")

        # Save checkpoint
        checkpoint_manager.save_checkpoint(
            model, optimizer, epoch, test_metrics, is_best=is_best
        )

        # Early stopping
        if early_stopping(test_metrics['rmse']):
            if early_stopping.early_stop:
                print(f"\nEarly stopping triggered at epoch {epoch}")
                break

    print(f"\n=== Training Complete ===")
    print(f"Best RMSE: {best_rmse:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Dense MoE on 10M dataset")

    # Data
    parser.add_argument('--data_dir', type=str, default='ml-10M100K',
                        help='Path to ml-10M100K directory')
    parser.add_argument('--test_ratio', type=float, default=0.2,
                        help='Test set ratio')
    parser.add_argument('--use_genres', action='store_true', default=True,
                        help='Use genre features')

    # Model
    parser.add_argument('--num_experts', type=int, default=8,
                        help='Number of experts')
    parser.add_argument('--embedding_dim', type=int, default=128,
                        help='Embedding dimension')
    parser.add_argument('--dropout', type=float, default=0.2,
                        help='Dropout probability')

    # Training
    parser.add_argument('--epochs', type=int, default=20,
                        help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=2048,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                        help='Weight decay')
    parser.add_argument('--patience', type=int, default=10,
                        help='Early stopping patience')
    parser.add_argument('--num_workers', type=int, default=8,
                        help='Number of data loading workers')

    # Checkpoint
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints_10m/dense_moe',
                        help='Checkpoint directory')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--resume', action='store_true',
                        help='Resume training from latest checkpoint')

    args = parser.parse_args()
    main(args)
