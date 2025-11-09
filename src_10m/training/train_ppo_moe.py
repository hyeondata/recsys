"""
Training script for PPO-MoE on 10M dataset (Batch-wise approach).
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
from models.ppo_moe import PPOMoE
from utils.trainer_utils import (
    EarlyStopping, CheckpointManager, set_seed, count_parameters, get_device
)
from utils.metrics import compute_all_metrics, compute_expert_distribution


def compute_gae(rewards, values, gamma=0.99, lam=0.95):
    """Compute Generalized Advantage Estimation."""
    advantages = []
    gae = 0

    for t in reversed(range(len(rewards))):
        if t == len(rewards) - 1:
            next_value = 0
        else:
            next_value = values[t + 1]

        delta = rewards[t] + gamma * next_value - values[t]
        gae = delta + gamma * lam * gae
        advantages.insert(0, gae)

    return torch.tensor(advantages, dtype=torch.float32)


def train_epoch(model, dataloader, optimizer, device, args):
    """Train for one epoch with PPO."""
    model.train()
    total_policy_loss = 0
    total_value_loss = 0
    total_rating_loss = 0
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

            # Collect episode
            with torch.no_grad():
                old_outputs = model(user_ids, movie_ids, genres)
                old_log_probs = old_outputs['log_probs']
                values = old_outputs['values']
                predictions = old_outputs['predictions']

                # Compute rewards
                rewards = -torch.abs(predictions - ratings)

                # Compute advantages
                advantages = rewards - values
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            # PPO update epochs
            for _ in range(args.ppo_epochs):
                outputs = model(user_ids, movie_ids, genres, old_outputs['expert_indices'])
                new_log_probs = outputs['log_probs']
                new_values = outputs['values']
                new_predictions = outputs['predictions']

                # PPO policy loss
                ratio = torch.exp(new_log_probs - old_log_probs)
                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 1 - args.clip_epsilon, 1 + args.clip_epsilon) * advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss
                value_loss = nn.MSELoss()(new_values, rewards)

                # Rating prediction loss
                rating_loss = nn.MSELoss()(new_predictions, ratings)

                # Entropy bonus
                policy_logits = model.policy(model.get_state(user_ids, movie_ids, genres)[0] if genres is None
                                             else torch.cat(model.get_state(user_ids, movie_ids, genres), dim=1))
                entropy = -(torch.softmax(policy_logits, dim=1) * torch.log_softmax(policy_logits, dim=1)).sum(dim=1).mean()

                # Total loss
                loss = (policy_loss +
                       args.value_coef * value_loss +
                       rating_loss -
                       args.entropy_coef * entropy)

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_rating_loss += rating_loss.item()
                num_batches += 1

            pbar.set_postfix({
                'policy_loss': policy_loss.item(),
                'rating_loss': rating_loss.item()
            })

    return {
        'policy_loss': total_policy_loss / num_batches,
        'value_loss': total_value_loss / num_batches,
        'rating_loss': total_rating_loss / num_batches
    }


def evaluate(model, dataloader, device):
    """Evaluate model."""
    model.eval()
    all_predictions = []
    all_targets = []
    all_expert_indices = []

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

                outputs = model(user_ids, movie_ids, genres)
                predictions = outputs['predictions']
                expert_indices = outputs['expert_indices']

                all_predictions.append(predictions.cpu())
                all_targets.append(ratings.cpu())
                all_expert_indices.append(expert_indices.cpu())

    all_predictions = torch.cat(all_predictions)
    all_targets = torch.cat(all_targets)
    all_expert_indices = torch.cat(all_expert_indices)

    metrics = compute_all_metrics(all_predictions, all_targets, denormalize=False)
    expert_dist = compute_expert_distribution(all_expert_indices)

    return metrics, expert_dist


def main(args):
    set_seed(args.seed)
    device = get_device()
    print(f"Using device: {device}")

    # Load data
    print("\n=== Loading Data ===")
    movie_preprocessor = MoviePreprocessor()
    movie_preprocessor.fit(os.path.join(args.data_dir, 'movies.dat'))

    all_ratings = load_ratings(os.path.join(args.data_dir, 'ratings.dat'))
    train_ratings, test_ratings = train_test_split_temporal(all_ratings, test_ratio=args.test_ratio)

    train_dataset = MovieRatingDataset(train_ratings, movie_preprocessor, use_genres=args.use_genres)
    test_dataset = MovieRatingDataset(test_ratings, movie_preprocessor, use_genres=args.use_genres)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size * 2, shuffle=False,
                             num_workers=args.num_workers, pin_memory=True)

    # Model
    print("\n=== Building Model ===")
    num_genres = movie_preprocessor.num_genres if args.use_genres else 0
    model = PPOMoE(
        num_users=train_dataset.num_users,
        num_movies=movie_preprocessor.num_movies,
        num_genres=num_genres,
        num_experts=args.num_experts,
        embedding_dim=args.embedding_dim,
        dropout=args.dropout
    ).to(device)

    print(f"Model parameters: {count_parameters(model):,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    early_stopping = EarlyStopping(patience=args.patience, mode='min')
    checkpoint_manager = CheckpointManager(args.checkpoint_dir, 'ppo_moe', max_keep=3)

    # Resume from checkpoint if specified
    start_epoch = 1
    best_rmse = float('inf')

    if args.resume:
        print("\n=== Resuming from Checkpoint ===")
        try:
            # Try to load latest checkpoint
            latest_checkpoint = checkpoint_manager.get_latest_checkpoint()
            if latest_checkpoint:
                checkpoint_info = checkpoint_manager.load_checkpoint(
                    model, optimizer, latest_checkpoint, scheduler, early_stopping
                )
                start_epoch = checkpoint_info['epoch'] + 1
                best_rmse = checkpoint_info['metrics'].get('rmse', float('inf'))
                print(f"Resumed from epoch {checkpoint_info['epoch']}")
                print(f"Previous best RMSE: {best_rmse:.4f}")
            else:
                print("No checkpoint found. Starting from scratch.")
        except Exception as e:
            print(f"Failed to load checkpoint: {e}")
            print("Starting from scratch.")

    # Training loop
    print("\n=== Training ===")
    if args.resume:
        print(f"Starting from epoch {start_epoch}")

    for epoch in range(start_epoch, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")

        losses = train_epoch(model, train_loader, optimizer, device, args)
        print(f"Train - Policy: {losses['policy_loss']:.4f}, Rating: {losses['rating_loss']:.4f}")

        test_metrics, expert_dist = evaluate(model, test_loader, device)
        print(f"Test - RMSE: {test_metrics['rmse']:.4f}, MAE: {test_metrics['mae']:.4f}")
        print(f"Expert distribution: {expert_dist}")

        scheduler.step(test_metrics['rmse'])

        is_best = test_metrics['rmse'] < best_rmse
        if is_best:
            best_rmse = test_metrics['rmse']
            print(f"New best RMSE: {best_rmse:.4f}")

        checkpoint_manager.save_checkpoint(model, optimizer, epoch, test_metrics, is_best=is_best,
                                          scheduler=scheduler, early_stopping=early_stopping)

        if early_stopping(test_metrics['rmse']):
            if early_stopping.early_stop:
                print(f"\nEarly stopping at epoch {epoch}")
                break

    print(f"\n=== Training Complete ===")
    print(f"Best RMSE: {best_rmse:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='ml-10M100K')
    parser.add_argument('--test_ratio', type=float, default=0.2)
    parser.add_argument('--use_genres', action='store_true', default=True)
    parser.add_argument('--num_experts', type=int, default=8)
    parser.add_argument('--embedding_dim', type=int, default=128)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--lr', type=float, default=0.0003)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--ppo_epochs', type=int, default=4)
    parser.add_argument('--clip_epsilon', type=float, default=0.2)
    parser.add_argument('--entropy_coef', type=float, default=0.01)
    parser.add_argument('--value_coef', type=float, default=0.5)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints_10m/ppo_moe')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--resume', action='store_true', help='Resume training from latest checkpoint')
    args = parser.parse_args()
    main(args)
