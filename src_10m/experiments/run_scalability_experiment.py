"""
Scalability experiment: Compare models with different numbers of experts.

Runs experiments with varying expert counts (e.g., 8, 16, 32, 64, 128)
to analyze scalability and efficiency trade-offs.
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from datetime import datetime

from data.movie_preprocessor import MoviePreprocessor
from data.dataset import load_ratings, train_test_split_temporal, MovieRatingDataset
from models.dense_moe import DenseMoE
from models.ppo_moe import PPOMoE
from models.grpo_moe import GRPOMoE
from utils.trainer_utils import (
    EarlyStopping, CheckpointManager, set_seed, get_device
)
from utils.metrics import compute_all_metrics
from efficiency_metrics import (
    EfficiencyTracker, count_model_parameters, compare_efficiency
)


def train_epoch_with_tracking(model, dataloader, criterion, optimizer, device, tracker, model_type='dense'):
    """Train for one epoch with efficiency tracking."""
    model.train()
    total_loss = 0
    num_batches = 0

    tracker.start_epoch()

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

            batch_size = user_ids.size(0)

            tracker.start_batch()
            optimizer.zero_grad()

            # Forward pass
            if model_type == 'dense':
                predictions, gate_probs = model(user_ids, movie_ids, genres)
            else:  # PPO/GRPO
                predictions, gate_probs, _, _ = model(user_ids, movie_ids, genres)

            # Compute loss
            loss = criterion(predictions, ratings)

            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            batch_time, throughput = tracker.end_batch(batch_size)

            # Track expert distribution
            tracker.record_expert_distribution(gate_probs.detach())

            # Track memory (every 10 batches to reduce overhead)
            if num_batches % 10 == 0:
                tracker.record_memory()

            total_loss += loss.item()
            num_batches += 1

            pbar.set_postfix({
                'loss': loss.item(),
                'throughput': f'{throughput:.1f} s/s'
            })

    epoch_time = tracker.end_epoch()
    avg_loss = total_loss / num_batches

    return avg_loss, epoch_time


def evaluate_with_tracking(model, dataloader, device, tracker, model_type='dense'):
    """Evaluate model with efficiency tracking."""
    model.eval()
    all_predictions = []
    all_targets = []
    all_gate_probs = []

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
                if model_type == 'dense':
                    predictions, gate_probs = model(user_ids, movie_ids, genres)
                else:
                    predictions, gate_probs, _, _ = model(user_ids, movie_ids, genres)

                all_predictions.append(predictions.cpu())
                all_targets.append(ratings.cpu())
                all_gate_probs.append(gate_probs.cpu())

    all_predictions = torch.cat(all_predictions)
    all_targets = torch.cat(all_targets)
    all_gate_probs = torch.cat(all_gate_probs)

    # Compute metrics
    metrics = compute_all_metrics(all_predictions, all_targets, denormalize=False)

    # Expert utilization on test set
    avg_expert_usage = all_gate_probs.mean(dim=0).numpy()
    metrics['test_expert_usage'] = avg_expert_usage.tolist()

    return metrics


def run_experiment(
    num_experts: int,
    model_type: str,
    args,
    train_loader,
    test_loader,
    num_users,
    num_movies,
    num_genres,
    device
):
    """
    Run a single experiment with specified number of experts.

    Args:
        num_experts: Number of experts to use
        model_type: 'dense', 'ppo', or 'grpo'
        args: Command line arguments
        train_loader: Training data loader
        test_loader: Test data loader
        num_users: Number of users
        num_movies: Number of movies
        num_genres: Number of genres
        device: Device to use

    Returns:
        Dict with results and efficiency metrics
    """
    print(f"\n{'='*60}")
    print(f"Running experiment: {model_type.upper()} with {num_experts} experts")
    print(f"{'='*60}")

    # Create model
    if model_type == 'dense':
        model = DenseMoE(
            num_users=num_users,
            num_movies=num_movies,
            num_genres=num_genres,
            num_experts=num_experts,
            embedding_dim=args.embedding_dim,
            dropout=args.dropout
        ).to(device)
    elif model_type == 'ppo':
        model = PPOMoE(
            num_users=num_users,
            num_movies=num_movies,
            num_genres=num_genres,
            num_experts=num_experts,
            embedding_dim=args.embedding_dim,
            dropout=args.dropout
        ).to(device)
    elif model_type == 'grpo':
        model = GRPOMoE(
            num_users=num_users,
            num_movies=num_movies,
            num_genres=num_genres,
            num_experts=num_experts,
            embedding_dim=args.embedding_dim,
            dropout=args.dropout
        ).to(device)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # Count parameters
    param_counts = count_model_parameters(model)
    print(f"Model parameters: {param_counts['total']:,}")

    # Optimizer and criterion
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    # Early stopping
    early_stopping = EarlyStopping(patience=args.patience, min_delta=args.min_delta)

    # Efficiency tracker
    tracker = EfficiencyTracker()

    # Training loop
    best_rmse = float('inf')
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")

        # Train
        train_loss, epoch_time = train_epoch_with_tracking(
            model, train_loader, criterion, optimizer, device, tracker, model_type
        )

        # Evaluate
        test_metrics = evaluate_with_tracking(model, test_loader, device, tracker, model_type)

        rmse = test_metrics['rmse']
        mae = test_metrics['mae']

        print(f"Epoch {epoch+1} - Loss: {train_loss:.4f}, "
              f"RMSE: {rmse:.4f}, MAE: {mae:.4f}, "
              f"Time: {epoch_time:.1f}s")

        # Track best model
        if rmse < best_rmse:
            best_rmse = rmse
            best_metrics = test_metrics.copy()

        # Early stopping
        early_stopping(rmse)
        if early_stopping.early_stop:
            print(f"Early stopping at epoch {epoch+1}")
            break

    # Get efficiency summary
    efficiency_summary = tracker.get_summary()

    # Compile results
    results = {
        'model_type': model_type,
        'num_experts': num_experts,
        'parameters': param_counts,
        'best_rmse': best_rmse,
        'best_metrics': best_metrics,
        'efficiency': efficiency_summary,
        'epochs_trained': epoch + 1,
        'early_stopped': early_stopping.early_stop
    }

    # Clean up
    del model
    tracker.clear_gpu_cache()

    return results


def main(args):
    # Set seed
    set_seed(args.seed)

    # Device
    device = get_device()
    print(f"Using device: {device}")

    # Load data (only once)
    print("\n=== Loading Data ===")
    movie_preprocessor = MoviePreprocessor()
    movie_preprocessor.fit(os.path.join(args.data_dir, 'movies.dat'))

    all_ratings = load_ratings(os.path.join(args.data_dir, 'ratings.dat'))
    train_ratings, test_ratings = train_test_split_temporal(all_ratings, test_ratio=args.test_ratio)

    train_dataset = MovieRatingDataset(train_ratings, movie_preprocessor, use_genres=args.use_genres)
    test_dataset = MovieRatingDataset(test_ratings, movie_preprocessor, use_genres=args.use_genres)

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

    print(f"Train: {len(train_dataset)} ratings, Test: {len(test_dataset)} ratings")

    num_genres = movie_preprocessor.num_genres if args.use_genres else 0

    # Run experiments for each expert count
    all_results = []

    for num_experts in args.expert_counts:
        result = run_experiment(
            num_experts=num_experts,
            model_type=args.model_type,
            args=args,
            train_loader=train_loader,
            test_loader=test_loader,
            num_users=train_dataset.num_users,
            num_movies=movie_preprocessor.num_movies,
            num_genres=num_genres,
            device=device
        )
        all_results.append(result)

        # Save intermediate results
        os.makedirs(args.output_dir, exist_ok=True)
        results_file = os.path.join(args.output_dir, f'scalability_results_{args.model_type}.json')
        with open(results_file, 'w') as f:
            json.dump(all_results, f, indent=2)

        print(f"\nIntermediate results saved to: {results_file}")

    # Compute comparison metrics
    print("\n" + "="*60)
    print("Computing Efficiency Comparison")
    print("="*60)

    # Prepare data for comparison
    comparison_data = []
    for r in all_results:
        data = {
            'num_experts': r['num_experts'],
            'rmse': r['best_rmse'],
            **r['efficiency']
        }
        comparison_data.append(data)

    comparison = compare_efficiency(comparison_data, baseline_experts=args.baseline_experts)

    # Save final results
    final_results = {
        'experiment': {
            'model_type': args.model_type,
            'timestamp': datetime.now().isoformat(),
            'expert_counts': args.expert_counts,
            'baseline_experts': args.baseline_experts,
            'epochs': args.epochs,
            'batch_size': args.batch_size
        },
        'individual_results': all_results,
        'comparison': comparison
    }

    final_file = os.path.join(args.output_dir, f'scalability_analysis_{args.model_type}.json')
    with open(final_file, 'w') as f:
        json.dump(final_results, f, indent=2)

    print(f"\nFinal results saved to: {final_file}")

    # Print summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)

    print(f"\nModel: {args.model_type.upper()}")
    print(f"\nExpert Count | RMSE   | Time/Epoch | Peak GPU Memory | Throughput")
    print("-" * 70)
    for r in all_results:
        num_exp = r['num_experts']
        rmse = r['best_rmse']
        time_per_epoch = r['efficiency'].get('avg_epoch_time', 0)
        peak_mem = r['efficiency'].get('peak_gpu_memory', 0)
        throughput = r['efficiency'].get('avg_throughput', 0)

        print(f"{num_exp:12d} | {rmse:.4f} | {time_per_epoch:10.1f}s | {peak_mem:14.2f} GB | {throughput:10.1f} s/s")

    print("\nScaling Efficiency (relative to baseline):")
    if 'comparisons' in comparison:
        print(f"\nExperts | Scale | Time Eff | Memory Eff | Entropy | Gini")
        print("-" * 65)
        for comp in comparison['comparisons']:
            n_exp = comp['num_experts']
            scale = comp['scale_factor']
            time_eff = comp.get('time_efficiency', 0)
            mem_eff = comp.get('memory_efficiency', 0)
            entropy = comp.get('expert_entropy', 0)
            gini = comp.get('expert_gini', 0)

            print(f"{n_exp:7d} | {scale:5.1f}x | {time_eff:8.2f} | {mem_eff:10.2f} | {entropy:7.3f} | {gini:4.3f}")

    print("\n" + "="*60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Scalability Experiment")

    # Data
    parser.add_argument('--data_dir', type=str, default='ml-10M100K',
                        help='Directory containing ratings.dat and movies.dat')
    parser.add_argument('--test_ratio', type=float, default=0.2,
                        help='Test set ratio')
    parser.add_argument('--use_genres', action='store_true', default=True,
                        help='Use genre features')

    # Experiment setup
    parser.add_argument('--model_type', type=str, default='dense',
                        choices=['dense', 'ppo', 'grpo'],
                        help='Model type to test')
    parser.add_argument('--expert_counts', type=int, nargs='+',
                        default=[8, 16, 32, 64],
                        help='List of expert counts to test (e.g., --expert_counts 8 16 32 64)')
    parser.add_argument('--baseline_experts', type=int, default=16,
                        help='Baseline number of experts for comparison')

    # Model architecture
    parser.add_argument('--embedding_dim', type=int, default=128,
                        help='Embedding dimension')
    parser.add_argument('--dropout', type=float, default=0.2,
                        help='Dropout rate')

    # Training
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of epochs per experiment')
    parser.add_argument('--batch_size', type=int, default=1024,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--patience', type=int, default=5,
                        help='Early stopping patience')
    parser.add_argument('--min_delta', type=float, default=0.001,
                        help='Early stopping min delta')

    # System
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')

    # Output
    parser.add_argument('--output_dir', type=str, default='results_10m/scalability',
                        help='Output directory for results')

    args = parser.parse_args()

    main(args)
