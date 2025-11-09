"""
Evaluation script for all models on 10M dataset.
"""
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse, torch, json
from torch.utils.data import DataLoader
from tqdm import tqdm
from data.movie_preprocessor import MoviePreprocessor
from data.dataset import load_ratings, MovieRatingDataset, train_test_split_temporal
from models.dense_moe import DenseMoE
from models.ppo_moe import PPOMoE
from models.grpo_moe import GRPOMoE
from utils.trainer_utils import get_device
from utils.metrics import compute_all_metrics, compute_expert_distribution


def load_model(model_class, checkpoint_path, num_users, num_movies, num_genres, num_experts, embedding_dim, device):
    model = model_class(num_users, num_movies, num_genres, num_experts, embedding_dim).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    return model


def evaluate_model(model, dataloader, device, model_type='dense'):
    model.eval()
    preds, targets, experts = [], [], []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"Evaluating {model_type}"):
            if len(batch) == 4:
                u, m, r, g = batch
                g = g.to(device)
            else:
                u, m, r = batch
                g = None
            u, m, r = u.to(device), m.to(device), r.to(device)
            
            if model_type == 'dense':
                pred, gate_probs = model(u, m, g)
                preds.append(pred.cpu())
            else:  # ppo or grpo
                out = model(u, m, g)
                preds.append(out['predictions'].cpu())
                experts.append(out['expert_indices'].cpu())
            targets.append(r.cpu())
    
    preds = torch.cat(preds)
    targets = torch.cat(targets)
    metrics = compute_all_metrics(preds, targets, False)
    
    if model_type != 'dense':
        experts = torch.cat(experts)
        expert_dist = compute_expert_distribution(experts)
        metrics['expert_distribution'] = expert_dist
    
    return metrics


def main(args):
    device = get_device()

    # Load data
    mp = MoviePreprocessor()
    mp.fit(os.path.join(args.data_dir, 'movies.dat'))

    # Load full ratings and split (same as training)
    all_ratings = load_ratings(os.path.join(args.data_dir, 'ratings.dat'))
    train_ratings, test_ratings = train_test_split_temporal(all_ratings, test_ratio=0.2)

    # Create datasets (train for num_users, test for evaluation)
    train_ds = MovieRatingDataset(train_ratings, mp, use_genres=args.use_genres)
    test_ds = MovieRatingDataset(test_ratings, mp, use_genres=args.use_genres)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    num_genres = mp.num_genres if args.use_genres else 0
    results = {}
    
    # Evaluate Dense MoE
    if args.dense_checkpoint:
        print("\n=== Evaluating Dense MoE ===")
        model = load_model(DenseMoE, args.dense_checkpoint, train_ds.num_users, mp.num_movies,
                          num_genres, args.num_experts, args.embedding_dim, device)
        results['dense'] = evaluate_model(model, test_loader, device, 'dense')
        print(f"Dense MoE - RMSE: {results['dense']['rmse']:.4f}")

    # Evaluate PPO-MoE
    if args.ppo_checkpoint:
        print("\n=== Evaluating PPO-MoE ===")
        model = load_model(PPOMoE, args.ppo_checkpoint, train_ds.num_users, mp.num_movies,
                          num_genres, args.num_experts, args.embedding_dim, device)
        results['ppo'] = evaluate_model(model, test_loader, device, 'ppo')
        print(f"PPO-MoE - RMSE: {results['ppo']['rmse']:.4f}")

    # Evaluate GRPO-MoE
    if args.grpo_checkpoint:
        print("\n=== Evaluating GRPO-MoE ===")
        model = load_model(GRPOMoE, args.grpo_checkpoint, train_ds.num_users, mp.num_movies,
                          num_genres, args.num_experts, args.embedding_dim, device)
        results['grpo'] = evaluate_model(model, test_loader, device, 'grpo')
        print(f"GRPO-MoE - RMSE: {results['grpo']['rmse']:.4f}")

    # Evaluate PPO Standard
    if args.ppo_standard_checkpoint:
        print("\n=== Evaluating PPO-MoE Standard ===")
        model = load_model(PPOMoE, args.ppo_standard_checkpoint, train_ds.num_users, mp.num_movies,
                          num_genres, args.num_experts, args.embedding_dim, device)
        results['ppo_standard'] = evaluate_model(model, test_loader, device, 'ppo')
        print(f"PPO-MoE Standard - RMSE: {results['ppo_standard']['rmse']:.4f}")

    # Evaluate GRPO Standard
    if args.grpo_standard_checkpoint:
        print("\n=== Evaluating GRPO-MoE Standard ===")
        model = load_model(GRPOMoE, args.grpo_standard_checkpoint, train_ds.num_users, mp.num_movies,
                          num_genres, args.num_experts, args.embedding_dim, device)
        results['grpo_standard'] = evaluate_model(model, test_loader, device, 'grpo')
        print(f"GRPO-MoE Standard - RMSE: {results['grpo_standard']['rmse']:.4f}")
    
    # Save results
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    with open(args.output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {args.output_file}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument('--data_dir', default='ml-10M100K')
    p.add_argument('--test_file', default='ml-10M100K/ratings.dat')
    p.add_argument('--use_genres', action='store_true', default=True)
    p.add_argument('--num_experts', type=int, default=16)
    p.add_argument('--embedding_dim', type=int, default=128)
    p.add_argument('--batch_size', type=int, default=2048)
    p.add_argument('--dense_checkpoint', type=str, default=None)
    p.add_argument('--ppo_checkpoint', type=str, default=None)
    p.add_argument('--grpo_checkpoint', type=str, default=None)
    p.add_argument('--ppo_standard_checkpoint', type=str, default=None)
    p.add_argument('--grpo_standard_checkpoint', type=str, default=None)
    p.add_argument('--output_file', default='results_10m/evaluation.json')
    main(p.parse_args())
