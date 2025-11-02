"""
모델 평가 스크립트
세 가지 MoE 모델(Dense, PPO, GRPO)을 평가
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
import json

from src.models import DenseMoE, PPOMoE, GRPOMoE
from src.data.movie_preprocessor import MoviePreprocessor
from src.data.user_preprocessor import UserPreprocessor
from src.data.enhanced_dataset import EnhancedMovieRatingDataset
from src.utils import (
    compute_all_metrics,
    compute_expert_distribution,
    compute_expert_performance,
    set_seed,
    get_device
)


def evaluate_dense_moe(model, data_loader, device):
    """
    Dense MoE 평가

    Args:
        model: DenseMoE 모델
        data_loader: 데이터 로더
        device: 디바이스

    Returns:
        dict: 평가 결과
    """
    model.eval()
    all_predictions = []
    all_targets = []
    all_gate_probs = []

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating Dense MoE"):
            user_id = batch['user_id'].to(device)
            movie_id = batch['movie_id'].to(device)
            age_group = batch['age_group'].to(device)
            gender = batch['gender'].to(device)
            occupation = batch['occupation'].to(device)
            rating = batch['rating'].to(device)

            # Forward
            outputs = model(user_id, movie_id, age_group, gender, occupation)

            all_predictions.append(outputs['rating'].cpu())
            all_targets.append(rating.cpu())
            all_gate_probs.append(outputs['gate_probs'].cpu())

    all_predictions = torch.cat(all_predictions)
    all_targets = torch.cat(all_targets)
    all_gate_probs = torch.cat(all_gate_probs)

    # 메트릭 계산
    metrics = compute_all_metrics(all_predictions, all_targets, denormalize=True)

    # Gate 확률 통계
    avg_gate_probs = all_gate_probs.mean(dim=0).numpy()
    gate_entropy = -(all_gate_probs * torch.log(all_gate_probs + 1e-10)).sum(dim=1).mean().item()

    metrics['avg_gate_probs'] = avg_gate_probs.tolist()
    metrics['gate_entropy'] = gate_entropy

    return metrics


def evaluate_rl_moe(model, data_loader, device, model_type='ppo'):
    """
    강화학습 MoE 평가 (PPO/GRPO)

    Args:
        model: PPOMoE 또는 GRPOMoE 모델
        data_loader: 데이터 로더
        device: 디바이스
        model_type: 'ppo' 또는 'grpo'

    Returns:
        dict: 평가 결과
    """
    model.eval()
    all_predictions = []
    all_targets = []
    all_actions = []

    with torch.no_grad():
        for batch in tqdm(data_loader, desc=f"Evaluating {model_type.upper()} MoE"):
            user_id = batch['user_id'].to(device)
            movie_id = batch['movie_id'].to(device)
            age_group = batch['age_group'].to(device)
            gender = batch['gender'].to(device)
            occupation = batch['occupation'].to(device)
            rating = batch['rating'].to(device)

            # Forward (deterministic)
            outputs = model(user_id, movie_id, age_group, gender, occupation, deterministic=True)

            all_predictions.append(outputs['rating'].cpu())
            all_targets.append(rating.cpu())
            all_actions.append(outputs['action'].cpu())

    all_predictions = torch.cat(all_predictions)
    all_targets = torch.cat(all_targets)
    all_actions = torch.cat(all_actions)

    # 메트릭 계산
    metrics = compute_all_metrics(all_predictions, all_targets, denormalize=True)

    # Expert 분포
    expert_dist = compute_expert_distribution(all_actions)
    metrics['expert_distribution'] = expert_dist

    # Expert 성능
    errors = torch.abs(all_predictions - all_targets)
    expert_perf = compute_expert_performance(all_actions, errors, num_experts=8)
    metrics['expert_performance'] = expert_perf

    return metrics


def load_model(model_type, checkpoint_path, model_args, device):
    """
    모델 로드

    Args:
        model_type: 'dense', 'ppo', 또는 'grpo'
        checkpoint_path: 체크포인트 파일 경로
        model_args: 모델 생성 인자
        device: 디바이스

    Returns:
        model: 로드된 모델
    """
    if model_type == 'dense':
        model = DenseMoE(**model_args).to(device)
    elif model_type == 'ppo':
        model = PPOMoE(**model_args).to(device)
    elif model_type == 'grpo':
        model = GRPOMoE(**model_args).to(device)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # 체크포인트 로드
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])

    print(f"Loaded {model_type.upper()} model from {checkpoint_path}")
    print(f"Checkpoint epoch: {checkpoint['epoch']}")
    print(f"Checkpoint metrics: {checkpoint['metrics']}")

    return model


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

    # 테스트 데이터셋
    test_dataset = EnhancedMovieRatingDataset(
        movie_data=movie_df,
        user_data=user_df,
        rating_path=args.test_rating_path,
        vocab=vocab,
        max_length=args.max_length
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers
    )

    print(f"Test size: {len(test_dataset)}")

    # 모델 인자
    model_args = {
        'num_users': user_metadata['num_users'],
        'num_movies': len(movie_df),
        'num_age_groups': user_metadata['num_age_groups'],
        'num_occupations': user_metadata['num_occupations'],
        'embedding_dim': args.embedding_dim,
        'num_experts': args.num_experts,
        'expert_hidden_dim': args.expert_hidden_dim,
        'dropout': args.dropout
    }

    results = {}

    # Dense MoE 평가
    if args.dense_checkpoint:
        print(f"\n{'='*50}")
        print("Evaluating Dense MoE")
        print(f"{'='*50}")

        dense_model_args = {**model_args, 'gating_hidden_dim': args.gating_hidden_dim}
        dense_model = load_model('dense', args.dense_checkpoint, dense_model_args, device)
        dense_metrics = evaluate_dense_moe(dense_model, test_loader, device)
        results['dense_moe'] = dense_metrics

        print("\nDense MoE Results:")
        for key, value in dense_metrics.items():
            if key not in ['avg_gate_probs', 'expert_distribution', 'expert_performance']:
                print(f"  {key}: {value:.4f}")

    # PPO-MoE 평가
    if args.ppo_checkpoint:
        print(f"\n{'='*50}")
        print("Evaluating PPO-MoE")
        print(f"{'='*50}")

        ppo_model_args = {
            **model_args,
            'policy_hidden_dim': args.policy_hidden_dim,
            'value_hidden_dim': args.value_hidden_dim,
            'clip_epsilon': args.clip_epsilon
        }
        ppo_model = load_model('ppo', args.ppo_checkpoint, ppo_model_args, device)
        ppo_metrics = evaluate_rl_moe(ppo_model, test_loader, device, model_type='ppo')
        results['ppo_moe'] = ppo_metrics

        print("\nPPO-MoE Results:")
        for key, value in ppo_metrics.items():
            if key not in ['expert_distribution', 'expert_performance']:
                print(f"  {key}: {value:.4f}")
        print(f"  Expert distribution: {ppo_metrics['expert_distribution']}")

    # GRPO-MoE 평가
    if args.grpo_checkpoint:
        print(f"\n{'='*50}")
        print("Evaluating GRPO-MoE")
        print(f"{'='*50}")

        grpo_model_args = {
            **model_args,
            'policy_hidden_dim': args.policy_hidden_dim,
            'clip_epsilon': args.clip_epsilon,
            'temperature': args.temperature
        }
        grpo_model = load_model('grpo', args.grpo_checkpoint, grpo_model_args, device)
        grpo_metrics = evaluate_rl_moe(grpo_model, test_loader, device, model_type='grpo')
        results['grpo_moe'] = grpo_metrics

        print("\nGRPO-MoE Results:")
        for key, value in grpo_metrics.items():
            if key not in ['expert_distribution', 'expert_performance']:
                print(f"  {key}: {value:.4f}")
        print(f"  Expert distribution: {grpo_metrics['expert_distribution']}")

    # 결과 비교
    if len(results) > 1:
        print(f"\n{'='*50}")
        print("Model Comparison")
        print(f"{'='*50}")
        print(f"{'Model':<15} {'MSE':<10} {'RMSE':<10} {'MAE':<10}")
        print("-" * 50)
        for model_name, metrics in results.items():
            print(f"{model_name:<15} {metrics['mse']:<10.4f} {metrics['rmse']:<10.4f} {metrics['mae']:<10.4f}")

    # 결과 저장
    if args.output_file:
        with open(args.output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate MoE models")

    # Data
    parser.add_argument("--data_dir", type=str, default="ml-100k", help="Data directory")
    parser.add_argument("--test_rating_path", type=str, default="ml-100k/u1.test", help="Test rating file")
    parser.add_argument("--max_length", type=int, default=10, help="Max title length")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers")

    # Model checkpoints
    parser.add_argument("--dense_checkpoint", type=str, default=None, help="Dense MoE checkpoint")
    parser.add_argument("--ppo_checkpoint", type=str, default=None, help="PPO-MoE checkpoint")
    parser.add_argument("--grpo_checkpoint", type=str, default=None, help="GRPO-MoE checkpoint")

    # Model args
    parser.add_argument("--embedding_dim", type=int, default=64, help="Embedding dimension")
    parser.add_argument("--num_experts", type=int, default=8, help="Number of experts")
    parser.add_argument("--expert_hidden_dim", type=int, default=256, help="Expert hidden dimension")
    parser.add_argument("--gating_hidden_dim", type=int, default=128, help="Gating hidden dimension (Dense)")
    parser.add_argument("--policy_hidden_dim", type=int, default=128, help="Policy hidden dimension (RL)")
    parser.add_argument("--value_hidden_dim", type=int, default=128, help="Value hidden dimension (PPO)")
    parser.add_argument("--clip_epsilon", type=float, default=0.2, help="Clip epsilon (RL)")
    parser.add_argument("--temperature", type=float, default=1.0, help="Temperature (GRPO)")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")

    # Misc
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output_file", type=str, default="evaluation_results.json", help="Output file")

    args = parser.parse_args()

    if not any([args.dense_checkpoint, args.ppo_checkpoint, args.grpo_checkpoint]):
        parser.error("At least one checkpoint must be provided")

    main(args)
