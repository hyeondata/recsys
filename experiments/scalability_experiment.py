"""
확장성 실험 스크립트
다양한 Expert 개수로 모델을 학습하고 성능을 비교

이 스크립트는 "확장성"과 "효율성"의 차이를 명확하게 보여줍니다:
- Dense MoE: 모든 Expert 사용 → Expert 수 증가 시 계산량 선형 증가
- RL MoE: 단일 Expert 선택 → Expert 수 증가해도 계산량 일정
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import argparse
import json
import time
import psutil
import os
from tqdm import tqdm

from src.models import DenseMoE
from src.models.ppo_moe import PPOMoE
from src.models.grpo_moe import GRPOMoE
from src.data.movie_preprocessor import MoviePreprocessor
from src.data.user_preprocessor import UserPreprocessor
from src.data.enhanced_dataset import EnhancedMovieRatingDataset
from src.utils import (
    compute_all_metrics,
    set_seed,
    count_parameters,
    get_device,
)


def measure_inference_time(model, test_loader, device, num_iterations=5):
    """
    추론 시간 측정

    Args:
        model: 모델
        test_loader: 테스트 데이터 로더
        device: 디바이스
        num_iterations: 측정 반복 횟수

    Returns:
        float: 평균 추론 시간 (초)
    """
    model.eval()
    times = []

    # 워밍업
    with torch.no_grad():
        for batch in test_loader:
            user_id = batch['user_id'].to(device)
            movie_id = batch['movie_id'].to(device)
            age_group = batch['age_group'].to(device)
            gender = batch['gender'].to(device)
            occupation = batch['occupation'].to(device)
            _ = model(user_id, movie_id, age_group, gender, occupation)
            break

    # 실제 측정
    with torch.no_grad():
        for _ in range(num_iterations):
            start_time = time.time()
            for batch in test_loader:
                user_id = batch['user_id'].to(device)
                movie_id = batch['movie_id'].to(device)
                age_group = batch['age_group'].to(device)
                gender = batch['gender'].to(device)
                occupation = batch['occupation'].to(device)
                _ = model(user_id, movie_id, age_group, gender, occupation)
            end_time = time.time()
            times.append(end_time - start_time)

    return sum(times) / len(times)


def measure_memory_usage(model, batch, device):
    """
    메모리 사용량 측정

    Args:
        model: 모델
        batch: 배치 데이터
        device: 디바이스

    Returns:
        dict: 메모리 정보
    """
    if device.type == 'cuda':
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        model.eval()
        with torch.no_grad():
            user_id = batch['user_id'].to(device)
            movie_id = batch['movie_id'].to(device)
            age_group = batch['age_group'].to(device)
            gender = batch['gender'].to(device)
            occupation = batch['occupation'].to(device)
            _ = model(user_id, movie_id, age_group, gender, occupation)

        memory_allocated = torch.cuda.max_memory_allocated() / 1024**2  # MB
        memory_reserved = torch.cuda.max_memory_reserved() / 1024**2  # MB

        return {
            'allocated_mb': memory_allocated,
            'reserved_mb': memory_reserved
        }
    else:
        return {
            'allocated_mb': 0,
            'reserved_mb': 0
        }


def train_single_epoch(model, train_loader, optimizer, device, model_type='dense'):
    """
    단일 epoch 학습 (간소화 버전)

    Args:
        model: 모델
        train_loader: 학습 데이터 로더
        optimizer: Optimizer
        device: 디바이스
        model_type: 모델 타입 ('dense', 'ppo', 'grpo')

    Returns:
        dict: 학습 메트릭
    """
    model.train()
    total_loss = 0
    num_batches = 0

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

        total_loss += loss.item()
        num_batches += 1

    return {
        'loss': total_loss / num_batches
    }


def evaluate_model(model, test_loader, device):
    """
    모델 평가

    Args:
        model: 모델
        test_loader: 테스트 데이터 로더
        device: 디바이스

    Returns:
        dict: 평가 메트릭
    """
    model.eval()
    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            user_id = batch['user_id'].to(device)
            movie_id = batch['movie_id'].to(device)
            age_group = batch['age_group'].to(device)
            gender = batch['gender'].to(device)
            occupation = batch['occupation'].to(device)
            rating = batch['rating'].to(device)

            outputs = model(user_id, movie_id, age_group, gender, occupation)
            predictions = outputs['rating']

            all_predictions.append(predictions.cpu())
            all_targets.append(rating.cpu())

    all_predictions = torch.cat(all_predictions)
    all_targets = torch.cat(all_targets)

    metrics = compute_all_metrics(all_predictions, all_targets)
    return metrics


def run_experiment(args):
    """
    확장성 실험 실행

    Args:
        args: 실험 설정

    Returns:
        dict: 실험 결과
    """
    # 시드 설정
    set_seed(args.seed)

    # 디바이스 설정
    device = get_device()
    print(f"Using device: {device}")

    # 데이터 전처리
    print("\nLoading and preprocessing data...")
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

    test_dataset = EnhancedMovieRatingDataset(
        movie_data=movie_df,
        user_data=user_df,
        rating_path=args.test_rating_path,
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

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers
    )

    print(f"Train size: {len(train_dataset)}, Test size: {len(test_dataset)}")

    results = {
        'config': vars(args),
        'models': {}
    }

    # 각 모델 타입에 대해 실험
    for model_name in args.models:
        print(f"\n{'='*70}")
        print(f"Experimenting with {model_name.upper()}")
        print(f"{'='*70}")

        model_results = {}

        # 다양한 expert 개수로 실험
        for num_experts in args.num_experts_list:
            print(f"\n--- Number of Experts: {num_experts} ---")

            # 모델 생성
            if model_name == 'dense':
                model = DenseMoE(
                    num_users=user_metadata['num_users'],
                    num_movies=len(movie_df),
                    num_age_groups=user_metadata['num_age_groups'],
                    num_occupations=user_metadata['num_occupations'],
                    embedding_dim=args.embedding_dim,
                    num_experts=num_experts,
                    expert_hidden_dim=args.expert_hidden_dim,
                    gating_hidden_dim=args.gating_hidden_dim,
                    dropout=args.dropout
                ).to(device)
            elif model_name == 'ppo':
                model = PPOMoE(
                    num_users=user_metadata['num_users'],
                    num_movies=len(movie_df),
                    num_age_groups=user_metadata['num_age_groups'],
                    num_occupations=user_metadata['num_occupations'],
                    embedding_dim=args.embedding_dim,
                    num_experts=num_experts,
                    expert_hidden_dim=args.expert_hidden_dim,
                    policy_hidden_dim=args.policy_hidden_dim,
                    dropout=args.dropout
                ).to(device)
            elif model_name == 'grpo':
                model = GRPOMoE(
                    num_users=user_metadata['num_users'],
                    num_movies=len(movie_df),
                    num_age_groups=user_metadata['num_age_groups'],
                    num_occupations=user_metadata['num_occupations'],
                    embedding_dim=args.embedding_dim,
                    num_experts=num_experts,
                    expert_hidden_dim=args.expert_hidden_dim,
                    policy_hidden_dim=args.policy_hidden_dim,
                    dropout=args.dropout
                ).to(device)
            else:
                raise ValueError(f"Unknown model: {model_name}")

            num_params = count_parameters(model)
            print(f"Model parameters: {num_params:,}")

            # Optimizer
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

            # 학습 시간 측정
            print("Training...")
            start_time = time.time()
            for epoch in range(1, args.num_epochs + 1):
                train_metrics = train_single_epoch(
                    model, train_loader, optimizer, device, model_name
                )
                print(f"Epoch {epoch}/{args.num_epochs} - Loss: {train_metrics['loss']:.4f}")
            train_time = time.time() - start_time

            # 평가
            print("Evaluating...")
            test_metrics = evaluate_model(model, test_loader, device)

            # 추론 시간 측정
            print("Measuring inference time...")
            inference_time = measure_inference_time(model, test_loader, device)

            # 메모리 측정
            print("Measuring memory usage...")
            sample_batch = next(iter(test_loader))
            memory_info = measure_memory_usage(model, sample_batch, device)

            # 결과 저장
            model_results[num_experts] = {
                'num_parameters': num_params,
                'train_time': train_time,
                'inference_time': inference_time,
                'memory': memory_info,
                'test_metrics': test_metrics
            }

            print(f"\nResults for {model_name.upper()} with {num_experts} experts:")
            print(f"  Parameters: {num_params:,}")
            print(f"  Train Time: {train_time:.2f}s")
            print(f"  Inference Time: {inference_time:.4f}s")
            print(f"  RMSE: {test_metrics['rmse']:.4f}")
            print(f"  Memory Allocated: {memory_info['allocated_mb']:.2f} MB")

        results['models'][model_name] = model_results

    # 결과 저장
    output_path = Path(args.output_dir) / f"scalability_results_{args.experiment_id}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*70}")
    print(f"Results saved to: {output_path}")
    print(f"{'='*70}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Scalability Experiment for MoE Models")

    # Data
    parser.add_argument("--data_dir", type=str, default="ml-100k", help="Data directory")
    parser.add_argument("--train_rating_path", type=str, default="ml-100k/u1.base", help="Training rating file")
    parser.add_argument("--test_rating_path", type=str, default="ml-100k/u1.test", help="Test rating file")
    parser.add_argument("--max_length", type=int, default=10, help="Max title length")

    # Experiment
    parser.add_argument("--models", nargs='+', default=['dense', 'ppo', 'grpo'],
                       help="Models to experiment with")
    parser.add_argument("--num_experts_list", nargs='+', type=int, default=[4, 8, 16, 32],
                       help="List of expert counts to test")
    parser.add_argument("--num_epochs", type=int, default=5,
                       help="Number of training epochs per configuration")
    parser.add_argument("--experiment_id", type=str, default="exp1",
                       help="Experiment identifier")

    # Model
    parser.add_argument("--embedding_dim", type=int, default=64, help="Embedding dimension")
    parser.add_argument("--expert_hidden_dim", type=int, default=256, help="Expert hidden dimension")
    parser.add_argument("--gating_hidden_dim", type=int, default=128, help="Gating hidden dimension (Dense)")
    parser.add_argument("--policy_hidden_dim", type=int, default=128, help="Policy hidden dimension (RL)")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")

    # Training
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")

    # Misc
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers")
    parser.add_argument("--output_dir", type=str, default="experiments/results", help="Output directory")

    args = parser.parse_args()
    run_experiment(args)
