"""
Quick scalability experiment with a single model.

This is a simplified version for testing with fewer epochs.
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import subprocess


def main():
    parser = argparse.ArgumentParser(description="Quick Scalability Experiment")

    parser.add_argument('--model_type', type=str, default='dense',
                        choices=['dense', 'ppo', 'grpo'],
                        help='Model type (default: dense)')
    parser.add_argument('--expert_counts', type=str, default='8,16,32',
                        help='Comma-separated expert counts (default: 8,16,32)')
    parser.add_argument('--epochs', type=int, default=5,
                        help='Epochs per experiment (default: 5)')
    parser.add_argument('--batch_size', type=int, default=1024,
                        help='Batch size (default: 1024)')

    args = parser.parse_args()

    expert_counts = ' '.join(args.expert_counts.split(','))

    cmd = [
        'uv', 'run', 'python3', 'src_10m/experiments/run_scalability_experiment.py',
        '--model_type', args.model_type,
        '--expert_counts', *expert_counts.split(),
        '--epochs', str(args.epochs),
        '--batch_size', str(args.batch_size),
        '--data_dir', 'ml-10M100K',
        '--output_dir', 'results_10m/scalability',
        '--baseline_experts', '16'
    ]

    print(f"Running: {' '.join(cmd)}")
    print()

    subprocess.run(cmd)

    print("\n" + "="*60)
    print("Experiment complete!")
    print("="*60)
    print("\nTo visualize results, run:")
    print("  uv run python3 src_10m/experiments/visualize_scalability.py")


if __name__ == "__main__":
    main()
