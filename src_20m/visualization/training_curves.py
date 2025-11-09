"""
Training Curves Visualization
학습 곡선 시각화 (논문용)
"""

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from pathlib import Path
import argparse
import json


def setup_publication_style():
    """논문 품질 플롯 스타일"""
    plt.rcParams.update({
        'figure.figsize': (6, 4),
        'figure.dpi': 300,
        'savefig.dpi': 600,
        'savefig.bbox': 'tight',
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif'],
        'font.size': 10,
        'axes.labelsize': 11,
        'axes.titlesize': 12,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'legend.fontsize': 9,
        'lines.linewidth': 1.5,
        'lines.markersize': 4,
        'axes.grid': True,
        'grid.alpha': 0.3,
    })

    return {
        'dense': '#0173B2',
        'ppo': '#DE8F05',
        'grpo': '#029E73',
    }


def plot_training_curves_combined(training_logs, save_path, colors):
    """
    모든 모델의 학습 곡선을 하나의 figure에

    Args:
        training_logs: 학습 로그 딕셔너리 {model_name: {epochs, train_loss, val_loss, ...}}
        save_path: 저장 경로
        colors: 색상 딕셔너리
    """
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    model_labels = {'dense_moe': 'Dense-MoE', 'ppo_moe': 'PPO-MoE', 'grpo_moe': 'GRPO-MoE'}

    # (a) Training Loss
    ax = axes[0, 0]
    for model_name, logs in training_logs.items():
        if 'train_loss' in logs:
            epochs = logs.get('epochs', list(range(1, len(logs['train_loss'])+1)))
            ax.plot(epochs, logs['train_loss'],
                   marker='o', markersize=3,
                   label=model_labels.get(model_name, model_name),
                   color=colors.get(model_name, 'gray'),
                   linewidth=1.5, alpha=0.9)

    ax.set_xlabel('Epoch', fontweight='bold')
    ax.set_ylabel('Training Loss', fontweight='bold')
    ax.legend(loc='best', frameon=True, framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    ax.text(-0.15, 1.05, '(a)', transform=ax.transAxes,
           fontsize=12, fontweight='bold', va='top')

    # (b) Validation RMSE
    ax = axes[0, 1]
    for model_name, logs in training_logs.items():
        if 'val_rmse' in logs:
            epochs = logs.get('epochs', list(range(1, len(logs['val_rmse'])+1)))
            ax.plot(epochs, logs['val_rmse'],
                   marker='s', markersize=3,
                   label=model_labels.get(model_name, model_name),
                   color=colors.get(model_name, 'gray'),
                   linewidth=1.5, alpha=0.9)

    ax.set_xlabel('Epoch', fontweight='bold')
    ax.set_ylabel('Validation RMSE', fontweight='bold')
    ax.legend(loc='best', frameon=True, framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    ax.text(-0.15, 1.05, '(b)', transform=ax.transAxes,
           fontsize=12, fontweight='bold', va='top')

    # (c) Validation MAE
    ax = axes[1, 0]
    for model_name, logs in training_logs.items():
        if 'val_mae' in logs:
            epochs = logs.get('epochs', list(range(1, len(logs['val_mae'])+1)))
            ax.plot(epochs, logs['val_mae'],
                   marker='^', markersize=3,
                   label=model_labels.get(model_name, model_name),
                   color=colors.get(model_name, 'gray'),
                   linewidth=1.5, alpha=0.9)

    ax.set_xlabel('Epoch', fontweight='bold')
    ax.set_ylabel('Validation MAE', fontweight='bold')
    ax.legend(loc='best', frameon=True, framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    ax.text(-0.15, 1.05, '(c)', transform=ax.transAxes,
           fontsize=12, fontweight='bold', va='top')

    # (d) Learning Rate (if available)
    ax = axes[1, 1]
    has_lr = False
    for model_name, logs in training_logs.items():
        if 'learning_rate' in logs:
            has_lr = True
            epochs = logs.get('epochs', list(range(1, len(logs['learning_rate'])+1)))
            ax.plot(epochs, logs['learning_rate'],
                   marker='d', markersize=3,
                   label=model_labels.get(model_name, model_name),
                   color=colors.get(model_name, 'gray'),
                   linewidth=1.5, alpha=0.9)

    if has_lr:
        ax.set_xlabel('Epoch', fontweight='bold')
        ax.set_ylabel('Learning Rate', fontweight='bold')
        ax.set_yscale('log')
        ax.legend(loc='best', frameon=True, framealpha=0.9)
        ax.grid(True, alpha=0.3, linestyle='--', which='both')
        ax.set_axisbelow(True)
    else:
        ax.text(0.5, 0.5, 'Learning Rate\nData Not Available',
               ha='center', va='center', transform=ax.transAxes,
               fontsize=11, style='italic', color='gray')
        ax.axis('off')

    ax.text(-0.15, 1.05, '(d)', transform=ax.transAxes,
           fontsize=12, fontweight='bold', va='top')

    plt.tight_layout()
    plt.savefig(save_path / 'training_curves.pdf', dpi=600, bbox_inches='tight')
    plt.savefig(save_path / 'training_curves.png', dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✓ Saved: training_curves.pdf/.png")


def plot_convergence_comparison(training_logs, save_path, colors):
    """
    수렴 속도 비교 (단일 플롯)

    Args:
        training_logs: 학습 로그
        save_path: 저장 경로
        colors: 색상 딕셔너리
    """
    fig, ax = plt.subplots(figsize=(6, 4))

    model_labels = {'dense_moe': 'Dense-MoE', 'ppo_moe': 'PPO-MoE', 'grpo_moe': 'GRPO-MoE'}
    markers = {'dense_moe': 'o', 'ppo_moe': 's', 'grpo_moe': '^'}

    for model_name, logs in training_logs.items():
        if 'val_rmse' in logs:
            epochs = logs.get('epochs', list(range(1, len(logs['val_rmse'])+1)))
            ax.plot(epochs, logs['val_rmse'],
                   marker=markers.get(model_name, 'o'),
                   markersize=5, markevery=max(1, len(epochs)//10),
                   label=model_labels.get(model_name, model_name),
                   color=colors.get(model_name, 'gray'),
                   linewidth=2, alpha=0.9)

    ax.set_xlabel('Epoch', fontweight='bold')
    ax.set_ylabel('Validation RMSE', fontweight='bold')
    ax.set_title('Convergence Comparison', fontweight='bold', fontsize=12)
    ax.legend(loc='best', frameon=True, framealpha=0.95, edgecolor='black')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)

    plt.tight_layout()
    plt.savefig(save_path / 'convergence_comparison.pdf', dpi=600, bbox_inches='tight')
    plt.savefig(save_path / 'convergence_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✓ Saved: convergence_comparison.pdf/.png")


def create_dummy_training_logs():
    """
    실제 학습 로그가 없을 때 사용할 더미 데이터

    Returns:
        dict: 더미 학습 로그
    """
    epochs = list(range(1, 11))

    # Dense MoE (빠른 수렴)
    dense_train_loss = [0.0660 - i*0.0016 for i in range(10)]
    dense_val_rmse = [1.0414 - i*0.0061 + np.random.normal(0, 0.003) for i in range(10)]
    dense_val_mae = [0.8382 - i*0.0063 + np.random.normal(0, 0.003) for i in range(10)]

    # PPO-MoE (불안정)
    ppo_train_loss = [0.7164, 0.4339, 0.4655, 0.4821, 0.4502, 0.4389, 0.4277, 0.4199, 0.4166, 0.4142]
    ppo_val_rmse = [1.1192, 1.0880, 1.0659, 1.0621, 1.0584, 1.0563, 1.0549, 1.0538, 1.0531, 1.0527]
    ppo_val_mae = [0.9043, 0.8735, 0.8575, 0.8543, 0.8509, 0.8491, 0.8478, 0.8469, 0.8463, 0.8459]

    # GRPO-MoE (안정적)
    grpo_train_loss = [0.5085, 0.5155, 0.4949, 0.4852, 0.4774, 0.4708, 0.4655, 0.4611, 0.4573, 0.4541]
    grpo_val_rmse = [1.0969, 1.0501, 1.0412, 1.0391, 1.0372, 1.0358, 1.0347, 1.0339, 1.0333, 1.0328]
    grpo_val_mae = [0.8802, 0.8376, 0.8276, 0.8257, 0.8241, 0.8228, 0.8218, 0.8210, 0.8204, 0.8199]

    return {
        'dense_moe': {
            'epochs': epochs,
            'train_loss': dense_train_loss,
            'val_rmse': dense_val_rmse,
            'val_mae': dense_val_mae,
            'learning_rate': [0.001 * (0.5 ** (i//3)) for i in range(10)]
        },
        'ppo_moe': {
            'epochs': epochs,
            'train_loss': ppo_train_loss,
            'val_rmse': ppo_val_rmse,
            'val_mae': ppo_val_mae,
            'learning_rate': [0.0003] * 10
        },
        'grpo_moe': {
            'epochs': epochs,
            'train_loss': grpo_train_loss,
            'val_rmse': grpo_val_rmse,
            'val_mae': grpo_val_mae,
            'learning_rate': [0.0003] * 10
        }
    }


def main(args):
    """메인 함수"""
    colors = setup_publication_style()

    # 학습 로그 로드
    if args.training_logs_file and Path(args.training_logs_file).exists():
        with open(args.training_logs_file, 'r') as f:
            training_logs = json.load(f)
        print(f"✓ Loaded training logs from: {args.training_logs_file}")
    else:
        print("⚠ No training logs file found. Using dummy data for demonstration.")
        training_logs = create_dummy_training_logs()

    # 출력 디렉토리
    save_path = Path(args.output_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    print(f"✓ Output directory: {save_path}")

    print("\nGenerating training curve visualizations...")

    # 시각화 생성
    plot_training_curves_combined(training_logs, save_path, colors)
    plot_convergence_comparison(training_logs, save_path, colors)

    print(f"\n✅ Training curve visualizations saved!")
    print("\nGenerated files:")
    print("  - training_curves.pdf/.png")
    print("  - convergence_comparison.pdf/.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate training curve visualizations"
    )

    parser.add_argument(
        "--training_logs_file",
        type=str,
        default=None,
        help="Path to training logs JSON file (optional)"
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/publication",
        help="Output directory"
    )

    args = parser.parse_args()
    main(args)
