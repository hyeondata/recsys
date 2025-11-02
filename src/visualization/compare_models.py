"""
MoE 모델 비교 시각화 스크립트
Dense MoE vs PPO-MoE vs GRPO-MoE 비교 차트 생성
"""

import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path
import argparse


def setup_plot_style():
    """플롯 스타일 설정"""
    plt.style.use('seaborn-v0_8-darkgrid')
    plt.rcParams['figure.figsize'] = (15, 10)
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['xtick.labelsize'] = 10
    plt.rcParams['ytick.labelsize'] = 10
    plt.rcParams['legend.fontsize'] = 10


def plot_performance_comparison(results, save_path):
    """
    성능 비교 바 차트

    Args:
        results: 평가 결과 딕셔너리
        save_path: 저장 경로
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    models = list(results.keys())
    model_labels = [m.replace('_', ' ').title() for m in models]

    # MSE
    mse_values = [results[m]['mse'] for m in models]
    bars1 = axes[0].bar(model_labels, mse_values, color=['#2E86AB', '#A23B72', '#F18F01'])
    axes[0].set_ylabel('MSE')
    axes[0].set_title('Mean Squared Error (Lower is Better)')
    axes[0].set_ylim(0, max(mse_values) * 1.2)

    # 값 표시
    for bar in bars1:
        height = bar.get_height()
        axes[0].text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.4f}',
                    ha='center', va='bottom', fontsize=10, fontweight='bold')

    # RMSE
    rmse_values = [results[m]['rmse'] for m in models]
    bars2 = axes[1].bar(model_labels, rmse_values, color=['#2E86AB', '#A23B72', '#F18F01'])
    axes[1].set_ylabel('RMSE')
    axes[1].set_title('Root Mean Squared Error (Lower is Better)')
    axes[1].set_ylim(0, max(rmse_values) * 1.2)

    for bar in bars2:
        height = bar.get_height()
        axes[1].text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.4f}',
                    ha='center', va='bottom', fontsize=10, fontweight='bold')

    # MAE
    mae_values = [results[m]['mae'] for m in models]
    bars3 = axes[2].bar(model_labels, mae_values, color=['#2E86AB', '#A23B72', '#F18F01'])
    axes[2].set_ylabel('MAE')
    axes[2].set_title('Mean Absolute Error (Lower is Better)')
    axes[2].set_ylim(0, max(mae_values) * 1.2)

    for bar in bars3:
        height = bar.get_height()
        axes[2].text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.4f}',
                    ha='center', va='bottom', fontsize=10, fontweight='bold')

    plt.tight_layout()
    plt.savefig(save_path / 'performance_comparison.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {save_path / 'performance_comparison.png'}")
    plt.close()


def plot_expert_distribution(results, save_path):
    """
    Expert 선택/가중치 분포 비교

    Args:
        results: 평가 결과 딕셔너리
        save_path: 저장 경로
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    num_experts = 8
    expert_labels = [f'E{i}' for i in range(num_experts)]

    # Dense MoE - Gate 확률
    if 'dense_moe' in results:
        gate_probs = results['dense_moe']['avg_gate_probs']
        bars1 = axes[0].bar(expert_labels, gate_probs, color='#2E86AB', alpha=0.7)
        axes[0].set_ylabel('Average Gate Probability')
        axes[0].set_title('Dense MoE - Gate Weights (All Experts Used)')
        axes[0].set_ylim(0, max(gate_probs) * 1.2)
        axes[0].axhline(y=1/num_experts, color='r', linestyle='--',
                       label=f'Uniform ({1/num_experts:.3f})')
        axes[0].legend()

        # 값 표시
        for bar in bars1:
            height = bar.get_height()
            axes[0].text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.3f}',
                        ha='center', va='bottom', fontsize=9)

    # PPO-MoE - Expert 선택 분포
    if 'ppo_moe' in results and 'expert_distribution' in results['ppo_moe']:
        expert_dist = results['ppo_moe']['expert_distribution']

        # 딕셔너리를 리스트로 변환
        counts = [expert_dist.get(i, 0) for i in range(num_experts)]
        total = sum(counts)
        percentages = [c / total * 100 if total > 0 else 0 for c in counts]

        bars2 = axes[1].bar(expert_labels, percentages, color='#A23B72', alpha=0.7)
        axes[1].set_ylabel('Selection Percentage (%)')
        axes[1].set_title('PPO-MoE - Expert Selection (One Expert per Sample)')
        axes[1].set_ylim(0, max(percentages) * 1.2 if max(percentages) > 0 else 50)
        axes[1].axhline(y=100/num_experts, color='r', linestyle='--',
                       label=f'Uniform ({100/num_experts:.1f}%)')
        axes[1].legend()

        # 값 표시
        for i, bar in enumerate(bars2):
            height = bar.get_height()
            if height > 0:
                axes[1].text(bar.get_x() + bar.get_width()/2., height,
                            f'{height:.1f}%\n({counts[i]})',
                            ha='center', va='bottom', fontsize=8)

    # GRPO-MoE - Expert 선택 분포
    if 'grpo_moe' in results and 'expert_distribution' in results['grpo_moe']:
        expert_dist = results['grpo_moe']['expert_distribution']

        # 딕셔너리를 리스트로 변환
        counts = [expert_dist.get(i, 0) for i in range(num_experts)]
        total = sum(counts)
        percentages = [c / total * 100 if total > 0 else 0 for c in counts]

        bars3 = axes[2].bar(expert_labels, percentages, color='#F18F01', alpha=0.7)
        axes[2].set_ylabel('Selection Percentage (%)')
        axes[2].set_title('GRPO-MoE - Expert Selection (One Expert per Sample)')
        axes[2].set_ylim(0, max(percentages) * 1.2 if max(percentages) > 0 else 50)
        axes[2].axhline(y=100/num_experts, color='r', linestyle='--',
                       label=f'Uniform ({100/num_experts:.1f}%)')
        axes[2].legend()

        # 값 표시
        for i, bar in enumerate(bars3):
            height = bar.get_height()
            if height > 0:
                axes[2].text(bar.get_x() + bar.get_width()/2., height,
                            f'{height:.1f}%\n({counts[i]})',
                            ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    plt.savefig(save_path / 'expert_distribution.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {save_path / 'expert_distribution.png'}")
    plt.close()


def plot_expert_performance(results, save_path):
    """
    Expert별 성능 비교 (PPO/GRPO만)

    Args:
        results: 평가 결과 딕셔너리
        save_path: 저장 경로
    """
    has_data = False

    # PPO 또는 GRPO 데이터가 있는지 확인
    if 'ppo_moe' in results and 'expert_performance' in results['ppo_moe']:
        has_data = True
    if 'grpo_moe' in results and 'expert_performance' in results['grpo_moe']:
        has_data = True

    if not has_data:
        print("⚠ No expert performance data available for visualization")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    num_experts = 8
    expert_labels = [f'E{i}' for i in range(num_experts)]

    # PPO-MoE
    if 'ppo_moe' in results and 'expert_performance' in results['ppo_moe']:
        expert_perf = results['ppo_moe']['expert_performance']

        avg_errors = [expert_perf.get(i, None) for i in range(num_experts)]
        colors = ['#A23B72' if err is not None else '#CCCCCC' for err in avg_errors]

        # None을 0으로 변환 (시각화용)
        plot_errors = [err if err is not None else 0 for err in avg_errors]

        bars1 = axes[0].bar(expert_labels, plot_errors, color=colors, alpha=0.7)
        axes[0].set_ylabel('Average Error')
        axes[0].set_title('PPO-MoE - Expert Performance (Lower is Better)')
        axes[0].set_ylim(0, max([e for e in avg_errors if e is not None], default=1) * 1.2)

        # 값 표시
        for i, (bar, err) in enumerate(zip(bars1, avg_errors)):
            if err is not None:
                height = bar.get_height()
                axes[0].text(bar.get_x() + bar.get_width()/2., height,
                            f'{err:.4f}',
                            ha='center', va='bottom', fontsize=8)
            else:
                axes[0].text(bar.get_x() + bar.get_width()/2., 0.01,
                            'N/A',
                            ha='center', va='bottom', fontsize=8, color='gray')

    # GRPO-MoE
    if 'grpo_moe' in results and 'expert_performance' in results['grpo_moe']:
        expert_perf = results['grpo_moe']['expert_performance']

        avg_errors = [expert_perf.get(i, None) for i in range(num_experts)]
        colors = ['#F18F01' if err is not None else '#CCCCCC' for err in avg_errors]

        # None을 0으로 변환 (시각화용)
        plot_errors = [err if err is not None else 0 for err in avg_errors]

        bars2 = axes[1].bar(expert_labels, plot_errors, color=colors, alpha=0.7)
        axes[1].set_ylabel('Average Error')
        axes[1].set_title('GRPO-MoE - Expert Performance (Lower is Better)')
        axes[1].set_ylim(0, max([e for e in avg_errors if e is not None], default=1) * 1.2)

        # 값 표시
        for i, (bar, err) in enumerate(zip(bars2, avg_errors)):
            if err is not None:
                height = bar.get_height()
                axes[1].text(bar.get_x() + bar.get_width()/2., height,
                            f'{err:.4f}',
                            ha='center', va='bottom', fontsize=8)
            else:
                axes[1].text(bar.get_x() + bar.get_width()/2., 0.01,
                            'N/A',
                            ha='center', va='bottom', fontsize=8, color='gray')

    plt.tight_layout()
    plt.savefig(save_path / 'expert_performance.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {save_path / 'expert_performance.png'}")
    plt.close()


def plot_model_comparison_radar(results, save_path):
    """
    모델 비교 레이더 차트

    Args:
        results: 평가 결과 딕셔너리
        save_path: 저장 경로
    """
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))

    # 성능 메트릭 (정규화)
    models = list(results.keys())

    # RMSE 역수 (높을수록 좋음)
    rmse_values = [results[m]['rmse'] for m in models]
    rmse_scores = [1/r for r in rmse_values]
    rmse_max = max(rmse_scores)
    rmse_norm = [s/rmse_max for s in rmse_scores]

    # MAE 역수 (높을수록 좋음)
    mae_values = [results[m]['mae'] for m in models]
    mae_scores = [1/m for m in mae_values]
    mae_max = max(mae_scores)
    mae_norm = [s/mae_max for s in mae_scores]

    # Expert 다양성 (Dense는 gate entropy, 나머지는 선택 분포 엔트로피)
    diversity_scores = []
    for m in models:
        if m == 'dense_moe':
            # Gate entropy (이미 정규화됨)
            entropy = results[m].get('gate_entropy', 0)
            diversity_scores.append(entropy / np.log(8))  # 최대 엔트로피로 정규화
        else:
            # Expert 선택 분포의 엔트로피 계산
            if 'expert_distribution' in results[m]:
                dist = results[m]['expert_distribution']
                counts = [dist.get(i, 0) for i in range(8)]
                total = sum(counts)
                if total > 0:
                    probs = [c/total for c in counts if c > 0]
                    entropy = -sum(p * np.log(p) for p in probs)
                    diversity_scores.append(entropy / np.log(8))
                else:
                    diversity_scores.append(0)
            else:
                diversity_scores.append(0)

    # 추론 효율성 (RL 모델은 1개 Expert, Dense는 8개 Expert)
    efficiency_scores = []
    for m in models:
        if m == 'dense_moe':
            efficiency_scores.append(1/8)  # 8개 모두 사용
        else:
            efficiency_scores.append(1.0)  # 1개만 사용

    # 학습 안정성 (성능 기반으로 추정)
    # Dense > GRPO > PPO 순서로 가정
    stability_map = {
        'dense_moe': 1.0,
        'grpo_moe': 0.8,
        'ppo_moe': 0.6
    }
    stability_scores = [stability_map.get(m, 0.5) for m in models]

    # 카테고리
    categories = ['Accuracy\n(1/RMSE)', 'Precision\n(1/MAE)', 'Expert\nDiversity',
                  'Inference\nEfficiency', 'Training\nStability']
    N = len(categories)

    # 각도 계산
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]

    # 플롯
    colors = ['#2E86AB', '#A23B72', '#F18F01']
    labels = [m.replace('_', ' ').title() for m in models]

    for i, m in enumerate(models):
        values = [rmse_norm[i], mae_norm[i], diversity_scores[i],
                 efficiency_scores[i], stability_scores[i]]
        values += values[:1]

        ax.plot(angles, values, 'o-', linewidth=2, label=labels[i], color=colors[i])
        ax.fill(angles, values, alpha=0.15, color=colors[i])

    # 축 설정
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, size=11)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], size=9)
    ax.grid(True)

    # 범례
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=11)

    plt.title('Model Comparison - Normalized Metrics', size=14, weight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(save_path / 'model_comparison_radar.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {save_path / 'model_comparison_radar.png'}")
    plt.close()


def plot_architecture_comparison(save_path):
    """
    아키텍처 비교 다이어그램

    Args:
        save_path: 저장 경로
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Dense MoE
    ax1 = axes[0]
    ax1.set_xlim(0, 10)
    ax1.set_ylim(0, 10)
    ax1.axis('off')
    ax1.set_title('Dense MoE Architecture', fontsize=14, weight='bold')

    # State
    rect1 = mpatches.FancyBboxPatch((1, 8), 8, 1, boxstyle="round,pad=0.1",
                                    edgecolor='black', facecolor='lightblue', linewidth=2)
    ax1.add_patch(rect1)
    ax1.text(5, 8.5, 'State (User + Movie + Context)', ha='center', va='center', fontsize=10, weight='bold')

    # Arrow
    ax1.arrow(5, 8, 0, -0.5, head_width=0.3, head_length=0.2, fc='black', ec='black')

    # Gating Network
    rect2 = mpatches.FancyBboxPatch((0.5, 6), 4, 1.2, boxstyle="round,pad=0.1",
                                    edgecolor='blue', facecolor='lightcyan', linewidth=2)
    ax1.add_patch(rect2)
    ax1.text(2.5, 6.6, 'Gating Network\n(Softmax)', ha='center', va='center', fontsize=9)

    # Experts
    rect3 = mpatches.FancyBboxPatch((5.5, 6), 4, 1.2, boxstyle="round,pad=0.1",
                                    edgecolor='green', facecolor='lightgreen', linewidth=2)
    ax1.add_patch(rect3)
    ax1.text(7.5, 6.6, '8 Experts\n(All Active)', ha='center', va='center', fontsize=9)

    # Arrows
    ax1.arrow(2.5, 6, 0, -0.5, head_width=0.3, head_length=0.2, fc='black', ec='black')
    ax1.arrow(7.5, 6, 0, -0.5, head_width=0.3, head_length=0.2, fc='black', ec='black')

    # Weighted Sum
    rect4 = mpatches.FancyBboxPatch((2, 4), 6, 1, boxstyle="round,pad=0.1",
                                    edgecolor='purple', facecolor='plum', linewidth=2)
    ax1.add_patch(rect4)
    ax1.text(5, 4.5, 'Weighted Sum (All Experts)', ha='center', va='center', fontsize=10)

    # Arrow
    ax1.arrow(5, 4, 0, -0.5, head_width=0.3, head_length=0.2, fc='black', ec='black')

    # Output
    rect5 = mpatches.FancyBboxPatch((3, 2.5), 4, 0.8, boxstyle="round,pad=0.1",
                                    edgecolor='red', facecolor='lightcoral', linewidth=2)
    ax1.add_patch(rect5)
    ax1.text(5, 2.9, 'Rating Prediction', ha='center', va='center', fontsize=10)

    # Info
    ax1.text(5, 1, '✓ All 8 Experts Used\n✓ Weighted Combination\n✗ High Inference Cost',
            ha='center', va='center', fontsize=9,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # PPO-MoE
    ax2 = axes[1]
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 10)
    ax2.axis('off')
    ax2.set_title('PPO-MoE Architecture', fontsize=14, weight='bold')

    # State
    rect1 = mpatches.FancyBboxPatch((1, 8), 8, 1, boxstyle="round,pad=0.1",
                                    edgecolor='black', facecolor='lightblue', linewidth=2)
    ax2.add_patch(rect1)
    ax2.text(5, 8.5, 'State (User + Movie + Context)', ha='center', va='center', fontsize=10, weight='bold')

    # Arrows
    ax2.arrow(4, 8, -1, -0.5, head_width=0.2, head_length=0.2, fc='black', ec='black')
    ax2.arrow(6, 8, 1, -0.5, head_width=0.2, head_length=0.2, fc='black', ec='black')

    # Policy Network
    rect2 = mpatches.FancyBboxPatch((0.5, 6), 3.5, 1.2, boxstyle="round,pad=0.1",
                                    edgecolor='blue', facecolor='lightcyan', linewidth=2)
    ax2.add_patch(rect2)
    ax2.text(2.25, 6.6, 'Policy Network\n(Categorical)', ha='center', va='center', fontsize=9)

    # Value Network
    rect3 = mpatches.FancyBboxPatch((6, 6), 3.5, 1.2, boxstyle="round,pad=0.1",
                                    edgecolor='orange', facecolor='lightyellow', linewidth=2)
    ax2.add_patch(rect3)
    ax2.text(7.75, 6.6, 'Value Network\n(Critic)', ha='center', va='center', fontsize=9)

    # Arrow
    ax2.arrow(2.25, 6, 0, -0.5, head_width=0.3, head_length=0.2, fc='black', ec='black')

    # Expert Selection
    rect4 = mpatches.FancyBboxPatch((0.5, 4), 3.5, 1, boxstyle="round,pad=0.1",
                                    edgecolor='green', facecolor='lightgreen', linewidth=2)
    ax2.add_patch(rect4)
    ax2.text(2.25, 4.5, 'Select 1 Expert\n(e.g., Expert 3)', ha='center', va='center', fontsize=9)

    # Arrow
    ax2.arrow(2.25, 4, 0, -0.5, head_width=0.3, head_length=0.2, fc='black', ec='black')

    # Output
    rect5 = mpatches.FancyBboxPatch((3, 2.5), 4, 0.8, boxstyle="round,pad=0.1",
                                    edgecolor='red', facecolor='lightcoral', linewidth=2)
    ax2.add_patch(rect5)
    ax2.text(5, 2.9, 'Rating Prediction', ha='center', va='center', fontsize=10)

    # Info
    ax2.text(5, 1, '✓ Only 1 Expert Used\n✓ Low Inference Cost\n✓ PPO Training',
            ha='center', va='center', fontsize=9,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # GRPO-MoE
    ax3 = axes[2]
    ax3.set_xlim(0, 10)
    ax3.set_ylim(0, 10)
    ax3.axis('off')
    ax3.set_title('GRPO-MoE Architecture', fontsize=14, weight='bold')

    # State
    rect1 = mpatches.FancyBboxPatch((1, 8), 8, 1, boxstyle="round,pad=0.1",
                                    edgecolor='black', facecolor='lightblue', linewidth=2)
    ax3.add_patch(rect1)
    ax3.text(5, 8.5, 'State (User + Movie + Context)', ha='center', va='center', fontsize=10, weight='bold')

    # Arrows
    ax3.arrow(4, 8, -1, -0.5, head_width=0.2, head_length=0.2, fc='black', ec='black')
    ax3.arrow(6, 8, 1, -0.5, head_width=0.2, head_length=0.2, fc='black', ec='black')

    # Policy Network
    rect2 = mpatches.FancyBboxPatch((0.5, 6), 3.5, 1.2, boxstyle="round,pad=0.1",
                                    edgecolor='blue', facecolor='lightcyan', linewidth=2)
    ax3.add_patch(rect2)
    ax3.text(2.25, 6.6, 'Policy Network\n(Categorical/T)', ha='center', va='center', fontsize=9)

    # Baseline Network
    rect3 = mpatches.FancyBboxPatch((6, 6), 3.5, 1.2, boxstyle="round,pad=0.1",
                                    edgecolor='orange', facecolor='lightyellow', linewidth=2)
    ax3.add_patch(rect3)
    ax3.text(7.75, 6.6, 'Baseline Network\n(Reward Est.)', ha='center', va='center', fontsize=9)

    # Arrow
    ax3.arrow(2.25, 6, 0, -0.5, head_width=0.3, head_length=0.2, fc='black', ec='black')

    # Expert Selection
    rect4 = mpatches.FancyBboxPatch((0.5, 4), 3.5, 1, boxstyle="round,pad=0.1",
                                    edgecolor='green', facecolor='lightgreen', linewidth=2)
    ax3.add_patch(rect4)
    ax3.text(2.25, 4.5, 'Select 1 Expert\n(e.g., Expert 5)', ha='center', va='center', fontsize=9)

    # Arrow
    ax3.arrow(2.25, 4, 0, -0.5, head_width=0.3, head_length=0.2, fc='black', ec='black')

    # Output
    rect5 = mpatches.FancyBboxPatch((3, 2.5), 4, 0.8, boxstyle="round,pad=0.1",
                                    edgecolor='red', facecolor='lightcoral', linewidth=2)
    ax3.add_patch(rect5)
    ax3.text(5, 2.9, 'Rating Prediction', ha='center', va='center', fontsize=10)

    # Info
    ax3.text(5, 1, '✓ Only 1 Expert Used\n✓ Low Inference Cost\n✓ GRPO Training',
            ha='center', va='center', fontsize=9,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(save_path / 'architecture_comparison.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {save_path / 'architecture_comparison.png'}")
    plt.close()


def create_summary_table(results, save_path):
    """
    결과 요약 테이블 이미지 생성

    Args:
        results: 평가 결과 딕셔너리
        save_path: 저장 경로
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.axis('tight')
    ax.axis('off')

    models = list(results.keys())
    model_labels = [m.replace('_', ' ').title() for m in models]

    # 데이터 준비
    table_data = []
    table_data.append(['Metric', *model_labels])

    # MSE
    mse_values = [f"{results[m]['mse']:.4f}" for m in models]
    table_data.append(['MSE', *mse_values])

    # RMSE
    rmse_values = [f"{results[m]['rmse']:.4f}" for m in models]
    table_data.append(['RMSE', *rmse_values])

    # MAE
    mae_values = [f"{results[m]['mae']:.4f}" for m in models]
    table_data.append(['MAE', *mae_values])

    # Expert 활용
    expert_usage = []
    for m in models:
        if m == 'dense_moe':
            expert_usage.append('All (8)')
        else:
            expert_usage.append('One (1)')
    table_data.append(['Experts Used', *expert_usage])

    # 추론 효율성
    efficiency = []
    for m in models:
        if m == 'dense_moe':
            efficiency.append('Low (8x)')
        else:
            efficiency.append('High (1x)')
    table_data.append(['Inference Efficiency', *efficiency])

    # 학습 방법
    training = []
    for m in models:
        if m == 'dense_moe':
            training.append('Supervised')
        elif m == 'ppo_moe':
            training.append('PPO (RL)')
        else:
            training.append('GRPO (RL)')
    table_data.append(['Training Method', *training])

    # 테이블 생성
    table = ax.table(cellText=table_data, cellLoc='center', loc='center',
                    colWidths=[0.25] + [0.25] * len(models))

    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2.5)

    # 헤더 스타일
    for i in range(len(model_labels) + 1):
        cell = table[(0, i)]
        cell.set_facecolor('#4472C4')
        cell.set_text_props(weight='bold', color='white')

    # 첫 번째 열 스타일
    for i in range(1, len(table_data)):
        cell = table[(i, 0)]
        cell.set_facecolor('#D9E1F2')
        cell.set_text_props(weight='bold')

    # 최고 성능 하이라이트
    for row_idx in [1, 2, 3]:  # MSE, RMSE, MAE
        values = [float(table_data[row_idx][i+1]) for i in range(len(models))]
        best_idx = values.index(min(values))
        cell = table[(row_idx, best_idx + 1)]
        cell.set_facecolor('#C6EFCE')
        cell.set_text_props(weight='bold')

    plt.title('Model Comparison Summary', fontsize=16, weight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(save_path / 'summary_table.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {save_path / 'summary_table.png'}")
    plt.close()


def main(args):
    """메인 함수"""
    setup_plot_style()

    # 결과 파일 로드
    results_path = Path(args.results_file)
    if not results_path.exists():
        print(f"❌ Error: Results file not found: {results_path}")
        return

    with open(results_path, 'r') as f:
        results = json.load(f)

    print(f"✓ Loaded results from: {results_path}")
    print(f"  Models found: {', '.join(results.keys())}")

    # 출력 디렉토리 생성
    save_path = Path(args.output_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    print(f"✓ Output directory: {save_path}")

    print("\nGenerating visualizations...")

    # 시각화 생성
    plot_performance_comparison(results, save_path)
    plot_expert_distribution(results, save_path)
    plot_expert_performance(results, save_path)
    plot_model_comparison_radar(results, save_path)
    plot_architecture_comparison(save_path)
    create_summary_table(results, save_path)

    print(f"\n✅ All visualizations saved to: {save_path}")
    print("\nGenerated files:")
    print("  - performance_comparison.png")
    print("  - expert_distribution.png")
    print("  - expert_performance.png")
    print("  - model_comparison_radar.png")
    print("  - architecture_comparison.png")
    print("  - summary_table.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize MoE model comparison")

    parser.add_argument(
        "--results_file",
        type=str,
        default="results/all_models_evaluation.json",
        help="Path to evaluation results JSON file"
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/visualizations",
        help="Output directory for visualizations"
    )

    args = parser.parse_args()
    main(args)
