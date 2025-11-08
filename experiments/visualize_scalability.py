"""
확장성 실험 결과 시각화

확장성(Scalability)과 효율성(Efficiency)의 차이를 명확하게 보여주는 시각화:
1. Expert 개수 vs 학습 시간
2. Expert 개수 vs 추론 시간
3. Expert 개수 vs 메모리 사용량
4. Expert 개수 vs RMSE (성능)
5. Expert 개수 vs 파라미터 수
"""

import json
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
from pathlib import Path
import numpy as np

# 한글 폰트 설정
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

# 스타일 설정
sns.set_style("whitegrid")
sns.set_palette("husl")


def load_results(result_path):
    """
    실험 결과 로드

    Args:
        result_path: 결과 JSON 파일 경로

    Returns:
        dict: 실험 결과
    """
    with open(result_path, 'r') as f:
        return json.load(f)


def extract_data(results):
    """
    시각화를 위한 데이터 추출

    Args:
        results: 실험 결과

    Returns:
        dict: 모델별 데이터
    """
    data = {}

    for model_name, model_results in results['models'].items():
        num_experts_list = sorted([int(k) for k in model_results.keys()])

        data[model_name] = {
            'num_experts': num_experts_list,
            'num_parameters': [],
            'train_time': [],
            'inference_time': [],
            'memory_allocated': [],
            'rmse': [],
            'mae': [],
            'mse': []
        }

        for num_experts in num_experts_list:
            result = model_results[str(num_experts)]
            data[model_name]['num_parameters'].append(result['num_parameters'])
            data[model_name]['train_time'].append(result['train_time'])
            data[model_name]['inference_time'].append(result['inference_time'])
            data[model_name]['memory_allocated'].append(result['memory']['allocated_mb'])
            data[model_name]['rmse'].append(result['test_metrics']['rmse'])
            data[model_name]['mae'].append(result['test_metrics']['mae'])
            data[model_name]['mse'].append(result['test_metrics']['mse'])

    return data


def plot_scalability_comparison(data, output_dir, dpi=600):
    """
    확장성 비교 그래프 생성

    Args:
        data: 추출된 데이터
        output_dir: 출력 디렉토리
        dpi: 이미지 해상도
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # 색상 및 마커 설정
    colors = {'dense': '#E74C3C', 'ppo': '#3498DB', 'grpo': '#2ECC71'}
    markers = {'dense': 'o', 'ppo': 's', 'grpo': '^'}
    labels = {'dense': 'Dense MoE', 'ppo': 'PPO-MoE', 'grpo': 'GRPO-MoE'}

    # 1. 전체 비교 (2x3 서브플롯)
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('MoE Scalability Analysis: Dense vs RL-based', fontsize=16, fontweight='bold')

    metrics = [
        ('train_time', 'Training Time (seconds)', 'Training Time vs Number of Experts'),
        ('inference_time', 'Inference Time (seconds)', 'Inference Time vs Number of Experts'),
        ('memory_allocated', 'Memory (MB)', 'Memory Usage vs Number of Experts'),
        ('rmse', 'RMSE', 'RMSE vs Number of Experts (lower is better)'),
        ('num_parameters', 'Parameters', 'Model Parameters vs Number of Experts'),
        ('mae', 'MAE', 'MAE vs Number of Experts (lower is better)')
    ]

    for idx, (metric, ylabel, title) in enumerate(metrics):
        row = idx // 3
        col = idx % 3
        ax = axes[row, col]

        for model_name, model_data in data.items():
            ax.plot(
                model_data['num_experts'],
                model_data[metric],
                marker=markers[model_name],
                linewidth=2,
                markersize=8,
                label=labels[model_name],
                color=colors[model_name]
            )

        ax.set_xlabel('Number of Experts', fontsize=11, fontweight='bold')
        ax.set_ylabel(ylabel, fontsize=11, fontweight='bold')
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_xticks(data[list(data.keys())[0]]['num_experts'])

    plt.tight_layout()
    plt.savefig(output_path / 'scalability_overview.png', dpi=dpi, bbox_inches='tight')
    plt.savefig(output_path / 'scalability_overview.pdf', dpi=dpi, bbox_inches='tight')
    print(f"Saved: {output_path / 'scalability_overview.png'}")

    # 2. 학습 시간 확장성 (상세)
    fig, ax = plt.subplots(figsize=(10, 6))

    for model_name, model_data in data.items():
        ax.plot(
            model_data['num_experts'],
            model_data['train_time'],
            marker=markers[model_name],
            linewidth=3,
            markersize=10,
            label=labels[model_name],
            color=colors[model_name]
        )

    ax.set_xlabel('Number of Experts', fontsize=13, fontweight='bold')
    ax.set_ylabel('Training Time (seconds)', fontsize=13, fontweight='bold')
    ax.set_title('Training Time Scalability', fontsize=15, fontweight='bold')
    ax.legend(fontsize=12, loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.set_xticks(data[list(data.keys())[0]]['num_experts'])

    # Dense MoE의 선형 증가 강조
    if 'dense' in data:
        dense_data = data['dense']
        # 최소제곱법으로 선형 근사
        z = np.polyfit(dense_data['num_experts'], dense_data['train_time'], 1)
        p = np.poly1d(z)
        ax.plot(
            dense_data['num_experts'],
            p(dense_data['num_experts']),
            '--',
            color=colors['dense'],
            alpha=0.5,
            linewidth=2,
            label=f'Dense MoE (Linear Fit: y={z[0]:.2f}x+{z[1]:.2f})'
        )
        ax.legend(fontsize=11, loc='upper left')

    plt.tight_layout()
    plt.savefig(output_path / 'train_time_scalability.png', dpi=dpi, bbox_inches='tight')
    plt.savefig(output_path / 'train_time_scalability.pdf', dpi=dpi, bbox_inches='tight')
    print(f"Saved: {output_path / 'train_time_scalability.png'}")

    # 3. 추론 시간 확장성 (상세)
    fig, ax = plt.subplots(figsize=(10, 6))

    for model_name, model_data in data.items():
        ax.plot(
            model_data['num_experts'],
            model_data['inference_time'],
            marker=markers[model_name],
            linewidth=3,
            markersize=10,
            label=labels[model_name],
            color=colors[model_name]
        )

    ax.set_xlabel('Number of Experts', fontsize=13, fontweight='bold')
    ax.set_ylabel('Inference Time (seconds)', fontsize=13, fontweight='bold')
    ax.set_title('Inference Time Scalability', fontsize=15, fontweight='bold')
    ax.legend(fontsize=12, loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.set_xticks(data[list(data.keys())[0]]['num_experts'])

    plt.tight_layout()
    plt.savefig(output_path / 'inference_time_scalability.png', dpi=dpi, bbox_inches='tight')
    plt.savefig(output_path / 'inference_time_scalability.pdf', dpi=dpi, bbox_inches='tight')
    print(f"Saved: {output_path / 'inference_time_scalability.png'}")

    # 4. 효율성 비교 (시간 대비 성능)
    fig, ax = plt.subplots(figsize=(10, 6))

    for model_name, model_data in data.items():
        # Efficiency = 1 / (RMSE * inference_time)
        efficiency = [1 / (rmse * time) for rmse, time in
                     zip(model_data['rmse'], model_data['inference_time'])]

        ax.plot(
            model_data['num_experts'],
            efficiency,
            marker=markers[model_name],
            linewidth=3,
            markersize=10,
            label=labels[model_name],
            color=colors[model_name]
        )

    ax.set_xlabel('Number of Experts', fontsize=13, fontweight='bold')
    ax.set_ylabel('Efficiency (1 / (RMSE × Time))', fontsize=13, fontweight='bold')
    ax.set_title('Model Efficiency: Higher is Better', fontsize=15, fontweight='bold')
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(data[list(data.keys())[0]]['num_experts'])

    plt.tight_layout()
    plt.savefig(output_path / 'efficiency_comparison.png', dpi=dpi, bbox_inches='tight')
    plt.savefig(output_path / 'efficiency_comparison.pdf', dpi=dpi, bbox_inches='tight')
    print(f"Saved: {output_path / 'efficiency_comparison.png'}")

    # 5. 정규화된 비교 (모든 메트릭을 0-1 범위로)
    fig, ax = plt.subplots(figsize=(12, 6))

    metrics_to_normalize = ['train_time', 'inference_time', 'memory_allocated', 'rmse']
    x_positions = np.arange(len(data[list(data.keys())[0]]['num_experts']))
    bar_width = 0.25

    for model_idx, (model_name, model_data) in enumerate(data.items()):
        normalized_scores = []

        for num_experts in model_data['num_experts']:
            # 각 메트릭을 정규화 (낮을수록 좋음)
            scores = []
            for metric in metrics_to_normalize:
                all_values = [data[m][metric] for m in data.keys()]
                all_values_flat = [item for sublist in all_values for item in sublist]
                max_val = max(all_values_flat)
                min_val = min(all_values_flat)

                idx = model_data['num_experts'].index(num_experts)
                value = model_data[metric][idx]

                # 정규화 (낮을수록 좋으므로 반전)
                normalized = 1 - (value - min_val) / (max_val - min_val) if max_val != min_val else 0.5
                scores.append(normalized)

            # 평균 점수
            normalized_scores.append(np.mean(scores))

        offset = (model_idx - 1) * bar_width
        ax.bar(
            x_positions + offset,
            normalized_scores,
            bar_width,
            label=labels[model_name],
            color=colors[model_name],
            alpha=0.8
        )

    ax.set_xlabel('Number of Experts', fontsize=13, fontweight='bold')
    ax.set_ylabel('Normalized Score (higher is better)', fontsize=13, fontweight='bold')
    ax.set_title('Overall Performance Score (normalized)', fontsize=15, fontweight='bold')
    ax.set_xticks(x_positions)
    ax.set_xticklabels(data[list(data.keys())[0]]['num_experts'])
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim([0, 1])

    plt.tight_layout()
    plt.savefig(output_path / 'normalized_comparison.png', dpi=dpi, bbox_inches='tight')
    plt.savefig(output_path / 'normalized_comparison.pdf', dpi=dpi, bbox_inches='tight')
    print(f"Saved: {output_path / 'normalized_comparison.png'}")

    plt.close('all')


def generate_comparison_table(data, output_dir):
    """
    비교 테이블 생성 (Markdown 형식)

    Args:
        data: 추출된 데이터
        output_dir: 출력 디렉토리
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    table_path = output_path / 'scalability_comparison_table.md'

    with open(table_path, 'w') as f:
        f.write("# MoE Scalability Comparison\n\n")

        # 각 expert 개수별 비교
        for num_experts in data[list(data.keys())[0]]['num_experts']:
            f.write(f"\n## {num_experts} Experts\n\n")
            f.write("| Model | Parameters | Train Time (s) | Inference Time (s) | Memory (MB) | RMSE | MAE |\n")
            f.write("|-------|-----------|---------------|-------------------|-------------|------|-----|\n")

            for model_name, model_data in data.items():
                idx = model_data['num_experts'].index(num_experts)
                f.write(f"| {model_name.upper()} | "
                       f"{model_data['num_parameters'][idx]:,} | "
                       f"{model_data['train_time'][idx]:.2f} | "
                       f"{model_data['inference_time'][idx]:.4f} | "
                       f"{model_data['memory_allocated'][idx]:.2f} | "
                       f"{model_data['rmse'][idx]:.4f} | "
                       f"{model_data['mae'][idx]:.4f} |\n")

        # 확장성 분석
        f.write("\n## Scalability Analysis\n\n")
        f.write("### Training Time Growth Rate\n\n")
        f.write("| Model | 4→8 | 8→16 | 16→32 | Average Growth |\n")
        f.write("|-------|-----|------|-------|----------------|\n")

        for model_name, model_data in data.items():
            growths = []
            for i in range(len(model_data['num_experts']) - 1):
                growth = (model_data['train_time'][i+1] - model_data['train_time'][i]) / model_data['train_time'][i] * 100
                growths.append(growth)

            avg_growth = np.mean(growths)
            growth_str = " | ".join([f"{g:.1f}%" for g in growths])
            f.write(f"| {model_name.upper()} | {growth_str} | {avg_growth:.1f}% |\n")

    print(f"Saved: {table_path}")


def main(args):
    """
    메인 함수

    Args:
        args: 커맨드라인 인자
    """
    print(f"Loading results from: {args.result_path}")
    results = load_results(args.result_path)

    print("Extracting data...")
    data = extract_data(results)

    print("Generating visualizations...")
    plot_scalability_comparison(data, args.output_dir, args.dpi)

    print("Generating comparison table...")
    generate_comparison_table(data, args.output_dir)

    print(f"\n{'='*70}")
    print("Visualization complete!")
    print(f"Output directory: {args.output_dir}")
    print(f"{'='*70}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize Scalability Experiment Results")

    parser.add_argument("--result_path", type=str, required=True,
                       help="Path to experiment results JSON file")
    parser.add_argument("--output_dir", type=str, default="experiments/visualizations",
                       help="Output directory for plots")
    parser.add_argument("--dpi", type=int, default=600,
                       help="Image resolution (DPI)")

    args = parser.parse_args()
    main(args)
