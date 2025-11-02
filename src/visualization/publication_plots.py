"""
Publication-Quality Visualization for MoE Models
논문 제출용 고품질 시각화 생성
"""

import json
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from pathlib import Path
import argparse

# Publication-quality 설정
def setup_publication_style():
    """논문 품질 플롯 스타일 설정"""

    # 폰트 설정 (LaTeX 스타일)
    plt.rcParams.update({
        # Figure
        'figure.figsize': (6, 4),
        'figure.dpi': 300,
        'savefig.dpi': 600,
        'savefig.format': 'pdf',
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.1,

        # Fonts
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif'],
        'font.size': 10,
        'axes.labelsize': 11,
        'axes.titlesize': 12,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'legend.fontsize': 9,

        # Lines and markers
        'lines.linewidth': 1.5,
        'lines.markersize': 6,
        'patch.linewidth': 0.5,

        # Grid
        'axes.grid': True,
        'grid.alpha': 0.3,
        'grid.linewidth': 0.5,

        # Spines
        'axes.linewidth': 0.8,
        'axes.edgecolor': 'black',

        # Legend
        'legend.framealpha': 0.8,
        'legend.edgecolor': '0.8',
        'legend.fancybox': True,
    })

    # Colorblind-friendly palette
    return {
        'dense': '#0173B2',  # Blue
        'ppo': '#DE8F05',    # Orange
        'grpo': '#029E73',   # Green
        'colors': ['#0173B2', '#DE8F05', '#029E73']
    }


def plot_performance_bars(results, save_path, colors):
    """
    성능 비교 Bar Chart (개별 파일)

    Args:
        results: 평가 결과
        save_path: 저장 경로
        colors: 색상 딕셔너리
    """
    metrics = ['mse', 'rmse', 'mae']
    metric_names = ['MSE', 'RMSE', 'MAE']

    models = list(results.keys())
    model_labels = ['Dense-MoE', 'PPO-MoE', 'GRPO-MoE']

    for metric, metric_name in zip(metrics, metric_names):
        fig, ax = plt.subplots(figsize=(5, 4))

        values = [results[m][metric] for m in models]
        x_pos = np.arange(len(models))

        bars = ax.bar(x_pos, values, width=0.6,
                     color=colors['colors'],
                     edgecolor='black', linewidth=0.8)

        # 값 표시
        for i, (bar, val) in enumerate(zip(bars, values)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{val:.4f}',
                   ha='center', va='bottom', fontsize=9, fontweight='bold')

        ax.set_ylabel(metric_name, fontweight='bold')
        ax.set_xlabel('Model', fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(model_labels, rotation=0)
        ax.set_ylim(0, max(values) * 1.15)

        # 그리드 스타일
        ax.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.5)
        ax.set_axisbelow(True)

        plt.tight_layout()

        # PDF와 PNG 모두 저장
        plt.savefig(save_path / f'{metric}_comparison.pdf', dpi=600, bbox_inches='tight')
        plt.savefig(save_path / f'{metric}_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()

        print(f"✓ Saved: {metric}_comparison.pdf/.png")


def plot_combined_performance(results, save_path, colors):
    """
    통합 성능 비교 (논문용 단일 figure)

    Args:
        results: 평가 결과
        save_path: 저장 경로
        colors: 색상 딕셔너리
    """
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))

    models = list(results.keys())
    model_labels = ['Dense-MoE', 'PPO-MoE', 'GRPO-MoE']
    metrics = ['mse', 'rmse', 'mae']
    metric_names = ['MSE', 'RMSE', 'MAE']

    for idx, (ax, metric, metric_name) in enumerate(zip(axes, metrics, metric_names)):
        values = [results[m][metric] for m in models]
        x_pos = np.arange(len(models))

        bars = ax.bar(x_pos, values, width=0.6,
                     color=colors['colors'],
                     edgecolor='black', linewidth=0.8)

        # 값 표시
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{val:.3f}',
                   ha='center', va='bottom', fontsize=8, fontweight='bold')

        ax.set_ylabel(metric_name, fontweight='bold')
        ax.set_xlabel('Model', fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(model_labels, rotation=15, ha='right')
        ax.set_ylim(0, max(values) * 1.18)
        ax.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.5)
        ax.set_axisbelow(True)

        # 서브플롯 레이블
        ax.text(-0.15, 1.05, f'({chr(97+idx)})', transform=ax.transAxes,
               fontsize=12, fontweight='bold', va='top')

    plt.tight_layout()
    plt.savefig(save_path / 'performance_combined.pdf', dpi=600, bbox_inches='tight')
    plt.savefig(save_path / 'performance_combined.png', dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✓ Saved: performance_combined.pdf/.png")


def plot_expert_distribution_publication(results, save_path, colors):
    """
    Expert 분포 (논문용)

    Args:
        results: 평가 결과
        save_path: 저장 경로
        colors: 색상 딕셔너리
    """
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))

    num_experts = 8
    expert_labels = [f'E{i}' for i in range(num_experts)]

    # Dense MoE - Gate 확률
    if 'dense_moe' in results:
        ax = axes[0]
        gate_probs = results['dense_moe']['avg_gate_probs']

        bars = ax.bar(expert_labels, gate_probs, width=0.6,
                     color=colors['dense'], alpha=0.8,
                     edgecolor='black', linewidth=0.8)

        ax.axhline(y=1/num_experts, color='red', linestyle='--',
                  linewidth=1.5, label='Uniform', alpha=0.7)

        ax.set_ylabel('Gate Probability', fontweight='bold')
        ax.set_xlabel('Expert ID', fontweight='bold')
        ax.set_title('Dense-MoE', fontweight='bold')
        ax.set_ylim(0, max(gate_probs) * 1.15)
        ax.legend(loc='upper right', frameon=True)
        ax.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.5)
        ax.set_axisbelow(True)

        # 서브플롯 레이블
        ax.text(-0.15, 1.05, '(a)', transform=ax.transAxes,
               fontsize=12, fontweight='bold', va='top')

    # PPO-MoE - Expert 선택
    if 'ppo_moe' in results and 'expert_distribution' in results['ppo_moe']:
        ax = axes[1]
        expert_dist = results['ppo_moe']['expert_distribution']

        counts = [expert_dist.get(i, 0) for i in range(num_experts)]
        total = sum(counts)
        percentages = [c / total * 100 if total > 0 else 0 for c in counts]

        bars = ax.bar(expert_labels, percentages, width=0.6,
                     color=colors['ppo'], alpha=0.8,
                     edgecolor='black', linewidth=0.8)

        ax.axhline(y=100/num_experts, color='red', linestyle='--',
                  linewidth=1.5, label='Uniform', alpha=0.7)

        ax.set_ylabel('Selection (%)', fontweight='bold')
        ax.set_xlabel('Expert ID', fontweight='bold')
        ax.set_title('PPO-MoE', fontweight='bold')
        ax.set_ylim(0, max(percentages) * 1.15 if max(percentages) > 0 else 50)
        ax.legend(loc='upper right', frameon=True)
        ax.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.5)
        ax.set_axisbelow(True)

        ax.text(-0.15, 1.05, '(b)', transform=ax.transAxes,
               fontsize=12, fontweight='bold', va='top')

    # GRPO-MoE - Expert 선택
    if 'grpo_moe' in results and 'expert_distribution' in results['grpo_moe']:
        ax = axes[2]
        expert_dist = results['grpo_moe']['expert_distribution']

        counts = [expert_dist.get(i, 0) for i in range(num_experts)]
        total = sum(counts)
        percentages = [c / total * 100 if total > 0 else 0 for c in counts]

        bars = ax.bar(expert_labels, percentages, width=0.6,
                     color=colors['grpo'], alpha=0.8,
                     edgecolor='black', linewidth=0.8)

        ax.axhline(y=100/num_experts, color='red', linestyle='--',
                  linewidth=1.5, label='Uniform', alpha=0.7)

        ax.set_ylabel('Selection (%)', fontweight='bold')
        ax.set_xlabel('Expert ID', fontweight='bold')
        ax.set_title('GRPO-MoE', fontweight='bold')
        ax.set_ylim(0, max(percentages) * 1.15 if max(percentages) > 0 else 50)
        ax.legend(loc='upper right', frameon=True)
        ax.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.5)
        ax.set_axisbelow(True)

        ax.text(-0.15, 1.05, '(c)', transform=ax.transAxes,
               fontsize=12, fontweight='bold', va='top')

    plt.tight_layout()
    plt.savefig(save_path / 'expert_distribution.pdf', dpi=600, bbox_inches='tight')
    plt.savefig(save_path / 'expert_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✓ Saved: expert_distribution.pdf/.png")


def plot_comparison_table(results, save_path):
    """
    LaTeX 형식 비교 테이블 생성

    Args:
        results: 평가 결과
        save_path: 저장 경로
    """
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.axis('tight')
    ax.axis('off')

    models = list(results.keys())
    model_labels = ['Dense-MoE', 'PPO-MoE', 'GRPO-MoE']

    # 테이블 데이터
    table_data = [
        ['Metric', *model_labels],
        ['MSE', *[f"{results[m]['mse']:.4f}" for m in models]],
        ['RMSE', *[f"{results[m]['rmse']:.4f}" for m in models]],
        ['MAE', *[f"{results[m]['mae']:.4f}" for m in models]],
        ['Experts', 'All (8)', 'One (1)', 'One (1)'],
        ['Training', 'Supervised', 'PPO', 'GRPO'],
    ]

    # 테이블 생성
    table = ax.table(cellText=table_data, cellLoc='center', loc='center',
                    colWidths=[0.25] * 4)

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)

    # 스타일링
    for i in range(4):
        cell = table[(0, i)]
        cell.set_facecolor('#E0E0E0')
        cell.set_text_props(weight='bold')
        cell.set_edgecolor('black')
        cell.set_linewidth(1.5)

    for i in range(1, 6):
        cell = table[(i, 0)]
        cell.set_facecolor('#F5F5F5')
        cell.set_text_props(weight='bold')
        cell.set_edgecolor('black')
        cell.set_linewidth(1)

    # 최고 성능 하이라이트
    for row_idx in [1, 2, 3]:
        values = [float(table_data[row_idx][i+1]) for i in range(3)]
        best_idx = values.index(min(values))
        cell = table[(row_idx, best_idx + 1)]
        cell.set_facecolor('#C6EFCE')
        cell.set_text_props(weight='bold')
        cell.set_edgecolor('black')
        cell.set_linewidth(1)

    for key, cell in table.get_celld().items():
        cell.set_edgecolor('black')
        cell.set_linewidth(0.5)

    plt.tight_layout()
    plt.savefig(save_path / 'comparison_table.pdf', dpi=600, bbox_inches='tight')
    plt.savefig(save_path / 'comparison_table.png', dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✓ Saved: comparison_table.pdf/.png")


def generate_latex_table(results, save_path):
    """
    LaTeX 테이블 코드 생성

    Args:
        results: 평가 결과
        save_path: 저장 경로
    """
    models = list(results.keys())
    model_labels = ['Dense-MoE', 'PPO-MoE', 'GRPO-MoE']

    latex_code = r"""\begin{table}[t]
\centering
\caption{Performance comparison of MoE models on MovieLens 100k dataset}
\label{tab:performance}
\begin{tabular}{lccc}
\toprule
\textbf{Metric} & \textbf{Dense-MoE} & \textbf{PPO-MoE} & \textbf{GRPO-MoE} \\
\midrule
"""

    # MSE
    mse_vals = [results[m]['mse'] for m in models]
    best_mse = min(mse_vals)
    mse_row = "MSE & "
    for val in mse_vals:
        if val == best_mse:
            mse_row += f"\\textbf{{{val:.4f}}} & "
        else:
            mse_row += f"{val:.4f} & "
    mse_row = mse_row.rstrip(" & ") + " \\\\\n"
    latex_code += mse_row

    # RMSE
    rmse_vals = [results[m]['rmse'] for m in models]
    best_rmse = min(rmse_vals)
    rmse_row = "RMSE & "
    for val in rmse_vals:
        if val == best_rmse:
            rmse_row += f"\\textbf{{{val:.4f}}} & "
        else:
            rmse_row += f"{val:.4f} & "
    rmse_row = rmse_row.rstrip(" & ") + " \\\\\n"
    latex_code += rmse_row

    # MAE
    mae_vals = [results[m]['mae'] for m in models]
    best_mae = min(mae_vals)
    mae_row = "MAE & "
    for val in mae_vals:
        if val == best_mae:
            mae_row += f"\\textbf{{{val:.4f}}} & "
        else:
            mae_row += f"{val:.4f} & "
    mae_row = mae_row.rstrip(" & ") + " \\\\\n"
    latex_code += mae_row

    latex_code += r"""\midrule
Experts Used & All (8) & One (1) & One (1) \\
Training Method & Supervised & PPO & GRPO \\
\bottomrule
\end{tabular}
\end{table}
"""

    # 저장
    with open(save_path / 'comparison_table.tex', 'w') as f:
        f.write(latex_code)

    print(f"✓ Saved: comparison_table.tex")


def main(args):
    """메인 함수"""
    # 스타일 설정
    colors = setup_publication_style()

    # 결과 로드
    results_path = Path(args.results_file)
    if not results_path.exists():
        print(f"❌ Error: Results file not found: {results_path}")
        return

    with open(results_path, 'r') as f:
        results = json.load(f)

    print(f"✓ Loaded results from: {results_path}")
    print(f"  Models: {', '.join(results.keys())}")

    # 출력 디렉토리
    save_path = Path(args.output_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    print(f"✓ Output directory: {save_path}")

    print("\nGenerating publication-quality visualizations...")

    # 시각화 생성
    plot_performance_bars(results, save_path, colors)
    plot_combined_performance(results, save_path, colors)
    plot_expert_distribution_publication(results, save_path, colors)
    plot_comparison_table(results, save_path)
    generate_latex_table(results, save_path)

    print(f"\n✅ All publication-quality visualizations saved!")
    print("\nGenerated files:")
    print("  PDF (600 DPI):")
    print("    - mse_comparison.pdf")
    print("    - rmse_comparison.pdf")
    print("    - mae_comparison.pdf")
    print("    - performance_combined.pdf")
    print("    - expert_distribution.pdf")
    print("    - comparison_table.pdf")
    print("  PNG (300 DPI): Same filenames with .png")
    print("  LaTeX: comparison_table.tex")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate publication-quality visualizations for MoE models"
    )

    parser.add_argument(
        "--results_file",
        type=str,
        default="results/all_models_evaluation.json",
        help="Path to evaluation results JSON"
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/publication",
        help="Output directory for publication figures"
    )

    args = parser.parse_args()
    main(args)
