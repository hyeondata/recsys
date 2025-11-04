"""
논문용 종합 지표 비교 시각화
- 확장성 (Scalability): 파라미터 수
- 효율성 (Efficiency): 학습/추론 시간, 메모리
- 성능 (Performance): MAE, RMSE
"""

import json
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from pathlib import Path
import argparse


def setup_publication_style():
    """논문 품질 플롯 스타일 설정"""
    plt.rcParams.update({
        'figure.figsize': (6, 4),
        'figure.dpi': 300,
        'savefig.dpi': 600,
        'savefig.format': 'pdf',
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.1,
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif'],
        'font.size': 10,
        'axes.labelsize': 11,
        'axes.titlesize': 12,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'legend.fontsize': 9,
        'lines.linewidth': 1.5,
        'lines.markersize': 6,
        'axes.grid': True,
        'grid.alpha': 0.3,
        'grid.linewidth': 0.5,
        'axes.linewidth': 0.8,
        'legend.framealpha': 0.8,
    })

    return {
        'dense': '#0173B2',
        'ppo': '#DE8F05',
        'grpo': '#029E73',
        'ppo_std': '#CC79A7',
        'grpo_std': '#F0E442',
        'colors': ['#0173B2', '#DE8F05', '#029E73', '#CC79A7', '#F0E442']
    }


def plot_comprehensive_comparison(results, save_path, colors):
    """
    종합 비교 시각화: 4가지 관점
    - (a) Performance: MAE, RMSE
    - (b) Scalability: Parameters
    - (c) Training Efficiency: Time per epoch
    - (d) Inference Efficiency: Inference time
    """
    fig = plt.figure(figsize=(14, 8))
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

    models = list(results.keys())
    model_labels = []
    model_colors = []

    for m in models:
        if 'dense' in m:
            model_labels.append('Dense')
            model_colors.append(colors['dense'])
        elif 'ppo' in m and 'standard' in m:
            model_labels.append('PPO-Std')
            model_colors.append(colors['ppo_std'])
        elif 'ppo' in m:
            model_labels.append('PPO')
            model_colors.append(colors['ppo'])
        elif 'grpo' in m and 'standard' in m:
            model_labels.append('GRPO-Std')
            model_colors.append(colors['grpo_std'])
        elif 'grpo' in m:
            model_labels.append('GRPO')
            model_colors.append(colors['grpo'])

    # (a) Performance: MAE & RMSE
    ax1 = fig.add_subplot(gs[0, 0])
    x = np.arange(len(models))
    width = 0.35

    mae_vals = [results[m]['mae'] for m in models]
    rmse_vals = [results[m]['rmse'] for m in models]

    ax1.bar(x - width/2, mae_vals, width, label='MAE',
            color='#4ECDC4', edgecolor='black', linewidth=0.8)
    ax1.bar(x + width/2, rmse_vals, width, label='RMSE',
            color='#FF6B6B', edgecolor='black', linewidth=0.8)

    # 값 표시
    for i, (mae, rmse) in enumerate(zip(mae_vals, rmse_vals)):
        ax1.text(i - width/2, mae + 0.02, f'{mae:.3f}',
                ha='center', va='bottom', fontsize=8)
        ax1.text(i + width/2, rmse + 0.02, f'{rmse:.3f}',
                ha='center', va='bottom', fontsize=8)

    ax1.set_ylabel('Error', fontweight='bold')
    ax1.set_xlabel('Model', fontweight='bold')
    ax1.set_title('(a) Performance Metrics', fontweight='bold', loc='left')
    ax1.set_xticks(x)
    ax1.set_xticklabels(model_labels, rotation=15, ha='right')
    ax1.legend(loc='upper right', frameon=True)
    ax1.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.5)
    ax1.set_axisbelow(True)

    # (b) Scalability: Parameters
    ax2 = fig.add_subplot(gs[0, 1])
    params = [results[m].get('parameters', 750000) / 1e6 for m in models]

    bars = ax2.bar(x, params, width=0.6, color=model_colors,
                   edgecolor='black', linewidth=0.8)

    for i, (bar, val) in enumerate(zip(bars, params)):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.2f}M',
                ha='center', va='bottom', fontsize=8, fontweight='bold')

    ax2.set_ylabel('Parameters (Million)', fontweight='bold')
    ax2.set_xlabel('Model', fontweight='bold')
    ax2.set_title('(b) Model Scalability', fontweight='bold', loc='left')
    ax2.set_xticks(x)
    ax2.set_xticklabels(model_labels, rotation=15, ha='right')
    ax2.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.5)
    ax2.set_axisbelow(True)

    # (c) Training Efficiency
    ax3 = fig.add_subplot(gs[1, 0])
    train_time = [results[m].get('train_time_per_epoch', 60) for m in models]

    bars = ax3.bar(x, train_time, width=0.6, color=model_colors,
                   edgecolor='black', linewidth=0.8)

    for i, (bar, val) in enumerate(zip(bars, train_time)):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.1f}s',
                ha='center', va='bottom', fontsize=8, fontweight='bold')

    ax3.set_ylabel('Time (seconds)', fontweight='bold')
    ax3.set_xlabel('Model', fontweight='bold')
    ax3.set_title('(c) Training Efficiency (per epoch)', fontweight='bold', loc='left')
    ax3.set_xticks(x)
    ax3.set_xticklabels(model_labels, rotation=15, ha='right')
    ax3.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.5)
    ax3.set_axisbelow(True)

    # (d) Inference Efficiency
    ax4 = fig.add_subplot(gs[1, 1])
    inference_time = [results[m].get('inference_time_ms', 1.0) for m in models]

    bars = ax4.bar(x, inference_time, width=0.6, color=model_colors,
                   edgecolor='black', linewidth=0.8)

    for i, (bar, val) in enumerate(zip(bars, inference_time)):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.2f}',
                ha='center', va='bottom', fontsize=8, fontweight='bold')

    ax4.set_ylabel('Time (ms per sample)', fontweight='bold')
    ax4.set_xlabel('Model', fontweight='bold')
    ax4.set_title('(d) Inference Efficiency', fontweight='bold', loc='left')
    ax4.set_xticks(x)
    ax4.set_xticklabels(model_labels, rotation=15, ha='right')
    ax4.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.5)
    ax4.set_axisbelow(True)

    plt.tight_layout()
    plt.savefig(save_path / 'comprehensive_comparison.pdf', dpi=600, bbox_inches='tight')
    plt.savefig(save_path / 'comprehensive_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✓ Saved: comprehensive_comparison.pdf/.png")


def plot_performance_vs_efficiency(results, save_path, colors):
    """
    Performance vs Efficiency 산점도
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    models = list(results.keys())
    model_labels = []
    model_colors = []

    for m in models:
        if 'dense' in m:
            model_labels.append('Dense')
            model_colors.append(colors['dense'])
        elif 'ppo' in m and 'standard' in m:
            model_labels.append('PPO-Std')
            model_colors.append(colors['ppo_std'])
        elif 'ppo' in m:
            model_labels.append('PPO')
            model_colors.append(colors['ppo'])
        elif 'grpo' in m and 'standard' in m:
            model_labels.append('GRPO-Std')
            model_colors.append(colors['grpo_std'])
        elif 'grpo' in m:
            model_labels.append('GRPO')
            model_colors.append(colors['grpo'])

    # (a) RMSE vs Training Time
    ax1 = axes[0]
    rmse_vals = [results[m]['rmse'] for m in models]
    train_time = [results[m].get('train_time_per_epoch', 60) for m in models]

    for i, (rmse, time, label, color) in enumerate(zip(rmse_vals, train_time, model_labels, model_colors)):
        ax1.scatter(time, rmse, s=200, c=[color], alpha=0.7,
                   edgecolors='black', linewidth=1.5, marker='o')
        ax1.annotate(label, (time, rmse),
                    textcoords="offset points", xytext=(0,10),
                    ha='center', fontsize=9, fontweight='bold')

    ax1.set_xlabel('Training Time per Epoch (s)', fontweight='bold')
    ax1.set_ylabel('RMSE', fontweight='bold')
    ax1.set_title('(a) Performance vs Training Efficiency', fontweight='bold', loc='left')
    ax1.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    ax1.set_axisbelow(True)

    # (b) RMSE vs Inference Time
    ax2 = axes[1]
    inference_time = [results[m].get('inference_time_ms', 1.0) for m in models]

    for i, (rmse, time, label, color) in enumerate(zip(rmse_vals, inference_time, model_labels, model_colors)):
        ax2.scatter(time, rmse, s=200, c=[color], alpha=0.7,
                   edgecolors='black', linewidth=1.5, marker='o')
        ax2.annotate(label, (time, rmse),
                    textcoords="offset points", xytext=(0,10),
                    ha='center', fontsize=9, fontweight='bold')

    ax2.set_xlabel('Inference Time (ms per sample)', fontweight='bold')
    ax2.set_ylabel('RMSE', fontweight='bold')
    ax2.set_title('(b) Performance vs Inference Efficiency', fontweight='bold', loc='left')
    ax2.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    ax2.set_axisbelow(True)

    plt.tight_layout()
    plt.savefig(save_path / 'performance_vs_efficiency.pdf', dpi=600, bbox_inches='tight')
    plt.savefig(save_path / 'performance_vs_efficiency.png', dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✓ Saved: performance_vs_efficiency.pdf/.png")


def generate_comprehensive_latex_table(results, save_path):
    """
    종합 지표 LaTeX 테이블 생성
    """
    models = list(results.keys())

    latex_code = r"""\begin{table*}[t]
\centering
\caption{Comprehensive comparison of MoE models on MovieLens 100k dataset}
\label{tab:comprehensive}
\begin{tabular}{lcccccc}
\toprule
\textbf{Model} & \textbf{Parameters} & \textbf{MAE} & \textbf{RMSE} & \textbf{Train Time} & \textbf{Inference} \\
 & (M) & & & (s/epoch) & (ms/sample) \\
\midrule
"""

    model_names = []
    for m in models:
        if 'dense' in m:
            model_names.append('Dense-MoE')
        elif 'ppo' in m and 'standard' in m:
            model_names.append('PPO-MoE (Standard)')
        elif 'ppo' in m:
            model_names.append('PPO-MoE (Batch)')
        elif 'grpo' in m and 'standard' in m:
            model_names.append('GRPO-MoE (Standard)')
        elif 'grpo' in m:
            model_names.append('GRPO-MoE (Batch)')

    # Find best values
    mae_vals = [results[m]['mae'] for m in models]
    rmse_vals = [results[m]['rmse'] for m in models]
    train_times = [results[m].get('train_time_per_epoch', 60) for m in models]
    infer_times = [results[m].get('inference_time_ms', 1.0) for m in models]

    best_mae = min(mae_vals)
    best_rmse = min(rmse_vals)
    best_train = min(train_times)
    best_infer = min(infer_times)

    for i, m in enumerate(models):
        params = results[m].get('parameters', 750000) / 1e6
        mae = results[m]['mae']
        rmse = results[m]['rmse']
        train_time = results[m].get('train_time_per_epoch', 60)
        infer_time = results[m].get('inference_time_ms', 1.0)

        row = f"{model_names[i]} & {params:.2f} & "

        # MAE
        if mae == best_mae:
            row += f"\\textbf{{{mae:.4f}}} & "
        else:
            row += f"{mae:.4f} & "

        # RMSE
        if rmse == best_rmse:
            row += f"\\textbf{{{rmse:.4f}}} & "
        else:
            row += f"{rmse:.4f} & "

        # Train time
        if train_time == best_train:
            row += f"\\textbf{{{train_time:.1f}}} & "
        else:
            row += f"{train_time:.1f} & "

        # Inference time
        if infer_time == best_infer:
            row += f"\\textbf{{{infer_time:.3f}}}"
        else:
            row += f"{infer_time:.3f}"

        row += " \\\\\n"
        latex_code += row

    latex_code += r"""\bottomrule
\end{tabular}
\end{table*}
"""

    with open(save_path / 'comprehensive_table.tex', 'w') as f:
        f.write(latex_code)

    print(f"✓ Saved: comprehensive_table.tex")


def main(args):
    """메인 함수"""
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

    print("\nGenerating comprehensive comparison visualizations...")

    # 시각화 생성
    plot_comprehensive_comparison(results, save_path, colors)
    plot_performance_vs_efficiency(results, save_path, colors)
    generate_comprehensive_latex_table(results, save_path)

    print(f"\n✅ All comprehensive visualizations saved!")
    print("\nGenerated files:")
    print("  - comprehensive_comparison.pdf/.png (4-panel comparison)")
    print("  - performance_vs_efficiency.pdf/.png (scatter plots)")
    print("  - comprehensive_table.tex (LaTeX table)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate comprehensive comparison visualizations for paper"
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
        help="Output directory for visualizations"
    )

    args = parser.parse_args()
    main(args)
