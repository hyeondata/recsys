"""
Visualization tools for scalability experiments.

Creates publication-quality plots showing:
- Performance vs. number of experts
- Training time vs. number of experts
- Memory usage vs. number of experts
- Expert utilization (entropy, Gini)
- Efficiency metrics
"""
import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List


# Publication settings
plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 11,
    'figure.titlesize': 18,
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
    'text.usetex': False,  # Set to True if LaTeX is installed
    'figure.dpi': 150,
    'savefig.dpi': 600,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1
})

# Colorblind-friendly palette
COLORS = {
    'dense': '#0173B2',  # Blue
    'ppo': '#DE8F05',    # Orange
    'grpo': '#029E73',   # Green
}


def load_results(file_path: str) -> Dict:
    """Load results from JSON file."""
    with open(file_path, 'r') as f:
        return json.load(f)


def plot_performance_scaling(results: List[Dict], output_file: str):
    """
    Plot performance (RMSE) vs. number of experts.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # Group by model type
    model_results = {}
    for result in results:
        model_type = result['experiment']['model_type']
        if model_type not in model_results:
            model_results[model_type] = {'experts': [], 'rmse': []}

        for res in result['individual_results']:
            model_results[model_type]['experts'].append(res['num_experts'])
            model_results[model_type]['rmse'].append(res['best_rmse'])

    # Plot each model
    for model_type, data in model_results.items():
        experts = np.array(data['experts'])
        rmse = np.array(data['rmse'])

        # Sort by experts
        sort_idx = np.argsort(experts)
        experts = experts[sort_idx]
        rmse = rmse[sort_idx]

        ax.plot(experts, rmse, 'o-', color=COLORS.get(model_type, 'gray'),
                linewidth=2, markersize=8, label=model_type.upper())

    ax.set_xlabel('Number of Experts')
    ax.set_ylabel('RMSE (Lower is Better)')
    ax.set_title('Performance Scaling with Expert Count')
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_xscale('log', base=2)

    plt.savefig(output_file, format='pdf', bbox_inches='tight')
    plt.savefig(output_file.replace('.pdf', '.png'), format='png', bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.close()


def plot_time_scaling(results: List[Dict], output_file: str):
    """
    Plot training time vs. number of experts.
    Shows scaling efficiency.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    for result in results:
        model_type = result['experiment']['model_type']
        experts = []
        times = []

        for res in result['individual_results']:
            experts.append(res['num_experts'])
            times.append(res['efficiency'].get('avg_epoch_time', 0))

        experts = np.array(experts)
        times = np.array(times)
        sort_idx = np.argsort(experts)
        experts = experts[sort_idx]
        times = times[sort_idx]

        color = COLORS.get(model_type, 'gray')

        # Absolute time
        ax1.plot(experts, times, 'o-', color=color, linewidth=2,
                 markersize=8, label=model_type.upper())

        # Efficiency (normalized by expert count)
        # Ideal: constant throughput regardless of experts
        baseline_time = times[0]
        baseline_experts = experts[0]
        expected_time = baseline_time * (experts / baseline_experts)
        efficiency = expected_time / times  # >1 is better

        ax2.plot(experts, efficiency, 'o-', color=color, linewidth=2,
                 markersize=8, label=model_type.upper())

    # Left plot: Absolute time
    ax1.set_xlabel('Number of Experts')
    ax1.set_ylabel('Average Time per Epoch (seconds)')
    ax1.set_title('Training Time Scaling')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax1.set_xscale('log', base=2)

    # Right plot: Efficiency
    ax2.set_xlabel('Number of Experts')
    ax2.set_ylabel('Time Efficiency (Higher is Better)')
    ax2.set_title('Computational Efficiency')
    ax2.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, label='Linear scaling')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    ax2.set_xscale('log', base=2)

    plt.tight_layout()
    plt.savefig(output_file, format='pdf', bbox_inches='tight')
    plt.savefig(output_file.replace('.pdf', '.png'), format='png', bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.close()


def plot_memory_scaling(results: List[Dict], output_file: str):
    """
    Plot memory usage vs. number of experts.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    for result in results:
        model_type = result['experiment']['model_type']
        experts = []
        gpu_memory = []
        parameters = []

        for res in result['individual_results']:
            experts.append(res['num_experts'])
            gpu_memory.append(res['efficiency'].get('peak_gpu_memory', 0))
            parameters.append(res['parameters']['total'])

        experts = np.array(experts)
        gpu_memory = np.array(gpu_memory)
        parameters = np.array(parameters)

        sort_idx = np.argsort(experts)
        experts = experts[sort_idx]
        gpu_memory = gpu_memory[sort_idx]
        parameters = parameters[sort_idx]

        color = COLORS.get(model_type, 'gray')

        # GPU memory
        ax1.plot(experts, gpu_memory, 'o-', color=color, linewidth=2,
                 markersize=8, label=model_type.upper())

        # Parameters (in millions)
        ax2.plot(experts, parameters / 1e6, 'o-', color=color, linewidth=2,
                 markersize=8, label=model_type.upper())

    # Left: GPU memory
    ax1.set_xlabel('Number of Experts')
    ax1.set_ylabel('Peak GPU Memory (GB)')
    ax1.set_title('GPU Memory Usage')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax1.set_xscale('log', base=2)

    # Right: Parameters
    ax2.set_xlabel('Number of Experts')
    ax2.set_ylabel('Model Parameters (Millions)')
    ax2.set_title('Model Size')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    ax2.set_xscale('log', base=2)

    plt.tight_layout()
    plt.savefig(output_file, format='pdf', bbox_inches='tight')
    plt.savefig(output_file.replace('.pdf', '.png'), format='png', bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.close()


def plot_expert_utilization(results: List[Dict], output_file: str):
    """
    Plot expert utilization metrics (entropy and Gini coefficient).
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    for result in results:
        model_type = result['experiment']['model_type']
        experts = []
        entropy = []
        gini = []

        for res in result['individual_results']:
            experts.append(res['num_experts'])
            entropy.append(res['efficiency'].get('normalized_entropy', 0))
            gini.append(res['efficiency'].get('gini', 0))

        experts = np.array(experts)
        entropy = np.array(entropy)
        gini = np.array(gini)

        sort_idx = np.argsort(experts)
        experts = experts[sort_idx]
        entropy = entropy[sort_idx]
        gini = gini[sort_idx]

        color = COLORS.get(model_type, 'gray')

        # Entropy (higher = more uniform)
        ax1.plot(experts, entropy, 'o-', color=color, linewidth=2,
                 markersize=8, label=model_type.upper())

        # Gini coefficient (lower = more uniform)
        ax2.plot(experts, gini, 'o-', color=color, linewidth=2,
                 markersize=8, label=model_type.upper())

    # Left: Entropy
    ax1.set_xlabel('Number of Experts')
    ax1.set_ylabel('Normalized Entropy (Higher = More Uniform)')
    ax1.set_title('Expert Usage Uniformity (Entropy)')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax1.set_xscale('log', base=2)
    ax1.set_ylim([0, 1])

    # Right: Gini
    ax2.set_xlabel('Number of Experts')
    ax2.set_ylabel('Gini Coefficient (Lower = More Uniform)')
    ax2.set_title('Expert Usage Concentration (Gini)')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    ax2.set_xscale('log', base=2)

    plt.tight_layout()
    plt.savefig(output_file, format='pdf', bbox_inches='tight')
    plt.savefig(output_file.replace('.pdf', '.png'), format='png', bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.close()


def plot_efficiency_comparison(results: List[Dict], output_file: str):
    """
    Comprehensive efficiency comparison plot.
    Shows trade-offs between performance, time, and memory.
    """
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)

    ax1 = fig.add_subplot(gs[0, 0])  # RMSE vs Experts
    ax2 = fig.add_subplot(gs[0, 1])  # Time vs Experts
    ax3 = fig.add_subplot(gs[0, 2])  # Memory vs Experts
    ax4 = fig.add_subplot(gs[1, 0])  # Throughput vs Experts
    ax5 = fig.add_subplot(gs[1, 1])  # RMSE vs Time (efficiency)
    ax6 = fig.add_subplot(gs[1, 2])  # RMSE vs Memory (efficiency)

    for result in results:
        model_type = result['experiment']['model_type']
        experts = []
        rmse = []
        time = []
        memory = []
        throughput = []

        for res in result['individual_results']:
            experts.append(res['num_experts'])
            rmse.append(res['best_rmse'])
            time.append(res['efficiency'].get('avg_epoch_time', 0))
            memory.append(res['efficiency'].get('peak_gpu_memory', 0))
            throughput.append(res['efficiency'].get('avg_throughput', 0))

        experts = np.array(experts)
        rmse = np.array(rmse)
        time = np.array(time)
        memory = np.array(memory)
        throughput = np.array(throughput)

        color = COLORS.get(model_type, 'gray')
        label = model_type.upper()

        # 1. RMSE vs Experts
        ax1.plot(experts, rmse, 'o-', color=color, linewidth=2, markersize=8, label=label)

        # 2. Time vs Experts
        ax2.plot(experts, time, 'o-', color=color, linewidth=2, markersize=8, label=label)

        # 3. Memory vs Experts
        ax3.plot(experts, memory, 'o-', color=color, linewidth=2, markersize=8, label=label)

        # 4. Throughput vs Experts
        ax4.plot(experts, throughput, 'o-', color=color, linewidth=2, markersize=8, label=label)

        # 5. RMSE vs Time (efficiency)
        ax5.scatter(time, rmse, c=[color]*len(time), s=experts*3, alpha=0.6, label=label)

        # 6. RMSE vs Memory (efficiency)
        ax6.scatter(memory, rmse, c=[color]*len(memory), s=experts*3, alpha=0.6, label=label)

    # Configure axes
    ax1.set_xlabel('Number of Experts')
    ax1.set_ylabel('RMSE')
    ax1.set_title('Performance')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax1.set_xscale('log', base=2)

    ax2.set_xlabel('Number of Experts')
    ax2.set_ylabel('Time/Epoch (s)')
    ax2.set_title('Training Time')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    ax2.set_xscale('log', base=2)

    ax3.set_xlabel('Number of Experts')
    ax3.set_ylabel('GPU Memory (GB)')
    ax3.set_title('Memory Usage')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    ax3.set_xscale('log', base=2)

    ax4.set_xlabel('Number of Experts')
    ax4.set_ylabel('Throughput (samples/s)')
    ax4.set_title('Throughput')
    ax4.grid(True, alpha=0.3)
    ax4.legend()
    ax4.set_xscale('log', base=2)

    ax5.set_xlabel('Time/Epoch (s)')
    ax5.set_ylabel('RMSE')
    ax5.set_title('Performance vs. Time\n(Size = Expert count)')
    ax5.grid(True, alpha=0.3)
    ax5.legend()

    ax6.set_xlabel('GPU Memory (GB)')
    ax6.set_ylabel('RMSE')
    ax6.set_title('Performance vs. Memory\n(Size = Expert count)')
    ax6.grid(True, alpha=0.3)
    ax6.legend()

    fig.suptitle('Comprehensive Efficiency Analysis', fontsize=18, fontweight='bold')

    plt.savefig(output_file, format='pdf', bbox_inches='tight')
    plt.savefig(output_file.replace('.pdf', '.png'), format='png', bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.close()


def create_summary_table(results: List[Dict], output_file: str):
    """Create a summary table in LaTeX format."""
    lines = []
    lines.append("\\begin{table}[htbp]")
    lines.append("\\centering")
    lines.append("\\caption{Scalability Experiment Results}")
    lines.append("\\label{tab:scalability}")
    lines.append("\\begin{tabular}{lccccc}")
    lines.append("\\hline")
    lines.append("Experts & RMSE & Time/Epoch (s) & Memory (GB) & Throughput (s/s) & Entropy \\\\")
    lines.append("\\hline")

    for result in results:
        model_type = result['experiment']['model_type']
        lines.append(f"\\multicolumn{{6}}{{c}}{{\\textbf{{{model_type.upper()}}}}} \\\\")

        for res in result['individual_results']:
            experts = res['num_experts']
            rmse = res['best_rmse']
            time = res['efficiency'].get('avg_epoch_time', 0)
            memory = res['efficiency'].get('peak_gpu_memory', 0)
            throughput = res['efficiency'].get('avg_throughput', 0)
            entropy = res['efficiency'].get('normalized_entropy', 0)

            lines.append(f"{experts} & {rmse:.4f} & {time:.1f} & {memory:.2f} & {throughput:.1f} & {entropy:.3f} \\\\")

        lines.append("\\hline")

    lines.append("\\end{tabular}")
    lines.append("\\end{table}")

    latex_content = "\n".join(lines)

    with open(output_file, 'w') as f:
        f.write(latex_content)

    print(f"Saved LaTeX table: {output_file}")
    print("\n" + latex_content)


def main(args):
    # Load all results
    results = []
    result_files = Path(args.results_dir).glob('scalability_analysis_*.json')

    for file in result_files:
        result = load_results(str(file))
        results.append(result)
        print(f"Loaded: {file}")

    if not results:
        print(f"No results found in {args.results_dir}")
        return

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nGenerating plots in: {output_dir}")

    # Generate plots
    plot_performance_scaling(results, str(output_dir / 'performance_scaling.pdf'))
    plot_time_scaling(results, str(output_dir / 'time_scaling.pdf'))
    plot_memory_scaling(results, str(output_dir / 'memory_scaling.pdf'))
    plot_expert_utilization(results, str(output_dir / 'expert_utilization.pdf'))
    plot_efficiency_comparison(results, str(output_dir / 'efficiency_comparison.pdf'))

    # Generate summary table
    create_summary_table(results, str(output_dir / 'summary_table.tex'))

    print("\nVisualization complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize scalability experiments")

    parser.add_argument('--results_dir', type=str,
                        default='results_10m/scalability',
                        help='Directory containing scalability_analysis_*.json files')
    parser.add_argument('--output_dir', type=str,
                        default='results_10m/scalability/plots',
                        help='Output directory for plots')

    args = parser.parse_args()
    main(args)
