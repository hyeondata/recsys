"""
Publication-quality plots for 10M dataset results.
"""
import json, argparse, os
import matplotlib.pyplot as plt
import numpy as np

plt.style.use('seaborn-v0_8-paper')
plt.rcParams['figure.dpi'] = 600
plt.rcParams['savefig.dpi'] = 600
plt.rcParams['font.size'] = 10
plt.rcParams['font.family'] = 'serif'


def plot_performance_comparison(results, output_dir):
    """Plot performance metrics comparison."""
    models = list(results.keys())
    metrics = ['mse', 'rmse', 'mae']
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    for idx, metric in enumerate(metrics):
        values = [results[m][metric] for m in models]
        axes[idx].bar(models, values, color='steelblue', alpha=0.8)
        axes[idx].set_ylabel(metric.upper())
        axes[idx].set_title(f'{metric.upper()} Comparison')
        axes[idx].grid(axis='y', alpha=0.3)
        
        # Add value labels
        for i, v in enumerate(values):
            axes[idx].text(i, v + max(values)*0.02, f'{v:.4f}', ha='center')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'performance_comparison.pdf'), bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'performance_comparison.png'), bbox_inches='tight')
    plt.close()
    print("Saved performance comparison plots")


def plot_expert_distribution(results, output_dir):
    """Plot expert distribution for RL models."""
    fig, axes = plt.subplots(1, len(results), figsize=(5*len(results), 4))
    if len(results) == 1:
        axes = [axes]
    
    for idx, (model_name, data) in enumerate(results.items()):
        if 'expert_distribution' in data:
            expert_dist = data['expert_distribution']
            experts = sorted(expert_dist.keys())
            counts = [expert_dist[e] for e in experts]
            
            axes[idx].bar([f'E{e}' for e in experts], counts, color='coral', alpha=0.8)
            axes[idx].set_title(f'{model_name.upper()} Expert Distribution')
            axes[idx].set_xlabel('Expert')
            axes[idx].set_ylabel('Count')
            axes[idx].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'expert_distribution.pdf'), bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'expert_distribution.png'), bbox_inches='tight')
    plt.close()
    print("Saved expert distribution plots")


def main(args):
    os.makedirs(args.output_dir, exist_ok=True)
    
    with open(args.results_file, 'r') as f:
        results = json.load(f)
    
    print(f"Loaded results for models: {list(results.keys())}")
    
    plot_performance_comparison(results, args.output_dir)
    plot_expert_distribution(results, args.output_dir)
    
    print(f"\nAll plots saved to {args.output_dir}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument('--results_file', required=True, help='JSON file with evaluation results')
    p.add_argument('--output_dir', default='results_10m/plots', help='Output directory')
    main(p.parse_args())
