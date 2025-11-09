"""
Efficiency metrics for measuring scalability and performance.

Tracks:
- Training time per epoch
- Memory usage (GPU/CPU)
- Expert utilization (entropy, gini coefficient)
- Throughput (samples/second)
- Model size (parameters)
"""
import time
import torch
import numpy as np
from typing import Dict, List, Optional
import psutil
import gc


class EfficiencyTracker:
    """Track efficiency metrics during training."""

    def __init__(self):
        self.reset()

    def reset(self):
        """Reset all metrics."""
        self.epoch_times = []
        self.batch_times = []
        self.gpu_memory_usage = []
        self.cpu_memory_usage = []
        self.expert_distributions = []
        self.throughput = []

    def start_epoch(self):
        """Start timing an epoch."""
        self.epoch_start_time = time.time()

    def end_epoch(self):
        """End timing an epoch."""
        epoch_time = time.time() - self.epoch_start_time
        self.epoch_times.append(epoch_time)
        return epoch_time

    def start_batch(self):
        """Start timing a batch."""
        self.batch_start_time = time.time()

    def end_batch(self, batch_size: int):
        """End timing a batch and compute throughput."""
        batch_time = time.time() - self.batch_start_time
        self.batch_times.append(batch_time)

        # Throughput (samples/second)
        throughput = batch_size / batch_time if batch_time > 0 else 0
        self.throughput.append(throughput)

        return batch_time, throughput

    def record_memory(self):
        """Record current GPU and CPU memory usage."""
        # GPU memory
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.memory_allocated() / (1024 ** 3)  # GB
            self.gpu_memory_usage.append(gpu_memory)

        # CPU memory
        process = psutil.Process()
        cpu_memory = process.memory_info().rss / (1024 ** 3)  # GB
        self.cpu_memory_usage.append(cpu_memory)

    def record_expert_distribution(self, gate_probs: torch.Tensor):
        """
        Record expert selection distribution.

        Args:
            gate_probs: [batch_size, num_experts] gate probabilities
        """
        # Average expert usage across batch
        expert_usage = gate_probs.mean(dim=0).cpu().numpy()
        self.expert_distributions.append(expert_usage)

    def compute_expert_metrics(self) -> Dict[str, float]:
        """
        Compute expert utilization metrics.

        Returns:
            entropy: Shannon entropy (higher = more uniform)
            gini: Gini coefficient (0 = uniform, 1 = concentrated)
            std: Standard deviation of expert usage
            max_usage: Maximum expert usage
            min_usage: Minimum expert usage
        """
        if not self.expert_distributions:
            return {}

        # Average expert usage across all batches
        avg_expert_usage = np.mean(self.expert_distributions, axis=0)

        # Normalize to probability distribution
        prob = avg_expert_usage / avg_expert_usage.sum()

        # Shannon entropy
        entropy = -np.sum(prob * np.log(prob + 1e-10))
        max_entropy = np.log(len(prob))
        normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0

        # Gini coefficient
        sorted_usage = np.sort(avg_expert_usage)
        n = len(sorted_usage)
        index = np.arange(1, n + 1)
        gini = (2 * np.sum(index * sorted_usage)) / (n * np.sum(sorted_usage)) - (n + 1) / n

        # Standard deviation
        std = np.std(avg_expert_usage)

        # Min/max usage
        max_usage = np.max(avg_expert_usage)
        min_usage = np.min(avg_expert_usage)

        return {
            'entropy': entropy,
            'normalized_entropy': normalized_entropy,
            'gini': gini,
            'std': std,
            'max_usage': max_usage,
            'min_usage': min_usage,
            'usage_ratio': max_usage / (min_usage + 1e-10)
        }

    def get_summary(self) -> Dict[str, float]:
        """Get summary of all efficiency metrics."""
        summary = {}

        # Time metrics
        if self.epoch_times:
            summary['avg_epoch_time'] = np.mean(self.epoch_times)
            summary['total_training_time'] = np.sum(self.epoch_times)

        if self.batch_times:
            summary['avg_batch_time'] = np.mean(self.batch_times)
            summary['std_batch_time'] = np.std(self.batch_times)

        # Throughput
        if self.throughput:
            summary['avg_throughput'] = np.mean(self.throughput)
            summary['std_throughput'] = np.std(self.throughput)

        # Memory
        if self.gpu_memory_usage:
            summary['avg_gpu_memory'] = np.mean(self.gpu_memory_usage)
            summary['peak_gpu_memory'] = np.max(self.gpu_memory_usage)

        if self.cpu_memory_usage:
            summary['avg_cpu_memory'] = np.mean(self.cpu_memory_usage)
            summary['peak_cpu_memory'] = np.max(self.cpu_memory_usage)

        # Expert metrics
        expert_metrics = self.compute_expert_metrics()
        summary.update(expert_metrics)

        return summary

    def clear_gpu_cache(self):
        """Clear GPU cache to free memory."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()


def count_model_parameters(model: torch.nn.Module) -> Dict[str, int]:
    """
    Count model parameters.

    Returns:
        total: Total parameters
        trainable: Trainable parameters
        non_trainable: Non-trainable parameters
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return {
        'total': total,
        'trainable': trainable,
        'non_trainable': total - trainable
    }


def compute_flops_per_sample(
    num_experts: int,
    embedding_dim: int,
    state_dim: int,
    expert_hidden_dims: List[int]
) -> int:
    """
    Estimate FLOPs per sample (forward pass only).

    Args:
        num_experts: Number of experts
        embedding_dim: Embedding dimension
        state_dim: State dimension
        expert_hidden_dims: Hidden layer dimensions in expert network

    Returns:
        Approximate FLOPs per sample
    """
    flops = 0

    # Embedding lookup (negligible)

    # Gating network (state_dim -> num_experts)
    flops += state_dim * num_experts

    # Expert networks (assuming all experts are evaluated)
    # Each expert: state_dim + genres -> hidden layers -> 1
    prev_dim = state_dim + 128  # Including genre features
    for hidden_dim in expert_hidden_dims:
        flops += num_experts * (prev_dim * hidden_dim)
        prev_dim = hidden_dim

    # Final layer (hidden -> 1)
    flops += num_experts * prev_dim

    # Weighted combination
    flops += num_experts

    return flops


def compare_efficiency(
    results: List[Dict],
    baseline_experts: int = 16
) -> Dict:
    """
    Compare efficiency across different expert counts.

    Args:
        results: List of result dicts with 'num_experts' and metrics
        baseline_experts: Baseline number of experts for comparison

    Returns:
        Comparison metrics showing scaling behavior
    """
    if not results:
        return {}

    # Sort by num_experts
    results = sorted(results, key=lambda x: x['num_experts'])

    # Find baseline
    baseline = None
    for r in results:
        if r['num_experts'] == baseline_experts:
            baseline = r
            break

    if baseline is None:
        baseline = results[0]

    comparison = {
        'baseline_experts': baseline['num_experts'],
        'comparisons': []
    }

    for r in results:
        n_experts = r['num_experts']
        scale_factor = n_experts / baseline['num_experts']

        comp = {
            'num_experts': n_experts,
            'scale_factor': scale_factor,
        }

        # Time scaling (ideal: linear or sublinear)
        if 'avg_epoch_time' in baseline and 'avg_epoch_time' in r:
            time_ratio = r['avg_epoch_time'] / baseline['avg_epoch_time']
            comp['time_scaling'] = time_ratio
            comp['time_efficiency'] = scale_factor / time_ratio  # >1 is good

        # Memory scaling
        if 'peak_gpu_memory' in baseline and 'peak_gpu_memory' in r:
            memory_ratio = r['peak_gpu_memory'] / baseline['peak_gpu_memory']
            comp['memory_scaling'] = memory_ratio
            comp['memory_efficiency'] = scale_factor / memory_ratio

        # Throughput scaling
        if 'avg_throughput' in baseline and 'avg_throughput' in r:
            throughput_ratio = r['avg_throughput'] / baseline['avg_throughput']
            comp['throughput_scaling'] = throughput_ratio  # Should stay ~1

        # Expert utilization
        if 'normalized_entropy' in r:
            comp['expert_entropy'] = r['normalized_entropy']
        if 'gini' in r:
            comp['expert_gini'] = r['gini']

        # Performance
        if 'rmse' in r:
            comp['rmse'] = r['rmse']

        comparison['comparisons'].append(comp)

    return comparison
