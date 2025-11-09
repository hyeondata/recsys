#!/bin/bash
# Run scalability experiments for all models with different expert counts
#
# Usage:
#   bash run_all_scalability_experiments.sh
#
# This will run experiments with 8, 16, 32, 64, and 128 experts
# for Dense, PPO, and GRPO models.

set -e  # Exit on error

# Configuration
DATA_DIR="ml-10M100K"
OUTPUT_DIR="results_10m/scalability"
EPOCHS=10  # Reduce for faster experiments
BATCH_SIZE=1024
EXPERT_COUNTS="8 16 32 64 128"

echo "=========================================="
echo "Scalability Experiments"
echo "=========================================="
echo "Expert counts: ${EXPERT_COUNTS}"
echo "Epochs per experiment: ${EPOCHS}"
echo "Batch size: ${BATCH_SIZE}"
echo "Output: ${OUTPUT_DIR}"
echo "=========================================="
echo ""

# Create output directory
mkdir -p ${OUTPUT_DIR}

# Run Dense MoE experiments
echo "Running Dense MoE experiments..."
uv run python3 src_10m/experiments/run_scalability_experiment.py \
    --model_type dense \
    --expert_counts ${EXPERT_COUNTS} \
    --epochs ${EPOCHS} \
    --batch_size ${BATCH_SIZE} \
    --data_dir ${DATA_DIR} \
    --output_dir ${OUTPUT_DIR} \
    --baseline_experts 16

echo ""
echo "Dense MoE complete!"
echo ""

# Run PPO-MoE experiments
echo "Running PPO-MoE experiments..."
uv run python3 src_10m/experiments/run_scalability_experiment.py \
    --model_type ppo \
    --expert_counts ${EXPERT_COUNTS} \
    --epochs ${EPOCHS} \
    --batch_size 512 \
    --data_dir ${DATA_DIR} \
    --output_dir ${OUTPUT_DIR} \
    --baseline_experts 16

echo ""
echo "PPO-MoE complete!"
echo ""

# Run GRPO-MoE experiments
echo "Running GRPO-MoE experiments..."
uv run python3 src_10m/experiments/run_scalability_experiment.py \
    --model_type grpo \
    --expert_counts ${EXPERT_COUNTS} \
    --epochs ${EPOCHS} \
    --batch_size 512 \
    --data_dir ${DATA_DIR} \
    --output_dir ${OUTPUT_DIR} \
    --baseline_experts 16

echo ""
echo "GRPO-MoE complete!"
echo ""

# Generate visualizations
echo "=========================================="
echo "Generating visualizations..."
echo "=========================================="

uv run python3 src_10m/experiments/visualize_scalability.py \
    --results_dir ${OUTPUT_DIR} \
    --output_dir ${OUTPUT_DIR}/plots

echo ""
echo "=========================================="
echo "All experiments complete!"
echo "=========================================="
echo "Results saved to: ${OUTPUT_DIR}"
echo "Plots saved to: ${OUTPUT_DIR}/plots"
echo ""
echo "Files generated:"
echo "  - scalability_analysis_dense.json"
echo "  - scalability_analysis_ppo.json"
echo "  - scalability_analysis_grpo.json"
echo "  - plots/performance_scaling.pdf"
echo "  - plots/time_scaling.pdf"
echo "  - plots/memory_scaling.pdf"
echo "  - plots/expert_utilization.pdf"
echo "  - plots/efficiency_comparison.pdf"
echo "  - plots/summary_table.tex"
echo "=========================================="
