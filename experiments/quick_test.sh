#!/bin/bash

# 빠른 확장성 테스트 (2개 expert만, 3 epochs)
# 개발/디버깅용

echo "========================================"
echo "Quick Scalability Test (Development)"
echo "========================================"
echo ""

EXPERIMENT_ID="quick_test_$(date +%Y%m%d_%H%M%S)"

echo "Running quick test with:"
echo "  - Expert counts: 4, 8"
echo "  - Epochs: 3"
echo "  - Models: dense, ppo"
echo ""

uv run python3 experiments/scalability_experiment.py \
    --data_dir ml-100k \
    --train_rating_path ml-100k/u1.base \
    --test_rating_path ml-100k/u1.test \
    --models dense ppo \
    --num_experts_list 4 8 \
    --num_epochs 3 \
    --experiment_id $EXPERIMENT_ID \
    --output_dir experiments/results \
    --batch_size 256 \
    --lr 0.001 \
    --seed 42

RESULT_FILE="experiments/results/scalability_results_${EXPERIMENT_ID}.json"

if [ -f "$RESULT_FILE" ]; then
    echo ""
    echo "Test completed! Generating visualizations..."

    uv run python3 experiments/visualize_scalability.py \
        --result_path $RESULT_FILE \
        --output_dir "experiments/visualizations/$EXPERIMENT_ID" \
        --dpi 300

    echo ""
    echo "Results: experiments/visualizations/$EXPERIMENT_ID/"
fi
