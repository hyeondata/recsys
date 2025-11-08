#!/bin/bash

# 확장성 실험 실행 스크립트
# Expert 개수를 늘려서 Dense MoE와 RL-based MoE의 확장성 차이를 비교

echo "========================================"
echo "MoE Scalability Experiment"
echo "========================================"
echo ""

# 실험 ID (타임스탬프)
EXPERIMENT_ID=$(date +%Y%m%d_%H%M%S)

echo "Experiment ID: $EXPERIMENT_ID"
echo ""

# 기본 설정
DATA_DIR="ml-100k"
TRAIN_RATING="ml-100k/u1.base"
TEST_RATING="ml-100k/u1.test"

# Expert 개수 리스트 (4, 8, 16, 32)
EXPERT_COUNTS="4 8 16 32"

# 테스트할 모델
MODELS="dense ppo grpo"

# 학습 epoch 수 (빠른 실험을 위해 5 epoch만)
NUM_EPOCHS=5

# 출력 디렉토리
OUTPUT_DIR="experiments/results"

echo "Configuration:"
echo "  Data: $DATA_DIR"
echo "  Expert counts: $EXPERT_COUNTS"
echo "  Models: $MODELS"
echo "  Epochs per config: $NUM_EPOCHS"
echo ""

# 실험 실행
echo "Starting scalability experiment..."
echo ""

uv run python3 experiments/scalability_experiment.py \
    --data_dir $DATA_DIR \
    --train_rating_path $TRAIN_RATING \
    --test_rating_path $TEST_RATING \
    --models $MODELS \
    --num_experts_list $EXPERT_COUNTS \
    --num_epochs $NUM_EPOCHS \
    --experiment_id $EXPERIMENT_ID \
    --output_dir $OUTPUT_DIR \
    --batch_size 256 \
    --lr 0.001 \
    --seed 42

# 실험 결과 확인
RESULT_FILE="$OUTPUT_DIR/scalability_results_${EXPERIMENT_ID}.json"

if [ -f "$RESULT_FILE" ]; then
    echo ""
    echo "========================================"
    echo "Experiment completed successfully!"
    echo "========================================"
    echo ""
    echo "Results saved to: $RESULT_FILE"
    echo ""

    # 시각화 생성
    echo "Generating visualizations..."
    echo ""

    uv run python3 experiments/visualize_scalability.py \
        --result_path $RESULT_FILE \
        --output_dir "experiments/visualizations/$EXPERIMENT_ID" \
        --dpi 600

    echo ""
    echo "========================================"
    echo "Visualization completed!"
    echo "========================================"
    echo ""
    echo "Check results in:"
    echo "  - Results: $RESULT_FILE"
    echo "  - Plots: experiments/visualizations/$EXPERIMENT_ID/"
    echo ""
else
    echo ""
    echo "ERROR: Experiment failed. Results file not found."
    echo ""
fi
