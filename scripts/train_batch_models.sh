#!/bin/bash
# Train Refined RF and Embedded XGB on all windows

WINDOWS=(1 5 10 15 20)

for w in "${WINDOWS[@]}"; do
    echo "=================================================="
    echo "Training Wrapper for Window Size: $w"
    echo "=================================================="
    
    # 1. Random Forest (Refined)
    echo ">> Training Random Forest (Refined)..."
    venv/bin/python3 scripts/train_random_forest_model.py \
        --window_size $w \
        --binary_target \
        --n_trials 30 \
        --model_filename "rf_tiny_refined_w${w}_binary.pkl" \
        --metrics_filename "rf_tiny_refined_metrics_w${w}_binary.txt"

    # 2. XGBoost (Embedded)
    echo ">> Training XGBoost (Embedded)..."
    venv/bin/python3 scripts/train_xgboost_model.py \
        --window_size $w \
        --binary_target \
        --n_trials 30 \
        --model_filename "xgb_embedded_w${w}_binary.pkl" \
        --metrics_filename "xgb_embedded_metrics_w${w}_binary.txt"

    # 3. Decision Tree (Optimized)
    echo ">> Training Decision Tree (Optimized)..."
    venv/bin/python3 scripts/train_decision_tree_model.py \
        --window_size $w \
        --binary_target \
        --n_trials 30 \
        --model_filename "dt_optimized_w${w}_binary.pkl" \
        --metrics_filename "dt_optimized_metrics_w${w}_binary.txt"
        
    echo "Finished Window Size: $w"
done
