#!/bin/bash
# Train Refined RF and Embedded XGB on all windows (Multi-label)

WINDOWS=(1 5 10 15 20)

for w in "${WINDOWS[@]}"; do
    echo "=================================================="
    echo "Training Wrapper for Window Size: $w (Multi-label)"
    echo "=================================================="
    
    # 1. Random Forest (Refined)
    echo ">> Training Random Forest (Refined)..."
    venv/bin/python3 scripts/train_random_forest_model.py \
        --window_size $w \
        --n_trials 30 \
        --feature_set full \
        --optimize_metric f1

    # 2. XGBoost (Embedded)
    echo ">> Training XGBoost (Embedded)..."
    venv/bin/python3 scripts/train_xgboost_model.py \
        --window_size $w \
        --n_trials 30 \
        --feature_set full \
        --optimize_metric f1

    # 3. Decision Tree (Optimized)
    echo ">> Training Decision Tree (Optimized)..."
    venv/bin/python3 scripts/train_decision_tree_model.py \
        --window_size $w \
        --n_trials 30 \
        --feature_set full \
        --optimize_metric f1
        
    # Validation & Cleanup
    echo ">> Syncing filesystem and cooling down..."
    sync
    sleep 5
    
    echo "Finished Window Size: $w"
done
