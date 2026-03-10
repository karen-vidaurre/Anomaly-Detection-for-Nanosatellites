#!/bin/bash
# Train Refined RF and Embedded XGB on all windows (Multi-label)

WINDOWS=(1 5 10 15 20)

for w in "${WINDOWS[@]}"; do
    echo "=================================================="
    echo "Training Wrapper for Window Size: $w (Multi-label)"
    echo "=================================================="
    
    # 1. Random Forest (Refined)
    # n_jobs=1: avoids spawning N worker copies of the 218MB feature matrix
    # optuna_subsample=0.25: runs each trial on 70K rows instead of 280K (final model uses full data)
    echo ">> Training Random Forest (Refined)..."
    venv/bin/python3 scripts/train_random_forest_model.py \
        --window_size $w \
        --n_trials 30 \
        --feature_set full \
        --optimize_metric f1 \
        --n_jobs 1 \
        --optuna_subsample 0.25

    # 2. XGBoost (Embedded)
    # XGBoost hist method is already memory-efficient; subsample still helps across 50 trials
    echo ">> Training XGBoost (Embedded)..."
    venv/bin/python3 scripts/train_xgboost_model.py \
        --window_size $w \
        --n_trials 30 \
        --feature_set full \
        --optimize_metric f1 \
        --optuna_subsample 0.25

    # 3. Decision Tree (Optimized)
    echo ">> Training Decision Tree (Optimized)..."
    venv/bin/python3 scripts/train_decision_tree_model.py \
        --window_size $w \
        --n_trials 30 \
        --feature_set full \
        --optimize_metric f1 \
        --optuna_subsample 0.25
        
    # Validation & Cleanup
    echo ">> Syncing filesystem and cooling down..."
    sync
    sleep 5
    
    echo "Finished Window Size: $w"
done
