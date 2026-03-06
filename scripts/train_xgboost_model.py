"""
Train XGBoost model with Optuna tuning.
"""

import argparse
import logging
import os
import sys
import time
from typing import Callable, Tuple
import joblib

import numpy as np
import pandas as pd
import optuna
from xgboost import XGBClassifier
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    f1_score,
    fbeta_score,
    hamming_loss,
    recall_score,
)
from sklearn.preprocessing import StandardScaler

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Train XGBoost model with Optuna tuning."
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/processed/tabular",
        help="Path to processed data.",
    )
    parser.add_argument(
        "--model_dir", type=str, default="models", help="Directory to save models."
    )
    parser.add_argument(
        "--window_size", type=int, default=1, help="Window size to train on."
    )
    parser.add_argument(
        "--binary_target", action="store_true", help="Use binary target (Fault/Normal)."
    )
    parser.add_argument(
        "--n_trials", type=int, default=20, help="Number of Optuna trials."
    )
    parser.add_argument(
        "--model_filename", type=str, default=None, help="Output model filename."
    )
    parser.add_argument(
        "--metrics_filename", type=str, default=None, help="Output metrics filename."
    )
    parser.add_argument(
        "--normalize",
        action="store_true",
        help="Normalize features using StandardScaler.",
    )
    parser.add_argument(
        "--optimize_metric",
        type=str,
        default="f1",
        choices=["f1", "f2", "recall"],
        help="Metric to optimize (f1, f2, recall).",
    )
    parser.add_argument(
        "--feature_set",
        type=str,
        default="full",
        choices=["full", "stat_only"],
        help="full: Use all features. stat_only: Drop raw/mean/min/max.",
    )
    return parser.parse_args()


def load_datasets(
    data_dir: str, window_size: int, binary_target: bool, feature_set: str
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Loads train, validation, and test CSVs and applies feature filtering.
    """
    mode_str = "binary" if binary_target else "multi"
    base_path = os.path.join(data_dir, f"w{window_size}", mode_str)

    train_path = os.path.join(base_path, "train.csv")
    val_path = os.path.join(base_path, "val.csv")
    test_path = os.path.join(base_path, "test.csv")

    if not os.path.exists(train_path):
        logger.error("Datasets not found in %s", base_path)
        sys.exit(1)

    logger.info("Loading training data from %s...", train_path)
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)

    # Identify Targets
    target_cols = [
        c for c in train_df.columns if c.startswith("fault_") or c == "any_fault"
    ]

    # Identify Features
    all_cols = train_df.columns
    feature_cols = [
        c for c in all_cols
        if c not in target_cols and c != "Time" and not c.startswith("point_error")
    ]

    # Filter Features
    if feature_set == "stat_only":
        logger.info(
            "Applying 'stat_only' filter: Dropping Raw, Mean, Min, Max, Median."
        )
        STAT_SUFFIXES = (
            "_delta", "_accel",
            "_var_", "_std_", "_kurt_", "_skew_", "_range_", "_dev_",
            "_mean_", "_min_", "_max_", "_median_",
        )
        filtered_cols = [col for col in feature_cols if any(s in col for s in STAT_SUFFIXES)]

        if len(filtered_cols) == 0:
            logger.error("Feature set 'stat_only' resulted in 0 features!")
            sys.exit(1)
        feature_cols = filtered_cols
        logger.info("Features reduced to %s.", len(feature_cols))

    # Apply Selection and Cast
    train_df = train_df[feature_cols + target_cols]
    val_df = val_df[feature_cols + target_cols]
    test_df = test_df[feature_cols + target_cols]

    # Convert features to float32
    train_df[feature_cols] = train_df[feature_cols].astype("float32")
    val_df[feature_cols] = val_df[feature_cols].astype("float32")
    test_df[feature_cols] = test_df[feature_cols].astype("float32")

    return train_df, val_df, test_df, feature_cols, target_cols


def get_objective(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    binary_target: bool,
    scale_pos_weight: float,
    optimize_metric: str = "f1",
) -> Callable:
    """
    Returns the objective function for Optuna hyperparameter optimization.
    """

    def objective(trial: optuna.Trial) -> float:
        # --- TUNE WEIGHTING STRATEGY ---
        weight_strategy = trial.suggest_categorical(
            "weight_strategy", ["balanced", "none"]
        )
        current_scale_weight = (
            scale_pos_weight if weight_strategy == "balanced" else 1.0
        )

        params = {
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "n_estimators": trial.suggest_int("n_estimators", 20, 100),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "gamma": trial.suggest_float("gamma", 0.0, 5.0),
            "n_jobs": -1,
            "random_state": 42,
            "tree_method": "hist",
            "grow_policy": trial.suggest_categorical(
                "grow_policy", ["depthwise", "lossguide"]
            ),
        }

        # If lossguide, limit leaves
        if params["grow_policy"] == "lossguide":
            params["max_leaves"] = trial.suggest_int("max_leaves", 16, 64)

        if binary_target:
            params["objective"] = "binary:logistic"
            params["eval_metric"] = "logloss"
            params["scale_pos_weight"] = current_scale_weight
        else:
            params["objective"] = "multi:softprob"
            params["eval_metric"] = "mlogloss"

        clf = XGBClassifier(**params)

        # Fit the model
        clf.fit(x_train, y_train, verbose=False)
        # Evaluate the model
        y_pred = clf.predict(x_val)

        # Calculate the score
        if binary_target:
            if optimize_metric == "f2":
                score = fbeta_score(
                    y_val, y_pred, beta=2, average="binary", zero_division=0
                )
            elif optimize_metric == "recall":
                score = recall_score(y_val, y_pred, average="binary", zero_division=0)
            else:
                score = f1_score(y_val, y_pred, average="binary", zero_division=0)
        else:
            score = f1_score(y_val, y_pred, average="macro", zero_division=0)

        return score

    return objective


def save_feature_importance(model, feature_names, output_path):
    """
    Extracts and saves feature importance.
    """
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("Rank,Feature,Importance\n")
        for i in range(len(feature_names)):
            idx = indices[i]
            if importances[idx] > 0:
                f.write(f"{i+1},{feature_names[idx]},{importances[idx]:.6f}\n")

    logger.info("Feature importance saved to %s", output_path)


def get_system_metrics(model, x_test, model_path):
    """
    Calculates Model Size (Storage) and Inference Latency (Speed).
    Useful for embedded feasibility analysis.
    """
    # Model Size
    if not os.path.exists(model_path):
        return 0.0, 0.0

    size_bytes = os.path.getsize(model_path)
    size_kb = size_bytes / 1024

    # Inference Time
    # Warm-up run (to load into cache)
    _ = model.predict(x_test[:100])

    # Timing run
    start_time = time.perf_counter()
    _ = model.predict(x_test)
    end_time = time.perf_counter()

    total_time = end_time - start_time
    # Average time per single sample in microseconds
    latency_us = (total_time / len(x_test)) * 1e6

    return size_kb, latency_us


def main():
    """
    Main function to train the XGBoost model.
    """
    args = parse_arguments()

    mode_str = "binary" if args.binary_target else "multi"

    # Naming convention updates
    if args.model_filename is None:
        args.model_filename = (
            f"xgb_model_w{args.window_size}_{mode_str}_{args.feature_set}.pkl"
        )
    if args.metrics_filename is None:
        args.metrics_filename = (
            f"xgb_metrics_w{args.window_size}_{mode_str}_{args.feature_set}.txt"
        )

    imp_filename = (
        f"xgb_importance_w{args.window_size}_{mode_str}_{args.feature_set}.csv"
    )

    # Paths
    os.makedirs(args.model_dir, exist_ok=True)
    model_path = os.path.join(args.model_dir, args.model_filename)
    metrics_path = os.path.join(args.model_dir, args.metrics_filename)
    imp_path = os.path.join(args.model_dir, imp_filename)

    # Load Data
    train_df, val_df, test_df, feature_cols, target_cols = load_datasets(
        args.data_dir, args.window_size, args.binary_target, args.feature_set
    )

    # Prepare NumPy arrays
    x_train = train_df[feature_cols].to_numpy()
    y_train = train_df[target_cols].to_numpy()

    x_val = val_df[feature_cols].to_numpy()
    y_val = val_df[target_cols].to_numpy()

    x_test = test_df[feature_cols].to_numpy()
    y_test = test_df[target_cols].to_numpy()

    if args.binary_target:
        y_train = y_train.ravel()
        y_val = y_val.ravel()
        y_test = y_test.ravel()

    # Normalize if requested (skip if builder already normalized)
    scaler_path = os.path.join(
        args.data_dir, f"w{args.window_size}",
        "binary" if args.binary_target else "multi",
        "scaler_params.json",
    )
    if os.path.exists(scaler_path):
        logger.info("Using pre-normalized data from builder (scaler_params.json found).")
        if args.normalize:
            logger.warning(
                "--normalize flag is ignored: builder normalization already applied."
            )
    elif args.normalize:
        logger.info("Normalizing features using StandardScaler...")
        scaler = StandardScaler()
        x_train = scaler.fit_transform(x_train)
        x_val = scaler.transform(x_val)
        x_test = scaler.transform(x_test)

    # Calculate scale_pos_weight base
    base_scale_pos_weight = 1.0
    if args.binary_target:
        num_neg = np.sum(y_train == 0)
        num_pos = np.sum(y_train == 1)
        base_scale_pos_weight = num_neg / num_pos if num_pos > 0 else 1.0
        logger.info("Base Imbalance Ratio: %.2f", base_scale_pos_weight)

    logger.info(
        "Starting Optuna Optimization (%d trials)... Mode: %s",
        args.n_trials,
        args.feature_set,
    )

    objective = get_objective(
        x_train,
        y_train,
        x_val,
        y_val,
        args.binary_target,
        base_scale_pos_weight,
        args.optimize_metric,
    )
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=args.n_trials)

    logger.info("Best Params: %s", study.best_params)

    # Retrain best model
    best_params = study.best_params

    # Handle the custom weight strategy logic for final model
    weight_strategy = best_params.pop("weight_strategy", "none")
    final_scale_weight = base_scale_pos_weight if weight_strategy == "balanced" else 1.0

    best_params["n_jobs"] = -1
    best_params["random_state"] = 42
    best_params["tree_method"] = "hist"

    if args.binary_target:
        best_params["objective"] = "binary:logistic"
        best_params["eval_metric"] = "logloss"
        best_params["scale_pos_weight"] = final_scale_weight
    else:
        best_params["objective"] = "multi:softprob"
        best_params["eval_metric"] = "mlogloss"

    clf = XGBClassifier(**best_params)

    clf.fit(x_train, y_train, verbose=False)

    best_thresh = 0.5
    if args.binary_target:
        val_probs = clf.predict_proba(x_val)[:, 1]
        best_f1_val = 0.0
        for thresh in np.arange(0.1, 0.9, 0.05):
            y_val_pred = (val_probs > thresh).astype(int)
            f1_val = f1_score(y_val, y_val_pred, average="binary", zero_division=0)
            if f1_val > best_f1_val:
                best_f1_val = f1_val
                best_thresh = thresh
        logger.info("Best Threshold: %.2f (Val F1: %.4f)", best_thresh, best_f1_val)

    # Evaluate
    if args.binary_target:
        y_pred = (clf.predict_proba(x_test)[:, 1] > best_thresh).astype(int)
        target_names = ["Normal", "Fault"]
    else:
        y_pred = clf.predict(x_test)
        target_names = target_cols

    accuracy = accuracy_score(y_test, y_pred)
    hamming = hamming_loss(y_test, y_pred)
    if args.binary_target:
        f1 = f1_score(y_test, y_pred, average="binary", zero_division=0)
    else:
        f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)

    report = classification_report(
        y_test, y_pred, target_names=target_names, zero_division=0
    )

    logger.info("Test F1: %.4f", f1)

    # Save Artifacts
    joblib.dump(clf, model_path)
    logger.info("Model saved to %s", model_path)

    # Calculate System Metrics
    size_kb, latency_us = get_system_metrics(clf, x_test, model_path)
    logger.info("Model Size: %.2f KB", size_kb)
    logger.info("Inference Time: %.2f µs/sample", latency_us)

    with open(metrics_path, "w", encoding="utf-8") as f:
        f.write(f"Feature Set: {args.feature_set}\n")
        f.write(f"Weight Strategy: {weight_strategy}\n")
        f.write(f"Best Params: {best_params}\n")
        f.write(f"Best Threshold: {best_thresh:.2f}\n")
        f.write(f"Test F1 Score: {f1:.4f}\n")
        f.write(f"Test Accuracy: {accuracy:.4f}\n")
        f.write(f"Test Hamming Loss: {hamming:.4f}\n")
        f.write(f"Model Size (KB): {size_kb:.4f}\n")
        f.write(f"Inference Time (us/sample): {latency_us:.4f}\n")
        f.write(f"\nClassification Report:\n{report}\n")

    save_feature_importance(clf, feature_cols, imp_path)

    logger.info("Done.")


if __name__ == "__main__":
    main()
