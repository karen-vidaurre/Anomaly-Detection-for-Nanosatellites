"""
This script trains a Decision Tree model on the processed datasets.
"""

import argparse
import logging
import os
import sys
import time
from typing import Tuple, Callable

import joblib
import numpy as np
import optuna
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    f1_score,
    accuracy_score,
    hamming_loss,
    classification_report,
    fbeta_score,
    recall_score,
)
from sklearn.preprocessing import StandardScaler

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
    parser = argparse.ArgumentParser(description="Train a Decision Tree model.")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/processed/tabular",
        help="Root directory where processed datasets are stored.",
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        default="models",
        help="Directory to save the model and metrics.",
    )
    parser.add_argument(
        "--model_filename",
        type=str,
        default=None,
        help="Filename. Defaults to 'model_w{window}_{type}.pkl'.",
    )
    parser.add_argument(
        "--metrics_filename",
        type=str,
        default=None,
        help="Filename. Defaults to 'metrics_w{window}_{type}.txt'.",
    )
    parser.add_argument(
        "--n_trials", type=int, default=30, help="Number of Optuna trials."
    )
    parser.add_argument(
        "--window_size",
        type=int,
        default=1,
        help="Window size to train on.",
    )
    parser.add_argument(
        "--binary_target",
        action="store_true",
        help="If set, uses binary targets (0=Normal, 1=Any Fault).",
    )
    parser.add_argument(
        "--normalize",
        action="store_true",
        help="Normalize features using StandardScaler (Optional for Trees).",
    )
    parser.add_argument(
        "--optimize_metric",
        type=str,
        default="f1",
        choices=["f1", "f2", "recall"],
        help="Metric to optimize.",
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
    Loads train/val/test and filters features based on feature_set mode.
    """
    mode_str = "binary" if binary_target else "multi"
    base_path = os.path.join(data_dir, f"w{window_size}", mode_str)

    train_path = os.path.join(base_path, "train.csv")
    val_path = os.path.join(base_path, "val.csv")
    test_path = os.path.join(base_path, "test.csv")

    if not os.path.exists(train_path):
        logger.error("Datasets not found in: %s", base_path)
        sys.exit(1)

    logger.info("Loading data from: %s", base_path)
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

    # --- FEATURE SELECTION ---
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
            logger.error(
                "Feature set 'stat_only' resulted in 0 features! Check column names."
            )
            sys.exit(1)

        feature_cols = filtered_cols
        logger.info(
            "Features reduced from %d to %d.",
            len(all_cols) - len(target_cols),
            len(feature_cols),
        )

    # Apply selection
    x_train = train_df[feature_cols].astype("float32")
    x_val = val_df[feature_cols].astype("float32")
    x_test = test_df[feature_cols].astype("float32")

    y_train = train_df[target_cols]
    y_val = val_df[target_cols]
    y_test = test_df[target_cols]

    return x_train, y_train, x_val, y_val, x_test, y_test, feature_cols


def get_objective(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    binary_target: bool,
    optimize_metric: str = "f1",
) -> Callable:
    """
    Returns an Optuna objective function for hyperparameter tuning.
    """

    def objective(trial: optuna.Trial) -> float:
        # Search Space
        max_depth = trial.suggest_int("max_depth", 5, 15)
        min_samples_split = trial.suggest_int("min_samples_split", 10, 200)
        min_samples_leaf = trial.suggest_int("min_samples_leaf", 5, 100)
        criterion = trial.suggest_categorical("criterion", ["gini", "entropy"])
        ccp_alpha = trial.suggest_float("ccp_alpha", 1e-5, 1e-2, log=True)
        class_weight_opt = trial.suggest_categorical(
            "class_weight", ["balanced", "None"]
        )
        cw = None if class_weight_opt == "None" else "balanced"

        clf = DecisionTreeClassifier(
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            ccp_alpha=ccp_alpha,
            criterion=criterion,
            class_weight=cw,
            random_state=42,
        )

        clf.fit(x_train, y_train)
        y_pred = clf.predict(x_val)

        # Calculate score based on metric
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


def evaluate_model(
    clf, x_test, y_test, target_names, binary_target, threshold=0.5
) -> Tuple[float, float, float, str]:
    """
    Evaluates the model on the test set.
    """
    if binary_target:
        y_prob = clf.predict_proba(x_test)[:, 1]
        y_pred = (y_prob > threshold).astype(int)
        f1_metric = f1_score(y_test, y_pred, average="binary", zero_division=0)
        report = classification_report(
            y_test, y_pred, target_names=["Normal", "Fault"], zero_division=0
        )
    else:
        y_pred = clf.predict(x_test)
        f1_metric = f1_score(y_test, y_pred, average="macro", zero_division=0)
        report = classification_report(
            y_test, y_pred, target_names=target_names, zero_division=0
        )

    subset_acc = accuracy_score(y_test, y_pred)
    hamming = hamming_loss(y_test, y_pred)
    return subset_acc, hamming, f1_metric, report


def save_feature_importance(model, feature_names, output_path) -> None:
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


def get_system_metrics(model, x_test, model_path) -> Tuple[float, float]:
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
    # Warm-up run
    _ = model.predict(x_test[:100])

    # Timing run
    start_time = time.perf_counter()
    _ = model.predict(x_test)  # Predict the whole test set
    end_time = time.perf_counter()

    total_time = end_time - start_time
    # Average time per single sample in microseconds
    latency_us = (total_time / len(x_test)) * 1e6

    return size_kb, latency_us


def main() -> None:
    """
    Main function to train the Decision Tree model.
    """
    args = parse_arguments()

    # Naming convention updates
    mode_suffix = "binary" if args.binary_target else "multi"
    if args.model_filename is None:
        args.model_filename = (
            f"dt_model_w{args.window_size}_{mode_suffix}_{args.feature_set}.pkl"
        )
    if args.metrics_filename is None:
        args.metrics_filename = (
            f"dt_metrics_w{args.window_size}_{mode_suffix}_{args.feature_set}.txt"
        )

    imp_filename = (
        f"dt_importance_w{args.window_size}_{mode_suffix}_{args.feature_set}.csv"
    )

    # Load Data
    x_train_df, y_train_df, x_val_df, y_val_df, x_test_df, y_test_df, feature_names = (
        load_datasets(
            args.data_dir, args.window_size, args.binary_target, args.feature_set
        )
    )

    # Get target names
    if args.binary_target:
        target_names = ["Normal", "Fault"]
    else:
        target_names = list(y_train_df.columns)

    # Convert to Numpy
    x_train = x_train_df.to_numpy()
    y_train = y_train_df.to_numpy()
    x_val = x_val_df.to_numpy()
    y_val = y_val_df.to_numpy()
    x_test = x_test_df.to_numpy()
    y_test = y_test_df.to_numpy()

    if args.binary_target:
        y_train = y_train.ravel()
        y_val = y_val.ravel()
        y_test = y_test.ravel()

    # Normalize
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
        scaler = StandardScaler()
        x_train = scaler.fit_transform(x_train)
        x_val = scaler.transform(x_val)
        x_test = scaler.transform(x_test)

    # Optuna
    logger.info(
        "Starting Optuna (%d trials)... Mode: %s", args.n_trials, args.feature_set
    )
    study = optuna.create_study(direction="maximize")
    objective = get_objective(
        x_train, y_train, x_val, y_val, args.binary_target, args.optimize_metric
    )
    study.optimize(objective, n_trials=args.n_trials)

    logger.info("Best Params: %s", study.best_trial.params)

    best_params = study.best_trial.params
    if "class_weight" in best_params:
        if best_params["class_weight"] == "None":
            best_params["class_weight"] = None
        else:
            best_params["class_weight"] = "balanced"

    best_clf = DecisionTreeClassifier(random_state=42, **best_params)
    best_clf.fit(x_train, y_train)

    best_thresh = 0.5
    if args.binary_target:
        val_probs = best_clf.predict_proba(x_val)[:, 1]
        best_f1_val = 0.0
        for thresh in np.arange(0.1, 0.9, 0.05):
            y_val_pred = (val_probs > thresh).astype(int)
            f1_val = f1_score(y_val, y_val_pred, average="binary", zero_division=0)
            if f1_val > best_f1_val:
                best_f1_val = f1_val
                best_thresh = thresh
        logger.info("Best Threshold: %.2f (Val F1: %.4f)", best_thresh, best_f1_val)

    # Evaluate
    acc, hamm, f1, report = evaluate_model(
        best_clf, x_test, y_test, target_names, args.binary_target, best_thresh
    )

    logger.info("Test F1: %.4f", f1)

    # Save Paths
    model_path = os.path.join(args.model_dir, args.model_filename)
    metrics_path = os.path.join(args.model_dir, args.metrics_filename)
    imp_path = os.path.join(args.model_dir, imp_filename)

    if not os.path.exists(args.model_dir):
        os.makedirs(args.model_dir)

    # Save Model
    joblib.dump(best_clf, model_path)
    logger.info("Model saved to %s", model_path)

    # Calculate System Metrics
    size_kb, latency_us = get_system_metrics(best_clf, x_test, model_path)
    logger.info("Model Size: %.2f KB", size_kb)
    logger.info("Inference Time: %.2f µs/sample", latency_us)

    # 3. SAVE METRICS TEXT FILE (Updated)
    with open(metrics_path, "w", encoding="utf-8") as f:
        f.write(f"Mode: {'Binary' if args.binary_target else 'Multi-Label'}\n")
        f.write(f"Window Size: {args.window_size}\n")
        f.write(f"Feature Set: {args.feature_set}\n")
        f.write(f"Best Params: {best_params}\n")
        f.write(f"Best CV Score: {study.best_value}\n")
        f.write(f"Best Threshold: {best_thresh:.2f}\n")
        f.write(f"Test Subset Accuracy: {acc:.4f}\n")
        f.write(f"Test Hamming Loss: {hamm:.4f}\n")
        f.write(f"Test F1 Score: {f1:.4f}\n")
        # --- NEW LINES ---
        f.write(f"Model Size (KB): {size_kb:.4f}\n")
        f.write(f"Inference Time (us/sample): {latency_us:.4f}\n")
        # -----------------
        f.write(f"\nClassification Report:\n{report}\n")

    # Save Feature Importance
    save_feature_importance(best_clf, feature_names, imp_path)


if __name__ == "__main__":
    main()
