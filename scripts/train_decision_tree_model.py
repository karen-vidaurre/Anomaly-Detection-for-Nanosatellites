import argparse
import logging
import os
import sys
import gc
from typing import Tuple, List, Dict, Optional, Any, Callable

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
    parser = argparse.ArgumentParser(description="Train a Decision Tree model.")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/processed/tabular",  # Updated default to match your new structure
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
    # --- NEW ARGUMENT FOR RESEARCH COMPARISON ---
    parser.add_argument(
        "--feature_set",
        type=str,
        default="full",
        choices=["full", "stat_only"],
        help="full: Use all features. stat_only: Drop raw/mean/min/max (emulate statistical model constraints).",
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
        logger.error(f"Datasets not found in {base_path}")
        sys.exit(1)

    logger.info(f"Loading data from {base_path}...")
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)

    # 1. Identify Targets
    target_cols = [
        c for c in train_df.columns if c.startswith("fault_") or c == "any_fault"
    ]

    # 2. Identify Features
    all_cols = train_df.columns
    feature_cols = [c for c in all_cols if c not in target_cols and c != "Time"]

    # --- RESEARCH FEATURE SELECTION ---
    if feature_set == "stat_only":
        logger.info(
            "Applying 'stat_only' filter: Dropping Raw, Mean, Min, Max, Median."
        )
        # We only keep columns that contain '_delta', '_var', '_accel'
        # OR columns that do NOT have the banned suffixes.

        # Banned: Raw sensors usually have no suffix, or specific ones like _mean
        # Strategy: Keep only those with accepted suffixes.
        accepted_suffixes = ["_delta", "_var", "_accel", "_std"]

        filtered_cols = []
        for col in feature_cols:
            if any(s in col for s in accepted_suffixes):
                filtered_cols.append(col)

        if len(filtered_cols) == 0:
            logger.error(
                "Feature set 'stat_only' resulted in 0 features! Check column names."
            )
            sys.exit(1)

        feature_cols = filtered_cols
        logger.info(
            f"Features reduced from {len(all_cols)-len(target_cols)} to {len(feature_cols)}."
        )

    # Apply selection
    X_train = train_df[feature_cols].astype("float32")
    X_val = val_df[feature_cols].astype("float32")
    X_test = test_df[feature_cols].astype("float32")

    y_train = train_df[target_cols]
    y_val = val_df[target_cols]
    y_test = test_df[target_cols]

    return X_train, y_train, X_val, y_val, X_test, y_test, feature_cols


def get_objective(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    binary_target: bool,
    optimize_metric: str = "f1",
) -> Callable:

    def objective(trial: optuna.Trial) -> float:
        # Search Space
        max_depth = trial.suggest_int("max_depth", 3, 30)
        min_samples_split = trial.suggest_int("min_samples_split", 2, 100)
        min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 50)
        criterion = trial.suggest_categorical("criterion", ["gini", "entropy"])

        # IMPROVEMENT: Tune class_weight
        class_weight_opt = trial.suggest_categorical(
            "class_weight", ["balanced", "None"]
        )
        cw = None if class_weight_opt == "None" else "balanced"

        clf = DecisionTreeClassifier(
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            criterion=criterion,
            class_weight=cw,
            random_state=42,
        )

        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_val)

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


def evaluate_model(clf, X_test, y_test, target_names, binary_target):
    y_pred = clf.predict(X_test)

    subset_acc = accuracy_score(y_test, y_pred)
    hamming = hamming_loss(y_test, y_pred)

    if binary_target:
        f1_metric = f1_score(y_test, y_pred, average="binary", zero_division=0)
        report = classification_report(
            y_test, y_pred, target_names=["Normal", "Fault"], zero_division=0
        )
    else:
        f1_metric = f1_score(y_test, y_pred, average="macro", zero_division=0)
        # Handle case where target_names length mismatches classes in y_test
        report = classification_report(
            y_test, y_pred, target_names=target_names, zero_division=0
        )

    return subset_acc, hamming, f1_metric, report


def save_feature_importance(model, feature_names, output_path):
    """
    Extracts and saves feature importance. Crucial for research analysis.
    """
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]

    with open(output_path, "w") as f:
        f.write("Rank,Feature,Importance\n")
        for i in range(len(feature_names)):
            idx = indices[i]
            # Only save non-zero features to save space/noise
            if importances[idx] > 0:
                f.write(f"{i+1},{feature_names[idx]},{importances[idx]:.6f}\n")

    logger.info(f"Feature importance saved to {output_path}")


def main():
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

    # Load Data (Pandas)
    X_train_df, y_train_df, X_val_df, y_val_df, X_test_df, y_test_df, feature_names = (
        load_datasets(
            args.data_dir, args.window_size, args.binary_target, args.feature_set
        )
    )

    # Get target names
    if args.binary_target:
        target_names = ["Normal", "Fault"]
    else:
        target_names = list(y_train_df.columns)

    # Convert to Numpy for Sklearn
    X_train = X_train_df.to_numpy()
    y_train = y_train_df.to_numpy()
    X_val = X_val_df.to_numpy()
    y_val = y_val_df.to_numpy()
    X_test = X_test_df.to_numpy()
    y_test = y_test_df.to_numpy()

    # Flatten y if binary
    if args.binary_target:
        y_train = y_train.ravel()
        y_val = y_val.ravel()
        y_test = y_test.ravel()

    # Normalize (Optional)
    if args.normalize:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)
        X_test = scaler.transform(X_test)

    # Optuna
    logger.info(f"Starting Optuna ({args.n_trials} trials)... Mode: {args.feature_set}")
    study = optuna.create_study(direction="maximize")
    objective = get_objective(
        X_train, y_train, X_val, y_val, args.binary_target, args.optimize_metric
    )
    study.optimize(objective, n_trials=args.n_trials)

    logger.info(f"Best Params: {study.best_trial.params}")

    # Retrain on Train + Val
    X_full = np.concatenate([X_train, X_val], axis=0)
    y_full = np.concatenate([y_train, y_val], axis=0)

    best_params = study.best_trial.params
    # Handle string conversion for class_weight
    if "class_weight" in best_params:
        if best_params["class_weight"] == "None":
            best_params["class_weight"] = None
        else:
            best_params["class_weight"] = "balanced"

    best_clf = DecisionTreeClassifier(random_state=42, **best_params)
    best_clf.fit(X_full, y_full)

    # Evaluate
    acc, hamm, f1, report = evaluate_model(
        best_clf, X_test, y_test, target_names, args.binary_target
    )

    logger.info(f"Test F1: {f1:.4f}")

    # Save
    model_path = os.path.join(args.model_dir, args.model_filename)
    metrics_path = os.path.join(args.model_dir, args.metrics_filename)
    imp_path = os.path.join(args.model_dir, imp_filename)

    # Save Model
    if not os.path.exists(args.model_dir):
        os.makedirs(args.model_dir)
    joblib.dump(best_clf, model_path)

    # Save Metrics
    with open(metrics_path, "w") as f:
        f.write(f"Feature Set: {args.feature_set}\n")
        f.write(f"Best Params: {best_params}\n")
        f.write(f"Test F1: {f1}\n")
        f.write(f"Test Accuracy: {acc}\n\n")
        f.write(report)

    # Save Feature Importance (Research Requirement)
    save_feature_importance(best_clf, feature_names, imp_path)


if __name__ == "__main__":
    main()
