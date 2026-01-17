import argparse
import logging
import os
import sys
from typing import Callable, Tuple, List
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
    # --- RESEARCH IMPROVEMENT: Feature Set Selection ---
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
    Loads train, validation, and test CSVs and applies feature filtering.
    """
    mode_str = "binary" if binary_target else "multi"
    base_path = os.path.join(data_dir, f"w{window_size}", mode_str)

    train_path = os.path.join(base_path, "train.csv")
    val_path = os.path.join(base_path, "val.csv")
    test_path = os.path.join(base_path, "test.csv")

    if not os.path.exists(train_path):
        logger.error(f"Datasets not found in {base_path}.")
        sys.exit(1)

    logger.info(f"Loading training data from {train_path}...")
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

    # --- RESEARCH IMPROVEMENT: Filter Features ---
    if feature_set == "stat_only":
        logger.info(
            "Applying 'stat_only' filter: Dropping Raw, Mean, Min, Max, Median."
        )
        accepted_suffixes = ["_delta", "_var", "_accel", "_std"]
        filtered_cols = []
        for col in feature_cols:
            if any(s in col for s in accepted_suffixes):
                filtered_cols.append(col)

        if len(filtered_cols) == 0:
            logger.error("Feature set 'stat_only' resulted in 0 features!")
            sys.exit(1)
        feature_cols = filtered_cols
        logger.info(f"Features reduced to {len(feature_cols)}.")

    # Apply Selection and Cast
    train_df = train_df[feature_cols + target_cols]  # Keep targets for splitting later
    val_df = val_df[feature_cols + target_cols]
    test_df = test_df[feature_cols + target_cols]

    # Convert features to float32
    train_df[feature_cols] = train_df[feature_cols].astype("float32")
    val_df[feature_cols] = val_df[feature_cols].astype("float32")
    test_df[feature_cols] = test_df[feature_cols].astype("float32")

    return train_df, val_df, test_df, feature_cols, target_cols


def get_objective(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    binary_target: bool,
    scale_pos_weight: float,
    optimize_metric: str = "f1",
) -> Callable:

    def objective(trial: optuna.Trial) -> float:
        # --- RESEARCH IMPROVEMENT: Tune Class Weighting Strategy ---
        # "balanced": Use calculated scale_pos_weight
        # "none": Use default (1.0)
        weight_strategy = trial.suggest_categorical(
            "weight_strategy", ["balanced", "none"]
        )

        current_scale_weight = (
            scale_pos_weight if weight_strategy == "balanced" else 1.0
        )

        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 500),
            "max_depth": trial.suggest_int("max_depth", 3, 15),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "gamma": trial.suggest_float("gamma", 0.0, 5.0),
            "n_jobs": -1,
            "random_state": 42,
            "tree_method": "hist",
        }

        if binary_target:
            params["objective"] = "binary:logistic"
            params["eval_metric"] = "logloss"
            params["scale_pos_weight"] = current_scale_weight
        else:
            params["objective"] = "multi:softprob"
            params["eval_metric"] = "mlogloss"
            # Scale pos weight not directly supported in multi:softprob via this API param
            # ignoring for multi-class for now

        clf = XGBClassifier(**params)
        clf.fit(X_train, y_train, verbose=False)
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


def save_feature_importance(model, feature_names, output_path):
    """
    Extracts and saves feature importance.
    """
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]

    with open(output_path, "w") as f:
        f.write("Rank,Feature,Importance\n")
        for i in range(len(feature_names)):
            idx = indices[i]
            if importances[idx] > 0:
                f.write(f"{i+1},{feature_names[idx]},{importances[idx]:.6f}\n")

    logger.info(f"Feature importance saved to {output_path}")


def main():
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
    X_train = train_df[feature_cols].to_numpy()
    y_train = train_df[target_cols].to_numpy()

    X_val = val_df[feature_cols].to_numpy()
    y_val = val_df[target_cols].to_numpy()

    X_test = test_df[feature_cols].to_numpy()
    y_test = test_df[target_cols].to_numpy()

    if args.binary_target:
        y_train = y_train.ravel()
        y_val = y_val.ravel()
        y_test = y_test.ravel()

    # Normalize if requested
    if args.normalize:
        logger.info("Normalizing features using StandardScaler...")
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)
        X_test = scaler.transform(X_test)

    # Calculate scale_pos_weight base
    base_scale_pos_weight = 1.0
    if args.binary_target:
        num_neg = np.sum(y_train == 0)
        num_pos = np.sum(y_train == 1)
        base_scale_pos_weight = num_neg / num_pos if num_pos > 0 else 1.0
        logger.info(f"Base Imbalance Ratio: {base_scale_pos_weight:.2f}")

    logger.info(
        f"Starting Optuna Optimization ({args.n_trials} trials)... Mode: {args.feature_set}"
    )

    objective = get_objective(
        X_train,
        y_train,
        X_val,
        y_val,
        args.binary_target,
        base_scale_pos_weight,
        args.optimize_metric,
    )
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=args.n_trials)

    logger.info(f"Best Params: {study.best_params}")

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

    X_train_full = np.vstack([X_train, X_val])
    y_train_full = np.concatenate([y_train, y_val])

    clf.fit(X_train_full, y_train_full, verbose=False)

    # Evaluate
    y_pred = clf.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    hamming = hamming_loss(y_test, y_pred)
    if args.binary_target:
        f1 = f1_score(y_test, y_pred, average="binary", zero_division=0)
        target_names = ["Normal", "Fault"]
    else:
        f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
        target_names = target_cols

    report = classification_report(
        y_test, y_pred, target_names=target_names, zero_division=0
    )

    logger.info(f"Test F1: {f1:.4f}")

    # Save Artifacts
    joblib.dump(clf, model_path)

    with open(metrics_path, "w") as f:
        f.write(f"Feature Set: {args.feature_set}\n")
        f.write(f"Weight Strategy: {weight_strategy}\n")
        f.write(f"Best Params: {best_params}\n")
        f.write(f"Test F1 Score: {f1:.4f}\n\n")
        f.write(report)

    # Save Feature Importance
    save_feature_importance(clf, feature_cols, imp_path)

    logger.info("Done.")


if __name__ == "__main__":
    main()
