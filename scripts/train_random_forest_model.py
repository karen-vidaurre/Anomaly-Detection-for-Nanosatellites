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
from sklearn.ensemble import RandomForestClassifier
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
    parser = argparse.ArgumentParser(description="Train a Random Forest model.")
    parser.add_argument("--data_dir", type=str, default="data/processed/tabular")
    parser.add_argument("--model_dir", type=str, default="models")
    parser.add_argument("--model_filename", type=str, default=None)
    parser.add_argument("--metrics_filename", type=str, default=None)
    parser.add_argument("--n_trials", type=int, default=30)
    parser.add_argument("--window_size", type=int, default=1)
    parser.add_argument("--binary_target", action="store_true")
    parser.add_argument("--normalize", action="store_true")
    parser.add_argument("--optimize_metric", type=str, default="f1", choices=["f1", "f2", "recall"])
    parser.add_argument("--feature_set", type=str, default="full", choices=["full", "stat_only"])
    parser.add_argument("--n_jobs", type=int, default=-1, help="Number of cores to use. Set to 1 if memory is an issue.")
    return parser.parse_args()


def load_datasets(
    data_dir: str, window_size: int, binary_target: bool, feature_set: str
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
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

    target_cols = [c for c in train_df.columns if c.startswith("fault_") or c == "any_fault"]
    all_cols = train_df.columns
    feature_cols = [c for c in all_cols if c not in target_cols and c != "Time"]

    if feature_set == "stat_only":
        accepted_suffixes = ["_delta", "_var", "_accel", "_std"]
        feature_cols = [col for col in feature_cols if any(s in col for s in accepted_suffixes)]
        if not feature_cols:
            logger.error("Feature set 'stat_only' resulted in 0 features!")
            sys.exit(1)

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
    optimize_metric: str,
    n_jobs: int,
) -> Callable:
    def objective(trial: optuna.Trial) -> float:
        n_estimators = trial.suggest_int("n_estimators", 20, 64)      
        max_depth = trial.suggest_int("max_depth", 8, 14)             
        min_samples_split = trial.suggest_int("min_samples_split", 20, 120) 
        min_samples_leaf = trial.suggest_int("min_samples_leaf", 10, 40)     
        
        max_leaf_nodes = trial.suggest_int("max_leaf_nodes", 80, 128) 
        max_samples = trial.suggest_float("max_samples", 0.6, 0.9)   
        class_weight_opt = trial.suggest_categorical("class_weight", ["balanced", "None"])
        cw = None if class_weight_opt == "None" else "balanced"

        clf = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_leaf_nodes=max_leaf_nodes,
            max_samples=max_samples,
            class_weight=cw,
            n_jobs=n_jobs,
            random_state=42,
        )

        clf.fit(x_train, y_train)
        y_pred = clf.predict(x_val)

        if binary_target:
            if optimize_metric == "f2":
                score = fbeta_score(y_val, y_pred, beta=2, average="binary", zero_division=0)
            elif optimize_metric == "recall":
                score = recall_score(y_val, y_pred, average="binary", zero_division=0)
            else:
                score = f1_score(y_val, y_pred, average="binary", zero_division=0)
        else:
            score = f1_score(y_val, y_pred, average="macro", zero_division=0)

        return score

    return objective


def evaluate_model(
    clf, x_test, y_test, target_names, binary_target
) -> Tuple[float, float, float, str]:
    y_pred = clf.predict(x_test)
    subset_acc = accuracy_score(y_test, y_pred)
    hamming = hamming_loss(y_test, y_pred)

    if binary_target:
        f1_metric = f1_score(y_test, y_pred, average="binary", zero_division=0)
        report = classification_report(
            y_test, y_pred, target_names=["Normal", "Fault"], zero_division=0
        )
    else:
        f1_metric = f1_score(y_test, y_pred, average="macro", zero_division=0)
        report = classification_report(
            y_test, y_pred, target_names=target_names, zero_division=0
        )

    return subset_acc, hamming, f1_metric, report


def save_feature_importance(model, feature_names, output_path) -> None:
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
    if not os.path.exists(model_path):
        return 0.0, 0.0

    size_bytes = os.path.getsize(model_path)
    size_kb = size_bytes / 1024

    _ = model.predict(x_test[:100])
    start_time = time.perf_counter()
    _ = model.predict(x_test)
    end_time = time.perf_counter()

    total_time = end_time - start_time
    latency_us = (total_time / len(x_test)) * 1e6

    return size_kb, latency_us


def main() -> None:
    args = parse_arguments()

    mode_suffix = "binary" if args.binary_target else "multi"
    if args.model_filename is None:
        args.model_filename = f"rf_model_w{args.window_size}_{mode_suffix}_{args.feature_set}.pkl"
    if args.metrics_filename is None:
        args.metrics_filename = f"rf_metrics_w{args.window_size}_{mode_suffix}_{args.feature_set}.txt"

    imp_filename = f"rf_importance_w{args.window_size}_{mode_suffix}_{args.feature_set}.csv"

    x_train_df, y_train_df, x_val_df, y_val_df, x_test_df, y_test_df, feature_names = (
        load_datasets(args.data_dir, args.window_size, args.binary_target, args.feature_set)
    )

    if args.binary_target:
        target_names = ["Normal", "Fault"]
    else:
        target_names = list(y_train_df.columns)

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

    if args.normalize:
        scaler = StandardScaler()
        x_train = scaler.fit_transform(x_train)
        x_val = scaler.transform(x_val)
        x_test = scaler.transform(x_test)

    logger.info("Starting Optuna (%d trials)...", args.n_trials)
    study = optuna.create_study(direction="maximize")
    objective = get_objective(
        x_train, y_train, x_val, y_val, args.binary_target, args.optimize_metric, args.n_jobs
    )
    study.optimize(objective, n_trials=args.n_trials)

    logger.info("Best Params: %s", study.best_trial.params)

    x_full = np.concatenate([x_train, x_val], axis=0)
    y_full = np.concatenate([y_train, y_val], axis=0)

    best_params = study.best_trial.params
    if "class_weight" in best_params:
        if best_params["class_weight"] == "None":
            best_params["class_weight"] = None
        else:
            best_params["class_weight"] = "balanced"

    best_clf = RandomForestClassifier(
        n_jobs=args.n_jobs, random_state=42, **best_params
    )
    best_clf.fit(x_full, y_full)

    acc, hamm, f1, report = evaluate_model(
        best_clf, x_test, y_test, target_names, args.binary_target
    )

    logger.info("Test F1: %.4f", f1)

    model_path = os.path.join(args.model_dir, args.model_filename)
    metrics_path = os.path.join(args.model_dir, args.metrics_filename)
    imp_path = os.path.join(args.model_dir, imp_filename)

    if not os.path.exists(args.model_dir):
        os.makedirs(args.model_dir)

    joblib.dump(best_clf, model_path)
    logger.info("Model saved to %s", model_path)

    size_kb, latency_us = get_system_metrics(best_clf, x_test, model_path)
    logger.info("Model Size: %.2f KB", size_kb)
    logger.info("Inference Time: %.2f µs/sample", latency_us)

    with open(metrics_path, "w", encoding="utf-8") as f:
        f.write(f"Mode: {'Binary' if args.binary_target else 'Multi-Label'}\n")
        f.write(f"Window Size: {args.window_size}\n")
        f.write(f"Feature Set: {args.feature_set}\n")
        f.write(f"Best Params: {best_params}\n")
        f.write(f"Best CV Score: {study.best_value}\n")
        f.write(f"Test Subset Accuracy: {acc:.4f}\n")
        f.write(f"Test Hamming Loss: {hamm:.4f}\n")
        f.write(f"Test F1 Score: {f1:.4f}\n")
        f.write(f"Model Size (KB): {size_kb:.4f}\n")
        f.write(f"Inference Time (us/sample): {latency_us:.4f}\n")
        f.write(f"\nClassification Report:\n{report}\n")

    save_feature_importance(best_clf, feature_names, imp_path)


if __name__ == "__main__":
    main()
