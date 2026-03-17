import argparse
import gc
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
from sklearn.multioutput import MultiOutputClassifier
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
    parser.add_argument("--optuna_subsample", type=float, default=1.0,
                        help="Fraction of training data to use during Optuna search (e.g. 0.25). Final model always trains on full data.")
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
    feature_cols = [
        c for c in all_cols
        if c not in target_cols and c != "Time" and not c.startswith("point_error")
    ]

    if feature_set == "stat_only":
        STAT_SUFFIXES = (
            "_delta", "_accel",
            "_var_", "_std_", "_kurt_", "_skew_", "_range_", "_dev_",
            "_mean_", "_min_", "_max_", "_median_",
        )
        feature_cols = [col for col in feature_cols if any(s in col for s in STAT_SUFFIXES)]
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

        base_clf = RandomForestClassifier(
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

        if binary_target:
            clf = base_clf
        else:
            clf = MultiOutputClassifier(base_clf, n_jobs=1)

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

        del clf
        gc.collect()
        return score

    return objective


def evaluate_model(
    clf, x_test, y_test, target_names, binary_target, threshold=0.5
) -> Tuple[float, float, float, str]:
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
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "estimators_"):
        importances = np.mean(
            [est.feature_importances_ for est in model.estimators_], axis=0
        )
    else:
        logger.warning("Model has no feature_importances_. Skipping.")
        return

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

    # Subsample training data for Optuna search to reduce memory per trial.
    # The final model is always retrained on the full dataset.
    if args.optuna_subsample < 1.0:
        n_sub = max(1, int(len(x_train) * args.optuna_subsample))
        rng = np.random.default_rng(42)
        idx = rng.choice(len(x_train), size=n_sub, replace=False)
        x_train_opt, y_train_opt = x_train[idx], y_train[idx]
        logger.info("Optuna subsample: %d / %d rows (%.0f%%)", n_sub, len(x_train), args.optuna_subsample * 100)
    else:
        x_train_opt, y_train_opt = x_train, y_train

    logger.info("Starting Optuna (%d trials)...", args.n_trials)
    study = optuna.create_study(direction="maximize")
    objective = get_objective(
        x_train_opt, y_train_opt, x_val, y_val, args.binary_target, args.optimize_metric, args.n_jobs
    )
    study.optimize(objective, n_trials=args.n_trials, gc_after_trial=True)

    del x_train_opt, y_train_opt
    gc.collect()

    logger.info("Best Params: %s", study.best_trial.params)

    best_params = study.best_trial.params
    if "class_weight" in best_params:
        if best_params["class_weight"] == "None":
            best_params["class_weight"] = None
        else:
            best_params["class_weight"] = "balanced"

    base_clf = RandomForestClassifier(
        n_jobs=args.n_jobs, random_state=42, **best_params
    )

    if args.binary_target:
        best_clf = base_clf
    else:
        best_clf = MultiOutputClassifier(base_clf, n_jobs=1)

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

    acc, hamm, f1, report = evaluate_model(
        best_clf, x_test, y_test, target_names, args.binary_target, best_thresh
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
        f.write(f"Best Threshold: {best_thresh:.2f}\n")
        f.write(f"Test Subset Accuracy: {acc:.4f}\n")
        f.write(f"Test Hamming Loss: {hamm:.4f}\n")
        f.write(f"Test F1 Score: {f1:.4f}\n")
        f.write(f"Model Size (KB): {size_kb:.4f}\n")
        f.write(f"Inference Time (us/sample): {latency_us:.4f}\n")
        f.write(f"\nClassification Report:\n{report}\n")

    save_feature_importance(best_clf, feature_names, imp_path)


if __name__ == "__main__":
    main()
