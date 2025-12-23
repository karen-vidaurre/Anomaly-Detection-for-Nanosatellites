import argparse
import logging
import os
import sys
import gc
from typing import Tuple, List, Dict, Optional, Any

import joblib
import numpy as np
import optuna
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit, cross_validate
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    f1_score,
    accuracy_score,
    hamming_loss,
    classification_report,
    make_scorer,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def parse_arguments() -> argparse.Namespace:
    """Parses command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train a Decision Tree model (Binary or Multi-label) with Optuna."
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="resources/raw/sim_001_full_overlaps.csv",
        help="Path to the input CSV data.",
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
        help="Window size. 1=Raw, >1=Rolling Variance/Mean.",
    )
    parser.add_argument(
        "--n_splits", type=int, default=5, help="Number of Time Series CV splits."
    )
    parser.add_argument(
        "--binary_target",
        action="store_true",
        help="If set, trains a binary classifier (0=Normal, 1=Any Fault) instead of multi-label.",
    )

    return parser.parse_args()


def load_data(data_path: str) -> pd.DataFrame:
    """Loads data from CSV."""
    if not os.path.exists(data_path):
        logger.error(f"Data file not found at {data_path}")
        sys.exit(1)

    logger.info(f"Loading data from {data_path}...")
    try:
        df = pd.read_csv(data_path)
        df.columns = [c.strip() for c in df.columns]

        float_cols = df.select_dtypes(include=["float64"]).columns
        if len(float_cols) > 0:
            df[float_cols] = df[float_cols].astype("float32")

        return df
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        sys.exit(1)


def save_data_subsets(
    X_train, X_test, y_train, y_test, output_dir, window_size, is_binary
):
    """
    Saves the train and test subsets to CSV files with proper naming.
    """
    os.makedirs(output_dir, exist_ok=True)

    mode_str = "binary" if is_binary else "multi"

    train_full = pd.concat([X_train, y_train], axis=1)
    test_full = pd.concat([X_test, y_test], axis=1)

    train_filename = f"train_split_w{window_size}_{mode_str}.csv"
    test_filename = f"test_split_w{window_size}_{mode_str}.csv"

    train_full.to_csv(os.path.join(output_dir, train_filename), index=False)
    test_full.to_csv(os.path.join(output_dir, test_filename), index=False)

    logger.info(
        f"Subconjuntos guardados en: {output_dir} ({train_filename}, {test_filename})"
    )


def preprocess_data(
    df: pd.DataFrame, window_size: int, binary_target: bool
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    """
    Separates features and targets. Handles Binary vs Multi-label logic.
    """
    MAX_WINDOW_SIZE = 20

    feature_prefixes = (
        "Vout_ss",
        "mout_mtq",
        "out_gps",
        "wout_gyro",
        "point_error",
        "Bout_magn",
    )
    raw_feature_cols = [
        c for c in df.columns if any(c.startswith(p) for p in feature_prefixes)
    ]
    target_cols = [c for c in df.columns if c.startswith("fault_")]

    if not raw_feature_cols:
        logger.error("No feature columns found based on prefixes.")
        sys.exit(1)
    if not target_cols:
        logger.error("No target columns found starting with 'fault_'.")
        sys.exit(1)

    # --- Feature Engineering ---
    features_list = []
    features_list.append(df[raw_feature_cols])

    # Delta
    delta_df = df[raw_feature_cols].diff().fillna(0).astype("float32")
    delta_df.columns = [f"{c}_delta" for c in raw_feature_cols]
    features_list.append(delta_df)

    if window_size > 1:
        # Rolling Variance
        var_df = (
            df[raw_feature_cols].rolling(window=window_size).var().astype("float32")
        )
        var_df.columns = [f"{c}_var_{window_size}" for c in raw_feature_cols]
        features_list.append(var_df)

        # Rolling Mean
        mean_df = (
            df[raw_feature_cols].rolling(window=window_size).mean().astype("float32")
        )
        mean_df.columns = [f"{c}_mean_{window_size}" for c in raw_feature_cols]
        features_list.append(mean_df)

    # Concatenate features
    X_full = pd.concat(features_list, axis=1)

    # Cleanup intermediate frames
    del features_list, delta_df
    if window_size > 1:
        del var_df, mean_df
    gc.collect()

    # Extract Targets
    y_raw = df[target_cols]

    if binary_target:
        logger.info("Transforming targets to BINARY mode (0=Normal, 1=Any Fault).")
        y_final = y_raw.max(axis=1).to_frame(name="any_fault")
        final_target_cols = ["any_fault"]
    else:
        logger.info("Keeping targets in MULTI-LABEL mode.")
        y_final = y_raw
        final_target_cols = target_cols

    # Combine X and y for consistent dropping
    data_combined = pd.concat([X_full, y_final], axis=1)

    # Cleanup X_full and y_final mostly to free reference, though data_combined holds data
    del X_full, y_final
    gc.collect()

    # Global Truncation
    logger.info(f"Applying Global Truncation: Dropping first {MAX_WINDOW_SIZE} rows.")
    data_combined = data_combined.iloc[MAX_WINDOW_SIZE:].reset_index(drop=True)

    before_drop = len(data_combined)
    data_combined = data_combined.dropna()
    after_drop = len(data_combined)

    if before_drop != after_drop:
        logger.info(f"Dropped {before_drop - after_drop} additional rows due to NaNs.")

    # Calculate number of feature columns: Total columns - number of target columns
    num_features = len(data_combined.columns) - len(final_target_cols)
    X = data_combined.iloc[:, :num_features].astype("float32")
    y = data_combined.iloc[:, num_features:]

    if binary_target:
        y = y["any_fault"]

    # Cleanup big combined df
    del data_combined
    gc.collect()

    logger.info(f"Features: {X.shape[1]}, Targets shape: {y.shape}")
    return X, y, final_target_cols


def get_objective(X: np.ndarray, y: np.ndarray, n_splits: int, binary_target: bool):
    """
    Returns the Optuna objective function with Time Series CV.
    """

    def objective(trial: optuna.Trial) -> float:
        # Search Space
        max_depth = trial.suggest_int("max_depth", 3, 20)
        min_samples_split = trial.suggest_int("min_samples_split", 2, 50)
        min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 20)
        criterion = trial.suggest_categorical("criterion", ["gini", "entropy"])
        class_weight = trial.suggest_categorical("class_weight", [None, "balanced"])

        clf = DecisionTreeClassifier(
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            criterion=criterion,
            class_weight=class_weight,
            random_state=42,
        )

        tscv = TimeSeriesSplit(n_splits=n_splits)

        if binary_target:
            scorer = make_scorer(f1_score, average="binary", zero_division=0)
        else:
            scorer = make_scorer(f1_score, average="macro", zero_division=0)

        scores = cross_validate(clf, X, y, cv=tscv, scoring=scorer, n_jobs=-1)
        mean_score = scores["test_score"].mean()

        return mean_score

    return objective


def evaluate_model(
    clf: DecisionTreeClassifier,
    X_test: np.ndarray,
    y_test: np.ndarray,
    target_cols: List[str],
    binary_target: bool,
) -> Tuple[float, float, float, str]:
    """
    Evaluates the model on the test set.
    """
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
        report = classification_report(
            y_test, y_pred, target_names=target_cols, zero_division=0
        )

    return subset_acc, hamming, f1_metric, report


def save_artifacts(
    model: DecisionTreeClassifier, metrics: dict, model_path: str, metrics_path: str
):
    """
    Saves the model and metrics to disk.
    """
    model_dir = os.path.dirname(model_path)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
        logger.info(f"Created directory {model_dir}")

    joblib.dump(model, model_path)
    logger.info(f"Model saved to {model_path}")

    with open(metrics_path, "w") as f:
        for key, value in metrics.items():
            f.write(f"{key}: {value}\n")
            if key == "Classification Report":
                f.write("\n")

    logger.info(f"Metrics saved to {metrics_path}")


def main():
    args = parse_arguments()

    mode_suffix = "binary" if args.binary_target else "multi"

    if args.model_filename is None:
        args.model_filename = f"dt_model_w{args.window_size}_{mode_suffix}.pkl"
    if args.metrics_filename is None:
        args.metrics_filename = f"dt_metrics_w{args.window_size}_{mode_suffix}.txt"

    # Paths
    model_path = os.path.join(args.model_dir, args.model_filename)
    metrics_path = os.path.join(args.model_dir, args.metrics_filename)

    # Load and Preprocess
    df = load_data(args.data_path)
    X, y, target_cols = preprocess_data(df, args.window_size, args.binary_target)

    # Cleanup original df
    del df
    gc.collect()

    logger.info("Splitting data (Shuffle=False for Time Series)...")

    # Method: 80% Train/Val (for CV), 20% Test (Holdout)
    split_idx = int(len(X) * 0.8)
    X_train_full = X.iloc[:split_idx]
    X_test = X.iloc[split_idx:]
    y_train_full = y.iloc[:split_idx]
    y_test = y.iloc[split_idx:]

    logger.info(f"Train+Val size: {len(X_train_full)}")
    logger.info(f"Test size:      {len(X_test)}")

    save_data_subsets(
        X_train_full,
        X_test,
        y_train_full,
        y_test,
        "data/processed/splits",
        args.window_size,
        args.binary_target,
    )

    # Convert to NumPy arrays for training (more efficient)
    logger.info("Converting datasets to NumPy arrays for optimization loop...")
    X_train_np = X_train_full.to_numpy(dtype=np.float32)
    y_train_np = y_train_full.to_numpy()  # Keep targets as inferred type (int/float)
    X_test_np = X_test.to_numpy(dtype=np.float32)
    y_test_np = y_test.to_numpy()

    # Delete DataFrame versions to free memory
    del X_train_full, X_test, y_train_full, y_test, X, y
    gc.collect()

    logger.info(
        f"Starting Optuna Optimization ({args.n_trials} trials, {args.n_splits}-split TSCV)..."
    )
    study = optuna.create_study(direction="maximize")
    objective = get_objective(X_train_np, y_train_np, args.n_splits, args.binary_target)
    study.optimize(objective, n_trials=args.n_trials)

    logger.info("\nBest Trial:")
    logger.info(study.best_trial.params)
    logger.info(f"Best CV Score: {study.best_value:.4f}")

    # Retrain best model on all available training data (Train + Val)
    logger.info("Retraining best model on full training set...")
    best_params = study.best_trial.params

    if best_params.get("class_weight") == "None":
        best_params["class_weight"] = None

    best_clf = DecisionTreeClassifier(random_state=42, **best_params)
    best_clf.fit(X_train_np, y_train_np)

    # Evaluate on final holdout Test set
    logger.info("Evaluating on Holdout Test Set...")
    subset_acc, hamming, f1_metric, report = evaluate_model(
        best_clf, X_test_np, y_test_np, target_cols, args.binary_target
    )

    logger.info(f"Test Subset Accuracy: {subset_acc:.4f}")
    logger.info(f"Test Hamming Loss:    {hamming:.4f}")
    logger.info(f"Test F1 Score:        {f1_metric:.4f}")

    metrics = {
        "Mode": "Binary" if args.binary_target else "Multi-label",
        "Window Size": args.window_size,
        "Best Params": best_params,
        "Best CV Score": study.best_value,
        "Test Subset Accuracy": f"{subset_acc:.4f}",
        "Test Hamming Loss": f"{hamming:.4f}",
        "Test F1 Score": f"{f1_metric:.4f}",
        "Classification Report": report,
    }

    save_artifacts(best_clf, metrics, model_path, metrics_path)


if __name__ == "__main__":
    main()
