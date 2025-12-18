import argparse
import logging
import os
import sys
from typing import Tuple, List, Dict, Optional, Any

import joblib
import numpy as np
import optuna
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit, cross_validate, train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score, accuracy_score, hamming_loss, classification_report, make_scorer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def parse_arguments() -> argparse.Namespace:
    """Parses command line arguments."""
    parser = argparse.ArgumentParser(description="Train a Decision Tree model with Optuna and Time Series CV.")
    parser.add_argument('--data_path', type=str, default='data/raw/sim_001_full_overlaps.csv', help='Path to the input CSV data.')
    parser.add_argument('--model_dir', type=str, default='models', help='Directory to save the model and metrics.')
    parser.add_argument('--model_filename', type=str, default='decision_tree_model.pkl', help='Filename for the saved model.')
    parser.add_argument('--metrics_filename', type=str, default='decision_tree_metrics.txt', help='Filename for the metrics report.')
    parser.add_argument('--n_trials', type=int, default=50, help='Number of Optuna trials.')
    parser.add_argument('--n_splits', type=int, default=5, help='Number of Time Series CV splits.')
    
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
        return df
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        sys.exit(1)

def preprocess_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    """Separates features and targets."""
    feature_prefixes = ('Vout_ss', 'mout_mtq', 'out_gps', 'wout_gyro', 'point_error', 'Bout_magn')
    feature_cols = [c for c in df.columns if any(c.startswith(p) for p in feature_prefixes)]
    target_cols = [c for c in df.columns if c.startswith('fault_')]

    if not feature_cols:
        logger.error("No feature columns found based on prefixes.")
        sys.exit(1)
    if not target_cols:
        logger.error("No target columns found starting with 'fault_'.")
        sys.exit(1)

    X = df[feature_cols]
    y = df[target_cols]
    
    logger.info(f"Features: {len(feature_cols)}, Targets: {len(target_cols)}")
    return X, y, target_cols

def get_objective(X: pd.DataFrame, y: pd.DataFrame, n_splits: int):
    """Returns the Optuna objective function with Time Series CV."""
    
    def objective(trial: optuna.Trial) -> float:
        # Search Space
        max_depth = trial.suggest_int('max_depth', 3, 20)
        min_samples_split = trial.suggest_int('min_samples_split', 2, 50)
        min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 20)
        criterion = trial.suggest_categorical('criterion', ['gini', 'entropy'])
        class_weight = trial.suggest_categorical('class_weight', [None, 'balanced'])

        clf = DecisionTreeClassifier(
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            criterion=criterion,
            class_weight=class_weight,
            random_state=42
        )

        # Time Series Cross-Validation
        tscv = TimeSeriesSplit(n_splits=n_splits)
        
        # We optimize for F1 Macro
        scorer = make_scorer(f1_score, average='macro', zero_division=0)
        
        scores = cross_validate(clf, X, y, cv=tscv, scoring=scorer, n_jobs=-1)
        mean_score = scores['test_score'].mean()
        
        return mean_score

    return objective

def evaluate_model(clf: DecisionTreeClassifier, X_test: pd.DataFrame, y_test: pd.DataFrame, target_cols: List[str]) -> Tuple[float, float, float, str]:
    """Evaluates the model on the test set."""
    y_pred = clf.predict(X_test)
    
    subset_acc = accuracy_score(y_test, y_pred)
    hamming = hamming_loss(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average='macro', zero_division=0)
    report = classification_report(y_test, y_pred, target_names=target_cols, zero_division=0)
    
    return subset_acc, hamming, f1_macro, report

def save_artifacts(model: DecisionTreeClassifier, metrics: dict, model_path: str, metrics_path: str):
    """Saves the model and metrics to disk."""
    model_dir = os.path.dirname(model_path)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
        logger.info(f"Created directory {model_dir}")

    joblib.dump(model, model_path)
    logger.info(f"Model saved to {model_path}")

    with open(metrics_path, 'w') as f:
        for key, value in metrics.items():
            f.write(f"{key}: {value}\n")
            if key == "Classification Report":
                f.write("\n")
        
    logger.info(f"Metrics saved to {metrics_path}")

def main():
    args = parse_arguments()
    
    # Paths
    model_path = os.path.join(args.model_dir, args.model_filename)
    metrics_path = os.path.join(args.model_dir, args.metrics_filename)

    # Load and Preprocess
    df = load_data(args.data_path)
    X, y, target_cols = preprocess_data(df)

    logger.info("Splitting data (Shuffle=False for Time Series)...")
    
    # Method: 80% Train/Val (for CV), 20% Test (Holdout)
    split_idx = int(len(X) * 0.8)
    X_train_full, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train_full, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    logger.info(f"Train+Val size: {len(X_train_full)}")
    logger.info(f"Test size:      {len(X_test)}")

    logger.info(f"Starting Optuna Optimization ({args.n_trials} trials, {args.n_splits}-split TSCV)...")
    study = optuna.create_study(direction='maximize')
    objective = get_objective(X_train_full, y_train_full, args.n_splits)
    study.optimize(objective, n_trials=args.n_trials)

    logger.info("\nBest Trial:")
    logger.info(study.best_trial.params)
    logger.info(f"Best CV F1-Macro: {study.best_value:.4f}")

    # Retrain best model on all available training data (Train + Val)
    logger.info("Retraining best model on full training set...")
    best_params = study.best_trial.params
    
    # Handle None for categorical
    if best_params.get('class_weight') == 'None':
         best_params['class_weight'] = None

    best_clf = DecisionTreeClassifier(random_state=42, **best_params)
    best_clf.fit(X_train_full, y_train_full)

    # Evaluate on final holdout Test set
    logger.info("Evaluating on Holdout Test Set...")
    subset_acc, hamming, f1_macro, report = evaluate_model(best_clf, X_test, y_test, target_cols)

    logger.info(f"Test Subset Accuracy: {subset_acc:.4f}")
    logger.info(f"Test Hamming Loss:    {hamming:.4f}")
    logger.info(f"Test F1 Macro:        {f1_macro:.4f}")
    
    metrics = {
        "Best Params": best_params,
        "Best CV Score": study.best_value,
        "Test Subset Accuracy": f"{subset_acc:.4f}",
        "Test Hamming Loss": f"{hamming:.4f}",
        "Test F1 Macro": f"{f1_macro:.4f}",
        "Classification Report": report
    }

    save_artifacts(best_clf, metrics, model_path, metrics_path)

if __name__ == "__main__":
    main()
