"""
This script trains a Conv1D-LSTM model on the processed 3D datasets.
"""

import argparse
import logging
import os
import sys
import json
import numpy as np
import optuna
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, optimizers, mixed_precision

# Enable Mixed Precision for GPU acceleration
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score
)
from sklearn.utils import class_weight

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a Conv1D-LSTM model.")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/processed/lstm/w20",
        help="Directory containing X_train.npy, y_train.npy, etc.",
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        default="models/convlstm",  # Changed to convlstm
        help="Directory to save the model and metrics.",
    )
    parser.add_argument(
        "--n_trials", 
        type=int, 
        default=25, 
        help="Number of Optuna trials."
    )
    parser.add_argument(
        "--epochs", 
        type=int, 
        default=30, 
        help="Number of epochs for final training."
    )
    parser.add_argument(
        "--batch_size_list",
        type=str,
        default="64,128",
        help="Comma-separated list of batch sizes to try.",
    )
    return parser.parse_args()


def load_data(data_dir: str):
    logger.info(f"Loading data from {data_dir}...")
    try:
        X_train = np.load(os.path.join(data_dir, "X_train.npy"))
        y_train = np.load(os.path.join(data_dir, "y_train.npy"))
        X_val = np.load(os.path.join(data_dir, "X_val.npy"))
        y_val = np.load(os.path.join(data_dir, "y_val.npy"))
        X_test = np.load(os.path.join(data_dir, "X_test.npy"))
        y_test = np.load(os.path.join(data_dir, "y_test.npy"))
        
        logger.info(f"X_train: {X_train.shape}, y_train: {y_train.shape}")
        return X_train, y_train, X_val, y_val, X_test, y_test
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        sys.exit(1)


def build_convlstm_model(
    input_shape,
    n_filters=64,
    kernel_size=3,
    n_units=64,
    n_layers=1,
    dropout=0.3,
    lr=1e-3
):
    """
    Builds the Conv1D + LSTM model using Keras.
    """
    model = models.Sequential()
    model.add(layers.Input(shape=input_shape))
    
    # Conv1D Layer for Feature Extraction
    model.add(layers.Conv1D(
        filters=n_filters, 
        kernel_size=kernel_size, 
        activation='relu', 
        padding='same'
    ))
    # Optional pooling to reduce dimensionality/noise
    model.add(layers.MaxPooling1D(pool_size=2))

    # LSTM Layers
    for i in range(n_layers):
        return_seq = i < (n_layers - 1)
        model.add(
            layers.LSTM(
                n_units,
                return_sequences=return_seq
            )
        )
        model.add(layers.Dropout(dropout))

    # Binary Classification Output
    model.add(layers.Dense(1, activation="sigmoid"))
    
    model.compile(
        optimizer=optimizers.Adam(learning_rate=lr),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    return model


class ValF1Callback(tf.keras.callbacks.Callback):
    """Computes binary F1 on the validation set at each epoch end and stores it as val_f1."""
    def __init__(self, X_val, y_val):
        super().__init__()
        self.X_val = X_val
        self.y_val = y_val

    def on_epoch_end(self, epoch, logs=None):
        y_pred = (self.model.predict(self.X_val, verbose=0) > 0.5).astype(int).ravel()
        logs['val_f1'] = f1_score(self.y_val, y_pred, average='macro', zero_division=0)


def get_objective(X_train, y_train, X_val, y_val, batch_sizes, class_weights):
    input_shape = (X_train.shape[1], X_train.shape[2])

    def objective(trial):
        # Search Space
        n_filters = trial.suggest_int("n_filters", 32, 128)
        kernel_size = trial.suggest_int("kernel_size", 3, 7)
        n_units = trial.suggest_int("n_units", 32, 128)
        n_layers = trial.suggest_int("n_layers", 1, 2)
        dropout = trial.suggest_float("dropout", 0.2, 0.5)
        lr = trial.suggest_float("lr", 1e-4, 5e-3, log=True)
        batch_size = trial.suggest_categorical("batch_size", batch_sizes)

        model = build_convlstm_model(
            input_shape=input_shape,
            n_filters=n_filters,
            kernel_size=kernel_size,
            n_units=n_units,
            n_layers=n_layers,
            dropout=dropout,
            lr=lr
        )

        model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=15,
            batch_size=batch_size,
            class_weight=class_weights,
            verbose=0,
            callbacks=[
                callbacks.EarlyStopping(monitor="val_loss", patience=5)
            ]
        )

        y_pred_prob = model.predict(X_val, verbose=0)
        y_pred = (y_pred_prob > 0.5).astype(int).ravel()

        f1 = f1_score(y_val, y_pred, average='macro', zero_division=0)
        return f1

    return objective


def evaluate_model(model, X_test, y_test, threshold=0.5):
    y_pred_prob = model.predict(X_test, verbose=0)
    y_pred = (y_pred_prob > threshold).astype(int).ravel()
    
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    report = classification_report(y_test, y_pred, target_names=["Nominal", "Anomaly"])
    cm = confusion_matrix(y_test, y_pred)
    
    return acc, f1, report, cm


def main():
    args = parse_arguments()
    
    SEED = 42
    np.random.seed(SEED)
    tf.random.set_seed(SEED)
    
    os.makedirs(args.model_dir, exist_ok=True)
    X_train, y_train, X_val, y_val, X_test, y_test = load_data(args.data_dir)
    
    logger.info("Computing class weights...")
    cw = class_weight.compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weights_dict = dict(zip(np.unique(y_train), cw))
    logger.info(f"Class Weights: {class_weights_dict}")

    try:
        batch_sizes = [int(x) for x in args.batch_size_list.split(",")]
    except ValueError:
        logger.error("Invalid batch_size_list")
        sys.exit(1)
        
    logger.info(f"Starting Optuna Optimization ({args.n_trials} trials)...")
    study = optuna.create_study(direction="maximize")
    objective = get_objective(X_train, y_train, X_val, y_val, batch_sizes, class_weights_dict)
    study.optimize(objective, n_trials=args.n_trials)
    
    best_params = study.best_params
    logger.info(f"Best Params: {best_params}")
    
    logger.info("Retraining best model...")
    input_shape = (X_train.shape[1], X_train.shape[2])
    
    final_model = build_convlstm_model(
        input_shape=input_shape,
        n_filters=best_params["n_filters"],
        kernel_size=best_params["kernel_size"],
        n_units=best_params["n_units"],
        n_layers=best_params["n_layers"],
        dropout=best_params["dropout"],
        lr=best_params["lr"]
    )
    
    checkpoint_path = os.path.join(args.model_dir, "best_convlstm_model.keras")
    checkpoint = callbacks.ModelCheckpoint(
        checkpoint_path,
        monitor="val_f1",
        mode="max",
        save_best_only=True,
        verbose=1
    )

    final_model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=args.epochs,
        batch_size=best_params["batch_size"],
        class_weight=class_weights_dict,
        callbacks=[ValF1Callback(X_val, y_val), checkpoint],
        verbose=1
    )
    
    logger.info(f"Loading best model from {checkpoint_path}...")
    best_model = models.load_model(checkpoint_path)
    
    logger.info("Optimizing Threshold for F1...")
    val_probs = best_model.predict(X_val, verbose=0)
    best_thresh = 0.5
    best_f1 = 0.0
    
    for thresh in np.arange(0.1, 0.9, 0.05):
        y_val_pred = (val_probs > thresh).astype(int).ravel()
        f1 = f1_score(y_val, y_val_pred, average='macro', zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = thresh
            
    logger.info(f"Best Threshold: {best_thresh:.2f} (Val F1: {best_f1:.4f})")

    logger.info("Evaluating on Test Set...")
    acc, f1, report, cm = evaluate_model(best_model, X_test, y_test, threshold=best_thresh)
    
    logger.info(f"Test Accuracy: {acc:.4f}")
    logger.info(f"Test F1 Score: {f1:.4f}")
    logger.info("\n" + report)
    
    import time
    start_time = time.perf_counter()
    _ = best_model.predict(X_test[:100], verbose=0)
    t0 = time.perf_counter()
    _ = best_model.predict(X_test, verbose=0)
    t1 = time.perf_counter()
    
    latency_us = ((t1 - t0) / len(X_test)) * 1e6
    model_size_kb = os.path.getsize(checkpoint_path) / 1024
    
    metrics_file = os.path.join(args.model_dir, "metrics.txt")
    with open(metrics_file, "w") as f:
        f.write(f"Best Params: {best_params}\n")
        f.write(f"Best Threshold: {best_thresh:.2f}\n")
        f.write(f"Test Accuracy: {acc:.4f}\n")
        f.write(f"Test F1 Score: {f1:.4f}\n")
        f.write(f"Model Size: {model_size_kb:.2f} KB\n")
        f.write(f"Inference Latency: {latency_us:.2f} us/sample\n")
        f.write("\nClassification Report:\n")
        f.write(report)
        f.write("\nConfusion Matrix:\n")
        f.write(str(cm))
        
    logger.info(f"Metrics saved to {metrics_file}")
    logger.info("Done.")

if __name__ == "__main__":
    main()
