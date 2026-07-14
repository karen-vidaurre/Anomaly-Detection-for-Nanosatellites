import kagglehub
kagglehub.login()

litzyconde3500_data_lstm_280326_path = kagglehub.dataset_download('litzyconde3500/data-lstm-280326')

print('Data source import complete.')

import os
import gc
import json
import logging
import sys
import time
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, optimizers, regularizers

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    classification_report, confusion_matrix,
    f1_score, accuracy_score, roc_auc_score,
    roc_curve, hamming_loss,
)

RUN_WINDOW_SIZE = 20
N_OPTUNA_TRIALS = 30
FINAL_EPOCHS    = 100
PATIENCE        = 15
SEED            = 42

DATA_BASE_DIR   = "/kaggle/input/datasets/litzyconde3500/data-lstm-280326"
OUTPUT_BASE_DIR = f"/kaggle/working/conv_ae_results_ws{RUN_WINDOW_SIZE}"
os.makedirs(OUTPUT_BASE_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

os.environ["PYTHONHASHSEED"] = str(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

physical_devices = tf.config.list_physical_devices("GPU")
USE_GPU = len(physical_devices) > 0
if USE_GPU:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    logger.info(f"GPU: {physical_devices[0].name}")
else:
    logger.info("CPU mode")


def load_split(ws: int, split: str) -> tuple[np.ndarray, np.ndarray]:
    split_dir = os.path.join(DATA_BASE_DIR, f"ws{ws}", split)
    X = np.load(os.path.join(split_dir, "X.npy")).astype(np.float32)
    y = np.load(os.path.join(split_dir, "y_bin.npy")).astype(np.int8)
    logger.info(f"  [{split:5s}] X={X.shape}  anomalies={y.sum()} ({100*y.mean():.1f}%)")
    return X, y


def load_metadata(ws: int) -> dict:
    path = os.path.join(DATA_BASE_DIR, f"ws{ws}", "metadata.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


class ReconF1Callback(callbacks.Callback):
    
    def __init__(self, X_val: np.ndarray, y_val: np.ndarray, n_thresholds: int = 60):
        super().__init__()
        self.X_val = X_val
        self.y_val = y_val
        self.n_thresholds = n_thresholds

    def on_epoch_end(self, epoch, logs=None):
        recon = self.model.predict(self.X_val, verbose=0)
        err = np.mean(np.square(recon - self.X_val), axis=(1, 2))
        lo, hi = np.percentile(err, 1), np.percentile(err, 99)
        best_f1 = 0.0
        for t in np.linspace(lo, hi, self.n_thresholds):
            f1 = f1_score(self.y_val, (err > t).astype(int), zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
        logs["val_recon_f1"] = float(best_f1)
        if (epoch + 1) % 10 == 0:
            logger.info(f"    epoch {epoch+1:3d}  val_recon_f1={best_f1:.4f}")


def build_conv_autoencoder(
    input_shape: tuple,
    filters: int = 32,
    n_layers: int = 2,
    kernel_size: int = 5,
    latent_dim: int = 16,
    dropout: float = 0.2,
    lr: float = 1e-3,
    l2_reg: float = 1e-4,
) -> tf.keras.Model:
    inp = layers.Input(shape=input_shape, name="dwt_detail_input")
    x = inp

    #Encoder
    f = filters
    for i in range(n_layers):
        x = layers.Conv1D(
            f, kernel_size, padding="same", activation="relu",
            kernel_regularizer=regularizers.l2(l2_reg), name=f"enc_conv_{i}",
        )(x)
        x = layers.BatchNormalization(name=f"enc_bn_{i}")(x)
        x = layers.Dropout(dropout, name=f"enc_drop_{i}")(x)
        f = max(f // 2, 8)

    shape_before_flatten = x.shape[1:]
    flat_dim = int(shape_before_flatten[0] * shape_before_flatten[1])
    x = layers.Flatten(name="flatten")(x)
    latent = layers.Dense(latent_dim, activation="relu", name="latent")(x)

    #Decoder
    x = layers.Dense(flat_dim, activation="relu", name="dec_dense")(latent)
    x = layers.Reshape(shape_before_flatten, name="dec_reshape")(x)
    f = int(shape_before_flatten[-1])
    for i in range(n_layers):
        f = min(f * 2, filters)
        x = layers.Conv1D(
            f, kernel_size, padding="same", activation="relu",
            kernel_regularizer=regularizers.l2(l2_reg), name=f"dec_conv_{i}",
        )(x)
        x = layers.BatchNormalization(name=f"dec_bn_{i}")(x)
        x = layers.Dropout(dropout, name=f"dec_drop_{i}")(x)

    out = layers.Conv1D(
        input_shape[-1], kernel_size, padding="same", activation="linear",
        dtype="float32", name="reconstruction",
    )(x)

    model = models.Model(inp, out, name=f"conv_ae_ws{RUN_WINDOW_SIZE}")
    model.compile(
        optimizer=optimizers.Adam(learning_rate=lr, clipnorm=1.0),
        loss="mse",
    )
    return model

def make_objective(X_tr_normal, X_val, y_val):
    def objective(trial):
        filters     = trial.suggest_categorical("filters", [16, 32, 64])
        n_layers    = trial.suggest_int("n_layers", 1, 3)
        kernel_size = trial.suggest_categorical("kernel_size", [3, 5, 7])
        latent_dim  = trial.suggest_categorical("latent_dim", [8, 16, 32])
        dropout     = trial.suggest_float("dropout", 0.1, 0.4)
        lr          = trial.suggest_float("lr", 5e-4, 5e-3, log=True)
        l2_reg      = trial.suggest_float("l2_reg", 1e-5, 1e-3, log=True)
        batch_sz    = trial.suggest_categorical("batch_size", [64, 128, 256])

        model = build_conv_autoencoder(
            (X_tr_normal.shape[1], X_tr_normal.shape[2]),
            filters, n_layers, kernel_size, latent_dim, dropout, lr, l2_reg,
        )

        recon_f1_cb = ReconF1Callback(X_val, y_val)
        model.fit(
            X_tr_normal, X_tr_normal,
            validation_data=(X_val, X_val),
            epochs=25,
            batch_size=batch_sz,
            callbacks=[
                recon_f1_cb,
                callbacks.EarlyStopping(
                    monitor="val_recon_f1", mode="max",
                    patience=6, restore_best_weights=True,
                ),
            ],
            verbose=0,
        )

        recon = model.predict(X_val, verbose=0)
        err = np.mean(np.square(recon - X_val), axis=(1, 2))
        lo, hi = np.percentile(err, 1), np.percentile(err, 99)
        best_f1 = 0.0
        for t in np.linspace(lo, hi, 60):
            f1 = f1_score(y_val, (err > t).astype(int), zero_division=0)
            best_f1 = max(best_f1, f1)

        del model
        gc.collect()
        tf.keras.backend.clear_session()
        return best_f1

    return objective


def run_optuna(X_tr_normal, X_val, y_val) -> dict:
    db_path    = os.path.join(OUTPUT_BASE_DIR, f"optuna_conv_ae_ws{RUN_WINDOW_SIZE}.db")
    study_name = f"conv_ae_ws{RUN_WINDOW_SIZE}"
    storage    = f"sqlite:///{db_path}"

    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=SEED),
        load_if_exists=True,
    )

    completed = len([t for t in study.trials
                     if t.state == optuna.trial.TrialState.COMPLETE])
    remaining = N_OPTUNA_TRIALS - completed

    if remaining > 0:
        logger.info(f"Optuna: {completed} prevs, {remaining} rest")
        study.optimize(
            make_objective(X_tr_normal, X_val, y_val),
            n_trials=remaining,
            gc_after_trial=True,
        )

    logger.info(f"Best params: {study.best_params}")
    with open(os.path.join(OUTPUT_BASE_DIR, "best_params.json"), "w") as f:
        json.dump(study.best_params, f, indent=4)

    return study.best_params

def find_best_threshold(model, X_val, y_val, n_thresholds: int = 150) -> tuple[float, float, np.ndarray]:
    recon = model.predict(X_val, verbose=0)
    err = np.mean(np.square(recon - X_val), axis=(1, 2))
    lo, hi = np.percentile(err, 0.5), np.percentile(err, 99.5)
    best_t, best_f1 = float(np.median(err)), 0.0
    for t in np.linspace(lo, hi, n_thresholds):
        f1 = f1_score(y_val, (err > t).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, t
    logger.info(f"Optimal reconstruction threshold: {best_t:.6f}  val_F1={best_f1:.4f}")
    return float(best_t), float(best_f1), err


def export_tflite(model, X_sample: np.ndarray, out_path: str) -> float:
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    def representative_dataset():
        for i in range(min(200, len(X_sample))):
            yield [X_sample[i:i + 1]]

    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.float32
    converter.inference_output_type = tf.float32

    try:
        tflite_model = converter.convert()
        with open(out_path, "wb") as f:
            f.write(tflite_model)
        size_kb = os.path.getsize(out_path) / 1024
        logger.info(f"TFLite int8 saved: {out_path}  ({size_kb:.2f} KB)")
        return size_kb
    except Exception as e:
        logger.warning(f"TFLite int8 failed ({e}), try float16")
        converter2 = tf.lite.TFLiteConverter.from_keras_model(model)
        converter2.optimizations = [tf.lite.Optimize.DEFAULT]
        converter2.target_spec.supported_types = [tf.float16]
        tflite_model = converter2.convert()
        with open(out_path, "wb") as f:
            f.write(tflite_model)
        size_kb = os.path.getsize(out_path) / 1024
        logger.info(f"TFLite float16 saved: {out_path}  ({size_kb:.2f} KB)")
        return size_kb


def evaluate_tflite(tflite_path: str, X_test: np.ndarray,
                    y_test: np.ndarray, threshold: float) -> dict:
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()
    inp_idx = interpreter.get_input_details()[0]["index"]
    out_idx = interpreter.get_output_details()[0]["index"]

    errs = []
    for i in range(len(X_test)):
        interpreter.set_tensor(inp_idx, X_test[i:i + 1])
        interpreter.invoke()
        recon = interpreter.get_tensor(out_idx)[0]
        errs.append(np.mean(np.square(recon - X_test[i])))

    err = np.array(errs, dtype=np.float32)
    y_pred = (err > threshold).astype(int)

    return {
        "f1":       float(f1_score(y_test, y_pred, zero_division=0)),
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "auc":      float(roc_auc_score(y_test, err)),
    }

def plot_results(history_dict, y_test, y_pred, err_test, metrics, out_dir,
                 X_test, recon_test):
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(f"Conv1D Autoencoder — ws={RUN_WINDOW_SIZE} (input: DWT coef. D)", fontsize=13)

    if "loss" in history_dict:
        axes[0, 0].plot(history_dict["loss"], label="Train")
        axes[0, 0].plot(history_dict.get("val_loss", []), label="Val")
        axes[0, 0].set_title("Reconstruction loss (MSE)"); axes[0, 0].legend(); axes[0, 0].grid(True)

    if "val_recon_f1" in history_dict:
        axes[0, 1].plot(history_dict["val_recon_f1"], color="darkorange")
        axes[0, 1].set_title("Val recon-F1 per epoch"); axes[0, 1].grid(True)

    axes[0, 2].hist(err_test[y_test == 0], bins=40, alpha=0.6, label="Nominal", color="steelblue")
    axes[0, 2].hist(err_test[y_test == 1], bins=40, alpha=0.6, label="Anomaly", color="salmon")
    axes[0, 2].axvline(metrics["best_threshold"], color="k", ls="--",
                       label=f"thresh={metrics['best_threshold']:.4f}")
    axes[0, 2].set_title("Reconstruction error distribution"); axes[0, 2].legend(); axes[0, 2].grid(True)

    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", ax=axes[1, 0], cmap="Blues",
                xticklabels=["Nominal", "Anomaly"],
                yticklabels=["Nominal", "Anomaly"])
    axes[1, 0].set_title(f"Confusion matrix (thresh={metrics['best_threshold']:.4f})")

    fpr, tpr, _ = roc_curve(y_test, err_test)
    axes[1, 1].plot(fpr, tpr, label=f"AUC={metrics['test_auc']:.3f}")
    axes[1, 1].plot([0, 1], [0, 1], "k--", alpha=0.4)
    axes[1, 1].set_title("ROC (score = reconstruction error)"); axes[1, 1].legend(); axes[1, 1].grid(True)

    idx_norm = np.where(y_test == 0)[0]
    idx_anom = np.where(y_test == 1)[0]
    if len(idx_norm) > 0:
        i = idx_norm[0]
        axes[1, 2].plot(X_test[i, :, 0], label="Original (nominal)", color="steelblue")
        axes[1, 2].plot(recon_test[i, :, 0], "--", label="Reconstructed", color="steelblue", alpha=0.7)
    if len(idx_anom) > 0:
        j = idx_anom[0]
        axes[1, 2].plot(X_test[j, :, 0], label="Original (anomaly)", color="salmon")
        axes[1, 2].plot(recon_test[j, :, 0], "--", label="Reconstructed", color="salmon", alpha=0.7)
    axes[1, 2].set_title("Sample reconstruction (channel 0)")
    axes[1, 2].legend(fontsize=7); axes[1, 2].grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "training_plots.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved")


def main():
    ws = RUN_WINDOW_SIZE
    logger.info(f"CONV1D AUTOENCODER (binary, reconstruction-error) — window_size={ws}")
    logger.info(f"Input: DWT coefficients, detail D (no approximation)")

    ws_dir = os.path.join(DATA_BASE_DIR, f"ws{ws}")
    if not os.path.isdir(ws_dir):
        raise FileNotFoundError(f"No founded: {ws_dir}\n")

    meta = load_metadata(ws)
    logger.info(f"Metadata: {meta}")

    X_train, y_train = load_split(ws, "train")
    X_val,   y_val   = load_split(ws, "val")
    X_test,  y_test  = load_split(ws, "test")

    input_shape = (X_train.shape[1], X_train.shape[2])
    logger.info(f"input_shape = {input_shape}  (len_D, n_channels)")

    X_train_normal = X_train[y_train == 0]
    logger.info(f"Training on nominal-only subset: {X_train_normal.shape}")

    checkpoint_path = os.path.join(OUTPUT_BASE_DIR, "best_model.keras")
    tflite_path     = os.path.join(OUTPUT_BASE_DIR, "best_model_int8.tflite")
    training_flag   = os.path.join(OUTPUT_BASE_DIR, "training_done.json")
    history_path    = os.path.join(OUTPUT_BASE_DIR, "training_history.json")

    best_params = run_optuna(X_train_normal, X_val, y_val)
    gc.collect(); tf.keras.backend.clear_session()

    if os.path.exists(training_flag) and os.path.exists(checkpoint_path):
        best_model   = models.load_model(checkpoint_path)
        flag_data    = json.load(open(training_flag))
        elapsed      = flag_data.get("elapsed_sec", 0)
        history_dict = json.load(open(history_path)) if os.path.exists(history_path) else {}
    else:
        logger.info("Trainning")
        final_model = build_conv_autoencoder(
            input_shape,
            filters     = best_params["filters"],
            n_layers    = best_params["n_layers"],
            kernel_size = best_params["kernel_size"],
            latent_dim  = best_params["latent_dim"],
            dropout     = best_params["dropout"],
            lr          = best_params["lr"],
            l2_reg      = best_params["l2_reg"],
        )
        final_model.summary(print_fn=logger.info)

        recon_f1_cb = ReconF1Callback(X_val, y_val)

        t0 = time.time()
        hist_obj = final_model.fit(
            X_train_normal, X_train_normal,
            validation_data=(X_val, X_val),
            epochs=FINAL_EPOCHS,
            batch_size=best_params["batch_size"],
            callbacks=[
                recon_f1_cb,
                callbacks.ModelCheckpoint(
                    checkpoint_path,
                    monitor="val_recon_f1", mode="max",
                    save_best_only=True, verbose=0,
                ),
                callbacks.EarlyStopping(
                    monitor="val_recon_f1", mode="max",
                    patience=PATIENCE, restore_best_weights=True,
                ),
                callbacks.ReduceLROnPlateau(
                    monitor="val_loss", factor=0.5,
                    patience=6, min_lr=1e-6, verbose=0,
                ),
            ],
            verbose=1,
        )
        elapsed = time.time() - t0
        logger.info(f"Trainning: {elapsed / 60:.1f} min")

        history_dict = {k: [float(v) for v in vals]
                        for k, vals in hist_obj.history.items()}
        with open(history_path, "w") as f:
            json.dump(history_dict, f, indent=4)

        with open(training_flag, "w") as f:
            json.dump({
                "elapsed_sec": elapsed,
                "timestamp":   time.strftime("%Y-%m-%d %H:%M:%S"),
            }, f, indent=4)

        best_model = models.load_model(checkpoint_path)

    best_threshold, val_f1, _ = find_best_threshold(best_model, X_val, y_val)

    recon_test = best_model.predict(X_test, verbose=0)
    err_test = np.mean(np.square(recon_test - X_test), axis=(1, 2))
    y_pred_test = (err_test > best_threshold).astype(int)

    keras_size_kb = os.path.getsize(checkpoint_path) / 1024

    t_inf = time.time()
    _ = best_model.predict(X_test, verbose=0)
    inf_us = (time.time() - t_inf) * 1e6 / len(X_test)

    tflite_size_kb = export_tflite(
        best_model,
        X_sample=X_train_normal[:200],
        out_path=tflite_path,
    )

    tflite_metrics = evaluate_tflite(tflite_path, X_test, y_test, best_threshold)
    logger.info(f"TFLite F1={tflite_metrics['f1']:.4f}  "
                f"AUC={tflite_metrics['auc']:.4f}  "
                f"size={tflite_size_kb:.2f} KB")

    metrics = {
        "window_size": ws,
        "input": {
            "type": "DWT detail coef D",
            "shape": list(input_shape),
            "wavelet": "db4",
            "level": 1,
        },
        "approach":             "1D Conv Autoencoder, trained on nominal-only windows; "
                                 "anomaly score = mean squared reconstruction error",
        "best_params":          best_params,
        "best_threshold":       best_threshold,
        "val_f1_at_ckpt":       val_f1,
        "test_f1":              float(f1_score(y_test, y_pred_test, zero_division=0)),
        "test_accuracy":        float(accuracy_score(y_test, y_pred_test)),
        "test_auc":             float(roc_auc_score(y_test, err_test)),
        "test_hamming_loss":    float(hamming_loss(y_test, y_pred_test)),
        "keras_model_size_kb":  round(keras_size_kb, 4),
        "tflite_model_size_kb": round(tflite_size_kb, 4),
        "tflite_f1":            tflite_metrics["f1"],
        "tflite_auc":           tflite_metrics["auc"],
        "inference_time_us":    round(inf_us, 4),
        "train_time_min":       round(elapsed / 60, 2),
    }

    report = classification_report(
        y_test, y_pred_test,
        target_names=["Normal", "Anomaly"],
        digits=4,
    )
    logger.info(f"\nWindow Size: {ws}")
    logger.info(f"Best Params: {best_params}")
    logger.info(f"Best Threshold: {best_threshold:.6f}")
    logger.info(f"Test F1 Score: {metrics['test_f1']:.4f}")
    logger.info(f"Test Accuracy: {metrics['test_accuracy']:.4f}")
    logger.info(f"Test Hamming Loss: {metrics['test_hamming_loss']:.4f}")
    logger.info(f"Keras Model Size (KB): {keras_size_kb:.4f}")
    logger.info(f"TFLite int8 Size (KB): {tflite_size_kb:.4f}")
    logger.info(f"TFLite F1: {tflite_metrics['f1']:.4f}")
    logger.info(f"Inference Time (us/sample): {inf_us:.4f}")
    logger.info(f"\nClassification Report:\n{report}")

    with open(os.path.join(OUTPUT_BASE_DIR, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=4)
    with open(os.path.join(OUTPUT_BASE_DIR, "classification_report.txt"), "w") as f:
        f.write(f"Window Size: {ws}\n")
        f.write(f"Approach: {metrics['approach']}\n")
        f.write(f"Best Params: {best_params}\n")
        f.write(f"Best Threshold: {best_threshold:.6f}\n")
        f.write(f"Test F1: {metrics['test_f1']:.4f}\n")
        f.write(f"Keras size KB: {keras_size_kb:.4f}\n")
        f.write(f"TFLite int8 size KB: {tflite_size_kb:.4f}\n")
        f.write(f"TFLite F1: {tflite_metrics['f1']:.4f}\n\n")
        f.write(report)

    plot_results(
        history_dict, y_test, y_pred_test, err_test,
        metrics, OUTPUT_BASE_DIR, X_test, recon_test,
    )

    logger.info(f"\nSaved in: {OUTPUT_BASE_DIR}")


if __name__ == "__main__":
    main()
