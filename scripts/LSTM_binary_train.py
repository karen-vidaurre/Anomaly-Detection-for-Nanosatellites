import os
import gc
import json
import logging
import sys
import time
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import joblib
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
from sklearn.utils import class_weight as sk_class_weight

RUN_WINDOW_SIZE = 20
N_OPTUNA_TRIALS = 30
FINAL_EPOCHS    = 80
PATIENCE        = 12
SEED            = 42

DATA_BASE_DIR   = "/kaggle/input/datasets/litzyconde3500/data-lstm-280326"
OUTPUT_BASE_DIR = f"/kaggle/working/lstm_results_ws{RUN_WINDOW_SIZE}"
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

class ValF1Callback(tf.keras.callbacks.Callback):
    def __init__(self, X_val: np.ndarray, y_val: np.ndarray, threshold: float = 0.5):
        super().__init__()
        self.X_val     = X_val
        self.y_val     = y_val
        self.threshold = threshold

    def on_epoch_end(self, epoch, logs=None):
        y_prob = self.model.predict(self.X_val, verbose=0).ravel()
        y_pred = (y_prob > self.threshold).astype(int)
        f1 = f1_score(self.y_val, y_pred, average="binary", zero_division=0)
        logs["val_f1"] = float(f1)
        if (epoch + 1) % 10 == 0:
            logger.info(f"    epoch {epoch+1:3d}  val_f1={f1:.4f}")

def build_lstm(
    input_shape: tuple,
    n_units: int = 32,
    n_layers: int = 1,
    dropout: float = 0.3,
    lr: float = 1e-3,
    l2_reg: float = 1e-4,
) -> tf.keras.Model:
    inp = layers.Input(shape=input_shape, name="dwt_detail_input")
    x   = inp

    for i in range(n_layers):
        return_seq = (i < n_layers - 1)
        x = layers.LSTM(
            n_units,
            return_sequences=return_seq,
            kernel_regularizer=regularizers.l2(l2_reg),
            recurrent_regularizer=regularizers.l2(l2_reg / 2),
            name=f"lstm_{i}",
        )(x)
        x = layers.Dropout(dropout, name=f"drop_{i}")(x)
        if return_seq:
            x = layers.LayerNormalization(name=f"layernorm_{i}")(x)

    x   = layers.Dense(max(8, n_units // 2), activation="relu",
                       kernel_regularizer=regularizers.l2(l2_reg), name="dense_head")(x)
    x   = layers.Dropout(dropout / 2, name="drop_head")(x)
    out = layers.Dense(1, activation="sigmoid", dtype="float32", name="output")(x)

    model = models.Model(inp, out, name=f"lstm_ws{RUN_WINDOW_SIZE}")
    model.compile(
        optimizer=optimizers.Adam(learning_rate=lr, clipnorm=1.0),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")],
    )
    return model

def make_objective(X_tr, y_tr, X_val, y_val, cw_dict):
    def objective(trial):
        n_units  = trial.suggest_categorical("n_units",  [16, 32, 48, 64])
        n_layers = trial.suggest_int("n_layers", 1, 2)
        dropout  = trial.suggest_float("dropout", 0.1, 0.45)
        lr       = trial.suggest_float("lr", 5e-4, 5e-3, log=True)
        l2_reg   = trial.suggest_float("l2_reg", 1e-5, 1e-3, log=True)
        batch_sz = trial.suggest_categorical("batch_size", [64, 128, 256])

        model = build_lstm(
            (X_tr.shape[1], X_tr.shape[2]),
            n_units, n_layers, dropout, lr, l2_reg,
        )

        val_f1_cb = ValF1Callback(X_val, y_val)
        model.fit(
            X_tr, y_tr,
            validation_data=(X_val, y_val),
            epochs=25,
            batch_size=batch_sz,
            class_weight=cw_dict,
            callbacks=[
                val_f1_cb,
                callbacks.EarlyStopping(
                    monitor="val_f1", mode="max",
                    patience=6, restore_best_weights=True,
                ),
            ],
            verbose=0,
        )

        y_prob = model.predict(X_val, verbose=0).ravel()
        score  = f1_score(y_val, (y_prob > 0.5).astype(int),
                          average="binary", zero_division=0)
        del model
        gc.collect()
        tf.keras.backend.clear_session()
        return score

    return objective


def run_optuna(X_tr, y_tr, X_val, y_val, cw_dict) -> dict:
    db_path    = os.path.join(OUTPUT_BASE_DIR, f"optuna_ws{RUN_WINDOW_SIZE}.db")
    study_name = f"lstm_ws{RUN_WINDOW_SIZE}"
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
        logger.info(f"Optuna: {completed} prevs, {remaining} rest...")
        study.optimize(
            make_objective(X_tr, y_tr, X_val, y_val, cw_dict),
            n_trials=remaining,
            gc_after_trial=True,
        )

    logger.info(f"Best params: {study.best_params}")
    with open(os.path.join(OUTPUT_BASE_DIR, "best_params.json"), "w") as f:
        json.dump(study.best_params, f, indent=4)

    return study.best_params

def find_best_threshold(model, X_val, y_val) -> tuple[float, float]:
    y_prob = model.predict(X_val, verbose=0).ravel()
    best_thresh, best_f1 = 0.5, 0.0
    for t in np.arange(0.10, 0.91, 0.02):
        f1 = f1_score(y_val, (y_prob > t).astype(int),
                      average="binary", zero_division=0)
        if f1 > best_f1:
            best_f1, best_thresh = f1, t
    logger.info(f"Optimal threshold: {best_thresh:.2f}  val_F1={best_f1:.4f}")
    return float(best_thresh), float(best_f1)

def export_tflite(model, X_sample: np.ndarray, out_path: str) -> float:

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    def representative_dataset():
        for i in range(min(200, len(X_sample))):
            yield [X_sample[i:i+1]]

    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type  = tf.float32
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

    preds = []
    for i in range(len(X_test)):
        interpreter.set_tensor(inp_idx, X_test[i:i+1])
        interpreter.invoke()
        preds.append(interpreter.get_tensor(out_idx)[0, 0])

    y_prob = np.array(preds, dtype=np.float32)
    y_pred = (y_prob > threshold).astype(int)

    return {
        "f1":       float(f1_score(y_test, y_pred, zero_division=0)),
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "auc":      float(roc_auc_score(y_test, y_prob)),
    }

def plot_results(history_dict, y_test, y_pred, y_prob, metrics, out_dir):
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(f"LSTM Binary — ws={RUN_WINDOW_SIZE} (input: DWT coef. D)", fontsize=13)

    if "loss" in history_dict:
        axes[0, 0].plot(history_dict["loss"], label="Train")
        axes[0, 0].plot(history_dict.get("val_loss", []), label="Val")
        axes[0, 0].set_title("Loss"); axes[0, 0].legend(); axes[0, 0].grid(True)

    if "auc" in history_dict:
        axes[0, 1].plot(history_dict["auc"], label="Train AUC")
        axes[0, 1].plot(history_dict.get("val_auc", []), label="Val AUC")
        axes[0, 1].set_title("AUC"); axes[0, 1].legend(); axes[0, 1].grid(True)

    if "val_f1" in history_dict:
        axes[0, 2].plot(history_dict["val_f1"], color="darkorange")
        axes[0, 2].set_title("Val F1 per epoch"); axes[0, 2].grid(True)

    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", ax=axes[1, 0], cmap="Blues",
                xticklabels=["Nominal", "Anomaly"],
                yticklabels=["Nominal", "Anomaly"])
    axes[1, 0].set_title(f"Confusion matrix (thresh={metrics['best_threshold']:.2f})")

    fpr, tpr, _ = roc_curve(y_test, y_prob)
    axes[1, 1].plot(fpr, tpr, label=f"AUC={metrics['test_auc']:.3f}")
    axes[1, 1].plot([0, 1], [0, 1], "k--", alpha=0.4)
    axes[1, 1].set_title("ROC"); axes[1, 1].legend(); axes[1, 1].grid(True)

    axes[1, 2].hist(y_prob[y_test == 0], bins=40, alpha=0.6, label="Nominal",  color="steelblue")
    axes[1, 2].hist(y_prob[y_test == 1], bins=40, alpha=0.6, label="Anomaly", color="salmon")
    axes[1, 2].axvline(metrics["best_threshold"], color="k", ls="--",
                       label=f"thresh={metrics['best_threshold']:.2f}")
    axes[1, 2].set_title("Distribution P(anomaly)"); axes[1, 2].legend(); axes[1, 2].grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "training_plots.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved")

def main():
    ws = RUN_WINDOW_SIZE
    logger.info(f"LSTM BINARY — window_size={ws}")
    logger.info(f"Input: DWT coefficients, detail D (no approximation)")

    ws_dir = os.path.join(DATA_BASE_DIR, f"ws{ws}")
    if not os.path.isdir(ws_dir):
        raise FileNotFoundError(
            f"No founded: {ws_dir}\n"
        )

    meta = load_metadata(ws)
    logger.info(f"Metadata: {meta}")

    X_train, y_train = load_split(ws, "train")
    X_val,   y_val   = load_split(ws, "val")
    X_test,  y_test  = load_split(ws, "test")

    input_shape = (X_train.shape[1], X_train.shape[2])
    logger.info(f"input_shape = {input_shape}  (len_D, n_channels)")

    cw = sk_class_weight.compute_class_weight(
        "balanced", classes=np.unique(y_train), y=y_train
    )
    cw_dict = dict(zip(np.unique(y_train).tolist(), cw.tolist()))
    logger.info(f"class_weights: {cw_dict}")

    checkpoint_path  = os.path.join(OUTPUT_BASE_DIR, "best_model.keras")
    tflite_path      = os.path.join(OUTPUT_BASE_DIR, "best_model_int8.tflite")
    training_flag    = os.path.join(OUTPUT_BASE_DIR, "training_done.json")
    history_path     = os.path.join(OUTPUT_BASE_DIR, "training_history.json")

    best_params = run_optuna(X_train, y_train, X_val, y_val, cw_dict)
    gc.collect(); tf.keras.backend.clear_session()

    if os.path.exists(training_flag) and os.path.exists(checkpoint_path):
        best_model   = models.load_model(checkpoint_path)
        flag_data    = json.load(open(training_flag))
        elapsed      = flag_data.get("elapsed_sec", 0)
        val_f1_hist  = flag_data.get("val_f1_history", [])
        history_dict = json.load(open(history_path)) if os.path.exists(history_path) else {}
    else:
        logger.info("Trainning")
        final_model = build_lstm(
            input_shape,
            n_units  = best_params["n_units"],
            n_layers = best_params["n_layers"],
            dropout  = best_params["dropout"],
            lr       = best_params["lr"],
            l2_reg   = best_params["l2_reg"],
        )
        final_model.summary(print_fn=logger.info)

        val_f1_cb = ValF1Callback(X_val, y_val)

        t0 = time.time()
        hist_obj = final_model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=FINAL_EPOCHS,
            batch_size=best_params["batch_size"],
            class_weight=cw_dict,
            callbacks=[
                val_f1_cb,
                callbacks.ModelCheckpoint(
                    checkpoint_path,
                    monitor="val_f1", mode="max",
                    save_best_only=True, verbose=0,
                ),
                callbacks.EarlyStopping(
                    monitor="val_f1", mode="max",
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
        history_dict["val_f1"] = [float(v) for v in val_f1_cb.model_f1_history
                                   ] if hasattr(val_f1_cb, "model_f1_history") else []
        with open(history_path, "w") as f:
            json.dump(history_dict, f, indent=4)

        val_f1_hist = [float(v) for v in val_f1_cb.__dict__.get("val_f1_history",
                        hist_obj.history.get("val_f1", []))]

        with open(training_flag, "w") as f:
            json.dump({
                "elapsed_sec":     elapsed,
                "val_f1_history":  val_f1_hist,
                "timestamp":       time.strftime("%Y-%m-%d %H:%M:%S"),
            }, f, indent=4)

        best_model = models.load_model(checkpoint_path)
    best_threshold, val_f1 = find_best_threshold(best_model, X_val, y_val)

    y_prob_test = best_model.predict(X_test, verbose=0).ravel()
    y_pred_test = (y_prob_test > best_threshold).astype(int)

    keras_size_kb = os.path.getsize(checkpoint_path) / 1024

    t_inf = time.time()
    _ = best_model.predict(X_test, verbose=0)
    inf_us = (time.time() - t_inf) * 1e6 / len(X_test)

    tflite_size_kb = export_tflite(
        best_model,
        X_sample=X_train[:200],
        out_path=tflite_path,
    )

    tflite_metrics = evaluate_tflite(tflite_path, X_test, y_test, best_threshold)
    logger.info(f"TFLite F1={tflite_metrics['f1']:.4f}  "
                f"AUC={tflite_metrics['auc']:.4f}  "
                f"size={tflite_size_kb:.2f} KB")

    metrics = {
        "window_size":        ws,
        "input": {
            "type":   "DWT detail coef D",
            "shape":  list(input_shape),
            "wavelet": "db4",
            "level":   1,
        },
        "best_params":          best_params,
        "best_threshold":       best_threshold,
        "val_f1_at_ckpt":       val_f1,
        "test_f1":              float(f1_score(y_test, y_pred_test, zero_division=0)),
        "test_accuracy":        float(accuracy_score(y_test, y_pred_test)),
        "test_auc":             float(roc_auc_score(y_test, y_prob_test)),
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
    logger.info(f"Input type: DWT detail coefficients D (NOT approximation A)")
    logger.info(f"Best Params: {best_params}")
    logger.info(f"Best Threshold: {best_threshold:.2f}")
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
        f.write(f"Input: DWT detail coef D\n")
        f.write(f"Best Params: {best_params}\n")
        f.write(f"Best Threshold: {best_threshold:.2f}\n")
        f.write(f"Test F1: {metrics['test_f1']:.4f}\n")
        f.write(f"Keras size KB: {keras_size_kb:.4f}\n")
        f.write(f"TFLite int8 size KB: {tflite_size_kb:.4f}\n")
        f.write(f"TFLite F1: {tflite_metrics['f1']:.4f}\n\n")
        f.write(report)

    plot_results(
        history_dict, y_test, y_pred_test, y_prob_test,
        metrics, OUTPUT_BASE_DIR,
    )

    logger.info(f"\nSaved in: {OUTPUT_BASE_DIR}")

if __name__ == "__main__":
    main()

import os
import gc
import json
import time
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import tensorflow as tf
from tensorflow.keras import models

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    classification_report, confusion_matrix,
    f1_score, accuracy_score, roc_auc_score,
    roc_curve, hamming_loss, precision_score, recall_score,
)

RUN_WINDOW_SIZE  = 20
DATA_BASE_DIR    = "/kaggle/input/datasets/litzyconde3500/data-lstm-280326"
OUTPUT_BASE_DIR  = f"/kaggle/working/lstm_results_ws{RUN_WINDOW_SIZE}"
CHECKPOINT_PATH  = os.path.join(OUTPUT_BASE_DIR, "best_model.keras")
BEST_PARAMS_PATH = os.path.join(OUTPUT_BASE_DIR, "best_params.json")
HISTORY_PATH     = os.path.join(OUTPUT_BASE_DIR, "training_history.json")
TRAINING_FLAG    = os.path.join(OUTPUT_BASE_DIR, "training_done.json")
os.makedirs(OUTPUT_BASE_DIR, exist_ok=True)

def load_split(ws, split):
    d = os.path.join(DATA_BASE_DIR, f"ws{ws}", split)
    X = np.load(os.path.join(d, "X.npy")).astype(np.float32)
    y = np.load(os.path.join(d, "y_bin.npy")).astype(np.int8)
    print(f"  [{split:5s}] X={X.shape}  anomaly={y.sum()} ({100*y.mean():.1f}%)")
    return X, y


def find_best_threshold(model, X_val, y_val):
    y_prob = model.predict(X_val, verbose=0).ravel()
    best_t, best_f1 = 0.5, 0.0
    for t in np.arange(0.10, 0.91, 0.02):
        f1 = f1_score(y_val, (y_prob > t).astype(int), average="binary", zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, t
    print(f"  Optimal threshold: {best_t:.2f}  val_F1={best_f1:.4f}")
    return float(best_t), float(best_f1)


def plot_results(history_dict, y_test, y_pred, y_prob, metrics, out_dir):
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(
        f"LSTM Binary — ws={RUN_WINDOW_SIZE}  "
        f"(DWT coef. D + rolling_mean | F1={metrics['keras_f1']:.4f})",
        fontsize=13,
    )

    if "loss" in history_dict:
        axes[0, 0].plot(history_dict["loss"], label="Train")
        axes[0, 0].plot(history_dict.get("val_loss", []), label="Val")
        axes[0, 0].set_title("Loss"); axes[0, 0].legend(); axes[0, 0].grid(True)

    if "auc" in history_dict:
        axes[0, 1].plot(history_dict.get("auc", []),     label="Train AUC")
        axes[0, 1].plot(history_dict.get("val_auc", []), label="Val AUC")
        axes[0, 1].set_title("AUC"); axes[0, 1].legend(); axes[0, 1].grid(True)

    val_f1_hist = history_dict.get("val_f1", [])
    if val_f1_hist:
        axes[0, 2].plot(val_f1_hist, color="darkorange", label="Val F1")
        axes[0, 2].set_title("Val F1 per epoch")
        axes[0, 2].legend(); axes[0, 2].grid(True)

    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", ax=axes[1, 0], cmap="Blues",
                xticklabels=["Normal", "Anomaly"],
                yticklabels=["Normal", "Anomaly"])
    axes[1, 0].set_title(f"Confusion matrix (thresh={metrics['best_threshold']:.2f})")

    fpr, tpr, _ = roc_curve(y_test, y_prob)
    axes[1, 1].plot(fpr, tpr, label=f"AUC={metrics['keras_auc']:.4f}")
    axes[1, 1].plot([0, 1], [0, 1], "k--", alpha=0.4)
    axes[1, 1].set_title("ROC"); axes[1, 1].legend(); axes[1, 1].grid(True)

    axes[1, 2].hist(y_prob[y_test == 0], bins=40, alpha=0.6, label="Normal",  color="steelblue")
    axes[1, 2].hist(y_prob[y_test == 1], bins=40, alpha=0.6, label="Anomaly",   color="salmon")
    axes[1, 2].axvline(metrics["best_threshold"], color="k", ls="--",
                       label=f"thresh={metrics['best_threshold']:.2f}")
    axes[1, 2].set_title("Distribution P(anomaly)")
    axes[1, 2].legend(); axes[1, 2].grid(True)

    plt.tight_layout()
    path = os.path.join(out_dir, "training_plots.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Figura: {path}")

def main():
    ws = RUN_WINDOW_SIZE
    print(f"RECOVERY JUST METRICS — window_size={ws}")

    if not os.path.exists(CHECKPOINT_PATH):
        raise FileNotFoundError(f"No founded: {CHECKPOINT_PATH}")

    X_val,  y_val  = load_split(ws, "val")
    X_test, y_test = load_split(ws, "test")

    keras_model   = models.load_model(CHECKPOINT_PATH)
    keras_size_kb = os.path.getsize(CHECKPOINT_PATH) / 1024
    print(f"  Size: {keras_size_kb:.2f} KB")

    best_threshold, val_f1 = find_best_threshold(keras_model, X_val, y_val)

    y_prob = keras_model.predict(X_test, verbose=0).ravel()
    y_pred = (y_prob > best_threshold).astype(int)

    t0 = time.time()
    _ = keras_model.predict(X_test, verbose=0)
    inf_us = (time.time() - t0) * 1e6 / len(X_test)

    best_params = {}
    if os.path.exists(BEST_PARAMS_PATH):
        with open(BEST_PARAMS_PATH) as f:
            best_params = json.load(f)

    metrics = {
        "window_size":       ws,
        "input": {
            "type":    "DWT detail coef D + rolling_mean",
            "shape":   list(X_test.shape[1:]),
            "wavelet": "db4",
            "level":   1,
        },
        "best_params":       best_params,
        "best_threshold":    best_threshold,
        "val_f1_at_ckpt":    val_f1,
        "keras_f1":          float(f1_score(y_test, y_pred, zero_division=0)),
        "keras_precision":   float(precision_score(y_test, y_pred, zero_division=0)),
        "keras_recall":      float(recall_score(y_test, y_pred, zero_division=0)),
        "keras_accuracy":    float(accuracy_score(y_test, y_pred)),
        "keras_auc":         float(roc_auc_score(y_test, y_prob)),
        "keras_hamming":     float(hamming_loss(y_test, y_pred)),
        "keras_size_kb":     round(keras_size_kb, 4),
        "keras_inf_us":      round(inf_us, 4),
        "tflite_note":       "Not generated — GPU environment forces CudnnRNNV3 "
                             "which is incompatible with TFLite converter. "
                             "Future windows trained with CUDA_VISIBLE_DEVICES='' "
                             "will produce a TFLite-compatible model.",
    }

    report = classification_report(
        y_test, y_pred,
        target_names=["Normal", "Fault"],
        digits=4,
    )

    print(f"Window Size:      {ws}")
    print(f"Input type:       DWT detail coef D + rolling_mean")
    print(f"Best Params:      {best_params}")
    print(f"Best Threshold:   {best_threshold:.2f}")
    print(f"Test F1 Score:    {metrics['keras_f1']:.4f}")
    print(f"Test Precision:   {metrics['keras_precision']:.4f}")
    print(f"Test Recall:      {metrics['keras_recall']:.4f}")
    print(f"Test Accuracy:    {metrics['keras_accuracy']:.4f}")
    print(f"Test AUC:         {metrics['keras_auc']:.4f}")
    print(f"Test Hamming:     {metrics['keras_hamming']:.4f}")
    print(f"Model Size (KB):  {keras_size_kb:.4f}")
    print(f"Inference (µs):   {inf_us:.4f}")
    print(f"\nClassification Report:\n{report}")

    metrics_path = os.path.join(OUTPUT_BASE_DIR, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)

    report_path = os.path.join(OUTPUT_BASE_DIR, "classification_report.txt")
    with open(report_path, "w") as f:
        f.write(f"Window Size: {ws}\n")
        f.write(f"Input: DWT detail coef D + rolling_mean\n")
        f.write(f"Best Params: {best_params}\n")
        f.write(f"Best Threshold: {best_threshold:.2f}\n\n")
        f.write(f"Test F1 Score:    {metrics['keras_f1']:.4f}\n")
        f.write(f"Test Precision:   {metrics['keras_precision']:.4f}\n")
        f.write(f"Test Recall:      {metrics['keras_recall']:.4f}\n")
        f.write(f"Test Accuracy:    {metrics['keras_accuracy']:.4f}\n")
        f.write(f"Test AUC:         {metrics['keras_auc']:.4f}\n")
        f.write(f"Test Hamming:     {metrics['keras_hamming']:.4f}\n")
        f.write(f"Model Size (KB):  {keras_size_kb:.4f}\n")
        f.write(f"Inference (µs):   {inf_us:.4f}\n\n")
        f.write(report)
        f.write(f"\nTFLite note: {metrics['tflite_note']}\n")

    print(f"\nSaved: {metrics_path}")
    print(f"Saved: {report_path}")

    history_dict = {}
    if os.path.exists(HISTORY_PATH):
        with open(HISTORY_PATH) as f:
            history_dict = json.load(f)
    elif os.path.exists(TRAINING_FLAG):
        with open(TRAINING_FLAG) as f:
            flag = json.load(f)
        history_dict["val_f1"] = flag.get("val_f1_history", [])

    plot_results(history_dict, y_test, y_pred, y_prob, metrics, OUTPUT_BASE_DIR)

    print(f"\nFiles in {OUTPUT_BASE_DIR}:")
    for fname in sorted(os.listdir(OUTPUT_BASE_DIR)):
        fpath = os.path.join(OUTPUT_BASE_DIR, fname)
        if os.path.isfile(fpath):
            print(f"  {fname:45s}  {os.path.getsize(fpath)/1024:8.2f} KB")


main()