import os
import gc
import json
import time
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, optimizers, regularizers
from tensorflow.keras import backend as K

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

from sklearn.metrics import (
    classification_report, f1_score, precision_score,
    recall_score, accuracy_score, hamming_loss,
)
from sklearn.utils import class_weight as sk_class_weight

import logging, sys
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

RUN_WINDOW_SIZE  = 10
N_OPTUNA_TRIALS  = 25
FINAL_EPOCHS     = 80
PATIENCE         = 12
SEED             = 42
THRESHOLD        = 0.5

FAULT_NAMES = [
    "fault_ss1", "fault_ss2", "fault_ss3", "fault_ss4", "fault_ss5", "fault_ss6",
    "fault_magn1", "fault_magn2", "fault_magn3",
    "fault_gyro1", "fault_gyro2", "fault_gyro3",
    "fault_mtq1", "fault_mtq2", "fault_mtq3",
]
N_LABELS = len(FAULT_NAMES)

DATA_BASE_DIR   = "/kaggle/input/datasets/litzyconde3500/data-lstm-280326"
OUTPUT_BASE_DIR = f"/kaggle/working/lstm_multi_results_ws{RUN_WINDOW_SIZE}"
os.makedirs(OUTPUT_BASE_DIR, exist_ok=True)

os.environ["PYTHONHASHSEED"] = str(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

gpus = tf.config.list_physical_devices("GPU")
if gpus:
    tf.config.experimental.set_memory_growth(gpus[0], True)
    logger.info(f"GPU : {gpus[0].name}")
else:
    logger.info("CPU used")

def load_split(ws, split):
    d = os.path.join(DATA_BASE_DIR, f"ws{ws}", split)
    X   = np.load(os.path.join(d, "X.npy")).astype(np.float32)
    yb  = np.load(os.path.join(d, "y_bin.npy")).astype(np.int8)
    ych = np.load(os.path.join(d, "y_ch.npy")).astype(np.float32)
    logger.info(
        f"  [{split:5s}] X={X.shape}  "
        f"anomalías={yb.sum()} ({100*yb.mean():.1f}%)  "
        f"y_ch={ych.shape}"
    )
    return X, yb, ych

class ValMacroF1Callback(tf.keras.callbacks.Callback):
    def __init__(self, X_val, y_val_ch, y_val_bin, threshold=0.5):
        super().__init__()
        self.X_val     = X_val
        self.y_val_ch  = y_val_ch
        self.y_val_bin = y_val_bin
        self.threshold = threshold
        self.history   = []

    def on_epoch_end(self, epoch, logs=None):
        y_prob = self.model.predict(self.X_val, verbose=0)
        y_pred = (y_prob > self.threshold).astype(int)

        mask = self.y_val_bin == 1
        if mask.sum() == 0:
            f1 = 0.0
        else:
            f1 = f1_score(
                self.y_val_ch[mask],
                y_pred[mask],
                average="macro", zero_division=0,
            )

        logs["val_macro_f1"] = float(f1)
        self.history.append(float(f1))
        if (epoch + 1) % 10 == 0:
            logger.info(f"    época {epoch+1:3d}  val_macro_f1={f1:.4f}")

def build_lstm_multilabel(input_shape, n_units=64, n_layers=1,
                           dropout=0.3, lr=1e-3, l2_reg=1e-4):

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

    x   = layers.Dense(
        max(16, n_units // 2), activation="relu",
        kernel_regularizer=regularizers.l2(l2_reg),
        name="dense_head",
    )(x)
    x   = layers.Dropout(dropout / 2, name="drop_head")(x)

    out = layers.Dense(N_LABELS, activation="sigmoid",
                       dtype="float32", name="output")(x)

    model = models.Model(inp, out, name=f"lstm_multi_ws{RUN_WINDOW_SIZE}")
    model.compile(
        optimizer=optimizers.Adam(learning_rate=lr, clipnorm=1.0),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model

def compute_sample_weights(y_ch):

    n_active = y_ch.sum(axis=1)
    weights  = np.where(n_active > 0, 2.0, 1.0)   # Anomalies carry more weight
    return weights.astype(np.float32)

def make_objective(X_tr, y_tr_ch, y_tr_bin,
                   X_val, y_val_ch, y_val_bin):
    def objective(trial):
        n_units  = trial.suggest_categorical("n_units",  [32, 64, 96])
        n_layers = trial.suggest_int("n_layers", 1, 2)
        dropout  = trial.suggest_float("dropout", 0.1, 0.45)
        lr       = trial.suggest_float("lr", 5e-4, 5e-3, log=True)
        l2_reg   = trial.suggest_float("l2_reg", 1e-5, 1e-3, log=True)
        batch_sz = trial.suggest_categorical("batch_size", [64, 128, 256])
        threshold = trial.suggest_float("threshold", 0.3, 0.7)

        model = build_lstm_multilabel(
            (X_tr.shape[1], X_tr.shape[2]),
            n_units, n_layers, dropout, lr, l2_reg,
        )
        sw = compute_sample_weights(y_tr_ch)
        val_f1_cb = ValMacroF1Callback(X_val, y_val_ch, y_val_bin, threshold)

        model.fit(
            X_tr, y_tr_ch,
            sample_weight=sw,
            validation_data=(X_val, y_val_ch),
            epochs=20,
            batch_size=batch_sz,
            callbacks=[
                val_f1_cb,
                callbacks.EarlyStopping(
                    monitor="val_macro_f1", mode="max",
                    patience=5, restore_best_weights=True,
                ),
            ],
            verbose=0,
        )

        y_prob = model.predict(X_val, verbose=0)
        y_pred = (y_prob > threshold).astype(int)
        mask   = y_val_bin == 1
        score  = f1_score(
            y_val_ch[mask], y_pred[mask],
            average="macro", zero_division=0,
        ) if mask.sum() > 0 else 0.0

        del model; gc.collect(); K.clear_session()
        return score

    return objective


def run_optuna(X_tr, y_tr_ch, y_tr_bin,
               X_val, y_val_ch, y_val_bin):
    db   = os.path.join(OUTPUT_BASE_DIR, f"optuna_multi_ws{RUN_WINDOW_SIZE}.db")
    name = f"lstm_multi_ws{RUN_WINDOW_SIZE}"
    study = optuna.create_study(
        study_name=name, storage=f"sqlite:///{db}",
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=SEED),
        load_if_exists=True,
    )
    done = len([t for t in study.trials
                if t.state == optuna.trial.TrialState.COMPLETE])
    rem  = N_OPTUNA_TRIALS - done
    if rem > 0:
        logger.info(f"Optuna: {done} previos, {rem} restantes...")
        study.optimize(
            make_objective(X_tr, y_tr_ch, y_tr_bin,
                           X_val, y_val_ch, y_val_bin),
            n_trials=rem, gc_after_trial=True,
        )
    logger.info(f"Mejores parámetros: {study.best_params}")
    with open(os.path.join(OUTPUT_BASE_DIR, "best_params.json"), "w") as f:
        json.dump(study.best_params, f, indent=4)
    return study.best_params

# Search for the optimal threshold per channel
def find_best_threshold_multi(model, X_val, y_val_ch, y_val_bin):
    y_prob = model.predict(X_val, verbose=0)
    mask   = y_val_bin == 1
    best_t, best_f1 = 0.5, 0.0
    for t in np.arange(0.10, 0.91, 0.02):
        y_pred = (y_prob > t).astype(int)
        f1 = f1_score(
            y_val_ch[mask], y_pred[mask],
            average="macro", zero_division=0,
        ) if mask.sum() > 0 else 0.0
        if f1 > best_f1:
            best_f1, best_t = f1, t
    logger.info(f"Threshold óptimo (multilabel): {best_t:.2f}  val_MacroF1={best_f1:.4f}")
    return float(best_t), float(best_f1)

def plot_results(history_dict, val_f1_hist,
                 y_test_ch, y_pred_anom, y_prob_anom,
                 metrics, out_dir):
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    ws = RUN_WINDOW_SIZE
    fig.suptitle(
        f"LSTM Multilabel — ws={ws}  "
        f"Macro-F1={metrics['macro_f1']:.4f}  "
        f"Micro-F1={metrics['micro_f1']:.4f}",
        fontsize=13,
    )

    # Loss
    if "loss" in history_dict:
        axes[0, 0].plot(history_dict["loss"],             label="Train")
        axes[0, 0].plot(history_dict.get("val_loss", []), label="Val")
        axes[0, 0].set_title("Loss"); axes[0, 0].legend(); axes[0, 0].grid(True)

    # Val Macro-F1
    if val_f1_hist:
        axes[0, 1].plot(val_f1_hist, color="darkorange", label="Val Macro-F1")
        axes[0, 1].set_title("Val Macro-F1 per epoch")
        axes[0, 1].legend(); axes[0, 1].grid(True)

    # F1 per channell
    f1_per_ch = [
        float(f1_score(y_test_ch[:, i], y_pred_anom[:, i], zero_division=0))
        for i in range(N_LABELS)
    ]
    short_names = [n.replace("fault_", "") for n in FAULT_NAMES]
    axes[0, 2].barh(short_names, f1_per_ch, color="steelblue", alpha=0.8)
    axes[0, 2].axvline(np.mean(f1_per_ch), color="red", ls="--",
                       label=f"macro={np.mean(f1_per_ch):.3f}")
    axes[0, 2].set_title("F1 per channel (in anomaly windows)")
    axes[0, 2].legend(); axes[0, 2].grid(True, axis="x")
    axes[0, 2].set_xlim(0, 1)

    # Heatmap
    cm_multi = np.zeros((N_LABELS, N_LABELS))
    for i in range(N_LABELS):
        for j in range(N_LABELS):
            cm_multi[i, j] = ((y_pred_anom[:, i] == 1) & (y_test_ch[:, j] == 1)).sum()
    sns.heatmap(cm_multi, ax=axes[1, 0], cmap="Blues",
                xticklabels=short_names, yticklabels=short_names,
                annot=False)
    axes[1, 0].set_title("Predicted vs. actual co-occurrence (anomalous windows)")
    axes[1, 0].tick_params(axis="x", rotation=90)
    axes[1, 0].tick_params(axis="y", rotation=0)

    # Precision vs Recall
    prec_ch = [float(precision_score(y_test_ch[:, i], y_pred_anom[:, i], zero_division=0))
               for i in range(N_LABELS)]
    rec_ch  = [float(recall_score(y_test_ch[:, i], y_pred_anom[:, i], zero_division=0))
               for i in range(N_LABELS)]
    axes[1, 1].scatter(prec_ch, rec_ch, c="steelblue", s=60, alpha=0.8)
    for i, name in enumerate(short_names):
        axes[1, 1].annotate(name, (prec_ch[i], rec_ch[i]),
                            fontsize=7, ha="left", va="bottom")
    axes[1, 1].set_xlabel("Precision"); axes[1, 1].set_ylabel("Recall")
    axes[1, 1].set_title("Precision vs. Recall by Channel")
    axes[1, 1].set_xlim(0, 1); axes[1, 1].set_ylim(0, 1)
    axes[1, 1].plot([0, 1], [0, 1], "k--", alpha=0.3)
    axes[1, 1].grid(True, alpha=0.4)

    # Average probability distribution
    mean_prob_anom = y_prob_anom[y_test_ch.sum(axis=1) > 0].mean(axis=0)
    mean_prob_nom  = y_prob_anom[y_test_ch.sum(axis=1) == 0].mean(axis=0) \
                     if (y_test_ch.sum(axis=1) == 0).any() else np.zeros(N_LABELS)
    x_pos = np.arange(N_LABELS)
    axes[1, 2].bar(x_pos - 0.2, mean_prob_anom, 0.4, label="Prob mean (actual anomaly)", color="salmon")
    axes[1, 2].bar(x_pos + 0.2, mean_prob_nom,  0.4, label="Average probability (nominal value)", color="steelblue")
    axes[1, 2].axhline(metrics["best_threshold"], color="k", ls="--",
                       label=f"threshold={metrics['best_threshold']:.2f}")
    axes[1, 2].set_xticks(x_pos); axes[1, 2].set_xticklabels(short_names, rotation=90, fontsize=7)
    axes[1, 2].set_title("Average probability per channel (in out-of-band test windows)")
    axes[1, 2].legend(fontsize=8); axes[1, 2].grid(True, axis="y")

    plt.tight_layout()
    path = os.path.join(out_dir, "training_plots_multi.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Figura: {path}")

def main():
    ws = RUN_WINDOW_SIZE
    logger.info(f"LSTM MULTILABEL (GPU) — window_size={ws}")
    logger.info(f"Output: {N_LABELS} canales — {FAULT_NAMES}")
    logger.info(f"Evaluation: only on anomalous frames (mask_anom)")

    ckpt_path    = os.path.join(OUTPUT_BASE_DIR, "best_model_multi.keras")
    flag_path    = os.path.join(OUTPUT_BASE_DIR, "training_done_multi.json")
    history_path = os.path.join(OUTPUT_BASE_DIR, "training_history_multi.json")

    X_train, y_bin_train, y_ch_train = load_split(ws, "train")
    X_val,   y_bin_val,   y_ch_val   = load_split(ws, "val")
    X_test,  y_bin_test,  y_ch_test  = load_split(ws, "test")

    input_shape = (X_train.shape[1], X_train.shape[2])
    logger.info(f"input_shape = {input_shape}")
    logger.info(f"y_ch_train shape = {y_ch_train.shape}")

    assert y_ch_train.shape[1] == N_LABELS, \
        f"y_ch has {y_ch_train.shape[1]} columns, wait for {N_LABELS}"

    logger.info("Distribution of faults by channel (train):")
    for i, name in enumerate(FAULT_NAMES):
        n = int(y_ch_train[:, i].sum())
        logger.info(f"  {name:15s}: {n:6d} ({100*n/len(y_ch_train):.1f}%)")

    # 2. Sample weights
    sw_train = compute_sample_weights(y_ch_train)
    logger.info(f"Sample weights: {sw_train[sw_train > 1].sum():.0f} weighted windows 2.0")

    # 3. Optuna
    best_params = run_optuna(
        X_train, y_ch_train, y_bin_train,
        X_val,   y_ch_val,   y_bin_val,
    )
    gc.collect(); K.clear_session()

    # 4. Train
    if os.path.exists(flag_path) and os.path.exists(ckpt_path):
        best_model   = models.load_model(ckpt_path)
        flag_data    = json.load(open(flag_path))
        elapsed      = flag_data.get("elapsed_sec", 0)
        val_f1_hist  = flag_data.get("val_f1_history", [])
        history_dict = json.load(open(history_path)) if os.path.exists(history_path) else {}
    else:
        best_threshold_optuna = best_params.pop("threshold", 0.5)

        final_model = build_lstm_multilabel(
            input_shape,
            n_units  = best_params["n_units"],
            n_layers = best_params["n_layers"],
            dropout  = best_params["dropout"],
            lr       = best_params["lr"],
            l2_reg   = best_params["l2_reg"],
        )
        final_model.summary(print_fn=logger.info)

        val_f1_cb = ValMacroF1Callback(
            X_val, y_ch_val, y_bin_val,
            threshold=best_threshold_optuna,
        )

        t0 = time.time()
        hist_obj = final_model.fit(
            X_train, y_ch_train,
            sample_weight=sw_train,
            validation_data=(X_val, y_ch_val),
            epochs=FINAL_EPOCHS,
            batch_size=best_params["batch_size"],
            callbacks=[
                val_f1_cb,
                callbacks.ModelCheckpoint(
                    ckpt_path,
                    monitor="val_macro_f1", mode="max",
                    save_best_only=True, verbose=0,
                ),
                callbacks.EarlyStopping(
                    monitor="val_macro_f1", mode="max",
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
        logger.info(f"Trainning: {elapsed/60:.1f} min")

        history_dict = {k: [float(v) for v in vals]
                        for k, vals in hist_obj.history.items()}
        val_f1_hist  = [float(v) for v in val_f1_cb.history]
        history_dict["val_macro_f1"] = val_f1_hist

        with open(history_path, "w") as f:
            json.dump(history_dict, f, indent=4)
        with open(flag_path, "w") as f:
            json.dump({
                "elapsed_sec":    elapsed,
                "val_f1_history": val_f1_hist,
                "timestamp":      time.strftime("%Y-%m-%d %H:%M:%S"),
            }, f, indent=4)

        best_model = models.load_model(ckpt_path)

    #5. Optimal threshold in validation
    best_threshold, val_macro_f1 = find_best_threshold_multi(
        best_model, X_val, y_ch_val, y_bin_val
    )

    # 6. Test
    y_prob_test = best_model.predict(X_test, verbose=0)
    y_pred_test = (y_prob_test > best_threshold).astype(int)

    mask_anom = y_bin_test == 1
    logger.info(f"Abnormal windows in testing: {mask_anom.sum()} of {len(mask_anom)}")

    y_ch_anom   = y_ch_test[mask_anom]
    y_pred_anom = y_pred_test[mask_anom]
    y_prob_anom = y_prob_test[mask_anom]

    # Inference
    t0 = time.time()
    _ = best_model.predict(X_test, verbose=0)
    inf_us = (time.time() - t0) * 1e6 / len(X_test)
    keras_size_kb = os.path.getsize(ckpt_path) / 1024

    # Metrics on anomalous windows
    macro_f1  = float(f1_score(y_ch_anom, y_pred_anom, average="macro",    zero_division=0))
    micro_f1  = float(f1_score(y_ch_anom, y_pred_anom, average="micro",    zero_division=0))
    macro_pre = float(precision_score(y_ch_anom, y_pred_anom, average="macro", zero_division=0))
    macro_rec = float(recall_score(y_ch_anom, y_pred_anom, average="macro",    zero_division=0))
    h_loss    = float(hamming_loss(y_ch_anom, y_pred_anom))

    report = classification_report(
        y_ch_anom, y_pred_anom,
        target_names=FAULT_NAMES, digits=4,
    )

    metrics = {
        "window_size":      ws,
        "input": {
            "type":    "DWT detail coef D + rolling_mean",
            "shape":   list(input_shape),
            "wavelet": "db4", "level": 1,
        },
        "evaluation":       "Only on anomalous windows (mask_anom = y_bin==1)",
        "best_params":      best_params,
        "best_threshold":   best_threshold,
        "val_macro_f1":     val_macro_f1,
        "macro_f1":         macro_f1,
        "micro_f1":         micro_f1,
        "macro_precision":  macro_pre,
        "macro_recall":     macro_rec,
        "hamming_loss":     h_loss,
        "model_size_kb":    round(keras_size_kb, 4),
        "inference_us":     round(inf_us, 4),
        "train_time_min":   round(elapsed / 60, 2) if not isinstance(elapsed, dict) else 0,
        "n_anomalous_test": int(mask_anom.sum()),
        "n_total_test":     int(len(mask_anom)),
    }

    logger.info(f"Window Size:        {ws}")
    logger.info(f"Evaluation:         only anomalous windows ({mask_anom.sum()} samples)")
    logger.info(f"Best Params:        {best_params}")
    logger.info(f"Best Threshold:     {best_threshold:.2f}")
    logger.info(f"Macro-Precision:    {macro_pre:.4f}")
    logger.info(f"Macro-Recall:       {macro_rec:.4f}")
    logger.info(f"Macro-F1:           {macro_f1:.4f}")
    logger.info(f"Micro-F1:           {micro_f1:.4f}")
    logger.info(f"Hamming Loss:       {h_loss:.4f}")
    logger.info(f"Model Size (KB):    {keras_size_kb:.4f}")
    logger.info(f"Inference (µs):     {inf_us:.4f}")
    logger.info(f"\nClassification Report (anomalous windows):\n{report}")

    with open(os.path.join(OUTPUT_BASE_DIR, "metrics_multi.json"), "w") as f:
        json.dump(metrics, f, indent=4)

    with open(os.path.join(OUTPUT_BASE_DIR, "classification_report_multi.txt"), "w") as f:
        f.write(f"Window Size: {ws}\n")
        f.write(f"Evaluation: ONLY anomalous windows (mask_anom)\n")
        f.write(f"Anomalous windows in test: {mask_anom.sum()} / {len(mask_anom)}\n")
        f.write(f"Best Params: {best_params}\n")
        f.write(f"Best Threshold: {best_threshold:.2f}\n\n")
        f.write(f"Macro-Precision: {macro_pre:.4f}\n")
        f.write(f"Macro-Recall:    {macro_rec:.4f}\n")
        f.write(f"Macro-F1:        {macro_f1:.4f}\n")
        f.write(f"Micro-F1:        {micro_f1:.4f}\n")
        f.write(f"Hamming Loss:    {h_loss:.4f}\n")
        f.write(f"Model Size KB:   {keras_size_kb:.4f}\n")
        f.write(f"Inference µs:    {inf_us:.4f}\n\n")
        f.write(report)

    plot_results(
        history_dict, val_f1_hist,
        y_ch_anom, y_pred_anom, y_prob_anom,
        metrics, OUTPUT_BASE_DIR,
    )

    logger.info(f"\Files in {OUTPUT_BASE_DIR}:")
    for fname in sorted(os.listdir(OUTPUT_BASE_DIR)):
        fpath = os.path.join(OUTPUT_BASE_DIR, fname)
        if os.path.isfile(fpath):
            logger.info(f"  {fname:50s}  {os.path.getsize(fpath)/1024:8.2f} KB")

if __name__ == "__main__":
    main()