import os, gc, json, time, warnings
warnings.filterwarnings("ignore")

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, optimizers, regularizers
from tensorflow.keras import backend as K
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

from sklearn.metrics import (
    classification_report, f1_score, precision_score,
    recall_score, hamming_loss,
)

import logging, sys
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

RUN_WINDOW_SIZE = 15
N_OPTUNA_TRIALS = 25
FINAL_EPOCHS    = 80
PATIENCE        = 12
SEED            = 42

FAULT_NAMES = [
    "fault_ss1", "fault_ss2", "fault_ss3", "fault_ss4", "fault_ss5", "fault_ss6",
    "fault_magn1", "fault_magn2", "fault_magn3",
    "fault_gyro1", "fault_gyro2", "fault_gyro3",
    "fault_mtq1",  "fault_mtq2",  "fault_mtq3",
]
N_LABELS = len(FAULT_NAMES)

DATA_BASE_DIR   = "/kaggle/input/datasets/litzyconde/data-lstm-280326"
OUTPUT_BASE_DIR = f"/kaggle/working/cnn_lstm_multi_results_ws{RUN_WINDOW_SIZE}"
os.makedirs(OUTPUT_BASE_DIR, exist_ok=True)

os.environ["PYTHONHASHSEED"] = str(SEED)
np.random.seed(SEED); tf.random.set_seed(SEED)

gpus = tf.config.list_physical_devices("GPU")
if gpus:
    tf.config.experimental.set_memory_growth(gpus[0], True)
    logger.info(f"GPU: {gpus[0].name}")

def load_split(ws, split):
    d = os.path.join(DATA_BASE_DIR, f"ws{ws}", split)
    X   = np.load(os.path.join(d, "X.npy")).astype(np.float32)
    yb  = np.load(os.path.join(d, "y_bin.npy")).astype(np.int8)
    ych = np.load(os.path.join(d, "y_ch.npy")).astype(np.float32)
    logger.info(f"  [{split:5s}] X={X.shape}  anom={yb.sum()} ({100*yb.mean():.1f}%)")
    return X, yb, ych

def sample_weights(y_ch):
    return np.where(y_ch.sum(axis=1) > 0, 2.0, 1.0).astype(np.float32)

class ValMacroF1(tf.keras.callbacks.Callback):
    def __init__(self, X_val, y_ch_val, y_bin_val, threshold=0.5):
        super().__init__()
        self.X = X_val; self.yc = y_ch_val; self.yb = y_bin_val
        self.thr = threshold; self.history = []

    def on_epoch_end(self, epoch, logs=None):
        prob = self.model.predict(self.X, verbose=0)
        pred = (prob > self.thr).astype(int)
        mask = self.yb == 1
        f1 = f1_score(self.yc[mask], pred[mask],
                      average="macro", zero_division=0) if mask.sum() > 0 else 0.0
        logs["val_macro_f1"] = float(f1); self.history.append(float(f1))
        if (epoch + 1) % 10 == 0:
            logger.info(f"    época {epoch+1:3d}  val_macro_f1={f1:.4f}")

def build_cnn_lstm(input_shape, n_filters_1=32, n_filters_2=64,
                   kernel_size=3, lstm_units=64, n_lstm_layers=1,
                   dropout=0.3, lr=1e-3, l2_reg=1e-4):

    inp = layers.Input(shape=input_shape, name="dwt_detail_input")

    # Block CNN 1
    x = layers.Conv1D(
        n_filters_1, kernel_size,
        padding="same", activation="relu",
        kernel_regularizer=regularizers.l2(l2_reg),
    )(inp)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout * 0.5)(x)

    # Block CNN 2
    x = layers.Conv1D(
        n_filters_2, kernel_size,
        padding="same", activation="relu",
        kernel_regularizer=regularizers.l2(l2_reg),
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout * 0.5)(x)

    # Block LSTM
    for i in range(n_lstm_layers):
        return_seq = (i < n_lstm_layers - 1)
        x = layers.LSTM(
            lstm_units,
            return_sequences=return_seq,
            kernel_regularizer=regularizers.l2(l2_reg),
            recurrent_regularizer=regularizers.l2(l2_reg / 2),
            name=f"lstm_{i}",
        )(x)
        x = layers.Dropout(dropout, name=f"drop_lstm_{i}")(x)

    # Head classification
    x   = layers.Dense(max(16, lstm_units // 2), activation="relu",
                       kernel_regularizer=regularizers.l2(l2_reg),
                       name="dense_head")(x)
    x   = layers.Dropout(dropout / 2)(x)
    out = layers.Dense(N_LABELS, activation="sigmoid",
                       dtype="float32", name="output")(x)

    model = models.Model(inp, out, name=f"cnn_lstm_ws{RUN_WINDOW_SIZE}")
    model.compile(
        optimizer=optimizers.Adam(learning_rate=lr, clipnorm=1.0),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model

def make_objective(X_tr, y_tr_ch, y_tr_bin, X_val, y_val_ch, y_val_bin):
    def objective(trial):
        n_filters_1  = trial.suggest_categorical("n_filters_1", [16, 32, 48])
        n_filters_2  = trial.suggest_categorical("n_filters_2", [32, 64, 96])
        kernel_size  = trial.suggest_categorical("kernel_size", [2, 3])
        lstm_units   = trial.suggest_categorical("lstm_units",  [32, 64, 96])
        n_lstm       = trial.suggest_int("n_lstm_layers",  1, 2)
        dropout      = trial.suggest_float("dropout",  0.1, 0.4)
        lr           = trial.suggest_float("lr",       5e-4, 5e-3, log=True)
        l2_reg       = trial.suggest_float("l2_reg",   1e-5, 1e-3, log=True)
        batch_sz     = trial.suggest_categorical("batch_size", [64, 128, 256])
        threshold    = trial.suggest_float("threshold", 0.3, 0.7)

        model = build_cnn_lstm(
            (X_tr.shape[1], X_tr.shape[2]),
            n_filters_1, n_filters_2, kernel_size, lstm_units, n_lstm,
            dropout, lr, l2_reg,
        )
        sw = sample_weights(y_tr_ch)
        cb = ValMacroF1(X_val, y_val_ch, y_val_bin, threshold)

        model.fit(
            X_tr, y_tr_ch, sample_weight=sw,
            validation_data=(X_val, y_val_ch),
            epochs=20, batch_size=batch_sz,
            callbacks=[
                cb,
                callbacks.EarlyStopping(monitor="val_macro_f1", mode="max",
                                        patience=5, restore_best_weights=True),
            ],
            verbose=0,
        )
        prob = model.predict(X_val, verbose=0)
        pred = (prob > threshold).astype(int)
        mask = y_val_bin == 1
        score = f1_score(y_val_ch[mask], pred[mask],
                         average="macro", zero_division=0) if mask.sum() > 0 else 0.0
        del model; gc.collect(); K.clear_session()
        return score
    return objective

def run_optuna(X_tr, y_tr_ch, y_tr_bin, X_val, y_val_ch, y_val_bin):
    db   = os.path.join(OUTPUT_BASE_DIR, f"optuna_cnn_lstm_ws{RUN_WINDOW_SIZE}.db")
    name = f"cnn_lstm_ws{RUN_WINDOW_SIZE}"
    study = optuna.create_study(
        study_name=name, storage=f"sqlite:///{db}",
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=SEED),
        load_if_exists=True,
    )
    done = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
    rem  = N_OPTUNA_TRIALS - done
    if rem > 0:
        logger.info(f"Optuna CNN+LSTM: {done} prevs, {rem} rest...")
        study.optimize(
            make_objective(X_tr, y_tr_ch, y_tr_bin, X_val, y_val_ch, y_val_bin),
            n_trials=rem, gc_after_trial=True,
        )
    logger.info(f"Best params: {study.best_params}")
    with open(os.path.join(OUTPUT_BASE_DIR, "best_params.json"), "w") as f:
        json.dump(study.best_params, f, indent=4)
    return study.best_params


def find_threshold(model, X_val, y_val_ch, y_val_bin):
    prob = model.predict(X_val, verbose=0)
    mask = y_val_bin == 1
    best_t, best_f1 = 0.5, 0.0
    for t in np.arange(0.10, 0.91, 0.02):
        pred = (prob > t).astype(int)
        f1 = f1_score(y_val_ch[mask], pred[mask],
                      average="macro", zero_division=0) if mask.sum() > 0 else 0.0
        if f1 > best_f1:
            best_f1, best_t = f1, t
    logger.info(f"Threshold óptimo: {best_t:.2f}  val_MacroF1={best_f1:.4f}")
    return float(best_t), float(best_f1)

def save_results(model, X_te, yc_te, yb_te, best_threshold,
                 best_params, histd, vf1h, keras_size_kb, elapsed, out_dir):
    mask   = yb_te == 1
    prob   = model.predict(X_te, verbose=0)
    pred   = (prob > best_threshold).astype(int)
    yca    = yc_te[mask]; pa = pred[mask]

    macro_f1  = float(f1_score(yca, pa, average="macro",    zero_division=0))
    micro_f1  = float(f1_score(yca, pa, average="micro",    zero_division=0))
    macro_pre = float(precision_score(yca, pa, average="macro", zero_division=0))
    macro_rec = float(recall_score(yca, pa,    average="macro", zero_division=0))
    h_loss    = float(hamming_loss(yca, pa))
    report    = classification_report(yca, pa, target_names=FAULT_NAMES, digits=4)

    metrics = {
        "model": "CNN+LSTM", "window_size": RUN_WINDOW_SIZE,
        "best_params": best_params, "best_threshold": best_threshold,
        "macro_f1": macro_f1, "micro_f1": micro_f1,
        "macro_precision": macro_pre, "macro_recall": macro_rec,
        "hamming_loss": h_loss,
        "model_size_kb": round(keras_size_kb, 4),
        "train_time_min": round(elapsed / 60, 2),
        "n_anomalous_test": int(mask.sum()),
    }

    logger.info(f"\nCNN+LSTM — ws={RUN_WINDOW_SIZE}")
    logger.info(f"Macro-F1: {macro_f1:.4f} · Micro-F1: {micro_f1:.4f}")
    logger.info(f"Hamming:  {h_loss:.4f} · Size: {keras_size_kb:.2f} KB")
    logger.info(f"\n{report}")

    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=4)
    with open(os.path.join(out_dir, "classification_report.txt"), "w") as f:
        f.write(f"CNN+LSTM — ws={RUN_WINDOW_SIZE}\n")
        f.write(f"Macro-F1: {macro_f1:.4f}\nMicro-F1: {micro_f1:.4f}\n")
        f.write(f"Hamming: {h_loss:.4f}\nSize KB: {keras_size_kb:.4f}\n\n")
        f.write(report)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f"CNN+LSTM Multilabel — ws={RUN_WINDOW_SIZE}  Macro-F1={macro_f1:.4f}", fontsize=13)
    if "loss" in histd:
        axes[0].plot(histd["loss"], label="Train")
        axes[0].plot(histd.get("val_loss", []), label="Val")
        axes[0].set_title("Loss"); axes[0].legend(); axes[0].grid(True)
    if vf1h:
        axes[1].plot(vf1h, color="darkorange")
        axes[1].set_title("Val Macro-F1"); axes[1].grid(True)
    f1_ch = [float(f1_score(yca[:, i], pa[:, i], zero_division=0)) for i in range(N_LABELS)]
    short = [n.replace("fault_", "") for n in FAULT_NAMES]
    axes[2].barh(short, f1_ch, color="steelblue", alpha=0.8)
    axes[2].axvline(macro_f1, color="red", ls="--", label=f"macro={macro_f1:.3f}")
    axes[2].set_title("F1 per channel canal"); axes[2].legend()
    axes[2].set_xlim(0, 1); axes[2].grid(True, axis="x")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "plots.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

def main():
    ws = RUN_WINDOW_SIZE
    logger.info(f"CNN+LSTM MULTILABEL (GPU) — ws={ws}")

    ckpt  = os.path.join(OUTPUT_BASE_DIR, "best_model.keras")
    flag  = os.path.join(OUTPUT_BASE_DIR, "training_done.json")
    histf = os.path.join(OUTPUT_BASE_DIR, "training_history.json")

    X_tr,  yb_tr,  yc_tr  = load_split(ws, "train")
    X_val, yb_val, yc_val = load_split(ws, "val")
    X_te,  yb_te,  yc_te  = load_split(ws, "test")
    shape = (X_tr.shape[1], X_tr.shape[2])

    best_params = run_optuna(X_tr, yc_tr, yb_tr, X_val, yc_val, yb_val)
    gc.collect(); K.clear_session()
    thr_opt = best_params.pop("threshold", 0.5)

    if os.path.exists(flag) and os.path.exists(ckpt):
        model   = models.load_model(ckpt)
        fd      = json.load(open(flag))
        elapsed = fd.get("elapsed_sec", 0)
        vf1h    = fd.get("val_f1_history", [])
        histd   = json.load(open(histf)) if os.path.exists(histf) else {}
    else:
        model = build_cnn_lstm(
            shape,
            n_filters_1  = best_params["n_filters_1"],
            n_filters_2  = best_params["n_filters_2"],
            kernel_size  = best_params["kernel_size"],
            lstm_units   = best_params["lstm_units"],
            n_lstm_layers = best_params["n_lstm_layers"],
            dropout      = best_params["dropout"],
            lr           = best_params["lr"],
            l2_reg       = best_params["l2_reg"],
        )
        model.summary(print_fn=logger.info)
        sw = sample_weights(yc_tr)
        cb = ValMacroF1(X_val, yc_val, yb_val, thr_opt)

        t0 = time.time()
        hist = model.fit(
            X_tr, yc_tr, sample_weight=sw,
            validation_data=(X_val, yc_val),
            epochs=FINAL_EPOCHS, batch_size=best_params["batch_size"],
            callbacks=[
                cb,
                callbacks.ModelCheckpoint(ckpt, monitor="val_macro_f1",
                                          mode="max", save_best_only=True, verbose=0),
                callbacks.EarlyStopping(monitor="val_macro_f1", mode="max",
                                        patience=PATIENCE, restore_best_weights=True),
                callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                                            patience=6, min_lr=1e-6, verbose=0),
            ],
            verbose=1,
        )
        elapsed = time.time() - t0
        vf1h    = [float(v) for v in cb.history]
        histd   = {k: [float(v) for v in vs] for k, vs in hist.history.items()}
        histd["val_macro_f1"] = vf1h
        with open(histf, "w") as f: json.dump(histd, f, indent=4)
        with open(flag, "w") as f:
            json.dump({"elapsed_sec": elapsed, "val_f1_history": vf1h,
                       "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")}, f, indent=4)
        model = models.load_model(ckpt)

    best_threshold, _ = find_threshold(model, X_val, yc_val, yb_val)
    keras_size_kb = os.path.getsize(ckpt) / 1024
    save_results(model, X_te, yc_te, yb_te, best_threshold,
                 best_params, histd, vf1h, keras_size_kb, elapsed, OUTPUT_BASE_DIR)
    logger.info(f"Ready — ws={ws}.")


if __name__ == "__main__":
    main()