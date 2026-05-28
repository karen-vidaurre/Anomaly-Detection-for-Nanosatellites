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

RUN_WINDOW_SIZE = 5
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

DATA_BASE_DIR   = "/kaggle/input/datasets/litzycondebts03/data-lstm-280326"
OUTPUT_BASE_DIR = f"/kaggle/working/bilstm_att_multi_ws{RUN_WINDOW_SIZE}"
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

class TemporalAttention(layers.Layer):

    def __init__(self, units=64, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.W = layers.Dense(units, activation="tanh", use_bias=True)
        self.v = layers.Dense(1, use_bias=False)

    def call(self, hidden_states, training=None):
        # hidden_states: (batch, T, hidden_dim)
        score  = self.W(hidden_states)        # (batch, T, units)
        score  = self.v(score)                # (batch, T, 1)
        alpha  = tf.nn.softmax(score, axis=1) # (batch, T, 1)
        context = tf.reduce_sum(alpha * hidden_states, axis=1)  # (batch, hidden_dim)
        return context, alpha

    def get_config(self):
        config = super().get_config()
        config.update({"units": self.units})
        return config

def build_bilstm_attention(input_shape, n_units=64, n_layers=1,
                            dropout=0.3, attention_units=32,
                            lr=1e-3, l2_reg=1e-4):

    inp = layers.Input(shape=input_shape, name="dwt_detail_input")
    x   = inp

    for i in range(n_layers):
        x = layers.Bidirectional(
            layers.LSTM(
                n_units,
                return_sequences=True,
                kernel_regularizer=regularizers.l2(l2_reg),
                recurrent_regularizer=regularizers.l2(l2_reg / 2),
                name=f"lstm_{i}",
            ),
            name=f"bilstm_{i}",
        )(x)
        x = layers.Dropout(dropout, name=f"drop_{i}")(x)
        if i < n_layers - 1:
            x = layers.LayerNormalization(name=f"layernorm_{i}")(x)

    context, attention_weights = TemporalAttention(
        units=attention_units, name="temporal_attention"
    )(x)

    x   = layers.Dense(max(16, n_units), activation="relu",
                       kernel_regularizer=regularizers.l2(l2_reg),
                       name="dense_head")(context)
    x   = layers.Dropout(dropout / 2, name="drop_head")(x)
    out = layers.Dense(N_LABELS, activation="sigmoid",
                       dtype="float32", name="output")(x)

    model = models.Model(inp, out, name=f"bilstm_att_ws{RUN_WINDOW_SIZE}")
    model.compile(
        optimizer=optimizers.Adam(learning_rate=lr, clipnorm=1.0),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    attention_model = models.Model(
        inp, attention_weights, name="attention_extractor"
    )

    return model, attention_model

def get_attention_weights(attention_model, X_sample):
    return attention_model.predict(X_sample, verbose=0)

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

def make_objective(X_tr, y_tr_ch, y_tr_bin, X_val, y_val_ch, y_val_bin):
    def objective(trial):
        n_units         = trial.suggest_categorical("n_units",  [32, 48, 64])
        n_layers        = trial.suggest_int("n_layers",   1, 2)
        dropout         = trial.suggest_float("dropout",  0.1, 0.45)
        attention_units = trial.suggest_categorical("attention_units", [16, 32, 48])
        lr              = trial.suggest_float("lr",       5e-4, 5e-3, log=True)
        l2_reg          = trial.suggest_float("l2_reg",   1e-5, 1e-3, log=True)
        batch_sz        = trial.suggest_categorical("batch_size", [64, 128, 256])
        threshold       = trial.suggest_float("threshold", 0.3, 0.7)

        model, _ = build_bilstm_attention(
            (X_tr.shape[1], X_tr.shape[2]),
            n_units, n_layers, dropout, attention_units, lr, l2_reg,
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
    db   = os.path.join(OUTPUT_BASE_DIR, f"optuna_bilstm_att_ws{RUN_WINDOW_SIZE}.db")
    name = f"bilstm_att_ws{RUN_WINDOW_SIZE}"
    study = optuna.create_study(
        study_name=name, storage=f"sqlite:///{db}",
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=SEED),
        load_if_exists=True,
    )
    done = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
    rem  = N_OPTUNA_TRIALS - done
    if rem > 0:
        logger.info(f"Optuna BiLSTM+Att: {done} prev, {rem} rest...")
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

def save_results(model, attention_model, X_te, yc_te, yb_te,
                 best_threshold, best_params, histd, vf1h,
                 keras_size_kb, elapsed, out_dir):
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
        "model": "BiLSTM+TemporalAttention", "window_size": RUN_WINDOW_SIZE,
        "best_params": best_params, "best_threshold": best_threshold,
        "macro_f1": macro_f1, "micro_f1": micro_f1,
        "macro_precision": macro_pre, "macro_recall": macro_rec,
        "hamming_loss": h_loss,
        "model_size_kb": round(keras_size_kb, 4),
        "train_time_min": round(elapsed / 60, 2),
        "n_anomalous_test": int(mask.sum()),
    }

    logger.info(f"\nBiLSTM+Att — ws={RUN_WINDOW_SIZE}")
    logger.info(f"Macro-F1: {macro_f1:.4f} · Micro-F1: {micro_f1:.4f}")
    logger.info(f"Hamming:  {h_loss:.4f} · Size: {keras_size_kb:.2f} KB")
    logger.info(f"\n{report}")

    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=4)
    with open(os.path.join(out_dir, "classification_report.txt"), "w") as f:
        f.write(f"Model: BiLSTM + Temporal Attention\nWindow: {RUN_WINDOW_SIZE}\n")
        f.write(f"Macro-F1: {macro_f1:.4f}\nMicro-F1: {micro_f1:.4f}\n")
        f.write(f"Hamming: {h_loss:.4f}\nSize KB: {keras_size_kb:.4f}\n\n")
        f.write(report)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        f"BiLSTM + Temporal Attention — ws={RUN_WINDOW_SIZE}  "
        f"Macro-F1={macro_f1:.4f}",
        fontsize=13,
    )

    if "loss" in histd:
        axes[0, 0].plot(histd["loss"], label="Train")
        axes[0, 0].plot(histd.get("val_loss", []), label="Val")
        axes[0, 0].set_title("Loss"); axes[0, 0].legend(); axes[0, 0].grid(True)

    if vf1h:
        axes[0, 1].plot(vf1h, color="darkorange")
        axes[0, 1].set_title("Val Macro-F1"); axes[0, 1].grid(True)

    f1_ch = [float(f1_score(yca[:, i], pa[:, i], zero_division=0)) for i in range(N_LABELS)]
    short = [n.replace("fault_", "") for n in FAULT_NAMES]
    axes[1, 0].barh(short, f1_ch, color="steelblue", alpha=0.8)
    axes[1, 0].axvline(macro_f1, color="red", ls="--", label=f"macro={macro_f1:.3f}")
    axes[1, 0].set_title("F1 per channel (anomaly windows)")
    axes[1, 0].legend(); axes[1, 0].set_xlim(0, 1); axes[1, 0].grid(True, axis="x")

    try:
        att_weights = get_attention_weights(
            attention_model, X_te[mask][:500]
        )  # (N, T, 1)
        att_mean = att_weights[:, :, 0].mean(axis=0)  # (T,)
        T = len(att_mean)
        axes[1, 1].bar(range(T), att_mean, color="purple", alpha=0.7)
        axes[1, 1].set_title("Map of average coverage (DWT coefficient)\n"
                              "Higher value = more effective at detecting faults")
        axes[1, 1].set_xlabel("DWT detail coefficient")
        axes[1, 1].set_ylabel("Target weight")
        axes[1, 1].grid(True, axis="y")

        np.save(os.path.join(out_dir, "attention_weights_mean.npy"), att_mean)
    except Exception as e:
        logger.warning(f"The service map could not be retrieved: {e}")
        axes[1, 1].text(0.5, 0.5, "Not available",
                        ha="center", va="center", transform=axes[1, 1].transAxes)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "plots.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

def main():
    ws = RUN_WINDOW_SIZE
    logger.info(f"BiLSTM + TEMPORAL ATTENTION MULTILABEL (GPU) — ws={ws}")

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
        model = models.load_model(
            ckpt,
            custom_objects={"TemporalAttention": TemporalAttention}
        )
        fd      = json.load(open(flag))
        elapsed = fd.get("elapsed_sec", 0)
        vf1h    = fd.get("val_f1_history", [])
        histd   = json.load(open(histf)) if os.path.exists(histf) else {}
        _, attention_model = build_bilstm_attention(
            shape,
            n_units         = best_params["n_units"],
            n_layers        = best_params["n_layers"],
            dropout         = best_params["dropout"],
            attention_units = best_params["attention_units"],
            lr              = best_params["lr"],
            l2_reg          = best_params["l2_reg"],
        )
        attention_model.set_weights(
            [w for w in model.weights
             if "temporal_attention" in w.name or "output" in w.name
             or "bilstm" in w.name or "dense" in w.name]
        )
    else:
        model, attention_model = build_bilstm_attention(
            shape,
            n_units         = best_params["n_units"],
            n_layers        = best_params["n_layers"],
            dropout         = best_params["dropout"],
            attention_units = best_params["attention_units"],
            lr              = best_params["lr"],
            l2_reg          = best_params["l2_reg"],
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
        model = models.load_model(
            ckpt,
            custom_objects={"TemporalAttention": TemporalAttention}
        )
        _, attention_model = build_bilstm_attention(
            shape,
            n_units         = best_params["n_units"],
            n_layers        = best_params["n_layers"],
            dropout         = best_params["dropout"],
            attention_units = best_params["attention_units"],
            lr              = best_params["lr"],
            l2_reg          = best_params["l2_reg"],
        )
        try:
            shared_layers = [l for l in model.layers
                             if "bilstm" in l.name or "temporal_attention" in l.name
                             or "dense_head" in l.name or "output" in l.name]
            for sl in shared_layers:
                try:
                    attention_model.get_layer(sl.name).set_weights(sl.get_weights())
                except Exception:
                    pass
        except Exception as e:
            logger.warning(f"We were unable to copy the weights to the service model: {e}")

    best_threshold, _ = find_threshold(model, X_val, yc_val, yb_val)
    keras_size_kb = os.path.getsize(ckpt) / 1024

    save_results(
        model, attention_model,
        X_te, yc_te, yb_te,
        best_threshold, best_params, histd, vf1h,
        keras_size_kb, elapsed, OUTPUT_BASE_DIR,
    )
    logger.info(f"Ready — ws={ws}.")


if __name__ == "__main__":
    main()