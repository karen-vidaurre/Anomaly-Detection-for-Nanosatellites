from google.colab import drive
drive.mount("/content/drive")

import os, json, time, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    classification_report, f1_score, precision_score,
    recall_score, hamming_loss, accuracy_score,
)

warnings.filterwarnings("ignore")

DATA_PATH = "/content/drive/MyDrive/Anomaly_codes_Mar/sim_357_bal_t1.csv"
OUT_DIR   = "/content/drive/MyDrive/Anomaly_codes_Mar/variance_multilabel"
os.makedirs(OUT_DIR, exist_ok=True)

WINDOW_SIZES = [5, 10, 15, 20]
STEP_SIZE    = 1
TRAIN_RATIO  = 0.70
VAL_RATIO    = 0.15
SEED         = 42

USE_MASK = True

N_THRESHOLD_STEPS = 100

FAULT_NAMES = [
    "fault_ss1", "fault_ss2", "fault_ss3", "fault_ss4", "fault_ss5", "fault_ss6",
    "fault_magn1", "fault_magn2", "fault_magn3",
    "fault_gyro1", "fault_gyro2", "fault_gyro3",
    "fault_mtq1",  "fault_mtq2",  "fault_mtq3",
]
N_LABELS = len(FAULT_NAMES)
np.random.seed(SEED)

print(f"OUT_DIR:  {OUT_DIR}")
print(f"USE_MASK: {USE_MASK}")

def load_data(path):
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    df[df.select_dtypes("float64").columns] = \
        df.select_dtypes("float64").astype("float32")
    print(f"CSV ready: {df.shape}")
    return df


def build_windows(X_raw, y_bin_raw, y_ch_raw, W, step=1):
    N, F = X_raw.shape
    n_win = (N - W) // step + 1
    rs, cs = X_raw.strides
    X_win = np.lib.stride_tricks.as_strided(
        X_raw, shape=(n_win, W, F), strides=(rs * step, rs, cs)
    ).copy()
    y_bin = np.array([int(y_bin_raw[i:i+W].max()) for i in range(n_win)], dtype=np.int8)
    y_ch  = np.array([y_ch_raw[i:i+W].max(axis=0) for i in range(n_win)],  dtype=np.int8)
    return X_win, y_bin, y_ch

def split_chrono(arrays, train_ratio, val_ratio, gap):
    N = len(arrays[0])
    te = int(N * train_ratio)
    vs = te + gap
    ve = int(N * (train_ratio + val_ratio))
    ts = ve + gap

    def sl(i0, i1): return tuple(a[i0:i1] for a in arrays)
    train, val, test = sl(0, te), sl(vs, ve), sl(ts, N)

    for name, s in [("TRAIN", train), ("VAL", val), ("TEST", test)]:
        n = len(s[0])
        na = int(s[1].sum()) if n else 0
        print(f"  {name:5s}: {n:7d} | anómalas={na} ({100*na/max(n,1):.1f}%)")
    return train, val, test

def normalize_nominal(X_train, y_train_bin):
    nom  = X_train[y_train_bin == 0]
    mean = nom.mean(axis=(0, 1))
    std  = nom.std(axis=(0, 1)) + 1e-8
    return mean, std

def apply_norm(X, mean, std):
    return (X - mean[None, None, :]) / std[None, None, :]


def window_std(X_win):
    return X_win.std(axis=1)

def calibrate_tau_range(sigma_nom, percentile_low=5, percentile_high=99):

    all_sigma = sigma_nom.ravel()
    tau_min = float(np.percentile(all_sigma, percentile_low))
    tau_max = float(np.percentile(all_sigma, percentile_high))

    tau_min_ch = np.percentile(sigma_nom, percentile_low,  axis=0)  # (F,)
    tau_max_ch = np.percentile(sigma_nom, percentile_high, axis=0)  # (F,)

    print(f"  τ range ranking: [{tau_min:.4f}, {tau_max:.4f}]")
    print(f"  τ range per channel (min): {np.round(tau_min_ch, 4)}")
    print(f"  τ range per channel (max): {np.round(tau_max_ch, 4)}")
    return tau_min, tau_max, tau_min_ch, tau_max_ch

def calibrate_k_range(sigma_val):
    k_min, k_max = 0.5, 4.0
    print(f"  k range (fixed, dimensionless): [{k_min}, {k_max}]")
    return k_min, k_max

class VarianceMultilabelV2:
    def __init__(self):
        self.tau = None

    def fit(self, X_val, y_val_bin, y_val_ch, tau_min_ch, tau_max_ch, use_mask=True):
        sigma_val = window_std(X_val)      # (N, F)
        self.tau  = np.zeros(N_LABELS)

        if use_mask:
            mask = y_val_bin == 1
        else:
            mask = np.ones(len(y_val_bin), dtype=bool)

        for ch in range(N_LABELS):
            taus   = np.linspace(tau_min_ch[ch], tau_max_ch[ch], N_THRESHOLD_STEPS)
            best_t, best_f1 = taus[0], -1.0
            for t in taus:
                y_pred_ch = (sigma_val[mask, ch] > t).astype(int)
                f1 = f1_score(y_val_ch[mask, ch], y_pred_ch, zero_division=0)
                if f1 > best_f1:
                    best_f1, best_t = f1, t
            self.tau[ch] = best_t

        y_pred_all = (sigma_val[mask] > self.tau[None, :]).astype(int)
        macro_f1   = f1_score(y_val_ch[mask], y_pred_all, average="macro", zero_division=0)
        print(f"  [Variance v2] τ per canal: {np.round(self.tau, 4)}")
        print(f"  [Variance v2] val_MacroF1={macro_f1:.4f}")

    def predict(self, X):
        sigma = window_std(X)                              # (N, F)
        y_ch  = (sigma > self.tau[None, :]).astype(int)   # broadcast τ per channel
        y_bin = (y_ch.sum(axis=1) > 0).astype(int)
        return y_bin, y_ch

class AdaptiveVarianceMultilabelV2:

    def __init__(self):
        self.k = None

    def fit(self, X_val, y_val_bin, y_val_ch, k_min, k_max, use_mask=True):
        sigma_val  = window_std(X_val)
        mu_sigma   = sigma_val.mean(axis=1, keepdims=True)
        sig_sigma  = sigma_val.std(axis=1,  keepdims=True)

        ks = np.linspace(k_min, k_max, N_THRESHOLD_STEPS)

        if use_mask:
            mask = y_val_bin == 1
        else:
            mask = np.ones(len(y_val_bin), dtype=bool)

        best_k, best_f1 = ks[0], -1.0
        for k in ks:
            thr    = mu_sigma + k * sig_sigma              # (N, 1)
            y_pred = (sigma_val > thr).astype(int)
            f1 = f1_score(y_val_ch[mask], y_pred[mask],
                          average="macro", zero_division=0)
            if f1 > best_f1:
                best_f1, best_k = f1, k

        self.k = best_k
        print(f"  [Adaptive v2] k={self.k:.4f}  val_MacroF1={best_f1:.4f}")

    def predict(self, X):
        sigma    = window_std(X)
        mu       = sigma.mean(axis=1, keepdims=True)
        sig      = sigma.std(axis=1,  keepdims=True)
        thr      = mu + self.k * sig
        y_ch     = (sigma > thr).astype(int)
        y_bin    = (y_ch.sum(axis=1) > 0).astype(int)
        return y_bin, y_ch

def evaluate(y_true_ch, y_pred_ch, y_true_bin, y_pred_bin, label, use_mask):
    if use_mask:
        mask = y_true_bin == 1
        n_e  = mask.sum()
        if n_e == 0:
            print(f"  [{label}] No anomaly windows — skip")
            return {}
        yc, yp = y_true_ch[mask], y_pred_ch[mask]
        print(f"  [{label}] Test in {n_e} anomaly windows {len(y_true_bin)}")
    else:
        yc, yp = y_true_ch, y_pred_ch
        print(f"  [{label}] Test in all {len(y_true_bin)} windows")

    m = {
        "label":            label,
        "use_mask":         use_mask,
        "n_evaluated":      int(len(yc)),
        "macro_f1":         float(f1_score(yc, yp, average="macro",    zero_division=0)),
        "micro_f1":         float(f1_score(yc, yp, average="micro",    zero_division=0)),
        "macro_precision":  float(precision_score(yc, yp, average="macro", zero_division=0)),
        "macro_recall":     float(recall_score(yc, yp,    average="macro", zero_division=0)),
        "hamming_loss":     float(hamming_loss(yc, yp)),
        "binary_f1":        float(f1_score(y_true_bin, y_pred_bin, zero_division=0)),
        "binary_precision": float(precision_score(y_true_bin, y_pred_bin, zero_division=0)),
        "binary_recall":    float(recall_score(y_true_bin, y_pred_bin,    zero_division=0)),
        "binary_accuracy":  float(accuracy_score(y_true_bin, y_pred_bin)),
    }

    print(f"\n  ── {label} ──")
    print(f"  [Binary]    F1={m['binary_f1']:.4f}  "
          f"Prec={m['binary_precision']:.4f}  Rec={m['binary_recall']:.4f}")
    print(f"  [Multilabel] Macro-F1={m['macro_f1']:.4f}  "
          f"Micro-F1={m['micro_f1']:.4f}  Hamming={m['hamming_loss']:.4f}")
    print(f"\n{classification_report(yc, yp, target_names=FAULT_NAMES, digits=4)}")
    return m


def save_report(metrics, model_name, tau_info, ws, out_dir, use_mask, yc, yp):
    report = classification_report(yc, yp, target_names=FAULT_NAMES, digits=4)
    path   = os.path.join(out_dir, f"report_{model_name}_ws{ws}.txt")
    with open(path, "w") as f:
        f.write(f"{model_name} — ws={ws}\n")
        f.write(f"USE_MASK={use_mask}\n")
        f.write(f"Parámetro: {tau_info}\n\n")
        f.write(f"Binary F1:       {metrics.get('binary_f1', 0):.4f}\n")
        f.write(f"Binary Prec:     {metrics.get('binary_precision', 0):.4f}\n")
        f.write(f"Binary Recall:   {metrics.get('binary_recall', 0):.4f}\n\n")
        f.write(f"Multilabel Macro-F1:  {metrics.get('macro_f1', 0):.4f}\n")
        f.write(f"Multilabel Micro-F1:  {metrics.get('micro_f1', 0):.4f}\n")
        f.write(f"Hamming Loss:         {metrics.get('hamming_loss', 0):.4f}\n\n")
        f.write(report)


def plot_f1_channels(y_true_ch, y_pred_ch, y_true_bin, model_name, ws, out_dir, use_mask):
    if use_mask:
        mask = y_true_bin == 1
        yc, yp = y_true_ch[mask], y_pred_ch[mask]
    else:
        yc, yp = y_true_ch, y_pred_ch

    f1_ch = [float(f1_score(yc[:, i], yp[:, i], zero_division=0)) for i in range(N_LABELS)]
    short = [n.replace("fault_", "") for n in FAULT_NAMES]

    fig, ax = plt.subplots(figsize=(10, 5))
    colors  = (["steelblue"] * 6 + ["darkorange"] * 3 +
               ["firebrick"] * 3 + ["seagreen"] * 3)
    ax.barh(short, f1_ch, color=colors, alpha=0.8)
    ax.axvline(np.mean(f1_ch), color="black", ls="--",
               label=f"macro={np.mean(f1_ch):.3f}")
    ax.set_title(f"{model_name} — F1 por canal — ws={ws}\n"
                 f"{'(solo anómalas)' if use_mask else '(todas)'}")
    ax.set_xlim(0, 1); ax.legend(); ax.grid(True, axis="x")

    from matplotlib.patches import Patch
    legend_elems = [
        Patch(facecolor="steelblue",   label="Sun sensors (ss1-6)"),
        Patch(facecolor="darkorange",  label="Magnetometers (magn1-3)"),
        Patch(facecolor="firebrick",   label="Gyroscopes (gyro1-3)"),
        Patch(facecolor="seagreen",    label="Magnetorquers (mtq1-3)"),
    ]
    ax.legend(handles=legend_elems + [
        plt.Line2D([0], [0], color="black", ls="--",
                   label=f"Macro F1={np.mean(f1_ch):.3f}")
    ], loc="lower right", fontsize=8)

    plt.tight_layout()
    path = os.path.join(out_dir, f"f1_channels_{model_name.lower().replace(' ', '_')}_ws{ws}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)

def plot_comparison(results_ws, ws, out_dir, use_mask):
    var_m = results_ws.get("variance_v2", {})
    ada_m = results_ws.get("adaptive_v2", {})

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(
        f"Variance v2 — Multilabel — ws={ws}  "
        f"{'(only anomaly)' if use_mask else '(all)'}",
        fontsize=12,
    )
    for ax, key, title in [
        (axes[0], "binary_f1",  "F1 binary (detection)"),
        (axes[1], "macro_f1",   "Macro-F1 Multilabel"),
        (axes[2], "hamming_loss", "Hamming Loss (best)"),
    ]:
        vals = [var_m.get(key, 0), ada_m.get(key, 0)]
        bars = ax.bar(["Variance\n(per channel)", "Adaptive\nVariance"],
                      vals, color=["steelblue", "salmon"], alpha=0.8)
        ax.set_title(title)
        ax.set_ylim(0, max(1, max(vals) * 1.2))
        ax.grid(True, axis="y")
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{v:.3f}", ha="center", fontsize=11)

    plt.tight_layout()
    path = os.path.join(out_dir, f"comparison_ws{ws}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)

def main():
    df = load_data(DATA_PATH)

    fault_cols  = [c for c in df.columns if c.startswith("fault_")]
    signal_cols = [c for c in df.columns
                   if c not in fault_cols
                   and c.lower() != "time"
                   and not c.startswith("point_error")]

    print(f"Signals: {len(signal_cols)}  |  Fault: {len(fault_cols)}")
    X_raw     = df[signal_cols].values.astype(np.float32)
    y_bin_raw = df[fault_cols].values.max(axis=1).astype(np.int8)
    y_ch_raw  = df[fault_cols].values.astype(np.int8)

    all_results = {}

    for W in WINDOW_SIZES:
        print(f"WINDOW_SIZE = {W}")

        ws_dir = os.path.join(OUT_DIR, f"ws{W}")
        os.makedirs(ws_dir, exist_ok=True)

        # 1. Windows
        X_win, y_bin, y_ch = build_windows(X_raw, y_bin_raw, y_ch_raw, W, STEP_SIZE)
        print(f"Windows: {X_win.shape}  anomaly={y_bin.sum()} ({100*y_bin.mean():.1f}%)")

        # 2. Split
        (X_tr, yb_tr, yc_tr), (X_val, yb_val, yc_val), (X_te, yb_te, yc_te) = \
            split_chrono((X_win, y_bin, y_ch), TRAIN_RATIO, VAL_RATIO, gap=W)

        # 3. Normalization z-score
        mean_feat, std_feat = normalize_nominal(X_tr, yb_tr)
        X_tr_s  = apply_norm(X_tr,  mean_feat, std_feat)
        X_val_s = apply_norm(X_val, mean_feat, std_feat)
        X_te_s  = apply_norm(X_te,  mean_feat, std_feat)
        np.save(os.path.join(ws_dir, "mean_feat.npy"), mean_feat)
        np.save(os.path.join(ws_dir, "std_feat.npy"),  std_feat)

        # 4. Calibrate threshold ranges based on the data
        # σ of the nominal Windows in validation this is the distribution
        # of “nominal noise” that τ must exceed to be classified as an anomaly
        sigma_val     = window_std(X_val_s)         # (N_val, F)
        sigma_val_nom = sigma_val[yb_val == 0]      # only nominal Windows
        print("\nCalibrate threshold ranges:")
        tau_min, tau_max, tau_min_ch, tau_max_ch = calibrate_tau_range(sigma_val_nom)
        k_min, k_max = calibrate_k_range(sigma_val)

        ws_results = {}

        print("\n── Variance-Based v2 (per-channel threshold) ──")
        var_model = VarianceMultilabelV2()
        var_model.fit(X_val_s, yb_val, yc_val, tau_min_ch, tau_max_ch, use_mask=USE_MASK)

        t0 = time.time()
        yb_pred_var, yc_pred_var = var_model.predict(X_te_s)
        inf_var = (time.time() - t0) * 1e6 / len(X_te_s)

        var_m = evaluate(yc_te, yc_pred_var, yb_te, yb_pred_var, "Variance v2", USE_MASK)
        var_m["tau_per_channel"] = var_model.tau.tolist()
        var_m["inference_us"]    = round(inf_var, 4)
        var_m["window_size"]     = W
        ws_results["variance_v2"] = var_m

        if USE_MASK:
            mask_te = yb_te == 1
            yc_eval, yp_eval_var = yc_te[mask_te], yc_pred_var[mask_te]
        else:
            yc_eval, yp_eval_var = yc_te, yc_pred_var

        save_report(var_m, "variance_v2", f"τ={np.round(var_model.tau, 4)}", W, ws_dir, USE_MASK, yc_eval, yp_eval_var)
        plot_f1_channels(yc_te, yc_pred_var, yb_te, "Variance v2", W, ws_dir, USE_MASK)

        print("\n── Adaptive Variance v2 ──")
        ada_model = AdaptiveVarianceMultilabelV2()
        ada_model.fit(X_val_s, yb_val, yc_val, k_min, k_max, use_mask=USE_MASK)

        t0 = time.time()
        yb_pred_ada, yc_pred_ada = ada_model.predict(X_te_s)
        inf_ada = (time.time() - t0) * 1e6 / len(X_te_s)

        ada_m = evaluate(yc_te, yc_pred_ada, yb_te, yb_pred_ada, "Adaptive v2", USE_MASK)
        ada_m["k"]           = float(ada_model.k)
        ada_m["inference_us"] = round(inf_ada, 4)
        ada_m["window_size"]  = W
        ws_results["adaptive_v2"] = ada_m

        if USE_MASK:
            yp_eval_ada = yc_pred_ada[mask_te]
        else:
            yp_eval_ada = yc_pred_ada

        save_report(ada_m, "adaptive_v2", f"k={ada_model.k:.4f}", W, ws_dir, USE_MASK, yc_eval, yp_eval_ada)
        plot_f1_channels(yc_te, yc_pred_ada, yb_te, "Adaptive v2", W, ws_dir, USE_MASK)
        plot_comparison(ws_results, W, ws_dir, USE_MASK)

        with open(os.path.join(ws_dir, "metrics.json"), "w") as f:
            json.dump(ws_results, f, indent=4)

        np.save(os.path.join(ws_dir, "sigma_val_nom_stats.npy"),
                np.stack([sigma_val_nom.mean(axis=0),
                          sigma_val_nom.std(axis=0),
                          np.percentile(sigma_val_nom, 5,  axis=0),
                          np.percentile(sigma_val_nom, 95, axis=0)]))

        all_results[f"ws{W}"] = ws_results
        print(f"\n  Saved: {ws_dir}")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        f"Variance v2 — Evolution with Window Size\n"
        f"{'(sonly anomaly)' if USE_MASK else '(all)'}", fontsize=12
    )
    for ax, metric, title in [
        (axes[0], "macro_f1",  "Macro-F1 Multilabel"),
        (axes[1], "binary_f1", "F1 binary"),
    ]:
        for key, label, color in [
            ("variance_v2", "Variance (per channel)", "steelblue"),
            ("adaptive_v2", "Adaptive Variance",    "salmon"),
        ]:
            vals = [all_results.get(f"ws{W}", {}).get(key, {}).get(metric, 0)
                    for W in WINDOW_SIZES]
            ax.plot(WINDOW_SIZES, vals, marker="o", label=label, color=color)
        ax.set_xlabel("Window size"); ax.set_title(title)
        ax.legend(); ax.grid(True); ax.set_ylim(0, 1)
        ax.set_xticks(WINDOW_SIZES)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "summary_evolution.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    with open(os.path.join(OUT_DIR, "all_results.json"), "w") as f:
        json.dump(all_results, f, indent=4)

    print(f"\nSqaved in: {OUT_DIR}")

main()