import os
import argparse
import numpy as np
import pywt
import pickle
import json
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    accuracy_score,
    confusion_matrix,
    classification_report
)

# ============================================================
# Arguments
# ============================================================

parser = argparse.ArgumentParser()
parser.add_argument("--window_size", type=int, required=True)
parser.add_argument("--metric", type=str, default="f1",
                    choices=["f1", "f2", "precision", "recall"],
                    help="Métrica para elegir el threshold óptimo por canal")
args = parser.parse_args()

W_PARAM    = args.window_size
OPT_METRIC = args.metric

BASE_DIR    = "./data/processed"
DATA_DIR    = os.path.join(BASE_DIR, f"windows_{W_PARAM}")
MODEL_DIR   = "./models"
RESULTS_DIR = "./results/ML5_wav_LR_AT"

os.makedirs(MODEL_DIR,   exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# ============================================================
# Load data
# ============================================================

X_train   = np.load(os.path.join(DATA_DIR, "train/X.npy"))
y_train   = np.load(os.path.join(DATA_DIR, "train/y_bin.npy"))
y_ch_train = np.load(os.path.join(DATA_DIR, "train", "y_ch.npy"))

X_val     = np.load(os.path.join(DATA_DIR, "val/X.npy"))
y_val     = np.load(os.path.join(DATA_DIR, "val/y_bin.npy"))
y_ch_val  = np.load(os.path.join(DATA_DIR, "val", "y_ch.npy"))

X_test    = np.load(os.path.join(DATA_DIR, "test/X.npy"))
y_test    = np.load(os.path.join(DATA_DIR, "test/y_bin.npy"))
y_ch_test = np.load(os.path.join(DATA_DIR, "test", "y_ch.npy"))

N_CHANNELS = y_ch_train.shape[1]

print(f"\nAdaptive per-channel thresholding — window size {W_PARAM}")
print(f"Optimization metric : {OPT_METRIC}")
print(f"Train samples       : {len(X_train)}")
print(f"Channels            : {N_CHANNELS}")

# Eliminar última columna (pointing accuracy)
X_train = X_train[:, :, :-1]
X_val   = X_val[:,   :, :-1]
X_test  = X_test[:,  :, :-1]

print(f"Train shape         : {X_train.shape}")

# ============================================================
# Normalización de entrada (solo con muestras normales)
# ============================================================

X_train_nom = X_train[y_train == 0]
mean_feat   = X_train_nom.mean(axis=(0, 1))
std_feat    = X_train_nom.std(axis=(0, 1))  + 1e-8

def scale_windows(X, mean, std):
    return (X - mean[None, None, :]) / std[None, None, :]

X_train = scale_windows(X_train, mean_feat, std_feat)
X_val   = scale_windows(X_val,   mean_feat, std_feat)
X_test  = scale_windows(X_test,  mean_feat, std_feat)

# ============================================================
# DWT
# ============================================================

def compute_dwt_windows(X, wavelet="db4", level=1):
    """
    X : (N, window_size, n_features)
    Retorna:
        A : coeficientes de aproximación  (N, len_A, n_features)
        D : coeficientes de detalle concat (N, len_D, n_features)
    """
    if X.ndim == 2:
        X = X[..., np.newaxis]

    N, W, F = X.shape
    A_list, D_list = [], []

    for i in range(N):
        A_ch, D_ch = [], []
        for ch in range(F):
            coeffs = pywt.wavedec(X[i, :, ch], wavelet, level=level)
            A_ch.append(coeffs[0])
            D_ch.append(np.concatenate(coeffs[1:]))
        A_list.append(np.stack(A_ch, axis=1))
        D_list.append(np.stack(D_ch, axis=1))

    return np.array(A_list), np.array(D_list)

print("\nComputando DWT...")
A_train, D_train = compute_dwt_windows(X_train)
A_val,   D_val   = compute_dwt_windows(X_val)
A_test,  D_test  = compute_dwt_windows(X_test)

# ============================================================
# Normalización DWT (solo con muestras normales)
# ============================================================

D_train_nom = D_train[y_train == 0]
mean_dwt    = D_train_nom.mean(axis=(0, 1), keepdims=True)
std_dwt     = D_train_nom.std(axis=(0, 1),  keepdims=True) + 1e-8

D_train_n = (D_train - mean_dwt) / std_dwt
D_val_n   = (D_val   - mean_dwt) / std_dwt
D_test_n  = (D_test  - mean_dwt) / std_dwt

# ============================================================
# Features de energía
# ============================================================

def compute_energy_features(D):
    """Energía por coeficiente DWT: (N, n_features)"""
    return np.sum(D ** 2, axis=1)

E_train = compute_energy_features(D_train_n)
E_val   = compute_energy_features(D_val_n)
E_test  = compute_energy_features(D_test_n)

print(f"Feature shape (energía): {E_train.shape}")

# ============================================================
# Entrenamiento — un clasificador LR por canal
# ============================================================

print("\nEntrenando modelos por canal...")
models = []
for ch in range(N_CHANNELS):
    clf = LogisticRegression(max_iter=1000, class_weight="balanced")
    clf.fit(E_train, y_ch_train[:, ch])
    models.append(clf)

# Probabilidades (fijas, independientes del threshold)
probs_val  = np.stack([models[ch].predict_proba(E_val)[:, 1]  for ch in range(N_CHANNELS)], axis=1)
probs_test = np.stack([models[ch].predict_proba(E_test)[:, 1] for ch in range(N_CHANNELS)], axis=1)

# ============================================================
# Búsqueda de threshold adaptativo por canal
# ============================================================

THRESHOLDS = np.arange(0.10, 0.96, 0.05).round(2)

def score_threshold(y_true, probs, threshold, metric):
    """Calcula la métrica elegida para un canal y un threshold."""
    y_pred = (probs > threshold).astype(int)
    if metric == "f1":
        return f1_score(y_true, y_pred, zero_division=0)
    elif metric == "f2":
        p = precision_score(y_true, y_pred, zero_division=0)
        r = recall_score(y_true,    y_pred, zero_division=0)
        return (5 * p * r) / (4 * p + r + 1e-12)
    elif metric == "precision":
        return precision_score(y_true, y_pred, zero_division=0)
    elif metric == "recall":
        return recall_score(y_true, y_pred, zero_division=0)

print(f"\n{'Canal':>6} | {'Threshold':>9} | {'Val score':>10} | Sweep")
print("-" * 70)

best_thresholds  = np.zeros(N_CHANNELS)
best_val_scores  = np.zeros(N_CHANNELS)
sweep_results    = {}

for ch in range(N_CHANNELS):
    scores = []
    for thr in THRESHOLDS:
        s = score_threshold(y_ch_val[:, ch], probs_val[:, ch], thr, OPT_METRIC)
        scores.append(s)

    scores = np.array(scores)
    best_idx = int(np.argmax(scores))

    best_thresholds[ch] = THRESHOLDS[best_idx]
    best_val_scores[ch] = scores[best_idx]

    # Barra visual del sweep
    bar = "".join(
        "█" if i == best_idx else ("▓" if s > 0.5 else "░")
        for i, s in enumerate(scores)
    )
    print(f"  ch{ch:02d} | {best_thresholds[ch]:>9.2f} | {best_val_scores[ch]:>10.4f} | {bar}")

    sweep_results[f"ch{ch}"] = {
        "best_threshold" : float(best_thresholds[ch]),
        f"val_{OPT_METRIC}" : float(best_val_scores[ch]),
        "sweep"          : {float(t): float(s) for t, s in zip(THRESHOLDS, scores)}
    }

print(f"\nThreshold medio    : {best_thresholds.mean():.4f}")
print(f"Threshold min/max  : {best_thresholds.min():.2f} / {best_thresholds.max():.2f}")
print(f"Score medio (val)  : {best_val_scores.mean():.4f}")

# ============================================================
# Predicción final en test con thresholds adaptativos
# ============================================================

y_pred_adaptive = np.zeros_like(y_ch_test)
for ch in range(N_CHANNELS):
    y_pred_adaptive[:, ch] = (probs_test[:, ch] > best_thresholds[ch]).astype(int)

# Comparativa con threshold fijo 0.7
y_pred_fixed = (probs_test > 0.7).astype(int)

# ============================================================
# Evaluación y comparativa
# ============================================================

def multilabel_metrics(y_true, y_pred, label=""):
    macro_f1 = f1_score(y_true, y_pred, average="macro",  zero_division=0)
    micro_f1 = f1_score(y_true, y_pred, average="micro",  zero_division=0)
    macro_p  = precision_score(y_true, y_pred, average="macro", zero_division=0)
    macro_r  = recall_score(y_true,    y_pred, average="macro", zero_division=0)
    acc      = accuracy_score(y_true.ravel(), y_pred.ravel())
    print(f"\n  {label}")
    print(f"    Macro F1   : {macro_f1:.4f}")
    print(f"    Micro F1   : {micro_f1:.4f}")
    print(f"    Macro Prec : {macro_p:.4f}")
    print(f"    Macro Rec  : {macro_r:.4f}")
    print(f"    Accuracy   : {acc:.4f}")
    return {"macro_f1": macro_f1, "micro_f1": micro_f1,
            "macro_precision": macro_p, "macro_recall": macro_r,
            "accuracy": acc}

print("\n" + "=" * 50)
print("RESULTADOS EN TEST")
print("=" * 50)
metrics_adaptive = multilabel_metrics(y_ch_test, y_pred_adaptive, "Threshold adaptativo por canal")
metrics_fixed    = multilabel_metrics(y_ch_test, y_pred_fixed,    "Threshold fijo 0.70")

# Mejora relativa
delta_macro = metrics_adaptive["macro_f1"] - metrics_fixed["macro_f1"]
print(f"\n  Delta Macro F1 (adaptativo - fijo): {delta_macro:+.4f}")

# Report detallado por canal
report_adaptive = classification_report(
    y_ch_test, y_pred_adaptive,
    target_names=[f"ch{i}" for i in range(N_CHANNELS)],
    digits=4, output_dict=True
)

report_fixed = classification_report(
    y_ch_test, y_pred_fixed,
    target_names=[f"ch{i}" for i in range(N_CHANNELS)],
    digits=4, output_dict=True
)

# Tabla por canal
print(f"\n{'Canal':>6} | {'Thr':>5} | {'F1 adap':>8} | {'F1 fixed':>8} | {'Delta':>7} | {'P adap':>7} | {'R adap':>7}")
print("-" * 70)
for ch in range(N_CHANNELS):
    f1_a = report_adaptive[f"ch{ch}"]["f1-score"]
    f1_f = report_fixed[f"ch{ch}"]["f1-score"]
    p_a  = report_adaptive[f"ch{ch}"]["precision"]
    r_a  = report_adaptive[f"ch{ch}"]["recall"]
    delta = f1_a - f1_f
    mark = " ▲" if delta > 0.005 else (" ▼" if delta < -0.005 else "  ")
    print(f"  ch{ch:02d} | {best_thresholds[ch]:>5.2f} | {f1_a:>8.4f} | {f1_f:>8.4f} | {delta:>+7.4f}{mark} | {p_a:>7.4f} | {r_a:>7.4f}")

# ============================================================
# Guardar todo
# ============================================================

EXP_DIR = os.path.join(RESULTS_DIR, f"window_{W_PARAM}")
os.makedirs(EXP_DIR, exist_ok=True)

# Modelos
with open(os.path.join(EXP_DIR, "lr_models.pkl"), "wb") as f:
    pickle.dump(models, f)

# Thresholds adaptativos
threshold_payload = {
    "opt_metric"       : OPT_METRIC,
    "thresholds"       : {f"ch{ch}": float(best_thresholds[ch]) for ch in range(N_CHANNELS)},
    "val_scores"       : {f"ch{ch}": float(best_val_scores[ch]) for ch in range(N_CHANNELS)},
    "mean_threshold"   : float(best_thresholds.mean()),
    "mean_val_score"   : float(best_val_scores.mean()),
}
with open(os.path.join(EXP_DIR, "adaptive_thresholds.json"), "w") as f:
    json.dump(threshold_payload, f, indent=4)

# Sweep completo
with open(os.path.join(EXP_DIR, "threshold_sweep_per_channel.json"), "w") as f:
    json.dump(sweep_results, f, indent=4)

# Reports
with open(os.path.join(EXP_DIR, "classification_report_adaptive.json"), "w") as f:
    json.dump(report_adaptive, f, indent=4)

with open(os.path.join(EXP_DIR, "classification_report_fixed07.json"), "w") as f:
    json.dump(report_fixed, f, indent=4)

# Comparativa resumida
comparison = {
    "adaptive" : {k: round(v, 4) for k, v in metrics_adaptive.items()},
    "fixed_07" : {k: round(v, 4) for k, v in metrics_fixed.items()},
    "delta"    : {k: round(metrics_adaptive[k] - metrics_fixed[k], 4) for k in metrics_adaptive}
}
with open(os.path.join(EXP_DIR, "comparison_adaptive_vs_fixed.json"), "w") as f:
    json.dump(comparison, f, indent=4)

# Matrices de confusión (adaptativo)
conf_mats = {}
for ch in range(N_CHANNELS):
    cm = confusion_matrix(y_ch_test[:, ch], y_pred_adaptive[:, ch])
    conf_mats[f"ch{ch}"] = cm.tolist()
with open(os.path.join(EXP_DIR, "confusion_matrices_adaptive.json"), "w") as f:
    json.dump(conf_mats, f, indent=4)

# Normalización
np.save(os.path.join(EXP_DIR, "dwt_mean.npy"),   mean_dwt)
np.save(os.path.join(EXP_DIR, "dwt_std.npy"),    std_dwt)
np.save(os.path.join(EXP_DIR, "input_mean.npy"), mean_feat)
np.save(os.path.join(EXP_DIR, "input_std.npy"),  std_feat)

print(f"\nTodo guardado en: {EXP_DIR}")
print("Archivos generados:")
for fname in [
    "lr_models.pkl",
    "adaptive_thresholds.json",
    "threshold_sweep_per_channel.json",
    "classification_report_adaptive.json",
    "classification_report_fixed07.json",
    "comparison_adaptive_vs_fixed.json",
    "confusion_matrices_adaptive.json",
    "dwt_mean.npy", "dwt_std.npy",
    "input_mean.npy", "input_std.npy",
]:
    print(f"  {fname}")