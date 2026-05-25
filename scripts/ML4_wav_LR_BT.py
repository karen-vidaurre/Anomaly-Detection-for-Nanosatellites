import os
import argparse
import numpy as np
# import tensorflow as tf
import pywt
import pickle
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    accuracy_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)
# ============================================================
# Arguments
# ============================================================

parser = argparse.ArgumentParser()
parser.add_argument("--window_size", type=int, required=True)
args = parser.parse_args()

W_PARAM = args.window_size

BASE_DIR = "./data/processed"
DATA_DIR = os.path.join(BASE_DIR, f"windows_{W_PARAM}")

MODEL_DIR = "./models"
RESULTS_DIR = "./results/ML2_wav_LR_BT"

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# ============================================================
# Load data
# ============================================================

X_train = np.load(os.path.join(DATA_DIR, "train/X.npy"))
y_train = np.load(os.path.join(DATA_DIR, "train/y_bin.npy"))
y_ch_train = np.load(os.path.join(DATA_DIR, "train", "y_ch.npy"))

X_val = np.load(os.path.join(DATA_DIR, "val/X.npy"))
y_val = np.load(os.path.join(DATA_DIR, "val/y_bin.npy"))
y_ch_val   = np.load(os.path.join(DATA_DIR, "val",   "y_ch.npy"))

X_test = np.load(os.path.join(DATA_DIR, "test/X.npy"))
y_test = np.load(os.path.join(DATA_DIR, "test/y_bin.npy"))
y_ch_test  = np.load(os.path.join(DATA_DIR, "test",  "y_ch.npy"))

print(f"\nMultilabel thresholding for window size {W_PARAM}")
print(f"Train samples: {len(X_train)}")
# To remove pointing accuracy from training
X_train = X_train[:, :, 0:-1]
X_val   = X_val[:, :, 0:-1]
X_test  = X_test[:, :, 0:-1]

print(f"\nTrain data shape{X_train.shape}")

# ============================================================
# Normalization
# ============================================================

X_train_nom = X_train[y_train == 0]

mean_feat = X_train_nom.mean(axis=(0,1))
std_feat  = X_train_nom.std(axis=(0,1)) + 1e-8
# # Save scalers
# SCALER_DIR = os.path.join(DATA_DIR, "scalers")
# os.makedirs(SCALER_DIR, exist_ok=True)

# with open(os.path.join(SCALER_DIR, "feature_scaler.pkl"), "wb") as f:
#     pickle.dump({
#         "mean": mean_feat,
#         "std": std_feat
#     }, f)

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
    X: (N_windows, window_size, n_channels) o (N_windows, window_size)
    return:
        A: Approximation coefficients (low-frequency)
        D: Detail coefficients (high-frequency, concatenated)
    """
    if X.ndim == 2:
        X = X[..., np.newaxis]

    N, W, F = X.shape

    A_list = []
    D_list = []

    for i in range(N):
        A_ch = []
        D_ch = []

        for ch in range(F):
            coeffs = pywt.wavedec(X[i, :, ch], wavelet, level=level)

            A = coeffs[0]                  # Approximation
            D = np.concatenate(coeffs[1:]) # All detail levels

            A_ch.append(A)
            D_ch.append(D)

        A_list.append(np.stack(A_ch, axis=1))
        D_list.append(np.stack(D_ch, axis=1))

    return np.array(A_list), np.array(D_list)

A_train, D_train = compute_dwt_windows(X_train)
A_val,   D_val   = compute_dwt_windows(X_val)
A_test,  D_test  = compute_dwt_windows(X_test)

# ============================================================
# Detector
# ============================================================
# D_train_nom = D_train[y_train == 0] # Check without this
mean = np.mean(D_train, axis=(0,1), keepdims=True)
std  = np.std(D_train, axis=(0,1), keepdims=True) + 1e-8

D_train_n = (D_train - mean) / std
D_val_n   = (D_val   - mean) / std
D_test_n  = (D_test  - mean) / std

def compute_energy_features(D):
    return np.sum(D**2, axis=1)

E_train = compute_energy_features(D_train_n)
E_val   = compute_energy_features(D_val_n)
E_test  = compute_energy_features(D_test_n)

# ============================================================
# Threshold sweep + guardar mejor modelo
# ============================================================

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, classification_report
import json, pickle
import numpy as np

THRESHOLDS = [0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9]

# --- Entrenar modelos (una sola vez) ---
models = []
for ch in range(15):
    clf = LogisticRegression(max_iter=1000, class_weight='balanced')
    clf.fit(E_train, y_ch_train[:, ch])
    models.append(clf)

# Probabilidades sobre val y test (fijas, independientes del threshold)
probs_val  = np.stack([models[ch].predict_proba(E_val)[:, 1]  for ch in range(15)], axis=1)
probs_test = np.stack([models[ch].predict_proba(E_test)[:, 1] for ch in range(15)], axis=1)

# --- Sweep sobre val para elegir mejor threshold ---
print(f"\n{'Threshold':>10} | {'Macro F1':>10} | {'Micro F1':>10} | {'Accuracy':>10}")
print("-" * 50)

best_threshold = None
best_f1 = -1
results_by_threshold = {}

for thr in THRESHOLDS:
    y_pred_val = (probs_val > thr).astype(int)

    macro_f1 = f1_score(y_ch_val, y_pred_val, average='macro', zero_division=0)
    micro_f1 = f1_score(y_ch_val, y_pred_val, average='micro', zero_division=0)
    acc      = (y_pred_val == y_ch_val).mean()

    results_by_threshold[thr] = {
        "macro_f1": round(macro_f1, 4),
        "micro_f1": round(micro_f1, 4),
        "accuracy": round(float(acc), 4),
    }

    marker = " <-- mejor" if macro_f1 > best_f1 else ""
    print(f"{thr:>10.2f} | {macro_f1:>10.4f} | {micro_f1:>10.4f} | {acc:>10.4f}{marker}")

    if macro_f1 > best_f1:
        best_f1 = macro_f1
        best_threshold = thr

print(f"\nMejor threshold: {best_threshold}  (macro F1 val = {best_f1:.4f})")

# --- Evaluar en test con el mejor threshold ---
y_pred_test = (probs_test > best_threshold).astype(int)

report = classification_report(
    y_ch_test, y_pred_test,
    target_names=[f"ch{i}" for i in range(15)],
    digits=4, output_dict=True
)

# --- Guardar ---
EXP_DIR = os.path.join(RESULTS_DIR, f"window_{W_PARAM}")
os.makedirs(EXP_DIR, exist_ok=True)

# Modelos
with open(os.path.join(EXP_DIR, "lr_models.pkl"), "wb") as f:
    pickle.dump(models, f)

# Threshold elegido
with open(os.path.join(EXP_DIR, "best_threshold.json"), "w") as f:
    json.dump({"best_threshold": best_threshold, "val_macro_f1": best_f1}, f, indent=4)

# Resultados de todos los thresholds (útil para analizar después)
with open(os.path.join(EXP_DIR, "threshold_sweep.json"), "w") as f:
    json.dump(results_by_threshold, f, indent=4)

# Report final
with open(os.path.join(EXP_DIR, "classification_report.json"), "w") as f:
    json.dump(report, f, indent=4)

# Normalización
np.save(os.path.join(EXP_DIR, "dwt_mean.npy"), mean)
np.save(os.path.join(EXP_DIR, "dwt_std.npy"), std)
np.save(os.path.join(EXP_DIR, "input_mean.npy"), mean_feat)
np.save(os.path.join(EXP_DIR, "input_std.npy"), std_feat)

print(f"\nResultados guardados en: {EXP_DIR}")