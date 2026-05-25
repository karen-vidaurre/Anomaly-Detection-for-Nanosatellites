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
RESULTS_DIR = "./results/ML1_wav_th_wom"

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

mask_anom = y_val == 1

E_val_anom = E_val[mask_anom]
y_ch_val_anom = y_ch_val[mask_anom]

# Find best threshold
# def find_best_thresholds(E_val, y_val, metric="f1"):

#     thresholds = np.zeros(E_val.shape[1])

#     for ch in range(E_val.shape[1]):

#         best_th = 0
#         best_score = -1

#         vals = E_val[:, ch]
#         th_range = np.linspace(vals.min(), vals.max(), 100)

#         for th in th_range:

#             pred = (vals > th).astype(int)

#             if metric == "f1":
#                 score = f1_score(y_val[:, ch], pred, zero_division=0)
#             elif metric == "precision":
#                 score = precision_score(y_val[:, ch], pred, zero_division=0)
#             elif metric == "recall":
#                 score = recall_score(y_val[:, ch], pred, zero_division=0)

#             if score > best_score:
#                 best_score = score
#                 best_th = th

#         thresholds[ch] = best_th

#     return thresholds
def find_best_thresholds(E_val, y_val):

    thresholds = np.zeros(E_val.shape[1])

    for ch in range(E_val.shape[1]):

        best_th = 0
        best_f1 = 0

        vals = E_val[:, ch]

        th_range = np.linspace(vals.min(), vals.max(), 100)

        for th in th_range:

            pred = (vals > th).astype(int)

            from sklearn.metrics import f1_score

            f1 = f1_score(y_val[:, ch], pred)

            if f1 > best_f1:
                best_f1 = f1
                best_th = th

        thresholds[ch] = best_th

    return thresholds

TH = find_best_thresholds(E_val, y_ch_val)
# TH = find_best_thresholds(E_val_anom, y_ch_val_anom)
# TH = find_best_thresholds(E_val_anom, y_ch_val_anom, metric="precision")

y_pred = (E_test > TH).astype(int)
# ============================================================
# Evaluation
# ============================================================

report = classification_report(
    y_ch_test,
    y_pred,
    target_names=[f"ch{i}" for i in range(y_ch_test.shape[1])],
    digits=4,
    output_dict=True
)

print(classification_report(
    y_ch_test,
    y_pred,
    target_names=[f"ch{i}" for i in range(y_ch_test.shape[1])],
    digits=4
))


# ============================================================
# Save everything
# ============================================================

EXP_DIR = os.path.join(RESULTS_DIR, f"window_{W_PARAM}")
os.makedirs(EXP_DIR, exist_ok=True)

# Save thresholds
np.save(os.path.join(EXP_DIR, "thresholds.npy"), TH)

# Save normalization params
np.save(os.path.join(EXP_DIR, "dwt_mean.npy"), mean)
np.save(os.path.join(EXP_DIR, "dwt_std.npy"), std)

# Save feature scaling
np.save(os.path.join(EXP_DIR, "input_mean.npy"), mean_feat)
np.save(os.path.join(EXP_DIR, "input_std.npy"), std_feat)

# Agregar esto en tu script, justo después de y_pred
from sklearn.metrics import hamming_loss

hamming_sc = 1 - hamming_loss(y_ch_test, y_pred)
exact_match = accuracy_score(y_ch_test, y_pred)

print(f"Hamming Score : {hamming_sc*100:.2f}%")
print(f"Exact Match   : {exact_match*100:.2f}%")

# Guardarlo en el JSON también
report["hamming_score"] = hamming_sc
report["exact_match_accuracy"] = exact_match

# with open(os.path.join(EXP_DIR, "classification_report.json"), "w") as f:
#     json.dump(report, f, indent=4)
# Save classification report
import json
with open(os.path.join(EXP_DIR, "classification_report.json"), "w") as f:
    json.dump(report, f, indent=4)

# Save confusion matrices per channel
conf_mats = {}
for ch in range(y_ch_test.shape[1]):
    cm = confusion_matrix(y_ch_test[:, ch], y_pred[:, ch])
    conf_mats[f"ch{ch}"] = cm.tolist()

with open(os.path.join(EXP_DIR, "confusion_matrices.json"), "w") as f:
    json.dump(conf_mats, f, indent=4)

print(f"\nResults saved in: {EXP_DIR}")
