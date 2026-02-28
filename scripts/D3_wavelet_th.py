import os
import argparse
import numpy as np
import tensorflow as tf
import pywt
import pickle
from sklearn.metrics import (
    f1_score,
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
RESULTS_DIR = "./results/D3_wavelet_th"

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# ============================================================
# Load data
# ============================================================

X_train = np.load(os.path.join(DATA_DIR, "train/X.npy"))
y_train = np.load(os.path.join(DATA_DIR, "train/y_bin.npy"))

X_val = np.load(os.path.join(DATA_DIR, "val/X.npy"))
y_val = np.load(os.path.join(DATA_DIR, "val/y_bin.npy"))

X_test = np.load(os.path.join(DATA_DIR, "test/X.npy"))
y_test = np.load(os.path.join(DATA_DIR, "test/y_bin.npy"))

print(f"\nTraining CNN for window size {W_PARAM}")
print(f"Train samples: {len(X_train)}")
# To remove pointing accuracy from training
X_train = X_train[:, :, 0:-1]
X_val   = X_val[:, :, 0:-1]
X_test  = X_test[:, :, 0:-1]

print(f"\nTrain data shape{X_train.shape}")
# print(X_train[0,:,-1])

# ============================================================
# Normalization
# ============================================================

X_train_nom = X_train[y_train == 0]

mean_feat = X_train_nom.mean(axis=(0,1))
std_feat  = X_train_nom.std(axis=(0,1)) + 1e-8
# Save scalers
SCALER_DIR = os.path.join(DATA_DIR, "scalers")
os.makedirs(SCALER_DIR, exist_ok=True)

with open(os.path.join(SCALER_DIR, "feature_scaler.pkl"), "wb") as f:
    pickle.dump({
        "mean": mean_feat,
        "std": std_feat
    }, f)

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
def adaptive_variance_detector(window, k=1.0):
    stds = np.std(window, axis=0)
    mu = np.mean(stds)
    sigma = np.std(stds)
    return 1 if np.any(stds > mu + k * sigma) else 0
from sklearn.metrics import f1_score

# For D coefficients
thresholds = np.linspace(0.002, 0.09, 30)
best_th = thresholds[0]
best_f1 = 0

for th in thresholds:
    y_val_pred = np.array([
        adaptive_variance_detector(w, th) for w in D_val
    ])
    f1 = f1_score(y_val, y_val_pred)
    if f1 > best_f1:
        best_f1 = f1
        best_th = th

print(f"Best threshold D = {best_th:.4f}, F1_val = {best_f1:.4f})")

y_test_predD = np.array([
    adaptive_variance_detector(w, best_th) for w in D_test
])


# For A coefficients
thresholds = np.linspace(0.002, 0.02, 30)
best_th = thresholds[0]
best_f1 = 0

for th in thresholds:
    y_val_pred = np.array([
        adaptive_variance_detector(w, th) for w in A_val
    ])
    f1 = f1_score(y_val, y_val_pred)
    if f1 > best_f1:
        best_f1 = f1
        best_th = th

print(f"Best threshold A= {best_th:.4f}, F1_val = {best_f1:.4f})")
y_test_predA = np.array([
    adaptive_variance_detector(w, best_th) for w in A_test
])

# ============================================================
# Evaluation
# ============================================================
# y_pred_prob = model.predict(X_test)
y_pred_wavelet = np.logical_or(y_test_predD, y_test_predA).astype(int)
# Metrics

f1 = f1_score(y_test, y_pred_wavelet)
# roc = roc_auc_score(y_test, y_pred_prob)
cm = confusion_matrix(y_test, y_pred_wavelet)

print("F1-score:", f1)
# print("ROC-AUC:", roc)
print("Confusion Matrix:\n", cm)
report = classification_report(y_test, y_pred_wavelet, digits=4)

print("\nTest Results:\n")
print(report)

# ============================================================
# Save model and results
# ============================================================

with open(os.path.join(RESULTS_DIR, f"variance_w{W_PARAM}.txt"), "w") as f:
    f.write(f"Best threshold: {best_th}\n")
    f.write(report)
print("\nResults saved successfully.")