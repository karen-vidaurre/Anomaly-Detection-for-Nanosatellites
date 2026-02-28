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
RESULTS_DIR = "./results/D1_variance_th"

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
SCALER_DIR = os.path.join(DATA_DIR, "scalers_detection")
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

def variance_based_detector(window, threshold_std):
    stds = np.std(window, axis=0)
    return 1 if np.any(stds > threshold_std) else 0

# Select best threshold

thresholds = np.linspace(0.5, 3.0, 50)
best_th = thresholds[0]
best_f1 = 0

for th in thresholds:
    y_val_pred = np.array([
        variance_based_detector(w, th) for w in X_val
    ])
    f1 = f1_score(y_val, y_val_pred)
    if f1 > best_f1:
        best_f1 = f1
        best_th = th

print(f"Best threshold = {best_th:.4f}, F1_val = {best_f1:.4f}")

y_test_pred = np.array([
    variance_based_detector(w, best_th) for w in X_test
])
f1 = f1_score(y_test, y_test_pred)
cm = confusion_matrix(y_test, y_test_pred)

print("F1-score:", f1)
# print("ROC-AUC:", roc)
print("Confusion Matrix:\n", cm)

# print("\nClassification Report:\n")
# print(classification_report(y_test, y_test_pred))
report = classification_report(y_test, y_test_pred, digits=4)

print("\nTest Results:\n")
print(report)
with open(os.path.join(RESULTS_DIR, f"variance_w{W_PARAM}.txt"), "w") as f:
    f.write(f"Best threshold: {best_th}\n")
    f.write(report)
