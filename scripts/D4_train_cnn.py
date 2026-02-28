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
RESULTS_DIR = "./results"

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

    N, W, F = X.shape
    D_list = []

    for i in range(N):
        D_ch = []
        for ch in range(F):
            coeffs = pywt.wavedec(X[i, :, ch], wavelet, level=level)
            D = np.concatenate(coeffs[1:])
            D_ch.append(D)

        D_list.append(np.stack(D_ch, axis=1))

    return np.array(D_list)

D_train = compute_dwt_windows(X_train)
D_val   = compute_dwt_windows(X_val)
D_test  = compute_dwt_windows(X_test)

# Normalization DWT (only train)
mean = np.mean(D_train, axis=(0,1), keepdims=True)
std  = np.std(D_train, axis=(0,1), keepdims=True) + 1e-8
# Save scalers DWT
with open(os.path.join(SCALER_DIR, "dwt_scaler.pkl"), "wb") as f:
    pickle.dump({
        "mean": mean,
        "std": std
    }, f)

D_train = (D_train - mean) / std
D_val   = (D_val   - mean) / std
D_test  = (D_test  - mean) / std

# ============================================================
# Model CNN
# ============================================================

model = tf.keras.Sequential([
    tf.keras.Input(shape=(D_train.shape[1], D_train.shape[2])),

    tf.keras.layers.Conv1D(8, 3, padding="same", activation="relu"),
    tf.keras.layers.BatchNormalization(),

    tf.keras.layers.Conv1D(8, 3, padding="same", activation="relu"),
    tf.keras.layers.GlobalAveragePooling1D(),

    tf.keras.layers.Dense(8, activation="relu"),
    tf.keras.layers.Dense(1, activation="sigmoid")
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-3),
    loss="binary_crossentropy",
    metrics=[
        "accuracy",
        tf.keras.metrics.AUC(name="auc")
    ]
)

callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=10,
        restore_best_weights=True
    ),
    tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(MODEL_DIR, f"cnn_w{W_PARAM}.keras"),
        monitor="val_loss",
        save_best_only=True
    )
]

history = model.fit(
    D_train,
    y_train,
    validation_data=(D_val, y_val),
    epochs=30,
    batch_size=64,
    callbacks=callbacks, 
    verbose=1
)

# ============================================================
# Evaluation
# ============================================================
# y_pred_prob = model.predict(X_test)
y_pred = (model.predict(D_test) > 0.5).astype(int)

# Métricas

f1 = f1_score(y_test, y_pred)
# roc = roc_auc_score(y_test, y_pred_prob)
cm = confusion_matrix(y_test, y_pred)

print("F1-score:", f1)
# print("ROC-AUC:", roc)
print("Confusion Matrix:\n", cm)

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))
report = classification_report(y_test, y_pred, digits=4)

print("\nTest Results:\n")
print(report)

# ============================================================
# Save model and results
# ============================================================

model.save(os.path.join(MODEL_DIR, f"cnn_w{W_PARAM}.keras"))

np.save(os.path.join(RESULTS_DIR, f"w{W_PARAM}_history.npy"), history.history)

with open(os.path.join(RESULTS_DIR, f"w{W_PARAM}_classification.txt"), "w") as f:
    f.write(report)
total_params = model.count_params()
size_mb = total_params * 4 / (1024 * 1024)

print(f"Approx model size in memory: {size_mb:.2f} MB")
print("\nModel and results saved successfully.")