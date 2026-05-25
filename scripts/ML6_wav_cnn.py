import os
import argparse
import numpy as np
import pywt
import json
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    hamming_loss,
    accuracy_score
)
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks

# ============================================================
# Arguments
# ============================================================

parser = argparse.ArgumentParser()
parser.add_argument("--window_size", type=int, default=20)
parser.add_argument("--epochs",      type=int, default=50)
parser.add_argument("--batch_size",  type=int, default=64)
args = parser.parse_args()

W_PARAM    = args.window_size
EPOCHS     = args.epochs
BATCH_SIZE = args.batch_size

BASE_DIR    = "./data/processed"
DATA_DIR    = os.path.join(BASE_DIR, f"windows_{W_PARAM}")
MODEL_DIR   = "./models"
RESULTS_DIR = "./results/CNN_wav_multilabel"

os.makedirs(MODEL_DIR,   exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# ============================================================
# Load data
# ============================================================

X_train   = np.load(os.path.join(DATA_DIR, "train/X.npy"))
y_train   = np.load(os.path.join(DATA_DIR, "train/y_bin.npy"))
y_ch_train= np.load(os.path.join(DATA_DIR, "train/y_ch.npy"))

X_val     = np.load(os.path.join(DATA_DIR, "val/X.npy"))
y_val     = np.load(os.path.join(DATA_DIR, "val/y_bin.npy"))
y_ch_val  = np.load(os.path.join(DATA_DIR, "val/y_ch.npy"))

X_test    = np.load(os.path.join(DATA_DIR, "test/X.npy"))
y_test    = np.load(os.path.join(DATA_DIR, "test/y_bin.npy"))
y_ch_test = np.load(os.path.join(DATA_DIR, "test/y_ch.npy"))

# Remover pointing accuracy (igual que en tu script original)
X_train = X_train[:, :, 0:-1]
X_val   = X_val[:,   :, 0:-1]
X_test  = X_test[:,  :, 0:-1]

N_CHANNELS = y_ch_train.shape[1]  # 15
print(f"\nCNN Wavelet Multilabel — window size {W_PARAM}")
print(f"Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")
print(f"Input shape: {X_train.shape} | Labels: {N_CHANNELS} channels")

# ============================================================
# Normalización (igual que tu script — solo sobre muestras nominales)
# ============================================================

mean_feat = X_train[y_train == 0].mean(axis=(0, 1))
std_feat  = X_train[y_train == 0].std(axis=(0, 1))  + 1e-8

def scale_windows(X, mean, std):
    return (X - mean[None, None, :]) / std[None, None, :]

X_train = scale_windows(X_train, mean_feat, std_feat)
X_val   = scale_windows(X_val,   mean_feat, std_feat)
X_test  = scale_windows(X_test,  mean_feat, std_feat)

# ============================================================
# DWT — coeficientes de detalle D (igual que en thresholding)
# ============================================================

def compute_dwt_windows(X, wavelet="db4", level=1):
    if X.ndim == 2:
        X = X[..., np.newaxis]
    N, W, F = X.shape
    D_list = []
    for i in range(N):
        D_ch = []
        for ch in range(F):
            coeffs = pywt.wavedec(X[i, :, ch], wavelet, level=level)
            D = np.concatenate(coeffs[1:])  # solo detalle
            D_ch.append(D)
        D_list.append(np.stack(D_ch, axis=-1))  # (D_len, F)
    return np.array(D_list)  # (N, D_len, F)

print("\nComputando DWT...")
D_train = compute_dwt_windows(X_train)
D_val   = compute_dwt_windows(X_val)
D_test  = compute_dwt_windows(X_test)

# Normalización sobre los coeficientes de detalle
mean_d = D_train.mean(axis=(0, 1), keepdims=True)
std_d  = D_train.std(axis=(0, 1),  keepdims=True) + 1e-8

D_train = (D_train - mean_d) / std_d
D_val   = (D_val   - mean_d) / std_d
D_test  = (D_test  - mean_d) / std_d

print(f"Shape DWT input para CNN: {D_train.shape}")
# Esperado: (N, D_len, 15) — D_len depende del window size y level

# ============================================================
# Modelo CNN — 15 cabezas sigmoid
# ============================================================

def build_cnn_multilabel(input_shape, n_labels=15):
    """
    input_shape: (D_len, n_features) — coeficientes de detalle normalizados
    n_labels: número de canales ADCS (15)
    
    Arquitectura:
      Conv1D x2 → GlobalAvgPool → Dense → Dropout → Dense(15, sigmoid)
    Pequeña por diseño para ser compatible con STM32 si se cuantiza.
    """
    inp = layers.Input(shape=input_shape, name="dwt_detail_input")

    # Bloque convolucional 1
    x = layers.Conv1D(32, kernel_size=3, padding="same", activation="relu",
                      name="conv1")(inp)
    x = layers.BatchNormalization(name="bn1")(x)

    # Bloque convolucional 2
    x = layers.Conv1D(64, kernel_size=3, padding="same", activation="relu",
                      name="conv2")(x)
    x = layers.BatchNormalization(name="bn2")(x)

    # Bloque convolucional 3 — captura patrones a mayor escala
    x = layers.Conv1D(64, kernel_size=5, padding="same", activation="relu",
                      name="conv3")(x)
    x = layers.BatchNormalization(name="bn3")(x)

    # Pooling global — colapsa la dimensión temporal
    x = layers.GlobalAveragePooling1D(name="gap")(x)

    # Cabeza densa
    x = layers.Dense(128, activation="relu", name="dense1")(x)
    x = layers.Dropout(0.3, name="dropout")(x)

    # 15 salidas independientes con sigmoid
    # Cada neurona aprende su propio threshold por canal
    out = layers.Dense(n_labels, activation="sigmoid", name="output")(x)

    model = models.Model(inputs=inp, outputs=out, name="CNN_Wavelet_Multilabel")
    return model

input_shape = D_train.shape[1:]  # (D_len, n_features)
model = build_cnn_multilabel(input_shape, n_labels=N_CHANNELS)
model.summary()

# ============================================================
# Compilación — Binary Crossentropy por canal (multilabel estándar)
# ============================================================

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="binary_crossentropy",
    metrics=[
        tf.keras.metrics.AUC(multi_label=True, name="auc"),
        tf.keras.metrics.BinaryAccuracy(name="bin_acc")
    ]
)

# ============================================================
# Callbacks
# ============================================================

EXP_DIR = os.path.join(RESULTS_DIR, f"window_{W_PARAM}")
os.makedirs(EXP_DIR, exist_ok=True)

cb_list = [
    # Detener si val_loss no mejora en 10 epochs
    callbacks.EarlyStopping(
        monitor="val_loss", patience=10,
        restore_best_weights=True, verbose=1
    ),
    # Guardar el mejor modelo
    callbacks.ModelCheckpoint(
        filepath=os.path.join(MODEL_DIR, f"cnn_multilabel_w{W_PARAM}.keras"),
        monitor="val_loss", save_best_only=True, verbose=1
    ),
    # Reducir LR si se estanca
    callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.5,
        patience=5, min_lr=1e-6, verbose=1
    ),
    # Log de entrenamiento
    callbacks.CSVLogger(os.path.join(EXP_DIR, "training_log.csv"))
]

# ============================================================
# Entrenamiento
# ============================================================

print(f"\nEntrenando CNN multilabel — W={W_PARAM}, epochs={EPOCHS}, batch={BATCH_SIZE}")

history = model.fit(
    D_train, y_ch_train,
    validation_data=(D_val, y_ch_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=cb_list,
    verbose=1
)

# ============================================================
# Evaluación — threshold 0.5 por defecto
# ============================================================

y_prob = model.predict(D_test, verbose=0)       # (N, 15) probabilidades
y_pred = (y_prob >= 0.5).astype(int)            # binarizar

# --- Búsqueda de threshold óptimo por canal sobre validación ---
# (igual que tu thresholding, pero sobre probabilidades en vez de energía)
def find_best_thresholds_prob(y_prob_val, y_true_val):
    thresholds = np.zeros(y_prob_val.shape[1])
    for ch in range(y_prob_val.shape[1]):
        best_th, best_f1 = 0.5, 0.0
        for th in np.linspace(0.1, 0.9, 80):
            pred = (y_prob_val[:, ch] >= th).astype(int)
            from sklearn.metrics import f1_score
            f1 = f1_score(y_true_val[:, ch], pred, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_th = th
        thresholds[ch] = best_th
    return thresholds

y_prob_val = model.predict(D_val, verbose=0)
TH_opt = find_best_thresholds_prob(y_prob_val, y_ch_val)
y_pred_opt = (y_prob >= TH_opt).astype(int)

print("\n" + "="*60)
print("RESULTADOS CON THRESHOLD = 0.5")
print("="*60)
report_05 = classification_report(
    y_ch_test, y_pred,
    target_names=[f"ch{i}" for i in range(N_CHANNELS)],
    digits=4, output_dict=True
)
print(classification_report(
    y_ch_test, y_pred,
    target_names=[f"ch{i}" for i in range(N_CHANNELS)],
    digits=4
))

print("\n" + "="*60)
print("RESULTADOS CON THRESHOLD ÓPTIMO POR CANAL (val F1)")
print("="*60)
report_opt = classification_report(
    y_ch_test, y_pred_opt,
    target_names=[f"ch{i}" for i in range(N_CHANNELS)],
    digits=4, output_dict=True
)
print(classification_report(
    y_ch_test, y_pred_opt,
    target_names=[f"ch{i}" for i in range(N_CHANNELS)],
    digits=4
))

# ============================================================
# Hamming Score y Exact Match
# ============================================================

for label, y_p, rep in [("th=0.5", y_pred, report_05),
                         ("th=opt", y_pred_opt, report_opt)]:
    hs = 1 - hamming_loss(y_ch_test, y_p)
    em = accuracy_score(y_ch_test, y_p)
    rep["hamming_score"]         = float(hs)
    rep["exact_match_accuracy"]  = float(em)
    rep["optimal_thresholds"]    = TH_opt.tolist() if label == "th=opt" else [0.5]*N_CHANNELS
    print(f"[{label}] Hamming Score: {hs*100:.2f}% | Exact Match: {em*100:.2f}%")

# ============================================================
# Guardar resultados
# ============================================================

np.save(os.path.join(EXP_DIR, "y_prob.npy"),     y_prob)
np.save(os.path.join(EXP_DIR, "y_pred_05.npy"),  y_pred)
np.save(os.path.join(EXP_DIR, "y_pred_opt.npy"), y_pred_opt)
np.save(os.path.join(EXP_DIR, "thresholds_opt.npy"), TH_opt)
np.save(os.path.join(EXP_DIR, "dwt_mean.npy"),   mean_d)
np.save(os.path.join(EXP_DIR, "dwt_std.npy"),    std_d)
np.save(os.path.join(EXP_DIR, "input_mean.npy"), mean_feat)
np.save(os.path.join(EXP_DIR, "input_std.npy"),  std_feat)

with open(os.path.join(EXP_DIR, "report_th05.json"), "w") as f:
    json.dump(report_05, f, indent=4)

with open(os.path.join(EXP_DIR, "report_th_opt.json"), "w") as f:
    json.dump(report_opt, f, indent=4)

# Confusion matrices
conf_mats = {}
for ch in range(N_CHANNELS):
    cm = confusion_matrix(y_ch_test[:, ch], y_pred_opt[:, ch])
    conf_mats[f"ch{ch}"] = cm.tolist()

with open(os.path.join(EXP_DIR, "confusion_matrices.json"), "w") as f:
    json.dump(conf_mats, f, indent=4)

# Tamaño del modelo
model_path = os.path.join(MODEL_DIR, f"cnn_multilabel_w{W_PARAM}.keras")
model_size_kb = os.path.getsize(model_path) / 1024 if os.path.exists(model_path) else 0
print(f"\nModel size: {model_size_kb:.3f} KB")
print(f"Results saved in: {EXP_DIR}")