import os
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--window_size", type=int, required=True)
args = parser.parse_args()

WINDOW_SIZE = args.window_size

BASE_DIR = "./data/processed"
SAVE_DIR = os.path.join(BASE_DIR, f"windows_{WINDOW_SIZE}")

TRAIN_RATIO = 0.7
VAL_RATIO = 0.15

X = np.load(os.path.join(SAVE_DIR, "X_windows.npy"))
y_bin = np.load(os.path.join(SAVE_DIR, "y_binary.npy"))
y_ch  = np.load(os.path.join(SAVE_DIR, "y_channel.npy"))

N, W, F = X.shape

print(f"\nLoaded dataset windows_{WINDOW_SIZE}")
print(f"Total windows: {N}")
print(f"Anomaly %: {100 * sum(y_bin)/N:.2f}%\n")


def save_split_with_gap_npy(X, y_bin, y_ch, output_dir, gap):

    N = len(X)

    train_end = int(N * TRAIN_RATIO)

    val_start = train_end + gap
    val_end = int(N * (TRAIN_RATIO + VAL_RATIO))
    test_start = val_end + gap

    splits = {
        "train": (0, train_end),
        "val":   (val_start, val_end),
        "test":  (test_start, N),
    }

    for name, (i_start, i_end) in splits.items():
        split_dir = os.path.join(output_dir, name)
        os.makedirs(split_dir, exist_ok=True)

        np.save(os.path.join(split_dir, "X.npy"), X[i_start:i_end])
        np.save(os.path.join(split_dir, "y_bin.npy"), y_bin[i_start:i_end])
        np.save(os.path.join(split_dir, "y_ch.npy"), y_ch[i_start:i_end])

        print(f"{name.upper()}: {i_end - i_start} samples")


save_split_with_gap_npy(
    X,
    y_bin,
    y_ch,
    SAVE_DIR,
    gap=W  # clave para evitar leakage
)

print("Split completed.\n")