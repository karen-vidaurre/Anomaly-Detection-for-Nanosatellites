import argparse
import logging
import os
import sys
import json

import pandas as pd
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate 3D LSTM datasets (Samples, TimeSteps, Features)."
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="data/raw/sim_001_full_overlaps.csv",
        help="Path to raw CSV.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/processed/lstm",
        help="Base directory to save processed datasets.",
    )
    parser.add_argument(
        "--window_size",
        type=int,
        default=20,
        help="Window size (number of time steps).",
    )
    parser.add_argument(
        "--step_size",
        type=int,
        default=1,
        help="Step size (stride) for sliding window.",
    )
    return parser.parse_args()


def load_data(data_path: str) -> pd.DataFrame:
    """
    Loads data from a CSV file.
    """
    if not os.path.exists(data_path):
        logger.error(f"Data file not found at {data_path}")
        sys.exit(1)

    logger.info(f"Loading data from {data_path}...")
    try:
        df = pd.read_csv(data_path)
        df.columns = [c.strip() for c in df.columns]

        # Optimize types
        float_cols = df.select_dtypes(include=["float64"]).columns
        if len(float_cols) > 0:
            df[float_cols] = df[float_cols].astype("float32")

        return df
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        sys.exit(1)


def validate_time_continuity(
    df: pd.DataFrame, time_col: str = "Time", expected_freq: float = 0.1
) -> None:
    """
    Ensures data doesn't have time gaps.
    """
    if time_col not in df.columns:
        logger.warning(
            f"Time column '{time_col}' not found. Skipping continuity check."
        )
        return

    logger.info("Running Time Continuity Sanity Check...")
    dt = df[time_col].diff().fillna(expected_freq)
    gaps = df[np.abs(dt) > (expected_freq * 1.5)]

    if len(gaps) > 0:
        logger.error(
            f"FATAL METHODOLOGY ERROR: Found {len(gaps)} time gaps in the data."
        )
        sys.exit(1)
    else:
        logger.info("Sanity Check Passed: Time continuity is valid.")


def create_sliding_windows(data: np.ndarray, window_size: int, step: int) -> np.ndarray:
    """
    Creates 3D sliding windows from 2D data using numpy stride tricks.
    Returns shape: (NumSamples, WindowSize, NumFeatures)
    """
    num_samples, num_features = data.shape
    num_windows = (num_samples - window_size) // step + 1

    if num_windows <= 0:
        return np.empty((0, window_size, num_features))

    row_stride = data.strides[0]
    col_stride = data.strides[1]

    new_strides = (row_stride * step, row_stride, col_stride)
    new_shape = (num_windows, window_size, num_features)

    windows = np.lib.stride_tricks.as_strided(
        data, shape=new_shape, strides=new_strides
    )

    return windows


def main():
    args = parse_arguments()

    # Create output directory
    out_path = os.path.join(args.output_dir, f"w{args.window_size}")
    os.makedirs(out_path, exist_ok=True)

    # Load Data
    df = load_data(args.data_path)
    validate_time_continuity(df)

    # Define Feature Columns (Raw Sensors)
    feature_prefixes = (
        "Vout_ss",
        "mout_mtq",
        "out_gps",
        "wout_gyro",
        "point_error",
        "Bout_magn",
    )
    feature_cols = [
        c for c in df.columns if any(c.startswith(p) for p in feature_prefixes)
    ]

    # Define Target Columns
    target_cols = [c for c in df.columns if c.startswith("fault_")]

    if not feature_cols:
        logger.error("No feature columns found.")
        sys.exit(1)

    logger.info(f"Selected {len(feature_cols)} feature columns.")

    # Extract Features and Targets
    # Convert to binary target immediately as per LSTM requirements
    y_binary = df[target_cols].max(axis=1).values.astype(int)
    X_raw = df[feature_cols].values.astype(np.float32)

    # Create Windows
    logger.info(
        f"Creating sliding windows (Size={args.window_size}, Step={args.step_size})..."
    )

    # Windowing X: (N_windows, WindowSize, Features)
    X_windows = create_sliding_windows(X_raw, args.window_size, args.step_size)

    # Windowing y: Take label at the end of each window
    y_windows = y_binary[args.window_size - 1 :: args.step_size]

    # Ensure lengths match
    if len(X_windows) != len(y_windows):
        min_len = min(len(X_windows), len(y_windows))
        X_windows = X_windows[:min_len]
        y_windows = y_windows[:min_len]

    logger.info(f"Windowed data shape: X={X_windows.shape}, y={y_windows.shape}")

    # --- Splitting (Train 70%, Val 15%, Test 15%) ---
    # We apply a GAP to prevent leakage between sets.

    N = len(X_windows)
    train_split_ratio = 0.70
    val_split_ratio = 0.15

    train_end = int(N * train_split_ratio)

    # Gap calculation in terms of indices (windows)
    # If Step=1, Window=20, we need to skip 20 windows to clear the buffer.
    gap_windows = int(np.ceil(args.window_size / args.step_size))

    val_start = train_end + gap_windows
    val_end_raw = int(N * (train_split_ratio + val_split_ratio))
    val_end = val_end_raw if val_start < val_end_raw else val_start

    test_start = val_end + gap_windows

    X_train = X_windows[:train_end]
    y_train = y_windows[:train_end]

    X_val = X_windows[val_start:val_end]
    y_val = y_windows[val_start:val_end]

    X_test = X_windows[test_start:]
    y_test = y_windows[test_start:]

    logger.info(f"Train: {len(X_train)} samples")
    logger.info(f"Val:   {len(X_val)} samples (Gap: {gap_windows})")
    logger.info(f"Test:  {len(X_test)} samples (Gap: {gap_windows})")

    # --- Standardization (Normalization) ---
    logger.info("Computing normalization stats on Nominal Training data...")

    # Nominal indices in training set
    nom_mask = y_train == 0
    X_train_nom = X_train[nom_mask]

    if len(X_train_nom) == 0:
        logger.warning(
            "No nominal data found in training set! Using all training data for stats."
        )
        X_train_ref = X_train
    else:
        X_train_ref = X_train_nom

    # Calculate Mean and Std
    # Axis=(0, 1) means we aggregate over Samples and TimeSteps, getting 1 mean per Feature.
    mean_feat = X_train_ref.mean(axis=(0, 1))
    std_feat = X_train_ref.std(axis=(0, 1)) + 1e-8

    def scale_array(X, mean, std):
        return (X - mean[None, None, :]) / std[None, None, :]

    X_train_scaled = scale_array(X_train, mean_feat, std_feat)
    X_val_scaled = scale_array(X_val, mean_feat, std_feat)
    X_test_scaled = scale_array(X_test, mean_feat, std_feat)

    # --- Save ---
    logger.info(f"Saving datasets to {out_path}...")

    np.save(os.path.join(out_path, "X_train.npy"), X_train_scaled)
    np.save(os.path.join(out_path, "y_train.npy"), y_train)

    np.save(os.path.join(out_path, "X_val.npy"), X_val_scaled)
    np.save(os.path.join(out_path, "y_val.npy"), y_val)

    np.save(os.path.join(out_path, "X_test.npy"), X_test_scaled)
    np.save(os.path.join(out_path, "y_test.npy"), y_test)

    # Save Scaler Params
    scaler_params = {
        "mean": mean_feat.tolist(),
        "std": std_feat.tolist(),
        "feature_names": feature_cols,
    }
    with open(os.path.join(out_path, "scaler_params.json"), "w") as f:
        json.dump(scaler_params, f, indent=4)

    logger.info("Done.")


if __name__ == "__main__":
    main()
