import argparse
import logging
import os
import sys
import gc
from typing import List, Tuple

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
        description="Generate train/test datasets with sliding window features."
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="data/raw/sim_357_bal_t1.csv",
        help="Path to raw CSV.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/processed/tabular",
        help="Base directory to save processed datasets.",
    )
    parser.add_argument(
        "--window_sizes",
        type=str,
        default="1,5,10,15,20",
        help="Comma-separated list of window sizes to generate (e.g., '1,5,10').",
    )
    parser.add_argument(
        "--binary_target",
        action="store_true",
        help="If set, generates binary targets (0/1) instead of multi-label.",
    )
    return parser.parse_args()


def load_data(data_path: str) -> pd.DataFrame:
    """
    Loads data from a CSV file.

    Args:
        data_path (str): Path to the CSV file.

    Returns:
        pd.DataFrame: Loaded data.
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
    Ensures data doesn't have time gaps. Gaps corrupt rolling window calculations
    because rolling() assumes adjacent rows are adjacent in time.

    Args:
        df (pd.DataFrame): Dataframe containing a time column.
        time_col (str): Name of the time column.
        expected_freq (float): Expected sampling rate in seconds.
    """
    if time_col not in df.columns:
        logger.warning(
            f"Time column '{time_col}' not found. Skipping continuity check."
        )
        return

    logger.info("Running Time Continuity Sanity Check...")

    # Calculate time difference between rows
    # We fillna with expected_freq because the first row has no previous row
    dt = df[time_col].diff().fillna(expected_freq)

    # Check for gaps (allowing for tiny floating point errors, e.g., 1e-4)
    # If gap is significantly larger than expected (e.g. > 1.5x), it is a break.
    gaps = df[np.abs(dt) > (expected_freq * 1.5)]

    if len(gaps) > 0:
        logger.error(
            f"FATAL METHODOLOGY ERROR: Found {len(gaps)} time gaps in the data."
        )
        logger.error(
            f"Example gap at index {gaps.index[0]}: Delta was {dt.loc[gaps.index[0]]}"
        )
        logger.error(
            "Rolling window features will be mathematically incorrect across these gaps."
        )
        logger.error(
            "Please fix the raw CSV (fill gaps) or split processing by segment."
        )
        sys.exit(1)  # Fail fast to protect research integrity
    else:
        logger.info("Sanity Check Passed: Time continuity is valid.")


def generate_features(
    df: pd.DataFrame, window_size: int, binary_target: bool
) -> pd.DataFrame:
    """
    Applies feature engineering:
    - Delta
    - Rolling Mean / Var / Min / Max / Median

    Applies global truncation and returns computed dataframe with targets.

    Args:
        df (pd.DataFrame): Input dataframe.
        window_size (int): Window size for feature engineering.
        binary_target (bool): Whether to use binary target.

    Returns:
        pd.DataFrame: Computed dataframe with features and targets.
    """
    MAX_WINDOW_SIZE = 20

    # Sanity Check
    validate_time_continuity(df)

    # Base dataset feature column names prefixes
    feature_prefixes = (
        "Vout_ss",
        "mout_mtq",
        "out_gps",
        "wout_gyro",
        "Bout_magn",
    )
    raw_feature_cols = [
        c for c in df.columns if any(c.startswith(p) for p in feature_prefixes)
    ]

    # Target column names prefixes
    target_cols = [c for c in df.columns if c.startswith("fault_")]

    if not raw_feature_cols:
        raise ValueError("No feature columns found.")
    if not target_cols:
        raise ValueError("No target columns found.")

    # --- Feature Engineering ---
    features_list = []

    # Keeps Time if present (Useful for visualization later)
    if "Time" in df.columns:
        features_list.append(df[["Time"]])

    features_list.append(df[raw_feature_cols])

    # Delta
    delta_df = df[raw_feature_cols].diff().fillna(0).astype("float32")
    delta_df.columns = [f"{c}_delta" for c in raw_feature_cols]
    features_list.append(delta_df)

    if window_size > 1:
        # Rolling Variance
        var_df = (
            df[raw_feature_cols].rolling(window=window_size).var().astype("float32")
        )
        var_df.columns = [f"{c}_var_{window_size}" for c in raw_feature_cols]
        features_list.append(var_df)

        # Rolling Mean
        mean_df = (
            df[raw_feature_cols].rolling(window=window_size).mean().astype("float32")
        )
        mean_df.columns = [f"{c}_mean_{window_size}" for c in raw_feature_cols]
        features_list.append(mean_df)

        # Rolling Min
        min_df = (
            df[raw_feature_cols].rolling(window=window_size).min().astype("float32")
        )
        min_df.columns = [f"{c}_min_{window_size}" for c in raw_feature_cols]
        features_list.append(min_df)

        # Rolling Max
        max_df = (
            df[raw_feature_cols].rolling(window=window_size).max().astype("float32")
        )
        max_df.columns = [f"{c}_max_{window_size}" for c in raw_feature_cols]
        features_list.append(max_df)

        # Rolling Median
        median_df = (
            df[raw_feature_cols].rolling(window=window_size).median().astype("float32")
        )
        median_df.columns = [f"{c}_median_{window_size}" for c in raw_feature_cols]
        features_list.append(median_df)

    X_full = pd.concat(features_list, axis=1)

    # Cleanup
    del features_list, delta_df
    if window_size > 1:
        del var_df, mean_df, min_df, max_df, median_df
    gc.collect()

    # Targets
    y_raw = df[target_cols]
    if binary_target:
        y_final = y_raw.max(axis=1).to_frame(name="any_fault")
    else:
        y_final = y_raw

    data_combined = pd.concat([X_full, y_final], axis=1)

    del X_full, y_final
    gc.collect()

    # --- Global Truncation ---
    # We drop the first MAX_WINDOW_SIZE rows for ALL window sizes to ensure
    # that different window sizes can be compared on the EXACT same timeframe.
    logger.info(f"Applying Global Truncation: Dropping first {MAX_WINDOW_SIZE} rows.")
    data_combined = data_combined.iloc[MAX_WINDOW_SIZE:].reset_index(drop=True)

    # Drop NaNs that might have been created by diff/rolling
    initial_len = len(data_combined)
    data_combined = data_combined.dropna()
    dropped_len = initial_len - len(data_combined)

    if dropped_len > 0:
        logger.warning(f"Dropped {dropped_len} additional rows due to NaNs.")

    return data_combined


def save_split_with_gap(
    df: pd.DataFrame,
    output_dir: str,
    gap: int,
    train_split_ratio: float = 0.7,
    val_split_ratio: float = 0.15,
) -> None:
    """
    Splits data time-wise into Train, Validation, and Test sets with a purge GAP.

    The GAP is essential for sliding window datasets to prevent data leakage.
    If Train ends at index T, and Window=5:
      - Train contains windows ending at 0..T (using data up to T).
      - Validation MUST start at T+5 to ensure its first window (using T..T+5)
        does not overlap with T.

    Args:
        df (pd.DataFrame): The dataframe to split.
        output_dir (str): Directory to save split files.
        gap (int): Number of samples to drop between splits (should be window_size).
        train_split_ratio (float): Ratio of data for training.
        val_split_ratio (float): Ratio of data for validation.
    """
    n = len(df)

    # Indices
    train_end = int(n * train_split_ratio)

    # Val starts after gap
    val_start = train_end + gap
    val_end_raw = int(n * (train_split_ratio + val_split_ratio))
    # Ensure Val has size
    if val_start >= val_end_raw:
        logger.warning(f"Gap {gap} consumes entire validation set! Check ratios.")
        val_end = val_start  # Empty
    else:
        val_end = val_end_raw

    # Test starts after gap (relative to val_end)
    test_start = val_end + gap

    train_df = df.iloc[:train_end]
    val_df = df.iloc[val_start:val_end]
    test_df = df.iloc[test_start:]

    os.makedirs(output_dir, exist_ok=True)

    train_path = os.path.join(output_dir, "train.csv")
    val_path = os.path.join(output_dir, "val.csv")
    test_path = os.path.join(output_dir, "test.csv")

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)

    logger.info(f"Saved split to {output_dir} (Gap={gap})")
    logger.info(f"  Train: {len(train_df)} rows")
    logger.info(f"  Val:   {len(val_df)} rows (dropped {gap} rows before)")
    logger.info(f"  Test:  {len(test_df)} rows (dropped {gap} rows before)")


def main():
    args = parse_arguments()

    # Parse sizes
    try:
        window_sizes = [int(x.strip()) for x in args.window_sizes.split(",")]
    except ValueError:
        logger.error(
            "Invalid format for window_sizes. Use comma separated integers (e.g. '1,5')."
        )
        sys.exit(1)

    mode_str = "binary" if args.binary_target else "multi"
    logger.info(f"Generating datasets for mode: {mode_str}")
    logger.info(f"Window sizes: {window_sizes}")

    df_raw = load_data(args.data_path)

    for w in window_sizes:
        logger.info(f"--- Processing Window Size: {w} ---")

        # Generate features and targets
        processed_df = generate_features(df_raw, w, args.binary_target)

        # Define output path
        # Structure: data/processed/w{size}/{mode}/
        out_path = os.path.join(args.output_dir, f"w{w}", mode_str)

        # Save split with GAP to prevent leakage
        save_split_with_gap(processed_df, out_path, gap=w)

        del processed_df
        gc.collect()

    logger.info("Done.")


if __name__ == "__main__":
    main()
