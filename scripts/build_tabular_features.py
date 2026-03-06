import argparse
import json
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
    parser.add_argument(
        "--balance_train",
        action="store_true",
        help="If set, undersample majority class in training split only.",
    )
    parser.add_argument(
        "--random_seed",
        type=int,
        default=42,
        help="Random seed for balancing reproducibility.",
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
    Applies feature engineering and returns a dataframe with targets.

    Features computed (per raw sensor column):
      - Raw values
      - Delta (first difference): detects abrupt changes
      - Accel (second difference of delta): detects spike onset/offset edges
      - Rolling Variance (_var_{w}): variance-based anomalies
      - Rolling Mean (_mean_{w}): smooth reference signal
      - Rolling Min (_min_{w}): saturation low
      - Rolling Max (_max_{w}): saturation high
      - Rolling Median (_median_{w}): robust central tendency
      - Rolling Std (_std_{w}): noise increase anomalies
      - Rolling Kurtosis (_kurt_{w}): impulsive spike shape
      - Rolling Skewness (_skew_{w}): fault onset asymmetry
      - Rolling Range (_range_{w}): clipping/saturation width (free: max-min)
      - Deviation from Rolling Mean (_dev_{w}): bias and drift (free: raw-mean)

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

    # Delta (first difference)
    delta_df = df[raw_feature_cols].diff().fillna(0).astype("float32")
    delta_df.columns = [f"{c}_delta" for c in raw_feature_cols]
    features_list.append(delta_df)

    # Accel (second difference of delta): detects spike onset/offset edges
    delta2_df = delta_df.diff().fillna(0).astype("float32")
    delta2_df.columns = [f"{c}_accel" for c in raw_feature_cols]
    features_list.append(delta2_df)

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

        # Rolling Std: targets noise increase anomalies
        std_df = (
            df[raw_feature_cols].rolling(window=window_size).std().astype("float32")
        )
        std_df.columns = [f"{c}_std_{window_size}" for c in raw_feature_cols]
        features_list.append(std_df)

        # Rolling Kurtosis: targets impulsive spike shape
        kurt_df = (
            df[raw_feature_cols]
            .rolling(window=window_size)
            .kurt()
            .fillna(0)
            .astype("float32")
        )
        kurt_df.columns = [f"{c}_kurt_{window_size}" for c in raw_feature_cols]
        features_list.append(kurt_df)

        # Rolling Skewness: targets fault onset asymmetry
        skew_df = (
            df[raw_feature_cols]
            .rolling(window=window_size)
            .skew()
            .fillna(0)
            .astype("float32")
        )
        skew_df.columns = [f"{c}_skew_{window_size}" for c in raw_feature_cols]
        features_list.append(skew_df)

        # Rolling Range: targets saturation/clipping (free: reuses max_df, min_df)
        range_vals = (max_df.values - min_df.values).astype("float32")
        range_df = pd.DataFrame(
            range_vals,
            columns=[f"{c}_range_{window_size}" for c in raw_feature_cols],
            index=df.index,
        )
        features_list.append(range_df)

        # Deviation from Rolling Mean: targets bias/drift (free: reuses mean_df)
        dev_vals = (df[raw_feature_cols].values - mean_df.values).astype("float32")
        dev_df = pd.DataFrame(
            dev_vals,
            columns=[f"{c}_dev_{window_size}" for c in raw_feature_cols],
            index=df.index,
        )
        features_list.append(dev_df)

    X_full = pd.concat(features_list, axis=1)

    # Cleanup
    del features_list, delta_df, delta2_df
    if window_size > 1:
        del var_df, mean_df, min_df, max_df, median_df
        del std_df, kurt_df, skew_df, range_df, dev_df
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


def balance_training_set(
    train_df: pd.DataFrame, target_col: str, random_seed: int
) -> pd.DataFrame:
    """
    Random undersample majority class to match minority class size.

    Args:
        train_df: Training DataFrame with target column present.
        target_col: Binary target column name (0=nominal, 1=anomaly).
        random_seed: Seed for reproducibility.

    Returns:
        Balanced DataFrame with equal nominal and anomaly rows, shuffled.
    """
    nominal_idx = train_df.index[train_df[target_col] == 0].tolist()
    anomaly_idx = train_df.index[train_df[target_col] == 1].tolist()

    n_minority = min(len(nominal_idx), len(anomaly_idx))

    rng = np.random.default_rng(random_seed)
    if len(nominal_idx) > n_minority:
        nominal_idx = rng.choice(nominal_idx, size=n_minority, replace=False).tolist()
    else:
        anomaly_idx = rng.choice(anomaly_idx, size=n_minority, replace=False).tolist()

    balanced_df = train_df.loc[sorted(nominal_idx + anomaly_idx)]
    balanced_df = balanced_df.sample(frac=1, random_state=random_seed).reset_index(drop=True)

    logger.info(
        f"Training balanced: Nominal={len(nominal_idx)}, Anomaly={len(anomaly_idx)}"
        f" → balanced to {n_minority} each"
    )
    return balanced_df


def save_split_with_gap(
    df: pd.DataFrame,
    output_dir: str,
    gap: int,
    train_split_ratio: float = 0.7,
    val_split_ratio: float = 0.15,
    balance_train: bool = False,
    random_seed: int = 42,
) -> None:
    """
    Splits data time-wise into Train, Validation, and Test sets with a purge GAP.

    The GAP is essential for sliding window datasets to prevent data leakage.
    If Train ends at index T, and Window=5:
      - Train contains windows ending at 0..T (using data up to T).
      - Validation MUST start at T+5 to ensure its first window (using T..T+5)
        does not overlap with T.

    After splitting:
      1. Optionally balance training set (undersample majority class).
      2. Fit z-score scaler on nominal training rows only (matching LSTM pipeline).
      3. Apply scaler to all three splits.
      4. Save CSVs and scaler_params.json and dataset_info.json.

    Args:
        df (pd.DataFrame): The dataframe to split.
        output_dir (str): Directory to save split files.
        gap (int): Number of samples to drop between splits (should be window_size).
        train_split_ratio (float): Ratio of data for training.
        val_split_ratio (float): Ratio of data for validation.
        balance_train (bool): Whether to undersample majority class in training set.
        random_seed (int): Random seed for balancing.
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

    train_df = df.iloc[:train_end].copy()
    val_df = df.iloc[val_start:val_end].copy()
    test_df = df.iloc[test_start:].copy()

    # Identify target and feature columns
    target_cols = [c for c in df.columns if c.startswith("fault_") or c == "any_fault"]
    non_feature_cols = target_cols + (["Time"] if "Time" in df.columns else [])
    feature_cols = [c for c in df.columns if c not in non_feature_cols]

    # Determine binary target column for nominal mask
    if "any_fault" in target_cols:
        binary_target_col = "any_fault"
    else:
        binary_target_col = target_cols[0] if target_cols else None

    # Log class distributions
    for split_name, split_df in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
        n_rows = len(split_df)
        if n_rows == 0:
            logger.warning(f"  {split_name}: 0 rows — check split ratios and gap.")
            continue
        if target_cols:
            y_bin = (split_df[target_cols].max(axis=1) > 0).astype(int)
            n_nom = int((y_bin == 0).sum())
            n_anom = int((y_bin == 1).sum())
            logger.info(
                f"  {split_name} class balance: Nominal={n_nom} ({100*n_nom/n_rows:.1f}%),"
                f" Anomaly={n_anom} ({100*n_anom/n_rows:.1f}%)"
            )

    # 1. Balance training set (if requested)
    if balance_train and binary_target_col is not None:
        train_df = balance_training_set(train_df, binary_target_col, random_seed)

    # 2. Fit z-score scaler on nominal training rows only
    if binary_target_col is not None:
        nom_mask = train_df[binary_target_col] == 0
    else:
        nom_mask = pd.Series(True, index=train_df.index)

    nom_train = train_df.loc[nom_mask, feature_cols]
    feat_mean = nom_train.mean()
    feat_std = nom_train.std() + 1e-8

    # 3. Apply scaler to all splits
    train_df[feature_cols] = ((train_df[feature_cols] - feat_mean) / feat_std).astype("float32")
    val_df[feature_cols] = ((val_df[feature_cols] - feat_mean) / feat_std).astype("float32")
    test_df[feature_cols] = ((test_df[feature_cols] - feat_mean) / feat_std).astype("float32")

    os.makedirs(output_dir, exist_ok=True)

    # 4. Save normalized CSVs
    train_df.to_csv(os.path.join(output_dir, "train.csv"), index=False)
    val_df.to_csv(os.path.join(output_dir, "val.csv"), index=False)
    test_df.to_csv(os.path.join(output_dir, "test.csv"), index=False)

    logger.info(f"Saved split to {output_dir} (Gap={gap})")
    logger.info(f"  Train: {len(train_df)} rows")
    logger.info(f"  Val:   {len(val_df)} rows (dropped {gap} rows before)")
    logger.info(f"  Test:  {len(test_df)} rows (dropped {gap} rows before)")

    # 5. Save scaler_params.json
    scaler_params = {
        "feature_cols": feature_cols,
        "mean": feat_mean.tolist(),
        "std": feat_std.tolist(),
        "fit_on": "nominal_train_only",
        "n_nominal_train_rows": int(nom_mask.sum()),
    }
    with open(os.path.join(output_dir, "scaler_params.json"), "w") as f:
        json.dump(scaler_params, f, indent=2)
    logger.info(
        f"  Scaler saved (fit on {scaler_params['n_nominal_train_rows']} nominal train rows)"
    )

    # 6. Save dataset_info.json
    def split_stats(split_df):
        if len(split_df) == 0:
            return {"rows": 0, "n_nominal": 0, "n_anomaly": 0}
        y_bin = (split_df[target_cols].max(axis=1) > 0).astype(int) if target_cols else pd.Series(0, index=split_df.index)
        return {
            "rows": len(split_df),
            "n_nominal": int((y_bin == 0).sum()),
            "n_anomaly": int((y_bin == 1).sum()),
        }

    dataset_info = {
        "gap": gap,
        "train_split_ratio": train_split_ratio,
        "val_split_ratio": val_split_ratio,
        "binary_target": "any_fault" in target_cols,
        "balance_train": balance_train,
        "random_seed": random_seed,
        "n_features": len(feature_cols),
        "feature_cols": feature_cols,
        "splits": {
            "train": split_stats(train_df),
            "val": split_stats(val_df),
            "test": split_stats(test_df),
        },
    }
    with open(os.path.join(output_dir, "dataset_info.json"), "w") as f:
        json.dump(dataset_info, f, indent=2)
    logger.info(f"  Dataset info saved to {output_dir}/dataset_info.json")


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
    logger.info(f"Balance train: {args.balance_train}, Random seed: {args.random_seed}")

    df_raw = load_data(args.data_path)

    for w in window_sizes:
        logger.info(f"--- Processing Window Size: {w} ---")

        # Generate features and targets
        processed_df = generate_features(df_raw, w, args.binary_target)

        # Define output path
        # Structure: data/processed/w{size}/{mode}/
        out_path = os.path.join(args.output_dir, f"w{w}", mode_str)

        # Save split with GAP to prevent leakage
        save_split_with_gap(
            processed_df,
            out_path,
            gap=w,
            balance_train=args.balance_train,
            random_seed=args.random_seed,
        )

        del processed_df
        gc.collect()

    logger.info("Done.")


if __name__ == "__main__":
    main()
