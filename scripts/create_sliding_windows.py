import argparse
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

def analyze_dataset(X, y_bin, y_ch, fault_cols, output_cols, df, window_size, output_dir):

    fig_dir = os.path.join(output_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    # ============================
    # 1. Binary Balance
    # ============================
    plt.figure()
    unique, counts = np.unique(y_bin, return_counts=True)
    plt.bar(['Nominal (0)', 'Anomaly (1)'], counts)
    plt.title(f"Binary Anomaly Balance (W={window_size})")
    plt.ylabel("Number of Windows")
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "binary_balance.png"), dpi=300)
    plt.close()

    # ============================
    # 2. Faults per channel
    # ============================
    fault_counts = y_ch.sum(axis=0)
    plt.figure(figsize=(10,5))
    plt.bar(fault_cols, fault_counts)
    plt.xticks(rotation=90)
    plt.title(f"Fault Count per Channel (W={window_size})")
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "fault_per_channel.png"), dpi=300)
    plt.close()

    # ============================
    # 3. PCA
    # ============================
    X_flat = X.reshape(X.shape[0], -1)
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_flat)

    plt.figure()
    plt.scatter(X_pca[:,0], X_pca[:,1], c=y_bin, s=5)
    plt.title(f"PCA Projection (W={window_size})")
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "pca_2d.png"), dpi=300)
    plt.close()

    # ============================
    # 4. t-SNE (subset for speed)
    # ============================

    subset = min(5000, len(X_flat))
    idx = np.random.choice(len(X_flat), subset, replace=False)

    # PCA
    pca = PCA(n_components=min(30, X_flat.shape[1]))
    X_reduced = pca.fit_transform(X_flat[idx])

    # t-SNE 
    tsne = TSNE(
        n_components=2,
        perplexity=30,
        learning_rate=200,
        max_iter=1000,
        init="pca",
        random_state=42
    )

    X_tsne = tsne.fit_transform(X_reduced)

    plt.figure(figsize=(6,5))
    plt.scatter(
        X_tsne[:, 0],
        X_tsne[:, 1],
        c=y_bin[idx],
        s=5,
        cmap="coolwarm"
    )

    plt.title(f"t-SNE Projection (W={window_size})")
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "tsne_2d.png"), dpi=300)
    plt.close()

    # ============================
    # 5. Correlation
    # ============================
    plt.figure(figsize=(8,6))
    sns.heatmap(df[output_cols].corr(), cmap="viridis")
    plt.title(f"Output Correlation (W={window_size})")
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "correlation.png"), dpi=300)
    plt.close()

    # ============================
    # 6. Fault Co-occurrence
    # ============================
    co_matrix = np.zeros((len(fault_cols), len(fault_cols)))

    for row in y_ch:
        co_matrix += np.outer(row, row)

    plt.figure(figsize=(8,7))
    sns.heatmap(co_matrix, xticklabels=fault_cols,
                yticklabels=fault_cols)
    plt.title(f"Fault Co-Occurrence (W={window_size})")
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "fault_cooccurrence.png"), dpi=300)
    plt.close()

    print("Figures saved to:", fig_dir)

def main(args):

    print("Loading dataset:", args.input_csv)
    df = pd.read_csv(args.input_csv)

    # Drop first row if needed
    # df = df.iloc[1:].reset_index(drop=True)

    print("Dataset shape:", df.shape)

    fault_cols = [c for c in df.columns if c.startswith("fault_")]
    output_cols = [c for c in df.columns if c not in fault_cols and c.lower() != "time"]

    print("Fault columns:", len(fault_cols))
    print("Output columns:", len(output_cols))

    window_size = args.window_size

    X = []
    y_bin = []
    y_ch = []

    for start in range(0, len(df) - window_size + 1):
        end = start + window_size
        window = df.iloc[start:end]

        X.append(window[output_cols].values)

        # Binary label
        y_bin.append(int(window[fault_cols].values.max() == 1))

        # Multi-label
        y_ch_vec = (window[fault_cols].values.max(axis=0)).astype(int)
        y_ch.append(y_ch_vec)

    X = np.array(X)
    y_bin = np.array(y_bin)
    y_ch = np.array(y_ch)

    print("Final dataset:")
    print("X:", X.shape)
    print("Binary distribution:", np.bincount(y_bin))
    print("Multi-label shape:", y_ch.shape)

    os.makedirs(args.output_dir, exist_ok=True)

    np.save(os.path.join(args.output_dir, "X_windows.npy"), X)
    np.save(os.path.join(args.output_dir, "y_binary.npy"), y_bin)
    np.save(os.path.join(args.output_dir, "y_channel.npy"), y_ch)

    print("Saved to:", args.output_dir)
    
    if args.analyze:
        analyze_dataset(
        X, y_bin, y_ch,
        fault_cols, output_cols,
        df, window_size,
        args.output_dir
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--window_size", type=int, default=20)
    parser.add_argument("--analyze", action="store_true",
                    help="Generate dataset analysis figures")
    args = parser.parse_args()
    main(args)

