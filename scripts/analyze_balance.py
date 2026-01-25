import pandas as pd

file_path = '/home/chris/Documents/Anomaly-Detection-for-Nanosatellites/data/raw/sim_balanced_dataset.csv'

# Read only the header first to identify fault columns
df_head = pd.read_csv(file_path, nrows=0)
fault_cols = [c for c in df_head.columns if c.startswith('fault_')]

print(f"Analyzing balance for {len(fault_cols)} fault columns: {fault_cols}")

# Read the file (using chunksize if it was huge, but 92MB fits in memory easily)
df = pd.read_csv(file_path)

total_rows = len(df)
print(f"Total rows: {total_rows}")

# 1. Individual fault counts
print("\n--- Individual Fault Counts ---")
fault_counts = df[fault_cols].sum().sort_values(ascending=False)
for col, count in fault_counts.items():
    percentage = (count / total_rows) * 100
    print(f"{col}: {count} ({percentage:.2f}%)")

# 2. Total faults per row
df['total_faults'] = df[fault_cols].sum(axis=1)

# 3. Normal vs Faulty
normal_count = (df['total_faults'] == 0).sum()
faulty_count = (df['total_faults'] > 0).sum()

print("\n--- Global Balance ---")
print(f"Normal (No Faults): {normal_count} ({(normal_count/total_rows)*100:.2f}%)")
print(f"Faulty (At least one): {faulty_count} ({(faulty_count/total_rows)*100:.2f}%)")

# 4. Multi-label check
multi_fault_count = (df['total_faults'] > 1).sum()
print(f"\nMulti-fault rows (simultaneous faults): {multi_fault_count} ({(multi_fault_count/total_rows)*100:.2f}%)")

# 5. Fault distribution in faulty rows
if faulty_count > 0:
    print("\n--- Distribution within Faulty Data ---")
    for col, count in fault_counts.items():
        pct_of_faulty = (count / faulty_count) * 100
        print(f"{col}: {pct_of_faulty:.2f}% of faulty samples")
