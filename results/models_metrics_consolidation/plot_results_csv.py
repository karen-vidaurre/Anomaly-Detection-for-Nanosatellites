import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib import rcParams
import seaborn as sns

CSV_PATH   = "binary_classification_results_consolidated.csv"
OUTPUT_IMG = "accuracy_vs_window_size.png"
Y_COLUMN   = "accuracy"   

rcParams["font.family"]     = "DejaVu Sans"
rcParams["font.size"]       = 10
rcParams["axes.labelsize"]  = 11
rcParams["xtick.labelsize"] = 10
rcParams["ytick.labelsize"] = 10
rcParams["legend.fontsize"] = 8.5

df = pd.read_csv(CSV_PATH)
df["window_size"] = pd.to_numeric(df["window_size"], errors="coerce")
df[Y_COLUMN]      = pd.to_numeric(df[Y_COLUMN],      errors="coerce")
df = df.dropna(subset=["window_size", Y_COLUMN])

modelos = list(dict.fromkeys(df["modelo"].tolist()))
n       = len(modelos)

ws_vals = sorted(df["window_size"].unique())

palette = sns.color_palette("viridis", n)
color_map = {modelo: palette[i] for i, modelo in enumerate(modelos)}

fig, ax = plt.subplots(figsize=(8.5, 5.2))
fig.patch.set_facecolor("white")
ax.set_facecolor("white")
ax.grid(True, linestyle="--", linewidth=0.6, color="#cccccc", alpha=0.8, zorder=0)
handles, labels = [], []

for modelo in modelos:
    subset = df[df["modelo"] == modelo].sort_values("window_size")
    if subset.empty:
        continue

    color = color_map[modelo]

    line, = ax.plot(
        subset["window_size"],
        subset[Y_COLUMN],
        color=color,
        marker="o",
        markersize=5,
        linewidth=1.6,
        markerfacecolor=color,
        markeredgecolor=color,
        zorder=3,
        label=modelo,
    )
    handles.append(line)
    labels.append(modelo)

ax.set_xlabel("Window size", fontsize=11, labelpad=6)
ax.set_ylabel("Accuracy (%)", fontsize=11, labelpad=6)

ax.set_xticks(ws_vals)
ax.xaxis.set_minor_locator(ticker.NullLocator())

y_min = max(0,   df[Y_COLUMN].min() - 3)
y_max = min(100, df[Y_COLUMN].max() + 3)
ax.set_ylim(y_min, y_max)
ax.yaxis.set_major_locator(ticker.MultipleLocator(5))

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_color("#999999")
ax.spines["bottom"].set_color("#999999")
ax.tick_params(colors="#444444")

ax.legend(
    handles, labels,
    loc="center left",
    bbox_to_anchor=(1.01, 0.5),
    fontsize=8.5,
    frameon=True,
    framealpha=0.95,
    edgecolor="#dddddd",
    handlelength=2.2,
    handletextpad=0.5,
    borderpad=0.6,
    labelspacing=0.4,
)

plt.tight_layout(pad=0.5)
plt.savefig(OUTPUT_IMG, dpi=200, bbox_inches="tight", facecolor="white")
print(f" Saved : {OUTPUT_IMG}")
plt.show()
