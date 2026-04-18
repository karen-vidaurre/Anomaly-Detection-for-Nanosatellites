"""
SHAP interpretability analysis for binary fault-detection models.

Loads three pre-trained models and their test sets, computes SHAP values,
and produces global and sample-level outputs saved to results/shap/.

Models analysed
---------------
  dt_model_w15_binary_full   — Decision Tree, W=15
  xgb_model_w5_binary_full   — XGBoost, W=5
  rf_model_w5_binary_full    — Random Forest, W=5
"""

import matplotlib

matplotlib.use("Agg")

import os
import warnings

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.multioutput import MultiOutputClassifier

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(ROOT, "models")
DATA_DIR = os.path.join(ROOT, "data", "processed", "tabular")
OUT_DIR = os.path.join(ROOT, "results", "shap")

os.makedirs(OUT_DIR, exist_ok=True)

MODELS = [
    {
        "label": "dt",
        "w": "w15",
        "model_file": "dt_model_w15_binary_full.pkl",
        "test_csv": os.path.join(DATA_DIR, "w15", "binary", "test.csv"),
    },
    {
        "label": "xgb",
        "w": "w5",
        "model_file": "xgb_model_w5_binary_full.pkl",
        "test_csv": os.path.join(DATA_DIR, "w5", "binary", "test.csv"),
    },
    {
        "label": "rf",
        "w": "w5",
        "model_file": "rf_model_w5_binary_full.pkl",
        "test_csv": os.path.join(DATA_DIR, "w5", "binary", "test.csv"),
    },
]

THRESHOLD = 0.5
INTERACTION_SUBSAMPLE = 2000
INTERACTION_SEED = 42


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Stat types present in data when the models were trained.
# The data builder was later extended with _delta, _accel, _min_, _max_, _median_,
# but those columns were not included when these three models were fitted.
_ORIGINAL_STAT_SUFFIXES = ("_var_", "_std_", "_kurt_", "_skew_", "_range_", "_dev_", "_mean_")


def get_feature_target_cols(df: pd.DataFrame, expected_n_features: int = None):
    """Return (feature_cols, target_cols) matching the pipeline column logic.

    If expected_n_features is provided and the full feature set does not match,
    fall back to the original 7-stat filter that was active when the models were trained.
    """
    target_cols = [c for c in df.columns if c.startswith("fault_") or c == "any_fault"]
    feature_cols = [
        c for c in df.columns
        if c not in target_cols and c != "Time" and not c.startswith("point_error")
    ]

    if expected_n_features is not None and len(feature_cols) != expected_n_features:
        filtered = [c for c in feature_cols if any(s in c for s in _ORIGINAL_STAT_SUFFIXES)]
        if len(filtered) == expected_n_features:
            feature_cols = filtered
        else:
            raise ValueError(
                f"Cannot match model's expected {expected_n_features} features. "
                f"Full set has {len(feature_cols)}, stat filter gives {len(filtered)}."
            )

    return feature_cols, target_cols


def get_estimator_for_shap(model):
    """Unwrap MultiOutputClassifier if needed; return the underlying estimator."""
    if isinstance(model, MultiOutputClassifier):
        return model.estimators_[0]
    return model


def get_proba(model, X: pd.DataFrame) -> np.ndarray:
    """Return predicted fault probability (class 1) as a 1-D array."""
    if isinstance(model, MultiOutputClassifier):
        # list of (n_samples, 2) arrays, one per output — use first output
        return model.predict_proba(X)[0][:, 1]
    return model.predict_proba(X)[:, 1]


def select_samples(y_true: np.ndarray, proba: np.ndarray, threshold: float):
    """
    Return (tp_idx, fp_idx, tn_idx) as positions in the original array.

    tp — correctly classified fault with highest predicted probability
    fp — misclassified nominal with highest predicted probability
    tn — correctly classified nominal with lowest predicted probability
    """
    pred = (proba >= threshold).astype(int)

    tp_mask = (y_true == 1) & (pred == 1)
    fp_mask = (y_true == 0) & (pred == 1)
    tn_mask = (y_true == 0) & (pred == 0)

    def _argmax_in(mask):
        indices = np.where(mask)[0]
        if len(indices) == 0:
            return None
        return indices[np.argmax(proba[indices])]

    def _argmin_in(mask):
        indices = np.where(mask)[0]
        if len(indices) == 0:
            return None
        return indices[np.argmin(proba[indices])]

    return _argmax_in(tp_mask), _argmax_in(fp_mask), _argmin_in(tn_mask)


def save_current_figure(path: str, dpi: int = 300):
    plt.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close("all")


# ---------------------------------------------------------------------------
# Main analysis loop
# ---------------------------------------------------------------------------

summary_lines = []
xgb_artifacts = {}  # store for interaction computation
all_sample_info = {}  # {tag: {case: (idx, proba)}} for final summary

for cfg in MODELS:
    label = cfg["label"]
    w = cfg["w"]
    tag = f"{label}_{w}"

    print(f"\n{'='*60}")
    print(f"  Processing: {tag}")
    print(f"{'='*60}")

    # --- Load model and data ---
    model_path = os.path.join(MODEL_DIR, cfg["model_file"])
    model = joblib.load(model_path)
    print(f"  Loaded model: {cfg['model_file']}")

    df = pd.read_csv(cfg["test_csv"])
    print(f"  Loaded test CSV: {cfg['test_csv']}  shape={df.shape}")

    estimator = get_estimator_for_shap(model)
    feature_cols, target_cols = get_feature_target_cols(df, expected_n_features=estimator.n_features_in_)
    X_test = df[feature_cols]
    y_test = df["any_fault"].values.astype(int)

    proba = get_proba(model, X_test)

    # --- SHAP explainer ---
    if label == "xgb":
        explainer = shap.TreeExplainer(estimator, model_output="raw")
    else:
        explainer = shap.TreeExplainer(estimator)

    print(f"  Computing SHAP values (this may take a moment)…")
    shap_exp = explainer(X_test)

    # For sklearn binary classifiers the Explanation has shape (n, f, 2);
    # index class 1 (fault).  For XGB raw output it is already (n, f).
    if shap_exp.values.ndim == 3:
        shap_exp_fault = shap_exp[:, :, 1]
    else:
        shap_exp_fault = shap_exp

    # --- Global: beeswarm ---
    beeswarm_path = os.path.join(OUT_DIR, f"shap_beeswarm_{tag}.png")
    shap.plots.beeswarm(shap_exp_fault, max_display=20, show=False)
    save_current_figure(beeswarm_path)
    print(f"  Saved: {beeswarm_path}")
    summary_lines.append(beeswarm_path)

    # --- Global: bar ---
    bar_path = os.path.join(OUT_DIR, f"shap_bar_{tag}.png")
    shap.plots.bar(shap_exp_fault, max_display=20, show=False)
    save_current_figure(bar_path)
    print(f"  Saved: {bar_path}")
    summary_lines.append(bar_path)

    # --- Global: CSV ---
    mean_abs = np.abs(shap_exp_fault.values).mean(axis=0)
    global_df = (
        pd.DataFrame({"Feature": feature_cols, "MeanAbsSHAP": mean_abs})
        .sort_values("MeanAbsSHAP", ascending=False)
        .reset_index(drop=True)
    )
    csv_path = os.path.join(OUT_DIR, f"shap_global_{tag}.csv")
    global_df.to_csv(csv_path, index=False)
    print(f"  Saved: {csv_path}")
    summary_lines.append(csv_path)

    # --- Sample selection ---
    tp_idx, fp_idx, tn_idx = select_samples(y_test, proba, THRESHOLD)
    cases = {"tp": tp_idx, "fp": fp_idx, "tn": tn_idx}

    print(f"  Selected samples:")
    for case, idx in cases.items():
        if idx is None:
            print(f"    {case}: NOT FOUND in test set")
        else:
            print(f"    {case}: index={idx}  p(fault)={proba[idx]:.4f}")

    # --- Waterfall plots ---
    for case, idx in cases.items():
        wf_path = os.path.join(OUT_DIR, f"shap_waterfall_{tag}_{case}.png")
        if idx is None:
            print(f"  Skipping waterfall ({case}): no matching sample")
            continue
        shap.plots.waterfall(shap_exp_fault[idx], max_display=15, show=False)
        save_current_figure(wf_path)
        print(f"  Saved: {wf_path}")
        summary_lines.append(wf_path)

    # Store sample info for final summary
    all_sample_info[tag] = {
        case: (idx, float(proba[idx]) if idx is not None else None)
        for case, idx in cases.items()
    }

    # Stash XGB artifacts for interaction computation
    if label == "xgb":
        xgb_artifacts["model"] = estimator
        xgb_artifacts["X_test"] = X_test
        xgb_artifacts["feature_cols"] = feature_cols

# ---------------------------------------------------------------------------
# XGB pairwise interaction values
# ---------------------------------------------------------------------------

print(f"\n{'='*60}")
print("  Computing XGB SHAP interaction values…")
print(f"{'='*60}")

xgb_model = xgb_artifacts["model"]
X_full = xgb_artifacts["X_test"]
feat_cols = xgb_artifacts["feature_cols"]

n_sub = min(INTERACTION_SUBSAMPLE, len(X_full))
X_sub = X_full.sample(n=n_sub, random_state=INTERACTION_SEED)
print(f"  Subsampled to {n_sub} rows (seed={INTERACTION_SEED})")

interact_explainer = shap.TreeExplainer(xgb_model, model_output="raw")
interaction_vals = interact_explainer.shap_interaction_values(X_sub)  # (n, f, f)

mean_abs_inter = np.abs(interaction_vals).mean(axis=0)  # (f, f)
np.fill_diagonal(mean_abs_inter, 0.0)

n_feat = len(feat_cols)
rows = []
for i in range(n_feat):
    for j in range(i + 1, n_feat):
        rows.append((feat_cols[i], feat_cols[j], mean_abs_inter[i, j]))

interact_df = (
    pd.DataFrame(rows, columns=["Feature_A", "Feature_B", "MeanAbsInteraction"])
    .sort_values("MeanAbsInteraction", ascending=False)
    .head(20)
    .reset_index(drop=True)
)

interact_path = os.path.join(OUT_DIR, "shap_interactions_xgb_w5.csv")
interact_df.to_csv(interact_path, index=False)
print(f"  Saved: {interact_path}")
summary_lines.append(interact_path)

# ---------------------------------------------------------------------------
# Final console summary
# ---------------------------------------------------------------------------

print(f"\n{'='*60}")
print("  SUMMARY — Files saved")
print(f"{'='*60}")
for path in summary_lines:
    print(f"  {path}")

print(f"\n{'='*60}")
print("  SUMMARY — Selected sample indices and probabilities")
print(f"{'='*60}")

for cfg in MODELS:
    tag = f"{cfg['label']}_{cfg['w']}"
    print(f"\n  {tag}:")
    for case, (idx, p) in all_sample_info.get(tag, {}).items():
        if idx is None:
            print(f"    {case}: NOT FOUND")
        else:
            print(f"    {case}: index={idx}  p(fault)={p:.4f}")
