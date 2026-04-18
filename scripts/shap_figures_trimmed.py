"""
Regenerate trimmed SHAP figures for the paper's interpretability section.

Produces four outputs with reduced display parameters (top-10 features):
  shap_beeswarm_xgb_w5_top10.png   — (10, 5) in, 300 DPI
  shap_beeswarm_rf_w5_top10.png    — (10, 5) in, 300 DPI
  shap_beeswarm_dt_w15_top10.png   — (10, 5) in, 300 DPI
  shap_waterfall_xgb_w5_fp_top10.png — (10, 6) in, 300 DPI

SHAP values are recomputed from the pre-trained models and test CSVs;
no cached arrays are required.
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

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(ROOT, "models")
DATA_DIR = os.path.join(ROOT, "data", "processed", "tabular")
OUT_DIR = os.path.join(ROOT, "results", "shap")

os.makedirs(OUT_DIR, exist_ok=True)

MODELS = [
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
    {
        "label": "dt",
        "w": "w15",
        "model_file": "dt_model_w15_binary_full.pkl",
        "test_csv": os.path.join(DATA_DIR, "w15", "binary", "test.csv"),
    },
]

THRESHOLD = 0.5
_ORIGINAL_STAT_SUFFIXES = ("_var_", "_std_", "_kurt_", "_skew_", "_range_", "_dev_", "_mean_")


def get_feature_target_cols(df, expected_n_features=None):
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
    if isinstance(model, MultiOutputClassifier):
        return model.estimators_[0]
    return model


def get_proba(model, X):
    if isinstance(model, MultiOutputClassifier):
        return model.predict_proba(X)[0][:, 1]
    return model.predict_proba(X)[:, 1]


def select_fp(y_true, proba, threshold):
    """Return index of the false positive with the highest predicted probability."""
    pred = (proba >= threshold).astype(int)
    fp_mask = (y_true == 0) & (pred == 1)
    indices = np.where(fp_mask)[0]
    if len(indices) == 0:
        return None
    return indices[np.argmax(proba[indices])]


def save_figure(path, figsize, dpi=300):
    plt.gcf().set_size_inches(figsize)
    plt.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close("all")
    print(f"  Saved: {path}")


for cfg in MODELS:
    label = cfg["label"]
    w = cfg["w"]
    tag = f"{label}_{w}"

    print(f"\n{'='*60}")
    print(f"  Processing: {tag}")
    print(f"{'='*60}")

    model = joblib.load(os.path.join(MODEL_DIR, cfg["model_file"]))
    df = pd.read_csv(cfg["test_csv"])
    estimator = get_estimator_for_shap(model)
    feature_cols, _ = get_feature_target_cols(df, expected_n_features=estimator.n_features_in_)
    X_test = df[feature_cols]
    y_test = df["any_fault"].values.astype(int)

    if label == "xgb":
        explainer = shap.TreeExplainer(estimator, model_output="raw")
    else:
        explainer = shap.TreeExplainer(estimator)

    print("  Computing SHAP values…")
    shap_exp = explainer(X_test)

    if shap_exp.values.ndim == 3:
        shap_exp_fault = shap_exp[:, :, 1]
    else:
        shap_exp_fault = shap_exp

    # Beeswarm — top 10
    beeswarm_path = os.path.join(OUT_DIR, f"shap_beeswarm_{tag}_top10.png")
    shap.plots.beeswarm(shap_exp_fault, max_display=10, show=False)
    save_figure(beeswarm_path, figsize=(10, 5))

    # Waterfall for XGB false positive only
    if label == "xgb":
        proba = get_proba(model, X_test)
        fp_idx = select_fp(y_test, proba, THRESHOLD)
        if fp_idx is None:
            print("  WARNING: no false positive found in XGB test set — skipping waterfall")
        else:
            print(f"  XGB FP sample: index={fp_idx}  p(fault)={proba[fp_idx]:.4f}")
            wf_path = os.path.join(OUT_DIR, "shap_waterfall_xgb_w5_fp_top10.png")
            shap.plots.waterfall(shap_exp_fault[fp_idx], max_display=10, show=False)
            save_figure(wf_path, figsize=(10, 6))

print(f"\n{'='*60}")
print("  Done.")
print(f"{'='*60}")
