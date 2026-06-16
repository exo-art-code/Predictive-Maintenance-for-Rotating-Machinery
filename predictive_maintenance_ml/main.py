"""
Predictive Maintenance for Rotating Machinery — main pipeline.

Usage:
    python main.py

Steps:
  1. Download CWRU bearing dataset
  2. Extract time & frequency domain features
  3. Compare classifiers (RF, SVM, XGBoost) via 5-fold CV
  4. Train best model, evaluate on held-out test set
  5. Predict Remaining Useful Life (RUL)
  6. Save plots to results/
"""

import numpy as np
from sklearn.model_selection import train_test_split

from src.data_loader      import download_and_load
from src.feature_engineering import build_feature_matrix, FEATURE_NAMES
from src.models           import (
    cross_validate_all, train_best_model,
    build_rul_targets, train_rul_model,
)
from src.visualize        import (
    plot_raw_signals, plot_fft, plot_feature_distributions,
    plot_confusion_matrix, plot_cv_comparison,
    plot_rul_predictions, plot_feature_importance,
)


def main():
    print("=" * 60)
    print("  Predictive Maintenance — CWRU Bearing Fault Detection")
    print("=" * 60)

    # ── 1. Load data ───────────────────────────────────────────────
    print("\n[1/6] Loading dataset …")
    df = download_and_load(max_segments_per_file=60)

    # ── 2. Visualise raw signals ───────────────────────────────────
    print("\n[2/6] Generating signal plots …")
    plot_raw_signals(df)
    plot_fft(df)

    # ── 3. Feature extraction ──────────────────────────────────────
    print("\n[3/6] Extracting features …")
    X, y, feat_names, label_map = build_feature_matrix(df)
    print(f"  Feature matrix shape: {X.shape}")
    print(f"  Classes: {list(label_map.keys())}")

    plot_feature_distributions(X, y, feat_names, label_map)

    # ── 4. Cross-validate classifiers ─────────────────────────────
    print("\n[4/6] Cross-validating classifiers (5-fold) …")
    cv_results = cross_validate_all(X, y, cv=5)
    plot_cv_comparison(cv_results)

    # ── 5. Train best model on 80/20 split ────────────────────────
    print("\n[5/6] Training XGBoost on 80 % split …")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    label_names = sorted(label_map, key=label_map.get)
    model, y_pred, report, cm, acc = train_best_model(
        X_train, y_train, X_test, y_test, label_names
    )
    plot_confusion_matrix(cm, label_names)
    plot_feature_importance(model, feat_names)

    # ── 6. RUL prediction ─────────────────────────────────────────
    print("\n[6/6] Remaining Useful Life prediction …")
    fault_sizes = df["fault_size"].to_numpy()
    rul_all     = build_rul_targets(fault_sizes)

    # Use the same 80/20 split indices
    idx       = np.arange(len(X))
    idx_tr, idx_te = train_test_split(idx, test_size=0.2,
                                      stratify=y, random_state=42)
    reg, scaler, rul_pred, rmse = train_rul_model(
        X[idx_tr], rul_all[idx_tr],
        X[idx_te],  rul_all[idx_te],
    )
    plot_rul_predictions(rul_all[idx_te], rul_pred)

    # ── Summary ───────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  Results Summary")
    print("=" * 60)
    for name, res in cv_results.items():
        print(f"  {name:20s}  CV acc = {res['mean_acc']:.2%} ± {res['std_acc']:.2%}")
    print(f"\n  XGBoost test accuracy : {acc:.2%}")
    print(f"  RUL prediction RMSE   : {rmse:.2f} %")
    print("\n  All plots saved to results/")
    print("=" * 60)


if __name__ == "__main__":
    main()
