"""
Fault classification models: Random Forest, SVM, XGBoost.
Also includes a Remaining Useful Life (RUL) regressor.
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score
)
import xgboost as xgb
import joblib
import os

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results")


def _make_rf():
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf",    RandomForestClassifier(
            n_estimators=200,
            max_depth=None,
            min_samples_split=4,
            random_state=42,
            n_jobs=-1,
        )),
    ])


def _make_svm():
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf",    SVC(
            kernel="rbf",
            C=10.0,
            gamma="scale",
            probability=True,
            random_state=42,
        )),
    ])


def _make_xgb():
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf",    xgb.XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            use_label_encoder=False,
            eval_metric="mlogloss",
            random_state=42,
            n_jobs=-1,
        )),
    ])


CLASSIFIERS = {
    "Random Forest": _make_rf,
    "SVM (RBF)":     _make_svm,
    "XGBoost":       _make_xgb,
}


def cross_validate_all(X: np.ndarray, y: np.ndarray, cv: int = 5) -> dict:
    """Run stratified k-fold CV on all classifiers and return mean accuracies."""
    skf     = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    results = {}
    for name, factory in CLASSIFIERS.items():
        model  = factory()
        scores = cross_val_score(model, X, y, cv=skf, scoring="accuracy", n_jobs=-1)
        results[name] = {
            "mean_acc": float(scores.mean()),
            "std_acc":  float(scores.std()),
            "scores":   scores.tolist(),
        }
        print(f"  {name:20s}  {scores.mean():.4f} ± {scores.std():.4f}")
    return results


def train_best_model(X_train, y_train, X_test, y_test,
                     label_names: list[str]) -> tuple:
    """
    Train XGBoost (typically best performer) and return
    (model, y_pred, report_str, conf_matrix).
    """
    model = _make_xgb()
    model.fit(X_train, y_train)
    y_pred  = model.predict(X_test)
    acc     = accuracy_score(y_test, y_pred)
    report  = classification_report(y_test, y_pred, target_names=label_names)
    cm      = confusion_matrix(y_test, y_pred)
    print(f"\nXGBoost test accuracy: {acc:.4f}")
    print(report)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    joblib.dump(model, os.path.join(RESULTS_DIR, "best_model.joblib"))
    return model, y_pred, report, cm, acc


# ── Remaining Useful Life regressor ───────────────────────────────────────────

def build_rul_targets(fault_sizes: np.ndarray, max_rul: float = 100.0) -> np.ndarray:
    """
    Simple proxy: map fault size (in inches) to a RUL percentage.
    0.000" → RUL = 100 %  (healthy)
    0.007" → RUL ≈ 65 %
    0.014" → RUL ≈ 30 %
    0.021" → RUL ≈  0 %
    """
    rul = max_rul * (1.0 - fault_sizes / 0.021)
    return np.clip(rul, 0, max_rul)


def train_rul_model(X_train, rul_train, X_test, rul_test):
    """Gradient Boosted Trees regressor for RUL prediction."""
    scaler  = StandardScaler()
    X_tr    = scaler.fit_transform(X_train)
    X_te    = scaler.transform(X_test)
    reg     = GradientBoostingRegressor(
        n_estimators=200, max_depth=4,
        learning_rate=0.05, random_state=42
    )
    reg.fit(X_tr, rul_train)
    rul_pred = reg.predict(X_te)
    rmse     = float(np.sqrt(np.mean((rul_pred - rul_test) ** 2)))
    print(f"RUL prediction RMSE: {rmse:.2f} %")
    joblib.dump((scaler, reg), os.path.join(RESULTS_DIR, "rul_model.joblib"))
    return reg, scaler, rul_pred, rmse
