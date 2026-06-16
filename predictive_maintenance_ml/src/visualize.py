"""
Plotting utilities — all figures are saved to results/.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
from scipy.fft import fft, fftfreq

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results")
FS = 12_000

PALETTE = {
    "Normal":      "#2196F3",
    "Inner_Race":  "#F44336",
    "Ball":        "#FF9800",
    "Outer_Race":  "#4CAF50",
}


def _save(fig, name):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {path}")


def plot_raw_signals(df, n_per_class: int = 1):
    """Time-domain waveforms for one sample per class."""
    classes = sorted(df["label"].unique())
    fig, axes = plt.subplots(len(classes), 1, figsize=(12, 3 * len(classes)),
                             sharex=True)
    for ax, cls in zip(axes, classes):
        sig  = df[df["label"] == cls]["signal"].iloc[0]
        t    = np.arange(len(sig)) / FS * 1000   # ms
        color = PALETTE.get(cls, "steelblue")
        ax.plot(t, sig, color=color, linewidth=0.7)
        ax.set_ylabel("Acceleration (g)")
        ax.set_title(cls.replace("_", " "), fontsize=11, color=color)
        ax.grid(alpha=0.3)
    axes[-1].set_xlabel("Time (ms)")
    fig.suptitle("Raw Vibration Signals by Fault Class", fontsize=13, y=1.01)
    _save(fig, "01_raw_signals.png")


def plot_fft(df):
    """Frequency spectra for one sample per class."""
    classes = sorted(df["label"].unique())
    fig, axes = plt.subplots(len(classes), 1, figsize=(12, 3 * len(classes)),
                             sharex=True)
    for ax, cls in zip(axes, classes):
        sig   = df[df["label"] == cls]["signal"].iloc[0]
        n     = len(sig)
        freq  = fftfreq(n, d=1.0 / FS)[:n // 2]
        mag   = np.abs(fft(sig))[:n // 2] * 2 / n
        color = PALETTE.get(cls, "steelblue")
        ax.plot(freq, mag, color=color, linewidth=0.7)
        ax.set_ylabel("|X(f)|")
        ax.set_title(cls.replace("_", " "), fontsize=11, color=color)
        ax.grid(alpha=0.3)
    axes[-1].set_xlabel("Frequency (Hz)")
    fig.suptitle("FFT Spectra by Fault Class", fontsize=13, y=1.01)
    _save(fig, "02_fft_spectra.png")


def plot_feature_distributions(X, y, feature_names, label_map):
    """Box plots of key features per class."""
    inv_map  = {v: k for k, v in label_map.items()}
    classes  = [inv_map[i] for i in sorted(inv_map)]
    key_feat = ["kurtosis", "crest_factor", "envelope_kurtosis",
                "rms", "band_energy_0_3k"]
    idx      = [feature_names.index(f) for f in key_feat if f in feature_names]

    fig, axes = plt.subplots(1, len(idx), figsize=(4 * len(idx), 5))
    for ax, i in zip(axes, idx):
        data_by_class = [X[y == label_map[cls], i] for cls in classes]
        bp = ax.boxplot(data_by_class, patch_artist=True, notch=False,
                        medianprops=dict(color="black", linewidth=2))
        for patch, cls in zip(bp["boxes"], classes):
            patch.set_facecolor(PALETTE.get(cls, "steelblue"))
            patch.set_alpha(0.75)
        ax.set_xticklabels([c.replace("_", "\n") for c in classes], fontsize=8)
        ax.set_title(feature_names[i], fontsize=10)
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle("Feature Distributions by Fault Class", fontsize=13)
    _save(fig, "03_feature_distributions.png")


def plot_confusion_matrix(cm, label_names):
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=[l.replace("_", " ") for l in label_names],
        yticklabels=[l.replace("_", " ") for l in label_names],
        ax=ax,
    )
    ax.set_xlabel("Predicted", fontsize=11)
    ax.set_ylabel("Actual", fontsize=11)
    ax.set_title("Confusion Matrix — XGBoost", fontsize=13)
    _save(fig, "04_confusion_matrix.png")


def plot_cv_comparison(cv_results: dict):
    names  = list(cv_results.keys())
    means  = [cv_results[n]["mean_acc"] for n in names]
    stds   = [cv_results[n]["std_acc"]  for n in names]

    fig, ax = plt.subplots(figsize=(7, 4))
    colors  = ["#2196F3", "#F44336", "#FF9800"]
    bars    = ax.bar(names, means, color=colors, alpha=0.8, edgecolor="black")
    ax.errorbar(names, means, yerr=stds, fmt="none",
                color="black", capsize=6, linewidth=2)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("5-Fold CV Accuracy")
    ax.set_title("Classifier Comparison (5-Fold Cross-Validation)")
    ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1))
    for bar, m in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2,
                m + 0.01, f"{m:.1%}", ha="center", fontsize=11)
    ax.grid(axis="y", alpha=0.3)
    _save(fig, "05_classifier_comparison.png")


def plot_rul_predictions(rul_test, rul_pred):
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(rul_test, rul_pred, alpha=0.5, edgecolors="white",
               linewidths=0.3, color="#2196F3", s=30)
    lim = [0, 105]
    ax.plot(lim, lim, "r--", linewidth=1.5, label="Perfect prediction")
    ax.set_xlabel("True RUL (%)")
    ax.set_ylabel("Predicted RUL (%)")
    ax.set_title("Remaining Useful Life Prediction")
    ax.legend()
    ax.grid(alpha=0.3)
    _save(fig, "06_rul_prediction.png")


def plot_feature_importance(model, feature_names):
    """Random Forest / XGBoost feature importances."""
    clf = model.named_steps["clf"]
    if not hasattr(clf, "feature_importances_"):
        return
    imp  = clf.feature_importances_
    idx  = np.argsort(imp)[::-1]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(range(len(imp)), imp[idx], color="#2196F3", alpha=0.8,
           edgecolor="black")
    ax.set_xticks(range(len(imp)))
    ax.set_xticklabels([feature_names[i] for i in idx],
                       rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Importance")
    ax.set_title("Feature Importances (XGBoost)")
    ax.grid(axis="y", alpha=0.3)
    _save(fig, "07_feature_importance.png")
