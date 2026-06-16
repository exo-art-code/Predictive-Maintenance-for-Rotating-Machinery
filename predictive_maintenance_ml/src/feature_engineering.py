"""
Time-domain and frequency-domain feature extraction from vibration signals.

These are standard features used in condition monitoring of rotating machinery.
Each feature has a physical interpretation — comments explain the mechanics.
"""

import numpy as np
import pandas as pd
from scipy.fft import fft, fftfreq
from scipy.stats import kurtosis, skew


FS = 12_000   # CWRU sampling frequency (Hz)


# ── Time-domain features ───────────────────────────────────────────────────────

def rms(x: np.ndarray) -> float:
    """Root Mean Square — energy of the signal, sensitive to load changes."""
    return float(np.sqrt(np.mean(x ** 2)))


def peak_value(x: np.ndarray) -> float:
    return float(np.max(np.abs(x)))


def crest_factor(x: np.ndarray) -> float:
    """Peak / RMS — high value indicates impulsive (fault) events."""
    r = rms(x)
    return float(peak_value(x) / r) if r > 0 else 0.0


def kurtosis_val(x: np.ndarray) -> float:
    """
    4th statistical moment — excellent early-fault indicator.
    Healthy bearings ~3 (Gaussian); faults push this to 6+.
    """
    return float(kurtosis(x))


def skewness(x: np.ndarray) -> float:
    return float(skew(x))


def shape_factor(x: np.ndarray) -> float:
    """RMS / mean(|x|) — dimensionless, load-independent."""
    mean_abs = np.mean(np.abs(x))
    return float(rms(x) / mean_abs) if mean_abs > 0 else 0.0


def impulse_factor(x: np.ndarray) -> float:
    mean_abs = np.mean(np.abs(x))
    return float(peak_value(x) / mean_abs) if mean_abs > 0 else 0.0


def clearance_factor(x: np.ndarray) -> float:
    sqrt_mean = np.mean(np.sqrt(np.abs(x))) ** 2
    return float(peak_value(x) / sqrt_mean) if sqrt_mean > 0 else 0.0


# ── Frequency-domain features ──────────────────────────────────────────────────

def _spectrum(x: np.ndarray):
    n    = len(x)
    freq = fftfreq(n, d=1.0 / FS)[:n // 2]
    mag  = np.abs(fft(x))[:n // 2] * 2 / n
    return freq, mag


def mean_frequency(x: np.ndarray) -> float:
    """Weighted centroid of the power spectrum."""
    freq, mag = _spectrum(x)
    power = mag ** 2
    total = power.sum()
    return float((freq * power).sum() / total) if total > 0 else 0.0


def spectral_rms(x: np.ndarray) -> float:
    _, mag = _spectrum(x)
    return float(np.sqrt(np.mean(mag ** 2)))


def spectral_kurtosis_mean(x: np.ndarray) -> float:
    """Mean of local spectral kurtosis — sensitive to non-stationary impulses."""
    _, mag = _spectrum(x)
    return float(kurtosis(mag))


def band_energy_ratio(x: np.ndarray,
                      low: float = 0, high: float = 3000) -> float:
    """
    Fraction of energy in a specific band relative to total.
    Fault characteristic frequencies fall in well-defined bands —
    this captures energy concentration there.
    """
    freq, mag = _spectrum(x)
    mask  = (freq >= low) & (freq < high)
    total = (mag ** 2).sum()
    return float((mag[mask] ** 2).sum() / total) if total > 0 else 0.0


def peak_frequency(x: np.ndarray) -> float:
    """Frequency at maximum spectral amplitude."""
    freq, mag = _spectrum(x)
    return float(freq[np.argmax(mag)])


# ── Envelope analysis (demodulation) ──────────────────────────────────────────

def envelope_rms(x: np.ndarray) -> float:
    """
    RMS of the amplitude envelope (Hilbert transform).
    Fault impulses amplitude-modulate the high-frequency carrier;
    demodulation isolates this modulation.
    """
    from scipy.signal import hilbert
    analytic = hilbert(x)
    envelope = np.abs(analytic)
    return float(rms(envelope))


def envelope_kurtosis(x: np.ndarray) -> float:
    from scipy.signal import hilbert
    analytic = hilbert(x)
    envelope = np.abs(analytic)
    return float(kurtosis(envelope))


# ── Feature vector ─────────────────────────────────────────────────────────────

FEATURE_FUNCS = {
    "rms":                rms,
    "peak":               peak_value,
    "crest_factor":       crest_factor,
    "kurtosis":           kurtosis_val,
    "skewness":           skewness,
    "shape_factor":       shape_factor,
    "impulse_factor":     impulse_factor,
    "clearance_factor":   clearance_factor,
    "mean_frequency":     mean_frequency,
    "spectral_rms":       spectral_rms,
    "spectral_kurtosis":  spectral_kurtosis_mean,
    "band_energy_0_3k":   lambda x: band_energy_ratio(x, 0, 3000),
    "band_energy_3k_6k":  lambda x: band_energy_ratio(x, 3000, 6000),
    "peak_frequency":     peak_frequency,
    "envelope_rms":       envelope_rms,
    "envelope_kurtosis":  envelope_kurtosis,
}

FEATURE_NAMES = list(FEATURE_FUNCS.keys())


def extract_features(signal: np.ndarray) -> np.ndarray:
    return np.array([fn(signal) for fn in FEATURE_FUNCS.values()])


def build_feature_matrix(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, list]:
    """
    Convert a DataFrame of raw segments into a (X, y, feature_names) tuple.
    """
    X = np.vstack([extract_features(sig) for sig in df["signal"]])
    label_map = {lab: i for i, lab in enumerate(sorted(df["label"].unique()))}
    y = df["label"].map(label_map).to_numpy()
    return X, y, FEATURE_NAMES, label_map
