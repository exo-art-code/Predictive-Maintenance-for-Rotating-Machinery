"""
Downloads and loads CWRU Bearing Fault Dataset.

The CWRU dataset contains vibration signals from drive-end and fan-end
accelerometers under four health conditions:
  - Normal
  - Inner Race Fault (IRF)
  - Ball Fault (BF)
  - Outer Race Fault (ORF)

Fault sizes: 0.007", 0.014", 0.021" inches
Load conditions: 0, 1, 2, 3 HP
"""

import os
import requests
import numpy as np
import pandas as pd
from tqdm import tqdm
import scipy.io as sio

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
BASE_URL = "https://engineering.case.edu/sites/default/files/"

# (filename, label, fault_size_inches, load_hp)
CWRU_FILES = [
    # Normal baseline
    ("97.mat",  "Normal", 0.000, 0),
    ("98.mat",  "Normal", 0.000, 1),
    ("99.mat",  "Normal", 0.000, 2),
    ("100.mat", "Normal", 0.000, 3),
    # Inner Race Fault — 0.007"
    ("105.mat", "Inner_Race", 0.007, 0),
    ("106.mat", "Inner_Race", 0.007, 1),
    ("107.mat", "Inner_Race", 0.007, 2),
    ("108.mat", "Inner_Race", 0.007, 3),
    # Inner Race Fault — 0.014"
    ("169.mat", "Inner_Race", 0.014, 0),
    ("170.mat", "Inner_Race", 0.014, 1),
    ("171.mat", "Inner_Race", 0.014, 2),
    ("172.mat", "Inner_Race", 0.014, 3),
    # Ball Fault — 0.007"
    ("118.mat", "Ball", 0.007, 0),
    ("119.mat", "Ball", 0.007, 1),
    ("120.mat", "Ball", 0.007, 2),
    ("121.mat", "Ball", 0.007, 3),
    # Ball Fault — 0.014"
    ("185.mat", "Ball", 0.014, 0),
    ("186.mat", "Ball", 0.014, 1),
    ("187.mat", "Ball", 0.014, 2),
    ("188.mat", "Ball", 0.014, 3),
    # Outer Race Fault — 0.007"  (centred @6)
    ("130.mat", "Outer_Race", 0.007, 0),
    ("131.mat", "Outer_Race", 0.007, 1),
    ("132.mat", "Outer_Race", 0.007, 2),
    ("133.mat", "Outer_Race", 0.007, 3),
    # Outer Race Fault — 0.014"
    ("197.mat", "Outer_Race", 0.014, 0),
    ("198.mat", "Outer_Race", 0.014, 1),
    ("199.mat", "Outer_Race", 0.014, 2),
    ("200.mat", "Outer_Race", 0.014, 3),
]

SEGMENT_LEN = 1024   # samples per segment (~85 ms at 12 kHz)
STEP        = 512    # 50 % overlap


def _download_file(filename: str) -> str:
    os.makedirs(DATA_DIR, exist_ok=True)
    local_path = os.path.join(DATA_DIR, filename)
    if os.path.exists(local_path):
        return local_path
    url = BASE_URL + filename
    resp = requests.get(url, stream=True, timeout=30)
    resp.raise_for_status()
    with open(local_path, "wb") as f:
        for chunk in resp.iter_content(chunk_size=8192):
            f.write(chunk)
    return local_path


def _extract_signal(mat_path: str) -> np.ndarray:
    """Return drive-end vibration array from a .mat file."""
    data = sio.loadmat(mat_path)
    # Key names vary — find the drive-end DE key
    for key in data:
        if "DE" in key and not key.startswith("_"):
            return data[key].flatten().astype(np.float64)
    # Fallback: return first numeric array
    for key in data:
        if not key.startswith("_"):
            arr = data[key]
            if isinstance(arr, np.ndarray) and arr.ndim >= 1:
                return arr.flatten().astype(np.float64)
    raise ValueError(f"No vibration signal found in {mat_path}")


def _segment(signal: np.ndarray, length: int = SEGMENT_LEN, step: int = STEP):
    """Sliding-window segmentation."""
    segments = []
    start = 0
    while start + length <= len(signal):
        segments.append(signal[start : start + length])
        start += step
    return np.array(segments)


def download_and_load(max_segments_per_file: int = 60) -> pd.DataFrame:
    """
    Download CWRU files and return a DataFrame where each row is one segment.

    Columns: signal (numpy array), label, fault_size, load
    """
    rows = []
    print("Downloading CWRU dataset …")
    for filename, label, fault_size, load in tqdm(CWRU_FILES):
        try:
            path   = _download_file(filename)
            signal = _extract_signal(path)
            segs   = _segment(signal)
            if max_segments_per_file:
                segs = segs[:max_segments_per_file]
            for seg in segs:
                rows.append({
                    "signal":     seg,
                    "label":      label,
                    "fault_size": fault_size,
                    "load":       load,
                })
        except Exception as e:
            print(f"  Warning: could not load {filename} — {e}")

    df = pd.DataFrame(rows)
    print(f"Loaded {len(df)} segments across {df['label'].nunique()} classes.")
    return df
