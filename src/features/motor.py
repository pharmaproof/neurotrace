"""
Motor Feature Pipeline — PD Stream
===================================
Extracts keystroke timing features from press/release logs.
Supports neuroQWERTY and Tappy dataset formats.
"""

import numpy as np
import pandas as pd
from scipy import stats
import pywt  # PyWavelets


# ── Parsing ──────────────────────────────────────────────────────────────────

def parse_keystroke_log(filepath: str) -> pd.DataFrame:
    """
    Load a neuroQWERTY-format keystroke log (tab-separated).

    Columns:  action (P/R),  timestamp (ms),  key (id)
    """
    df = pd.read_csv(
        filepath,
        sep='\t',
        header=None,
        names=['action', 'timestamp', 'key'],
    )
    df['timestamp'] = pd.to_numeric(df['timestamp'], errors='coerce')
    df = df.dropna(subset=['timestamp'])
    return df


def parse_tappy_log(filepath: str) -> pd.DataFrame:
    """
    Load a Tappy-format keystroke log (comma-separated).

    Expected columns: UserID, Date, Timestamp, Hand,
                      HoldTime, Direction, LatencyTime, FlightTime
    """
    df = pd.read_csv(filepath, header=0)
    df.columns = df.columns.str.strip()
    for col in ['HoldTime', 'FlightTime']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    return df


# ── Feature Extraction ────────────────────────────────────────────────────────

def compute_hold_flight_times(df: pd.DataFrame):
    """
    Compute hold times and flight times from press/release events.

    Returns
    -------
    hold_times  : np.ndarray  — duration each key is held (ms)
    flight_times: np.ndarray  — gap between consecutive key presses (ms)
    """
    presses  = df[df['action'] == 'P'].reset_index(drop=True)
    releases = df[df['action'] == 'R'].reset_index(drop=True)

    hold_times = []
    for _, press in presses.iterrows():
        rel = releases[
            (releases['key'] == press['key']) &
            (releases['timestamp'] > press['timestamp'])
        ]
        if not rel.empty:
            hold_times.append(rel.iloc[0]['timestamp'] - press['timestamp'])

    press_ts    = presses['timestamp'].values
    flight_times = np.diff(press_ts)

    return np.array(hold_times, dtype=float), flight_times.astype(float)


def _clean(arr: np.ndarray, lo: float = 0.0, hi: float = 2000.0) -> np.ndarray:
    """Remove invalid / outlier values from a timing array."""
    return arr[(arr > lo) & (arr < hi)]


def _stat_features(arr: np.ndarray, prefix: str) -> dict:
    """Seven statistical moments for a timing array."""
    arr = _clean(arr)
    if len(arr) < 2:
        return {f'{prefix}_{k}': 0.0 for k in
                ['mean', 'std', 'median', 'skew', 'kurt', 'iqr', 'cv']}
    return {
        f'{prefix}_mean':   float(np.mean(arr)),
        f'{prefix}_std':    float(np.std(arr)),
        f'{prefix}_median': float(np.median(arr)),
        f'{prefix}_skew':   float(stats.skew(arr)),
        f'{prefix}_kurt':   float(stats.kurtosis(arr)),
        f'{prefix}_iqr':    float(stats.iqr(arr)),
        f'{prefix}_cv':     float(np.std(arr) / (np.mean(arr) + 1e-9)),
    }


def _dwt_features(arr: np.ndarray, prefix: str,
                  wavelet: str = 'db4', level: int = 3) -> dict:
    """
    Discrete Wavelet Transform features — frequency-domain arrhythmia
    signal (SSRN 2025 approach).
    """
    arr = _clean(arr)
    feats: dict = {}
    if len(arr) < 2 ** level:
        for lvl in range(level + 1):
            feats[f'{prefix}_dwt_L{lvl}_energy'] = 0.0
            feats[f'{prefix}_dwt_L{lvl}_std']    = 0.0
        return feats

    coeffs = pywt.wavedec(arr, wavelet, level=level)
    for i, c in enumerate(coeffs):
        feats[f'{prefix}_dwt_L{i}_energy'] = float(np.sum(c ** 2))
        feats[f'{prefix}_dwt_L{i}_std']    = float(np.std(c))
    return feats


def extract_motor_features(hold_times: np.ndarray,
                            flight_times: np.ndarray) -> dict:
    """
    Extract statistical + DWT features from timing arrays.

    Returns a flat dict of ~28 numeric features per subject.
    """
    features: dict = {}
    features.update(_stat_features(hold_times,   'ht'))
    features.update(_stat_features(flight_times, 'ft'))
    features.update(_dwt_features(hold_times,    'ht'))
    features.update(_dwt_features(flight_times,  'ft'))
    return features


# ── Dataset Builder ───────────────────────────────────────────────────────────

def build_motor_dataset(subject_files: list, labels: list) -> pd.DataFrame:
    """
    Batch-process all subjects and return a feature DataFrame.

    Parameters
    ----------
    subject_files : list of filepaths to keystroke logs
    labels        : corresponding list of int labels (0=Healthy, 1=PD)
    """
    rows = []
    for filepath, label in zip(subject_files, labels):
        try:
            df  = parse_keystroke_log(filepath)
            ht, ft = compute_hold_flight_times(df)
            if len(ht) < 50:
                continue  # Too few keystrokes — skip this session
            feats = extract_motor_features(ht, ft)
            feats['label'] = int(label)
            rows.append(feats)
        except Exception as exc:
            print(f'[WARN] Skipping {filepath}: {exc}')
    return pd.DataFrame(rows)
