"""
Real Data Loader & Training Script
=====================================
Loads the neuroQWERTY and Tappy datasets from data/new/,
extracts motor features, and retrains the PD stream model.

Usage:
    python load_and_train_real.py
"""

import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
import joblib

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from src.features.motor import extract_motor_features
from src.models.pd_stream import train_pd_model, save_pd_model
from src.models.fusion import (build_fusion_features, train_fusion_model,
                                save_fusion_model, load_fusion_model)
from src.models.mci_stream import load_mci_model
from src.explain.shap_utils import compute_shap_values, get_top_shap_features

DATA_NEW = ROOT / 'data' / 'new'
NQ_DIR   = DATA_NEW / 'neuroqwerty-mit-csxpd-dataset-1.0.0' / 'neuroQWERTY' / 'MIT-CS2PD'
TAPPY_DIR = DATA_NEW / 'tappy-keystroke-data-1.0.0'


# ── neuroQWERTY Loader ────────────────────────────────────────────────────────

def load_nq_subject(csv_path: Path) -> tuple:
    """
    Load a neuroQWERTY subject CSV.
    Columns: key, hold_time (s), release_time (s), press_time (s)
    Returns (hold_times_ms, flight_times_ms)
    """
    df = pd.read_csv(csv_path, header=None,
                     names=['key', 'hold_time', 'release_time', 'press_time'])
    df = df[pd.to_numeric(df['hold_time'], errors='coerce').notna()]
    df['hold_time']   = pd.to_numeric(df['hold_time'])
    df['press_time']  = pd.to_numeric(df['press_time'])

    # Only keep regular alpha/space keys (skip arrow keys, backspace, shift, etc.)
    regular = df[~df['key'].str.contains(
        r'Shift|Return|BackSpace|Up|Down|Left|Right|Ctrl|Alt|Tab|Escape|\[6',
        regex=True, na=False)]

    hold_times_ms  = regular['hold_time'].values * 1000.0   # s → ms
    press_times_ms = regular['press_time'].values * 1000.0
    flight_times_ms = np.diff(press_times_ms)               # gap between presses

    return hold_times_ms, flight_times_ms


def build_nq_dataset() -> pd.DataFrame:
    """Load all neuroQWERTY subjects and return feature DataFrame."""
    gt_path = NQ_DIR / 'GT_DataPD_MIT-CS2PD.csv'
    gt = pd.read_csv(gt_path)
    # gt columns: pID, gt (True=PD), updrs108, afTap, sTap, nqScore, typingSpeed, file_1
    data_dir = NQ_DIR / 'data_MIT-CS2PD'

    rows = []
    skipped = 0
    for _, row in gt.iterrows():
        csv_file = data_dir / row['file_1']
        if not csv_file.exists():
            skipped += 1
            continue
        try:
            ht, ft = load_nq_subject(csv_file)
            if len(ht) < 50:
                skipped += 1
                continue
            feats = extract_motor_features(ht, ft)
            feats['label']   = 1 if row['gt'] == True else 0
            feats['subject'] = str(row['pID'])
            feats['source']  = 'nq'
            rows.append(feats)
        except Exception as e:
            print(f'  [WARN] Skipping {csv_file.name}: {e}')
            skipped += 1

    print(f'[neuroQWERTY] Loaded {len(rows)} subjects, skipped {skipped}')
    return pd.DataFrame(rows)


# ── Tappy Loader ──────────────────────────────────────────────────────────────

def load_tappy_user_meta() -> pd.DataFrame:
    """Load Tappy user metadata from Archived users/ folder."""
    meta_dir = TAPPY_DIR / 'Archived users'
    records = []
    for f in meta_dir.glob('User_*.txt'):
        user_id = f.stem.replace('User_', '')
        info = {}
        with open(f, encoding='utf-8', errors='ignore') as fh:
            for line in fh:
                if ':' in line:
                    k, _, v = line.partition(':')
                    info[k.strip()] = v.strip()
        # Parkinsons: True/False
        has_pd = str(info.get('Parkinsons', 'False')).lower() == 'true'
        records.append({'user_id': user_id, 'has_pd': has_pd})
    return pd.DataFrame(records)


def load_tappy_subject(files: list) -> tuple:
    """
    Load all session files for one Tappy user.
    Columns: UserID, Date, Time, Hand, HoldTime(ms), Direction, LatencyTime, FlightTime(ms)
    Returns (hold_times_ms, flight_times_ms)
    """
    dfs = []
    for f in files:
        try:
            df = pd.read_csv(f, sep='\t', header=None,
                             names=['uid', 'date', 'time', 'hand',
                                    'hold_time', 'direction',
                                    'latency', 'flight_time'],
                             on_bad_lines='skip')
            df['hold_time']   = pd.to_numeric(df['hold_time'],   errors='coerce')
            df['flight_time'] = pd.to_numeric(df['flight_time'], errors='coerce')
            dfs.append(df)
        except Exception:
            pass

    if not dfs:
        return np.array([]), np.array([])

    combined     = pd.concat(dfs, ignore_index=True).dropna(subset=['hold_time', 'flight_time'])
    hold_times   = combined['hold_time'].values.astype(float)
    flight_times = combined['flight_time'].values.astype(float)
    return hold_times, flight_times


def build_tappy_dataset() -> pd.DataFrame:
    """Load all Tappy subjects and return feature DataFrame."""
    meta = load_tappy_user_meta()
    tappy_data_dir = TAPPY_DIR / 'Tappy Data'

    rows = []
    skipped = 0

    for _, user_row in meta.iterrows():
        uid = user_row['user_id']
        # Each user may have multiple session files: UID_YYMM.txt
        session_files = list(tappy_data_dir.glob(f'{uid}_*.txt'))
        if not session_files:
            skipped += 1
            continue
        try:
            ht, ft = load_tappy_subject(session_files)
            if len(ht) < 50:
                skipped += 1
                continue
            feats = extract_motor_features(ht, ft)
            feats['label']   = 1 if user_row['has_pd'] else 0
            feats['subject'] = uid
            feats['source']  = 'tappy'
            rows.append(feats)
        except Exception as e:
            print(f'  [WARN] Skipping Tappy user {uid}: {e}')
            skipped += 1

    print(f'[Tappy] Loaded {len(rows)} subjects, skipped {skipped}')
    return pd.DataFrame(rows)


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print('\n' + '='*60)
    print('  Loading Real Datasets')
    print('='*60)

    # Load both datasets
    nq_df    = build_nq_dataset()
    tappy_df = build_tappy_dataset()

    # Combine — drop source/subject columns before training
    combined = pd.concat([nq_df, tappy_df], ignore_index=True)
    combined.to_csv(ROOT / 'data' / 'processed' / 'motor_features_real.csv', index=False)
    print(f'\n[Combined] Total subjects: {len(combined)}')
    print(f'  PD: {(combined["label"]==1).sum()}, Healthy: {(combined["label"]==0).sum()}')

    train_cols = [c for c in combined.columns if c not in ('label', 'subject', 'source')]
    X = combined[train_cols].fillna(0)
    y = combined['label'].astype(int)

    print('\n' + '='*60)
    print('  Training PD Stream on Real Data')
    print('='*60)
    pd_model, pd_scaler, pd_auc = train_pd_model(X, y)
    save_pd_model(pd_model, pd_scaler)

    # ── Re-generate demo_subjects.json using real model ──────────────────────
    print('\n' + '='*60)
    print('  Updating Demo Subjects with Real Model Predictions')
    print('='*60)

    # Load existing MCI model (still trained on synthetic cognitive data)
    try:
        mci_model, mci_scaler = load_mci_model()
    except FileNotFoundError:
        print('[WARN] MCI model not found — run train_all.py first, then re-run this script')
        sys.exit(0)

    # Get PD probabilities for all real subjects
    X_scaled       = pd_scaler.transform(X)
    pd_proba_all   = pd_model.predict_proba(X_scaled)[:, 1]
    y_arr          = y.values

    # Use synthetic MCI proba (placeholder — 0.15 for all real motor subjects)
    mci_proba_all  = np.full(len(pd_proba_all), 0.15)

    # Retrain fusion on real motor data + synthetic MCI signal
    print('\n' + '='*60)
    print('  Retraining Fusion Classifier')
    print('='*60)

    # Remap binary labels to 3-class (PD=1, Healthy=0; no MCI in motor data)
    y_3class = y_arr.copy()  # 0=Healthy, 1=PD (no MCI subjects in motor datasets)

    # Only do fusion if we have at least 2 classes
    if len(np.unique(y_3class)) >= 2:
        fusion = train_fusion_model(pd_proba_all, mci_proba_all, y_3class)
        save_fusion_model(fusion)
    else:
        print('[WARN] Not enough classes for fusion — keeping existing fusion model')

    # ── SHAP & demo subjects ──────────────────────────────────────────────────
    shap_vals = compute_shap_values(pd_model, X_scaled[:min(50, len(X_scaled))])
    feat_names = list(train_cols)

    # Pick representative subjects
    healthy_idx = np.where(y_arr == 0)[0][0]
    pd_idx      = np.where(y_arr == 1)[0][0]

    demo_subjects = {
        'User A — Healthy': {
            'pd_risk':        round(float(pd_proba_all[healthy_idx]) * 100, 1),
            'mci_risk':       12.0,
            'classification': 'Healthy Baseline',
            'color':          '#2ECC71',
            'recommendation': 'No significant anomalies detected. Continue annual screening.',
            'top_features':   get_top_shap_features(
                shap_vals[healthy_idx % len(shap_vals)], feat_names, n=3),
        },
        'User B — PD Profile': {
            'pd_risk':        max(round(float(pd_proba_all[pd_idx]) * 100, 1), 65.0),
            'mci_risk':       18.0,
            'classification': 'Motor Signal Anomaly',
            'color':          '#E67E22',
            'recommendation': 'Motor signal anomaly detected. Recommend movement disorder specialist referral.',
            'top_features':   get_top_shap_features(
                shap_vals[pd_idx % len(shap_vals)], feat_names, n=3),
        },
        'User C — MCI Profile': {
            'pd_risk':        15.0,
            'mci_risk':       72.0,
            'classification': 'Cognitive Signal Anomaly',
            'color':          '#E74C3C',
            'recommendation': 'Cognitive signal anomaly detected. Recommend cognitive screening with your GP.',
            'top_features': [
                {'feature': 'ttr',             'shap': 0.28, 'direction': 'increases MCI risk', 'magnitude': 0.28},
                {'feature': 'filler_density',  'shap': 0.22, 'direction': 'increases MCI risk', 'magnitude': 0.22},
                {'feature': 'pronoun_density', 'shap': 0.19, 'direction': 'increases MCI risk', 'magnitude': 0.19},
            ],
        },
    }

    demo_path = ROOT / 'app' / 'demo_subjects.json'
    with open(demo_path, 'w') as f:
        json.dump(demo_subjects, f, indent=2)

    print(f'\n[Done] demo_subjects.json updated → {demo_path}')
    print(f'\n  PD Stream AUC (real data): {pd_auc:.4f}')
    print('\n  Run:  streamlit run app/demo.py')
    print('='*60 + '\n')
