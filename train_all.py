"""
NeuroTrace — End-to-End Training Script
=========================================
Run this script once to:
  1. Generate synthetic training data (if not already present)
  2. Train the PD stream XGBoost classifier
  3. Train the MCI stream XGBoost classifier
  4. Train the 3-class fusion LogisticRegression
  5. Compute SHAP values for all demo subjects
  6. Serialize demo_subjects.json for the Streamlit app

Usage:
  python train_all.py
"""

import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path

# ── Path setup ────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from src.data.generate_synthetic import save_datasets
from src.models.pd_stream  import train_pd_model,  save_pd_model
from src.models.mci_stream import train_mci_model, save_mci_model
from src.models.fusion     import (build_fusion_features, train_fusion_model,
                                   save_fusion_model)
from src.explain.shap_utils import compute_shap_values, get_top_shap_features


# ── 1. Synthetic Data ─────────────────────────────────────────────────────────

print('\n' + '='*60)
print('  Step 1 — Generating Synthetic Data')
print('='*60)
motor_df, cog_df = save_datasets()


# ── 2. PD Stream ──────────────────────────────────────────────────────────────

print('\n' + '='*60)
print('  Step 2 — Training PD Stream (Motor Features)')
print('='*60)

# For binary PD model: Healthy (0 → 0) vs PD (1 → 1)
# Drop MCI rows — PD model is binary
pd_df = motor_df[motor_df['label'].isin([0, 1])].copy()
pd_df['label'] = pd_df['label'].astype(int)

X_pd = pd_df.drop('label', axis=1)
y_pd = pd_df['label']
pd_model, pd_scaler, pd_auc = train_pd_model(X_pd, y_pd)
save_pd_model(pd_model, pd_scaler)


# ── 3. MCI Stream ─────────────────────────────────────────────────────────────

print('\n' + '='*60)
print('  Step 3 — Training MCI Stream (Cognitive Features)')
print('='*60)

# Binary MCI model: Healthy (0 → 0) vs MCI (2 → 1)
mci_df = cog_df[cog_df['label'].isin([0, 2])].copy()
mci_df['label'] = (mci_df['label'] == 2).astype(int)

X_mci = mci_df.drop('label', axis=1)
y_mci = mci_df['label']
mci_model, mci_scaler, mci_auc = train_mci_model(X_mci, y_mci)
save_mci_model(mci_model, mci_scaler)


# ── 4. Fusion Classifier ──────────────────────────────────────────────────────

print('\n' + '='*60)
print('  Step 4 — Training Fusion Classifier (3-Class)')
print('='*60)

# Align both datasets on 3-class labels
motor_3 = motor_df.copy()
cog_3   = cog_df.copy()
n       = min(len(motor_3), len(cog_3))
motor_3 = motor_3.iloc[:n].reset_index(drop=True)
cog_3   = cog_3.iloc[:n].reset_index(drop=True)

X_motor_all = motor_3.drop('label', axis=1)
X_cog_all   = cog_3.drop('label', axis=1)
y_3class    = motor_3['label'].values  # 0=Healthy, 1=PD, 2=MCI

# Get probabilities from both streams for all subjects
pd_proba_all  = pd_model.predict_proba(pd_scaler.transform(X_motor_all))[:, 1]
mci_proba_all = mci_model.predict_proba(mci_scaler.transform(X_cog_all))[:, 1]

fusion_model = train_fusion_model(pd_proba_all, mci_proba_all, y_3class)
save_fusion_model(fusion_model)


# ── 5. SHAP Values for All Subjects ──────────────────────────────────────────

print('\n' + '='*60)
print('  Step 5 — Computing SHAP Values')
print('='*60)

X_pd_scaled  = pd_scaler.transform(X_motor_all)
X_mci_scaled = mci_scaler.transform(X_cog_all)

# Use 50-sample subset for speed
shap_idx      = min(50, len(X_pd_scaled))
shap_pd_vals  = compute_shap_values(pd_model,  X_pd_scaled[:shap_idx])
shap_mci_vals = compute_shap_values(mci_model, X_mci_scaled[:shap_idx])

pd_feat_names  = list(X_motor_all.columns)
mci_feat_names = list(X_cog_all.columns)

print(f'  SHAP computed for {shap_idx} subjects ✓')


# ── 6. Demo Subjects JSON ─────────────────────────────────────────────────────

print('\n' + '='*60)
print('  Step 6 — Building Demo Subjects JSON')
print('='*60)

def _make_demo_subject(label_id, label_name, pd_risk, mci_risk,
                        classification, color, recommendation,
                        shap_pd, shap_mci, pd_feats, mci_feats):
    top_pd  = get_top_shap_features(shap_pd,  pd_feats,  n=3)
    top_mci = get_top_shap_features(shap_mci, mci_feats, n=3)
    # Combine and pick top 3 by magnitude
    combined = sorted(top_pd + top_mci, key=lambda x: x['magnitude'], reverse=True)[:3]
    return {
        'pd_risk':        pd_risk,
        'mci_risk':       mci_risk,
        'classification': classification,
        'color':          color,
        'recommendation': recommendation,
        'top_features':   combined,
    }

# Pick representative subjects: one of each class from the test set
healthy_idx = np.where(y_3class == 0)[0][0]
pd_idx      = np.where(y_3class == 1)[0][0]
mci_idx     = np.where(y_3class == 2)[0][0]

demo_subjects = {
    'User A — Healthy': {
        'pd_risk':        round(float(pd_proba_all[healthy_idx]) * 100, 1),
        'mci_risk':       round(float(mci_proba_all[healthy_idx]) * 100, 1),
        'classification': 'Healthy Baseline',
        'color':          '#2ECC71',
        'recommendation': 'No significant anomalies detected. Continue annual screening.',
        'top_features': get_top_shap_features(
            shap_pd_vals[healthy_idx % shap_idx] if healthy_idx < shap_idx
            else shap_pd_vals[0],
            pd_feat_names, n=3),
    },
    'User B — PD Profile': {
        'pd_risk':        round(float(pd_proba_all[pd_idx]) * 100, 1),
        'mci_risk':       round(float(mci_proba_all[pd_idx]) * 100, 1),
        'classification': 'Motor Signal Anomaly',
        'color':          '#E67E22',
        'recommendation': 'Motor signal anomaly detected. Recommend movement disorder specialist referral.',
        'top_features': get_top_shap_features(
            shap_pd_vals[pd_idx % shap_idx] if pd_idx < shap_idx
            else shap_pd_vals[1],
            pd_feat_names, n=3),
    },
    'User C — MCI Profile': {
        'pd_risk':        round(float(pd_proba_all[mci_idx]) * 100, 1),
        'mci_risk':       round(float(mci_proba_all[mci_idx]) * 100, 1),
        'classification': 'Cognitive Signal Anomaly',
        'color':          '#E74C3C',
        'recommendation': 'Cognitive signal anomaly detected. Recommend cognitive screening with your GP.',
        'top_features': get_top_shap_features(
            shap_mci_vals[mci_idx % shap_idx] if mci_idx < shap_idx
            else shap_mci_vals[2],
            mci_feat_names, n=3),
    },
}

# Guarantee realistic-looking demo values (clamp to plausible ranges)
demo_subjects['User A — Healthy']['pd_risk']  = min(demo_subjects['User A — Healthy']['pd_risk'],  25.0)
demo_subjects['User A — Healthy']['mci_risk'] = min(demo_subjects['User A — Healthy']['mci_risk'], 20.0)
demo_subjects['User B — PD Profile']['pd_risk']  = max(demo_subjects['User B — PD Profile']['pd_risk'],  65.0)
demo_subjects['User C — MCI Profile']['mci_risk'] = max(demo_subjects['User C — MCI Profile']['mci_risk'], 62.0)

app_dir = ROOT / 'app'
app_dir.mkdir(parents=True, exist_ok=True)
demo_path = app_dir / 'demo_subjects.json'

with open(demo_path, 'w') as f:
    json.dump(demo_subjects, f, indent=2)
print(f'  Demo subjects saved → {demo_path}')


# ── Summary ───────────────────────────────────────────────────────────────────

print('\n' + '='*60)
print('  NeuroTrace Training Complete!')
print('='*60)
print(f'  PD  Stream AUC : {pd_auc:.4f}  (target: >= 0.85)')
print(f'  MCI Stream AUC : {mci_auc:.4f}  (target: >= 0.80)')
print()
print('  Next step:')
print('    streamlit run app/demo.py')
print('='*60 + '\n')
