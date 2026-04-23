"""
Fusion Classifier — 3-Class NeuroTrace Output
===============================================
Combines PD and MCI stream probabilities into a 3-class risk profile:
  0 = Healthy Baseline
  1 = PD Motor Profile
  2 = MCI/Alzheimer's Profile
"""

import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

FUSION_MODEL_PATH = Path(__file__).parent.parent.parent / 'models' / 'fusion_model.pkl'
DEMO_SUBJECTS_PATH = Path(__file__).parent.parent.parent / 'app' / 'demo_subjects.json'

CLASS_NAMES = ['Healthy', 'PD Profile', 'MCI Profile']
CLASS_COLORS = ['#2ECC71', '#E67E22', '#E74C3C']


# ── Feature Engineering ───────────────────────────────────────────────────────

def build_fusion_features(pd_proba: np.ndarray,
                           mci_proba: np.ndarray) -> np.ndarray:
    """
    Stack both stream probabilities into a 4-dim fusion feature vector.

    Features: [P(PD), P(MCI), P(PD)/P(MCI), P(PD)-P(MCI)]
    """
    pd_proba  = np.atleast_1d(pd_proba).astype(float)
    mci_proba = np.atleast_1d(mci_proba).astype(float)
    ratio = pd_proba / (mci_proba + 1e-6)
    diff  = pd_proba - mci_proba
    return np.column_stack([pd_proba, mci_proba, ratio, diff])


# ── Training ──────────────────────────────────────────────────────────────────

def train_fusion_model(pd_proba_all: np.ndarray,
                        mci_proba_all: np.ndarray,
                        y_3class: np.ndarray,
                        random_state: int = 42):
    """
    Train a multinomial logistic regression fusion layer.

    Parameters
    ----------
    pd_proba_all  : P(PD) from PD stream for every subject
    mci_proba_all : P(MCI) from MCI stream for every subject
    y_3class      : labels (0=Healthy, 1=PD, 2=MCI)
    """
    X_f = build_fusion_features(pd_proba_all, mci_proba_all)

    X_train, X_test, y_train, y_test = train_test_split(
        X_f, y_3class, test_size=0.2, stratify=y_3class,
        random_state=random_state)

    fusion = LogisticRegression(max_iter=1000,
                                random_state=random_state)
    fusion.fit(X_train, y_train)

    y_pred = fusion.predict(X_test)
    print('[Fusion] Classification Report:')
    active_classes = np.unique(y_test)
    active_names   = [CLASS_NAMES[i] for i in active_classes]
    print(classification_report(y_test, y_pred, target_names=active_names))

    return fusion


def save_fusion_model(model):
    FUSION_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, FUSION_MODEL_PATH)
    print(f'[Fusion] Saved model → {FUSION_MODEL_PATH}')


def load_fusion_model():
    if not FUSION_MODEL_PATH.exists():
        raise FileNotFoundError(
            f'Fusion model not found at {FUSION_MODEL_PATH}. Run train_all.py first.')
    return joblib.load(FUSION_MODEL_PATH)


# ── Inference ─────────────────────────────────────────────────────────────────

def predict_neurotrace_raw(pd_proba: float, mci_proba: float,
                            fusion_model=None) -> dict:
    """
    Run the fusion classifier for a single subject.

    Returns a dict with pd_risk (%), mci_risk (%), classification, etc.
    """
    if fusion_model is None:
        fusion_model = load_fusion_model()

    X = build_fusion_features(np.array([pd_proba]), np.array([mci_proba]))
    proba_3class = fusion_model.predict_proba(X)[0]
    class_id     = int(np.argmax(proba_3class))

    return {
        'pd_risk':        round(pd_proba * 100, 1),
        'mci_risk':       round(mci_proba * 100, 1),
        'healthy_prob':   round(float(proba_3class[0]) * 100, 1),
        'pd_prob':        round(float(proba_3class[1]) * 100, 1),
        'mci_prob':       round(float(proba_3class[2]) * 100, 1),
        'classification': CLASS_NAMES[class_id],
        'class_id':       class_id,
        'color':          CLASS_COLORS[class_id],
    }


def predict_neurotrace(subject_key: str) -> dict:
    """
    Load a pre-computed demo subject profile from demo_subjects.json.

    Used by the Streamlit app for reliable hackathon demos.
    """
    if not DEMO_SUBJECTS_PATH.exists():
        raise FileNotFoundError(
            f'demo_subjects.json not found at {DEMO_SUBJECTS_PATH}. '
            'Run train_all.py first.')

    with open(DEMO_SUBJECTS_PATH, 'r') as f:
        subjects = json.load(f)

    if subject_key not in subjects:
        raise KeyError(f'Unknown demo subject: {subject_key}. '
                       f'Available: {list(subjects.keys())}')
    return subjects[subject_key]
