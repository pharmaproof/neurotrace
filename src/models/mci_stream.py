"""
MCI Stream — Mild Cognitive Impairment Classifier
===================================================
Trains and loads the XGBoost binary classifier on cognitive / linguistic features.
"""

import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, classification_report
from xgboost import XGBClassifier

from imblearn.over_sampling import SMOTE


MODEL_PATH  = Path(__file__).parent.parent.parent / 'models' / 'mci_model.pkl'
SCALER_PATH = Path(__file__).parent.parent.parent / 'models' / 'mci_scaler.pkl'


def train_mci_model(X: pd.DataFrame, y: pd.Series,
                    smote: bool = True,
                    random_state: int = 42):
    """
    Train an XGBoost MCI classifier on linguistic features.

    Parameters
    ----------
    X            : feature DataFrame (all numeric, no label column)
    y            : binary labels (0=Healthy, 1=MCI/AD)
    smote        : whether to apply SMOTE oversampling
    random_state : reproducibility seed

    Returns
    -------
    model, scaler, auc  (float AUC on held-out test set)
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state, stratify=y)

    scaler    = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    if smote and y_train.nunique() > 1:
        sm = SMOTE(random_state=random_state)
        X_train_s, y_train = sm.fit_resample(X_train_s, y_train)

    model = XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        use_label_encoder=False,
        eval_metric='logloss',
        random_state=random_state,
        verbosity=0,
    )
    model.fit(X_train_s, y_train,
              eval_set=[(X_test_s, y_test)],
              verbose=False)

    proba = model.predict_proba(X_test_s)[:, 1]
    auc   = roc_auc_score(y_test, proba)
    print(f'[MCI Stream] Test AUC: {auc:.4f}')
    print(classification_report(y_test, (proba >= 0.5).astype(int),
                                 target_names=['Healthy', 'MCI']))

    return model, scaler, auc


def save_mci_model(model, scaler):
    """Persist model and scaler to disk."""
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model,  MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    print(f'[MCI Stream] Saved model → {MODEL_PATH}')


def load_mci_model():
    """Load model and scaler from disk."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f'MCI model not found at {MODEL_PATH}. Run train_all.py first.')
    model  = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    return model, scaler


def predict_mci_proba(X: np.ndarray, model=None, scaler=None) -> np.ndarray:
    """Return P(MCI) for each row of X (raw, unscaled features)."""
    if model is None or scaler is None:
        model, scaler = load_mci_model()
    X_s = scaler.transform(X)
    return model.predict_proba(X_s)[:, 1]


# ── Standalone training entrypoint ────────────────────────────────────────────

if __name__ == '__main__':
    data_path = (Path(__file__).parent.parent.parent
                 / 'data' / 'processed' / 'cognitive_features.csv')
    if not data_path.exists():
        raise FileNotFoundError(
            f'Cognitive features not found at {data_path}. '
            'Run python -m src.data.generate_synthetic first.')

    df = pd.read_csv(data_path)
    X  = df.drop('label', axis=1)
    y  = df['label']

    model, scaler, auc = train_mci_model(X, y)
    save_mci_model(model, scaler)
