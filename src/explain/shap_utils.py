"""
SHAP Explainability Utilities
==============================
Wraps TreeExplainer for XGBoost models and returns
human-readable top-feature explanations.
"""

import numpy as np
from typing import List


def compute_shap_values(model, X: np.ndarray) -> np.ndarray:
    """
    Compute SHAP values for all rows of X using TreeExplainer.

    Parameters
    ----------
    model : a fitted XGBoost or LightGBM model
    X     : 2-D numpy array (n_samples, n_features) — already scaled

    Returns
    -------
    shap_values : np.ndarray of shape (n_samples, n_features)
    """
    try:
        import shap
        explainer   = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)
        # For binary classifiers, shap_values may be a list [class0, class1]
        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # Take class-1 (positive) values
        return shap_values
    except ImportError:
        print('[WARN] shap not installed — returning zeros')
        return np.zeros_like(X)


def get_top_shap_features(shap_vals: np.ndarray,
                           feature_names: List[str],
                           n: int = 3) -> List[dict]:
    """
    Return the top-N features by absolute SHAP value for one prediction.

    Parameters
    ----------
    shap_vals     : 1-D array of SHAP values for a single sample
    feature_names : list of feature name strings
    n             : number of top features to return

    Returns
    -------
    List of dicts with keys: feature, shap, direction, magnitude
    """
    shap_vals    = np.array(shap_vals)
    abs_vals     = np.abs(shap_vals)
    top_idx      = np.argsort(abs_vals)[::-1][:n]

    results = []
    for i in top_idx:
        results.append({
            'feature':   feature_names[i],
            'shap':      float(shap_vals[i]),
            'magnitude': float(abs_vals[i]),
            'direction': 'increases' if shap_vals[i] > 0 else 'decreases',
        })
    return results


def explain_prediction(model, X_sample: np.ndarray,
                        feature_names: List[str],
                        n: int = 3) -> List[dict]:
    """
    Compute SHAP values for a single sample and return top-N features.

    Parameters
    ----------
    model         : fitted XGBoost model
    X_sample      : 2-D array with shape (1, n_features) — already scaled
    feature_names : list of feature name strings
    n             : number of top features to return
    """
    shap_vals = compute_shap_values(model, X_sample)
    if shap_vals.ndim == 2:
        shap_vals = shap_vals[0]
    return get_top_shap_features(shap_vals, feature_names, n)


def format_shap_explanation(top_features: List[dict], stream: str = 'risk') -> str:
    """Format top SHAP features as a human-readable bullet list."""
    lines = []
    for f in top_features:
        lines.append(
            f"• **{f['feature']}** ({f['shap']:+.3f}) — {f['direction']} {stream}"
        )
    return '\n'.join(lines)
