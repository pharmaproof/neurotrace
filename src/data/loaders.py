"""
Dataset Loaders
================
Utilities for loading real PhysioNet / DementiaBank datasets.
Falls back to synthetic data if real data is not present.
"""

import os
from pathlib import Path
import pandas as pd

# Canonical data paths
DATA_ROOT = Path(__file__).parent.parent.parent / 'data'

NQ_RAW_DIR   = DATA_ROOT / 'raw' / 'nqmitcsxpd'
TAPPY_RAW_DIR = DATA_ROOT / 'raw' / 'tappy'
DB_RAW_DIR   = DATA_ROOT / 'raw' / 'dementiabank'

MOTOR_PROCESSED   = DATA_ROOT / 'processed' / 'motor_features.csv'
COGNITIVE_PROCESSED = DATA_ROOT / 'processed' / 'cognitive_features.csv'


def load_motor_features() -> pd.DataFrame:
    """
    Load motor feature CSV. Falls back to generating synthetic data if not found.
    """
    if MOTOR_PROCESSED.exists():
        df = pd.read_csv(MOTOR_PROCESSED)
        print(f'[Loader] Loaded motor features from {MOTOR_PROCESSED} — {df.shape}')
        return df

    print('[Loader] Processed motor features not found. Generating synthetic data...')
    from src.data.generate_synthetic import save_datasets
    motor_df, _ = save_datasets()
    return motor_df


def load_cognitive_features() -> pd.DataFrame:
    """
    Load cognitive feature CSV. Falls back to generating synthetic data if not found.
    """
    if COGNITIVE_PROCESSED.exists():
        df = pd.read_csv(COGNITIVE_PROCESSED)
        print(f'[Loader] Loaded cognitive features from {COGNITIVE_PROCESSED} — {df.shape}')
        return df

    print('[Loader] Processed cognitive features not found. Generating synthetic data...')
    from src.data.generate_synthetic import save_datasets
    _, cog_df = save_datasets()
    return cog_df


def has_real_nq_data() -> bool:
    """Check if neuroQWERTY raw data has been downloaded."""
    return NQ_RAW_DIR.exists() and any(NQ_RAW_DIR.glob('*.txt'))


def has_real_tappy_data() -> bool:
    """Check if Tappy raw data has been downloaded."""
    return TAPPY_RAW_DIR.exists() and any(TAPPY_RAW_DIR.glob('*.csv'))


def has_real_dementiabank_data() -> bool:
    """Check if DementiaBank raw data has been downloaded."""
    return DB_RAW_DIR.exists() and any(DB_RAW_DIR.glob('**/*.cha'))


def data_status() -> dict:
    """Return a dict showing which real datasets are available."""
    return {
        'neuroQWERTY':    has_real_nq_data(),
        'tappy':          has_real_tappy_data(),
        'dementiabank':   has_real_dementiabank_data(),
        'motor_processed':    MOTOR_PROCESSED.exists(),
        'cognitive_processed': COGNITIVE_PROCESSED.exists(),
    }
