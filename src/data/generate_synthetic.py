"""
Synthetic Data Generator
=========================
Generates realistic synthetic keystroke + linguistic feature datasets
for 150 subjects (50 each: Healthy, PD, MCI) so the full pipeline
can be trained and demoed without downloading any real datasets.

Statistical parameters derived from the literature:
  - neuroQWERTY / Tappy datasets (PD keystroke features)
  - DementiaBank / Pitt Corpus (linguistic markers)
"""

import numpy as np
import pandas as pd
from pathlib import Path

# Feature means and stds calibrated per clinical literature
# ── Motor features (hold time / flight time statistics) ──────────────────────
#    Columns match output of extract_motor_features() in motor.py
MOTOR_PROFILES = {
    'healthy': {
        'ht_mean': (105, 12),   'ht_std': (18, 4),  'ht_median': (100, 10),
        'ht_skew': (0.4, 0.2),  'ht_kurt': (0.5, 0.3), 'ht_iqr': (22, 5),
        'ht_cv':   (0.17, 0.03),
        'ft_mean': (155, 18),   'ft_std': (28, 6),  'ft_median': (148, 15),
        'ft_skew': (0.5, 0.2),  'ft_kurt': (0.6, 0.3), 'ft_iqr': (35, 7),
        'ft_cv':   (0.18, 0.03),
        # DWT energy at 4 levels (L0–L3) for hold time
        'ht_dwt_L0_energy': (1800, 200), 'ht_dwt_L0_std': (30, 5),
        'ht_dwt_L1_energy': (900,  150), 'ht_dwt_L1_std': (22, 4),
        'ht_dwt_L2_energy': (400,   80), 'ht_dwt_L2_std': (15, 3),
        'ht_dwt_L3_energy': (180,   40), 'ht_dwt_L3_std': (10, 2),
        # DWT for flight time
        'ft_dwt_L0_energy': (3800, 350), 'ft_dwt_L0_std': (45, 8),
        'ft_dwt_L1_energy': (1900, 250), 'ft_dwt_L1_std': (32, 6),
        'ft_dwt_L2_energy': (850,  120), 'ft_dwt_L2_std': (22, 4),
        'ft_dwt_L3_energy': (380,   70), 'ft_dwt_L3_std': (14, 3),
    },
    'pd': {
        # PD: higher variability, longer hold times, asymmetric distributions
        'ht_mean': (145, 20),   'ht_std': (55, 12), 'ht_median': (135, 18),
        'ht_skew': (1.2, 0.4),  'ht_kurt': (2.1, 0.7), 'ht_iqr': (65, 15),
        'ht_cv':   (0.38, 0.07),
        'ft_mean': (210, 35),   'ft_std': (85, 18), 'ft_median': (195, 30),
        'ft_skew': (1.4, 0.5),  'ft_kurt': (2.5, 0.8), 'ft_iqr': (100, 22),
        'ft_cv':   (0.40, 0.08),
        'ht_dwt_L0_energy': (4200, 500),  'ht_dwt_L0_std': (68, 12),
        'ht_dwt_L1_energy': (2800, 400),  'ht_dwt_L1_std': (55, 10),
        'ht_dwt_L2_energy': (1800, 300),  'ht_dwt_L2_std': (42, 8),
        'ht_dwt_L3_energy': (900,  180),  'ht_dwt_L3_std': (28, 5),
        'ft_dwt_L0_energy': (8500, 900),  'ft_dwt_L0_std': (95, 18),
        'ft_dwt_L1_energy': (5200, 700),  'ft_dwt_L1_std': (72, 14),
        'ft_dwt_L2_energy': (2800, 450),  'ft_dwt_L2_std': (50, 10),
        'ft_dwt_L3_energy': (1400, 250),  'ft_dwt_L3_std': (32, 6),
    },
}
# MCI motor features are similar to healthy (cognitive disease, not motor)
MOTOR_PROFILES['mci'] = {k: (v[0] * 1.05, v[1] * 1.1)
                          for k, v in MOTOR_PROFILES['healthy'].items()}

# ── Cognitive features (linguistic markers) ───────────────────────────────────
COGNITIVE_PROFILES = {
    'healthy': {
        'ttr':             (0.72, 0.06),
        'mlu':             (9.5,  1.5),
        'filler_density':  (0.03, 0.01),
        'pronoun_density': (0.08, 0.02),
        'word_count':      (220,  45),
        'repetitions':     (2.5,  1.0),
        'sentence_count':  (22,   5),
        'unique_words':    (155,  30),
    },
    'mci': {
        # MCI: lower TTR, more fillers, more pronouns, fewer words
        'ttr':             (0.54, 0.08),
        'mlu':             (6.8,  1.8),
        'filler_density':  (0.09, 0.03),
        'pronoun_density': (0.16, 0.04),
        'word_count':      (135,  40),
        'repetitions':     (8.5,  3.0),
        'sentence_count':  (18,   5),
        'unique_words':    (72,   20),
    },
}
# PD cognitive features are similar to healthy (motor disease, not cognitive)
COGNITIVE_PROFILES['pd'] = {k: (v[0] * 0.97, v[1] * 1.05)
                              for k, v in COGNITIVE_PROFILES['healthy'].items()}


def generate_motor_dataset(n_per_class: int = 50,
                            random_state: int = 42) -> pd.DataFrame:
    """
    Generate synthetic motor feature dataset.

    Returns DataFrame with n_per_class * 3 rows and label column:
      0 = Healthy, 1 = PD, 2 = MCI
    """
    rng = np.random.default_rng(random_state)
    rows = []

    for label, profile_key in [(0, 'healthy'), (1, 'pd'), (2, 'mci')]:
        profile = MOTOR_PROFILES[profile_key]
        for _ in range(n_per_class):
            row = {}
            for feat, (mu, sigma) in profile.items():
                row[feat] = float(abs(rng.normal(mu, sigma)))  # clip negative
            row['label'] = label
            rows.append(row)

    df = pd.DataFrame(rows).sample(frac=1, random_state=random_state).reset_index(drop=True)
    return df


def generate_cognitive_dataset(n_per_class: int = 50,
                                random_state: int = 42) -> pd.DataFrame:
    """
    Generate synthetic cognitive feature dataset.

    Returns DataFrame with n_per_class * 3 rows and label column:
      0 = Healthy, 1 = PD, 2 = MCI
    """
    rng = np.random.default_rng(random_state)
    rows = []

    for label, profile_key in [(0, 'healthy'), (1, 'pd'), (2, 'mci')]:
        profile = COGNITIVE_PROFILES[profile_key]
        for _ in range(n_per_class):
            row = {}
            for feat, (mu, sigma) in profile.items():
                val = float(rng.normal(mu, sigma))
                # Clip proportional features to [0, 1]
                if feat in ('ttr', 'filler_density', 'pronoun_density'):
                    val = float(np.clip(val, 0.0, 1.0))
                else:
                    val = max(0.0, val)
                row[feat] = val
            row['label'] = label
            rows.append(row)

    df = pd.DataFrame(rows).sample(frac=1, random_state=random_state).reset_index(drop=True)
    return df


def save_datasets(out_dir: Path = None):
    """Generate and save both datasets to data/processed/."""
    if out_dir is None:
        out_dir = Path(__file__).parent.parent.parent / 'data' / 'processed'
    out_dir.mkdir(parents=True, exist_ok=True)

    motor_df = generate_motor_dataset()
    cog_df   = generate_cognitive_dataset()

    motor_path = out_dir / 'motor_features.csv'
    cog_path   = out_dir / 'cognitive_features.csv'

    motor_df.to_csv(motor_path, index=False)
    cog_df.to_csv(cog_path,   index=False)

    print(f'[Data] Motor features: {motor_df.shape}  → {motor_path}')
    print(f'[Data] Cognitive features: {cog_df.shape} → {cog_path}')
    return motor_df, cog_df


if __name__ == '__main__':
    save_datasets()
