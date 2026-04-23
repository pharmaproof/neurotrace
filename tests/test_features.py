"""
Unit Tests for NeuroTrace Feature Pipelines & Utilities
"""

import sys
from pathlib import Path

# Ensure project root is on path
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pytest


# ── Motor Feature Tests ───────────────────────────────────────────────────────

class TestMotorFeatures:
    def test_extract_motor_features_returns_dict(self):
        from src.features.motor import extract_motor_features
        ht = np.random.normal(120, 20, 200).astype(float)
        ft = np.random.normal(160, 30, 199).astype(float)
        result = extract_motor_features(ht, ft)
        assert isinstance(result, dict)
        assert len(result) > 0

    def test_expected_stat_keys_present(self):
        from src.features.motor import extract_motor_features
        ht = np.abs(np.random.normal(120, 20, 200))
        ft = np.abs(np.random.normal(160, 30, 199))
        result = extract_motor_features(ht, ft)
        for key in ['ht_mean', 'ht_std', 'ht_cv', 'ft_mean', 'ft_std', 'ft_cv']:
            assert key in result, f'Missing key: {key}'

    def test_dwt_keys_present(self):
        from src.features.motor import extract_motor_features
        ht = np.abs(np.random.normal(120, 20, 200))
        ft = np.abs(np.random.normal(160, 30, 199))
        result = extract_motor_features(ht, ft)
        for lvl in range(4):
            assert f'ht_dwt_L{lvl}_energy' in result
            assert f'ft_dwt_L{lvl}_energy' in result

    def test_empty_arrays_handled(self):
        from src.features.motor import extract_motor_features
        ht = np.array([])
        ft = np.array([])
        # Should not raise — returns zero-filled dict
        result = extract_motor_features(ht, ft)
        assert isinstance(result, dict)

    def test_compute_hold_flight_times_structure(self):
        import pandas as pd
        from src.features.motor import compute_hold_flight_times
        data = {
            'action':    ['P', 'P', 'R', 'R', 'P', 'R'],
            'timestamp': [100, 150, 220, 290, 350, 420],
            'key':       ['a', 'b', 'a', 'b', 'c', 'c'],
        }
        df = pd.DataFrame(data)
        ht, ft = compute_hold_flight_times(df)
        assert isinstance(ht, np.ndarray)
        assert isinstance(ft, np.ndarray)
        assert len(ht) >= 0
        assert len(ft) == 2  # diff of 3 press timestamps


# ── Cognitive Feature Tests ───────────────────────────────────────────────────

class TestCognitiveFeatures:
    def test_extract_linguistic_returns_dict(self):
        from src.features.cognitive import extract_linguistic_features
        text = 'The quick brown fox jumps over the lazy dog. It is a well known sentence.'
        result = extract_linguistic_features(text)
        assert isinstance(result, dict)

    def test_expected_keys_present(self):
        from src.features.cognitive import extract_linguistic_features
        text = 'Hello world um so you know I was thinking well anyway.'
        result = extract_linguistic_features(text)
        for key in ['ttr', 'mlu', 'filler_density', 'pronoun_density',
                    'word_count', 'repetitions', 'sentence_count', 'unique_words']:
            assert key in result, f'Missing key: {key}'

    def test_ttr_in_range(self):
        from src.features.cognitive import extract_linguistic_features
        text = 'cat cat cat dog dog'
        result = extract_linguistic_features(text)
        assert 0.0 <= result['ttr'] <= 1.0

    def test_empty_text_handled(self):
        from src.features.cognitive import extract_linguistic_features
        result = extract_linguistic_features('')
        assert result['ttr'] == 0.0
        assert result['word_count'] == 0

    def test_filler_word_detected(self):
        from src.features.cognitive import extract_linguistic_features
        text = 'um I was um thinking uh about this um you know'
        result = extract_linguistic_features(text)
        assert result['filler_density'] > 0.0

    def test_parse_cha_ignores_non_par_lines(self):
        import tempfile, os
        from src.features.cognitive import parse_cha_transcript
        cha_content = (
            '@Begin\n'
            '*INV: Tell me what you see.\n'
            '*PAR: I can see a woman washing dishes.\n'
            '*PAR: There is a boy um getting a cookie.\n'
            '@End\n'
        )
        with tempfile.NamedTemporaryFile('w', suffix='.cha', delete=False,
                                         encoding='utf-8') as f:
            f.write(cha_content)
            tmp_path = f.name
        try:
            result = parse_cha_transcript(tmp_path)
            assert 'washing dishes' in result
            assert 'INV' not in result
        finally:
            os.unlink(tmp_path)


# ── Synthetic Data Tests ──────────────────────────────────────────────────────

class TestSyntheticData:
    def test_motor_dataset_shape(self):
        from src.data.generate_synthetic import generate_motor_dataset
        df = generate_motor_dataset(n_per_class=10)
        assert df.shape[0] == 30  # 10 * 3 classes
        assert 'label' in df.columns

    def test_motor_dataset_labels(self):
        from src.data.generate_synthetic import generate_motor_dataset
        df = generate_motor_dataset(n_per_class=10)
        assert set(df['label'].unique()) == {0, 1, 2}

    def test_cognitive_dataset_shape(self):
        from src.data.generate_synthetic import generate_cognitive_dataset
        df = generate_cognitive_dataset(n_per_class=10)
        assert df.shape[0] == 30
        assert 'label' in df.columns

    def test_cognitive_ttr_range(self):
        from src.data.generate_synthetic import generate_cognitive_dataset
        df = generate_cognitive_dataset(n_per_class=20)
        assert (df['ttr'] >= 0).all() and (df['ttr'] <= 1).all()

    def test_pd_profile_has_higher_ht_cv(self):
        from src.data.generate_synthetic import generate_motor_dataset
        df = generate_motor_dataset(n_per_class=50)
        healthy_cv = df[df['label'] == 0]['ht_cv'].mean()
        pd_cv      = df[df['label'] == 1]['ht_cv'].mean()
        assert pd_cv > healthy_cv, 'PD subjects should have higher ht_cv than Healthy'

    def test_mci_profile_has_lower_ttr(self):
        from src.data.generate_synthetic import generate_cognitive_dataset
        df = generate_cognitive_dataset(n_per_class=50)
        healthy_ttr = df[df['label'] == 0]['ttr'].mean()
        mci_ttr     = df[df['label'] == 2]['ttr'].mean()
        assert mci_ttr < healthy_ttr, 'MCI subjects should have lower TTR than Healthy'


# ── SHAP Utility Tests ────────────────────────────────────────────────────────

class TestShapUtils:
    def test_get_top_shap_features_returns_list(self):
        from src.explain.shap_utils import get_top_shap_features
        shap_vals = np.array([0.1, -0.5, 0.3, -0.05, 0.8])
        feat_names = ['a', 'b', 'c', 'd', 'e']
        result = get_top_shap_features(shap_vals, feat_names, n=3)
        assert len(result) == 3

    def test_top_feature_is_largest_abs(self):
        from src.explain.shap_utils import get_top_shap_features
        shap_vals  = np.array([0.1, -0.9, 0.3])
        feat_names = ['a', 'b', 'c']
        result = get_top_shap_features(shap_vals, feat_names, n=1)
        assert result[0]['feature'] == 'b'
        assert result[0]['direction'] == 'decreases'

    def test_direction_classification(self):
        from src.explain.shap_utils import get_top_shap_features
        shap_vals  = np.array([0.5, -0.3])
        feat_names = ['pos_feat', 'neg_feat']
        result = get_top_shap_features(shap_vals, feat_names, n=2)
        directions = {r['feature']: r['direction'] for r in result}
        assert directions['pos_feat'] == 'increases'
        assert directions['neg_feat'] == 'decreases'


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
