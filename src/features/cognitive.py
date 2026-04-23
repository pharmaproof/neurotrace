"""
Cognitive Feature Pipeline — MCI Stream
=========================================
Extracts linguistic markers and optional BERT embeddings
from DementiaBank / Pitt Corpus transcripts.
"""

import re
import numpy as np
from collections import Counter
from pathlib import Path


# ── Transcript Parsing ────────────────────────────────────────────────────────

def parse_cha_transcript(filepath: str) -> str:
    """
    Extract participant speech from a CHAT (.cha) file.

    Lines starting with *PAR: are participant utterances.
    Returns a single concatenated string of clean speech.
    """
    speech_lines = []
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            if line.startswith('*PAR:'):
                text = line[5:]
                text = re.sub(r'\[.*?\]', '', text)        # Remove CHAT annotations
                text = re.sub(r'[&%@<>]\S*', '', text)     # Remove special markers
                text = re.sub(r'\s+', ' ', text).strip()
                if text:
                    speech_lines.append(text)
    return ' '.join(speech_lines)


def parse_text_transcript(filepath: str) -> str:
    """Load a plain-text transcript (one utterance per line)."""
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        return ' '.join(line.strip() for line in f if line.strip())


# ── Linguistic Feature Extraction ─────────────────────────────────────────────

FILLER_WORDS = {'um', 'uh', 'like', 'well', 'so', 'you know', 'hmm', 'er'}

PRONOUN_WORDS = {'it', 'they', 'he', 'she', 'this', 'that', 'there',
                 'them', 'these', 'those', 'its'}


def extract_linguistic_features(text: str) -> dict:
    """
    Extract lexical and syntactic markers linked to MCI in the literature.

    Features
    --------
    ttr             : Type-Token Ratio (lower → reduced vocabulary diversity)
    mlu             : Mean Length of Utterance in words
    filler_density  : Fraction of words that are filler words (um, uh, …)
    pronoun_density : Fraction of words that are vague pronouns
    word_count      : Total word count
    repetitions     : Number of content words repeated > 2 times
    sentence_count  : Number of utterances / sentences
    unique_words    : Vocabulary size
    """
    if not text or not text.strip():
        return _empty_linguistic_features()

    words     = text.lower().split()
    sentences = [s.strip() for s in re.split(r'[.!?]', text) if s.strip()]

    if not words:
        return _empty_linguistic_features()

    n = len(words)

    # Type-Token Ratio
    ttr = len(set(words)) / n

    # Mean Length of Utterance
    mlu = float(np.mean([len(s.split()) for s in sentences])) if sentences else 0.0

    # Filler word density
    filler_count   = sum(words.count(fw) for fw in FILLER_WORDS)
    filler_density = filler_count / n

    # Pronoun density
    pronoun_count   = sum(words.count(p) for p in PRONOUN_WORDS)
    pronoun_density = pronoun_count / n

    # Repetition rate — content words (len > 3) repeated > 2 times
    word_freq   = Counter(words)
    repetitions = sum(1 for w, c in word_freq.items() if c > 2 and len(w) > 3)

    return {
        'ttr':             ttr,
        'mlu':             mlu,
        'filler_density':  filler_density,
        'pronoun_density': pronoun_density,
        'word_count':      n,
        'repetitions':     repetitions,
        'sentence_count':  len(sentences),
        'unique_words':    len(set(words)),
    }


def _empty_linguistic_features() -> dict:
    """Return a zero-filled feature dict for empty / unparseable transcripts."""
    return {
        'ttr': 0.0, 'mlu': 0.0, 'filler_density': 0.0,
        'pronoun_density': 0.0, 'word_count': 0,
        'repetitions': 0, 'sentence_count': 0, 'unique_words': 0,
    }


# ── BERT Embeddings (optional) ────────────────────────────────────────────────

def get_bert_embedding(text: str,
                       model_name: str = 'bert-base-uncased',
                       max_len: int = 128) -> np.ndarray:
    """
    Return a 768-dim mean-pooled BERT embedding for the given text.

    This is optional — use linguistic features alone for the hackathon
    unless you have GPU and sufficient RAM (≥ 4 GB free).

    Returns zeros if transformers is not installed.
    """
    try:
        from transformers import AutoTokenizer, AutoModel
        import torch

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model     = AutoModel.from_pretrained(model_name)
        model.eval()

        inputs = tokenizer(
            text, return_tensors='pt',
            max_length=max_len, truncation=True, padding='max_length',
        )
        with torch.no_grad():
            outputs = model(**inputs)
        embedding = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
        return embedding  # shape (768,)
    except ImportError:
        print('[WARN] transformers not installed — returning zero embedding')
        return np.zeros(768, dtype=float)
    except Exception as exc:
        print(f'[WARN] BERT embedding failed: {exc}')
        return np.zeros(768, dtype=float)


# ── Dataset Builder ───────────────────────────────────────────────────────────

def build_cognitive_dataset(transcript_files: list,
                             labels: list,
                             use_bert: bool = False) -> 'pd.DataFrame':
    """
    Batch-process transcripts and return a feature DataFrame.

    Parameters
    ----------
    transcript_files : list of .cha or .txt filepaths
    labels           : corresponding int labels (0=Healthy, 1=MCI/AD)
    use_bert         : if True, append BERT embedding columns (slow, needs GPU)
    """
    import pandas as pd
    rows = []
    for filepath, label in zip(transcript_files, labels):
        try:
            fp = Path(filepath)
            text = (parse_cha_transcript(str(fp))
                    if fp.suffix == '.cha'
                    else parse_text_transcript(str(fp)))
            feats = extract_linguistic_features(text)
            if use_bert:
                emb = get_bert_embedding(text)
                for i, v in enumerate(emb):
                    feats[f'bert_{i}'] = float(v)
            feats['label'] = int(label)
            rows.append(feats)
        except Exception as exc:
            print(f'[WARN] Skipping {filepath}: {exc}')
    return pd.DataFrame(rows)
