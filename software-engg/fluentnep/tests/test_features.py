import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fluentnep.audio.features import compute_mfcc, trim_silence
from fluentnep.config import N_MFCC, SAMPLE_RATE


def _sine(freq=440.0, duration=1.0, sr=SAMPLE_RATE):
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    return (0.5 * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def test_mfcc_shape():
    y = _sine(duration=1.0)
    mfcc = compute_mfcc(y, sr=SAMPLE_RATE, n_mfcc=N_MFCC)
    assert mfcc.ndim == 2
    assert mfcc.shape[1] == N_MFCC
    assert mfcc.shape[0] > 0


def test_trim_silence_removes_leading_silence():
    silence = np.zeros(SAMPLE_RATE // 2, dtype=np.float32)
    tone = _sine(duration=0.5)
    y = np.concatenate([silence, tone])
    trimmed = trim_silence(y, top_db=30)
    assert len(trimmed) < len(y)
    assert len(trimmed) > 0
