"""Audio preprocessing: load, resample, VAD-trim, and MFCC feature extraction.

Mirrors Phase-2 Week 1-2 of the master plan: 16kHz resampling, 40-filterbank
MFCCs, and simple energy-based voice activity detection.
"""
from __future__ import annotations

import numpy as np
import librosa

from fluentnep.config import N_MFCC, SAMPLE_RATE


def load_audio(path: str, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Load an audio file (any format soundfile/librosa can decode) and
    resample to `sr` mono."""
    y, _ = librosa.load(path, sr=sr, mono=True)
    return y


def trim_silence(y: np.ndarray, top_db: int = 30) -> np.ndarray:
    """Energy-based voice activity detection: trims leading/trailing
    silence. Falls back to the original signal if trimming empties it."""
    trimmed, _ = librosa.effects.trim(y, top_db=top_db)
    return trimmed if trimmed.size > 0 else y


def compute_mfcc(y: np.ndarray, sr: int = SAMPLE_RATE, n_mfcc: int = N_MFCC) -> np.ndarray:
    """Returns MFCC features shaped (time, n_mfcc)."""
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    return mfcc.T.astype(np.float32)


def compute_mel_spectrogram(y: np.ndarray, sr: int = SAMPLE_RATE, n_mels: int = 80) -> np.ndarray:
    """Returns a log-mel spectrogram shaped (time, n_mels), used only for
    visualization (e.g. the proposal's spectrogram figure)."""
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
    log_mel = librosa.power_to_db(mel, ref=np.max)
    return log_mel.T.astype(np.float32)


def extract_features(path: str, sr: int = SAMPLE_RATE, n_mfcc: int = N_MFCC) -> np.ndarray:
    """Full pipeline: load -> trim -> MFCC. This is what both training data
    prep and real-time inference call."""
    y = load_audio(path, sr=sr)
    y = trim_silence(y)
    return compute_mfcc(y, sr=sr, n_mfcc=n_mfcc)


def extract_features_from_array(y: np.ndarray, sr: int = SAMPLE_RATE, n_mfcc: int = N_MFCC) -> np.ndarray:
    """Same as extract_features but for an in-memory waveform (used by the
    real-time microphone-streaming path)."""
    y = trim_silence(y)
    return compute_mfcc(y, sr=sr, n_mfcc=n_mfcc)
