"""Central configuration for the FluentNep pipeline."""
from dataclasses import dataclass
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data"
SYNTHETIC_DIR = DATA_DIR / "synthetic"
AUDIO_DIR = SYNTHETIC_DIR / "audio"
CHECKPOINT_DIR = ROOT_DIR / "checkpoints"
LOG_DIR = ROOT_DIR / "logs"

for d in (DATA_DIR, SYNTHETIC_DIR, AUDIO_DIR, CHECKPOINT_DIR, LOG_DIR):
    d.mkdir(parents=True, exist_ok=True)

SAMPLE_RATE = 16_000
N_MFCC = 40

DISFLUENCY_TAGS = [
    "O",            # fluent
    "FILLER",
    "REPETITION",
    "FALSE_START",
    "REPAIR",
    "PROLONGATION",
]
TAG2ID = {t: i for i, t in enumerate(DISFLUENCY_TAGS)}
ID2TAG = {i: t for t, i in TAG2ID.items()}


@dataclass
class AudioEncoderConfig:
    n_mfcc: int = N_MFCC
    d_model: int = 256
    n_heads: int = 8
    n_layers: int = 4
    dim_feedforward: int = 512
    dropout: float = 0.1


@dataclass
class DisfluencyTaggerConfig:
    d_model: int = 256
    n_heads: int = 8
    n_layers: int = 4
    dim_feedforward: int = 512
    dropout: float = 0.2
    n_tags: int = len(DISFLUENCY_TAGS)
