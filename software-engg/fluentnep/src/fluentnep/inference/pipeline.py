"""End-to-end inference: raw audio -> transcript -> disfluency tags ->
fluency score. This is what both the FastAPI WebSocket endpoint and any
batch/offline script call.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from fluentnep.audio.features import extract_features, extract_features_from_array
from fluentnep.config import CHECKPOINT_DIR, ID2TAG, SYNTHETIC_DIR, AudioEncoderConfig, DisfluencyTaggerConfig
from fluentnep.data.char_vocab import CharVocab
from fluentnep.data.word_vocab import WordVocab
from fluentnep.models.audio_encoder import AudioEncoder
from fluentnep.models.disfluency_tagger import DisfluencyTagger


@dataclass
class DisfluencyEvent:
    word: str
    tag: str
    index: int


@dataclass
class InferenceResult:
    transcript: str
    tokens: list[str]
    tags: list[str]
    disfluencies: list[DisfluencyEvent] = field(default_factory=list)
    fluency_score: float = 1.0
    n_words: int = 0
    n_disfluent_words: int = 0

    def to_dict(self) -> dict:
        return {
            "transcript": self.transcript,
            "tokens": self.tokens,
            "tags": self.tags,
            "disfluencies": [d.__dict__ for d in self.disfluencies],
            "fluency_score": round(self.fluency_score, 3),
            "n_words": self.n_words,
            "n_disfluent_words": self.n_disfluent_words,
        }


class FluentNepPipeline:
    """Loads both trained models once and exposes a single `run()` call for
    real-time and batch inference alike."""

    def __init__(
        self,
        audio_ckpt: Path = CHECKPOINT_DIR / "audio_encoder.pt",
        tagger_ckpt: Path = CHECKPOINT_DIR / "disfluency_tagger.pt",
        char_vocab_path: Path = SYNTHETIC_DIR / "char_vocab.json",
        word_vocab_path: Path = SYNTHETIC_DIR / "word_vocab.json",
        device: str | None = None,
    ):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        self.char_vocab = CharVocab.load(char_vocab_path)
        self.word_vocab = WordVocab.load(word_vocab_path)

        audio_state = torch.load(audio_ckpt, map_location=self.device)
        acfg = AudioEncoderConfig(**audio_state["config"])
        self.audio_encoder = AudioEncoder(
            n_mfcc=acfg.n_mfcc,
            d_model=acfg.d_model,
            n_heads=acfg.n_heads,
            n_layers=acfg.n_layers,
            dim_feedforward=acfg.dim_feedforward,
            dropout=acfg.dropout,
            vocab_size=audio_state["vocab_size"],
        ).to(self.device)
        self.audio_encoder.load_state_dict(audio_state["model_state"])
        self.audio_encoder.eval()

        tagger_state = torch.load(tagger_ckpt, map_location=self.device)
        tcfg = DisfluencyTaggerConfig(**tagger_state["config"])
        self.tagger = DisfluencyTagger(
            vocab_size=tagger_state["vocab_size"],
            d_model=tcfg.d_model,
            n_heads=tcfg.n_heads,
            n_layers=tcfg.n_layers,
            dim_feedforward=tcfg.dim_feedforward,
            dropout=tcfg.dropout,
            n_tags=tcfg.n_tags,
        ).to(self.device)
        self.tagger.load_state_dict(tagger_state["model_state"])
        self.tagger.eval()

    @torch.no_grad()
    def _transcribe(self, mfcc: np.ndarray) -> str:
        x = torch.tensor(mfcc, dtype=torch.float32, device=self.device).unsqueeze(0)
        logits = self.audio_encoder(x)
        ids = logits.argmax(dim=-1)[0].cpu().tolist()
        return self.char_vocab.ctc_greedy_decode(ids)

    @torch.no_grad()
    def _tag(self, tokens: list[str]) -> tuple[list[str], list[float]]:
        if not tokens:
            return [], []
        ids = torch.tensor([self.word_vocab.encode(tokens)], dtype=torch.long, device=self.device)
        logits = self.tagger(ids)
        probs = F.softmax(logits, dim=-1)[0]
        pred_ids = probs.argmax(dim=-1).cpu().tolist()
        confidences = probs.max(dim=-1).values.cpu().tolist()
        tags = [ID2TAG[i] for i in pred_ids]
        return tags, confidences

    def run(self, wav_path: str | None = None, waveform: np.ndarray | None = None, sr: int = 16_000) -> InferenceResult:
        if wav_path is not None:
            mfcc = extract_features(wav_path)
        elif waveform is not None:
            mfcc = extract_features_from_array(waveform, sr=sr)
        else:
            raise ValueError("Provide either wav_path or waveform")

        if mfcc.shape[0] < 8:  # too short for the CNN's two stride-2 pools
            return InferenceResult(transcript="", tokens=[], tags=[])

        transcript = self._transcribe(mfcc)
        tokens = transcript.split()
        tags, _confidences = self._tag(tokens)

        disfluencies = [
            DisfluencyEvent(word=w, tag=t, index=i)
            for i, (w, t) in enumerate(zip(tokens, tags))
            if t != "O"
        ]
        n_words = len(tokens)
        n_disfluent = len(disfluencies)
        fluency_score = 1.0 if n_words == 0 else max(0.0, 1.0 - n_disfluent / n_words)

        return InferenceResult(
            transcript=transcript,
            tokens=tokens,
            tags=tags,
            disfluencies=disfluencies,
            fluency_score=fluency_score,
            n_words=n_words,
            n_disfluent_words=n_disfluent,
        )

    def run_on_text(self, text: str) -> InferenceResult:
        """Bypasses the audio model — tags disfluencies directly from text.
        Used by the dashboard's "type instead of speak" fallback and for
        quickly sanity-checking the tagger alone."""
        tokens = text.split()
        tags, _ = self._tag(tokens)
        disfluencies = [
            DisfluencyEvent(word=w, tag=t, index=i)
            for i, (w, t) in enumerate(zip(tokens, tags))
            if t != "O"
        ]
        n_words = len(tokens)
        n_disfluent = len(disfluencies)
        fluency_score = 1.0 if n_words == 0 else max(0.0, 1.0 - n_disfluent / n_words)
        return InferenceResult(
            transcript=text,
            tokens=tokens,
            tags=tags,
            disfluencies=disfluencies,
            fluency_score=fluency_score,
            n_words=n_words,
            n_disfluent_words=n_disfluent,
        )
