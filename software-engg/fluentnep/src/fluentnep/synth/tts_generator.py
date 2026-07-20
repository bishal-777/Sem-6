"""Extends the synthetic disfluency generator to the audio level using TTS
(Phase-2 Week 4 of the master plan).

Each disfluent text sample is synthesized to a 16kHz mono WAV file. This
gives the AudioEncoder real acoustic signal (real pauses, real timing,
real filler-word pronunciation) to train the CTC model on, with zero manual
transcription — the transcript is known exactly because we generated it.

Limitation (documented, not hidden): gTTS speaks the romanized
Nepali-English text with an English voice, since no free neural TTS voice
exists for romanized code-mixed Nepali. Acoustic patterns (timing,
repetition, disfluency) are still real; pronunciation of Nepali words is
approximate. Swapping in a proper Nepali TTS voice (e.g. Coqui TTS
fine-tuned on Common Voice Nepali) later requires no changes downstream.
"""
from __future__ import annotations

import io
import json
import time
from pathlib import Path

import numpy as np
import soundfile as sf
from gtts import gTTS
from tqdm import tqdm

from fluentnep.config import AUDIO_DIR, SAMPLE_RATE


def synthesize_to_array(text: str, lang: str = "en", retries: int = 3) -> tuple[np.ndarray, int]:
    """Synthesizes `text` via gTTS and returns (waveform, sample_rate)."""
    last_err = None
    for attempt in range(retries):
        try:
            buf = io.BytesIO()
            gTTS(text=text, lang=lang, slow=False).write_to_fp(buf)
            buf.seek(0)
            y, sr = sf.read(buf, dtype="float32")
            if y.ndim > 1:
                y = y.mean(axis=1)
            return y, sr
        except Exception as e:  # network hiccups, rate limiting
            last_err = e
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"TTS synthesis failed for {text!r} after {retries} attempts") from last_err


def _resample(y: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    if orig_sr == target_sr:
        return y
    import librosa

    return librosa.resample(y, orig_sr=orig_sr, target_sr=target_sr)


def generate_audio_dataset(
    samples: list[dict],
    out_dir: Path = AUDIO_DIR,
    manifest_path: Path | None = None,
    sr: int = SAMPLE_RATE,
    limit: int | None = None,
) -> list[dict]:
    """Synthesizes audio for each sample dict (must have 'id' and 'text')
    and writes a JSONL manifest mapping id -> wav path + transcript + tags.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = manifest_path or (out_dir.parent / "audio_manifest.jsonl")

    todo = samples[:limit] if limit else samples
    manifest_entries = []
    failures = 0
    with open(manifest_path, "w") as f:
        for sample in tqdm(todo, desc="Synthesizing audio"):
            wav_path = out_dir / f"{sample['id']}.wav"
            try:
                y, orig_sr = synthesize_to_array(sample["text"])
                y = _resample(y, orig_sr, sr)
                sf.write(wav_path, y, sr)
            except Exception as e:
                failures += 1
                continue
            entry = {
                "id": sample["id"],
                "wav_path": str(wav_path),
                "text": sample["text"],
                "tokens": sample["tokens"],
                "tags": sample["tags"],
                "is_disfluent": sample["is_disfluent"],
                "duration_sec": len(y) / sr,
            }
            manifest_entries.append(entry)
            f.write(json.dumps(entry) + "\n")

    if failures:
        print(f"[tts_generator] {failures}/{len(todo)} samples failed TTS synthesis and were skipped.")
    return manifest_entries
