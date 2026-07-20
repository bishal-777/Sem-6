"""Generates a spectrogram + MFCC figure from a synthetic audio sample —
the "real evidence" artifact the master plan's proposal chapter wants
(Phase 0, Day 3 / Day 8: "include a spectrogram image of actual speech").

Usage:
  python scripts/plot_spectrogram.py --wav data/synthetic/audio/fluentnep_000001.wav
  python scripts/plot_spectrogram.py --random   # picks a random synthesized clip
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib.pyplot as plt
import librosa
import librosa.display

from fluentnep.config import SAMPLE_RATE, SYNTHETIC_DIR


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wav", type=str, default=None)
    parser.add_argument("--random", action="store_true")
    parser.add_argument("--out", type=str, default=str(SYNTHETIC_DIR / "spectrogram_example.png"))
    args = parser.parse_args()

    if args.wav:
        wav_path = args.wav
        text = ""
    else:
        manifest = SYNTHETIC_DIR / "audio_manifest.jsonl"
        entries = [json.loads(l) for l in manifest.read_text().splitlines() if l.strip()]
        entry = random.choice(entries) if args.random else entries[0]
        wav_path = entry["wav_path"]
        text = entry["text"]

    y, sr = librosa.load(wav_path, sr=SAMPLE_RATE)
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=80)
    log_mel = librosa.power_to_db(mel, ref=1.0)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)

    fig, axes = plt.subplots(2, 1, figsize=(10, 6))
    img0 = librosa.display.specshow(log_mel, sr=sr, x_axis="time", y_axis="mel", ax=axes[0])
    axes[0].set_title(f"Log-Mel Spectrogram — {Path(wav_path).name}")
    fig.colorbar(img0, ax=axes[0], format="%+2.0f dB")

    img1 = librosa.display.specshow(mfcc, sr=sr, x_axis="time", ax=axes[1])
    axes[1].set_title("MFCC (40 coefficients)")
    fig.colorbar(img1, ax=axes[1])

    if text:
        fig.suptitle(f'Transcript: "{text}"', fontsize=10)
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"Saved figure to {args.out}")


if __name__ == "__main__":
    main()
