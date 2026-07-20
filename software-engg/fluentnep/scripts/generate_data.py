"""Orchestrates synthetic data generation end to end:

  1. Generate N clean code-mixed sentences (text_generator)
  2. Inject disfluencies with aligned labels (disfluency_generator)
  3. Write the full text+label dataset to JSONL
  4. Synthesize audio for a subset via TTS (tts_generator)
  5. Build & save the char vocab (CTC) and word vocab (tagger)

Usage:
  python scripts/generate_data.py --n-text 800 --n-audio 400
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fluentnep.config import SYNTHETIC_DIR
from fluentnep.data.char_vocab import CharVocab
from fluentnep.data.word_vocab import WordVocab
from fluentnep.synth.disfluency_generator import DisfluencyGenerator
from fluentnep.synth.text_generator import generate_clean_corpus
from fluentnep.synth.tts_generator import generate_audio_dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-text", type=int, default=800, help="Number of text samples to generate")
    parser.add_argument("--n-audio", type=int, default=400, help="Number of those samples to synthesize as audio")
    parser.add_argument("--disfluent-ratio", type=float, default=0.7)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-audio", action="store_true")
    args = parser.parse_args()

    print(f"[1/5] Generating {args.n_text} clean code-mixed sentences...")
    clean_sentences = generate_clean_corpus(args.n_text, seed=args.seed)

    print(f"[2/5] Injecting synthetic disfluencies (ratio={args.disfluent_ratio})...")
    gen = DisfluencyGenerator(seed=args.seed)
    dataset = gen.generate_dataset(clean_sentences, disfluent_ratio=args.disfluent_ratio)

    text_path = SYNTHETIC_DIR / "text_samples.jsonl"
    with open(text_path, "w") as f:
        for sample in dataset:
            f.write(json.dumps(sample.to_dict()) + "\n")
    n_disfluent = sum(s.is_disfluent for s in dataset)
    print(f"      wrote {len(dataset)} samples ({n_disfluent} disfluent, {len(dataset) - n_disfluent} clean) -> {text_path}")

    print("[3/5] Building char vocab (CTC) and word vocab (tagger)...")
    char_vocab = CharVocab()
    char_vocab.save(SYNTHETIC_DIR / "char_vocab.json")
    word_vocab = WordVocab.build([s.tokens for s in dataset])
    word_vocab.save(SYNTHETIC_DIR / "word_vocab.json")
    print(f"      char vocab size={len(char_vocab)}, word vocab size={len(word_vocab)}")

    if args.skip_audio:
        print("[4/5] Skipping audio synthesis (--skip-audio)")
    else:
        print(f"[4/5] Synthesizing audio for {args.n_audio} samples via gTTS (this hits the network)...")
        samples_dicts = [s.to_dict() for s in dataset]
        entries = generate_audio_dataset(samples_dicts, limit=args.n_audio)
        print(f"      synthesized {len(entries)} audio files -> {SYNTHETIC_DIR / 'audio'}")

    print("[5/5] Done.")


if __name__ == "__main__":
    main()
