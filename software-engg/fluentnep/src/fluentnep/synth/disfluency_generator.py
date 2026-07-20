"""Synthetic Disfluency Generator — FluentNep's first research contribution.

Takes a clean code-mixed sentence and injects one or more of 5 disfluency
types, producing a token-aligned label sequence with zero manual annotation:

  FILLER        - insert a filler word before a content word
  REPETITION    - repeat a word 1-2 times
  FALSE_START   - truncate a word, abandon it, then say the full word
  REPAIR        - say a wrong word/span, then correct it
  PROLONGATION  - stretch the vowels of a word

Every injector operates on a list[(word, tag)] so label alignment can never
drift out of sync with the text.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field

from fluentnep.synth.vocab import FILLERS, REPAIR_CONNECTORS

Token = tuple[str, str]  # (word, tag)

VOWELS = "aeiou"


@dataclass
class DisfluentSample:
    id: str
    clean_text: str
    text: str
    tokens: list[str]
    tags: list[str]
    disfluency_types: list[str] = field(default_factory=list)
    is_disfluent: bool = False

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "clean_text": self.clean_text,
            "text": self.text,
            "tokens": self.tokens,
            "tags": self.tags,
            "disfluency_types": self.disfluency_types,
            "is_disfluent": self.is_disfluent,
        }


def _inject_filler(tokens: list[Token], rng: random.Random) -> list[Token]:
    if not tokens:
        return tokens
    idx = rng.randrange(len(tokens))
    filler = rng.choice(FILLERS)
    return tokens[:idx] + [(filler, "FILLER")] + tokens[idx:]


def _inject_repetition(tokens: list[Token], rng: random.Random) -> list[Token]:
    if not tokens:
        return tokens
    idx = rng.randrange(len(tokens))
    word, _ = tokens[idx]
    n_repeats = rng.choice([1, 1, 2])  # mostly single repeats
    repeats = [(word, "REPETITION") for _ in range(n_repeats)]
    return tokens[:idx] + repeats + tokens[idx:]


def _inject_false_start(tokens: list[Token], rng: random.Random) -> list[Token]:
    candidates = [i for i, (w, _) in enumerate(tokens) if len(w) >= 4]
    if not candidates:
        return tokens
    idx = rng.choice(candidates)
    word, _ = tokens[idx]
    cut = rng.randint(2, max(2, len(word) // 2))
    fragment = word[:cut] + "-"
    return tokens[:idx] + [(fragment, "FALSE_START")] + tokens[idx:]


def _inject_repair(tokens: list[Token], rng: random.Random) -> list[Token]:
    if not tokens:
        return tokens
    idx = rng.randrange(len(tokens))
    correct_word, _ = tokens[idx]
    # Fabricate a plausible "wrong" word by reusing another token in the
    # sentence, or a generic wrong number/word if the sentence is too short.
    other_words = [w for w, _ in tokens if w != correct_word]
    wrong_word = rng.choice(other_words) if other_words else "galat"
    connector = rng.choice(REPAIR_CONNECTORS)
    repair_span = [
        (wrong_word, "REPAIR"),
        (connector, "REPAIR"),
    ]
    return tokens[:idx] + repair_span + tokens[idx:]


def _inject_prolongation(tokens: list[Token], rng: random.Random) -> list[Token]:
    candidates = [i for i, (w, _) in enumerate(tokens) if any(v in w for v in VOWELS)]
    if not candidates:
        return tokens
    idx = rng.choice(candidates)
    word, _ = tokens[idx]
    # Stretch the first vowel found.
    for i, ch in enumerate(word):
        if ch in VOWELS:
            word = word[:i] + ch * 3 + word[i + 1 :]
            break
    tokens[idx] = (word, "PROLONGATION")
    return tokens


INJECTORS = {
    "FILLER": _inject_filler,
    "REPETITION": _inject_repetition,
    "FALSE_START": _inject_false_start,
    "REPAIR": _inject_repair,
    "PROLONGATION": _inject_prolongation,
}


class DisfluencyGenerator:
    """Injects synthetic disfluencies into clean code-mixed sentences."""

    def __init__(self, seed: int = 7):
        self.rng = random.Random(seed)

    def inject(self, clean_text: str, n_disfluencies: int | None = None) -> DisfluentSample:
        tokens: list[Token] = [(w, "O") for w in clean_text.split()]
        if n_disfluencies is None:
            n_disfluencies = self.rng.choice([1, 1, 2, 2, 3])

        types_applied: list[str] = []
        for _ in range(n_disfluencies):
            dtype = self.rng.choice(list(INJECTORS.keys()))
            new_tokens = INJECTORS[dtype](list(tokens), self.rng)
            if new_tokens != tokens:
                tokens = new_tokens
                types_applied.append(dtype)

        words = [w for w, _ in tokens]
        tags = [t for _, t in tokens]
        return DisfluentSample(
            id="",
            clean_text=clean_text,
            text=" ".join(words),
            tokens=words,
            tags=tags,
            disfluency_types=types_applied,
            is_disfluent=len(types_applied) > 0,
        )

    def make_fluent(self, clean_text: str) -> DisfluentSample:
        words = clean_text.split()
        return DisfluentSample(
            id="",
            clean_text=clean_text,
            text=clean_text,
            tokens=words,
            tags=["O"] * len(words),
            disfluency_types=[],
            is_disfluent=False,
        )

    def generate_dataset(
        self, clean_sentences: list[str], disfluent_ratio: float = 0.7
    ) -> list[DisfluentSample]:
        samples: list[DisfluentSample] = []
        for i, sentence in enumerate(clean_sentences):
            if self.rng.random() < disfluent_ratio:
                sample = self.inject(sentence)
            else:
                sample = self.make_fluent(sentence)
            sample.id = f"fluentnep_{i:06d}"
            samples.append(sample)
        return samples
