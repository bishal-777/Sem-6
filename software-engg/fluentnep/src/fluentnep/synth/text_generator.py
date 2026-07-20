"""Generates clean (disfluency-free) romanized Nepali-English code-mixed
sentences from templates, for use as the "ground truth fluent" base that the
DisfluencyGenerator then injects disfluencies into.
"""
import random

from fluentnep.synth.vocab import (
    CONNECTORS,
    ENGLISH_NOUNS,
    ENGLISH_VERBS,
    NEP_ADJ,
    NEP_TIME,
    PRONOUNS,
    SENTENCE_TEMPLATES,
    VERBS_NEP,
)


def generate_clean_sentence(rng: random.Random) -> str:
    template = rng.choice(SENTENCE_TEMPLATES)
    sentence = template.format(
        pron=rng.choice(PRONOUNS),
        verb_nep=rng.choice(VERBS_NEP),
        conn=rng.choice(CONNECTORS),
        eng_noun=rng.choice(ENGLISH_NOUNS),
        eng_verb=rng.choice(ENGLISH_VERBS),
        time=rng.choice(NEP_TIME),
        adj=rng.choice(NEP_ADJ),
    )
    return sentence


def generate_clean_corpus(n: int, seed: int = 42) -> list[str]:
    """Generate `n` unique clean code-mixed sentences."""
    rng = random.Random(seed)
    seen: set[str] = set()
    out: list[str] = []
    attempts = 0
    max_attempts = n * 50
    while len(out) < n and attempts < max_attempts:
        attempts += 1
        s = generate_clean_sentence(rng)
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out
