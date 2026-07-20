"""Word-level vocabulary for the DisfluencyTagger."""
from __future__ import annotations

import json
from pathlib import Path

PAD = "<pad>"
UNK = "<unk>"


class WordVocab:
    def __init__(self, words: list[str] | None = None):
        words = words or []
        self.id2word = [PAD, UNK] + words
        self.word2id = {w: i for i, w in enumerate(self.id2word)}

    def __len__(self) -> int:
        return len(self.id2word)

    @property
    def pad_id(self) -> int:
        return self.word2id[PAD]

    @property
    def unk_id(self) -> int:
        return self.word2id[UNK]

    def encode(self, tokens: list[str]) -> list[int]:
        return [self.word2id.get(t, self.unk_id) for t in tokens]

    @classmethod
    def build(cls, token_lists: list[list[str]], min_freq: int = 1) -> "WordVocab":
        freq: dict[str, int] = {}
        for tokens in token_lists:
            for t in tokens:
                freq[t] = freq.get(t, 0) + 1
        words = sorted([w for w, c in freq.items() if c >= min_freq])
        return cls(words)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.id2word[2:]))

    @classmethod
    def load(cls, path: str | Path) -> "WordVocab":
        words = json.loads(Path(path).read_text())
        return cls(words)
