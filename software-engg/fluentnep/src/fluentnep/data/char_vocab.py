"""Character-level vocabulary for the CTC AudioEncoder.

Index 0 is reserved for the CTC blank token, as required by
torch.nn.CTCLoss (blank=0).
"""
from __future__ import annotations

import json
from pathlib import Path

BLANK = "<blank>"

_ALPHABET = list("abcdefghijklmnopqrstuvwxyz")
_EXTRA = [" ", "'", "-"]


class CharVocab:
    def __init__(self, chars: list[str] | None = None):
        chars = chars or (_ALPHABET + _EXTRA)
        self.id2char = [BLANK] + chars
        self.char2id = {c: i for i, c in enumerate(self.id2char)}

    def __len__(self) -> int:
        return len(self.id2char)

    def encode(self, text: str) -> list[int]:
        text = text.lower()
        return [self.char2id[c] for c in text if c in self.char2id]

    def decode(self, ids: list[int]) -> str:
        return "".join(self.id2char[i] for i in ids if i != 0)

    def ctc_greedy_decode(self, ids: list[int]) -> str:
        """Collapse repeats and drop blanks, per standard CTC decoding."""
        out = []
        prev = None
        for i in ids:
            if i != prev and i != 0:
                out.append(self.id2char[i])
            prev = i
        return "".join(out)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.id2char[1:]))

    @classmethod
    def load(cls, path: str | Path) -> "CharVocab":
        chars = json.loads(Path(path).read_text())
        return cls(chars)
