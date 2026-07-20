"""PyTorch Dataset/collate implementations for both models."""
from __future__ import annotations

import json
from pathlib import Path

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from fluentnep.audio.features import extract_features
from fluentnep.data.char_vocab import CharVocab
from fluentnep.data.word_vocab import WordVocab


def load_jsonl(path: str | Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


class AudioCTCDataset(Dataset):
    """Loads (MFCC features, char-encoded transcript) pairs for the
    AudioEncoder / CTC training loop."""

    def __init__(self, manifest_entries: list[dict], char_vocab: CharVocab):
        self.entries = manifest_entries
        self.char_vocab = char_vocab

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int):
        entry = self.entries[idx]
        mfcc = extract_features(entry["wav_path"])  # (T, n_mfcc)
        target = self.char_vocab.encode(entry["text"])
        return {
            "mfcc": torch.tensor(mfcc, dtype=torch.float32),
            "target": torch.tensor(target, dtype=torch.long),
            "text": entry["text"],
        }


def ctc_collate_fn(batch: list[dict]):
    mfccs = [b["mfcc"] for b in batch]
    targets = [b["target"] for b in batch]

    input_lengths = torch.tensor([m.shape[0] for m in mfccs], dtype=torch.long)
    target_lengths = torch.tensor([len(t) for t in targets], dtype=torch.long)

    mfccs_padded = pad_sequence(mfccs, batch_first=True)  # (B, T_max, n_mfcc)
    targets_concat = torch.cat(targets)  # CTCLoss wants targets concatenated

    return {
        "mfcc": mfccs_padded,
        "input_lengths": input_lengths,
        "targets": targets_concat,
        "target_lengths": target_lengths,
        "texts": [b["text"] for b in batch],
    }


class DisfluencyDataset(Dataset):
    """Loads (word-token IDs, per-token disfluency tag IDs) pairs for the
    DisfluencyTagger training loop."""

    def __init__(self, samples: list[dict], word_vocab: WordVocab, tag2id: dict):
        self.samples = samples
        self.word_vocab = word_vocab
        self.tag2id = tag2id

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        token_ids = self.word_vocab.encode(sample["tokens"])
        tag_ids = [self.tag2id[t] for t in sample["tags"]]
        return {
            "tokens": torch.tensor(token_ids, dtype=torch.long),
            "tags": torch.tensor(tag_ids, dtype=torch.long),
        }


def tagger_collate_fn(batch: list[dict], pad_id: int = 0, ignore_index: int = -100):
    tokens = [b["tokens"] for b in batch]
    tags = [b["tags"] for b in batch]
    lengths = torch.tensor([len(t) for t in tokens], dtype=torch.long)

    tokens_padded = pad_sequence(tokens, batch_first=True, padding_value=pad_id)
    tags_padded = pad_sequence(tags, batch_first=True, padding_value=ignore_index)

    max_len = tokens_padded.size(1)
    key_padding_mask = torch.arange(max_len).unsqueeze(0) >= lengths.unsqueeze(1)  # True = pad

    return {
        "tokens": tokens_padded,
        "tags": tags_padded,
        "key_padding_mask": key_padding_mask,
        "lengths": lengths,
    }
