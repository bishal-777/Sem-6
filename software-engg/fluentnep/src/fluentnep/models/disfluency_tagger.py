"""Model 2 — DisfluencyTagger: Transformer sequence labeler that tags each
word token as fluent / filler / repetition / false_start / repair /
prolongation. Trained from scratch on the synthetic corpus.
"""
from __future__ import annotations

import torch
from torch import nn

from fluentnep.models.positional_encoding import PositionalEncoding


class DisfluencyTagger(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 4,
        dim_feedforward: int = 512,
        dropout: float = 0.2,
        n_tags: int = 6,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_enc = PositionalEncoding(d_model)
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=n_heads,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                batch_first=True,
            ),
            num_layers=n_layers,
        )
        self.tagger = nn.Linear(d_model, n_tags)

    def forward(self, x: torch.Tensor, src_key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        """x: (batch, seq_len) word-token IDs -> (batch, seq_len, n_tags)."""
        x = self.pos_enc(self.embedding(x))
        x = self.encoder(x, src_key_padding_mask=src_key_padding_mask)
        return self.tagger(x)
