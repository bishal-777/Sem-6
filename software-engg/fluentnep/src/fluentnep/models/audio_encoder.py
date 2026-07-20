"""Model 1 — AudioEncoder: CNN feature extractor + Transformer encoder +
CTC output head, trained from scratch (no pretrained weights), per the
master plan's architecture (Section 2.3).
"""
from __future__ import annotations

import torch
from torch import nn

from fluentnep.models.positional_encoding import PositionalEncoding


class AudioEncoder(nn.Module):
    def __init__(
        self,
        n_mfcc: int = 40,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 4,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        vocab_size: int = 30,
    ):
        super().__init__()
        self.n_mfcc = n_mfcc

        # CNN: extracts local acoustic patterns from MFCC frames.
        # Pools over the *frequency* axis only (/4), time axis untouched.
        # Character-level CTC needs roughly one output frame per character;
        # pooling time here (as a naive CNN+CTC design often does) collapses
        # a ~150-frame utterance to ~35 frames, which is shorter than most
        # transcripts -> every sample becomes an unsatisfiable CTC alignment
        # and the loss silently zeroes out for the whole training run. See
        # master plan Gate G4 troubleshooting: "Check CTC label lengths."
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1), nn.ReLU(), nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(), nn.BatchNorm2d(64),
            nn.MaxPool2d((1, 2)),
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.ReLU(), nn.BatchNorm2d(128),
            nn.MaxPool2d((1, 2)),
        )
        freq_out = n_mfcc // 4
        self.cnn_proj = nn.Linear(128 * freq_out, d_model)
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
        self.ctc_head = nn.Linear(d_model, vocab_size)

    def forward(self, x: torch.Tensor, src_key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        """x: (batch, time, n_mfcc) -> logits (batch, time, vocab_size)."""
        x = x.unsqueeze(1)  # (B, 1, T, n_mfcc)
        x = self.cnn(x)  # (B, 128, T, n_mfcc/4) -- time axis preserved
        B, C, T, F = x.shape
        x = x.permute(0, 2, 1, 3).reshape(B, T, C * F)
        x = self.cnn_proj(x)  # (B, T, d_model)
        x = self.pos_enc(x)
        x = self.encoder(x, src_key_padding_mask=src_key_padding_mask)
        return self.ctc_head(x)  # (B, T, vocab_size)

    @staticmethod
    def output_length(input_length: torch.Tensor) -> torch.Tensor:
        """Maps raw frame-count lengths to post-CNN time-step lengths, for
        CTCLoss's input_lengths argument. The CNN only pools frequency now,
        so the time dimension is unchanged."""
        return input_length
