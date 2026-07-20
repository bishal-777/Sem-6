"""Trains Model 1 — AudioEncoder (CNN + Transformer + CTC) — on the
synthesized audio corpus. Target from the master plan (Gate G5): WER < 40%
on a held-out split (that target assumes the full-scale real-data plan;
see README for what this scaled-down demo run actually achieves).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import jiwer
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split

from fluentnep.config import AudioEncoderConfig, CHECKPOINT_DIR, SYNTHETIC_DIR
from fluentnep.data.char_vocab import CharVocab
from fluentnep.data.datasets import AudioCTCDataset, ctc_collate_fn, load_jsonl
from fluentnep.models.audio_encoder import AudioEncoder


@torch.no_grad()
def evaluate(model, loader, char_vocab, device) -> float:
    model.eval()
    refs, hyps = [], []
    for batch in loader:
        mfcc = batch["mfcc"].to(device)
        logits = model(mfcc)
        pred_ids = logits.argmax(dim=-1).cpu().tolist()
        for ids, text in zip(pred_ids, batch["texts"]):
            hyps.append(char_vocab.ctc_greedy_decode(ids) or " ")
            refs.append(text)
    return jiwer.wer(refs, hyps)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--warmup-steps", type=int, default=200)
    parser.add_argument("--val-split", type=float, default=0.15)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    manifest_path = SYNTHETIC_DIR / "audio_manifest.jsonl"
    entries = load_jsonl(manifest_path)
    char_vocab = CharVocab.load(SYNTHETIC_DIR / "char_vocab.json")

    dataset = AudioCTCDataset(entries, char_vocab)
    n_val = max(1, int(len(dataset) * args.val_split))
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=ctc_collate_fn)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=ctc_collate_fn)

    cfg = AudioEncoderConfig()
    model = AudioEncoder(
        n_mfcc=cfg.n_mfcc,
        d_model=cfg.d_model,
        n_heads=cfg.n_heads,
        n_layers=cfg.n_layers,
        dim_feedforward=cfg.dim_feedforward,
        dropout=cfg.dropout,
        vocab_size=len(char_vocab),
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    ctc_loss = torch.nn.CTCLoss(blank=0, zero_infinity=True)

    # CTC from scratch collapses to "predict all blank" (a cheap local
    # minimum around loss ~= log(vocab_size)) almost every time without a
    # warmup: a too-large early update pushes every logit toward blank
    # before the model has learned anything useful to align with. A short
    # linear LR warmup is the standard fix.
    warmup_steps = max(1, args.warmup_steps)

    def lr_lambda(step: int) -> float:
        return min(1.0, (step + 1) / warmup_steps)

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    best_wer = float("inf")
    ckpt_path = CHECKPOINT_DIR / "audio_encoder.pt"

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        n_batches = 0
        for batch in train_loader:
            mfcc = batch["mfcc"].to(device)
            targets = batch["targets"].to(device)
            target_lengths = batch["target_lengths"].to(device)
            input_lengths = batch["input_lengths"].to(device)

            logits = model(mfcc)  # (B, T', vocab)
            output_lengths = AudioEncoder.output_length(input_lengths).clamp(min=1)
            # guard: CTC requires input_length >= target_length
            output_lengths = torch.minimum(output_lengths, torch.full_like(output_lengths, logits.size(1)))

            log_probs = F.log_softmax(logits, dim=-1).transpose(0, 1)  # (T', B, vocab)

            optimizer.zero_grad()
            loss = ctc_loss(log_probs, targets, output_lengths, target_lengths)
            if torch.isnan(loss):
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / max(n_batches, 1)
        val_wer = evaluate(model, val_loader, char_vocab, device)
        print(f"epoch {epoch:02d}  train_ctc_loss={avg_loss:.4f}  val_WER={val_wer:.4f}", flush=True)

        if val_wer < best_wer:
            best_wer = val_wer
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "config": cfg.__dict__,
                    "vocab_size": len(char_vocab),
                    "val_wer": val_wer,
                },
                ckpt_path,
            )

    print(f"Best val WER: {best_wer:.4f} (full-scale target < 0.40). Saved to {ckpt_path}")


if __name__ == "__main__":
    main()
