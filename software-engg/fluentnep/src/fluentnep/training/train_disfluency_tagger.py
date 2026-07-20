"""Trains Model 2 — DisfluencyTagger — on the synthetic text corpus.

Target from the master plan (Gate G6): macro F1 > 0.65 on a held-out split.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
from torch import nn
from torch.utils.data import DataLoader, random_split

from fluentnep.config import (
    CHECKPOINT_DIR,
    DISFLUENCY_TAGS,
    DisfluencyTaggerConfig,
    SYNTHETIC_DIR,
    TAG2ID,
)
from fluentnep.data.datasets import DisfluencyDataset, load_jsonl, tagger_collate_fn
from fluentnep.data.word_vocab import WordVocab
from fluentnep.models.disfluency_tagger import DisfluencyTagger


def macro_f1(all_preds: list[int], all_labels: list[int], n_tags: int) -> tuple[float, dict]:
    per_tag = {}
    f1s = []
    for tag_id in range(n_tags):
        tp = sum(1 for p, l in zip(all_preds, all_labels) if p == tag_id and l == tag_id)
        fp = sum(1 for p, l in zip(all_preds, all_labels) if p == tag_id and l != tag_id)
        fn = sum(1 for p, l in zip(all_preds, all_labels) if p != tag_id and l == tag_id)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        per_tag[DISFLUENCY_TAGS[tag_id]] = round(f1, 4)
        f1s.append(f1)
    return sum(f1s) / len(f1s), per_tag


def evaluate(model, loader, device) -> tuple[float, dict]:
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            tokens = batch["tokens"].to(device)
            tags = batch["tags"].to(device)
            mask = batch["key_padding_mask"].to(device)
            logits = model(tokens, src_key_padding_mask=mask)
            preds = logits.argmax(dim=-1)
            valid = tags != -100
            all_preds.extend(preds[valid].cpu().tolist())
            all_labels.extend(tags[valid].cpu().tolist())
    return macro_f1(all_preds, all_labels, len(DISFLUENCY_TAGS))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--val-split", type=float, default=0.15)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    samples = load_jsonl(SYNTHETIC_DIR / "text_samples.jsonl")
    word_vocab = WordVocab.load(SYNTHETIC_DIR / "word_vocab.json")

    dataset = DisfluencyDataset(samples, word_vocab, TAG2ID)
    n_val = max(1, int(len(dataset) * args.val_split))
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=tagger_collate_fn)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=tagger_collate_fn)

    cfg = DisfluencyTaggerConfig()
    model = DisfluencyTagger(
        vocab_size=len(word_vocab),
        d_model=cfg.d_model,
        n_heads=cfg.n_heads,
        n_layers=cfg.n_layers,
        dim_feedforward=cfg.dim_feedforward,
        dropout=cfg.dropout,
        n_tags=cfg.n_tags,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss(ignore_index=-100)

    best_f1 = 0.0
    ckpt_path = CHECKPOINT_DIR / "disfluency_tagger.pt"

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            tokens = batch["tokens"].to(device)
            tags = batch["tags"].to(device)
            mask = batch["key_padding_mask"].to(device)

            optimizer.zero_grad()
            logits = model(tokens, src_key_padding_mask=mask)
            loss = criterion(logits.reshape(-1, logits.size(-1)), tags.reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()

        val_f1, per_tag = evaluate(model, val_loader, device)
        avg_loss = total_loss / len(train_loader)
        print(f"epoch {epoch:02d}  train_loss={avg_loss:.4f}  val_macro_f1={val_f1:.4f}  per_tag={per_tag}")

        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "config": cfg.__dict__,
                    "vocab_size": len(word_vocab),
                    "val_macro_f1": val_f1,
                },
                ckpt_path,
            )

    print(f"Best val macro F1: {best_f1:.4f} (target > 0.65). Saved to {ckpt_path}")


if __name__ == "__main__":
    main()
