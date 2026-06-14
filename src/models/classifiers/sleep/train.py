#!/usr/bin/env python3
# SPDX-License-Identifier: CC-BY-NC-4.0
"""
Train the sleep PSG classifier (sleep stages or arousals).

Reads the participant split from
``data/sleep_psg/ts_haystack/participant_split.json`` and builds per-split
sample lists from the WFDB annotations. Signals are loaded on demand via
``loader.load_window`` (memmap slice + decimation to 100 Hz + per-channel
z-score), so no signal cache is materialised on disk.

Usage:
    .venv/bin/python3 scripts/train_sleep_classifier.py \
        --label-class sleep_stages --epochs 30 --batch-size 64
    .venv/bin/python3 scripts/train_sleep_classifier.py \
        --label-class arousals --epochs 30 --batch-size 64
"""

from __future__ import annotations

import argparse
import json
import random
import time
from collections import Counter
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import balanced_accuracy_score, f1_score
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from src.datasets.sleep_psg_haystack.loader import (
    EFFECTIVE_HZ,
    SOURCE_HZ,
    load_annotations,
    load_window,
)
from src.models.classifiers.sleep.model import (
    AROUSAL_CLASS_NAMES,
    SLEEP_STAGE_CLASS_NAMES,
    SleepClassifier,
    default_class_names,
    default_window_samples,
)


PARTICIPANT_SPLIT = Path("data/sleep_psg/ts_haystack/participant_split.json")
SPLIT_NAME_MAP = {"train": "train", "val": "val", "validation": "val", "test": "test"}

# Arousal sub-config: which raw classes count as positives, which are excluded
# from negative sampling, and the dominating "none" class.
KEPT_AROUSAL_CLASSES = {
    "rera",
    "hypopnea",
    "obstructive_apnea",
    "central_apnea",
    "mixed_apnea",
}


def _samples_to_ms(sample_idx: int) -> int:
    return int(round(sample_idx * 1000 / SOURCE_HZ))


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------


class SleepStagesClassifierDataset(Dataset):
    """30 s AASM epochs, one (subject, start_ms, end_ms, label) per row."""

    EPOCH_SECONDS = 30

    def __init__(self, subject_ids: List[str]):
        self.class_names = list(SLEEP_STAGE_CLASS_NAMES)
        self.label_to_idx = {n: i for i, n in enumerate(self.class_names)}
        self.window_samples = default_window_samples("sleep_stages", EFFECTIVE_HZ)
        self.entries: List[Tuple[str, int, int, int]] = []

        epoch_native = self.EPOCH_SECONDS * SOURCE_HZ
        for sid in tqdm(subject_ids, desc="indexing sleep_stages"):
            try:
                anns = load_annotations(sid, "sleep_stages")
            except Exception as e:
                print(f"  skip {sid}: {e}")
                continue
            for start, end, label in anns:
                if label not in self.label_to_idx:
                    continue
                pos = start
                while pos + epoch_native <= end:
                    ws = _samples_to_ms(pos)
                    we = _samples_to_ms(pos + epoch_native)
                    self.entries.append((sid, ws, we, self.label_to_idx[label]))
                    pos += epoch_native

    def set_epoch(self, epoch: int) -> None:  # noqa: ARG002
        return  # deterministic tiling

    @property
    def labels(self) -> np.ndarray:
        return np.array([e[3] for e in self.entries], dtype=np.int64)

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int):
        sid, ws, we, label = self.entries[idx]
        signal = load_window(sid, ws, we)  # (13, ~3000)
        if signal.shape[1] < self.window_samples:
            pad = self.window_samples - signal.shape[1]
            signal = np.pad(signal, ((0, 0), (0, pad)))
        elif signal.shape[1] > self.window_samples:
            signal = signal[:, : self.window_samples]
        return torch.from_numpy(signal).float(), int(label)


class ArousalsClassifierDataset(Dataset):
    """20 s windows around arousal events, plus K * n_events 'none' negatives.

    Random offsets and negatives are re-rolled each epoch via ``set_epoch``.
    """

    WINDOW_SECONDS = 20
    MIN_OVERLAP_SECONDS = 10
    NONE_GAP_SECONDS = 10  # negatives must be ≥ this far from any annotation

    def __init__(
        self,
        subject_ids: List[str],
        negative_k: float = 1.0,
        base_seed: int = 42,
    ):
        self.class_names = list(AROUSAL_CLASS_NAMES)  # [..., "none"]
        self.label_to_idx = {n: i for i, n in enumerate(self.class_names)}
        self.none_idx = self.label_to_idx["none"]
        self.window_samples = default_window_samples("arousals", EFFECTIVE_HZ)
        self.window_native = self.WINDOW_SECONDS * SOURCE_HZ
        self.min_overlap_native = self.MIN_OVERLAP_SECONDS * SOURCE_HZ
        self.gap_native = self.NONE_GAP_SECONDS * SOURCE_HZ
        self.negative_k = float(negative_k)
        self.base_seed = int(base_seed)

        # Per-subject cached annotations.
        self._subject_anns: List[Tuple[str, List[Tuple[int, int, str]], int]] = []
        self.entries: List[Tuple[str, int, int, int]] = []

        for sid in tqdm(subject_ids, desc="indexing arousals"):
            try:
                anns = load_annotations(sid, "arousals")
            except Exception as e:
                print(f"  skip {sid}: {e}")
                continue
            from src.datasets.sleep_psg_haystack.loader import parse_header
            try:
                hdr = parse_header(sid)
            except Exception:
                continue
            self._subject_anns.append((sid, anns, int(hdr["n_samples"])))

        self.set_epoch(0)

    def set_epoch(self, epoch: int) -> None:
        rng = random.Random(self.base_seed * 1_000_003 + epoch)
        entries: List[Tuple[str, int, int, int]] = []

        for sid, anns, total_samples in self._subject_anns:
            kept = [(s, e, lbl) for (s, e, lbl) in anns if lbl in KEPT_AROUSAL_CLASSES]

            # Positives — one window per kept event with random offset.
            # Required overlap: target = min(min_overlap_native, event_len).
            # Valid window starts ws satisfy:
            #   ws+W >= start + target  ->  ws >= start - W + target
            #   ws   <= end   - target
            for start, end, lbl in kept:
                event_len = max(1, end - start)
                target = min(self.min_overlap_native, event_len)
                low = start - self.window_native + target
                high = end - target
                low = max(0, low)
                high = min(total_samples - self.window_native, high)
                if high < low:
                    # Window doesn't fit in the recording for this event — skip.
                    continue
                ws_native = rng.randint(low, high)
                we_native = ws_native + self.window_native
                entries.append((
                    sid,
                    _samples_to_ms(ws_native),
                    _samples_to_ms(we_native),
                    self.label_to_idx[lbl],
                ))

            # Negatives — sampled from gaps that are ≥ gap_native away from any annotation.
            n_neg = int(round(self.negative_k * len(kept)))
            if n_neg > 0:
                forbidden = sorted(
                    (max(0, s - self.gap_native), min(total_samples, e + self.gap_native))
                    for s, e, _ in anns
                )
                # Merge forbidden intervals.
                merged: List[Tuple[int, int]] = []
                for s, e in forbidden:
                    if merged and s <= merged[-1][1]:
                        merged[-1] = (merged[-1][0], max(merged[-1][1], e))
                    else:
                        merged.append((s, e))
                # Compute gap intervals where a window of window_native fits.
                gaps: List[Tuple[int, int]] = []
                cursor = 0
                for s, e in merged:
                    if s - cursor >= self.window_native:
                        gaps.append((cursor, s - self.window_native))
                    cursor = e
                if total_samples - cursor >= self.window_native:
                    gaps.append((cursor, total_samples - self.window_native))

                if gaps:
                    weights = [g[1] - g[0] + 1 for g in gaps]
                    for _ in range(n_neg):
                        gap = rng.choices(gaps, weights=weights, k=1)[0]
                        ws_native = rng.randint(gap[0], gap[1])
                        we_native = ws_native + self.window_native
                        entries.append((
                            sid,
                            _samples_to_ms(ws_native),
                            _samples_to_ms(we_native),
                            self.none_idx,
                        ))

        self.entries = entries

    @property
    def labels(self) -> np.ndarray:
        return np.array([e[3] for e in self.entries], dtype=np.int64)

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int):
        sid, ws, we, label = self.entries[idx]
        signal = load_window(sid, ws, we)  # (13, ~2000)
        if signal.shape[1] < self.window_samples:
            pad = self.window_samples - signal.shape[1]
            signal = np.pad(signal, ((0, 0), (0, pad)))
        elif signal.shape[1] > self.window_samples:
            signal = signal[:, : self.window_samples]
        return torch.from_numpy(signal).float(), int(label)


# ---------------------------------------------------------------------------
# Train / eval loops
# ---------------------------------------------------------------------------


def train_one_epoch(model, loader, criterion, optimizer, device, max_grad_norm=1.0):
    model.train()
    if model.freeze_encoder:
        model.chronos_encoder.eval()
    total_loss = 0.0
    n_batches = 0
    for ts_batch, label_batch in tqdm(loader, desc="  train", leave=False):
        ts_batch = ts_batch.to(device)
        label_batch = label_batch.to(device, dtype=torch.long)
        logits = model(ts_batch)
        loss = criterion(logits, label_batch)
        optimizer.zero_grad()
        loss.backward()
        if max_grad_norm > 0:
            nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        total_loss += loss.item()
        n_batches += 1
    return total_loss / max(n_batches, 1)


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    n_batches = 0
    all_preds, all_labels = [], []
    for ts_batch, label_batch in tqdm(loader, desc="  eval", leave=False):
        ts_batch = ts_batch.to(device)
        label_batch = label_batch.to(device, dtype=torch.long)
        logits = model(ts_batch)
        loss = criterion(logits, label_batch)
        total_loss += loss.item()
        n_batches += 1
        all_preds.extend(logits.argmax(dim=-1).cpu().numpy().tolist())
        all_labels.extend(label_batch.cpu().numpy().tolist())
    p, l = np.array(all_preds), np.array(all_labels)
    return {
        "loss": total_loss / max(n_batches, 1),
        "accuracy": float((p == l).mean()) if len(p) else 0.0,
        "balanced_accuracy": float(balanced_accuracy_score(l, p)) if len(p) else 0.0,
        "macro_f1": float(f1_score(l, p, average="macro", zero_division=0)) if len(p) else 0.0,
        "per_class_f1": f1_score(l, p, average=None, zero_division=0).tolist() if len(p) else [],
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def build_dataset(label_class: str, subject_ids: List[str], negative_k: float, seed: int):
    if label_class == "sleep_stages":
        return SleepStagesClassifierDataset(subject_ids)
    if label_class == "arousals":
        return ArousalsClassifierDataset(subject_ids, negative_k=negative_k, base_seed=seed)
    raise ValueError(f"Unknown label_class: {label_class}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label-class", choices=["sleep_stages", "arousals"], required=True)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output-dir", type=str, default="results/sleep_classifier")
    ap.add_argument("--negative-k", type=float, default=1.0,
                    help="Arousals: K * n_events negatives sampled per subject per epoch")
    ap.add_argument("--max-grad-norm", type=float, default=1.0)
    ap.add_argument("--wandb", action="store_true", default=True)
    ap.add_argument("--no-wandb", dest="wandb", action="store_false")
    ap.add_argument("--wandb-project", type=str, default="ts-haystack-sleep-psg")
    ap.add_argument("--max-train-subjects", type=int, default=None,
                    help="Optional cap for fast iteration / smoke testing")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    output_dir = Path(args.output_dir) / args.label_class
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(f"Sleep PSG classifier — label_class={args.label_class}")
    print("=" * 60)
    print(f"Device: {device}")

    with open(PARTICIPANT_SPLIT) as f:
        split = json.load(f)

    train_subjects = split["train"]
    val_subjects = split.get("val") or split.get("validation") or []
    if args.max_train_subjects:
        train_subjects = train_subjects[: args.max_train_subjects]

    print(f"Train subjects: {len(train_subjects)}, val subjects: {len(val_subjects)}")

    print("\nBuilding train dataset...")
    train_ds = build_dataset(args.label_class, train_subjects, args.negative_k, args.seed)
    print(f"  Train windows: {len(train_ds)}")
    print("\nBuilding val dataset...")
    val_ds = build_dataset(args.label_class, val_subjects, args.negative_k, args.seed + 1)
    print(f"  Val windows: {len(val_ds)}")

    class_names = default_class_names(args.label_class)
    num_classes = len(class_names)

    label_counts = Counter(train_ds.labels.tolist())
    total = sum(label_counts.values())
    class_weights = torch.tensor(
        [total / (num_classes * max(1, label_counts.get(i, 0))) for i in range(num_classes)],
        dtype=torch.float32,
    ).to(device)
    print("Class counts:", {class_names[i]: label_counts.get(i, 0) for i in range(num_classes)})

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True,
        # persistent_workers=False: workers are re-spawned each epoch so they
        # pick up the updated entries after set_epoch() is called on the main
        # process copy of the dataset.
        persistent_workers=False,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True, persistent_workers=args.num_workers > 0,
    )

    print("\nInitializing SleepClassifier...")
    model = SleepClassifier(
        num_classes=num_classes,
        class_names=class_names,
        window_samples=default_window_samples(args.label_class, EFFECTIVE_HZ),
        n_channels=13,
        freeze_encoder=True,
        device=device,
    ).to(device)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_p = sum(p.numel() for p in model.parameters())
    print(f"  Trainable params: {trainable:,} / {total_p:,}")

    wandb_run = None
    if args.wandb:
        try:
            import wandb
            wandb_run = wandb.init(
                project=args.wandb_project,
                name=f"sleep_classifier_{args.label_class}",
                config=vars(args),
            )
        except Exception as e:
            print(f"  W&B init failed: {e}")

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.head.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val_f1 = -1.0
    best_epoch = 0
    print("\nTraining...")
    for epoch in range(1, args.epochs + 1):
        if hasattr(train_ds, "set_epoch"):
            train_ds.set_epoch(epoch)
        t0 = time.time()
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, device,
            max_grad_norm=args.max_grad_norm,
        )
        val = evaluate(model, val_loader, criterion, device)
        scheduler.step()
        print(
            f"Epoch {epoch:3d}/{args.epochs} | "
            f"train_loss={train_loss:.4f} | val_loss={val['loss']:.4f} | "
            f"val_acc={val['accuracy']:.4f} | val_bal={val['balanced_accuracy']:.4f} | "
            f"val_f1={val['macro_f1']:.4f} | {time.time() - t0:.1f}s"
        )
        if wandb_run:
            import wandb
            log = {
                "epoch": epoch,
                "train/loss": train_loss,
                "val/loss": val["loss"],
                "val/accuracy": val["accuracy"],
                "val/balanced_accuracy": val["balanced_accuracy"],
                "val/macro_f1": val["macro_f1"],
                "lr": scheduler.get_last_lr()[0],
            }
            for name, f1 in zip(class_names, val["per_class_f1"]):
                log[f"val/f1_{name}"] = f1
            wandb.log(log)
        if val["macro_f1"] > best_val_f1:
            best_val_f1 = val["macro_f1"]
            best_epoch = epoch
            model.save(output_dir / "best_classifier.pt")

    print(f"\nBest val macro F1: {best_val_f1:.4f} at epoch {best_epoch}")

    # ------------------------------------------------------------------
    # Test evaluation — reload best checkpoint
    # ------------------------------------------------------------------
    print("\nBuilding test dataset...")
    test_subjects = split.get("test", [])
    test_ds = build_dataset(args.label_class, test_subjects, args.negative_k, args.seed + 2)
    print(f"  Test windows: {len(test_ds)}")
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )
    print("Evaluating best checkpoint on test set...")
    best_model = SleepClassifier.load(output_dir / "best_classifier.pt", device=device)
    test = evaluate(best_model, test_loader, criterion, device)

    print(f"\nTest Results:")
    print(f"  Accuracy:          {test['accuracy']:.4f}")
    print(f"  Balanced Accuracy: {test['balanced_accuracy']:.4f}")
    print(f"  Macro F1:          {test['macro_f1']:.4f}")
    print(f"\n  Per-class F1:")
    for name, f1 in zip(class_names, test["per_class_f1"]):
        print(f"    {name:25s}: {f1:.4f}")

    results = {
        "args": vars(args),
        "best_epoch": best_epoch,
        "best_val_macro_f1": best_val_f1,
        "test_metrics": test,
        "class_names": class_names,
    }
    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    if wandb_run:
        import wandb
        wandb.run.summary["best_epoch"] = best_epoch
        wandb.run.summary["best_val_macro_f1"] = best_val_f1
        wandb.run.summary["test/accuracy"] = test["accuracy"]
        wandb.run.summary["test/balanced_accuracy"] = test["balanced_accuracy"]
        wandb.run.summary["test/macro_f1"] = test["macro_f1"]
        for name, f1 in zip(class_names, test["per_class_f1"]):
            wandb.run.summary[f"test/f1_{name}"] = f1
        wandb.finish()

    print(f"\nResults saved to {output_dir}")


if __name__ == "__main__":
    main()
