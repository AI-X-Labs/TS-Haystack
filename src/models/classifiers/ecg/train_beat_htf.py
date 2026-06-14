#!/usr/bin/env python3
# SPDX-License-Identifier: CC-BY-NC-4.0
"""Train the HTF beat classifier on LTAF (N / A / V).

Per-beat 2 s windows centered on R-peaks (matching the existing baseline)
plus RR-interval history and (teacher-forced) preceding-beat labels.

Usage:
    .venv/bin/python3 scripts/train_ecg_beat_htf.py \
        --epochs 15 --batch-size 256 --history-k 5
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
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from src.datasets.ltaf_haystack.loader import (
    LTAF_DATA_DIR,
    SOURCE_HZ,
    load_bout_signal,
    load_record_signals_mmap,
)
from src.models.classifiers.ecg.beat_htf import BEAT_CLASS_NAMES, EcgBeatHTFClassifier


HAYSTACK_DIR = LTAF_DATA_DIR / "ltaf_haystack"
SPLIT_MANIFEST = HAYSTACK_DIR / "split_manifest.json"
BEAT_TIMELINES_DIR = HAYSTACK_DIR / "beat_timelines"

WINDOW_SECONDS = 2
WINDOW_SAMPLES = WINDOW_SECONDS * SOURCE_HZ  # 256


def _zscore(chunk: np.ndarray) -> np.ndarray:
    mean = chunk.mean(axis=-1, keepdims=True)
    std = chunk.std(axis=-1, keepdims=True)
    return ((chunk - mean) / (std + 1e-6)).astype(np.float32, copy=False)


def _record_total_samples(record_id: str) -> int:
    return int(load_record_signals_mmap(record_id).shape[0])


class EcgBeatsHTFDataset(Dataset):
    """2 s windows centered on each beat R-peak with RR-interval and label history.

    For every beat in the record's beat timeline we pre-compute a (history_k,)
    vector of RR intervals (seconds) to the K preceding beats and a (history_k,)
    vector of preceding beat labels (-1 marks "no previous beat"). The kept
    label set is {N, A, V} (Q is dropped); preceding-beat positions whose
    symbol is Q are also marked -1 in the label history but their RR interval
    is still computed against their R-peak sample (since timing matters
    regardless of class).
    """

    def __init__(
        self,
        record_ids: List[str],
        history_k: int = 5,
        negative_k: float = 2.0,
        base_seed: int = 42,
    ):
        self.class_names = list(BEAT_CLASS_NAMES)
        self.label_to_idx = {n: i for i, n in enumerate(self.class_names)}
        self.history_k = history_k
        self.window_samples = WINDOW_SAMPLES
        self.half_native = WINDOW_SAMPLES // 2
        self.negative_k = float(negative_k)
        self.base_seed = int(base_seed)

        # Per-record cache for pulling history features.
        # beat_meta[rid] = (samples (N,), labels (N,) int8 with -1 for Q,
        #                   total_samples (int))
        self.beat_meta = {}
        # Per-record positive/negative beat indices (positions in beat_meta arrays).
        self._record_pos_neg: List[Tuple[str, np.ndarray, np.ndarray]] = []
        self.entries: List[Tuple[str, int, int]] = []  # (rid, beat_idx, label_idx)

        for rid in tqdm(record_ids, desc="indexing beats (HTF)"):
            beat_path = BEAT_TIMELINES_DIR / f"{rid}.parquet"
            if not beat_path.exists():
                print(f"  skip {rid}: no beat timeline")
                continue
            try:
                df = pd.read_parquet(beat_path, columns=["sample", "symbol"]).sort_values("sample")
                total_samples = _record_total_samples(rid)
            except Exception as e:
                print(f"  skip {rid}: {e}")
                continue

            samples = df["sample"].to_numpy(dtype=np.int64)
            symbols = df["symbol"].to_numpy()
            labels = np.full(len(samples), -1, dtype=np.int8)
            for sym, idx in self.label_to_idx.items():
                labels[symbols == sym] = idx

            # Beats whose 2 s window fits in the recording AND that are in {N,A,V}.
            valid = (
                (samples >= self.half_native)
                & (samples <= total_samples - self.half_native)
                & (labels >= 0)
            )
            valid_idx = np.flatnonzero(valid)
            if valid_idx.size == 0:
                continue

            valid_labels = labels[valid_idx]
            pos_mask = valid_labels != self.label_to_idx["N"]
            pos_idx = valid_idx[pos_mask]
            neg_idx = valid_idx[~pos_mask]

            self.beat_meta[rid] = (samples, labels, total_samples)
            self._record_pos_neg.append((rid, pos_idx, neg_idx))

        self.set_epoch(0)

    def set_epoch(self, epoch: int) -> None:
        rng = np.random.default_rng(self.base_seed * 1_000_003 + epoch)
        entries: List[Tuple[str, int, int]] = []
        for rid, pos_idx, neg_idx in self._record_pos_neg:
            samples, labels, _ = self.beat_meta[rid]
            for bi in pos_idx:
                entries.append((rid, int(bi), int(labels[bi])))
            n_target = int(round(self.negative_k * pos_idx.size))
            if n_target > 0 and neg_idx.size > 0:
                if n_target >= neg_idx.size:
                    picks = neg_idx
                else:
                    picks = rng.choice(neg_idx, size=n_target, replace=False)
                for bi in picks:
                    entries.append((rid, int(bi), int(labels[bi])))
        self.entries = entries

    @property
    def labels(self) -> np.ndarray:
        return np.array([e[2] for e in self.entries], dtype=np.int64)

    def __len__(self) -> int:
        return len(self.entries)

    def _history(self, rid: str, beat_idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """Return (rr_history (K,) seconds, label_history (K,) int with -1 missing).

        rr[k] = interval (in seconds) between beat (idx-k-1) and (idx-k);
        i.e. rr[0] is the immediately-preceding RR, rr[1] the one before
        that, etc. Missing positions (record start) get rr=0.0, label=-1.
        """
        samples, labels, _ = self.beat_meta[rid]
        rr = np.zeros(self.history_k, dtype=np.float32)
        lbl = np.full(self.history_k, -1, dtype=np.int64)
        for k in range(self.history_k):
            prev_pos = beat_idx - 1 - k
            if prev_pos < 0:
                break
            next_pos = prev_pos + 1  # always <= beat_idx for k >= 0
            rr[k] = (int(samples[next_pos]) - int(samples[prev_pos])) / SOURCE_HZ
            lbl[k] = int(labels[prev_pos])
        return rr, lbl

    def __getitem__(self, idx: int):
        rid, beat_idx, label = self.entries[idx]
        samples, _, _ = self.beat_meta[rid]
        center = int(samples[beat_idx])
        ws = center - self.half_native
        we = ws + self.window_samples
        signal = load_bout_signal(rid, ws, we)  # (L, 2)
        signal = np.ascontiguousarray(signal.T)  # (2, L)
        signal = _zscore(signal)
        if signal.shape[1] < self.window_samples:
            signal = np.pad(signal, ((0, 0), (0, self.window_samples - signal.shape[1])))
        elif signal.shape[1] > self.window_samples:
            signal = signal[:, : self.window_samples]
        rr, lbl = self._history(rid, beat_idx)
        return (
            torch.from_numpy(signal).float(),
            torch.from_numpy(rr).float(),
            torch.from_numpy(lbl).long(),
            int(label),
        )


def evaluate(model, loader, criterion, device, num_classes):
    model.eval()
    total_loss = 0.0
    n_batches = 0
    preds, labs = [], []
    with torch.no_grad():
        for x_time, rr, lbl, y in tqdm(loader, desc="  eval", leave=False):
            x_time = x_time.to(device)
            rr = rr.to(device)
            lbl = lbl.to(device)
            y = y.to(device, dtype=torch.long)
            logits = model(x_time, rr, lbl)
            loss = criterion(logits, y)
            total_loss += loss.item()
            n_batches += 1
            preds.extend(logits.argmax(-1).cpu().tolist())
            labs.extend(y.cpu().tolist())
    p, l = np.array(preds), np.array(labs)
    label_range = list(range(num_classes))
    return {
        "loss": total_loss / max(n_batches, 1),
        "accuracy": float((p == l).mean()) if len(p) else 0.0,
        "balanced_accuracy": float(balanced_accuracy_score(l, p)) if len(p) else 0.0,
        "macro_f1": float(f1_score(l, p, labels=label_range, average="macro", zero_division=0)) if len(p) else 0.0,
        "per_class_f1": f1_score(l, p, labels=label_range, average=None, zero_division=0).tolist() if len(p) else [],
        "confusion_matrix": confusion_matrix(l, p, labels=label_range).tolist() if len(p) else [],
    }


def train_one_epoch(model, loader, criterion, optimizer, device, max_grad_norm=1.0):
    model.train()
    total_loss = 0.0
    n_batches = 0
    for x_time, rr, lbl, y in tqdm(loader, desc="  train", leave=False):
        x_time = x_time.to(device)
        rr = rr.to(device)
        lbl = lbl.to(device)
        y = y.to(device, dtype=torch.long)
        logits = model(x_time, rr, lbl)
        loss = criterion(logits, y)
        optimizer.zero_grad()
        loss.backward()
        if max_grad_norm > 0:
            nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        total_loss += loss.item()
        n_batches += 1
    return total_loss / max(n_batches, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--history-k", type=int, default=5)
    ap.add_argument("--no-label-history", action="store_true")
    ap.add_argument("--negative-k", type=float, default=2.0)
    ap.add_argument("--time-channels", type=int, default=32)
    ap.add_argument("--freq-channels", type=int, default=32)
    ap.add_argument("--head-hidden", type=int, default=128)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--class-weight-power", type=float, default=0.5)
    ap.add_argument("--class-weight-cap", type=float, default=10.0)
    ap.add_argument("--label-smoothing", type=float, default=0.05)
    ap.add_argument("--output-dir", type=str, default="results/ecg_classifier/beats_htf")
    ap.add_argument("--max-grad-norm", type=float, default=1.0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Device: {device}, output: {output_dir}")

    with open(SPLIT_MANIFEST) as f:
        split = json.load(f)
    train_records = split["train"]
    val_records = split.get("validation", [])
    test_records = split.get("test", [])
    print(f"Train: {len(train_records)}, val: {len(val_records)}, test: {len(test_records)}")

    print("\nBuilding train...")
    train_ds = EcgBeatsHTFDataset(
        train_records, history_k=args.history_k,
        negative_k=args.negative_k, base_seed=args.seed,
    )
    print(f"  train entries: {len(train_ds)}")
    print("\nBuilding val...")
    val_ds = EcgBeatsHTFDataset(
        val_records, history_k=args.history_k,
        negative_k=args.negative_k, base_seed=args.seed + 1,
    )
    print(f"  val entries: {len(val_ds)}")
    print("\nBuilding test...")
    test_ds = EcgBeatsHTFDataset(
        test_records, history_k=args.history_k,
        negative_k=args.negative_k, base_seed=args.seed + 2,
    )
    print(f"  test entries: {len(test_ds)}")

    num_classes = len(BEAT_CLASS_NAMES)
    label_counts = Counter(train_ds.labels.tolist())
    total = sum(label_counts.values())
    raw_weights = [total / (num_classes * max(1, label_counts.get(i, 0)))
                   for i in range(num_classes)]
    dampened = [w ** args.class_weight_power for w in raw_weights]
    final_weights = [min(args.class_weight_cap, w) for w in dampened]
    class_weights = torch.tensor(final_weights, dtype=torch.float32).to(device)
    print(f"  Class counts: {dict(zip(BEAT_CLASS_NAMES, [label_counts.get(i,0) for i in range(num_classes)]))}")
    print(f"  Class weights: {dict(zip(BEAT_CLASS_NAMES, [round(w,3) for w in final_weights]))}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True,
                              persistent_workers=False)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True,
                            persistent_workers=args.num_workers > 0)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=True,
                             persistent_workers=args.num_workers > 0)

    model = EcgBeatHTFClassifier(
        num_classes=num_classes,
        class_names=BEAT_CLASS_NAMES,
        n_channels=2,
        window_samples=WINDOW_SAMPLES,
        history_k=args.history_k,
        history_use_labels=not args.no_label_history,
        time_base_channels=args.time_channels,
        freq_base_channels=args.freq_channels,
        head_hidden=args.head_hidden,
        dropout=args.dropout,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nHTF model params: {n_params:,}")

    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=args.label_smoothing)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val_f1 = -1.0
    best_epoch = 0
    history = []
    print("\nTraining...")
    for epoch in range(1, args.epochs + 1):
        train_ds.set_epoch(epoch)
        t0 = time.time()
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device,
                                     max_grad_norm=args.max_grad_norm)
        val = evaluate(model, val_loader, criterion, device, num_classes)
        scheduler.step()
        elapsed = time.time() - t0
        history.append({"epoch": epoch, "train_loss": train_loss, "val": val, "elapsed_s": elapsed})
        print(f"Epoch {epoch:3d}/{args.epochs} | "
              f"train_loss={train_loss:.4f} | val_loss={val['loss']:.4f} | "
              f"val_acc={val['accuracy']:.4f} | val_bal={val['balanced_accuracy']:.4f} | "
              f"val_f1={val['macro_f1']:.4f} | per-class={[round(x,3) for x in val['per_class_f1']]} | "
              f"{elapsed:.1f}s")
        if val["macro_f1"] > best_val_f1:
            best_val_f1 = val["macro_f1"]
            best_epoch = epoch
            model.save(output_dir / "best_classifier.pt")

    print(f"\nBest val macro F1: {best_val_f1:.4f} at epoch {best_epoch}")

    print("\nLoading best checkpoint, evaluating on test...")
    best_model = EcgBeatHTFClassifier.load(output_dir / "best_classifier.pt", device=device)
    test = evaluate(best_model, test_loader, criterion, device, num_classes)

    print(f"\nTest Results:")
    print(f"  Accuracy:          {test['accuracy']:.4f}")
    print(f"  Balanced Accuracy: {test['balanced_accuracy']:.4f}")
    print(f"  Macro F1:          {test['macro_f1']:.4f}")
    for name, f1 in zip(BEAT_CLASS_NAMES, test["per_class_f1"]):
        print(f"    {name}: F1={f1:.4f}")
    print("  Confusion matrix (rows=true, cols=pred, order N,A,V):")
    for row in test["confusion_matrix"]:
        print(f"    {row}")

    results = {
        "args": vars(args),
        "best_epoch": best_epoch,
        "best_val_macro_f1": best_val_f1,
        "test_metrics": test,
        "history": history,
        "class_names": BEAT_CLASS_NAMES,
        "n_params": n_params,
    }
    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_dir}/results.json")


if __name__ == "__main__":
    main()
