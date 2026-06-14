#!/usr/bin/env python3
# SPDX-License-Identifier: CC-BY-NC-4.0
"""Train RhythmResNet1D on the LTAF top-6 rhythm classes.

This is the production training recipe for the LTAF rhythm classifier.
Defaults reproduce the F1 0.658 (TTA-7) result.

Usage:
    .venv/bin/python3 -m src.models.classifiers.ecg.train_rhythm \
        --epochs 30 --batch-size 64 --lr 5e-4 \
        --base-channels 64 \
        --use-val-as-train \
        --output-dir results/ecg_classifier/sweep/c6_resnet1d_w10_e30_wide
"""

from __future__ import annotations

import argparse
import json
import random
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score
from torch.utils.data import ConcatDataset, DataLoader
from tqdm.auto import tqdm

from src.datasets.ltaf_haystack.loader import SOURCE_HZ
from src.datasets.ltaf_haystack.rhythm_classifier_dataset import (
    EcgRhythmsClassifierDataset,
    collect_labels,
    load_split_manifest,
    set_epoch_recursive,
)
from src.models.classifiers.ecg.rhythm import (
    RHYTHM_CLASS_NAMES_6,
    RhythmResNet1D,
)


def evaluate(model, loader, criterion, device, num_classes):
    model.eval()
    total = 0.0
    n = 0
    preds, labs = [], []
    with torch.no_grad():
        for x, y in tqdm(loader, desc="  eval", leave=False):
            x = x.to(device)
            y = y.to(device, dtype=torch.long)
            logits = model(x)
            loss = criterion(logits, y)
            total += loss.item()
            n += 1
            preds.extend(logits.argmax(-1).cpu().tolist())
            labs.extend(y.cpu().tolist())
    p, l = np.array(preds), np.array(labs)
    label_range = list(range(num_classes))
    return {
        "loss": total / max(n, 1),
        "accuracy": float((p == l).mean()) if len(p) else 0.0,
        "balanced_accuracy": float(balanced_accuracy_score(l, p)) if len(p) else 0.0,
        "macro_f1": float(f1_score(l, p, labels=label_range, average="macro", zero_division=0)) if len(p) else 0.0,
        "per_class_f1": f1_score(l, p, labels=label_range, average=None, zero_division=0).tolist() if len(p) else [],
        "confusion_matrix": confusion_matrix(l, p, labels=label_range).tolist() if len(p) else [],
    }


def train_one_epoch(model, loader, criterion, optimizer, device, max_grad_norm=1.0):
    model.train()
    total = 0.0
    n = 0
    for x, y in tqdm(loader, desc="  train", leave=False):
        x = x.to(device)
        y = y.to(device, dtype=torch.long)
        logits = model(x)
        loss = criterion(logits, y)
        optimizer.zero_grad()
        loss.backward()
        if max_grad_norm > 0:
            nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        total += loss.item()
        n += 1
    return total / max(n, 1)


def build_dataset(records, seed, window_sizes, encoder_window, classes):
    subsets = []
    for ws in window_sizes:
        ds = EcgRhythmsClassifierDataset(
            records, window_seconds=ws, encoder_window=encoder_window,
            base_seed=seed + int(ws * 10), classes=classes,
        )
        subsets.append(ds)
        print(f"    {ws}s: {len(ds)} windows")
    if len(subsets) == 1:
        return subsets[0]
    return ConcatDataset(subsets)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--window-sizes", type=float, nargs="+", default=[10.0])
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--base-channels", type=int, default=64)
    ap.add_argument("--class-weight-power", type=float, default=0.5)
    ap.add_argument("--class-weight-cap", type=float, default=10.0)
    ap.add_argument("--label-smoothing", type=float, default=0.1)
    ap.add_argument("--max-grad-norm", type=float, default=1.0)
    ap.add_argument("--output-dir", type=str, required=True)
    ap.add_argument("--use-val-as-train", action="store_true",
                    help="Combine train+val into one training set; still evaluates "
                         "on test. Last 8 records of the combined set are held out "
                         "as a proxy val for early-stopping.")
    ap.add_argument("--classes", type=str, nargs="+", default=RHYTHM_CLASS_NAMES_6,
                    help="Subset of LTAF rhythm classes to keep "
                         "(default: NSR AFIB SBR AB SVTA B).")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Device: {device}, output: {output_dir}")

    split = load_split_manifest()
    train_records = split["train"]
    val_records = split.get("validation", [])
    test_records = split.get("test", [])
    if args.use_val_as_train:
        train_records = train_records + val_records
        val_records = train_records[-8:]
        train_records = train_records[:-8]
        print("** using train+val merged for training **")
    print(f"Train: {len(train_records)}, val: {len(val_records)}, test: {len(test_records)}")

    classes = list(args.classes)
    print(f"Classes ({len(classes)}): {classes}")

    encoder_window = int(round(max(args.window_sizes) * SOURCE_HZ))
    print(f"Encoder window samples: {encoder_window} ({max(args.window_sizes)} s)")

    print("\nBuilding train dataset...")
    train_ds = build_dataset(train_records, args.seed, args.window_sizes, encoder_window, classes)
    print(f"  Train windows: {len(train_ds)}")
    print("\nBuilding val dataset...")
    val_ds = build_dataset(val_records, args.seed + 1, args.window_sizes, encoder_window, classes)
    print(f"  Val windows: {len(val_ds)}")

    num_classes = len(classes)
    label_counts = Counter(collect_labels(train_ds).tolist())
    total = sum(label_counts.values())
    raw_weights = [total / (num_classes * max(1, label_counts.get(i, 0)))
                   for i in range(num_classes)]
    dampened = [w ** args.class_weight_power for w in raw_weights]
    final_weights = [min(args.class_weight_cap, w) for w in dampened]
    class_weights = torch.tensor(final_weights, dtype=torch.float32).to(device)
    print(f"  Class counts: {dict(zip(classes, [label_counts.get(i,0) for i in range(num_classes)]))}")
    print(f"  Class weights: {dict(zip(classes, [round(w,3) for w in final_weights]))}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True,
                              persistent_workers=False)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True,
                            persistent_workers=args.num_workers > 0)

    model = RhythmResNet1D(
        num_classes=num_classes, class_names=classes, n_channels=2,
        base_channels=args.base_channels, dropout=args.dropout,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nRhythmResNet1D params: {n_params:,}")

    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=args.label_smoothing)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val_f1 = -1.0
    best_epoch = 0
    history = []
    print("\nTraining...")
    for epoch in range(1, args.epochs + 1):
        set_epoch_recursive(train_ds, epoch)
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
              f"val_f1={val['macro_f1']:.4f} | per-class={[round(x,2) for x in val['per_class_f1']]} | "
              f"{elapsed:.1f}s")
        if val["macro_f1"] > best_val_f1:
            best_val_f1 = val["macro_f1"]
            best_epoch = epoch
            model.save(output_dir / "best_classifier.pt")

    print(f"\nBest val macro F1: {best_val_f1:.4f} at epoch {best_epoch}")

    print("\nBuilding test dataset (largest window only)...")
    test_window_sizes = [max(args.window_sizes)]
    test_ds = build_dataset(test_records, args.seed + 2, test_window_sizes, encoder_window, classes)
    print(f"  Test windows: {len(test_ds)}")
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=True,
                             persistent_workers=args.num_workers > 0)
    best_model = RhythmResNet1D.load(output_dir / "best_classifier.pt", device=device)
    test = evaluate(best_model, test_loader, criterion, device, num_classes)
    print(f"\nTest Results:")
    print(f"  Accuracy:          {test['accuracy']:.4f}")
    print(f"  Balanced Accuracy: {test['balanced_accuracy']:.4f}")
    print(f"  Macro F1:          {test['macro_f1']:.4f}")
    for name, f1 in zip(classes, test["per_class_f1"]):
        print(f"    {name:6s}: F1={f1:.4f}")

    results = {
        "args": vars(args),
        "best_epoch": best_epoch,
        "best_val_macro_f1": best_val_f1,
        "test_metrics": test,
        "history": history,
        "class_names": classes,
        "n_params": n_params,
    }
    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_dir}/results.json")


if __name__ == "__main__":
    main()
