#!/usr/bin/env python3
# SPDX-License-Identifier: CC-BY-NC-4.0
"""
Standalone training script for the HAR classifier (Phase 1).

Trains an encoder + classification head on Capture-24 with WillettsSpecific2018
labels.  Supports frozen (feature extraction) and unfrozen (fine-tuning) modes.

Trains on mixed window sizes (1s, 3s, 6s, 10s) so the model learns
to handle zero-padded short bouts natively.

Prerequisites:
    # Extract windows at each size (all downsampled to 30 Hz)
    for W in 1 3 6 10; do
        python3 src/datasets/capture24/windows.py --window-size-s $W --source-hz 100 --downsample-hz 30
        python3 src/datasets/capture24/classification.py --window-size-s $W --effective-hz 30 --label-scheme WillettsSpecific2018 --min-confidence 0.6
    done

Usage:
    # Frozen encoder (fast, baseline)
    python3 scripts/train_classifier.py --epochs 50 --lr 1e-3 --batch-size 128

    # Fine-tune encoder from scratch (higher F1, slower)
    python3 scripts/train_classifier.py --no-freeze-encoder --finetune-after 5 --encoder-lr 1e-5 --epochs 30

    # Resume from frozen dual checkpoint and fine-tune encoder
    python3 scripts/train_classifier.py --resume results/classifier/dual/best_classifier.pt \
        --no-freeze-encoder --encoder-lr 1e-5 --epochs 20 --encoder-type dual
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as torchF
from torch.utils.data import ConcatDataset, DataLoader, Dataset
from sklearn.metrics import balanced_accuracy_score, f1_score
from tqdm.auto import tqdm

from src.datasets.capture24.classification import (
    ensure_numpy_cache,
    load_classification_metadata,
)
from src.models.classifiers.capture24.model import HARClassifier


ENCODER_WINDOW = 300  # 10 s @ 30 Hz — fixed encoder input size


# ---------------------------------------------------------------------------
# Dataset wrapper
# ---------------------------------------------------------------------------


class Capture24ClassificationDataset(Dataset):
    """PyTorch Dataset wrapping a Capture-24 classification split.

    Uses memory-mapped numpy arrays instead of loading everything into RAM.
    All windows are zero-padded to 300 samples (10 s @ 30 Hz) to match
    the encoder's expected input size.
    """

    def __init__(self, window_size_s: float, effective_hz: int, label_scheme: str, split: str):
        cache_dir = ensure_numpy_cache(window_size_s, effective_hz, label_scheme, split)
        self.x = np.load(cache_dir / "x.npy", mmap_mode="r")
        self.y = np.load(cache_dir / "y.npy", mmap_mode="r")
        self.z = np.load(cache_dir / "z.npy", mmap_mode="r")
        self.labels = np.load(cache_dir / "labels.npy", mmap_mode="r")
        self.window_size_s = window_size_s

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        ts = torch.from_numpy(
            np.stack([self.x[idx], self.y[idx], self.z[idx]])
        ).float()  # (3, L)
        # Zero-pad to encoder window size
        L = ts.shape[1]
        if L < ENCODER_WINDOW:
            ts = torchF.pad(ts, (0, ENCODER_WINDOW - L))
        return ts, int(self.labels[idx])


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def train_one_epoch(
    model: HARClassifier,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: str,
    freeze_encoder: bool = True,
    max_grad_norm: float = 0.0,
) -> float:
    model.train()
    # Only force eval on frozen encoders (preserves BN running stats);
    # when fine-tuning, let them train normally.
    if freeze_encoder:
        if model.encoder is not None:
            model.encoder.eval()
        if model.chronos_encoder is not None:
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
def evaluate(
    model: HARClassifier,
    loader: DataLoader,
    criterion: nn.Module,
    device: str,
) -> dict:
    model.eval()
    total_loss = 0.0
    n_batches = 0
    all_preds = []
    all_labels = []

    for ts_batch, label_batch in tqdm(loader, desc="  eval", leave=False):
        ts_batch = ts_batch.to(device)
        label_batch = label_batch.to(device, dtype=torch.long)

        logits = model(ts_batch)
        loss = criterion(logits, label_batch)

        total_loss += loss.item()
        n_batches += 1

        preds = logits.argmax(dim=-1).cpu().numpy()
        all_preds.extend(preds.tolist())
        all_labels.extend(label_batch.cpu().numpy().tolist())

    all_preds_np = np.array(all_preds)
    all_labels_np = np.array(all_labels)

    accuracy = (all_preds_np == all_labels_np).mean()
    balanced_acc = balanced_accuracy_score(all_labels_np, all_preds_np)
    macro_f1 = f1_score(all_labels_np, all_preds_np, average="macro", zero_division=0)
    per_class_f1 = f1_score(all_labels_np, all_preds_np, average=None, zero_division=0)

    return {
        "loss": total_loss / max(n_batches, 1),
        "accuracy": accuracy,
        "balanced_accuracy": balanced_acc,
        "macro_f1": macro_f1,
        "per_class_f1": per_class_f1.tolist(),
    }


def build_datasets(
    window_sizes: list[float],
    effective_hz: int,
    label_scheme: str,
    split: str,
) -> ConcatDataset:
    """Build a ConcatDataset from multiple window sizes."""
    datasets = []
    for ws in window_sizes:
        try:
            ds = Capture24ClassificationDataset(ws, effective_hz, label_scheme, split)
            print(f"  {ws}s: {len(ds)} windows")
            datasets.append(ds)
        except FileNotFoundError:
            print(f"  {ws}s: NOT FOUND — skipping (run windows.py + classification.py for this size)")
    if not datasets:
        raise RuntimeError("No datasets found for any window size")
    return ConcatDataset(datasets)


def main():
    parser = argparse.ArgumentParser(description="Train HAR classifier on Capture-24")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--output-dir", type=str, default="results/classifier")
    parser.add_argument(
        "--window-sizes", type=float, nargs="+", default=[1.0, 3.0, 6.0, 10.0],
        help="Window sizes in seconds to train on (default: 1 3 6 10)",
    )
    parser.add_argument("--effective-hz", type=int, default=30)
    parser.add_argument("--label-scheme", type=str, default="WillettsSpecific2018")
    parser.add_argument(
        "--encoder-type", type=str, default="oxwearables",
        choices=["oxwearables", "chronos2", "dual"],
        help="Encoder backend: oxwearables, chronos2, or dual (default: oxwearables)",
    )
    parser.add_argument("--chronos-model-id", type=str, default="amazon/chronos-2")
    parser.add_argument(
        "--resume", type=str, default=None,
        help="Path to a checkpoint to resume from (e.g. results/classifier/dual/best_classifier.pt)",
    )
    parser.add_argument("--freeze-encoder", action="store_true", default=True)
    parser.add_argument("--no-freeze-encoder", dest="freeze_encoder", action="store_false")
    parser.add_argument(
        "--encoder-lr", type=float, default=None,
        help="Learning rate for encoder params when fine-tuning (default: lr / 10)",
    )
    parser.add_argument(
        "--finetune-after", type=int, default=0,
        help="Train head-only for this many epochs before unfreezing encoder (default: 0 = immediate)",
    )
    parser.add_argument(
        "--max-grad-norm", type=float, default=1.0,
        help="Max gradient norm for clipping (0 = disabled, default: 1.0)",
    )
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging")
    parser.add_argument("--wandb-project", type=str, default="ts-haystack-classifier")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    mode_tag = "frozen" if args.freeze_encoder else "finetune"
    output_dir = Path(args.output_dir) / args.encoder_type / mode_tag
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("HAR Classifier Training (mixed window sizes)")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Encoder: {args.encoder_type}")
    print(f"Window sizes: {args.window_sizes}s @ {args.effective_hz}Hz")
    print(f"Label scheme: {args.label_scheme}")
    print(f"Freeze encoder: {args.freeze_encoder}")
    if not args.freeze_encoder:
        encoder_lr = args.encoder_lr if args.encoder_lr is not None else args.lr / 10
        print(f"Encoder LR: {encoder_lr}, Finetune after: {args.finetune_after} epochs")
    print(f"Epochs: {args.epochs}, LR: {args.lr}, Batch size: {args.batch_size}")
    print()

    # Load metadata from the largest window size for class info
    ref_ws = max(args.window_sizes)
    metadata = load_classification_metadata(ref_ws, args.effective_hz, args.label_scheme)
    class_names = metadata["class_names"]
    num_classes = metadata["num_classes"]
    print(f"Classes ({num_classes}): {class_names}")

    # Create mixed-size datasets
    print("\nLoading train datasets...")
    train_ds = build_datasets(args.window_sizes, args.effective_hz, args.label_scheme, "train")
    print(f"  Total train: {len(train_ds)}")

    print("\nLoading val datasets...")
    val_ds = build_datasets(args.window_sizes, args.effective_hz, args.label_scheme, "val")
    print(f"  Total val: {len(val_ds)}")

    # Test only on 10s windows (matches inference conditions)
    print("\nLoading test dataset (10s only)...")
    test_ds = Capture24ClassificationDataset(10.0, args.effective_hz, args.label_scheme, "test")
    print(f"  Test (10s): {len(test_ds)}")

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )

    # Compute class weights for imbalanced data (from train set)
    all_labels = np.concatenate([ds.labels[:] for ds in train_ds.datasets])
    label_counts = Counter(all_labels.tolist())
    total = sum(label_counts.values())
    class_weights = torch.tensor(
        [total / (num_classes * label_counts.get(i, 1)) for i in range(num_classes)],
        dtype=torch.float32,
    ).to(device)

    # Create or resume model
    if args.resume:
        print(f"\nResuming from checkpoint: {args.resume}")
        model = HARClassifier.load(
            args.resume, device=device, freeze_encoder=args.freeze_encoder,
        ).to(device)
        print(f"  Loaded encoder_type={model.encoder_type}, freeze={args.freeze_encoder}")
    else:
        print("\nInitializing model...")
        model = HARClassifier(
            num_classes=num_classes,
            class_names=class_names,
            encoder_type=args.encoder_type,
            pretrained_encoder=True,
            freeze_encoder=args.freeze_encoder,
            chronos_model_id=args.chronos_model_id,
            device=device,
        ).to(device)
    print(f"  Feature dim: {model.feature_dim}")

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Trainable params: {trainable_params:,} / {total_params:,}")

    # wandb setup
    wandb_run = None
    if args.wandb:
        try:
            import wandb

            wandb_run = wandb.init(
                project=args.wandb_project,
                name=f"har_{args.encoder_type}_{mode_tag}_{'_'.join(str(int(w)) for w in args.window_sizes)}s",
                config=vars(args),
            )
            print(f"  W&B run: {wandb.run.url}")
        except Exception as e:
            print(f"  W&B init failed: {e} — continuing without wandb")

    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # Build optimizer with discriminative LR when fine-tuning
    encoder_lr = args.encoder_lr if args.encoder_lr is not None else args.lr / 10
    currently_frozen = args.freeze_encoder or args.finetune_after > 0

    def build_optimizer(finetune_encoder: bool) -> torch.optim.AdamW:
        """Build optimizer with appropriate param groups."""
        if finetune_encoder:
            encoder_params = []
            if model.encoder is not None:
                encoder_params += list(model.encoder.parameters())
            if model.chronos_encoder is not None:
                encoder_params += list(model.chronos_encoder.parameters())
            param_groups = [
                {"params": list(model.head.parameters()), "lr": args.lr},
                {"params": encoder_params, "lr": encoder_lr},
            ]
        else:
            param_groups = [{"params": list(model.head.parameters()), "lr": args.lr}]
        return torch.optim.AdamW(param_groups, weight_decay=args.weight_decay)

    optimizer = build_optimizer(finetune_encoder=not currently_frozen)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Training loop
    best_val_f1 = 0.0
    best_epoch = 0
    print("\nTraining...")

    for epoch in range(1, args.epochs + 1):
        # Gradual unfreezing: unfreeze encoder after warmup epochs
        if (
            not args.freeze_encoder
            and args.finetune_after > 0
            and epoch == args.finetune_after + 1
            and currently_frozen
        ):
            print(f"\n  >>> Unfreezing encoder at epoch {epoch}")
            for p in model.parameters():
                p.requires_grad = True
            currently_frozen = False
            optimizer = build_optimizer(finetune_encoder=True)
            remaining = args.epochs - epoch + 1
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=remaining)

        t0 = time.time()
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, device,
            freeze_encoder=currently_frozen,
            max_grad_norm=args.max_grad_norm,
        )
        val_metrics = evaluate(model, val_loader, criterion, device)
        scheduler.step()
        elapsed = time.time() - t0

        print(
            f"Epoch {epoch:3d}/{args.epochs} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_metrics['loss']:.4f} | "
            f"val_acc={val_metrics['accuracy']:.4f} | "
            f"val_bal_acc={val_metrics['balanced_accuracy']:.4f} | "
            f"val_f1={val_metrics['macro_f1']:.4f} | "
            f"{elapsed:.1f}s"
        )

        if wandb_run:
            import wandb

            wandb.log({
                "epoch": epoch,
                "train/loss": train_loss,
                "val/loss": val_metrics["loss"],
                "val/accuracy": val_metrics["accuracy"],
                "val/balanced_accuracy": val_metrics["balanced_accuracy"],
                "val/macro_f1": val_metrics["macro_f1"],
                "lr": scheduler.get_last_lr()[0],
            })

        if val_metrics["macro_f1"] > best_val_f1:
            best_val_f1 = val_metrics["macro_f1"]
            best_epoch = epoch
            model.save(output_dir / "best_classifier.pt")

    print(f"\nBest val macro F1: {best_val_f1:.4f} at epoch {best_epoch}")

    # Test evaluation
    print("\nEvaluating on test set (10s windows)...")
    best_model = HARClassifier.load(output_dir / "best_classifier.pt", device=device)
    test_metrics = evaluate(best_model, test_loader, criterion, device)

    print(f"\nTest Results:")
    print(f"  Accuracy:          {test_metrics['accuracy']:.4f}")
    print(f"  Balanced Accuracy: {test_metrics['balanced_accuracy']:.4f}")
    print(f"  Macro F1:          {test_metrics['macro_f1']:.4f}")
    print(f"\n  Per-class F1:")
    for name, f1 in zip(class_names, test_metrics["per_class_f1"]):
        print(f"    {name:20s}: {f1:.4f}")

    # Save results
    results = {
        "args": vars(args),
        "best_epoch": best_epoch,
        "best_val_macro_f1": best_val_f1,
        "test_metrics": test_metrics,
        "class_names": class_names,
    }
    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    if wandb_run:
        import wandb

        wandb.run.summary["best_epoch"] = best_epoch
        wandb.run.summary["best_val_macro_f1"] = best_val_f1
        wandb.run.summary["test/accuracy"] = test_metrics["accuracy"]
        wandb.run.summary["test/balanced_accuracy"] = test_metrics["balanced_accuracy"]
        wandb.run.summary["test/macro_f1"] = test_metrics["macro_f1"]
        for name, f1 in zip(class_names, test_metrics["per_class_f1"]):
            wandb.run.summary[f"test/f1_{name}"] = f1
        wandb.finish()

    print(f"\nResults saved to {output_dir}")


if __name__ == "__main__":
    main()