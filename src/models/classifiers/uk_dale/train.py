#!/usr/bin/env python3
# SPDX-License-Identifier: CC-BY-NC-4.0
"""Train the UK-DALE appliance classifier.

Reads the per-week split from
``data/uk_dale/uk_dale_haystack/split_manifest.json`` and the bout-level
ground truth from ``bout_index.parquet``. Each training step samples a random
``window_samples``-long crop from a random (house, iso_week) tuple in the
split, calls ``load_meter_window_grid`` to pull mains active power onto a
regular 6 s grid, and computes per-sample multi-label appliance vectors from
overlapping bouts.

Usage:
    .venv/bin/python3 scripts/train_uk_dale_classifier.py \\
        --epochs 30 --steps-per-epoch 1000 --batch-size 32

The output is a single checkpoint at
``results/uk_dale_classifier/best_classifier.pt`` plus a ``results.json``
containing per-class F1 on the held-out test weeks.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, IterableDataset
from tqdm.auto import tqdm

from src.datasets.uk_dale_haystack.core.activity_regimes import V1_VOCAB
from src.datasets.uk_dale_haystack.loader import (
    NOMINAL_DT_S,
    load_meter_window_grid,
    mains_meter_id,
)
from src.models.classifiers.uk_dale.model import (
    UK_DALE_CLASS_NAMES,
    UKDaleClassifier,
    featurize_power,
)


SPLIT_MANIFEST = Path("data/uk_dale/uk_dale_haystack/split_manifest.json")
BOUT_INDEX = Path("data/uk_dale/uk_dale_haystack/bout_index.parquet")


# ---------------------------------------------------------------------------
# Time helpers
# ---------------------------------------------------------------------------


def iso_week_bounds_ns(week_str: str) -> tuple[int, int]:
    """Return (start_ns, end_ns) for an ISO week 'YYYY-WNN' (Mon 00:00 to Mon 00:00 UTC)."""
    year, w = week_str.split("-W")
    start = datetime.fromisocalendar(int(year), int(w), 1)
    end = start + timedelta(days=7)
    return int(start.timestamp() * 1e9), int(end.timestamp() * 1e9)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class UKDaleWindowDataset(IterableDataset):
    """Random crops from (house, iso_week) tuples in a given split.

    Each item yields ``(power, labels)`` where
        power:  (L,) float32 raw watts (caller featurizes)
        labels: (10, L) float32 multi-label per-sample appliance vector

    Sampling strategy (training mode):
      With probability ``stratify_prob`` the crop is anchored on a randomly
      chosen rare-class bout — drop the bout's start somewhere inside the
      crop, biased toward the centre. This guarantees gradient signal for
      classes like oven / toaster / kettle that natural random crops would
      miss almost entirely.

      Otherwise, fall back to a uniform random offset in a uniformly
      sampled (house, week) — captures the natural background distribution
      and keeps the negative rate honest.

    Validation/test mode (``deterministic=True``) tiles each week into
    contiguous non-overlapping crops in a fixed order so metrics are
    comparable across runs.
    """

    # Rare classes get oversampled via the stratified path. fridge_freezer
    # and washer_dryer are excluded — they're already abundant.
    RARE_CLASSES = (
        "dishwasher", "fridge", "hair_dryer", "kettle",
        "microwave", "oven", "toaster", "washing_machine",
    )

    def __init__(
        self,
        weeks: list[tuple[int, str]],
        bouts_by_house_week: dict[tuple[int, str], list[tuple[int, int, int]]],
        window_samples: int,
        steps_per_epoch: int,
        dt_s: float = NOMINAL_DT_S,
        seed: int = 0,
        deterministic: bool = False,
        stratify_prob: float = 0.6,
    ):
        super().__init__()
        self.weeks = list(weeks)
        self.bouts = bouts_by_house_week
        self.window_samples = int(window_samples)
        self.steps_per_epoch = int(steps_per_epoch)
        self.dt_s = float(dt_s)
        self.dt_ns = int(round(dt_s * 1e9))
        self.seed = int(seed)
        self.deterministic = bool(deterministic)
        self.stratify_prob = float(stratify_prob)
        self.window_dur_ns = self.window_samples * self.dt_ns

        # Index (class_idx -> [(house, week, b_start_ns, b_end_ns), ...])
        # for stratified sampling.
        self._bouts_by_class: dict[int, list[tuple[int, str, int, int]]] = {}
        rare_idxs = [APP_TO_IDX[a] for a in self.RARE_CLASSES]
        for (house, week), bouts in self.bouts.items():
            for app_idx, b_start, b_end in bouts:
                if app_idx in rare_idxs:
                    self._bouts_by_class.setdefault(app_idx, []).append(
                        (house, week, b_start, b_end)
                    )

    def _build_labels(
        self,
        house: int,
        week: str,
        grid_start_ns: int,
    ) -> np.ndarray:
        L = self.window_samples
        labels = np.zeros((NUM_CLASSES, L), dtype=np.float32)
        bouts = self.bouts.get((house, week), [])
        if not bouts:
            return labels
        sample_t = grid_start_ns + np.arange(L, dtype=np.int64) * self.dt_ns
        for app_idx, b_start, b_end in bouts:
            if b_end < grid_start_ns or b_start >= grid_start_ns + L * self.dt_ns:
                continue
            mask = (sample_t >= b_start) & (sample_t < b_end)
            labels[app_idx, mask] = 1.0
        return labels

    def _load_crop_signal(
        self, house: int, week: str, grid_start: int,
    ) -> np.ndarray | None:
        grid_end = grid_start + self.window_dur_ns
        try:
            power = load_meter_window_grid(
                house, mains_meter_id(house),
                grid_start, grid_end, dt_s=self.dt_s,
            )
        except Exception:
            return None
        if power.shape[0] != self.window_samples:
            pad = self.window_samples - power.shape[0]
            if pad > 0:
                power = np.concatenate([power, np.zeros(pad, dtype=power.dtype)])
            else:
                power = power[:self.window_samples]
        # Skip windows that fall in a mains-data gap. ``load_meter_window_grid``
        # zeros out grid points whose nearest sample is stale; in real
        # operation mains baseline is always > 50 W, so >50% zeros means we
        # landed in a gap and the bout-index labels are unreliable here.
        if (power < 1.0).mean() > 0.5:
            return None
        return power

    def _sample_uniform(self, rng: random.Random) -> tuple[int, str, int] | None:
        house, week = rng.choice(self.weeks)
        week_start, week_end = iso_week_bounds_ns(week)
        max_start = week_end - self.window_dur_ns
        if max_start <= week_start:
            return None
        grid_start = rng.randrange(week_start, max_start)
        grid_start = grid_start - (grid_start % self.dt_ns)
        return house, week, grid_start

    def _sample_anchored(self, rng: random.Random) -> tuple[int, str, int] | None:
        # Pick a rare class uniformly; pick a bout for that class uniformly.
        # This gives equal weight to each rare class regardless of total bout
        # count, which is what we want — kettle (~1k bouts) and oven (~50)
        # both end up well-sampled.
        classes = list(self._bouts_by_class.keys())
        if not classes:
            return self._sample_uniform(rng)
        c = rng.choice(classes)
        house, week, b_start, b_end = rng.choice(self._bouts_by_class[c])
        # Random offset of the bout start within the crop, biased toward the
        # centre but with margin so the bout's start isn't always at the edge.
        offset_in_crop = rng.randint(self.window_dur_ns // 6,
                                     5 * self.window_dur_ns // 6)
        grid_start = b_start - offset_in_crop
        grid_start = grid_start - (grid_start % self.dt_ns)
        # Clamp to the week bounds.
        week_start, week_end = iso_week_bounds_ns(week)
        grid_start = max(week_start, min(grid_start, week_end - self.window_dur_ns))
        return house, week, grid_start

    def _sample_crop(self, rng: random.Random) -> tuple[np.ndarray, np.ndarray]:
        for _ in range(32):  # bounded retry on signal-loading failure / data gaps
            choice = (
                self._sample_anchored(rng)
                if (rng.random() < self.stratify_prob and self._bouts_by_class)
                else self._sample_uniform(rng)
            )
            if choice is None:
                continue
            house, week, grid_start = choice
            power = self._load_crop_signal(house, week, grid_start)
            if power is None:
                continue
            labels = self._build_labels(house, week, grid_start)
            return power, labels
        # Fallback: zeros if we couldn't sample (very rare)
        return (np.zeros(self.window_samples, dtype=np.float32),
                np.zeros((NUM_CLASSES, self.window_samples), dtype=np.float32))

    def __iter__(self):
        worker = torch.utils.data.get_worker_info()
        worker_id = worker.id if worker is not None else 0
        if self.deterministic:
            # Tile each week into N = window_dur slices; deterministic across workers
            for i, (house, week) in enumerate(self.weeks):
                if i % max(1, (worker.num_workers if worker else 1)) != worker_id:
                    continue
                week_start, week_end = iso_week_bounds_ns(week)
                week_start = week_start + (-week_start % self.dt_ns)
                n_crops = (week_end - week_start) // self.window_dur_ns
                # Cap to keep eval cost reasonable: 64 crops per week => ~1.5 days
                n_crops = min(int(n_crops), 64)
                for k in range(n_crops):
                    grid_start = week_start + k * self.window_dur_ns
                    grid_end = grid_start + self.window_dur_ns
                    try:
                        power = load_meter_window_grid(
                            house, mains_meter_id(house),
                            grid_start, grid_end, dt_s=self.dt_s,
                        )
                    except Exception:
                        continue
                    if power.shape[0] != self.window_samples:
                        pad = self.window_samples - power.shape[0]
                        if pad > 0:
                            power = np.concatenate(
                                [power, np.zeros(pad, dtype=power.dtype)]
                            )
                        else:
                            power = power[:self.window_samples]
                    # Skip mains-data gaps (see _sample_crop note).
                    if (power < 1.0).mean() > 0.5:
                        continue
                    labels = self._build_labels(house, week, grid_start)
                    yield power.astype(np.float32), labels
        else:
            rng = random.Random(self.seed + worker_id * 1000003)
            for _ in range(self.steps_per_epoch):
                power, labels = self._sample_crop(rng)
                yield power.astype(np.float32), labels


def collate_fn(batch):
    powers = np.stack([b[0] for b in batch], axis=0)  # (B, L)
    labels = np.stack([b[1] for b in batch], axis=0)  # (B, 10, L)
    feats = featurize_power(powers)  # (B, 2, L)
    x = torch.from_numpy(feats).float()
    y = torch.from_numpy(labels).float()
    return x, y


# ---------------------------------------------------------------------------
# Bout index loading
# ---------------------------------------------------------------------------


NUM_CLASSES = len(UK_DALE_CLASS_NAMES)
APP_TO_IDX = {a: i for i, a in enumerate(UK_DALE_CLASS_NAMES)}


def build_bouts_lookup(
    bout_index_path: Path,
    weeks_filter: set[tuple[int, str]] | None = None,
) -> dict[tuple[int, str], list[tuple[int, int, int]]]:
    """Returns {(house, iso_week): [(app_idx, start_ns, end_ns), ...]}."""
    import pandas as pd

    df = pd.read_parquet(bout_index_path)
    if weeks_filter is not None:
        keys = [(int(h), str(w)) for h, w in weeks_filter]
        df = df[df.apply(lambda r: (int(r.house_id), str(r.iso_week)) in set(keys), axis=1)]
    out: dict[tuple[int, str], list[tuple[int, int, int]]] = {}
    for h, w, app, s_ns, e_ns in zip(
        df.house_id.to_numpy(),
        df.iso_week.to_numpy(),
        df.appliance.to_numpy(),
        df.start_ns.to_numpy(),
        df.end_ns.to_numpy(),
    ):
        app_idx = APP_TO_IDX.get(str(app))
        if app_idx is None:
            continue
        out.setdefault((int(h), str(w)), []).append(
            (app_idx, int(s_ns), int(e_ns))
        )
    return out


# ---------------------------------------------------------------------------
# Training / eval loops
# ---------------------------------------------------------------------------


def compute_pos_weight(
    weeks: list[tuple[int, str]],
    bouts: dict[tuple[int, str], list[tuple[int, int, int]]],
    cap: float = 100.0,
) -> torch.Tensor:
    """Estimate per-class positive fraction directly from bouts vs. total
    week duration. This avoids burning samples just to compute pos_weight.
    """
    total_dur_ns = 0
    per_class_dur_ns = np.zeros(NUM_CLASSES, dtype=np.int64)
    for house, week in weeks:
        ws, we = iso_week_bounds_ns(week)
        total_dur_ns += we - ws
        for app_idx, b_start, b_end in bouts.get((house, week), []):
            per_class_dur_ns[app_idx] += max(0, min(b_end, we) - max(b_start, ws))
    pos_frac = np.clip(per_class_dur_ns / max(1, total_dur_ns), 1e-5, 0.99)
    pos_weight = (1.0 - pos_frac) / pos_frac
    pos_weight = np.clip(pos_weight, 1.0, cap)
    return torch.tensor(pos_weight, dtype=torch.float32)


def train_one_epoch(model, loader, criterion, optimizer, device, max_grad_norm=1.0):
    model.train()
    total_loss = 0.0
    n_batches = 0
    for x, y in tqdm(loader, desc="  train", leave=False):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x)  # (B, 10, L)
        loss = criterion(logits, y)
        optimizer.zero_grad()
        loss.backward()
        if max_grad_norm > 0:
            nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        total_loss += loss.item()
        n_batches += 1
    return total_loss / max(n_batches, 1)


@torch.no_grad()
def evaluate(model, loader, criterion, device, threshold: float = 0.5):
    model.eval()
    total_loss = 0.0
    n_batches = 0
    # Aggregate counts per class
    tp = np.zeros(NUM_CLASSES, dtype=np.int64)
    fp = np.zeros(NUM_CLASSES, dtype=np.int64)
    fn = np.zeros(NUM_CLASSES, dtype=np.int64)
    n_pos = np.zeros(NUM_CLASSES, dtype=np.int64)
    n_total = 0
    for x, y in tqdm(loader, desc="  eval", leave=False):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x)
        loss = criterion(logits, y)
        total_loss += loss.item()
        n_batches += 1
        probs = torch.sigmoid(logits)
        preds = (probs > threshold).cpu().numpy()  # (B, 10, L)
        gt = y.cpu().numpy().astype(bool)
        n_total += gt.shape[0] * gt.shape[2]
        for c in range(NUM_CLASSES):
            p = preds[:, c, :]
            g = gt[:, c, :]
            tp[c] += int((p & g).sum())
            fp[c] += int((p & ~g).sum())
            fn[c] += int((~p & g).sum())
            n_pos[c] += int(g.sum())
    f1 = np.zeros(NUM_CLASSES, dtype=np.float64)
    prec = np.zeros(NUM_CLASSES, dtype=np.float64)
    rec = np.zeros(NUM_CLASSES, dtype=np.float64)
    for c in range(NUM_CLASSES):
        prec[c] = tp[c] / max(1, tp[c] + fp[c])
        rec[c] = tp[c] / max(1, tp[c] + fn[c])
        f1[c] = 2 * prec[c] * rec[c] / max(1e-9, prec[c] + rec[c])
    # Macro F1 over classes that actually appear in the eval set.
    present = n_pos > 0
    macro_f1 = float(f1[present].mean()) if present.any() else 0.0
    return {
        "loss": total_loss / max(n_batches, 1),
        "macro_f1": macro_f1,
        "per_class_f1": {UK_DALE_CLASS_NAMES[c]: float(f1[c]) for c in range(NUM_CLASSES)},
        "per_class_precision": {UK_DALE_CLASS_NAMES[c]: float(prec[c]) for c in range(NUM_CLASSES)},
        "per_class_recall": {UK_DALE_CLASS_NAMES[c]: float(rec[c]) for c in range(NUM_CLASSES)},
        "per_class_support": {UK_DALE_CLASS_NAMES[c]: int(n_pos[c]) for c in range(NUM_CLASSES)},
        "n_samples": n_total,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--steps-per-epoch", type=int, default=1000)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--window-samples", type=int, default=512,
                    help="Crop length in samples (1 sample = dt_s seconds)")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output-dir", type=str, default="results/uk_dale_classifier")
    ap.add_argument("--max-grad-norm", type=float, default=1.0)
    ap.add_argument("--pos-weight-cap", type=float, default=50.0)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--max-train-weeks", type=int, default=None)
    ap.add_argument("--wandb", action="store_true", default=False)
    ap.add_argument("--wandb-project", type=str, default="ts-haystack-uk-dale")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(f"UK-DALE classifier — window_samples={args.window_samples}")
    print(f"Device: {device}")
    print("=" * 60)

    # ---- splits ----
    split = json.load(open(SPLIT_MANIFEST))
    train_weeks = [(int(h), str(w)) for h, w in split["train"]]
    val_weeks = [(int(h), str(w)) for h, w in split.get("validation", [])]
    test_weeks = [(int(h), str(w)) for h, w in split.get("test", [])]
    if args.max_train_weeks:
        train_weeks = train_weeks[: args.max_train_weeks]
    print(f"Weeks: train={len(train_weeks)} val={len(val_weeks)} test={len(test_weeks)}")

    # ---- bouts ----
    print("Loading bout_index.parquet ...")
    all_weeks = set(train_weeks) | set(val_weeks) | set(test_weeks)
    bouts_lookup = build_bouts_lookup(BOUT_INDEX, weeks_filter=all_weeks)
    n_bouts = sum(len(v) for v in bouts_lookup.values())
    print(f"  {n_bouts} bouts indexed across {len(bouts_lookup)} (house, week) tuples")

    # ---- pos_weight from train weeks ----
    pos_weight = compute_pos_weight(train_weeks, bouts_lookup, cap=args.pos_weight_cap)
    print("Per-class pos_weight:", {
        UK_DALE_CLASS_NAMES[i]: round(float(pos_weight[i]), 2)
        for i in range(NUM_CLASSES)
    })

    # ---- datasets ----
    train_ds = UKDaleWindowDataset(
        train_weeks, bouts_lookup,
        window_samples=args.window_samples,
        steps_per_epoch=args.steps_per_epoch * args.batch_size,
        seed=args.seed, deterministic=False,
    )
    val_ds = UKDaleWindowDataset(
        val_weeks, bouts_lookup,
        window_samples=args.window_samples,
        steps_per_epoch=0,  # unused in deterministic mode
        seed=args.seed, deterministic=True,
    )
    test_ds = UKDaleWindowDataset(
        test_weeks, bouts_lookup,
        window_samples=args.window_samples,
        steps_per_epoch=0,
        seed=args.seed, deterministic=True,
    )

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size,
        num_workers=args.num_workers, pin_memory=True,
        collate_fn=collate_fn, persistent_workers=args.num_workers > 0,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size,
        num_workers=min(2, args.num_workers), pin_memory=True,
        collate_fn=collate_fn,
    )
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size,
        num_workers=min(2, args.num_workers), pin_memory=True,
        collate_fn=collate_fn,
    )

    # ---- model ----
    model = UKDaleClassifier().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {n_params:,}")

    # pos_weight is shape (10,); reshape to (10, 1) so it broadcasts across L
    # against logits of shape (B, 10, L) without colliding with the time axis.
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.view(NUM_CLASSES, 1).to(device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    wandb_run = None
    if args.wandb:
        try:
            import wandb
            wandb_run = wandb.init(
                project=args.wandb_project,
                name=f"uk_dale_classifier_w{args.window_samples}",
                config=vars(args),
            )
        except Exception as e:
            print(f"  W&B init failed: {e}")

    best_val_f1 = -1.0
    best_epoch = 0
    print("\nTraining...")
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, device,
            max_grad_norm=args.max_grad_norm,
        )
        val = evaluate(model, val_loader, criterion, device, threshold=args.threshold)
        scheduler.step()
        elapsed = time.time() - t0
        print(
            f"Epoch {epoch:3d}/{args.epochs} | "
            f"train_loss={train_loss:.4f} | val_loss={val['loss']:.4f} | "
            f"val_f1={val['macro_f1']:.4f} | {elapsed:.1f}s"
        )
        f1_str = "  ".join(
            f"{a[:6]}={val['per_class_f1'][a]:.2f}"
            for a in UK_DALE_CLASS_NAMES
        )
        print(f"      per-class F1: {f1_str}")
        if wandb_run:
            import wandb
            log = {
                "epoch": epoch,
                "train/loss": train_loss,
                "val/loss": val["loss"],
                "val/macro_f1": val["macro_f1"],
                "lr": scheduler.get_last_lr()[0],
            }
            for a in UK_DALE_CLASS_NAMES:
                log[f"val/f1_{a}"] = val["per_class_f1"][a]
                log[f"val/precision_{a}"] = val["per_class_precision"][a]
                log[f"val/recall_{a}"] = val["per_class_recall"][a]
            wandb.log(log)
        if val["macro_f1"] > best_val_f1:
            best_val_f1 = val["macro_f1"]
            best_epoch = epoch
            model.save(output_dir / "best_classifier.pt")

    print(f"\nBest val macro F1: {best_val_f1:.4f} at epoch {best_epoch}")

    # ---- test eval ----
    print("\nEvaluating best checkpoint on test weeks...")
    best = UKDaleClassifier.load(output_dir / "best_classifier.pt", device=device)
    test = evaluate(best, test_loader, criterion, device, threshold=args.threshold)
    print(f"Test macro F1: {test['macro_f1']:.4f}")
    print("  Per-class F1:")
    for a in UK_DALE_CLASS_NAMES:
        print(f"    {a:18s} f1={test['per_class_f1'][a]:.4f}  "
              f"prec={test['per_class_precision'][a]:.4f}  "
              f"rec={test['per_class_recall'][a]:.4f}  "
              f"support={test['per_class_support'][a]}")

    results = {
        "args": vars(args),
        "best_epoch": best_epoch,
        "best_val_macro_f1": best_val_f1,
        "test_metrics": test,
        "class_names": UK_DALE_CLASS_NAMES,
    }
    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_dir}")

    if wandb_run:
        import wandb
        wandb.run.summary["best_epoch"] = best_epoch
        wandb.run.summary["best_val_macro_f1"] = best_val_f1
        wandb.run.summary["test/macro_f1"] = test["macro_f1"]
        for a in UK_DALE_CLASS_NAMES:
            wandb.run.summary[f"test/f1_{a}"] = test["per_class_f1"][a]
        wandb.finish()


if __name__ == "__main__":
    main()
