#!/usr/bin/env python3
# SPDX-License-Identifier: CC-BY-NC-4.0
"""Train the beat-embedding rhythm classifier on LTAF.

For each rhythm bout: locate beats in window → run frozen HTF beats
classifier per beat → small Transformer over the beat embeddings →
6-class head.

Usage:
    .venv/bin/python scripts/train_rhythm_from_beats.py \
        --htf-checkpoint results/ecg_classifier/beats_htf/best_classifier.pt \
        --epochs 30 --batch-size 32 --lr 5e-4 \
        --output-dir results/ecg_classifier/sweep/c6_rhythm_from_beats_v1
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
from src.datasets.ltaf_haystack.rhythm_classifier_dataset import (
    SPLIT_MANIFEST,
    TIMELINES_DIR,
    RHYTHM_CLASS_NAMES,
)
from src.models.classifiers.ecg.beat_htf import (
    BEAT_CLASS_NAMES,
    EcgBeatHTFClassifier,
)
from src.models.classifiers.ecg.rhythm_from_beats import (
    RhythmFromBeats,
    htf_fused_features,
)


HAYSTACK_DIR = LTAF_DATA_DIR / "ltaf_haystack"
BEAT_TIMELINES_DIR = HAYSTACK_DIR / "beat_timelines"

BEAT_WINDOW_S = 2.0
BEAT_WINDOW_SAMPLES = int(BEAT_WINDOW_S * SOURCE_HZ)  # 256
HALF = BEAT_WINDOW_SAMPLES // 2
BEAT_LABEL_TO_IDX = {n: i for i, n in enumerate(BEAT_CLASS_NAMES)}  # N=0,A=1,V=2
HISTORY_K = 5
RR_NORMAL_S = 1.0  # Reference RR for HTF; actual model uses raw seconds.


def _zscore(chunk: np.ndarray) -> np.ndarray:
    mean = chunk.mean(axis=-1, keepdims=True)
    std = chunk.std(axis=-1, keepdims=True)
    return ((chunk - mean) / (std + 1e-6)).astype(np.float32, copy=False)


class RhythmFromBeatsDataset(Dataset):
    """Yields (beat_signals, rr_history, label_history, rr_to_prev, valid_mask, label).

    For each rhythm bout, we slice all R-peaks falling inside it, extract a
    2 s window per beat (R-peak centered), build a (history_k, num_beat_classes)
    label_history + (history_k,) RR_history per beat, and a (T,) RR-to-previous
    vector for the sequence model.
    """

    def __init__(
        self,
        record_ids: List[str],
        window_seconds: float = 10.0,
        max_beats: int = 64,
        history_k: int = HISTORY_K,
        base_seed: int = 42,
        classes: List[str] | None = None,
        min_beats: int = 4,
    ):
        self.class_names = list(classes) if classes else list(RHYTHM_CLASS_NAMES)
        self.label_to_idx = {n: i for i, n in enumerate(self.class_names)}
        self.window_seconds = float(window_seconds)
        self.window_samples = int(round(window_seconds * SOURCE_HZ))
        self.max_beats = int(max_beats)
        self.history_k = int(history_k)
        self.min_beats = int(min_beats)
        self.base_seed = int(base_seed)

        # Precomputed per-record arrays for fast in-window beat lookup.
        # Each entry: (rid, beat_samples_arr, beat_label_idxs_arr,
        #              bouts_list, total_samples)
        self._record_data: List[Tuple[str, np.ndarray, np.ndarray, list, int]] = []
        for rid in tqdm(record_ids, desc="indexing rhythms+beats"):
            timeline_path = TIMELINES_DIR / f"{rid}.parquet"
            beats_path = BEAT_TIMELINES_DIR / f"{rid}.parquet"
            if not timeline_path.exists() or not beats_path.exists():
                continue
            try:
                rdf = pd.read_parquet(timeline_path)
                bdf = pd.read_parquet(beats_path, columns=["sample", "symbol"])
                total_samples = int(load_record_signals_mmap(rid).shape[0])
            except Exception as e:
                print(f"  skip {rid}: {e}")
                continue
            # Keep only N/A/V beats.
            bdf = bdf[bdf["symbol"].isin(BEAT_CLASS_NAMES)]
            bsamples = bdf["sample"].to_numpy(dtype=np.int64)
            border = np.argsort(bsamples, kind="mergesort")
            bsamples = bsamples[border]
            blabels = np.array(
                [BEAT_LABEL_TO_IDX[s] for s in bdf["symbol"].to_numpy()[border]],
                dtype=np.int64,
            )
            bouts = []
            for row in rdf.itertuples(index=False):
                lbl = row.activity
                if lbl not in self.label_to_idx:
                    continue
                bouts.append((int(row.start_sample), int(row.end_sample), lbl))
            if not bouts:
                continue
            self._record_data.append((rid, bsamples, blabels, bouts, total_samples))

        self.entries: List[Tuple[str, int, int, int]] = []
        self.set_epoch(0)

    def set_epoch(self, epoch: int) -> None:
        rng = random.Random(self.base_seed * 1_000_003 + epoch)
        entries: List[Tuple[str, int, int, int]] = []
        min_overlap = max(1, self.window_samples // 2)
        for rid, bsamples, blabels, bouts, total_samples in self._record_data:
            for start, end, lbl in bouts:
                bout_len = max(1, end - start)
                target = min(min_overlap, bout_len)
                low = max(0, start - self.window_samples + target)
                high = min(total_samples - self.window_samples, end - target)
                if high < low:
                    continue
                ws = rng.randint(low, high)
                we = ws + self.window_samples
                # Need at least HALF samples on each side for centered beats.
                # The dataset still yields these; getitem handles padding for
                # near-edge beats.
                # Quick filter: at least min_beats beats falling inside [ws, we].
                # bsamples is sorted ascending.
                lo = np.searchsorted(bsamples, ws, side="left")
                hi = np.searchsorted(bsamples, we, side="right")
                if (hi - lo) < self.min_beats:
                    continue
                entries.append((rid, ws, we, self.label_to_idx[lbl]))
        self.entries = entries

    @property
    def labels(self) -> np.ndarray:
        return np.array([e[3] for e in self.entries], dtype=np.int64)

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int):
        rid, ws, we, label = self.entries[idx]
        # Find indices of beats whose centers fall in [ws, we].
        # We use the precomputed sorted arrays.
        for rdata in self._record_data:
            if rdata[0] == rid:
                _, bsamples, blabels, _, total_samples = rdata
                break
        lo = np.searchsorted(bsamples, ws, side="left")
        hi = np.searchsorted(bsamples, we, side="right")
        if (hi - lo) > self.max_beats:
            # Subsample evenly across the bout.
            picks = np.linspace(lo, hi - 1, self.max_beats).astype(np.int64)
        else:
            picks = np.arange(lo, hi)
        n_beats = len(picks)

        # Allocate outputs.
        beat_signals = np.zeros(
            (self.max_beats, 2, BEAT_WINDOW_SAMPLES), dtype=np.float32
        )
        rr_history = np.zeros((self.max_beats, self.history_k), dtype=np.float32)
        label_history = -np.ones((self.max_beats, self.history_k), dtype=np.int64)
        rr_extra = np.zeros((self.max_beats, 1), dtype=np.float32)
        valid_mask = np.zeros((self.max_beats,), dtype=bool)

        for slot, bi in enumerate(picks):
            sample = int(bsamples[bi])
            # Per-beat 2-s window centered on R-peak.
            bs_start = sample - HALF
            bs_end = bs_start + BEAT_WINDOW_SAMPLES
            # Clamp to record bounds; pad if near edges.
            clip_start = max(0, bs_start)
            clip_end = min(total_samples, bs_end)
            try:
                chunk = load_bout_signal(rid, clip_start, clip_end)  # (L, 2)
            except Exception:
                continue
            sig = np.ascontiguousarray(chunk.T)
            sig = _zscore(sig)
            target = np.zeros((2, BEAT_WINDOW_SAMPLES), dtype=np.float32)
            offset = clip_start - bs_start
            target[:, offset:offset + sig.shape[1]] = sig
            beat_signals[slot] = target

            # History: K preceding RR intervals and beat labels.
            #   rr[0] = sample[bi]   - sample[bi-1] (immediately preceding RR)
            #   rr[1] = sample[bi-1] - sample[bi-2]
            # label[k] = preceding-by-(k+1) beat's class.
            # Matches the convention in scripts/train_ecg_beat_htf.py.
            for k in range(self.history_k):
                prev_pos = bi - 1 - k
                if prev_pos < 0:
                    break
                next_pos = prev_pos + 1
                rr_s = (int(bsamples[next_pos]) - int(bsamples[prev_pos])) / SOURCE_HZ
                if rr_s > 3.0 or rr_s <= 0:
                    rr_s = 0.0
                rr_history[slot, k] = rr_s
                label_history[slot, k] = int(blabels[prev_pos])

            # rr_to_prev (for the sequence model): RR to the previous beat in
            # this *bout slot*, in seconds, normalized.
            if slot > 0:
                prev_sample_in_bout = int(bsamples[picks[slot - 1]])
                rr_extra[slot, 0] = max(
                    0.0, min(3.0, (sample - prev_sample_in_bout) / SOURCE_HZ)
                )
            valid_mask[slot] = True

        return (
            torch.from_numpy(beat_signals),         # (T, 2, 256)
            torch.from_numpy(rr_history),           # (T, K)
            torch.from_numpy(label_history),        # (T, K)
            torch.from_numpy(rr_extra),             # (T, 1)
            torch.from_numpy(valid_mask),           # (T,)
            int(label),
        )


def _embed_beats(htf: EcgBeatHTFClassifier, signals: torch.Tensor,
                 rr_history: torch.Tensor, label_history: torch.Tensor,
                 valid_mask: torch.Tensor) -> torch.Tensor:
    """Run frozen HTF on flattened beats, reshape back to (B, T, D)."""
    B, T, C, L = signals.shape
    flat_sig = signals.reshape(B * T, C, L)
    flat_rr = rr_history.reshape(B * T, -1)
    flat_lab = label_history.reshape(B * T, -1)
    feats = htf_fused_features(htf, flat_sig, flat_rr, flat_lab)  # (B*T, D)
    D = feats.shape[-1]
    feats = feats.reshape(B, T, D)
    # Zero out invalid positions to prevent gradient leakage from random padding.
    feats = feats * valid_mask.unsqueeze(-1).float()
    return feats


def evaluate(model, htf, loader, criterion, device, num_classes):
    model.eval(); htf.eval()
    total = 0.0; n = 0
    preds, labs = [], []
    with torch.no_grad():
        for signals, rr_hist, lab_hist, rr_extra, valid_mask, y in tqdm(
            loader, desc="  eval", leave=False
        ):
            signals = signals.to(device); rr_hist = rr_hist.to(device)
            lab_hist = lab_hist.to(device); rr_extra = rr_extra.to(device)
            valid_mask = valid_mask.to(device); y = y.to(device, dtype=torch.long)
            beat_feats = _embed_beats(htf, signals, rr_hist, lab_hist, valid_mask)
            logits = model(beat_feats, rr_extra, valid_mask)
            loss = criterion(logits, y)
            total += loss.item(); n += 1
            preds.extend(logits.argmax(-1).cpu().tolist())
            labs.extend(y.cpu().tolist())
    p, l = np.array(preds), np.array(labs)
    rng = list(range(num_classes))
    return {
        "loss": total / max(n, 1),
        "accuracy": float((p == l).mean()) if len(p) else 0.0,
        "balanced_accuracy": float(balanced_accuracy_score(l, p)) if len(p) else 0.0,
        "macro_f1": float(f1_score(l, p, labels=rng, average="macro", zero_division=0)) if len(p) else 0.0,
        "per_class_f1": f1_score(l, p, labels=rng, average=None, zero_division=0).tolist() if len(p) else [],
        "confusion_matrix": confusion_matrix(l, p, labels=rng).tolist() if len(p) else [],
    }


def train_one_epoch(model, htf, loader, criterion, optimizer, device, max_grad_norm=1.0):
    model.train(); htf.eval()  # HTF stays frozen
    total = 0.0; n = 0
    for signals, rr_hist, lab_hist, rr_extra, valid_mask, y in tqdm(
        loader, desc="  train", leave=False
    ):
        signals = signals.to(device); rr_hist = rr_hist.to(device)
        lab_hist = lab_hist.to(device); rr_extra = rr_extra.to(device)
        valid_mask = valid_mask.to(device); y = y.to(device, dtype=torch.long)
        with torch.no_grad():
            beat_feats = _embed_beats(htf, signals, rr_hist, lab_hist, valid_mask)
        logits = model(beat_feats, rr_extra, valid_mask)
        loss = criterion(logits, y)
        optimizer.zero_grad(); loss.backward()
        if max_grad_norm > 0:
            nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        total += loss.item(); n += 1
    return total / max(n, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--htf-checkpoint", required=True)
    ap.add_argument("--window-seconds", type=float, default=10.0)
    ap.add_argument("--max-beats", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--weight-decay", type=float, default=5e-2)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--d-model", type=int, default=128)
    ap.add_argument("--n-layers", type=int, default=4)
    ap.add_argument("--n-heads", type=int, default=4)
    ap.add_argument("--head-hidden", type=int, default=128)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--label-smoothing", type=float, default=0.1)
    ap.add_argument("--class-weight-power", type=float, default=0.5)
    ap.add_argument("--class-weight-cap", type=float, default=10.0)
    ap.add_argument("--max-grad-norm", type=float, default=1.0)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--use-val-as-train", action="store_true")
    ap.add_argument("--classes", type=str, nargs="+",
                    default=["NSR", "AFIB", "SBR", "AB", "SVTA", "B"])
    ap.add_argument("--no-eval-test", action="store_true")
    args = ap.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed); random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    output_dir = Path(args.output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Device: {device}, output: {output_dir}")

    with open(SPLIT_MANIFEST) as f:
        split = json.load(f)
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
    num_classes = len(classes)

    print("\nLoading frozen HTF beats classifier from", args.htf_checkpoint)
    htf = EcgBeatHTFClassifier.load(args.htf_checkpoint, device=device)
    for p in htf.parameters(): p.requires_grad = False
    htf.eval()
    print(f"  HTF params: {sum(p.numel() for p in htf.parameters()):,} (frozen)")

    print("\nBuilding train...")
    train_ds = RhythmFromBeatsDataset(
        train_records, window_seconds=args.window_seconds,
        max_beats=args.max_beats, base_seed=args.seed, classes=classes,
    )
    print(f"  Train bouts: {len(train_ds)}")
    print("Building val...")
    val_ds = RhythmFromBeatsDataset(
        val_records, window_seconds=args.window_seconds,
        max_beats=args.max_beats, base_seed=args.seed + 1, classes=classes,
    )
    print(f"  Val bouts: {len(val_ds)}")

    label_counts = Counter(train_ds.labels.tolist())
    total = sum(label_counts.values())
    raw = [total / (num_classes * max(1, label_counts.get(i, 0))) for i in range(num_classes)]
    dampened = [w ** args.class_weight_power for w in raw]
    final = [min(args.class_weight_cap, w) for w in dampened]
    class_weights = torch.tensor(final, dtype=torch.float32).to(device)
    print(f"  Class counts: {dict(zip(classes, [label_counts.get(i,0) for i in range(num_classes)]))}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)

    model = RhythmFromBeats(
        num_classes=num_classes, class_names=classes,
        beat_feat_dim=576, rr_extra_dim=1,
        d_model=args.d_model, n_layers=args.n_layers, n_heads=args.n_heads,
        max_beats=args.max_beats, dropout=args.dropout,
        head_hidden=args.head_hidden,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nRhythmFromBeats params: {n_params:,}")

    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=args.label_smoothing)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val_f1 = -1.0
    best_epoch = 0
    history = []
    print("\nTraining (HTF frozen)...")
    for epoch in range(1, args.epochs + 1):
        train_ds.set_epoch(epoch)
        t0 = time.time()
        train_loss = train_one_epoch(model, htf, train_loader, criterion, optimizer, device,
                                     max_grad_norm=args.max_grad_norm)
        val = evaluate(model, htf, val_loader, criterion, device, num_classes)
        scheduler.step()
        elapsed = time.time() - t0
        history.append({"epoch": epoch, "train_loss": train_loss, "val": val, "elapsed_s": elapsed})
        print(f"Epoch {epoch:3d}/{args.epochs} | train_loss={train_loss:.4f} | "
              f"val_acc={val['accuracy']:.4f} | val_bal={val['balanced_accuracy']:.4f} | "
              f"val_f1={val['macro_f1']:.4f} | per-class={[round(x,2) for x in val['per_class_f1']]} | "
              f"{elapsed:.1f}s", flush=True)
        if val["macro_f1"] > best_val_f1:
            best_val_f1 = val["macro_f1"]; best_epoch = epoch
            model.save(output_dir / "best_classifier.pt")

    print(f"\nBest val macro F1: {best_val_f1:.4f} at epoch {best_epoch}")

    test = None
    if not args.no_eval_test:
        print("\nBuilding test...")
        test_ds = RhythmFromBeatsDataset(
            test_records, window_seconds=args.window_seconds,
            max_beats=args.max_beats, base_seed=args.seed + 2, classes=classes,
        )
        print(f"  Test bouts: {len(test_ds)}")
        test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                                 num_workers=args.num_workers, pin_memory=True)
        best = RhythmFromBeats.load(output_dir / "best_classifier.pt", device=device)
        test = evaluate(best, htf, test_loader, criterion, device, num_classes)
        print(f"\nTest Results:")
        print(f"  Accuracy:          {test['accuracy']:.4f}")
        print(f"  Balanced Accuracy: {test['balanced_accuracy']:.4f}")
        print(f"  Macro F1:          {test['macro_f1']:.4f}")
        for name, f1 in zip(classes, test["per_class_f1"]):
            print(f"    {name:6s}: F1={f1:.4f}")

    results = {
        "args": vars(args), "best_epoch": best_epoch,
        "best_val_macro_f1": best_val_f1,
        "test_metrics": test, "history": history,
        "class_names": classes, "n_params": n_params,
    }
    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults -> {output_dir}/results.json")


if __name__ == "__main__":
    main()
