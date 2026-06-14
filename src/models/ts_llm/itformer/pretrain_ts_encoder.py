#!/usr/bin/env python3
"""Pretrain ITFormer's time series encoder with MAE-style masked reconstruction.

This is a standalone script — it does NOT use BaseModel, QADataset, or the SFT
training loop. It trains only the encoder on raw time series data from the
EngineMT-QA H5 file.

The saved checkpoint can be loaded by ITFormerModel via:
    config.model.extra_kwargs.ts_encoder_checkpoint: path/to/encoder.pt

Usage:
    python -m src.models.ts_llm.itformer.pretrain_ts_encoder \
        --data path/to/time_series_data.h5 \
        --output_dir results/pretrain_encoder \
        --epochs 10 --batch_size 12 --lr 1e-5
"""

import argparse
import time
from pathlib import Path

import h5py
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


class H5TimeSeriesDataset(Dataset):
    """Simple dataset that loads all time series from an H5 file into memory."""

    def __init__(self, h5_path: str):
        with h5py.File(h5_path, "r") as f:
            self.data = torch.tensor(f["seq_data"][:], dtype=torch.float32)
        print(f"Loaded {len(self.data)} samples, shape: {self.data.shape}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]  # (L, V)


def main():
    parser = argparse.ArgumentParser(description="Pretrain ITFormer TS encoder (MAE)")
    parser.add_argument("--data", type=str, required=True,
                        help="Path to time_series_data.h5")
    parser.add_argument("--output_dir", type=str, default="results/pretrain_encoder")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=12)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--n_heads", type=int, default=8)
    parser.add_argument("--e_layers", type=int, default=4)
    parser.add_argument("--patch_len", type=int, default=60)
    parser.add_argument("--stride", type=int, default=60)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--min_mask_ratio", type=float, default=0.7)
    parser.add_argument("--max_mask_ratio", type=float, default=0.8)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--save_every", type=int, default=100,
                        help="Save checkpoint every N steps (0 = end only)")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Dataset ---
    dataset = H5TimeSeriesDataset(args.data)
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True,
    )

    # --- Encoder in pretrain mode ---
    from src.models.ts_encoder.itformer_encoder import ITFormerTSEncoder

    encoder = ITFormerTSEncoder(
        output_dim=args.d_model,
        dropout=args.dropout,
        n_heads=args.n_heads,
        e_layers=args.e_layers,
        patch_len=args.patch_len,
        stride=args.stride,
        pretrain=True,
        min_mask_ratio=args.min_mask_ratio,
        max_mask_ratio=args.max_mask_ratio,
    ).to(device)

    total_params = sum(p.numel() for p in encoder.parameters())
    print(f"Encoder parameters: {total_params:,}")

    # --- Optimizer ---
    optimizer = AdamW(encoder.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # --- Training ---
    global_step = 0
    best_loss = float("inf")
    start = time.time()

    print(f"\nPretraining encoder for {args.epochs} epochs on {device}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Output: {output_dir}")
    print()

    for epoch in range(1, args.epochs + 1):
        encoder.train()
        epoch_loss = 0.0
        n_batches = 0

        pbar = tqdm(loader, desc=f"Epoch {epoch}/{args.epochs}")
        for batch in pbar:
            batch = batch.to(device)  # (B, L, V)
            optimizer.zero_grad()

            result = encoder(batch)  # dict with 'loss' and 'logits'
            loss = result["loss"]
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1
            global_step += 1

            pbar.set_postfix(loss=f"{epoch_loss / n_batches:.6f}")

            if args.save_every > 0 and global_step % args.save_every == 0:
                torch.save(encoder.state_dict(), output_dir / f"encoder_step_{global_step}.pt")

        avg_loss = epoch_loss / max(n_batches, 1)
        print(f"Epoch {epoch}: avg_loss={avg_loss:.6f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(encoder.state_dict(), output_dir / "encoder_best.pt")
            print(f"  Saved best encoder (loss={best_loss:.6f})")

    # Save final
    torch.save(encoder.state_dict(), output_dir / "encoder_final.pt")
    elapsed = time.time() - start
    print(f"\nDone in {elapsed / 60:.1f}m. Best loss: {best_loss:.6f}")
    print(f"Checkpoints at: {output_dir}")


if __name__ == "__main__":
    main()
