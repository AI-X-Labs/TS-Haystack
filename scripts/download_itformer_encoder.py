#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Anonymous Authors
# SPDX-License-Identifier: CC-BY-NC-4.0
"""
Download pretrained ITFormer time-series encoder from the ITFormer-3B HuggingFace checkpoint.

Downloads only shard 3 (which contains all TS encoder weights), extracts the
``ts_encoder.*`` keys, remaps them to TSLM-Bench naming conventions, and saves
a standalone encoder checkpoint.

Source:  pandalin98/ITFormer-3B  (Qwen2.5-3B backbone, encoder frozen during SFT)
Output:  checkpoints/itformer_ts_encoder_3b.pt

Usage:
    python scripts/download_itformer_encoder.py
    python scripts/download_itformer_encoder.py --output checkpoints/my_encoder.pt
    python scripts/download_itformer_encoder.py --keep-shard   # don't delete the 2.7 GB shard
"""

import argparse
import os
import sys
from pathlib import Path

HF_REPO = "pandalin98/ITFormer-3B"
SHARD_FILE = "model-00003-of-00003.safetensors"
ENCODER_PREFIX = "ts_encoder."

# Key remapping: original checkpoint → TSLM-Bench ITFormerTSEncoder
KEY_REMAP = [
    (".var_att_block.norm1.", ".var_att.norm."),
    (".var_att_block.attn_var.", ".var_att.attn."),
    (".seq_att_block.norm1.", ".seq_att.norm."),
    (".seq_att_block.attn_seq.", ".seq_att.attn."),
    (".feed_forward.", ".ff."),
]


def remap_key(key: str) -> str:
    """Remap a single state-dict key from original to TSLM-Bench naming."""
    for old, new in KEY_REMAP:
        key = key.replace(old, new)
    return key


def main():
    parser = argparse.ArgumentParser(
        description="Download pretrained ITFormer TS encoder from HuggingFace."
    )
    parser.add_argument(
        "--output",
        type=str,
        default="checkpoints/itformer_ts_encoder_3b.pt",
        help="Output path for the extracted encoder checkpoint (default: checkpoints/itformer_ts_encoder_3b.pt)",
    )
    parser.add_argument(
        "--keep-shard",
        action="store_true",
        help="Keep the downloaded shard file (2.7 GB) after extraction",
    )
    args = parser.parse_args()

    # Resolve paths relative to project root
    project_root = Path(__file__).resolve().parent.parent
    output_path = project_root / args.output

    # Check if output already exists
    if output_path.exists():
        print(f"Output already exists: {output_path}")
        print("Use a different --output path or delete the existing file.")
        sys.exit(0)

    # --- Step 1: Download shard 3 ---
    print(f"Downloading {SHARD_FILE} from {HF_REPO}...")
    print("(This is ~2.7 GB and contains all TS encoder weights)")

    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("ERROR: huggingface_hub is required. Install with: pip install huggingface_hub")
        sys.exit(1)

    shard_path = hf_hub_download(
        repo_id=HF_REPO,
        filename=SHARD_FILE,
    )
    print(f"Downloaded to: {shard_path}")

    # --- Step 2: Load and extract encoder weights ---
    print("Extracting TS encoder weights...")

    try:
        from safetensors.torch import load_file
    except ImportError:
        print("ERROR: safetensors is required. Install with: pip install safetensors")
        sys.exit(1)

    import torch

    full_state = load_file(shard_path)

    # Filter to ts_encoder.* keys and strip prefix
    encoder_state = {}
    for key, value in full_state.items():
        if key.startswith(ENCODER_PREFIX):
            clean_key = key[len(ENCODER_PREFIX):]
            remapped_key = remap_key(clean_key)
            encoder_state[remapped_key] = value

    if not encoder_state:
        print(f"ERROR: No keys with prefix '{ENCODER_PREFIX}' found in {SHARD_FILE}")
        sys.exit(1)

    print(f"Extracted {len(encoder_state)} encoder parameters")

    # --- Step 3: Save ---
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(encoder_state, output_path)
    print(f"Saved to: {output_path}")

    # --- Step 4: Summary ---
    total_params = sum(v.numel() for v in encoder_state.values())
    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"\nEncoder summary:")
    print(f"  Parameters: {total_params:,}")
    print(f"  File size:  {size_mb:.1f} MB")
    print(f"  Keys: {sorted(encoder_state.keys())[:5]} ... ({len(encoder_state)} total)")

    print(f"\nTo use in a config:")
    print(f"  model:")
    print(f"    extra_kwargs:")
    print(f"      ts_encoder_checkpoint: {args.output}")


if __name__ == "__main__":
    main()
