#!/usr/bin/env python3
# SPDX-License-Identifier: CC-BY-NC-4.0
"""Download the ARTS per-domain classifier checkpoints from Hugging Face.

All ARTS classifier tools are bundled in a single model repo,
``nz00shuuuu/arts-rlm-classifiers``, which mirrors the local checkpoint layout.
This script fetches the checkpoints used by the ARTS evaluation entrypoints
(``src.models.ts_llm.arts.*``) and places each at the relative path those
scripts read from by default:

  capture24 -> results/classifier/dual/best_classifier.pt
  sleep     -> results/sleep_classifier/{sleep_stages,arousals}/best_classifier.pt
  ecg       -> checkpoints/ltaf/{rhythm_resnet1d,beats_htf}/best_classifier.pt
  uk_dale   -> results/uk_dale_classifier/best_classifier.pt

(The bundle also contains additional ablation variants — single-encoder
Capture24 heads and fine-tuned UK-DALE heads — which this script does not fetch.)

Usage:
    python scripts/download_classifiers.py                       # all domains
    python scripts/download_classifiers.py --domain ecg
    python scripts/download_classifiers.py --domain sleep capture24
    python scripts/download_classifiers.py --output-root my_ckpts/
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

BUNDLE_REPO = "nz00shuuuu/arts-rlm-classifiers"

# domain -> checkpoint paths. These are identical in the bundle repo and locally,
# since the bundle mirrors the layout the ARTS scripts read from.
CLASSIFIER_CHECKPOINTS: dict[str, list[str]] = {
    "capture24": [
        "results/classifier/dual/best_classifier.pt",
    ],
    "sleep": [
        "results/sleep_classifier/sleep_stages/best_classifier.pt",
        "results/sleep_classifier/arousals/best_classifier.pt",
    ],
    "ecg": [
        "checkpoints/ltaf/rhythm_resnet1d/best_classifier.pt",
        "checkpoints/ltaf/beats_htf/best_classifier.pt",
    ],
    "uk_dale": [
        "results/uk_dale_classifier/best_classifier.pt",
    ],
}

ALL_DOMAINS = list(CLASSIFIER_CHECKPOINTS)


def _download(rel_path: str, output_root: Path) -> Path:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("ERROR: huggingface_hub is required. Install with: pip install huggingface_hub")
        sys.exit(1)

    dst = output_root / rel_path
    if dst.exists():
        print(f"[skip] {dst} already exists")
        return dst

    print(f"[hf]  {BUNDLE_REPO}/{rel_path}")
    src = hf_hub_download(repo_id=BUNDLE_REPO, filename=rel_path)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dst)
    print(f"       -> {dst} ({dst.stat().st_size / 1e6:.1f} MB)")
    return dst


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--domain", nargs="+", choices=ALL_DOMAINS + ["all"], default=["all"],
        help="Which classifier domain(s) to download (default: all).",
    )
    ap.add_argument(
        "--output-root", type=str, default=".",
        help="Root the checkpoints are placed under (default: repo root, matching "
             "the paths the ARTS scripts expect).",
    )
    args = ap.parse_args()

    domains = ALL_DOMAINS if "all" in args.domain else args.domain
    output_root = Path(args.output_root)

    for domain in domains:
        for rel_path in CLASSIFIER_CHECKPOINTS[domain]:
            _download(rel_path, output_root)

    print("\nDone.")


if __name__ == "__main__":
    main()
