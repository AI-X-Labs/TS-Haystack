#!/usr/bin/env python3
"""TS-Haystack TSLM training entry point.

Usage:
    python main.py --config configs/paper/flamingo_capture24_haystack_cot.yaml
    python main.py --config configs/paper/flamingo_capture24_haystack_cot.yaml --no-wandb
"""

import argparse
import sys


def main():
    parser = argparse.ArgumentParser(description="TS-Haystack training entry point")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--no-wandb", action="store_true")
    args = parser.parse_args()

    from scripts.train import main as train_main
    # Reconstruct sys.argv for train.py's argparse
    sys.argv = ["main.py", "--config", args.config]
    if args.no_wandb:
        sys.argv.append("--no-wandb")
    train_main()


if __name__ == "__main__":
    main()
