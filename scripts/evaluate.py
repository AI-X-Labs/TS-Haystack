#!/usr/bin/env python3
"""
Evaluate a trained TS-LLM on any registered dataset.

Uses the same YAML configs as training. Point to a checkpoint via --checkpoint
(local path) or via backbone.hf_checkpoint_repo in the config YAML.

Usage:
    # Evaluate with a local checkpoint
    python scripts/evaluate.py \
        --config configs/experiments/capture24_haystack_llama.yaml \
        --checkpoint results/capture24_haystack_cot/run_xyz/checkpoints/best_model.pt

    # Evaluate with HuggingFace checkpoint (set in config YAML)
    python scripts/evaluate.py \
        --config configs/experiments/capture24_haystack_llama.yaml

    # Quick evaluation on a few samples
    python scripts/evaluate.py \
        --config configs/experiments/smoke_test.yaml \
        --max-samples 20 --no-wandb
"""

import argparse
import json
import random
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from src.models.base import BaseModel
from src.models.registry import get_model_class
from src.datasets.registry import get_dataset_class
from src.utils.config import ExperimentConfig


# ==============================================================================
# Utility Functions
# ==============================================================================

def get_device() -> str:
    """Get the best available device."""
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def get_gpu_memory_mb() -> float:
    """Get current GPU memory usage in MB."""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024 / 1024
    return 0.0


def format_time(seconds: float) -> str:
    """Format seconds as human-readable time."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}m"
    else:
        return f"{seconds/3600:.1f}h"


# ==============================================================================
# Model Loading
# ==============================================================================

def _resolve_checkpoint(
    config: ExperimentConfig,
    cli_checkpoint: str | None,
) -> Path | None:
    """Resolve checkpoint path from CLI arg, config, or run directory.

    Priority:
        1. ``cli_checkpoint`` — explicit CLI path
        2. Auto-discover from run directory using ``config.runtime.run_name``:
           ``<output_dir>/<dataset_name>/<run_name>/checkpoints/best_model.pt``

    HuggingFace Hub checkpoints (``config.backbone.hf_checkpoint_repo``) are
    handled separately in ``load_model_for_eval`` since they require download.

    Returns:
        Resolved local Path, or None if no local checkpoint found.
    """
    # 1. Explicit CLI path
    if cli_checkpoint is not None:
        path = Path(cli_checkpoint)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        return path

    # 2. Auto-discover from run directory
    if config.runtime.run_name:
        run_dir = (
            config.runtime.output_dir
            / config.dataset.name
            / config.runtime.run_name
        )
        best_ckpt = run_dir / "checkpoints" / "best_model.pt"
        if best_ckpt.exists():
            print(f"  Auto-discovered checkpoint from run '{config.runtime.run_name}'")
            return best_ckpt

    return None


def load_model_for_eval(
    config: ExperimentConfig,
    device: str,
    checkpoint_path: str | None = None,
) -> BaseModel:
    """Load model and checkpoint for evaluation.

    Checkpoint priority:
        1. ``checkpoint_path`` CLI argument (local file)
        2. ``config.backbone.hf_checkpoint_repo`` (HuggingFace Hub)
        3. Auto-discover from run directory:
           ``<output_dir>/<dataset_name>/<run_name>/checkpoints/best_model.pt``

    Args:
        config: Experiment configuration.
        device: Target device.
        checkpoint_path: Optional local checkpoint path (overrides config).

    Returns:
        Model in eval mode on the target device.
    """
    print(f"\nInitializing {config.model.architecture} model "
          f"with LLM: {config.backbone.model_id}")

    model_cls = get_model_class(config.model.architecture)
    model = model_cls.from_config(config, device).to(device)

    # Resolve checkpoint path with priority chain
    resolved_path = _resolve_checkpoint(config, checkpoint_path)
    loaded_from = None

    if resolved_path is not None:
        print(f"  Loading checkpoint: {resolved_path}")
        model.load_from_file(str(resolved_path))
        loaded_from = str(resolved_path)

    elif config.backbone.hf_checkpoint_repo:
        # HuggingFace Hub checkpoint
        from huggingface_hub import hf_hub_download
        print(f"  Downloading checkpoint from: {config.backbone.hf_checkpoint_repo}")
        hf_path = hf_hub_download(
            repo_id=config.backbone.hf_checkpoint_repo,
            filename=config.backbone.hf_checkpoint_file,
        )
        print(f"  Loading checkpoint: {hf_path}")
        model.load_from_file(hf_path)
        loaded_from = config.backbone.hf_checkpoint_repo

    else:
        print("  Warning: No checkpoint specified. Using randomly initialized weights.")

    model.eval()

    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Total parameters: {total_params:,}")
    print(f"  Memory: {get_gpu_memory_mb():.1f} MB")
    if loaded_from:
        print(f"  Checkpoint: {loaded_from}")

    return model


# ==============================================================================
# Data Loading
# ==============================================================================

def create_eval_dataloader(
    config: ExperimentConfig,
    eos_token: str,
    split: str = "test",
    batch_size: int | None = None,
    max_samples: int | None = None,
) -> tuple[DataLoader, Any]:
    """Create a single dataloader for evaluation.

    Args:
        config: Experiment configuration.
        eos_token: Model's EOS token (injected into dataset).
        split: Dataset split to load.
        batch_size: Override config batch size.
        max_samples: Override config max_samples.

    Returns:
        Tuple of (dataloader, raw_dataset). The raw dataset is needed
        for calling extract_answer() and evaluate_answer().
    """
    dataset_name = config.dataset.name
    dataset_kwargs = dict(config.dataset.extra_kwargs)
    dataset_kwargs["EOS_TOKEN"] = eos_token

    dataset_cls = get_dataset_class(dataset_name)
    print(f"\nLoading dataset '{dataset_name}' ({dataset_cls.__name__}), "
          f"split={split}...")

    dataset = dataset_cls(split=split, **dataset_kwargs)
    raw_dataset = dataset  # Keep reference before potential Subset wrapping

    # Apply max_samples
    effective_max = max_samples if max_samples is not None else config.runtime.max_samples
    if effective_max is not None and len(dataset) > effective_max:
        print(f"  Limiting to {effective_max} samples")
        indices = random.sample(range(len(dataset)), effective_max)
        dataset = Subset(dataset, indices)

    print(f"  Samples: {len(dataset)}")

    # Identity collate — model.prepare_batch() handles collation
    effective_bs = batch_size if batch_size is not None else config.training.batch_size

    dl_cfg = config.training.dataloader
    num_workers = dl_cfg.get("num_workers", 8 if torch.cuda.is_available() else 0)
    prefetch_factor = dl_cfg.get("prefetch_factor", 4 if num_workers > 0 else None)
    pin_memory = dl_cfg.get("pin_memory", torch.cuda.is_available())

    loader = DataLoader(
        dataset,
        batch_size=effective_bs,
        shuffle=False,
        collate_fn=lambda batch: batch,
        num_workers=num_workers,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        persistent_workers=num_workers > 0,
    )

    return loader, raw_dataset


# ==============================================================================
# Evaluation
# ==============================================================================

def evaluate(
    model: BaseModel,
    dataloader: DataLoader,
    dataset,
    max_new_tokens: int = 500,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    """Run generation and evaluation on a dataset split.

    For each batch:
        1. Prepare inputs (prompt only, no labels).
        2. Generate predictions.
        3. Extract and evaluate answers using dataset-specific logic.

    Results are streamed to a JSONL file incrementally so that partial
    progress is preserved if the process is interrupted. This also reduces
    peak memory usage since per-sample entries are flushed to disk rather
    than accumulated in a list.

    Args:
        model: Model in eval mode.
        dataloader: DataLoader for the split.
        dataset: Raw QADataset instance (for extract_answer / evaluate_answer).
        max_new_tokens: Maximum tokens to generate per sample.
        output_dir: Directory for incremental JSONL output. If None,
            results are kept in memory only (legacy behaviour).

    Returns:
        Results dict with per-sample outputs, overall accuracy, and
        per-task accuracy breakdown.
    """
    model.eval()
    category_correct: dict[str, int] = {}
    category_total: dict[str, int] = {}
    category_metric_sums: dict[str, float] = {}
    category_key = dataset.category_key if dataset else None
    primary_metric = dataset.primary_metric if dataset else "accuracy"
    sample_idx = 0
    total_correct = 0
    total_metric_sum = 0.0
    total_metric_count = 0

    # Open a JSONL file for incremental writes
    jsonl_path = output_dir / "outputs.jsonl" if output_dir else None
    jsonl_file = open(jsonl_path, "w") if jsonl_path else None

    _CORE_KEYS = {"pre_prompt", "time_series", "time_series_text",
                  "post_prompt", "answer"}

    pbar = tqdm(dataloader, desc="Evaluating", leave=True)

    try:
        with torch.no_grad():
            for batch in pbar:
                # Prepare inputs (inference mode — no labels)
                eval_inputs = model.prepare_batch(batch, training=False)

                # Generate predictions — pass any extra keys from prepare_batch
                # (e.g. query_ids, stage for ITFormer) as kwargs to generate().
                extra_kwargs = {
                    k: v for k, v in eval_inputs.items()
                    if k not in ("time_series", "input_ids", "attention_mask", "labels")
                }
                token_ids = model.generate(
                    eval_inputs["time_series"],
                    eval_inputs["input_ids"],
                    eval_inputs["attention_mask"],
                    max_new_tokens=max_new_tokens,
                    **extra_kwargs,
                )
                predictions = model.tokenizer.batch_decode(
                    token_ids, skip_special_tokens=True,
                )

                # Evaluate each sample
                for sample, pred in zip(batch, predictions):
                    category = sample.get(category_key, "unknown") if category_key else "all"
                    question = sample.get("question", "")
                    ground_truth = dataset.get_ground_truth(sample) if dataset else ""

                    pred_answer = dataset.extract_answer(pred, sample)
                    eval_result = dataset.evaluate_answer(pred_answer, sample)
                    is_correct = eval_result["correct"]

                    # Track per-category metrics
                    if category not in category_correct:
                        category_correct[category] = 0
                        category_total[category] = 0
                        category_metric_sums[category] = 0.0
                    category_total[category] += 1
                    if is_correct:
                        category_correct[category] += 1
                        total_correct += 1
                    if primary_metric != "accuracy" and primary_metric in eval_result:
                        category_metric_sums[category] += eval_result[primary_metric]
                        total_metric_sum += eval_result[primary_metric]
                        total_metric_count += 1

                    entry = {
                        "sample_idx": sample_idx,
                        "question": question[:200],
                        "ground_truth": ground_truth,
                        "predicted_answer": pred_answer,
                        "prediction_raw": pred,
                        "correct": is_correct,
                    }
                    if category_key:
                        entry["category"] = category

                    # Forward dataset metadata (e.g. task_type, answer_type,
                    # context_length_samples) so aggregation scripts can slice
                    # results without knowing which dataset produced them.
                    for k, v in sample.items():
                        if k not in _CORE_KEYS and k not in entry:
                            if isinstance(v, (str, int, float, bool)):
                                entry[k] = v

                    # Include any extra eval metrics (e.g. iou)
                    for k, v in eval_result.items():
                        if k != "correct" and v is not None:
                            entry[k] = v

                    # Stream to disk instead of accumulating in memory
                    if jsonl_file:
                        jsonl_file.write(json.dumps(entry) + "\n")
                        jsonl_file.flush()

                    sample_idx += 1

                # Update progress bar with running metric
                if primary_metric == "accuracy":
                    running_val = total_correct / sample_idx if sample_idx else 0.0
                    pbar.set_postfix({"acc": f"{running_val*100:.1f}%"})
                else:
                    running_val = total_metric_sum / total_metric_count if total_metric_count else 0.0
                    pbar.set_postfix({primary_metric: f"{running_val:.4f}"})
    finally:
        if jsonl_file:
            jsonl_file.close()

    # Re-read outputs from JSONL to build final result dict
    if jsonl_path and jsonl_path.exists():
        with open(jsonl_path) as f:
            outputs = [json.loads(line) for line in f]
    else:
        outputs = []

    # Aggregate results
    result: dict[str, Any] = {
        "samples_evaluated": sample_idx,
        "outputs": outputs,
    }

    if primary_metric == "accuracy":
        result["accuracy"] = total_correct / sample_idx if sample_idx else 0.0
        if category_key:
            result["category_accuracy"] = {
                cat: category_correct[cat] / category_total[cat]
                for cat in sorted(category_correct)
            }
    else:
        result[primary_metric] = total_metric_sum / total_metric_count if total_metric_count else 0.0
        if category_key:
            result[f"category_{primary_metric}"] = {
                cat: category_metric_sums[cat] / category_total[cat] if category_total[cat] > 0 else 0.0
                for cat in sorted(category_metric_sums)
            }

    return result


# ==============================================================================
# Main
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a trained TS-LLM on any registered dataset",
        epilog="Uses the same YAML configs as training. "
               "See configs/experiments/ for examples.",
    )

    parser.add_argument(
        "--config", type=str, required=True,
        help="Path to experiment YAML config",
    )
    parser.add_argument(
        "--checkpoint", type=str, default=None,
        help="Path to local checkpoint (overrides config)",
    )
    parser.add_argument(
        "--split", type=str, default="test",
        choices=["train", "validation", "test"],
        help="Dataset split to evaluate (default: test)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=None,
        help="Override config batch size",
    )
    parser.add_argument(
        "--max-samples", type=int, default=5,
        help="Limit number of samples to evaluate",
    )
    parser.add_argument(
        "--max-new-tokens", type=int, default=500,
        help="Maximum tokens to generate per sample (default: 500)",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Output directory for results (default: results/eval/)",
    )
    parser.add_argument(
        "--no-wandb", action="store_true",
        help="Disable W&B logging (overrides config)",
    )

    args = parser.parse_args()

    # Load config
    config = ExperimentConfig.from_yaml(args.config)

    if args.no_wandb:
        wandb_cfg = config.training.logging.setdefault("wandb", {})
        wandb_cfg["enabled"] = False

    # Seed for reproducibility
    seed = config.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = get_device()

    # Resolve output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path("results") / "eval" / f"eval_{timestamp}"

    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print(f"Evaluation [{config.dataset.name}]")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  Architecture: {config.model.architecture}")
    print(f"  Encoder: {config.model.encoder.get('type', 'cnn_tokenizer')}")
    print(f"  Device: {device}")
    print(f"  Dataset: {config.dataset.name}")
    print(f"  Split: {args.split}")
    print(f"  Max new tokens: {args.max_new_tokens}")
    print(f"  Output dir: {output_dir}")

    # Save config snapshot
    with open(output_dir / "config.json", "w") as f:
        json.dump(config.to_dict(), f, indent=2)

    # Load model
    model = load_model_for_eval(config, device, checkpoint_path=args.checkpoint)

    # Create dataloader
    eos_token = model.get_eos_token()
    dataloader, dataset = create_eval_dataloader(
        config,
        eos_token=eos_token,
        split=args.split,
        batch_size=args.batch_size,
        max_samples=args.max_samples,
    )

    # Run evaluation
    print(f"\nRunning evaluation on {len(dataloader.dataset)} samples...")
    print("=" * 70)

    start_time = time.time()
    results = evaluate(
        model, dataloader, dataset,
        max_new_tokens=args.max_new_tokens,
        output_dir=output_dir,
    )
    eval_time = time.time() - start_time

    # Add metadata to results
    results["split"] = args.split
    resolved_ckpt = _resolve_checkpoint(config, args.checkpoint)
    results["checkpoint"] = (
        str(resolved_ckpt) if resolved_ckpt
        else config.backbone.hf_checkpoint_repo or "none"
    )
    results["config"] = args.config
    results["max_new_tokens"] = args.max_new_tokens
    results["eval_time_seconds"] = eval_time

    # Save results
    results_file = output_dir / f"eval_{args.split}.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)

    # Print summary
    primary_metric = dataset.primary_metric if dataset else "accuracy"
    metric_value = results.get(primary_metric, 0.0)

    print("\n" + "=" * 70)
    print("Evaluation Complete")
    print("=" * 70)
    print(f"  Split: {args.split}")
    print(f"  Samples: {results['samples_evaluated']}")
    if primary_metric == "accuracy":
        print(f"  Accuracy: {metric_value*100:.1f}%")
    else:
        print(f"  {primary_metric}: {metric_value:.4f}")
    print(f"  Time: {format_time(eval_time)}")
    print(f"  Results: {results_file}")

    cat_key = f"category_{primary_metric}" if primary_metric != "accuracy" else "category_accuracy"
    if results.get(cat_key):
        print(f"\n  Per-category {primary_metric}:")
        for cat, val in results[cat_key].items():
            cat_count = sum(
                1 for o in results["outputs"] if o.get("category") == cat
            )
            if primary_metric == "accuracy":
                print(f"    {cat}: {val*100:.1f}% ({cat_count} samples)")
            else:
                print(f"    {cat}: {val:.4f} ({cat_count} samples)")

    # Show a few sample predictions
    print("\n  Sample predictions:")
    for output in results["outputs"][:3]:
        status = "CORRECT" if output["correct"] else "WRONG"
        category = output.get("category", "")
        label = f"[{status}] {category}" if category else f"[{status}]"
        print(f"    {label}")
        print(f"      Q: {output['question'][:80]}...")
        print(f"      Gold: {output['ground_truth']}")
        print(f"      Pred: {output['predicted_answer']}")


if __name__ == "__main__":
    main()
