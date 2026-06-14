#!/usr/bin/env python3
"""
Train TS-LLM on any registered dataset.

All configuration is driven by a single YAML file (see configs/experiments/).

Usage:
    # Standard training run
    python scripts/train.py --config configs/experiments/capture24_haystack_llama.yaml

    # Disable W&B logging (overrides config)
    python scripts/train.py --config configs/experiments/capture24_haystack_llama.yaml --no-wandb
"""

import argparse
import gc
import json
import random
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

import torch
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import get_cosine_schedule_with_warmup, get_linear_schedule_with_warmup

from src.models.base import BaseModel
from src.models.registry import get_model_class
from src.datasets.registry import get_dataset_class
from src.training.logging import ExperimentLogger
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
# Model Setup
# ==============================================================================

def load_model(config: ExperimentConfig, device: str) -> BaseModel:
    """
    Load model using the registry.

    Each model class implements ``from_config()`` to extract the parameters
    it needs, so this function stays architecture-agnostic.
    """
    print(f"\nInitializing {config.model.architecture} model with LLM: {config.backbone.model_id}")

    model_cls = get_model_class(config.model.architecture)
    model = model_cls.from_config(config, device).to(device)

    # Optionally load pretrained weights from HuggingFace
    if config.backbone.hf_checkpoint_repo:
        try:
            from huggingface_hub import hf_hub_download
            print(f"  Downloading checkpoint from: {config.backbone.hf_checkpoint_repo}")
            checkpoint_path = hf_hub_download(
                repo_id=config.backbone.hf_checkpoint_repo,
                filename=config.backbone.hf_checkpoint_file,
            )
            print(f"  Loading checkpoint: {checkpoint_path}")
            model.load_from_file(checkpoint_path)
        except Exception as e:
            print(f"  Warning: Could not load from HuggingFace: {e}")
            print("  Proceeding with randomly initialized weights")

    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Memory: {get_gpu_memory_mb():.1f} MB")

    return model


def configure_training_mode(
    model: BaseModel,
    config: ExperimentConfig,
) -> dict[str, list]:
    """Configure which parameters are trainable based on ``config.training.backbone_training``.

    Modes:
        ``freeze`` — backbone LLM frozen, model components trained
            (per ``get_trainable_parameters()``).
        ``lora`` — same as freeze + LoRA adapters injected into the
            backbone LLM.
        ``full`` — all parameters trainable.

    Returns:
        Dictionary mapping group names to parameter lists, ready for
        the optimizer.
    """
    mode = config.training.backbone_training

    if mode == "freeze":
        model.requires_grad_(False)
        param_groups = model.get_trainable_parameters()
        for params in param_groups.values():
            for p in params:
                p.requires_grad = True
        print(f"\nTraining mode: freeze (backbone frozen)")

    elif mode == "lora":
        model.requires_grad_(False)

        # Inject LoRA into backbone
        backbone_module = model.get_backbone_module()
        if backbone_module is None:
            raise ValueError(
                f"{type(model).__name__} does not expose a backbone module. "
                "Implement get_backbone_module() to use lora mode."
            )
        model.backbone.apply_peft(backbone_module, config.training.lora)

        # Re-unfreeze trainable groups — inject_adapter_in_model re-freezes
        # all non-LoRA params inside the backbone, which clobbers cross-attn
        # and embeddings that live inside lang_encoder.
        param_groups = model.get_trainable_parameters()
        for params in param_groups.values():
            for p in params:
                p.requires_grad = True

        lora_params = [
            p for n, p in model.named_parameters()
            if "lora_" in n and p.requires_grad
        ]
        param_groups["backbone_lora"] = lora_params

        lora_cfg = config.training.lora
        print(f"\nTraining mode: lora (r={lora_cfg.get('r', 16)}, "
              f"alpha={lora_cfg.get('alpha', 32)})")

    elif mode == "full":
        model.requires_grad_(True)
        param_groups = model.get_trainable_parameters()
        # Add remaining backbone params as their own group
        grouped_ids = {id(p) for ps in param_groups.values() for p in ps}
        remaining = [p for p in model.parameters() if id(p) not in grouped_ids]
        if remaining:
            param_groups["backbone"] = remaining
        print(f"\nTraining mode: full (all params trainable)")

    else:
        raise ValueError(
            f"Unknown training.backbone_training: {mode!r}. "
            "Use 'freeze', 'lora', or 'full'."
        )

    # Print summary
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Trainable: {trainable_params:,} / {total_params:,} "
          f"({100 * trainable_params / total_params:.2f}%)")
    for group_name, params in param_groups.items():
        group_count = sum(p.numel() for p in params if p.requires_grad)
        print(f"  - {group_name}: {group_count:,}")

    return param_groups


# ==============================================================================
# Data Loading
# ==============================================================================

def create_dataloaders(
    config: ExperimentConfig,
    eos_token: str,
) -> tuple:
    """
    Create train, validation, and test dataloaders.

    Args:
        config: Experiment configuration
        eos_token: The model's EOS token, injected into datasets so they
            stay model-agnostic.

    Returns:
        Tuple of (train_loader, val_loader, test_loader,
                  train_dataset, val_dataset, test_dataset)
    """
    dataset_name = config.dataset.name
    dataset_kwargs = dict(config.dataset.extra_kwargs)
    dataset_kwargs["EOS_TOKEN"] = eos_token  # Bridge model → dataset
    dataset_cls = get_dataset_class(dataset_name)
    print(f"\nLoading dataset '{dataset_name}' ({dataset_cls.__name__})...")
    if dataset_kwargs:
        print(f"  Dataset kwargs: {dataset_kwargs}")

    train_dataset = dataset_cls(split="train", **dataset_kwargs)
    val_dataset = dataset_cls(split="validation", **dataset_kwargs)
    test_dataset = dataset_cls(split="test", **dataset_kwargs)

    # Keep original QADataset instances for evaluation methods
    # (Subset wrapping below hides them)
    raw_train_dataset = train_dataset
    raw_val_dataset = val_dataset
    raw_test_dataset = test_dataset

    # Apply max_samples if specified
    # max_samples controls train split; val/test get eval_samples_ratio (default 10%) each
    max_samples = config.runtime.max_samples
    if max_samples is not None:
        eval_samples = max(1, int(max_samples * config.runtime.eval_samples_ratio))
        print(f"  Limiting to {max_samples} train / {eval_samples} val / {eval_samples} test samples")
        from torch.utils.data import Subset
        import random

        if len(train_dataset) > max_samples:
            train_indices = random.sample(range(len(train_dataset)), max_samples)
            train_dataset = Subset(train_dataset, train_indices)
        if len(val_dataset) > eval_samples:
            val_indices = random.sample(range(len(val_dataset)), eval_samples)
            val_dataset = Subset(val_dataset, val_indices)
        if len(test_dataset) > eval_samples:
            test_indices = random.sample(range(len(test_dataset)), eval_samples)
            test_dataset = Subset(test_dataset, test_indices)

    print(f"  Train samples: {len(train_dataset)}")
    print(f"  Validation samples: {len(val_dataset)}")
    print(f"  Test samples: {len(test_dataset)}")

    # Identity collate — model.prepare_batch() handles patch-alignment
    def collate_fn(batch):
        return batch

    dl_cfg = config.training.dataloader
    num_workers = dl_cfg.get("num_workers", 8 if torch.cuda.is_available() else 0)
    prefetch_factor = dl_cfg.get("prefetch_factor", 4 if num_workers > 0 else None)
    pin_memory = dl_cfg.get("pin_memory", torch.cuda.is_available())

    if num_workers > 0:
        print(f"  DataLoader workers: {num_workers} (prefetch_factor={prefetch_factor})")

    loader_kwargs = dict(
        batch_size=config.training.batch_size,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        persistent_workers=num_workers > 0,
    )

    train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_dataset, shuffle=False, **loader_kwargs)

    return (train_loader, val_loader, test_loader,
            raw_train_dataset, raw_val_dataset, raw_test_dataset)


# ==============================================================================
# Checkpointing
# ==============================================================================

def save_checkpoint(
    model,
    optimizer: AdamW,
    scheduler,
    epoch: int,
    val_loss: float,
    output_dir: Path,
    config: ExperimentConfig,
    is_best: bool = False,
    batch_idx: Optional[int] = None,
    best_val_metric: Optional[float] = None,
) -> Path:
    """Save training checkpoint."""
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "epoch": epoch,
        "batch_idx": batch_idx,
        "val_loss": val_loss,
        "best_val_metric": best_val_metric,
        "config": config.to_dict(),
        "timestamp": datetime.now().isoformat(),
    }

    if is_best:
        path = checkpoint_dir / "best_model.pt"
    elif batch_idx is not None:
        path = checkpoint_dir / f"checkpoint_epoch_{epoch}_batch_{batch_idx}.pt"
    else:
        path = checkpoint_dir / f"checkpoint_epoch_{epoch}.pt"

    torch.save(checkpoint, path)

    # Also save as "latest" for easy resumption
    latest_path = checkpoint_dir / "latest_checkpoint.pt"
    torch.save(checkpoint, latest_path)

    if is_best:
        print(f"  Saved best checkpoint to {path}")
    elif batch_idx is not None:
        print(f"  [Checkpoint] Saved at epoch {epoch}, batch {batch_idx}")

    return path


def find_latest_checkpoint(output_dir: Path) -> Optional[str]:
    """Find the latest checkpoint in a run directory."""
    checkpoint_dir = output_dir / "checkpoints"
    latest_path = checkpoint_dir / "latest_checkpoint.pt"

    if latest_path.exists():
        return str(latest_path)
    return None


def load_checkpoint(
    path: str,
    model,
    optimizer: Optional[AdamW] = None,
    scheduler=None,
) -> Dict[str, Any]:
    """Load training checkpoint."""
    print(f"\nLoading checkpoint from: {path}")

    checkpoint = torch.load(path, map_location="cpu")

    model.load_state_dict(checkpoint["model_state"])

    if optimizer is not None and "optimizer_state" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state"])

    if scheduler is not None and "scheduler_state" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state"])

    print(f"  Resumed from epoch {checkpoint['epoch']}")
    print(f"  Validation loss: {checkpoint['val_loss']:.4f}")

    return checkpoint


# ==============================================================================
# Training Loop
# ==============================================================================

def train_epoch(
    model,
    train_loader: DataLoader,
    optimizer: AdamW,
    scheduler,
    config: ExperimentConfig,
    epoch: int,
    output_dir: Path,
    logger: Optional[ExperimentLogger] = None,
    global_step: int = 0,
) -> tuple:
    """
    Train for one epoch.

    Returns:
        Tuple of (average training loss, updated global step)
    """
    model.train()
    total_loss = 0.0
    num_batches = 0

    pbar = tqdm(train_loader, desc=f"Epoch {epoch} [Train]", leave=False)

    for batch_idx, batch in enumerate(pbar):

        optimizer.zero_grad()

        try:
            batch_inputs = model.prepare_batch(batch)
            output = model(**batch_inputs)
            loss = output["loss"]

            if torch.isnan(loss) or torch.isinf(loss):
                print(f"\n  Warning: Invalid loss at batch {batch_idx}, skipping")
                continue

            loss.backward()

            # Gradient clipping
            clip_grad_norm_(model.parameters(), config.training.max_grad_norm)

            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            num_batches += 1

            # Update progress bar
            avg_loss = total_loss / num_batches
            current_lr = scheduler.get_last_lr()[0]
            pbar.set_postfix({
                "loss": f"{avg_loss:.4f}",
                "lr": f"{current_lr:.2e}",
            })

            # Log step metrics
            if logger is not None:
                logger.log_metrics({
                    "train/loss": loss.item(),
                    "train/lr": current_lr,
                    "train/avg_loss": avg_loss,
                }, step=global_step)
            global_step += 1

            # Save intra-epoch checkpoint every N batches
            ckpt_interval = config.checkpoint_interval
            if ckpt_interval > 0 and (batch_idx + 1) % ckpt_interval == 0:
                save_checkpoint(
                    model, optimizer, scheduler, epoch,
                    val_loss=avg_loss,
                    output_dir=output_dir,
                    config=config,
                    batch_idx=batch_idx + 1,
                )

        except Exception as e:
            print(f"\n  Error at batch {batch_idx}: {e}")
            continue

    return total_loss / max(num_batches, 1), global_step


def validate(
    model,
    val_loader: DataLoader,
    epoch: int,
    dataset=None,
    output_dir: Optional[Path] = None,
    split_name: str = "val",
    log_outputs: bool = False,
    max_new_tokens: int = 500,
) -> Dict[str, Any]:
    """
    Validate the model with optional output logging.

    Returns:
        Dictionary with validation loss and optional metrics
    """
    model.eval()
    total_loss = 0.0
    num_batches = 0

    # For output logging
    logged_outputs = []
    samples_generated = 0
    category_correct = {}
    category_total = {}
    category_metric_sums: dict[str, float] = {}
    category_key = dataset.category_key if dataset else None
    primary_metric = dataset.primary_metric if dataset else "accuracy"

    pbar = tqdm(val_loader, desc=f"Epoch {epoch} [{split_name.upper()}]", leave=False)

    with torch.no_grad():
        for batch in pbar:
            try:
                # Compute loss
                batch_inputs = model.prepare_batch(batch)
                output = model(**batch_inputs)
                loss = output["loss"]

                if not torch.isnan(loss) and not torch.isinf(loss):
                    total_loss += loss.item()
                    num_batches += 1

                avg_loss = total_loss / max(num_batches, 1)
                pbar.set_postfix({"loss": f"{avg_loss:.4f}"})

                # Generate outputs for logging
                if log_outputs:
                    try:
                        eval_inputs = model.prepare_batch(batch, training=False)
                        gen_inputs = {k: v for k, v in eval_inputs.items() if k != "labels"}
                        token_ids = model.generate(**gen_inputs, max_new_tokens=max_new_tokens)
                        predictions = model.tokenizer.batch_decode(
                            token_ids, skip_special_tokens=True
                        )

                        for sample, pred in zip(batch, predictions):
                            category = sample.get(category_key, "unknown") if category_key else "all"
                            question = sample.get("question", "")
                            ground_truth = dataset.get_ground_truth(sample) if dataset else ""

                            pred_answer = dataset.extract_answer(pred, sample) if dataset else pred.strip()
                            eval_result = dataset.evaluate_answer(pred_answer, sample) if dataset else {"correct": False}
                            is_correct = eval_result["correct"]

                            if category not in category_correct:
                                category_correct[category] = 0
                                category_total[category] = 0
                                category_metric_sums[category] = 0.0
                            category_total[category] += 1
                            if is_correct:
                                category_correct[category] += 1
                            if primary_metric != "accuracy" and primary_metric in eval_result:
                                category_metric_sums[category] += eval_result[primary_metric]

                            entry = {
                                "sample_idx": samples_generated,
                                "question": question[:200],
                                "ground_truth": ground_truth,
                                "predicted_answer": pred_answer,
                                "prediction_raw": pred,
                                "correct": is_correct,
                            }
                            if category_key:
                                entry["category"] = category
                            # Include any extra eval metrics (e.g. iou)
                            for k, v in eval_result.items():
                                if k != "correct" and v is not None:
                                    entry[k] = v

                            logged_outputs.append(entry)
                            samples_generated += 1

                    except Exception:
                        pass  # Skip generation errors

            except Exception:
                continue

    avg_loss = total_loss / max(num_batches, 1)

    result = {
        "loss": avg_loss,
        "num_batches": num_batches,
    }

    # Add evaluation metrics if we logged outputs
    if logged_outputs:
        result["samples_evaluated"] = len(logged_outputs)

        if primary_metric == "accuracy":
            total_correct = sum(1 for o in logged_outputs if o["correct"])
            result["accuracy"] = total_correct / len(logged_outputs)
            if category_key:
                result["category_accuracy"] = {
                    cat: category_correct[cat] / category_total[cat] if category_total[cat] > 0 else 0.0
                    for cat in category_correct
                }
        else:
            metric_values = [o[primary_metric] for o in logged_outputs if primary_metric in o]
            result[primary_metric] = sum(metric_values) / len(metric_values) if metric_values else 0.0
            if category_key:
                result[f"category_{primary_metric}"] = {
                    cat: category_metric_sums[cat] / category_total[cat] if category_total[cat] > 0 else 0.0
                    for cat in category_metric_sums
                }

        # Save logged outputs to file
        if output_dir:
            logs_dir = output_dir / "output_logs"
            logs_dir.mkdir(parents=True, exist_ok=True)
            log_file = logs_dir / f"{split_name}_epoch_{epoch}.json"
            log_data: dict[str, Any] = {
                "epoch": epoch,
                "split": split_name,
                "loss": avg_loss,
                "outputs": logged_outputs,
            }
            if primary_metric == "accuracy":
                log_data["accuracy"] = result.get("accuracy", 0.0)
                log_data["category_accuracy"] = result.get("category_accuracy", {})
            else:
                log_data[primary_metric] = result.get(primary_metric, 0.0)
                cat_key = f"category_{primary_metric}"
                if cat_key in result:
                    log_data[cat_key] = result[cat_key]
            with open(log_file, "w") as f:
                json.dump(log_data, f, indent=2)

    return result


def train(
    config: ExperimentConfig,
) -> Dict[str, Any]:
    """
    Main training function.
    """
    # Seed everything for reproducibility
    seed = config.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = get_device()
    dataset_name = config.dataset.name
    num_epochs = config.training.num_epochs
    default_lr = config.training.learning_rate.get("default", 2e-4)

    # Build output directory: <output_dir>/<dataset>/<run_folder>
    base_dir = config.runtime.output_dir / dataset_name
    resume_from = config.runtime.resume_from

    if resume_from is None:
        if config.runtime.run_name:
            run_folder = config.runtime.run_name
            potential_output_dir = base_dir / run_folder

            if potential_output_dir.exists():
                latest_ckpt = find_latest_checkpoint(potential_output_dir)
                if latest_ckpt:
                    print(f"\nFound existing run '{run_folder}', auto-resuming from latest checkpoint...")
                    resume_from = latest_ckpt
                    output_dir = potential_output_dir
                else:
                    output_dir = potential_output_dir
            else:
                output_dir = potential_output_dir
        else:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            run_folder = f"run_{timestamp}"
            output_dir = base_dir / run_folder
    else:
        if config.runtime.run_name:
            output_dir = base_dir / config.runtime.run_name
        else:
            output_dir = config.runtime.output_dir

    if resume_from is not None and output_dir == Path("results"):
        checkpoint_path = Path(resume_from)
        if checkpoint_path.parent.name == "checkpoints":
            output_dir = checkpoint_path.parent.parent
        else:
            output_dir = checkpoint_path.parent

    print("=" * 70)
    print(f"Training [{dataset_name}]")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  Architecture: {config.model.architecture}")
    print(f"  Encoder: {config.model.encoder.get('type', 'cnn_tokenizer')}")
    print(f"  Device: {device}")
    print(f"  Dataset: {dataset_name}")
    print(f"  Batch size: {config.training.batch_size}")
    print(f"  Epochs: {num_epochs}")
    print(f"  Learning rate: {default_lr}")
    print(f"  Output dir: {output_dir}")

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save config (full structured dump)
    with open(output_dir / "config.json", "w") as f:
        json.dump(config.to_dict(), f, indent=2)
    from src.utils.config import save_yaml
    save_yaml(config.to_dict(), output_dir / "config.yaml")

    # Initialize experiment logger
    if not config.use_wandb:
        print("\n  NOTE: W&B logging is disabled. Using local-only logging.")
        print("  To enable W&B, set training.logging.wandb.enabled: true in config.")

    exp_logger = ExperimentLogger(
        experiment_name=config.runtime.run_name or output_dir.name,
        config=config.to_dict(),
        use_wandb=config.use_wandb,
        wandb_project=config.wandb_project,
        wandb_entity=config.wandb_entity,
        wandb_group=config.wandb_group,
        log_dir=str(output_dir),
    )

    # Load model
    model = load_model(config, device)

    # Configure which parameters are trainable (freeze / lora / full)
    param_groups = configure_training_mode(model, config)

    # Create dataloaders (+ raw dataset instances for evaluation methods)
    eos_token = model.get_eos_token()
    (train_loader, val_loader, test_loader,
     _, val_dataset, test_dataset) = create_dataloaders(config, eos_token=eos_token)

    # Setup optimizer with per-group learning rates
    lr_config = config.training.learning_rate
    optimizer_groups = []
    for group_name, params in param_groups.items():
        group_lr = lr_config.get(group_name, default_lr)
        optimizer_groups.append({"params": params, "lr": group_lr})

    optimizer = AdamW(
        optimizer_groups,
        lr=default_lr,
        weight_decay=config.training.weight_decay,
    )

    # Setup scheduler
    total_steps = num_epochs * len(train_loader)
    warmup_steps = int(config.training.warmup_ratio * total_steps)

    scheduler_type = getattr(config.training, "scheduler", None)
    scheduler_type = scheduler_type.get("type", "linear_with_warmup") if isinstance(scheduler_type, dict) else "linear_with_warmup"

    if scheduler_type == "cosine":
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )
    else:
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )

    print(f"\nTraining setup:")
    print(f"  Total steps: {total_steps}")
    print(f"  Warmup steps: {warmup_steps}")

    # Resume from checkpoint if specified
    start_epoch = 1
    best_val_metric = None

    # Determine the primary metric and whether higher is better
    val_primary_metric = val_dataset.primary_metric if val_dataset else "accuracy"
    # For loss-like metrics lower is better; for accuracy/f1/etc higher is better
    metric_higher_is_better = val_primary_metric != "loss"

    if resume_from:
        if config.runtime.resume_weights_only:
            checkpoint = load_checkpoint(resume_from, model)
            print(f"  Loaded model weights from: {resume_from}")
        else:
            checkpoint = load_checkpoint(
                resume_from, model, optimizer, scheduler
            )
            start_epoch = checkpoint["epoch"] + 1
            best_val_metric = checkpoint.get("best_val_metric", checkpoint.get("val_loss"))
            print(f"  Resuming training from epoch {start_epoch}")

    # Training history
    history = {
        "train_loss": [],
        "val_loss": [],
        "best_epoch": None,
        "best_val_metric": None,
        "best_val_metric_name": val_primary_metric,
    }

    epochs_no_improve = 0
    global_step = 0
    training_start_time = time.time()

    print("\n" + "=" * 70)
    print("Starting training...")
    print("=" * 70 + "\n")

    for epoch in range(start_epoch, num_epochs + 1):
        epoch_start_time = time.time()

        # Train
        train_loss, global_step = train_epoch(
            model, train_loader, optimizer, scheduler, config, epoch,
            output_dir=output_dir,
            logger=exp_logger, global_step=global_step,
        )

        # Validate with output logging
        val_result = validate(
            model, val_loader, epoch,
            dataset=val_dataset,
            output_dir=output_dir,
            split_name="val",
            log_outputs=True,
            max_new_tokens=config.training.eval_max_new_tokens,
        )
        val_loss = val_result["loss"]

        # Record history
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        val_primary_metric = val_dataset.primary_metric if val_dataset else "accuracy"
        val_metric_value = val_result.get(val_primary_metric)
        if val_metric_value is not None:
            hist_key = f"val_{val_primary_metric}"
            if hist_key not in history:
                history[hist_key] = []
            history[hist_key].append(val_metric_value)

        epoch_time = time.time() - epoch_start_time

        # Print epoch summary
        summary_str = (f"Epoch {epoch}/{num_epochs}: "
                      f"train_loss={train_loss:.4f}, "
                      f"val_loss={val_loss:.4f}")
        if val_metric_value is not None:
            if val_primary_metric == "accuracy":
                summary_str += f", val_acc={val_metric_value*100:.1f}%"
            else:
                summary_str += f", val_{val_primary_metric}={val_metric_value:.4f}"
        summary_str += f", time={format_time(epoch_time)}"
        print(summary_str)

        cat_breakdown_key = f"category_{val_primary_metric}" if val_primary_metric != "accuracy" else "category_accuracy"
        if cat_breakdown_key in val_result:
            for cat, val in sorted(val_result[cat_breakdown_key].items()):
                if val_primary_metric == "accuracy":
                    print(f"    {cat}: {val*100:.1f}%")
                else:
                    print(f"    {cat}: {val:.4f}")

        # Log epoch metrics
        epoch_metrics = {
            "train/epoch_loss": train_loss,
            "val/loss": val_loss,
        }
        if val_metric_value is not None:
            epoch_metrics[f"val/{val_primary_metric}"] = val_metric_value
        if cat_breakdown_key in val_result:
            for cat, val in val_result[cat_breakdown_key].items():
                epoch_metrics[f"val/{val_primary_metric}_{cat}"] = val
        exp_logger.log_epoch_metrics(epoch_metrics, epoch=epoch, step=global_step)

        # Check for improvement based on primary metric
        current_metric = val_metric_value if val_metric_value is not None else -val_loss
        is_improvement = (
            best_val_metric is None
            or (metric_higher_is_better and current_metric > best_val_metric)
            or (not metric_higher_is_better and current_metric < best_val_metric)
        )
        if is_improvement:
            best_val_metric = current_metric
            history["best_epoch"] = epoch
            history["best_val_metric"] = best_val_metric
            epochs_no_improve = 0

            checkpoint_path = save_checkpoint(
                model, optimizer, scheduler, epoch, val_loss,
                output_dir=output_dir, config=config, is_best=True,
                best_val_metric=best_val_metric,
            )
            exp_logger.log_artifact(checkpoint_path, artifact_type="model", name="best_model")
            print(f"  New best model! ({val_primary_metric}={current_metric:.4f})")
        else:
            epochs_no_improve += 1
            print(f"  No improvement for {epochs_no_improve} epochs")

        # Early stopping
        if epochs_no_improve >= config.early_stop_patience:
            print(f"\nEarly stopping triggered after {epoch} epochs")
            break

        # Save history
        with open(output_dir / "history.json", "w") as f:
            json.dump(history, f, indent=2)

        # Memory cleanup
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    total_time = time.time() - training_start_time

    # Final summary
    print("\n" + "=" * 70)
    print("Training Complete")
    print("=" * 70)
    print(f"  Total time: {format_time(total_time)}")
    print(f"  Best epoch: {history['best_epoch']}")
    print(f"  Best val {val_primary_metric}: {history['best_val_metric']:.4f}")
    print(f"  Checkpoint: {output_dir / 'checkpoints' / 'best_model.pt'}")

    # Load best model for final evaluation
    if history["best_epoch"] is not None:
        load_checkpoint(
            str(output_dir / "checkpoints" / "best_model.pt"),
            model
        )

    # Final test evaluation
    print("\nFinal test evaluation...")
    test_result = validate(
        model, test_loader, epoch=0,
        dataset=test_dataset,
        output_dir=output_dir,
        split_name="test",
        log_outputs=True,
        max_new_tokens=config.training.eval_max_new_tokens,
    )
    test_primary_metric = test_dataset.primary_metric if test_dataset else "accuracy"
    test_metric_value = test_result.get(test_primary_metric)
    print(f"  Test loss: {test_result['loss']:.4f}")
    if test_metric_value is not None:
        if test_primary_metric == "accuracy":
            print(f"  Test accuracy: {test_metric_value*100:.1f}%")
        else:
            print(f"  Test {test_primary_metric}: {test_metric_value:.4f}")
        test_cat_key = f"category_{test_primary_metric}" if test_primary_metric != "accuracy" else "category_accuracy"
        if test_cat_key in test_result:
            print(f"  Per-category {test_primary_metric}:")
            for cat, val in sorted(test_result[test_cat_key].items()):
                if test_primary_metric == "accuracy":
                    print(f"    {cat}: {val*100:.1f}%")
                else:
                    print(f"    {cat}: {val:.4f}")

    history["test_loss"] = test_result["loss"]
    if test_metric_value is not None:
        history[f"test_{test_primary_metric}"] = test_metric_value
        test_cat_key = f"category_{test_primary_metric}" if test_primary_metric != "accuracy" else "category_accuracy"
        if test_cat_key in test_result:
            history[f"test_{test_cat_key}"] = test_result[test_cat_key]

    # Log test metrics so they appear as chart data in W&B
    test_metrics = {"test/loss": test_result["loss"]}
    if test_metric_value is not None:
        test_metrics[f"test/{test_primary_metric}"] = test_metric_value
    test_cat_key = f"category_{test_primary_metric}" if test_primary_metric != "accuracy" else "category_accuracy"
    if test_cat_key in test_result:
        for cat, val in test_result[test_cat_key].items():
            test_metrics[f"test/{test_primary_metric}_{cat}"] = val
    exp_logger.log_metrics(test_metrics, step=global_step)

    # Save final history
    with open(output_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    # Log final summary and finish
    summary = {
        "best_epoch": history.get("best_epoch"),
        "best_val_loss": history.get("best_val_loss"),
        "test_loss": history.get("test_loss"),
    }
    test_metric_hist_key = f"test_{test_primary_metric}"
    if test_metric_hist_key in history:
        summary[test_metric_hist_key] = history[test_metric_hist_key]
    test_cat_hist_key = f"test_category_{test_primary_metric}" if test_primary_metric != "accuracy" else "test_category_accuracy"
    if test_cat_hist_key in history:
        summary[test_cat_hist_key] = history[test_cat_hist_key]
    exp_logger.log_summary(summary)
    exp_logger.finish()

    return history


# ==============================================================================
# Main
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Train TS-LLM on any registered dataset",
        epilog="All model, dataset, training and runtime settings are defined in the "
               "YAML config file. See configs/experiments/ for examples.",
    )

    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to experiment YAML config (e.g. configs/experiments/capture24_haystack_llama.yaml)",
    )
    parser.add_argument(
        "--no-wandb",
        action="store_true",
        help="Disable Weights & Biases logging (overrides config file)",
    )

    args = parser.parse_args()

    # Load config from YAML
    config = ExperimentConfig.from_yaml(args.config)

    # Apply --no-wandb override
    if args.no_wandb:
        wandb_cfg = config.training.logging.setdefault("wandb", {})
        wandb_cfg["enabled"] = False

    # Train
    train(config)


if __name__ == "__main__":
    main()
