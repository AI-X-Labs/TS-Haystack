#!/usr/bin/env python3
# SPDX-License-Identifier: CC-BY-NC-4.0
"""Orchestrator for the Anon-TSLM paper runs.

Iterates every config under ``configs/paper/*.yaml`` and runs the ones
whose results are missing. Each run is launched as a subprocess so CUDA
state is fully released on failure. On OOM, retries with this fallback
order (all retries wipe the run's checkpoint dir first — we retrain from
scratch rather than resume on a different data mix):

    1. Initial attempt as written.
    2. If ``training.batch_size > 1``: retry with ``batch_size: 1``.
    3. Drop the largest entry from
       ``dataset.extra_kwargs.context_lengths_seconds`` and retry.
    4. Repeat (3) until the context list is empty -> mark FAILED.

Idempotency: a config is considered DONE iff a
``<run_dir>/output_logs/test_epoch_*.json`` exists.

After a successful run, ``scripts/eval/eval_haystack_log.py`` is invoked
to emit ``<run_dir>/metrics_summary.json``. At the end, all per-run
summaries are rolled up into ``results/paper_summary.csv``.

Usage:
    uv run python scripts/run_paper_runs.py
    uv run python scripts/run_paper_runs.py --dry-run
    uv run python scripts/run_paper_runs.py --only itformer
    uv run python scripts/run_paper_runs.py --force chatts_ltaf_haystack
    uv run python scripts/run_paper_runs.py --no-eval
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Optional

import yaml


REPO_ROOT = Path(__file__).resolve().parent.parent
PAPER_CONFIGS_DIR = REPO_ROOT / "configs" / "paper"
REDUCED_RUNS_DIR = PAPER_CONFIGS_DIR / "_runs"
RESULTS_DIR = REPO_ROOT / "results"
MANIFEST_PATH = RESULTS_DIR / "paper_manifest.json"
SUMMARY_CSV = RESULTS_DIR / "paper_summary.csv"

OOM_PATTERNS = (
    "CUDA out of memory",
    "OutOfMemoryError",
    "torch.cuda.OutOfMemoryError",
    "CUDA error: out of memory",
)


# Default architecture run order for the paper. Configs whose stem starts
# with one of these prefixes (followed by ``_``) are scheduled in this
# order, regardless of alphabetical filename. Within an architecture,
# configs sort alphabetically by stem (so dataset order is stable).
# Stems whose architecture isn't listed fall through to a final bucket.
ARCH_PRIORITY = ("flamingo", "itformer", "chatts", "chattime")


def _config_sort_key(path: Path) -> tuple[int, str]:
    stem = path.stem
    for i, arch in enumerate(ARCH_PRIORITY):
        if stem.startswith(arch + "_"):
            return (i, stem)
    return (len(ARCH_PRIORITY), stem)


# ---------------------------------------------------------------------------
# YAML helpers
# ---------------------------------------------------------------------------

def load_yaml(path: Path) -> dict:
    with path.open() as f:
        return yaml.safe_load(f)


def dump_yaml(data: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        yaml.safe_dump(data, f, sort_keys=False)


def set_nested(d: dict, dotted: str, value: Any) -> None:
    keys = dotted.split(".")
    for k in keys[:-1]:
        d = d.setdefault(k, {})
    d[keys[-1]] = value


# ---------------------------------------------------------------------------
# Run-state helpers
# ---------------------------------------------------------------------------

def expected_result_dir(cfg: dict) -> Path:
    dataset_name = cfg["dataset"]["name"]
    run_name = cfg["runtime"]["run_name"]
    output_dir = cfg["runtime"].get("output_dir", "results")
    return (REPO_ROOT / output_dir / dataset_name / run_name).resolve()


def is_completed(cfg: dict) -> bool:
    logs = expected_result_dir(cfg) / "output_logs"
    if not logs.exists():
        return False
    return any(logs.glob("test_epoch_*.json"))


def wipe_checkpoints(result_dir: Path) -> None:
    ckpt = result_dir / "checkpoints"
    if ckpt.exists():
        print(f"  [orchestrator] wiping {ckpt}")
        shutil.rmtree(ckpt)


# ---------------------------------------------------------------------------
# Context-length resolution for the OOM fallback
# ---------------------------------------------------------------------------

def _ctx_sort_key(c: Any) -> tuple:
    try:
        return (0, float(c))
    except (TypeError, ValueError):
        return (1, str(c))


def _apply_min_ctx(ctxs: list, min_ctx: Optional[float]) -> list:
    """Filter out bins strictly below the floor. String entries like 'full'
    are kept because they're always ≥ any numeric floor."""
    if min_ctx is None:
        return list(ctxs)
    out = []
    for c in ctxs:
        try:
            if float(c) >= float(min_ctx):
                out.append(c)
        except (TypeError, ValueError):
            out.append(c)  # 'full' etc.
    return out


def resolve_context_lengths(cfg: dict, min_ctx: Optional[float] = None) -> list:
    """Materialize 'all' to a concrete list for the given dataset/config.

    Returns the list in its original loader order when possible. The OOM
    fallback sorts internally when it needs to pick "largest". If
    ``min_ctx`` is set, bins below the floor are removed so the
    orchestrator never trains on paper-incomparable short contexts.
    """
    ds = cfg.get("dataset", {}) or {}
    name = ds.get("name")
    extra = ds.get("extra_kwargs") or {}
    ctxs = extra.get("context_lengths_seconds", ["all"])

    if isinstance(ctxs, list) and len(ctxs) == 1 and str(ctxs[0]).lower() == "all":
        if name in ("ltaf_haystack", "ltaf_haystack_cot"):
            from src.datasets.ltaf_haystack.qa_loader import (
                LTAF_HAYSTACK_COT_DIR,
                LTAF_HAYSTACK_TASKS_DIR,
                get_available_context_lengths,
            )
            base_cfg = extra.get("base_dir")
            if base_cfg:
                base = Path(base_cfg)
            elif name == "ltaf_haystack_cot":
                base = Path(LTAF_HAYSTACK_COT_DIR)
            else:
                base = Path(LTAF_HAYSTACK_TASKS_DIR)
            resolved = list(get_available_context_lengths(base))
        elif name == "sleep_psg_haystack":
            from src.datasets.sleep_psg_haystack.qa_loader import (
                ALL_CONTEXT_LENGTHS_PER_LABEL,
            )
            label = extra["label_class"]
            resolved = list(ALL_CONTEXT_LENGTHS_PER_LABEL[label])
        elif name in ("capture24_haystack_classification", "capture24_haystack_cot"):
            from src.datasets.capture24_haystack.qa_loader import get_available_context_lengths
            use_cot = name.endswith("_cot")
            cot_small = bool(extra.get("cot_small", False))
            data_dir = ds.get("data_dir")
            resolved = list(get_available_context_lengths(
                use_cot=use_cot,
                cot_small=cot_small,
                base_dir=Path(data_dir) if data_dir else None,
            ))
        elif name == "uk_dale_haystack":
            from src.datasets.uk_dale_haystack.qa_loader import (
                UK_DALE_HAYSTACK_TASKS_DIR,
                get_available_context_lengths,
            )
            base_cfg = extra.get("base_dir")
            base = Path(base_cfg) if base_cfg else Path(UK_DALE_HAYSTACK_TASKS_DIR)
            resolved = list(get_available_context_lengths(base_dir=base))
        else:
            raise ValueError(
                f"Don't know how to resolve 'all' context lengths for dataset {name!r}. "
                f"Add handling in resolve_context_lengths()."
            )
    else:
        resolved = list(ctxs)

    filtered = _apply_min_ctx(resolved, min_ctx)
    if min_ctx is not None and len(filtered) < len(resolved):
        dropped = [c for c in resolved if c not in filtered]
        print(f"  [orchestrator] --min-ctx {min_ctx}: dropping bins {dropped}")
    return filtered


def drop_largest(ctxs: list) -> tuple[list, Any]:
    """Return (new list without the largest entry, the dropped entry)."""
    if not ctxs:
        return [], None
    sorted_ctxs = sorted(ctxs, key=_ctx_sort_key)
    largest = sorted_ctxs[-1]
    new = [c for c in ctxs if c != largest]
    return new, largest


# ---------------------------------------------------------------------------
# Subprocess launch with OOM detection
# ---------------------------------------------------------------------------

def _signal_group(proc: subprocess.Popen, sig: int) -> None:
    """Send ``sig`` to the subprocess's whole process group.

    ``Popen(..., start_new_session=True)`` puts the child in its own
    session/group so ``uv run``'s ``python main.py`` grandchild is also
    a group member. Without the group kill, SIGTERM only hits ``uv run``
    and the grandchild orphans — it keeps the GPU pinned *and* keeps the
    orchestrator's stdout/stderr pipes open, which blocks the pump
    threads on join() and deadlocks the orchestrator.
    """
    try:
        pgid = os.getpgid(proc.pid)
    except ProcessLookupError:
        return
    try:
        os.killpg(pgid, sig)
    except ProcessLookupError:
        pass


def launch_training(cfg_path: Path, log_path: Path) -> tuple[int, bool]:
    """Run training as a subprocess. Streams stdout+stderr to console AND a log
    file. Returns (returncode, oom_detected)."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["uv", "run", "python", "main.py", "--config", str(cfg_path)]
    print(f"  [orchestrator] launching: {' '.join(cmd)}")
    print(f"  [orchestrator] log: {log_path}")

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
        cwd=str(REPO_ROOT),
        env=os.environ.copy(),
        start_new_session=True,
    )

    oom_flag = {"value": False}
    killed_flag = {"value": False}

    def pump(src, dst_console, log_handle, scan: bool) -> None:
        for line in iter(src.readline, ""):
            log_handle.write(line)
            log_handle.flush()
            dst_console.write(line)
            dst_console.flush()
            if scan and any(p in line for p in OOM_PATTERNS):
                if not oom_flag["value"]:
                    oom_flag["value"] = True
                    # train.py swallows batch-level OOMs and keeps going, which
                    # would mask the failure and prevent the orchestrator's
                    # fallback from firing. Kill the whole process group on
                    # first OOM so uv's python grandchild dies too.
                    print(
                        "\n  [orchestrator] OOM detected in subprocess output, "
                        "terminating process group so fallback can take over.",
                        file=dst_console, flush=True,
                    )
                    _signal_group(proc, signal.SIGTERM)
                    killed_flag["value"] = True
        src.close()

    with log_path.open("w") as log_handle:
        t_out = threading.Thread(target=pump, args=(proc.stdout, sys.stdout, log_handle, True))
        t_err = threading.Thread(target=pump, args=(proc.stderr, sys.stderr, log_handle, True))
        t_out.start()
        t_err.start()
        try:
            # If the OOM-watcher terminates the group but it doesn't exit
            # within 30s, escalate to SIGKILL on the whole group.
            while True:
                try:
                    proc.wait(timeout=30)
                    break
                except subprocess.TimeoutExpired:
                    if killed_flag["value"]:
                        _signal_group(proc, signal.SIGKILL)
        except KeyboardInterrupt:
            _signal_group(proc, signal.SIGTERM)
            try:
                proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                _signal_group(proc, signal.SIGKILL)
            raise
        t_out.join()
        t_err.join()

    return proc.returncode, oom_flag["value"]


# ---------------------------------------------------------------------------
# Reduced-config writer
# ---------------------------------------------------------------------------

def write_reduced_config(
    base_cfg_path: Path,
    attempt_idx: int,
    overrides: dict,
) -> tuple[Path, dict]:
    cfg = load_yaml(base_cfg_path)
    for dotted, value in overrides.items():
        set_nested(cfg, dotted, value)
    out_path = REDUCED_RUNS_DIR / f"{base_cfg_path.stem}.attempt_{attempt_idx}.yaml"
    dump_yaml(cfg, out_path)
    return out_path, cfg


# ---------------------------------------------------------------------------
# Post-training eval
# ---------------------------------------------------------------------------

def run_eval(cfg: dict) -> Optional[Path]:
    run_dir = expected_result_dir(cfg)
    logs = run_dir / "output_logs"
    tests = sorted(logs.glob("test_epoch_*.json"))
    if not tests:
        return None
    test_log = tests[-1]
    summary = run_dir / "metrics_summary.json"
    cmd = [
        "uv", "run", "python", "scripts/eval/eval_haystack_log.py",
        str(test_log), "--save-json", str(summary),
    ]
    print(f"  [orchestrator] eval: {' '.join(cmd)}")
    rc = subprocess.call(cmd, cwd=str(REPO_ROOT))
    if rc != 0 or not summary.exists():
        print(f"  [orchestrator] eval failed rc={rc}")
        return None
    return summary


# ---------------------------------------------------------------------------
# Per-config runner with OOM fallback
# ---------------------------------------------------------------------------

def run_one(cfg_path: Path, min_ctx: Optional[float] = None) -> dict:
    base_cfg = load_yaml(cfg_path)
    if is_completed(base_cfg):
        return {
            "status": "skipped",
            "reason": "test_epoch log already present",
            "result_dir": str(expected_result_dir(base_cfg)),
            "attempts": [],
        }

    current_cfg_path = cfg_path
    current_cfg = base_cfg
    resolved_ctxs = resolve_context_lengths(current_cfg, min_ctx=min_ctx)
    if not resolved_ctxs:
        return {
            "status": "failed",
            "reason": f"--min-ctx {min_ctx} left no eligible bins",
            "result_dir": str(expected_result_dir(base_cfg)),
            "attempts": [],
        }
    current_bs = int(current_cfg.get("training", {}).get("batch_size", 2))
    attempts: list[dict] = []

    result_dir = expected_result_dir(base_cfg)

    for attempt_idx in range(1, 32):  # hard cap on retries
        print(f"\n---- attempt {attempt_idx}: bs={current_bs}, ctxs={resolved_ctxs}")
        log_path = result_dir / f"orchestrator_attempt_{attempt_idx}.log"
        rc, oom = launch_training(current_cfg_path, log_path)

        attempts.append({
            "attempt": attempt_idx,
            "config": str(current_cfg_path.relative_to(REPO_ROOT)),
            "batch_size": current_bs,
            "context_lengths": [str(c) for c in resolved_ctxs],
            "returncode": rc,
            "oom": oom,
            "log": str(log_path.relative_to(REPO_ROOT)),
        })

        if rc == 0 and is_completed(current_cfg):
            return {
                "status": "done",
                "result_dir": str(result_dir),
                "attempts": attempts,
                "final_batch_size": current_bs,
                "final_context_lengths": [str(c) for c in resolved_ctxs],
            }

        if not oom:
            return {
                "status": "failed",
                "reason": f"non-OOM exit rc={rc}",
                "result_dir": str(result_dir),
                "attempts": attempts,
            }

        # --- OOM: decide the next fallback --------------------------------
        wipe_checkpoints(result_dir)

        if current_bs > 1:
            current_bs = 1
            current_cfg_path, current_cfg = write_reduced_config(
                cfg_path, attempt_idx,
                {
                    "training.batch_size": current_bs,
                    "dataset.extra_kwargs.context_lengths_seconds": list(resolved_ctxs),
                },
            )
            print(f"  [orchestrator] OOM -> dropping batch_size to 1")
            continue

        new_ctxs, dropped = drop_largest(resolved_ctxs)
        new_ctxs = _apply_min_ctx(new_ctxs, min_ctx)
        if not new_ctxs:
            floor = f" (floor --min-ctx={min_ctx})" if min_ctx is not None else ""
            return {
                "status": "failed",
                "reason": f"OOM even at bs=1 and smallest allowed context{floor}",
                "result_dir": str(result_dir),
                "attempts": attempts,
            }
        resolved_ctxs = new_ctxs
        current_cfg_path, current_cfg = write_reduced_config(
            cfg_path, attempt_idx,
            {
                "training.batch_size": current_bs,
                "dataset.extra_kwargs.context_lengths_seconds": list(resolved_ctxs),
            },
        )
        print(f"  [orchestrator] OOM -> dropping ctx {dropped!r}, now {resolved_ctxs}")

    return {
        "status": "failed",
        "reason": "exceeded retry cap",
        "result_dir": str(result_dir),
        "attempts": attempts,
    }


# ---------------------------------------------------------------------------
# Summary aggregation
# ---------------------------------------------------------------------------

def aggregate_summary(manifest: dict) -> None:
    rows = []
    for stem, info in manifest.items():
        result_dir = Path(info.get("result_dir") or "")
        summary_path = result_dir / "metrics_summary.json"
        if not summary_path.exists():
            continue
        with summary_path.open() as f:
            m = json.load(f)
        acc = (m.get("accuracy") or {}).get("overall")
        f1 = (m.get("macro_f1_discrete") or {}).get("overall")
        iou = (m.get("mean_iou_time") or {}).get("overall")
        te = (m.get("mean_timestamp_error_s") or {}).get("overall")
        rows.append({
            "config": stem,
            "run": m.get("run"),
            "dataset": m.get("dataset_name"),
            "display_name": m.get("display_name"),
            "split": m.get("split"),
            "n_samples": m.get("n_samples"),
            "accuracy_overall": acc,
            "macro_f1_overall": f1,
            "mean_iou_overall": iou,
            "mean_timestamp_error_s_overall": te,
            "status": info.get("status"),
        })
    if not rows:
        print("  [orchestrator] no per-run summaries found; skipping CSV.")
        return
    SUMMARY_CSV.parent.mkdir(parents=True, exist_ok=True)
    with SUMMARY_CSV.open("w") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"  [orchestrator] wrote {SUMMARY_CSV} with {len(rows)} rows")


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------

def load_manifest() -> dict:
    if MANIFEST_PATH.exists():
        try:
            return json.loads(MANIFEST_PATH.read_text())
        except json.JSONDecodeError:
            return {}
    return {}


def save_manifest(manifest: dict) -> None:
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2, default=str))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true", help="Print plan, do nothing.")
    ap.add_argument(
        "--only",
        nargs="+",
        default=None,
        help="Keep configs whose stem contains ANY of these substrings "
             "(e.g. '--only ltaf_haystack capture24_haystack_cot').",
    )
    ap.add_argument(
        "--exclude",
        nargs="+",
        default=None,
        help="Drop configs whose stem contains ANY of these substrings "
             "(applied after --only).",
    )
    ap.add_argument("--force", help="Stem name to force re-run (wipes its result dir).")
    ap.add_argument("--no-eval", action="store_true", help="Skip post-training eval and CSV aggregation.")
    ap.add_argument(
        "--min-ctx",
        type=float,
        default=10,
        help="Paper-comparability floor (seconds). Bins below this are "
             "removed from the initial list and the OOM ctx-drop fallback "
             "will give up rather than go below. E.g. '--min-ctx 10' keeps "
             "capture24_haystack_cot's 2.56s out of the training mix.",
    )
    args = ap.parse_args()

    configs = sorted(PAPER_CONFIGS_DIR.glob("*.yaml"), key=_config_sort_key)
    if args.only:
        # Bucket by first-matching --only token to honor caller-specified
        # ordering (e.g. --only flamingo itformer chatts runs in that order).
        # Within a bucket, configs stay alphabetical.
        def _bucket(cfg):
            for i, tok in enumerate(args.only):
                if tok in cfg.stem:
                    return i
            return None
        configs = [(c, _bucket(c)) for c in configs]
        configs = [c for c, b in sorted(
            ((c, b) for c, b in configs if b is not None),
            key=lambda cb: (cb[1], cb[0].name),
        )]
    if args.exclude:
        configs = [c for c in configs if not any(tok in c.stem for tok in args.exclude)]
    if not configs:
        print(f"No configs matched under {PAPER_CONFIGS_DIR}")
        return

    manifest = load_manifest()

    if args.force:
        victims = [c for c in configs if c.stem == args.force]
        if not victims:
            print(f"--force {args.force!r} matched no configs.")
            sys.exit(2)
        for c in victims:
            cfg = load_yaml(c)
            rd = expected_result_dir(cfg)
            if rd.exists():
                print(f"[orchestrator] force wipe: {rd}")
                shutil.rmtree(rd)
            manifest.pop(c.stem, None)
        save_manifest(manifest)

    if args.dry_run:
        floor_note = f"  (--min-ctx {args.min_ctx})" if args.min_ctx is not None else ""
        print(f"\nOrchestrator plan — {len(configs)} configs under {PAPER_CONFIGS_DIR}{floor_note}:")
        for c in configs:
            cfg = load_yaml(c)
            rd = expected_result_dir(cfg)
            status = "DONE   (skip)" if is_completed(cfg) else "MISSING (run)"
            extra = ""
            if not is_completed(cfg):
                try:
                    ctxs = resolve_context_lengths(cfg, min_ctx=args.min_ctx)
                    extra = f"  ctxs={ctxs}"
                except Exception:  # noqa: BLE001
                    pass
            print(f"  {status}  {c.stem:<40}  -> {rd.relative_to(REPO_ROOT)}{extra}")
        return

    REDUCED_RUNS_DIR.mkdir(parents=True, exist_ok=True)

    for cfg_path in configs:
        stem = cfg_path.stem
        print(f"\n{'#' * 72}\n# {stem}\n{'#' * 72}")
        t0 = time.time()
        try:
            info = run_one(cfg_path, min_ctx=args.min_ctx)
        except KeyboardInterrupt:
            print("\n[orchestrator] interrupted — saving manifest and exiting.")
            save_manifest(manifest)
            raise
        except Exception as e:  # noqa: BLE001 — orchestrator catches so it can keep going
            info = {"status": "failed", "reason": f"orchestrator error: {e!r}", "attempts": []}
        info["wall_seconds"] = round(time.time() - t0, 1)
        manifest[stem] = info
        save_manifest(manifest)

        if info["status"] == "done" and not args.no_eval:
            cfg = load_yaml(cfg_path)
            summary_path = run_eval(cfg)
            if summary_path:
                info["metrics_summary"] = str(summary_path.relative_to(REPO_ROOT))
                manifest[stem] = info
                save_manifest(manifest)

        print(f"\n[orchestrator] {stem}: {info['status']} in {info['wall_seconds']}s")

    if not args.no_eval:
        aggregate_summary(manifest)


if __name__ == "__main__":
    main()
