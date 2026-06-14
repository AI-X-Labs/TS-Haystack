# SPDX-License-Identifier: CC-BY-NC-4.0
"""
Evaluate a haystack-family validation/test epoch log.

Reads ``results/<run>/output_logs/<split>_epoch_<N>.json`` and prints:
  * Accuracy per (task, context length) with marginals.
  * Macro-F1 per (task, context length) for discrete answer types
    (boolean / integer / category). Time-range / timestamp tasks get
    Mean IoU instead (no F1).

Breakdown layout: **rows = task, columns = context length.**

Supports the haystack-style datasets registered in
``src.datasets.registry``:

  * ``sleep_psg_haystack``
  * ``ltaf_haystack``, ``ltaf_haystack_cot``
  * ``capture24_haystack_classification``, ``capture24_haystack_cot``
  * ``uk_dale_haystack``

Context length and task are recovered from ``sample_idx`` by mirroring
the shard discovery loop of each dataset's loader against the live
parquet tree referenced by the run's ``config.yaml``.

Usage:
    python scripts/eval/eval_haystack_log.py \\
        results/<run>/output_logs/test_epoch_0.json
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import pyarrow.parquet as pq
import yaml

from src.datasets.constants import RAW_DATA
from src.datasets.ltaf_haystack.qa_loader import (
    LTAF_HAYSTACK_COT_DIR,
    LTAF_HAYSTACK_TASKS_DIR,
    _context_dir_name as _ltaf_ctx_dir_name,
    get_available_context_lengths as _ltaf_ctxs_on_disk,
    get_available_tasks as _ltaf_tasks_on_disk,
)
from src.datasets.sleep_psg_haystack.qa_loader import (
    ALL_CONTEXT_LENGTHS_PER_LABEL as SLEEP_CTXS_PER_LABEL,
    ALL_TASKS as SLEEP_ALL_TASKS,
    BASE_DIR as SLEEP_BASE_DIR,
)
from src.datasets.capture24_haystack.qa_loader import (
    ALL_TASKS as TSH_ALL_TASKS,
    TS_HAYSTACK_COT_DIR,
    TS_HAYSTACK_COT_SMALL_DIR,
    TS_HAYSTACK_TASKS_DIR,
    get_available_context_lengths as _tsh_ctxs_on_disk,
)
from src.datasets.uk_dale_haystack.qa_loader import (
    UK_DALE_HAYSTACK_TASKS_DIR,
    _context_dir_name as _ukd_ctx_dir_name,
    _split_dir_candidates as _ukd_split_dir_candidates,
    get_available_context_lengths as _ukd_ctxs_on_disk,
    get_available_tasks as _ukd_tasks_on_disk,
)
from src.datasets.capture24_haystack.utils import format_context_dir
from src.datasets.capture24_haystack.utils.answer_evaluation import (
    TIME_PATTERN,
    parse_time_range,
)
from src.datasets.capture24_haystack.utils.timestamp_utils import parse_time_string


SPLIT_DIR_MAP = {"val": "val", "validation": "val", "train": "train", "test": "test"}
SPLIT_DIR_CANDIDATES = {
    "val": ["val", "validation"],
    "train": ["train"],
    "test": ["test"],
}
DISCRETE_TYPES = {"boolean", "integer", "category"}
TIME_TYPES = {"time_range", "timestamp"}


# ---------------------------------------------------------------------------
# Shard representation
# ---------------------------------------------------------------------------

@dataclass
class Shard:
    ctx: Any
    ctx_label: str
    task: str
    n_rows: int
    answer_type: str
    path: Path


def _ctx_label(ctx: Any) -> str:
    if isinstance(ctx, str):
        return ctx
    return format_context_dir(ctx)


def _read_shard_meta(path: Path, ctx: Any, task: str) -> Optional[Shard]:
    """Return a Shard for an existing, non-empty parquet; None otherwise."""
    if not path.exists():
        return None
    meta = pq.read_metadata(path)
    n_rows = int(meta.num_rows)
    if n_rows <= 0:
        return None
    answer_type = pq.read_table(path, columns=["answer_type"]).column(0)[0].as_py()
    return Shard(
        ctx=ctx,
        ctx_label=_ctx_label(ctx),
        task=task,
        n_rows=n_rows,
        answer_type=answer_type,
        path=path,
    )


def _read_shard_meta_filtered_valid(path: Path, ctx: Any, task: str) -> Optional[Shard]:
    """Like _read_shard_meta but counts only rows with ``is_valid=True``.

    UK-DALE-Haystack's runtime loader drops generator-rejected rows
    (placeholder background metadata) via ``ds.filter(is_valid)``, so the
    sample count seen by the model is < parquet metadata num_rows whenever
    a shard contains invalid rows.
    """
    if not path.exists():
        return None
    meta = pq.read_metadata(path)
    if int(meta.num_rows) <= 0:
        return None
    schema_names = set(meta.schema.names)
    if "is_valid" in schema_names:
        valid_col = pq.read_table(path, columns=["is_valid"]).column(0).to_pylist()
        n_rows = sum(1 for v in valid_col if v)
    else:
        n_rows = int(meta.num_rows)
    if n_rows <= 0:
        return None
    answer_type = pq.read_table(path, columns=["answer_type"]).column(0)[0].as_py()
    return Shard(
        ctx=ctx,
        ctx_label=_ctx_label(ctx),
        task=task,
        n_rows=n_rows,
        answer_type=answer_type,
        path=path,
    )


# ---------------------------------------------------------------------------
# Per-dataset shard discovery (mirrors each loader's iteration order)
# ---------------------------------------------------------------------------

def _resolve_extra(cfg: dict) -> dict:
    return (cfg.get("dataset") or {}).get("extra_kwargs") or {}


def _discover_sleep_psg(
    cfg: dict,
    data_dir_override: Optional[Path],
    split: str,
) -> Tuple[List[Shard], Dict[str, Any]]:
    extra = _resolve_extra(cfg)
    label_class = extra.get("label_class")
    if label_class is None:
        raise ValueError("config.yaml missing dataset.extra_kwargs.label_class")

    tasks_req = extra.get("tasks", ["all"])
    ctxs_req = extra.get("context_lengths_seconds", ["all"])

    if "all" in [str(t) for t in tasks_req]:
        tasks = list(SLEEP_ALL_TASKS)
    else:
        tasks = list(tasks_req)

    if "all" in [str(c) for c in ctxs_req]:
        ctxs = list(SLEEP_CTXS_PER_LABEL[label_class])
    else:
        ctxs = list(ctxs_req)

    data_root = Path(data_dir_override) if data_dir_override else Path(
        cfg.get("dataset", {}).get("data_dir", "data")
    )
    base_dir = data_root / SLEEP_BASE_DIR.relative_to("data") / label_class / "tasks"

    shards: List[Shard] = []
    for ctx in ctxs:
        for task in tasks:
            p = base_dir / _ctx_label(ctx) / task / split / "data.parquet"
            s = _read_shard_meta(p, ctx, task)
            if s is not None:
                shards.append(s)

    info = {
        "label_class": label_class,
        "base_dir": base_dir,
        "display_name": f"sleep_psg_haystack [{label_class}]",
    }
    return shards, info


def _discover_ltaf(
    cfg: dict,
    data_dir_override: Optional[Path],
    split: str,
    use_cot: bool,
) -> Tuple[List[Shard], Dict[str, Any]]:
    extra = _resolve_extra(cfg)
    tasks_req = extra.get("tasks", ["all"])
    ctxs_req = extra.get("context_lengths_seconds", ["all"])
    base_dir_cfg = extra.get("base_dir")

    if data_dir_override:
        base_dir = Path(data_dir_override)
    elif base_dir_cfg:
        base_dir = Path(base_dir_cfg)
    else:
        base_dir = Path(LTAF_HAYSTACK_COT_DIR if use_cot else LTAF_HAYSTACK_TASKS_DIR)

    if tasks_req is None or (isinstance(tasks_req, list) and ("all" in [str(t) for t in tasks_req] or len(tasks_req) == 0)):
        tasks = _ltaf_tasks_on_disk(base_dir)
    else:
        # LTAF loader sorts tasks explicitly.
        tasks = sorted({str(t) for t in tasks_req if str(t).strip()})

    if ctxs_req is None or (isinstance(ctxs_req, list) and ("all" in [str(c) for c in ctxs_req] or len(ctxs_req) == 0)):
        ctxs = _ltaf_ctxs_on_disk(base_dir)
    else:
        ctxs = list(ctxs_req)

    shards: List[Shard] = []
    for ctx in ctxs:
        # LTAF's on-disk layout always uses the float-form ("100_0s"), even
        # for integer-equivalent values. Use LTAF's own helper for the path
        # so the two conventions stay in sync.
        ctx_dir = _ltaf_ctx_dir_name(float(ctx)) if not isinstance(ctx, str) else ctx
        for task in tasks:
            # Match LTAF's val/validation fallback: append any that exist.
            for split_dir in SPLIT_DIR_CANDIDATES.get(split, [split]):
                p = base_dir / ctx_dir / task / split_dir / "data.parquet"
                s = _read_shard_meta(p, ctx, task)
                if s is not None:
                    shards.append(s)

    display = "ltaf_haystack_cot" if use_cot else "ltaf_haystack"
    info = {"base_dir": base_dir, "display_name": display}
    return shards, info


def _discover_ts_haystack(
    cfg: dict,
    data_dir_override: Optional[Path],
    split: str,
    use_cot: bool,
    cot_small: bool,
) -> Tuple[List[Shard], Dict[str, Any]]:
    extra = _resolve_extra(cfg)
    tasks_req = extra.get("tasks", ["all"])
    ctxs_req = extra.get("context_lengths_seconds", ["all"])

    if data_dir_override:
        base_dir = Path(data_dir_override)
    elif cfg.get("dataset", {}).get("data_dir"):
        base_dir = Path(cfg["dataset"]["data_dir"])
    elif use_cot and cot_small:
        base_dir = Path(TS_HAYSTACK_COT_SMALL_DIR)
    elif use_cot:
        base_dir = Path(TS_HAYSTACK_COT_DIR)
    else:
        base_dir = Path(TS_HAYSTACK_TASKS_DIR)

    if "all" in [str(t) for t in tasks_req]:
        tasks = list(TSH_ALL_TASKS)
    else:
        tasks = list(tasks_req)

    if "all" in [str(c) for c in ctxs_req]:
        ctxs = _tsh_ctxs_on_disk(use_cot=use_cot, base_dir=base_dir, cot_small=cot_small)
    else:
        ctxs = list(ctxs_req)

    shards: List[Shard] = []
    for ctx in ctxs:
        for task in tasks:
            p = base_dir / _ctx_label(ctx) / task / split / "data.parquet"
            s = _read_shard_meta(p, ctx, task)
            if s is not None:
                shards.append(s)

    name = cfg.get("dataset", {}).get("name", "ts_haystack")
    info = {"base_dir": base_dir, "display_name": name}
    return shards, info


def _discover_uk_dale_haystack(
    cfg: dict,
    data_dir_override: Optional[Path],
    split: str,
) -> Tuple[List[Shard], Dict[str, Any]]:
    extra = _resolve_extra(cfg)
    tasks_req = extra.get("tasks", ["all"])
    ctxs_req = extra.get("context_lengths_seconds", ["all"])
    base_dir_cfg = extra.get("base_dir")

    if data_dir_override:
        base_dir = Path(data_dir_override)
    elif base_dir_cfg:
        base_dir = Path(base_dir_cfg)
    else:
        base_dir = Path(UK_DALE_HAYSTACK_TASKS_DIR)

    if tasks_req is None or (
        isinstance(tasks_req, list)
        and ("all" in [str(t) for t in tasks_req] or len(tasks_req) == 0)
    ):
        tasks = _ukd_tasks_on_disk(base_dir)
    else:
        # qa_loader sorts the tasks list before iterating (selected_tasks).
        tasks = sorted({str(t) for t in tasks_req if str(t).strip()})

    if ctxs_req is None or (
        isinstance(ctxs_req, list)
        and ("all" in [str(c) for c in ctxs_req] or len(ctxs_req) == 0)
    ):
        ctxs = _ukd_ctxs_on_disk(base_dir)
    else:
        ctxs = list(ctxs_req)

    shards: List[Shard] = []
    for ctx in ctxs:
        ctx_dir = (
            ctx
            if isinstance(ctx, str)
            else _ukd_ctx_dir_name(float(ctx))
        )
        for task in tasks:
            for split_dir in _ukd_split_dir_candidates(split):
                p = base_dir / ctx_dir / task / split_dir / "data.parquet"
                s = _read_shard_meta_filtered_valid(p, ctx, task)
                if s is not None:
                    shards.append(s)

    info = {"base_dir": base_dir, "display_name": "uk_dale_haystack"}
    return shards, info


DiscoverFn = Callable[[dict, Optional[Path], str], Tuple[List[Shard], Dict[str, Any]]]


def _get_discovery(cfg: dict) -> DiscoverFn:
    name = (cfg.get("dataset") or {}).get("name")
    if name == "sleep_psg_haystack":
        return _discover_sleep_psg
    if name == "ltaf_haystack":
        return lambda c, d, s: _discover_ltaf(c, d, s, use_cot=False)
    if name == "ltaf_haystack_cot":
        return lambda c, d, s: _discover_ltaf(c, d, s, use_cot=True)
    if name == "capture24_haystack_classification":
        return lambda c, d, s: _discover_ts_haystack(c, d, s, use_cot=False, cot_small=False)
    if name == "capture24_haystack_cot":
        extra = _resolve_extra(cfg)
        cot_small = bool(extra.get("cot_small", False))
        return lambda c, d, s: _discover_ts_haystack(c, d, s, use_cot=True, cot_small=cot_small)
    if name == "uk_dale_haystack":
        return _discover_uk_dale_haystack
    raise ValueError(
        f"Unsupported dataset.name {name!r}. This script handles haystack-style "
        f"datasets (sleep_psg_haystack, ltaf_haystack[_cot], ts_haystack_*, "
        f"uk_dale_haystack)."
    )


# ---------------------------------------------------------------------------
# sample_idx -> shard assignment
# ---------------------------------------------------------------------------

def assign_shard_per_sample(outputs: List[dict], shards: List[Shard]) -> List[Dict]:
    total = sum(s.n_rows for s in shards)
    if len(outputs) != total:
        raise ValueError(
            f"Row-count mismatch: log has {len(outputs)} samples but discovered "
            f"shards total {total}. Was runtime.max_samples set, or have shards "
            f"changed since training? Cannot recover ctx/task mapping."
        )
    bounds = []
    cum = 0
    for s in shards:
        bounds.append((cum, cum + s.n_rows, s))
        cum += s.n_rows

    annotated = []
    for out in outputs:
        idx = out["sample_idx"]
        shard = None
        for lo, hi, s in bounds:
            if lo <= idx < hi:
                shard = s
                break
        if shard is None:
            raise IndexError(f"sample_idx {idx} not covered by shard layout")
        if out.get("category") and out["category"] != shard.task:
            raise ValueError(
                f"Shard mismatch at sample_idx={idx}: log category="
                f"{out['category']!r} but shard task={shard.task!r} "
                f"(ctx={shard.ctx_label}). Mapping is unreliable."
            )
        annotated.append({
            **out,
            "ctx": shard.ctx,
            "ctx_label": shard.ctx_label,
            "task": shard.task,
            "answer_type": shard.answer_type,
        })
    return annotated


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

_SPECIAL_TOKEN_RE = re.compile(r"<\|[^|>]*\|>")


def _norm_discrete(s: str) -> str:
    # Some runtimes (e.g. Flamingo) leave HF special-token markers like
    # ``<|endofchunk|>`` on the gold normalized_gt, which silently breaks
    # exact-string F1. Strip them here so F1 lines up with `correct`.
    s = _SPECIAL_TOKEN_RE.sub("", s)
    s = s.strip().lower()
    return re.sub(r"[.,;:!?]+$", "", s)


def classification_macro_f1(samples: List[dict]) -> float:
    if not samples:
        return float("nan")
    gts = [_norm_discrete(str(s.get("normalized_gt") or s.get("ground_truth", ""))) for s in samples]
    preds = [_norm_discrete(str(s.get("normalized_pred") or s.get("predicted_answer", ""))) for s in samples]
    classes = sorted(set(gts) | set(preds))
    if not classes:
        return float("nan")
    f1s = []
    for c in classes:
        tp = sum(1 for g, p in zip(gts, preds) if g == c and p == c)
        fp = sum(1 for g, p in zip(gts, preds) if g != c and p == c)
        fn = sum(1 for g, p in zip(gts, preds) if g == c and p != c)
        if sum(1 for g in gts if g == c) == 0:
            continue
        denom = 2 * tp + fp + fn
        f1s.append(2 * tp / denom if denom > 0 else 0.0)
    return sum(f1s) / len(f1s) if f1s else float("nan")


def mean_iou(samples: List[dict]) -> Optional[float]:
    """Mean IoU over samples whose GT parses as a time range."""
    if not samples:
        return None
    ious = []
    for s in samples:
        gt_range = parse_time_range(str(s.get("ground_truth", "")))
        if gt_range is None:
            continue  # Not a time_range sample; see mean_timestamp_error_s.
        pred_range = parse_time_range(str(s.get("predicted_answer", "")))
        if pred_range is None:
            pred_range = parse_time_range(str(s.get("prediction_raw", "")))
        if pred_range is None:
            ious.append(0.0)
            continue
        gs, ge = gt_range
        ps, pe = pred_range
        inter = max(0.0, (min(ge, pe) - max(gs, ps)).total_seconds())
        union = (ge - gs).total_seconds() + (pe - ps).total_seconds() - inter
        ious.append(inter / union if union > 0 else 0.0)
    return sum(ious) / len(ious) if ious else None


def _parse_single_timestamp(text: str):
    """Parse the first TIME_PATTERN match in text, or None."""
    if not text:
        return None
    matches = re.findall(TIME_PATTERN, text, re.IGNORECASE)
    if not matches:
        return None
    try:
        return parse_time_string(matches[0])
    except ValueError:
        return None


def mean_timestamp_error_s(samples: List[dict]) -> Optional[float]:
    """Mean absolute seconds error over samples whose GT parses as a single
    timestamp (exactly one TIME_PATTERN match). Predictions that fail to
    parse count as a large error (using the largest observed GT–pred gap
    across the batch) so a missed localization is not silently dropped."""
    if not samples:
        return None
    errors: List[float] = []
    missed: List[dict] = []
    for s in samples:
        gt_text = str(s.get("ground_truth", ""))
        gt_matches = re.findall(TIME_PATTERN, gt_text, re.IGNORECASE)
        if len(gt_matches) != 1:
            continue
        gt_dt = _parse_single_timestamp(gt_text)
        if gt_dt is None:
            continue
        pred_dt = _parse_single_timestamp(str(s.get("predicted_answer", "")))
        if pred_dt is None:
            pred_dt = _parse_single_timestamp(str(s.get("prediction_raw", "")))
        if pred_dt is None:
            missed.append(s)
            continue
        errors.append(abs((pred_dt - gt_dt).total_seconds()))
    # Charge missed predictions a sentinel equal to the max observed error
    # (or 0 if no parseable predictions; then "missed" dominates the mean).
    if missed:
        penalty = max(errors) if errors else 0.0
        errors.extend([penalty] * len(missed))
    return sum(errors) / len(errors) if errors else None


# ---------------------------------------------------------------------------
# Table printing (rows=task, cols=ctx)
# ---------------------------------------------------------------------------

def _fmt_pct(x: Optional[float]) -> str:
    if x is None or (isinstance(x, float) and x != x):
        return "  —  "
    return f"{x*100:5.1f}"


def _fmt_val(x: Optional[float]) -> str:
    if x is None or (isinstance(x, float) and x != x):
        return "  —  "
    return f"{x:.3f}"


def print_header(title: str) -> None:
    print()
    print(title)
    print("-" * len(title))


def print_grid(
    row_keys: List[str],
    col_keys: List[str],
    cells: Dict[Tuple[str, str], str],
    row_marginals: Dict[str, str],
    col_marginals: Dict[str, str],
    overall: str,
    row_label: str = "task",
) -> None:
    col_w = max(12, max((len(c) for c in col_keys), default=6) + 2)
    row_w = max(len(row_label) + 2, max((len(r) for r in row_keys), default=4) + 2)

    header = f"{row_label:<{row_w}}" + "".join(f"{c:>{col_w}}" for c in col_keys) + f"{'all':>{col_w}}"
    print(header)
    print("-" * len(header))
    for r in row_keys:
        line = f"{r:<{row_w}}"
        for c in col_keys:
            line += f"{cells.get((r, c), '  —  '):>{col_w}}"
        line += f"{row_marginals.get(r, '  —  '):>{col_w}}"
        print(line)
    print("-" * len(header))
    foot = f"{'all':<{row_w}}"
    for c in col_keys:
        foot += f"{col_marginals.get(c, '  —  '):>{col_w}}"
    foot += f"{overall:>{col_w}}"
    print(foot)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def resolve_config(log_path: Path) -> dict:
    run_dir = log_path.parent.parent
    cfg_path = run_dir / "config.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(f"No config.yaml next to {run_dir}")
    with cfg_path.open() as f:
        return yaml.safe_load(f)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("log_file", type=Path, help="Path to <split>_epoch_N.json")
    ap.add_argument("--data-dir", type=Path, default=None,
                    help="Override dataset base directory (default: infer from config.yaml)")
    ap.add_argument("--save-json", type=Path, default=None,
                    help="Optional: also write aggregated metrics to JSON")
    args = ap.parse_args()

    log_path = args.log_file.resolve()
    with log_path.open() as f:
        log = json.load(f)

    cfg = resolve_config(log_path)

    if (cfg.get("runtime") or {}).get("max_samples") is not None:
        raise RuntimeError(
            "runtime.max_samples was set for this run; subset is random and "
            "sample_idx cannot be mapped deterministically."
        )

    split_name = log.get("split", "val")
    split_dir = SPLIT_DIR_MAP.get(split_name, split_name)

    discover = _get_discovery(cfg)
    shards, ds_info = discover(cfg, args.data_dir, split_dir)
    if not shards:
        raise RuntimeError(
            f"No shards discovered for split={split_dir} under {ds_info['base_dir']}"
        )
    samples = assign_shard_per_sample(log["outputs"], shards)

    # Preserve loader order for ctx and task while restricting to what actually
    # contributed samples in this split.
    ctx_order: List[str] = []
    task_order: List[str] = []
    task_answer_type: Dict[str, str] = {}
    for s in shards:
        if s.ctx_label not in ctx_order:
            ctx_order.append(s.ctx_label)
        if s.task not in task_order:
            task_order.append(s.task)
            task_answer_type[s.task] = s.answer_type

    # ------------------------------------------------------------------
    # Accuracy per (task, ctx) + marginals
    # ------------------------------------------------------------------
    correct = defaultdict(lambda: [0, 0])  # (task, ctx) -> [correct, total]
    for s in samples:
        key = (s["task"], s["ctx_label"])
        correct[key][1] += 1
        if s.get("correct"):
            correct[key][0] += 1

    cells_acc = {k: _fmt_pct(v[0] / v[1]) for k, v in correct.items() if v[1]}
    row_acc = {
        t: _fmt_pct(
            sum(v[0] for k, v in correct.items() if k[0] == t)
            / max(sum(v[1] for k, v in correct.items() if k[0] == t), 1)
        )
        for t in task_order
    }
    col_acc = {
        c: _fmt_pct(
            sum(v[0] for k, v in correct.items() if k[1] == c)
            / max(sum(v[1] for k, v in correct.items() if k[1] == c), 1)
        )
        for c in ctx_order
    }
    total_c = sum(v[0] for v in correct.values())
    total_n = sum(v[1] for v in correct.values())
    overall_acc = _fmt_pct(total_c / total_n) if total_n else "  —  "

    # ------------------------------------------------------------------
    # Macro-F1 per (task, ctx) for discrete tasks only
    # ------------------------------------------------------------------
    discrete_tasks = [t for t in task_order if task_answer_type[t] in DISCRETE_TYPES]
    time_tasks = [t for t in task_order if task_answer_type[t] in TIME_TYPES]

    groups: Dict[Tuple[str, str], List[dict]] = defaultdict(list)
    for s in samples:
        groups[(s["task"], s["ctx_label"])].append(s)

    cells_f1: Dict[Tuple[str, str], str] = {}
    for (t, c), grp in groups.items():
        if t in discrete_tasks:
            cells_f1[(t, c)] = _fmt_val(classification_macro_f1(grp))

    row_f1: Dict[str, str] = {}
    for t in discrete_tasks:
        pool = [s for s in samples if s["task"] == t]
        row_f1[t] = _fmt_val(classification_macro_f1(pool))

    col_f1: Dict[str, str] = {}
    for c in ctx_order:
        vals = []
        for t in discrete_tasks:
            grp = groups.get((t, c))
            if not grp:
                continue
            f1 = classification_macro_f1(grp)
            if f1 == f1:
                vals.append(f1)
        col_f1[c] = _fmt_val(sum(vals) / len(vals)) if vals else "  —  "

    all_f1s = []
    for t in discrete_tasks:
        pool = [s for s in samples if s["task"] == t]
        f1 = classification_macro_f1(pool)
        if f1 == f1:
            all_f1s.append(f1)
    overall_f1 = _fmt_val(sum(all_f1s) / len(all_f1s)) if all_f1s else "  —  "

    # ------------------------------------------------------------------
    # Mean IoU per (task, ctx) for time tasks
    # ------------------------------------------------------------------
    cells_iou: Dict[Tuple[str, str], str] = {}
    for (t, c), grp in groups.items():
        if t in time_tasks:
            iou = mean_iou(grp)
            if iou is not None:
                cells_iou[(t, c)] = _fmt_val(iou)

    row_iou: Dict[str, str] = {}
    for t in time_tasks:
        pool = [s for s in samples if s["task"] == t]
        iou = mean_iou(pool)
        if iou is not None:
            row_iou[t] = _fmt_val(iou)

    col_iou: Dict[str, str] = {}
    for c in ctx_order:
        pool = [s for s in samples if s["ctx_label"] == c and s["task"] in time_tasks]
        iou = mean_iou(pool)
        if iou is not None:
            col_iou[c] = _fmt_val(iou)

    overall_iou_val = mean_iou([s for s in samples if s["task"] in time_tasks])
    overall_iou = _fmt_val(overall_iou_val) if overall_iou_val is not None else "  —  "

    # ------------------------------------------------------------------
    # Mean timestamp error (seconds) for single-timestamp tasks.
    # Separated from IoU because units differ (seconds vs fraction).
    # ------------------------------------------------------------------
    cells_te: Dict[Tuple[str, str], str] = {}
    for (t, c), grp in groups.items():
        if t in time_tasks:
            te = mean_timestamp_error_s(grp)
            if te is not None:
                cells_te[(t, c)] = _fmt_val(te)

    row_te: Dict[str, str] = {}
    for t in time_tasks:
        pool = [s for s in samples if s["task"] == t]
        te = mean_timestamp_error_s(pool)
        if te is not None:
            row_te[t] = _fmt_val(te)

    col_te: Dict[str, str] = {}
    for c in ctx_order:
        pool = [s for s in samples if s["ctx_label"] == c and s["task"] in time_tasks]
        te = mean_timestamp_error_s(pool)
        if te is not None:
            col_te[c] = _fmt_val(te)

    overall_te_val = mean_timestamp_error_s(
        [s for s in samples if s["task"] in time_tasks]
    )
    overall_te = _fmt_val(overall_te_val) if overall_te_val is not None else "  —  "
    timestamp_tasks = [t for t in time_tasks if t in row_te]

    # ------------------------------------------------------------------
    # Print
    # ------------------------------------------------------------------
    print("=" * 72)
    print(f"Haystack eval — {ds_info['display_name']} — {log_path.name}")
    print(f"  run:          {log_path.parent.parent.name}")
    loss_str = f"{log.get('loss'):.4f}" if log.get("loss") is not None else "—"
    print(f"  split:        {split_name}  |  epoch: {log.get('epoch')}  "
          f"|  loss: {loss_str}  |  samples: {len(samples)}")
    print(f"  ctxs:         {ctx_order}")
    print(f"  tasks:        {task_order}  (time tasks: {time_tasks})")
    print("=" * 72)

    print_header("Accuracy (%) — rows: task, cols: context length")
    print_grid(task_order, ctx_order, cells_acc, row_acc, col_acc, overall_acc)

    if discrete_tasks:
        print_header("Macro-F1 — discrete tasks only (boolean / integer / category)")
        print_grid(discrete_tasks, ctx_order, cells_f1, row_f1, col_f1, overall_f1)

    range_tasks = [t for t in time_tasks if t in row_iou]
    if range_tasks:
        print_header("Mean IoU — time_range tasks only")
        print_grid(range_tasks, ctx_order, cells_iou, row_iou, col_iou, overall_iou)

    if timestamp_tasks:
        print_header("Mean timestamp error (seconds) — single-timestamp tasks only")
        print_grid(
            timestamp_tasks, ctx_order, cells_te, row_te, col_te, overall_te
        )

    if args.save_json:
        out = {
            "log_file": str(log_path),
            "run": log_path.parent.parent.name,
            "dataset_name": (cfg.get("dataset") or {}).get("name"),
            "display_name": ds_info["display_name"],
            "split": split_name,
            "epoch": log.get("epoch"),
            "loss": log.get("loss"),
            "n_samples": len(samples),
            "ctx_order": ctx_order,
            "task_order": task_order,
            "task_answer_type": task_answer_type,
            "accuracy": {
                "overall": total_c / total_n if total_n else None,
                "by_task_ctx": {
                    f"{t}|{c}": (correct[(t, c)][0] / correct[(t, c)][1])
                    for (t, c), v in correct.items() if v[1]
                },
            },
            "macro_f1_discrete": {
                "overall": sum(all_f1s) / len(all_f1s) if all_f1s else None,
                "by_task_ctx": {f"{t}|{c}": cells_f1[(t, c)].strip()
                                for (t, c) in cells_f1},
            },
            "mean_iou_time": {
                "overall": overall_iou_val,
                "by_task_ctx": {f"{t}|{c}": cells_iou[(t, c)].strip()
                                for (t, c) in cells_iou},
            },
            "mean_timestamp_error_s": {
                "overall": overall_te_val,
                "by_task_ctx": {f"{t}|{c}": cells_te[(t, c)].strip()
                                for (t, c) in cells_te},
            },
        }
        args.save_json.parent.mkdir(parents=True, exist_ok=True)
        with args.save_json.open("w") as f:
            json.dump(out, f, indent=2)
        print(f"\nSaved aggregated metrics to {args.save_json}")


if __name__ == "__main__":
    main()
