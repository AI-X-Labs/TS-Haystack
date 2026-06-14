#!/usr/bin/env python3
"""Plot GPT classifier benchmark results vs Flamingo baseline.

Reads summary.json files (recomputing from trajectories first) and produces:
  1. accuracy_by_context.png  — line plot, accuracy vs context length
  2. accuracy_by_task.png     — clustered bar chart, 3 bars × 10 tasks
  3. accuracy_context_x_task.png — 2×5 grid, one subplot per task

Usage:
    python3 scripts/eval/plot_gpt_benchmark.py
    python3 scripts/eval/plot_gpt_benchmark.py --output-dir results/gpt_benchmark/plots
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
RESULTS_ROOT = Path("results/gpt_benchmark")

ORACLE_DIR = RESULTS_ROOT / "gpt-5.4-mini-2026-03-17_oracle"
REAL_DIR = RESULTS_ROOT / "gpt-5.4-mini-2026-03-17_real_dual"

METHODS = {
    "Agentic with oracle class.": {"dir": ORACLE_DIR, "color": "#2196F3", "marker": "o"},
    "Agentic with real class.": {"dir": REAL_DIR, "color": "#FF9800", "marker": "s"},
    "Flamingo": {"dir": None, "color": "#4CAF50", "marker": "^"},
    "ITFormer": {"dir": None, "color": "#9C27B0", "marker": "D"},
}

# Flamingo results from paper (Capture-24 benchmark table)
# Keyed by context length in samples, then by task.
FLAMINGO_BY_CTX_TASK: dict[str, dict[str, float]] = {
    "256": {
        "existence": 57.3, "anomaly_detection": 52.0, "localization": 2.7,
        "state_query": 65.3, "multi_hop": 2.7, "comparison": 8.7,
        "antecedent": 7.3, "anomaly_localization": 34.0, "counting": 19.3,
        "ordering": 46.7,
    },
    "1000": {
        "existence": 47.3, "anomaly_detection": 46.7, "localization": 2.0,
        "state_query": 68.0, "multi_hop": 0.7, "comparison": 9.3,
        "antecedent": 13.3, "anomaly_localization": 30.7, "counting": 17.3,
        "ordering": 52.7,
    },
    "10000": {
        "existence": 54.0, "anomaly_detection": 45.3, "localization": 6.0,
        "state_query": 24.7, "multi_hop": 2.7, "comparison": 6.7,
        "antecedent": 11.3, "anomaly_localization": 34.0, "counting": 20.7,
        "ordering": 38.7,
    },
    "90000": {
        "existence": 53.3, "anomaly_detection": 50.0, "localization": 4.0,
        "state_query": 22.0, "multi_hop": 3.3, "comparison": 6.0,
        "antecedent": 8.7, "anomaly_localization": 27.3, "counting": 16.0,
        "ordering": 41.3,
    },
    "360000": {
        "existence": 51.3, "anomaly_detection": 48.0, "localization": 3.3,
        "state_query": 31.3, "multi_hop": 1.3, "comparison": 8.0,
        "antecedent": 9.3, "anomaly_localization": 28.7, "counting": 20.7,
        "ordering": 54.0,
    },
    "720000": {
        "existence": 50.0, "anomaly_detection": 54.7, "localization": 2.7,
        "state_query": 34.7, "multi_hop": 2.7, "comparison": 3.3,
        "antecedent": 9.3, "anomaly_localization": 24.0, "counting": 18.7,
        "ordering": 50.7,
    },
}

ITFORMER_BY_CTX_TASK: dict[str, dict[str, float]] = {
    "256": {
        "existence": 40.0, "anomaly_detection": 52.0, "localization": 2.0,
        "state_query": 4.0, "multi_hop": 0.0, "comparison": 2.0,
        "antecedent": 7.3, "anomaly_localization": 0.7, "counting": 12.7,
        "ordering": 30.7,
    },
    "1000": {
        "existence": 48.0, "anomaly_detection": 52.7, "localization": 2.7,
        "state_query": 3.3, "multi_hop": 1.3, "comparison": 0.7,
        "antecedent": 6.0, "anomaly_localization": 0.0, "counting": 17.3,
        "ordering": 35.3,
    },
    "10000": {
        "existence": 42.7, "anomaly_detection": 48.7, "localization": 1.3,
        "state_query": 4.7, "multi_hop": 0.0, "comparison": 1.3,
        "antecedent": 5.3, "anomaly_localization": 0.0, "counting": 20.0,
        "ordering": 30.0,
    },
}

TASKS = [
    "existence", "anomaly_detection", "localization", "state_query", "multi_hop",
    "comparison", "antecedent", "anomaly_localization", "counting", "ordering",
]
TASK_LABELS = [
    "Existence", "Anom. Det.", "Localization", "State Query", "Multi-Hop",
    "Comparison", "Antecedent", "Anom. Loc.", "Counting", "Ordering",
]

CTX_ORDER = ["256", "1000", "10000", "90000", "360000", "720000"]
CTX_TIME = {"256": "2.56s", "1000": "10s", "10000": "100s", "90000": "15min", "360000": "1h", "720000": "2h"}

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def recompute_summary(output_dir: Path) -> dict:
    """Recompute and save summary.json from trajectories."""
    from src.models.ts_llm.arts.capture24 import compute_summary
    summary = compute_summary(output_dir)
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    return summary


def load_per_task_context(output_dir: Path) -> dict[str, dict[str, float]]:
    """Load per-(context, task) accuracy from trajectory files.

    Returns {context_str: {task: accuracy_pct}}.
    """
    buckets: dict[str, dict[str, dict[str, int]]] = defaultdict(
        lambda: defaultdict(lambda: {"correct": 0, "total": 0})
    )
    for traj_path in sorted(output_dir.rglob("trajectories.jsonl")):
        with open(traj_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                ctx = str(rec.get("context_length_samples", 0))
                task = rec.get("task_type", "unknown")
                buckets[ctx][task]["total"] += 1
                if rec.get("correct", False):
                    buckets[ctx][task]["correct"] += 1

    out: dict[str, dict[str, float]] = {}
    for ctx, tasks in buckets.items():
        out[ctx] = {}
        for task, stats in tasks.items():
            out[ctx][task] = stats["correct"] / stats["total"] * 100 if stats["total"] > 0 else 0.0
    return out


HARDCODED_BASELINES: dict[str, dict[str, dict[str, float]]] = {
    "Flamingo": FLAMINGO_BY_CTX_TASK,
    "ITFormer": ITFORMER_BY_CTX_TASK,
}


def baseline_overall_by_context(data: dict[str, dict[str, float]]) -> dict[str, float]:
    """Compute average accuracy per context (mean over tasks) for a hardcoded baseline."""
    out = {}
    for ctx, tasks in data.items():
        vals = [tasks[t] for t in TASKS if t in tasks]
        out[ctx] = sum(vals) / len(vals) if vals else 0.0
    return out


def baseline_overall_by_task(data: dict[str, dict[str, float]]) -> dict[str, float]:
    """Compute average accuracy per task (mean over contexts) for a hardcoded baseline."""
    out = {}
    for task in TASKS:
        vals = [data[c][task] for c in CTX_ORDER if task in data.get(c, {})]
        out[task] = sum(vals) / len(vals) if vals else 0.0
    return out


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_accuracy_by_context(summaries: dict[str, dict], output_path: Path) -> None:
    """Line plot: accuracy vs context length for each method."""
    sns.set_theme(style="whitegrid", font_scale=1.2)
    fig, ax = plt.subplots(figsize=(10, 6))

    for method, cfg in METHODS.items():
        if method in HARDCODED_BASELINES:
            bl_ctx = baseline_overall_by_context(HARDCODED_BASELINES[method])
            ctxs = [c for c in CTX_ORDER if c in bl_ctx]
            vals = [bl_ctx[c] for c in ctxs]
        else:
            by_ctx = summaries[method].get("by_context", {})
            # Keys may be int or str depending on whether summary was just computed or loaded from JSON
            by_ctx_str = {str(k): v for k, v in by_ctx.items()}
            ctxs = [c for c in CTX_ORDER if c in by_ctx_str]
            vals = [by_ctx_str[c]["accuracy"] * 100 for c in ctxs]

        x_pos = [CTX_ORDER.index(c) for c in ctxs]
        ax.plot(x_pos, vals, marker=cfg["marker"], color=cfg["color"],
                linewidth=2.5, markersize=9, label=method)
        for xp, val in zip(x_pos, vals):
            ax.annotate(f"{val:.1f}%", xy=(xp, val), xytext=(0, 8),
                        textcoords="offset points", ha="center", va="bottom",
                        fontsize=9, color=cfg["color"], fontweight="bold")

    ax.set_xticks(range(len(CTX_ORDER)))
    ax.set_xticklabels([f"{c}\n({CTX_TIME[c]})" for c in CTX_ORDER])
    ax.set_xlabel("Context Length (samples / time)")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Accuracy by Context Length")
    ax.legend(loc="upper left")
    ax.set_ylim(0, 90)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {output_path}")


def plot_accuracy_by_task(summaries: dict[str, dict], output_path: Path) -> None:
    """Clustered bar chart: 3 bars × 10 tasks."""
    import numpy as np
    sns.set_theme(style="whitegrid", font_scale=1.1)
    fig, ax = plt.subplots(figsize=(14, 6))

    n_tasks = len(TASKS)
    x = np.arange(n_tasks)
    method_list = list(METHODS.keys())
    n_methods = len(method_list)
    width = 0.8 / n_methods

    for i, method in enumerate(method_list):
        cfg = METHODS[method]
        if method in HARDCODED_BASELINES:
            bl_task = baseline_overall_by_task(HARDCODED_BASELINES[method])
            vals = [bl_task.get(t, 0) for t in TASKS]
        else:
            by_task = summaries[method].get("by_task", {})
            vals = [by_task.get(t, {}).get("accuracy", 0) * 100 for t in TASKS]

        offset = (i - (n_methods - 1) / 2) * width
        bars = ax.bar(x + offset, vals, width, label=method, color=cfg["color"], edgecolor="white")
        for bar in bars:
            h = bar.get_height()
            ax.annotate(f"{h:.1f}", xy=(bar.get_x() + bar.get_width() / 2, h),
                        xytext=(0, 3), textcoords="offset points",
                        ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(TASK_LABELS)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Accuracy by Task Type")
    ax.legend(loc="upper right")
    ax.set_ylim(0, 105)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {output_path}")


def plot_context_x_task(per_task_ctx: dict[str, dict[str, dict[str, float]]], output_path: Path) -> None:
    """2×5 grid: one subplot per task, lines for each method across contexts."""
    sns.set_theme(style="whitegrid", font_scale=0.95)
    fig, axes = plt.subplots(2, 5, figsize=(22, 9), sharey=True)
    axes = axes.flatten()

    for i, (task, label) in enumerate(zip(TASKS, TASK_LABELS)):
        ax = axes[i]
        for method, cfg in METHODS.items():
            ctx_task = per_task_ctx[method]
            ctxs = [c for c in CTX_ORDER if c in ctx_task and task in ctx_task[c]]
            vals = [ctx_task[c][task] for c in ctxs]
            x_pos = [CTX_ORDER.index(c) for c in ctxs]
            ax.plot(x_pos, vals, marker=cfg["marker"], color=cfg["color"],
                    linewidth=2, markersize=6, label=method)

        ax.set_title(label, fontweight="bold")
        ax.set_xticks(range(len(CTX_ORDER)))
        ax.set_xticklabels([CTX_TIME[c] for c in CTX_ORDER], rotation=45, ha="right")
        ax.set_ylim(-2, 102)
        if i % 5 == 0:
            ax.set_ylabel("Accuracy (%)")

    axes[0].legend(loc="lower left", fontsize=8)
    fig.suptitle("Accuracy by Context Length per Task", fontsize=15, fontweight="bold", y=1.01)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {output_path}")


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------


def print_summary_table(
    summaries: dict[str, dict],
    per_task_ctx: dict[str, dict[str, dict[str, float]]],
) -> None:
    """Print a comprehensive accuracy table with deltas vs Flamingo."""

    baseline_ctx = {name: baseline_overall_by_context(data) for name, data in HARDCODED_BASELINES.items()}
    baseline_task = {name: baseline_overall_by_task(data) for name, data in HARDCODED_BASELINES.items()}

    # Helper: format delta
    def _delta(val: float, ref: float) -> str:
        d = val - ref
        return f"({'+'if d >= 0 else ''}{d:.1f}pp)"

    bl_names = list(HARDCODED_BASELINES.keys())
    sep = "-" * 130
    print(f"\n{'=' * 130}")
    print("BENCHMARK SUMMARY")
    print(f"{'=' * 130}")

    # --- By context length ---
    print(f"\n{sep}")
    bl_headers = " | ".join(f"{n:>10}" for n in bl_names)
    print(f"{'Context':<14} | {'Agentic (oracle)':>20} | {'Agentic (real)':>20} | {bl_headers} | "
          f"{'Oracle vs Flam.':>16} | {'Real vs Flam.':>16}")
    print(sep)

    oracle_overall_n = 0
    oracle_overall_c = 0
    real_overall_n = 0
    real_overall_c = 0
    bl_overall_sums = {n: 0.0 for n in bl_names}
    bl_overall_counts = {n: 0 for n in bl_names}

    for ctx in CTX_ORDER:
        flam_val = baseline_ctx["Flamingo"].get(ctx)

        oracle_str = ""
        real_str = ""
        oracle_delta = ""
        real_delta = ""

        for method in ["Agentic with oracle class.", "Agentic with real class."]:
            if method not in summaries:
                continue
            by_ctx = {str(k): v for k, v in summaries[method].get("by_context", {}).items()}
            if ctx not in by_ctx:
                continue
            stats = by_ctx[ctx]
            acc = stats["accuracy"] * 100
            n = stats["total"]
            c = stats["correct"]
            cell = f"{acc:5.1f}% ({c}/{n})"
            if method == "Agentic with oracle class.":
                oracle_str = cell
                oracle_overall_n += n
                oracle_overall_c += c
                if flam_val is not None:
                    oracle_delta = _delta(acc, flam_val)
            else:
                real_str = cell
                real_overall_n += n
                real_overall_c += c
                if flam_val is not None:
                    real_delta = _delta(acc, flam_val)

        bl_strs = []
        for n in bl_names:
            val = baseline_ctx[n].get(ctx)
            if val is not None:
                bl_strs.append(f"{val:5.1f}%")
                bl_overall_sums[n] += val
                bl_overall_counts[n] += 1
            else:
                bl_strs.append("")
        bl_col = " | ".join(f"{s:>10}" for s in bl_strs)

        label = f"{ctx} ({CTX_TIME[ctx]})"
        print(f"{label:<14} | {oracle_str:>20} | {real_str:>20} | {bl_col} | "
              f"{oracle_delta:>16} | {real_delta:>16}")

    # Overall row
    oracle_overall_acc = oracle_overall_c / oracle_overall_n * 100 if oracle_overall_n else 0
    real_overall_acc = real_overall_c / real_overall_n * 100 if real_overall_n else 0
    flam_overall_acc = bl_overall_sums["Flamingo"] / bl_overall_counts["Flamingo"] if bl_overall_counts["Flamingo"] else 0

    print(sep)
    oracle_ov_str = f"{oracle_overall_acc:5.1f}% ({oracle_overall_c}/{oracle_overall_n})" if oracle_overall_n else ""
    real_ov_str = f"{real_overall_acc:5.1f}% ({real_overall_c}/{real_overall_n})" if real_overall_n else ""
    bl_ov_strs = []
    for n in bl_names:
        if bl_overall_counts[n]:
            bl_ov_strs.append(f"{bl_overall_sums[n] / bl_overall_counts[n]:5.1f}%")
        else:
            bl_ov_strs.append("")
    bl_ov_col = " | ".join(f"{s:>10}" for s in bl_ov_strs)
    oracle_ov_delta = _delta(oracle_overall_acc, flam_overall_acc) if oracle_overall_n else ""
    real_ov_delta = _delta(real_overall_acc, flam_overall_acc) if real_overall_n else ""
    print(f"{'OVERALL':<14} | {oracle_ov_str:>20} | {real_ov_str:>20} | {bl_ov_col} | "
          f"{oracle_ov_delta:>16} | {real_ov_delta:>16}")

    # --- By task ---
    print(f"\n{sep}")
    bl_task_headers = " | ".join(f"{n:>10}" for n in bl_names)
    print(f"{'Task':<20} | {'Agentic (oracle)':>18} | {'Agentic (real)':>18} | {bl_task_headers} | "
          f"{'Oracle vs Flam.':>16} | {'Real vs Flam.':>16}")
    print(sep)

    for task, label in zip(TASKS, TASK_LABELS):
        flam_val = baseline_task["Flamingo"].get(task)

        oracle_str = ""
        real_str = ""
        oracle_delta = ""
        real_delta = ""

        for method in ["Agentic with oracle class.", "Agentic with real class."]:
            if method not in summaries:
                continue
            by_task = summaries[method].get("by_task", {})
            if task not in by_task:
                continue
            acc = by_task[task]["accuracy"] * 100
            cell = f"{acc:5.1f}%"
            if method == "Agentic with oracle class.":
                oracle_str = cell
                if flam_val is not None:
                    oracle_delta = _delta(acc, flam_val)
            else:
                real_str = cell
                if flam_val is not None:
                    real_delta = _delta(acc, flam_val)

        bl_task_strs = []
        for n in bl_names:
            val = baseline_task[n].get(task)
            if val is not None:
                bl_task_strs.append(f"{val:5.1f}%")
            else:
                bl_task_strs.append("")
        bl_task_col = " | ".join(f"{s:>10}" for s in bl_task_strs)

        print(f"{label:<20} | {oracle_str:>18} | {real_str:>18} | {bl_task_col} | "
              f"{oracle_delta:>16} | {real_delta:>16}")

    print(f"{'=' * 130}\n")


def print_head_to_head_table(
    summaries: dict[str, dict],
    per_task_ctx: dict[str, dict[str, dict[str, float]]],
) -> None:
    """Print a task × context grid comparing Agentic (real) vs Flamingo."""

    real_method = "Agentic with real class."
    real_ctx = per_task_ctx.get(real_method, {})
    flam_ctx = FLAMINGO_BY_CTX_TASK

    # Column widths
    task_w = 16
    ctx_w = 15  # each context column: "real% / flam%"
    sep_char = "-"

    header_ctxs = [f"{c} ({CTX_TIME[c]})" for c in CTX_ORDER]
    total_w = task_w + 3 + len(CTX_ORDER) * (ctx_w + 3) + ctx_w + 3  # +avg column
    sep = sep_char * total_w

    print(f"\n{'=' * total_w}")
    print("HEAD-TO-HEAD: Agentic (real) vs Flamingo  —  per task × context length")
    print(f"{'=' * total_w}")
    print(f"\n  Cell format: ARTS real% / Flamingo%\n")

    # Header row
    print(sep)
    hdr = f"{'Task':<{task_w}}"
    for label in header_ctxs:
        hdr += f" | {label:^{ctx_w}}"
    hdr += f" | {'Average':^{ctx_w}}"
    print(hdr)
    print(sep)

    # Accumulate column averages
    col_real_sums = {c: 0.0 for c in CTX_ORDER}
    col_flam_sums = {c: 0.0 for c in CTX_ORDER}
    col_counts = {c: 0 for c in CTX_ORDER}

    for task, label in zip(TASKS, TASK_LABELS):
        row = f"{label:<{task_w}}"
        task_real_sum = 0.0
        task_flam_sum = 0.0
        task_count = 0

        for ctx in CTX_ORDER:
            r_val = real_ctx.get(ctx, {}).get(task)
            f_val = flam_ctx.get(ctx, {}).get(task)

            if r_val is not None and f_val is not None:
                cell = f"{r_val:4.1f} / {f_val:4.1f}"
                task_real_sum += r_val
                task_flam_sum += f_val
                task_count += 1
                col_real_sums[ctx] += r_val
                col_flam_sums[ctx] += f_val
                col_counts[ctx] += 1
            elif r_val is not None:
                cell = f"{r_val:4.1f} /    —"
            elif f_val is not None:
                cell = f"   — / {f_val:4.1f}"
            else:
                cell = "—"
            row += f" | {cell:^{ctx_w}}"

        # Average across contexts for this task
        if task_count > 0:
            avg_r = task_real_sum / task_count
            avg_f = task_flam_sum / task_count
            avg_cell = f"{avg_r:4.1f} / {avg_f:4.1f}"
        else:
            avg_cell = "—"
        row += f" | {avg_cell:^{ctx_w}}"
        print(row)

    # Overall row
    print(sep)
    overall_row = f"{'OVERALL':<{task_w}}"
    grand_real = 0.0
    grand_flam = 0.0
    grand_count = 0
    for ctx in CTX_ORDER:
        if col_counts[ctx] > 0:
            r_avg = col_real_sums[ctx] / col_counts[ctx]
            f_avg = col_flam_sums[ctx] / col_counts[ctx]
            cell = f"{r_avg:4.1f} / {f_avg:4.1f}"
            grand_real += col_real_sums[ctx]
            grand_flam += col_flam_sums[ctx]
            grand_count += col_counts[ctx]
        else:
            cell = "—"
        overall_row += f" | {cell:^{ctx_w}}"

    if grand_count > 0:
        gr = grand_real / grand_count
        gf = grand_flam / grand_count
        avg_cell = f"{gr:4.1f} / {gf:4.1f}"
    else:
        avg_cell = "—"
    overall_row += f" | {avg_cell:^{ctx_w}}"
    print(overall_row)

    print(f"{'=' * total_w}\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Plot GPT benchmark results vs Flamingo")
    parser.add_argument("--output-dir", type=Path, default=RESULTS_ROOT,
                        help="Directory to save plots (default: results/gpt_benchmark/)")
    parser.add_argument("--skip-recompute", action="store_true",
                        help="Use existing summary.json without recomputing from trajectories")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Recompute / load summaries
    summaries: dict[str, dict] = {}
    per_task_ctx: dict[str, dict[str, dict[str, float]]] = {}

    for method, cfg in METHODS.items():
        if cfg["dir"] is None:
            continue
        d = cfg["dir"]
        if not d.exists():
            print(f"SKIP {method}: {d} not found")
            continue

        if args.skip_recompute:
            with open(d / "summary.json") as f:
                summaries[method] = json.load(f)
        else:
            print(f"Recomputing summary for {method}...")
            summaries[method] = recompute_summary(d)

        per_task_ctx[method] = load_per_task_context(d)

    # Hardcoded baselines from paper
    for bl_name, bl_data in HARDCODED_BASELINES.items():
        per_task_ctx[bl_name] = bl_data

    # 2. Generate plots
    print("Generating plots...")
    plot_accuracy_by_context(summaries, args.output_dir / "accuracy_by_context.png")
    plot_accuracy_by_task(summaries, args.output_dir / "accuracy_by_task.png")
    plot_context_x_task(per_task_ctx, args.output_dir / "accuracy_context_x_task.png")

    # 3. Print summary tables
    print_summary_table(summaries, per_task_ctx)
    print_head_to_head_table(summaries, per_task_ctx)

    print("Done.")


if __name__ == "__main__":
    main()
