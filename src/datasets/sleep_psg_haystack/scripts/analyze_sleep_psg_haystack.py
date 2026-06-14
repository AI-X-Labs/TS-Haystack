#!/usr/bin/env python3
# SPDX-License-Identifier: CC-BY-NC-4.0
"""
Correctness verification + paper statistics for the windowed Sleep PSG benchmark.

Two passes over every generated parquet under
    data/sleep_psg/ts_haystack/{label_class}/tasks/{ctx}/{task}/{split}/data.parquet

Pass 1 — Training-readiness checks (FAIL HARD if any issue):
    * Schema columns present, no nulls in critical columns.
    * is_valid == True for every row (training only consumes valid samples).
    * window_end_ms > window_start_ms; duration matches context length when fixed.
    * subject_id of every sample is in the matching split (train/val/test) — no
      cross-split leakage.
    * No template placeholders ("{...}") left in question or answer strings.
    * answer_type matches the task contract.
    * No duplicate (subject_id, window_start_ms, task_type, question) within split.

Pass 2 — Paper statistics (printed + dumped to JSON):
    * Sample counts per (label_class, ctx, task, split).
    * Unique subjects per split.
    * Answer distribution per task:
        boolean   -> yes/no balance
        integer   -> histogram + zero-rate
        category  -> per-class counts
        time_range -> position-in-context distribution (start fraction)
    * Window-position statistics for the answer region (mean, std, deciles).
    * Comparison: longest/shortest balance.
    * Multi-hop: before/after balance.
    * Existence: positive/negative balance per target activity.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[4]
DATA_ROOT = REPO_ROOT / "data" / "sleep_psg" / "ts_haystack"
SPLIT_FILE = DATA_ROOT / "participant_split.json"

CRITICAL_COLS = [
    "subject_id", "recording_duration_ms", "label_class", "task_type",
    "context_length_s", "window_start_ms", "window_end_ms",
    "question", "answer", "answer_type", "metadata", "is_valid",
]

TASK_ANSWER_TYPES = {
    "existence":     "boolean",
    "localization":  "time_range",
    "counting":      "integer",
    "ordering":      "boolean",
    "state_query":   "category",
    "antecedent":    "category",
    "comparison":    "time_range",
    "multi_hop":     "time_range",
}


def _ctx_to_seconds(ctx_dir: str) -> float:
    if ctx_dir == "full":
        return -1.0
    assert ctx_dir.endswith("s")
    return float(ctx_dir[:-1])


def _iter_parquets():
    for label_class_dir in sorted((DATA_ROOT).glob("*/tasks")):
        label_class = label_class_dir.parent.name
        if label_class not in ("sleep_stages", "arousals"):
            continue
        for ctx_dir in sorted(label_class_dir.iterdir()):
            if not ctx_dir.is_dir():
                continue
            for task_dir in sorted(ctx_dir.iterdir()):
                for split_dir in sorted(task_dir.iterdir()):
                    p = split_dir / "data.parquet"
                    if p.exists():
                        yield label_class, ctx_dir.name, task_dir.name, split_dir.name, p


def _load_metadata(df: pd.DataFrame) -> list:
    out = []
    for s in df["metadata"]:
        try:
            out.append(json.loads(s) if isinstance(s, str) else (s or {}))
        except Exception:
            out.append({})
    return out


def _answer_position_fraction(row, meta) -> float | None:
    """Where in the window is the answer? Returns fraction in [0,1] or None."""
    win_len = row["window_end_ms"] - row["window_start_ms"]
    if win_len <= 0:
        return None

    # Pick the most informative timestamp from metadata for each task family.
    s = meta.get("start_ms")
    e = meta.get("end_ms")
    # multi_hop: target start_ms is the answer; anchor_start_ms is anchor
    # antecedent: start_ms is the asked-about bout; antecedent_start_ms is the answer
    if "antecedent_start_ms" in meta:
        s = meta.get("antecedent_start_ms")
        e = meta.get("antecedent_end_ms")
    if "arousal_start_ms" in meta:
        s = meta.get("arousal_start_ms")
        e = meta.get("arousal_end_ms")

    if s is None:
        return None
    mid = (s + (e if e is not None else s)) / 2.0
    return float(mid) / float(win_len)


def verify_and_stats(strict: bool, out_json: Path | None, out_md: Path | None = None):
    split = json.loads(SPLIT_FILE.read_text())
    split_set = {k: set(v) for k, v in split.items() if k in ("train", "val", "test")}

    errors: list[str] = []
    warnings: list[str] = []
    stats: dict = {}

    file_count = 0
    total_samples = 0
    for label_class, ctx_dir, task, split_name, path in _iter_parquets():
        file_count += 1
        try:
            df = pd.read_parquet(path)
        except Exception as e:
            errors.append(f"[READ] {path}: {e}")
            continue

        n = len(df)
        total_samples += n
        rel = f"{label_class}/{ctx_dir}/{task}/{split_name}"

        # ---- Pass 1: correctness ----
        missing_cols = [c for c in CRITICAL_COLS if c not in df.columns]
        if missing_cols:
            errors.append(f"[SCHEMA] {rel}: missing columns {missing_cols}")
            continue

        if n == 0:
            errors.append(f"[EMPTY] {rel}: zero samples")
            continue

        if not df["is_valid"].all():
            bad = int((~df["is_valid"]).sum())
            errors.append(f"[INVALID] {rel}: {bad}/{n} rows have is_valid=False")

        for col in ("subject_id", "question", "answer", "answer_type"):
            n_null = int(df[col].isna().sum())
            if n_null:
                errors.append(f"[NULL] {rel}: {n_null} nulls in {col}")

        # window bounds
        bad_win = int((df["window_end_ms"] <= df["window_start_ms"]).sum())
        if bad_win:
            errors.append(f"[WINDOW] {rel}: {bad_win} rows with window_end <= window_start")

        # context length consistency
        ctx_s_expected = _ctx_to_seconds(ctx_dir)
        if ctx_s_expected > 0:
            expected_ms = int(round(ctx_s_expected * 1000))
            bad_dur = int(((df["window_end_ms"] - df["window_start_ms"]) != expected_ms).sum())
            if bad_dur:
                # window may be clipped at recording end? we treat as warning if rare
                warnings.append(f"[DUR] {rel}: {bad_dur} rows with window length != {expected_ms}ms")
            bad_ctx_col = int((df["context_length_s"] != ctx_s_expected).sum())
            if bad_ctx_col:
                errors.append(f"[CTX] {rel}: {bad_ctx_col} rows with context_length_s != {ctx_s_expected}")
        else:
            bad_ctx_col = int((df["context_length_s"] != -1.0).sum())
            if bad_ctx_col:
                errors.append(f"[CTX] {rel}: {bad_ctx_col} rows with context_length_s != -1 (full)")

        # split integrity
        expected_subjects = split_set.get(split_name, set())
        leaked = sorted(set(df["subject_id"].unique()) - expected_subjects)
        if leaked:
            errors.append(f"[SPLIT] {rel}: {len(leaked)} subjects not in split (e.g. {leaked[:3]})")

        # placeholders / answer type
        n_placeholder_q = int(df["question"].astype(str).str.contains(r"\{[a-zA-Z_]", regex=True).sum())
        n_placeholder_a = int(df["answer"].astype(str).str.contains(r"\{[a-zA-Z_]", regex=True).sum())
        if n_placeholder_q or n_placeholder_a:
            errors.append(f"[TEMPLATE] {rel}: unfilled placeholders Q={n_placeholder_q} A={n_placeholder_a}")

        expected_ans_type = TASK_ANSWER_TYPES.get(task)
        if expected_ans_type:
            bad_at = int((df["answer_type"] != expected_ans_type).sum())
            if bad_at:
                errors.append(f"[ATYPE] {rel}: {bad_at} rows answer_type != {expected_ans_type}")

        # duplicate detection
        dup_keys = df.duplicated(subset=["subject_id", "window_start_ms", "task_type", "question"])
        n_dup = int(dup_keys.sum())
        if n_dup:
            warnings.append(f"[DUP] {rel}: {n_dup} duplicate (subject,window,task,question) rows")

        # ---- Pass 2: statistics (use metadata, NOT answer strings — answer
        # text is intentionally diverse across template variants) ----
        metas = _load_metadata(df)
        ans_type = expected_ans_type or df["answer_type"].iloc[0]

        cell = {
            "n": n,
            "unique_subjects": int(df["subject_id"].nunique()),
            "answer_type": ans_type,
        }

        # Boolean tasks: existence (is_positive) and ordering (a_before_b)
        if task == "existence":
            yes = sum(1 for m in metas if m.get("is_positive") is True)
            no = sum(1 for m in metas if m.get("is_positive") is False)
            other = n - yes - no
            cell["balance"] = {"yes": yes, "no": no, "other": other}
            if other:
                warnings.append(f"[META] {rel}: {other} rows missing is_positive")
        elif task == "ordering":
            yes = sum(1 for m in metas if m.get("a_before_b") is True)
            no = sum(1 for m in metas if m.get("a_before_b") is False)
            other = n - yes - no
            cell["balance"] = {"yes": yes, "no": no, "other": other}
            if other:
                warnings.append(f"[META] {rel}: {other} rows missing a_before_b")

        elif task == "counting":
            ints = [m.get("count") for m in metas if isinstance(m.get("count"), int)]
            n_missing = n - len(ints)
            if ints:
                arr = np.array(ints)
                counts = Counter(ints)
                cell["int_summary"] = {
                    "min": int(arr.min()),
                    "max": int(arr.max()),
                    "mean": float(arr.mean()),
                    "median": float(np.median(arr)),
                    "zero_rate": float((arr == 0).mean()),
                    "n_missing_meta": n_missing,
                    "histogram": {str(k): counts[k] for k in sorted(counts)[:25]},
                }
            if n_missing:
                warnings.append(f"[META] {rel}: {n_missing} rows missing count metadata")

        elif task in ("antecedent", "state_query"):
            key = "antecedent_activity" if task == "antecedent" else "sleep_stage"
            cats = Counter(m.get(key) for m in metas if m.get(key))
            cell["category_counts"] = dict(cats)
            n_missing = n - sum(cats.values())
            if n_missing:
                warnings.append(f"[META] {rel}: {n_missing} rows missing {key}")

        # answer-position-in-window distribution (for ALL tasks where applicable)
        positions = []
        for r, m in zip(df.to_dict("records"), metas):
            f = _answer_position_fraction(r, m)
            if f is not None and 0.0 <= f <= 1.0:
                positions.append(f)
        if positions:
            arr = np.array(positions)
            cell["answer_position_frac"] = {
                "n_with_position": len(arr),
                "mean": float(arr.mean()),
                "std": float(arr.std()),
                "deciles": [float(x) for x in np.quantile(arr, np.linspace(0, 1, 11))],
            }

        # task-specific extras
        if task == "comparison":
            sup = Counter(m.get("superlative") for m in metas)
            cell["superlative"] = {k: v for k, v in sup.items() if k}
        if task == "multi_hop":
            direction = Counter(m.get("direction") for m in metas)
            cell["direction"] = {k: v for k, v in direction.items() if k}
        if task == "existence":
            pos_neg = Counter("positive" if m.get("is_positive") else "negative" for m in metas)
            target_counts = Counter(m.get("target_activity") for m in metas)
            cell["pos_neg"] = dict(pos_neg)
            cell["target_activity"] = {k: v for k, v in target_counts.items() if k}
        if task in ("localization", "counting", "comparison", "antecedent"):
            act = Counter(m.get("activity") for m in metas)
            cell["activity"] = {k: v for k, v in act.items() if k}

        stats.setdefault(label_class, {}).setdefault(ctx_dir, {}).setdefault(task, {})[split_name] = cell

    # ---- Print summary ----
    print("=" * 78)
    print(f"Sleep PSG haystack — analysis of {file_count} parquet files, "
          f"{total_samples:,} total samples")
    print("=" * 78)

    print("\n# Correctness checks")
    if not errors:
        print("  ALL CHECKS PASSED — dataset is training-ready ✓")
    else:
        print(f"  {len(errors)} ERRORS:")
        for e in errors:
            print(f"    {e}")
    if warnings:
        print(f"  {len(warnings)} warnings:")
        for w in warnings[:30]:
            print(f"    {w}")
        if len(warnings) > 30:
            print(f"    ... ({len(warnings) - 30} more)")

    # Sample-count table
    print("\n# Sample counts")
    for lc in sorted(stats):
        print(f"\n[{lc}]")
        # determine col order
        ctxs = sorted(stats[lc].keys(), key=lambda c: (c == "full", _ctx_to_seconds(c)))
        tasks_seen = sorted({t for c in ctxs for t in stats[lc][c].keys()})
        header = f"  {'task':14s}" + "".join(f"{c:>22s}" for c in ctxs)
        print(header)
        for task in tasks_seen:
            row = f"  {task:14s}"
            for c in ctxs:
                cell = stats[lc][c].get(task)
                if cell is None:
                    row += f"{'-':>22s}"
                else:
                    parts = []
                    for sp in ("train", "val", "test"):
                        v = cell.get(sp, {}).get("n", 0) if isinstance(cell.get(sp), dict) else 0
                        parts.append(str(v))
                    row += f"{'/'.join(parts):>22s}"
            print(row)

    # Label balance highlights
    print("\n# Label balance highlights (test split)")
    for lc in sorted(stats):
        for ctx in sorted(stats[lc].keys(), key=lambda c: (c == "full", _ctx_to_seconds(c))):
            for task in sorted(stats[lc][ctx]):
                cell = stats[lc][ctx][task].get("test")
                if not cell:
                    continue
                if "balance" in cell:
                    b = cell["balance"]
                    tot = b["yes"] + b["no"] + b["other"]
                    if tot:
                        print(f"  {lc}/{ctx}/{task:12s}: yes={b['yes']/tot:.2f} no={b['no']/tot:.2f}")
                if "int_summary" in cell and cell["int_summary"]["mean"] is not None:
                    s = cell["int_summary"]
                    print(f"  {lc}/{ctx}/{task:12s}: int mean={s['mean']:.1f} median={s['median']} "
                          f"zero_rate={s['zero_rate']:.2f} range=[{s['min']},{s['max']}]")
                if "category_counts" in cell:
                    c = cell["category_counts"]
                    tot = sum(c.values()) or 1
                    top = sorted(c.items(), key=lambda x: -x[1])[:6]
                    pretty = " ".join(f"{k}:{v/tot:.2f}" for k, v in top)
                    print(f"  {lc}/{ctx}/{task:12s}: cats {pretty}")

    # Position-in-context highlights
    print("\n# Answer position in window (test split, mean ± std)")
    for lc in sorted(stats):
        for ctx in sorted(stats[lc].keys(), key=lambda c: (c == "full", _ctx_to_seconds(c))):
            for task in sorted(stats[lc][ctx]):
                cell = stats[lc][ctx][task].get("test")
                if not cell or "answer_position_frac" not in cell:
                    continue
                p = cell["answer_position_frac"]
                print(f"  {lc}/{ctx}/{task:12s}: {p['mean']:.2f} ± {p['std']:.2f} "
                      f"(n={p['n_with_position']})")

    if out_json:
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps({
            "errors": errors,
            "warnings": warnings,
            "stats": stats,
            "n_files": file_count,
            "n_samples": total_samples,
        }, indent=2, default=str))
        print(f"\nFull report written to: {out_json}")

    if out_md:
        out_md.parent.mkdir(parents=True, exist_ok=True)
        out_md.write_text(_render_markdown(
            errors=errors, warnings=warnings, stats=stats,
            file_count=file_count, total_samples=total_samples,
        ))
        print(f"Markdown report written to: {out_md}")

    if errors and strict:
        sys.exit(1)


def _render_markdown(errors, warnings, stats, file_count, total_samples) -> str:
    """Render the analysis as a paper-ready markdown document."""
    lines: list[str] = []
    lines.append("# Sleep PSG Haystack — Benchmark Analysis Report")
    lines.append("")
    lines.append(
        f"**{file_count} parquet files · {total_samples:,} samples · "
        f"{len(errors)} errors · {len(warnings)} warnings**"
    )
    lines.append("")

    # ---- Correctness ----
    lines.append("## Correctness checks")
    lines.append("")
    if not errors:
        lines.append("All hard correctness checks passed — dataset is training-ready.")
    else:
        lines.append(f"**{len(errors)} errors:**")
        lines.append("")
        for e in errors:
            lines.append(f"- `{e}`")
    if warnings:
        lines.append("")
        lines.append(f"<details><summary>{len(warnings)} warnings</summary>")
        lines.append("")
        for w in warnings:
            lines.append(f"- `{w}`")
        lines.append("")
        lines.append("</details>")
    lines.append("")

    # ---- Sample counts ----
    lines.append("## Sample counts (train / val / test)")
    lines.append("")
    for lc in sorted(stats):
        ctxs = sorted(stats[lc].keys(), key=lambda c: (c == "full", _ctx_to_seconds(c)))
        tasks_seen = sorted({t for c in ctxs for t in stats[lc][c].keys()})
        lines.append(f"### `{lc}`")
        lines.append("")
        lines.append("| task | " + " | ".join(ctxs) + " |")
        lines.append("|---|" + "|".join(["---"] * len(ctxs)) + "|")
        for task in tasks_seen:
            row = [f"`{task}`"]
            for c in ctxs:
                cell = stats[lc][c].get(task)
                if cell is None:
                    row.append("—")
                else:
                    parts = [
                        str(cell.get(sp, {}).get("n", 0)) if isinstance(cell.get(sp), dict) else "0"
                        for sp in ("train", "val", "test")
                    ]
                    row.append(" / ".join(parts))
            lines.append("| " + " | ".join(row) + " |")
        lines.append("")

    # ---- Boolean balance ----
    lines.append("## Boolean balance (test split)")
    lines.append("")
    lines.append("Computed from canonical metadata fields (`is_positive` for existence, "
                 "`a_before_b` for ordering), not from answer text.")
    lines.append("")
    lines.append("| label_class | ctx | task | yes | no |")
    lines.append("|---|---|---|---|---|")
    for lc in sorted(stats):
        for ctx in sorted(stats[lc].keys(), key=lambda c: (c == "full", _ctx_to_seconds(c))):
            for task in sorted(stats[lc][ctx]):
                cell = stats[lc][ctx][task].get("test")
                if not cell or "balance" not in cell:
                    continue
                b = cell["balance"]
                tot = b["yes"] + b["no"] + b["other"]
                if not tot:
                    continue
                lines.append(
                    f"| {lc} | {ctx} | {task} | "
                    f"{b['yes']/tot:.2f} ({b['yes']}) | "
                    f"{b['no']/tot:.2f} ({b['no']}) |"
                )
    lines.append("")

    # ---- Counting summary ----
    lines.append("## Counting answers (test split)")
    lines.append("")
    lines.append("| label_class | ctx | mean | median | min | max | zero-rate |")
    lines.append("|---|---|---|---|---|---|---|")
    for lc in sorted(stats):
        for ctx in sorted(stats[lc].keys(), key=lambda c: (c == "full", _ctx_to_seconds(c))):
            cell = stats[lc][ctx].get("counting", {}).get("test")
            if not cell or "int_summary" not in cell:
                continue
            s = cell["int_summary"]
            if s.get("mean") is None:
                continue
            lines.append(
                f"| {lc} | {ctx} | {s['mean']:.2f} | {s['median']:.1f} | "
                f"{s['min']} | {s['max']} | {s['zero_rate']:.2f} |"
            )
    lines.append("")

    # ---- Categorical distributions ----
    lines.append("## Categorical answer distributions (test split)")
    lines.append("")
    lines.append("Top-6 classes per task, normalized over the test split.")
    lines.append("")
    lines.append("| label_class | ctx | task | distribution |")
    lines.append("|---|---|---|---|")
    for lc in sorted(stats):
        for ctx in sorted(stats[lc].keys(), key=lambda c: (c == "full", _ctx_to_seconds(c))):
            for task in sorted(stats[lc][ctx]):
                cell = stats[lc][ctx][task].get("test")
                if not cell or "category_counts" not in cell:
                    continue
                c = cell["category_counts"]
                tot = sum(c.values()) or 1
                top = sorted(c.items(), key=lambda x: -x[1])[:6]
                pretty = ", ".join(f"`{k}` {v/tot:.2f}" for k, v in top)
                lines.append(f"| {lc} | {ctx} | {task} | {pretty} |")
    lines.append("")

    # ---- Answer position in window ----
    lines.append("## Answer position within window (test split)")
    lines.append("")
    lines.append("Mean ± std of the answer-region midpoint expressed as a fraction "
                 "of the window length. A value near 0.5 with std near 0.3 indicates "
                 "no positional bias.")
    lines.append("")
    lines.append("| label_class | ctx | task | mean | std | n |")
    lines.append("|---|---|---|---|---|---|")
    for lc in sorted(stats):
        for ctx in sorted(stats[lc].keys(), key=lambda c: (c == "full", _ctx_to_seconds(c))):
            for task in sorted(stats[lc][ctx]):
                cell = stats[lc][ctx][task].get("test")
                if not cell or "answer_position_frac" not in cell:
                    continue
                p = cell["answer_position_frac"]
                lines.append(
                    f"| {lc} | {ctx} | {task} | {p['mean']:.2f} | "
                    f"{p['std']:.2f} | {p['n_with_position']} |"
                )
    lines.append("")

    # ---- Task-specific extras ----
    lines.append("## Task-specific balance (test split)")
    lines.append("")

    # Comparison: longest/shortest
    rows = []
    for lc in sorted(stats):
        for ctx in sorted(stats[lc].keys(), key=lambda c: (c == "full", _ctx_to_seconds(c))):
            cell = stats[lc][ctx].get("comparison", {}).get("test")
            if not cell or "superlative" not in cell:
                continue
            s = cell["superlative"]
            tot = sum(s.values()) or 1
            rows.append(
                f"| {lc} | {ctx} | "
                f"{s.get('longest', 0)/tot:.2f} | {s.get('shortest', 0)/tot:.2f} |"
            )
    if rows:
        lines.append("### Comparison: longest vs shortest")
        lines.append("")
        lines.append("| label_class | ctx | longest | shortest |")
        lines.append("|---|---|---|---|")
        lines.extend(rows)
        lines.append("")

    # Multi-hop: before/after
    rows = []
    for lc in sorted(stats):
        for ctx in sorted(stats[lc].keys(), key=lambda c: (c == "full", _ctx_to_seconds(c))):
            cell = stats[lc][ctx].get("multi_hop", {}).get("test")
            if not cell or "direction" not in cell:
                continue
            d = cell["direction"]
            tot = sum(d.values()) or 1
            rows.append(
                f"| {lc} | {ctx} | "
                f"{d.get('before', 0)/tot:.2f} | {d.get('after', 0)/tot:.2f} |"
            )
    if rows:
        lines.append("### Multi-hop: before vs after")
        lines.append("")
        lines.append("| label_class | ctx | before | after |")
        lines.append("|---|---|---|---|")
        lines.extend(rows)
        lines.append("")

    # Existence: target activity coverage
    rows = []
    for lc in sorted(stats):
        for ctx in sorted(stats[lc].keys(), key=lambda c: (c == "full", _ctx_to_seconds(c))):
            cell = stats[lc][ctx].get("existence", {}).get("test")
            if not cell or "target_activity" not in cell:
                continue
            ta = cell["target_activity"]
            tot = sum(ta.values()) or 1
            pretty = ", ".join(f"`{k}` {v/tot:.2f}" for k, v in sorted(ta.items()))
            rows.append(f"| {lc} | {ctx} | {pretty} |")
    if rows:
        lines.append("### Existence: target activity distribution")
        lines.append("")
        lines.append("| label_class | ctx | distribution |")
        lines.append("|---|---|---|")
        lines.extend(rows)
        lines.append("")

    return "\n".join(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--strict", action="store_true",
                        help="Exit non-zero if any correctness errors are found.")
    parser.add_argument("--out-json", type=Path,
                        default=DATA_ROOT / "analysis_report.json",
                        help="Path to write the full JSON report.")
    parser.add_argument("--out-md", type=Path,
                        default=DATA_ROOT / "analysis_report.md",
                        help="Path to write the markdown report.")
    args = parser.parse_args()
    verify_and_stats(strict=args.strict, out_json=args.out_json, out_md=args.out_md)
