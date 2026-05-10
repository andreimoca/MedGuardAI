"""Trim over-represented classes in sft_pairs.jsonl + drop tier-format mismatches.

Two passes:

  1. Drop rows whose answer doesn't match their declared triage section
     (e.g. a 'low'-tagged insomnia row whose answer is an emergency template).
     This catches off-theme drift the generator/judge let through.

  2. Down-sample the high_emergency class to a target count, balanced across
     themes (round-robin pick by theme), so the fine-tuned model isn't
     trigger-happy on emergencies.

Usage:
    # Default: drop mismatches, trim emergencies to 40
    python rebalance_dataset.py

    # Custom target per emergency theme
    python rebalance_dataset.py --emergency-keep 40

    # Dry-run: report what would change without writing
    python rebalance_dataset.py --dry-run
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import defaultdict
from typing import Any

DEFAULT_PATH = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "data", "sft_pairs.jsonl")
)


def load_all(path: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def summarise(rows: list[dict[str, Any]]) -> str:
    by_task = defaultdict(int)
    by_triage = defaultdict(int)
    for row in rows:
        by_task[row.get("task_type", "?")] += 1
        if row.get("task_type") == "symptom":
            by_triage[row.get("section", "?")] += 1
    return (
        f"  total: {len(rows)}\n"
        f"  by task_type: {dict(by_task)}\n"
        f"  symptom triage: {dict(by_triage)}"
    )


def is_tier_mismatched(row: dict[str, Any]) -> str | None:
    """Return a reason if the answer tier doesn't match the row's section."""
    if row.get("task_type") != "symptom":
        return None
    tier = (row.get("section") or "").strip()
    answer = (row.get("answer") or "").lower()

    has_emergency_marker = (
        "[emergency]" in answer
        or "call 911" in answer
        or "call your local emergency" in answer
        or "get to an emergency room" in answer
    )
    has_doctor_recommendation = (
        "see a healthcare provider" in answer
        or "see your doctor" in answer
        or "see a clinician" in answer
        or "medical evaluation" in answer
        or "see a doctor" in answer
    )

    if tier == "low" and has_emergency_marker:
        return "low tier but answer uses emergency template"
    if tier == "high_emergency" and not has_emergency_marker:
        return "high_emergency tier but answer lacks emergency template"
    if tier == "medium" and has_emergency_marker:
        return "medium tier but answer uses emergency template"
    if tier == "medium" and not has_doctor_recommendation:
        return "medium tier but answer lacks clinician referral"
    return None


def stratified_sample_per_theme(
    rows: list[dict[str, Any]], target_total: int, seed: int
) -> list[dict[str, Any]]:
    """Round-robin pick across themes until we hit target_total."""
    by_theme: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        theme = row.get("source_id") or row.get("meta", {}).get("label") or "?"
        by_theme[theme].append(row)

    rng = random.Random(seed)
    for theme in by_theme:
        rng.shuffle(by_theme[theme])

    kept: list[dict[str, Any]] = []
    themes = list(by_theme.keys())
    rng.shuffle(themes)
    pointers = {theme: 0 for theme in themes}
    while len(kept) < target_total:
        progressed = False
        for theme in themes:
            if pointers[theme] < len(by_theme[theme]):
                kept.append(by_theme[theme][pointers[theme]])
                pointers[theme] += 1
                progressed = True
                if len(kept) >= target_total:
                    break
        if not progressed:
            break
    return kept


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--path", default=DEFAULT_PATH)
    parser.add_argument(
        "--emergency-keep", type=int, default=40,
        help="target count of high_emergency symptom pairs to keep",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not os.path.exists(args.path):
        print(f"no file at {args.path}")
        return 1

    rows = load_all(args.path)
    print("BEFORE:")
    print(summarise(rows))

    # PASS 1: drop tier-format mismatches.
    cleaned: list[dict[str, Any]] = []
    dropped_reasons: dict[str, int] = defaultdict(int)
    for row in rows:
        reason = is_tier_mismatched(row)
        if reason:
            dropped_reasons[reason] += 1
        else:
            cleaned.append(row)
    if dropped_reasons:
        print("\ntier-mismatch cleanup:")
        for reason, count in dropped_reasons.items():
            print(f"  dropped {count}: {reason}")

    # PASS 2: down-sample emergencies.
    emergencies = [
        r for r in cleaned
        if r.get("task_type") == "symptom" and r.get("section") == "high_emergency"
    ]
    others = [
        r for r in cleaned
        if not (r.get("task_type") == "symptom" and r.get("section") == "high_emergency")
    ]

    if len(emergencies) <= args.emergency_keep:
        kept_emergencies = emergencies
        print(
            f"\nemergencies already at/below target "
            f"({len(emergencies)} <= {args.emergency_keep}) — keeping all"
        )
    else:
        kept_emergencies = stratified_sample_per_theme(
            emergencies, args.emergency_keep, args.seed
        )
        print(
            f"\ntrimming high_emergency: {len(emergencies)} -> {len(kept_emergencies)}"
        )

    new_rows = others + kept_emergencies
    random.Random(args.seed).shuffle(new_rows)

    print("\nAFTER:")
    print(summarise(new_rows))

    if args.dry_run:
        print("\n(dry-run, no file written)")
        return 0

    backup = args.path + ".bak"
    os.replace(args.path, backup)
    with open(args.path, "w", encoding="utf-8") as f:
        for row in new_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"\nwrote {len(new_rows)} rows to {args.path}")
    print(f"original backed up to {backup}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
