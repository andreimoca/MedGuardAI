"""Compare two or more evaluation runs and emit a single comparison markdown.

Each run is a directory under `evaluation/results/<label>/` containing
`summary.json` (always) and `cases.jsonl` (needed for `--per-case`), produced by
`run_eval.py --label <label>`. Typical workflow:

    # swap the model loaded in LM Studio between each, judge stays constant
    python run_eval.py --label base --model-override gemma-3-4b
    python run_eval.py --label sft  --model-override medguard/medguard-gemma-3-4b
    python run_eval.py --label dpo  --model-override medguard/medguard-gemma-3-4b-dpo
    python compare_runs.py --runs base sft dpo --per-case

Outputs `evaluation/results/comparison.md`.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Optional

RESULTS_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "results"))

# (key in summary by_category, display name, "higher is better"?)
DEEPEVAL_METRICS = [
    ("avg_faithfulness", "Faithfulness", True),
    ("avg_answer_relevancy", "Answer relevancy", True),
    ("avg_hallucination", "Hallucination (↓)", False),
    ("avg_medical_safety", "Medical safety (G-Eval)", True),
]
CLASSICAL_METRICS = [
    ("avg_rouge1_f", "ROUGE-1 F", "n_rouge1_f", True),
    ("avg_rougeL_f", "ROUGE-L F", "n_rougeL_f", True),
    ("avg_bleu", "BLEU", "n_bleu", True),
]


def load_summary(label: str, results_root: str) -> Optional[dict[str, Any]]:
    path = os.path.join(results_root, label, "summary.json")
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_cases(label: str, results_root: str) -> list[dict[str, Any]]:
    path = os.path.join(results_root, label, "cases.jsonl")
    rows: list[dict[str, Any]] = []
    if not os.path.exists(path):
        return rows
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def overall_metrics(summary: dict[str, Any]) -> dict[str, Optional[float]]:
    """Collapse the by_category breakdown into one number per metric."""
    by_cat = summary.get("by_category", {})
    out: dict[str, Optional[float]] = {}

    total = summary.get("total") or sum(c.get("total", 0) for c in by_cat.values())
    rule_passed = summary.get("rule_passed", sum(c.get("rule_passed", 0) for c in by_cat.values()))
    out["rule_pass_rate"] = round(rule_passed / total, 4) if total else None
    out["_rule_passed"] = rule_passed
    out["_total"] = total

    # Weighted-by-N average for latency and DeepEval metrics.
    def wavg(metric_key: str, weight_key: str = "total") -> Optional[float]:
        num = 0.0
        den = 0.0
        for c in by_cat.values():
            v = c.get(metric_key)
            w = c.get(weight_key, 0)
            if v is not None and w:
                num += v * w
                den += w
        return round(num / den, 4) if den else None

    out["avg_latency_s"] = wavg("avg_latency_s")
    for key, _name, _hib in DEEPEVAL_METRICS:
        out[key] = wavg(key)
    for key, _name, nkey, _hib in CLASSICAL_METRICS:
        out[key] = wavg(key, nkey)
        out[nkey] = int(sum(c.get(nkey, 0) for c in by_cat.values())) or None
    return out


def _fmt(v: Optional[float]) -> str:
    return "—" if v is None else f"{v:.4f}".rstrip("0").rstrip(".") if isinstance(v, float) else str(v)


def _delta(first: Optional[float], last: Optional[float], higher_is_better: bool) -> str:
    if first is None or last is None:
        return "—"
    d = last - first
    if abs(d) < 1e-9:
        return "0"
    arrow = "⬆" if (d > 0) == higher_is_better else "⬇"
    return f"{d:+.4f}".rstrip("0").rstrip(".") + f" {arrow}"


def _best_idx(values: list[Optional[float]], higher_is_better: bool) -> Optional[int]:
    cand = [(i, v) for i, v in enumerate(values) if v is not None]
    if not cand:
        return None
    return (max if higher_is_better else min)(cand, key=lambda t: t[1])[0]


def build_markdown(labels: list[str], summaries: dict[str, dict], overalls: dict[str, dict],
                   cases_by_label: dict[str, list], per_case: bool) -> str:
    first, last = labels[0], labels[-1]
    L = ["# MedGuardAI — Evaluation Comparison", "",
         f"Runs compared (in order): {', '.join('`' + l + '`' for l in labels)}.  ",
         "Same `eval_set.jsonl`, same DeepSeek judge — only the generation model differs between runs.", ""]

    # --- 1. overall table ---
    L += ["## 1. Overall", "",
          "| Metric | " + " | ".join(labels) + f" | Δ ({last}−{first}) |",
          "|---|" + "---|" * (len(labels) + 1)]

    def row(name: str, key: str, higher_is_better: bool, pct: bool = False) -> str:
        vals = [overalls[l].get(key) for l in labels]
        bi = _best_idx(vals, higher_is_better)
        cells = []
        for i, v in enumerate(vals):
            if v is None:
                cells.append("—")
            else:
                s = f"{v * 100:.1f}%" if pct else _fmt(v)
                cells.append(f"**{s}**" if i == bi and len(labels) > 1 else s)
        return f"| {name} | " + " | ".join(cells) + f" | {_delta(vals[0], vals[-1], higher_is_better)} |"

    L.append(row("Rule pass rate", "rule_pass_rate", True, pct=True))
    for key, name, hib in DEEPEVAL_METRICS:
        L.append(row(name, key, hib))
    for key, name, _nkey, hib in CLASSICAL_METRICS:
        L.append(row(name, key, hib))
    L.append(row("Avg latency (s)", "avg_latency_s", False))
    # raw counts line
    counts = " · ".join(f"{l}: {overalls[l].get('_rule_passed')}/{overalls[l].get('_total')}" for l in labels)
    L += ["", f"Rule-check raw counts — {counts}.", ""]

    # --- 2. per-category rule pass rate ---
    cats = []
    for l in labels:
        for c in summaries[l].get("by_category", {}):
            if c not in cats:
                cats.append(c)
    L += ["## 2. Rule pass rate by category", "",
          "| Category | " + " | ".join(labels) + " |",
          "|---|" + "---|" * len(labels)]
    for cat in cats:
        cells = []
        for l in labels:
            c = summaries[l].get("by_category", {}).get(cat)
            if not c:
                cells.append("—")
            else:
                cells.append(f"{c.get('rule_passed')}/{c.get('total')} ({c.get('rule_pass_rate', 0) * 100:.0f}%)")
        L.append(f"| {cat} | " + " | ".join(cells) + " |")
    L.append("")

    # --- 3. per-case flips (regressions / fixes) ---
    if per_case and cases_by_label.get(first) and cases_by_label.get(last):
        def passed_map(rows: list[dict]) -> dict[str, bool]:
            m: dict[str, bool] = {}
            for r in rows:
                q = r.get("query", "")
                m[q] = bool(r.get("rule_check", {}).get("rule_passed"))
            return m
        pm_first, pm_last = passed_map(cases_by_label[first]), passed_map(cases_by_label[last])
        common = [q for q in pm_first if q in pm_last]
        regressions = [q for q in common if pm_first[q] and not pm_last[q]]
        fixes = [q for q in common if not pm_first[q] and pm_last[q]]

        # Map query -> latest case result (for the failing-check detail).
        last_by_q = {r.get("query", ""): r for r in cases_by_label[last]}
        first_by_q = {r.get("query", ""): r for r in cases_by_label[first]}

        def fail_reason(r: dict) -> str:
            rc = r.get("rule_check", {})
            bits = []
            if rc.get("must_contain_misses"):
                bits.append(f"missing {rc['must_contain_misses']}")
            if rc.get("must_contain_any_satisfied") is False:
                bits.append("missing all must_contain_any")
            if rc.get("must_not_contain_violations"):
                bits.append(f"contains {rc['must_not_contain_violations']}")
            if rc.get("tier_match") is False:
                bits.append(f"wrong tier (expected {rc.get('tier_expected')})")
            return "; ".join(bits) or "(unknown)"

        L += [f"## 3. Per-case rule-check flips ({first} → {last})", ""]
        L.append(f"**Regressions ({len(regressions)})** — passed in `{first}`, now failing in `{last}`:")
        L.append("")
        if regressions:
            for q in regressions:
                L.append(f"- `{q[:110]}` — {fail_reason(last_by_q.get(q, {}))}")
        else:
            L.append("- _none_ 🎉")
        L += ["", f"**Newly fixed ({len(fixes)})** — failed in `{first}`, now passing in `{last}`:", ""]
        if fixes:
            for q in fixes:
                L.append(f"- `{q[:110]}` — was: {fail_reason(first_by_q.get(q, {}))}")
        else:
            L.append("- _none_")
        L.append("")

    L += ["## Notes", "",
          "- **Rule pass rate** is deterministic (must_contain / must_contain_any / must_not_contain / expected tier) — treat it as the headline number.",
          "- DeepEval / G-Eval metrics use an LLM judge and wobble ±0.05–0.10 run-to-run; read them as directional, not exact.",
          "- **Hallucination** is inverted: lower is better.",
          "- ROUGE / BLEU only cover cases that ship an `expected_output` reference answer.",
          "- All runs used the same eval set and the same judge model; only the model under test changed."]
    return "\n".join(L)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", nargs="+", required=True, help="run labels in order (e.g. base sft dpo)")
    parser.add_argument("--results-root", default=RESULTS_ROOT)
    parser.add_argument("--out", default=os.path.join(RESULTS_ROOT, "comparison.md"))
    parser.add_argument("--per-case", action="store_true", help="include the regressions/fixes section (needs cases.jsonl)")
    args = parser.parse_args()

    summaries: dict[str, dict] = {}
    overalls: dict[str, dict] = {}
    cases_by_label: dict[str, list] = {}
    for label in args.runs:
        s = load_summary(label, args.results_root)
        if s is None:
            print(f"!! no summary.json for run '{label}' under {args.results_root} — skipping")
            continue
        summaries[label] = s
        overalls[label] = overall_metrics(s)
        if args.per_case:
            cases_by_label[label] = load_cases(label, args.results_root)

    labels = [l for l in args.runs if l in summaries]
    if len(labels) < 2:
        print("need at least 2 runs with a summary.json to compare")
        return 1

    md = build_markdown(labels, summaries, overalls, cases_by_label, args.per_case)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(md)
    print(f"wrote comparison of {len(labels)} runs to {args.out}")
    # Also echo the headline numbers.
    for l in labels:
        o = overalls[l]
        print(f"  {l}: rule {o.get('_rule_passed')}/{o.get('_total')} "
              f"({(o.get('rule_pass_rate') or 0) * 100:.1f}%)  "
              f"faith={_fmt(o.get('avg_faithfulness'))}  halluc={_fmt(o.get('avg_hallucination'))}  "
              f"safety={_fmt(o.get('avg_medical_safety'))}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
