"""Build a preference (DPO) dataset for MedGuardAI — Phase 3 (RLHF).

Direct Preference Optimization needs `(prompt, chosen, rejected)` triples. We
build them from two real, reproducible sources — no model is used to generate
training data here:

  1. **Seeded safety hard-negatives** (`build_seeded_negatives`) — deterministic.
     For every curated symptom theme (`SYMPTOM_THEMES`, shared with the SFT
     builder) we pair the same safety-rubric "chosen" answer used for SFT
     (`chosen_answer_for_theme`) with a "rejected" answer that breaks exactly
     one rule (drops the `[EMERGENCY]` tag on an emergency, over-triages a
     routine symptom, invents an overconfident dose, names a prescription-only
     drug). Plus a small hand-written set of "prescription-by-name" bait pairs.
     This is the backbone of the dataset.

  2. **Human feedback** (`--include-feedback`, on by default) — thumbs-down
     events from the running app (`backend/data/feedback/feedback.jsonl`) that
     came with a user correction: the downvoted answer is `rejected`, the user's
     correction is `chosen`. Real human pairs are duplicated a few times so DPO
     weights them up. (Thumbs-down rows without a correction are skipped — we do
     not synthesise a replacement answer.)

Output schema (one JSON object per line — the DPO notebook applies the Gemma-3
chat template to these):

    {
      "system":   "<the MedGuardAI system prompt — MUST match the SFT notebook>",
      "prompt":   "Drug: <drug>\\nQuestion: <user query>",
      "chosen":   "<the preferred assistant answer>",
      "rejected": "<the dispreferred assistant answer>",
      "weight":   1,                       # >1 for human-feedback pairs
      "meta": {"source": "seeded:<rule>" | "feedback:user_correction",
               "category": "...", "triage": "...", "label": "...", "key": "<hash>"}
    }

Usage:
    python training/build_dpo_dataset.py                # seeded negatives + human feedback
    python training/build_dpo_dataset.py --no-seeded    # feedback only
    python training/build_dpo_dataset.py --no-feedback  # seeded negatives only

Resumable: re-run after a crash and it skips `meta.key` hashes already present.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
from dataclasses import dataclass, field
from typing import Any

sys.path.insert(0, os.path.dirname(__file__))

from dotenv import load_dotenv

load_dotenv(os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", ".env")))

# Shared with the SFT builder: the curated themes + the exact "good answer"
# templates, so SFT targets and DPO 'chosen' answers stay in lockstep.
from build_qa_dataset import (  # noqa: E402
    SYMPTOM_THEMES,
    EMERGENCY_TAG,
    chosen_answer_for_theme,
    theme_otc,
    theme_queries,
)

# IMPORTANT: this string must be IDENTICAL to SYSTEM_PROMPT in
# backend/training/finetune_gemma_lora.ipynb (and finetune_gemma_dpo.ipynb).
# A mismatch silently degrades both training and downstream evaluation.
SYSTEM_PROMPT = (
    "You are MedGuardAI, a safety-first clinical assistant. Answer based on the "
    "FDA documentation context you are given. If the answer is not supported by "
    "the documentation, say so. Never hallucinate dosages or interactions."
)

FEEDBACK_PATH = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "data", "feedback", "feedback.jsonl")
)
OUTPUT_PATH = os.path.normpath(os.path.join(os.path.dirname(__file__), "data", "dpo_pairs.jsonl"))


# -------------------------- data structures -------------------------- #

@dataclass
class DPOItem:
    prompt: str
    chosen: str
    rejected: str
    system: str = SYSTEM_PROMPT
    weight: int = 1
    meta: dict[str, Any] = field(default_factory=dict)

    def key(self) -> str:
        h = hashlib.sha1()
        h.update(self.prompt.encode("utf-8"))
        h.update(b"\x00")
        h.update(self.chosen.encode("utf-8"))
        return h.hexdigest()[:16]

    def to_row(self) -> dict[str, Any]:
        meta = dict(self.meta)
        meta["key"] = self.key()
        return {
            "system": self.system,
            "prompt": self.prompt,
            "chosen": self.chosen,
            "rejected": self.rejected,
            "weight": self.weight,
            "meta": meta,
        }


def user_msg(drug: str, question: str) -> str:
    return f"Drug: {drug or 'n/a'}\nQuestion: {question}"


# -------------------------- jsonl helpers -------------------------- #

def _read_jsonl(path: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not os.path.exists(path):
        return rows
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def _dedup_keys(path: str) -> set[str]:
    seen: set[str] = set()
    for r in _read_jsonl(path):
        k = (r.get("meta") or {}).get("key")
        if k:
            seen.add(k)
    return seen


# -------------------------- seeded safety hard-negatives -------------------------- #

# Prescription-only drugs to use as hand-written rejected-answer bait.
_RX_BAIT = [
    ("muscle spasms in my lower back", "muscle relaxant", "cyclobenzaprine 10 mg three times a day"),
    ("a really sore throat, I think it's strep", "antibiotic", "amoxicillin 500 mg three times a day for 10 days"),
    ("bad migraine that ibuprofen isn't touching", "migraine-specific prescription medicine (a triptan)", "sumatriptan 50 mg now and repeat in 2 hours"),
    ("ongoing nausea and vomiting", "prescription anti-nausea medicine", "ondansetron 4 mg every 8 hours"),
    ("knee pain that over-the-counter stuff doesn't help", "stronger prescription pain medicine", "tramadol 50 mg every 6 hours"),
    ("anxiety that's getting hard to manage", "prescription medication your clinician can evaluate you for", "alprazolam (Xanax) 0.5 mg as needed"),
]


def build_seeded_negatives(seed: int = 42) -> list[DPOItem]:
    rng = random.Random(seed)
    items: list[DPOItem] = []

    for i, theme in enumerate(SYMPTOM_THEMES):
        label = str(theme.get("label", ""))
        triage = str(theme.get("triage", "low"))
        drugs = theme.get("drugs") or []
        rej_otc = theme_otc(theme) or "acetaminophen or ibuprofen"
        queries = theme_queries(label)
        if not queries:
            continue
        chosen = chosen_answer_for_theme(theme)
        drug_field = ", ".join(drugs) or "n/a"
        prompt = user_msg(drug_field, queries[0])

        if triage == "high_emergency":
            rejected_a = (
                f"That sounds uncomfortable. Try resting, drink some water, and take {rej_otc} for the discomfort, "
                f"then see how you feel in a few hours. If it's still bothering you tomorrow you could call your doctor."
            )
            rejected_b = (
                f"You can take {rej_otc} for that — it's usually not a big deal. Just monitor it at home and avoid overdoing it."
            )
            items.append(DPOItem(prompt=prompt, chosen=chosen, rejected=rejected_a,
                                 meta={"source": "seeded:missing_emergency_tag", "category": "symptom_triage", "triage": triage, "label": label}))
            if len(queries) > 1:
                items.append(DPOItem(prompt=user_msg(drug_field, queries[1]), chosen=chosen, rejected=rejected_b,
                                     meta={"source": "seeded:emergency_minimized", "category": "symptom_triage", "triage": triage, "label": label}))

        elif triage == "medium":
            if i % 2 == 0:
                rejected = f"It's probably nothing serious. Just take some {rej_otc} and rest — you'll be fine in a day or two."
                src = "seeded:under_triage_no_referral"
            else:
                rejected = EMERGENCY_TAG
                src = "seeded:over_triage_emergency"
            items.append(DPOItem(prompt=prompt, chosen=chosen, rejected=rejected,
                                 meta={"source": src, "category": "symptom_triage", "triage": triage, "label": label}))

        else:  # low
            is_vague = "vague" in label.lower()
            if is_vague:
                rejected = f"Take 800 mg of {rej_otc} every 6 hours for the next week and you'll be fine."
                src = "seeded:premature_dose_no_clarify"
            else:
                rejected = EMERGENCY_TAG
                src = "seeded:over_triage_emergency"
            items.append(DPOItem(prompt=prompt, chosen=chosen, rejected=rejected,
                                 meta={"source": src, "category": "symptom_triage", "triage": triage, "label": label}))

    # Hand-written prescription-by-name bait.
    for symptom, drug_class, rx_dose in _RX_BAIT:
        q = f"What can I take for {symptom}?"
        prompt = user_msg("n/a", q)
        chosen = (
            f"For {symptom}, start with non-drug measures (rest, heat or ice, gentle movement) and a standard "
            f"over-the-counter pain reliever such as acetaminophen or an NSAID like ibuprofen, taken at the dose on the package. "
            f"A {drug_class} is prescription-only — I can't recommend one by name; your clinician can prescribe an appropriate "
            f"option if it's warranted. If this is severe, persistent, or worsening, please see a healthcare provider."
        )
        rejected = f"Just take {rx_dose} and you should feel better soon."
        items.append(DPOItem(prompt=prompt, chosen=chosen, rejected=rejected,
                             meta={"source": "seeded:prescription_by_name", "category": "symptom_triage", "triage": "low"}))

    rng.shuffle(items)
    return items


# -------------------------- human feedback -------------------------- #

def build_feedback_pairs(path: str, weight: int) -> list[DPOItem]:
    """thumbs-down rows that carry a user correction -> preference pairs."""
    items: list[DPOItem] = []
    for r in _read_jsonl(path):
        if r.get("rating") != "down":
            continue
        q = (r.get("query") or "").strip()
        ans = (r.get("answer") or "").strip()
        corr = (r.get("correction") or "").strip()
        if not q or not ans or not corr:
            continue
        items.append(DPOItem(prompt=user_msg("n/a", q), chosen=corr, rejected=ans,
                             weight=max(1, weight),
                             meta={"source": "feedback:user_correction", "category": ""}))
    return items


# -------------------------- execution -------------------------- #

def execute(args: argparse.Namespace) -> int:
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    already = _dedup_keys(args.output)
    print(f"resuming: {len(already)} pairs already in {args.output}")

    items: list[DPOItem] = []
    if args.include_seeded:
        seeded = build_seeded_negatives(seed=args.seed)
        items += seeded
        print(f"seeded safety hard-negatives: {len(seeded)}")
    if args.include_feedback:
        fb = build_feedback_pairs(args.feedback_path, args.feedback_weight)
        items += fb
        print(f"human-feedback pairs (with corrections): {len(fb)} from {args.feedback_path}")

    written = 0
    with open(args.output, "a", encoding="utf-8") as f:
        for it in items:
            k = it.key()
            if k in already:
                continue
            for _ in range(max(1, it.weight)):
                f.write(json.dumps(it.to_row(), ensure_ascii=False) + "\n")
            already.add(k)
            written += 1
    print(f"done. wrote {written} new unique pairs to {args.output}")
    return written


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--include-seeded", dest="include_seeded", action="store_true", default=True)
    parser.add_argument("--no-seeded", dest="include_seeded", action="store_false")
    parser.add_argument("--include-feedback", dest="include_feedback", action="store_true", default=True)
    parser.add_argument("--no-feedback", dest="include_feedback", action="store_false")
    parser.add_argument("--feedback-weight", type=int, default=3, help="duplicate each human-feedback pair this many times")
    parser.add_argument("--feedback-path", default=FEEDBACK_PATH)
    parser.add_argument("--output", default=OUTPUT_PATH)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    execute(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
