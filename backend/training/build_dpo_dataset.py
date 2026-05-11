"""Build a preference (DPO) dataset for MedGuardAI — Phase 3 (RLHF).

Direct Preference Optimization needs `(prompt, chosen, rejected)` triples. We
build them from three complementary sources, mixed into one JSONL:

  1. **Seeded safety hard-negatives** (`build_seeded_negatives`) — deterministic,
     zero API cost. For every curated symptom theme we synthesize a "chosen"
     answer that follows the safety rubric (right triage tier, no invented
     dosing, no prescription-only drug by name, asks a clarifying question for
     vague queries) and a "rejected" answer that breaks exactly one rule
     (drops the `[EMERGENCY]` tag on an emergency, over-triages a routine
     symptom, invents an overconfident dose, names a prescription-only drug).
     This is the backbone of the dataset.

  2. **Degraded SFT answers** (`--candidate-mode degrade`, default) — sample
     prompts from `sft_pairs.jsonl`; the SFT answer (already judge-validated
     when that dataset was built) is the `chosen`; a remote LLM (DeepSeek) is
     asked to rewrite it into a subtly *worse* answer (looser safety caveat,
     over-specific dosing, removed emergency framing) which becomes `rejected`.
     An optional preference judge confirms `chosen > rejected` before keeping
     the pair.

  3. **Human feedback** (`--prompts-from feedback`) — thumbs-down events from
     the running app (`backend/data/feedback/feedback.jsonl`): the downvoted
     answer is the `rejected`; the user's correction (if any) is the `chosen`,
     otherwise a remote LLM writes an "ideal safe answer" for the query and a
     judge keeps it only if it beats the downvoted answer. Real human pairs are
     duplicated a few times so DPO weights them up.

The output schema (one JSON object per line — the DPO notebook applies the
Gemma-3 chat template to these):

    {
      "system":   "<the MedGuardAI system prompt — MUST match the SFT notebook>",
      "prompt":   "Drug: <drug>\\nQuestion: <user query>",
      "chosen":   "<the preferred assistant answer>",
      "rejected": "<the dispreferred assistant answer>",
      "weight":   1,                       # >1 for human-feedback pairs
      "meta": {"source": "seeded:<rule>" | "degrade" | "feedback:user_correction" | ...,
               "category": "...", "triage": "...", "drug": "...", "key": "<hash>"}
    }

Backends (env, same convention as build_qa_dataset.py):
    DATASET_LLM_URL / DATASET_LLM_KEY / DATASET_LLM_MODEL  -> the judge / rewriter (DeepSeek)
    LOCAL_LLM_URL / LOCAL_LLM_MODEL                        -> the SFT model (for --candidate-mode sft-vs-perturbed)

Usage:
    # cheapest: deterministic seeded negatives only
    python training/build_dpo_dataset.py --candidate-mode none

    # default: seeded negatives + degraded SFT answers (DeepSeek), judged
    python training/build_dpo_dataset.py --prompts-from sft --n-prompts 300 --concurrency 15

    # fold in collected human feedback
    python training/build_dpo_dataset.py --prompts-from feedback --include-seeded

    # use the local SFT model to generate two candidates and let the judge pick (slow — LM Studio serializes)
    python training/build_dpo_dataset.py --candidate-mode sft-vs-perturbed --n-prompts 120 --concurrency 2
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import random
import re
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Optional

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "src")))

from dotenv import load_dotenv
from openai import AsyncOpenAI

load_dotenv(os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", ".env")))

# Reuse the SFT pipeline's plumbing — async budget guard, chat helper, the
# curated symptom themes, and the symptom-task builder (for FDA excerpts).
from build_qa_dataset import (  # noqa: E402
    call_chat,
    chunk_long_text,
    make_client_and_model,
    SYMPTOM_THEMES,
    _state,
)

# IMPORTANT: this string must be IDENTICAL to SYSTEM_PROMPT in
# backend/training/finetune_gemma_lora.ipynb (and finetune_gemma_dpo.ipynb).
# A mismatch silently degrades both training and downstream evaluation.
SYSTEM_PROMPT = (
    "You are MedGuardAI, a safety-first clinical assistant. Answer based on the "
    "FDA documentation context you are given. If the answer is not supported by "
    "the documentation, say so. Never hallucinate dosages or interactions."
)

SFT_PATH = os.path.normpath(os.path.join(os.path.dirname(__file__), "data", "sft_pairs.jsonl"))
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


# -------------------------- remote-LLM prompts -------------------------- #

DEGRADE_PROMPT = """You are helping build a preference dataset for a safety-first
medical assistant. Given a GOOD answer, write a SUBTLY WORSE alternative answer
to the same question.

The worse answer should still look plausible at a glance but break exactly ONE
of these safety rules:
  - drop or weaken the "see a healthcare provider" / clarifying-question framing,
  - state a more specific dose / contraindication / interaction than the source
    supports (i.e. make something up with false confidence),
  - if the good answer contains "[EMERGENCY]" or tells the user to seek
    emergency care, REMOVE that and instead give casual at-home advice,
  - recommend a prescription-only medication BY NAME (e.g. cyclobenzaprine,
    tramadol, an antibiotic) instead of deferring to a clinician.

Keep it roughly the same length and tone. Output ONLY the worse answer text — no
preamble, no explanation, no labels.

QUESTION:
{question}

FDA CONTEXT (what is actually supported):
{context}

GOOD ANSWER:
{good}
"""

IDEAL_ANSWER_PROMPT = """You are MedGuardAI, a safety-first clinical assistant.
Write the IDEAL response to the user query below, following these rules:
  - If the query describes a true emergency (chest pain, anaphylaxis, stroke
    signs, suspected overdose, severe breathing difficulty, suicidal intent),
    briefly acknowledge it, state you are not equipped for emergencies, then
    output "[EMERGENCY] Please call your local emergency number (911 in the US)
    or get to an emergency room immediately." and do NOT recommend any medication.
  - For a vague non-emergency symptom, ask 1-2 short clarifying questions, then
    give safe non-drug guidance and a standard OTC option (no specific mg unless
    you are sure it is the labeled OTC dose), and end with "see a healthcare
    provider if symptoms persist or worsen".
  - For a clear non-emergency question, answer it directly and helpfully; for
    anything that needs evaluation, name a specific timeframe to see a clinician.
  - NEVER recommend a prescription-only medication by name. NEVER invent dosages,
    contraindications, or interactions.

Output ONLY the response text.

USER QUERY:
{question}
"""

JUDGE_PREF_PROMPT = """You are comparing two candidate answers from a safety-first
medical assistant. Pick the BETTER one.

Prefer the answer that:
  1. uses "[EMERGENCY]" / tells the user to seek emergency care IF AND ONLY IF
     the query is a true emergency;
  2. never invents a dose, contraindication, or interaction not in the FDA
     context;
  3. never names a prescription-only medication (it should defer to a clinician);
  4. asks a brief clarifying question for vague non-emergency symptoms;
  5. is helpful and concrete, not dismissive.

Reply with EXACTLY one character: "A" or "B". Nothing else.

QUERY:
{question}

FDA CONTEXT:
{context}

CANDIDATE A:
{a}

CANDIDATE B:
{b}
"""


# -------------------------- remote-LLM helpers -------------------------- #

async def degrade_answer(client: AsyncOpenAI, model: str, question: str, good: str, context: str) -> Optional[str]:
    raw = await call_chat(
        client, model,
        DEGRADE_PROMPT.format(question=question, context=chunk_long_text(context or "(none)", 2000), good=good),
        temperature=0.7,
    )
    if not raw or raw.startswith("__ERROR__") or raw == "__ABORTED__":
        return None
    raw = raw.strip()
    # If the model just echoed the good answer, skip — no contrast.
    if raw.lower() == good.strip().lower() or len(raw) < 15:
        return None
    return raw


async def ideal_answer(client: AsyncOpenAI, model: str, question: str) -> Optional[str]:
    raw = await call_chat(client, model, IDEAL_ANSWER_PROMPT.format(question=question), temperature=0.2)
    if not raw or raw.startswith("__ERROR__") or raw == "__ABORTED__" or len(raw.strip()) < 15:
        return None
    return raw.strip()


async def judge_pref(client: AsyncOpenAI, model: str, question: str, context: str, a: str, b: str) -> Optional[str]:
    """Return 'A' or 'B' (the better of the two), or None if the verdict is
    inconsistent under position swap (kills position bias) or the judge errors."""
    ctx = chunk_long_text(context or "(none)", 1800)

    async def one(x: str, y: str) -> Optional[str]:
        raw = await call_chat(client, model, JUDGE_PREF_PROMPT.format(question=question, context=ctx, a=x, b=y), temperature=0.0)
        if not raw or raw.startswith("__ERROR__") or raw == "__ABORTED__":
            return None
        raw = raw.strip().upper()
        if raw.startswith("A"):
            return "first"
        if raw.startswith("B"):
            return "second"
        return None

    v1 = await one(a, b)          # a is "first"
    v2 = await one(b, a)          # b is "first"
    if v1 is None or v2 is None:
        return None
    # Consistent iff v1 says first-better AND v2 says second-better (both pick `a`),
    # or v1 says second-better AND v2 says first-better (both pick `b`).
    picked_a = (v1 == "first") and (v2 == "second")
    picked_b = (v1 == "second") and (v2 == "first")
    if picked_a:
        return "A"
    if picked_b:
        return "B"
    return None


# -------------------------- prompt sourcing -------------------------- #

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


def load_prompts_from_sft(path: str, n: int, seed: int) -> list[dict[str, Any]]:
    """Sample n SFT rows. Returns [{question_user, gold, context, drug, category, triage}]."""
    rows = _read_jsonl(path)
    if not rows:
        return []
    rng = random.Random(seed)
    rng.shuffle(rows)
    out: list[dict[str, Any]] = []
    for r in rows:
        q = (r.get("question") or "").strip()
        a = (r.get("answer") or "").strip()
        if not q or not a:
            continue
        drug = str(r.get("drug") or "n/a")
        triage = ""
        if r.get("task_type") == "symptom":
            triage = str(r.get("section") or "")
        out.append({
            "prompt": user_msg(drug, q),
            "question": q,
            "gold": a,
            "context": str(r.get("source") or ""),
            "drug": drug,
            "category": "symptom_triage" if r.get("task_type") == "symptom" else "drug_qa",
            "triage": triage,
        })
        if len(out) >= n:
            break
    return out


def load_prompts_from_feedback(path: str) -> list[dict[str, Any]]:
    """thumbs-down rows -> [{prompt, question, rejected, correction|None}]."""
    out: list[dict[str, Any]] = []
    for r in _read_jsonl(path):
        if r.get("rating") != "down":
            continue
        q = (r.get("query") or "").strip()
        ans = (r.get("answer") or "").strip()
        if not q or not ans:
            continue
        out.append({
            "prompt": user_msg("n/a", q),
            "question": q,
            "rejected": ans,
            "correction": (r.get("correction") or None),
        })
    return out


# -------------------------- seeded safety hard-negatives -------------------------- #

_EM_TAG = "[EMERGENCY] Please call your local emergency number (911 in the US) or get to an emergency room immediately."

# Prescription-only drugs to use as hand-written rejected-answer bait.
_RX_BAIT = [
    ("muscle spasms in my lower back", "muscle relaxant", "cyclobenzaprine 10 mg three times a day"),
    ("a really sore throat, I think it's strep", "antibiotic", "amoxicillin 500 mg three times a day for 10 days"),
    ("bad migraine that ibuprofen isn't touching", "migraine-specific prescription medicine (a triptan)", "sumatriptan 50 mg now and repeat in 2 hours"),
    ("ongoing nausea and vomiting", "prescription anti-nausea medicine", "ondansetron 4 mg every 8 hours"),
    ("knee pain that over-the-counter stuff doesn't help", "stronger prescription pain medicine", "tramadol 50 mg every 6 hours"),
    ("anxiety that's getting hard to manage", "prescription medication your clinician can evaluate you for", "alprazolam (Xanax) 0.5 mg as needed"),
]


def _theme_queries(label: str) -> list[str]:
    """Cheap natural-ish phrasings derived from a theme label."""
    core = re.sub(r"\s*\(.*?\)\s*", " ", label).strip()
    core = re.sub(r"\s+", " ", core)
    paren = None
    m = re.search(r"\(([^)]+)\)", label)
    if m:
        paren = re.sub(r"\s+", " ", m.group(1)).strip()
    qs = [f"I have {core}", f"{core[:1].upper()}{core[1:]} — what should I do?"]
    if paren and len(paren) < 90 and not paren.lower().startswith("fast"):
        qs.append(f"I'm experiencing {paren}")
    # de-dup while preserving order
    seen: set[str] = set()
    uniq = []
    for q in qs:
        if q.lower() not in seen:
            seen.add(q.lower())
            uniq.append(q)
    return uniq[:2]


def _otc_for(theme: dict[str, Any]) -> str:
    drugs = theme.get("drugs") or []
    if drugs:
        return drugs[0]
    return "acetaminophen or ibuprofen"


def build_seeded_negatives(seed: int = 42) -> list[DPOItem]:
    rng = random.Random(seed)
    items: list[DPOItem] = []

    for i, theme in enumerate(SYMPTOM_THEMES):
        label = str(theme.get("label", ""))
        triage = str(theme.get("triage", "low"))
        otc = _otc_for(theme)
        core = re.sub(r"\s*\(.*?\)\s*", " ", label).strip() or label
        queries = _theme_queries(label)
        if not queries:
            continue
        q = queries[0]
        prompt = user_msg(", ".join(theme.get("drugs") or []) or "n/a", q)

        if triage == "high_emergency":
            chosen = (
                f"I hear that you're dealing with {core.lower()}, and I'm sorry — that sounds frightening. "
                f"I'm an AI assistant and I am not the right tool for an acute medical emergency. "
                f"{_EM_TAG} These situations can be time-critical, so please don't wait or rely on a chatbot for this."
            )
            rejected_a = (
                f"That sounds uncomfortable. Try resting, drink some water, and take {otc} for the discomfort, "
                f"then see how you feel in a few hours. If it's still bothering you tomorrow you could call your doctor."
            )
            rejected_b = (
                f"You can take {otc} for that — it's usually not a big deal. Just monitor it at home and avoid overdoing it."
            )
            items.append(DPOItem(prompt=prompt, chosen=chosen, rejected=rejected_a,
                                 meta={"source": "seeded:missing_emergency_tag", "category": "symptom_triage", "triage": triage, "label": label}))
            if len(queries) > 1:
                items.append(DPOItem(prompt=user_msg(", ".join(theme.get("drugs") or []) or "n/a", queries[1]),
                                     chosen=chosen, rejected=rejected_b,
                                     meta={"source": "seeded:emergency_minimized", "category": "symptom_triage", "triage": triage, "label": label}))

        elif triage == "medium":
            red_a, red_b = "trouble breathing", "confusion, fainting, or sudden severe pain"
            chosen = (
                f"This could be a few different things, and the safe move is to get it checked. "
                f"For symptom relief in the meantime, a standard over-the-counter option like {otc} can help — follow the dose on the package. "
                f"I'd recommend seeing a doctor within the next day or so for a proper evaluation. "
                f"Go to the ER right away if you develop {red_a}, {red_b}, or it suddenly gets much worse."
            )
            if i % 2 == 0:
                rejected = f"It's probably nothing serious. Just take some {otc} and rest — you'll be fine in a day or two."
                src = "seeded:under_triage_no_referral"
            else:
                rejected = _EM_TAG
                src = "seeded:over_triage_emergency"
            items.append(DPOItem(prompt=prompt, chosen=chosen, rejected=rejected,
                                 meta={"source": src, "category": "symptom_triage", "triage": triage, "label": label}))

        else:  # low
            is_vague = "vague" in label.lower() or label.lower().startswith("vague")
            chosen = (
                f"A couple of quick questions to point you in the right direction: how long has this been going on, "
                f"how bad is it, and is anything else happening alongside it? In the meantime, rest, fluids, and a standard "
                f"over-the-counter option like {otc} (follow the package dose) usually help. "
                f"If it persists beyond a few days or gets worse, please see a healthcare provider."
            )
            if is_vague:
                rejected = f"Take 800 mg of {otc} every 6 hours for the next week and you'll be fine."
                src = "seeded:premature_dose_no_clarify"
            else:
                rejected = _EM_TAG
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


# -------------------------- execution -------------------------- #

def _dedup_keys(path: str) -> set[str]:
    seen: set[str] = set()
    for r in _read_jsonl(path):
        k = (r.get("meta") or {}).get("key")
        if k:
            seen.add(k)
    return seen


async def execute(args: argparse.Namespace) -> None:
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    already = _dedup_keys(args.output)
    print(f"resuming: {len(already)} pairs already in {args.output}")

    out_f = open(args.output, "a", encoding="utf-8")
    written = 0
    lock = asyncio.Lock()

    async def emit(item: DPOItem) -> None:
        nonlocal written
        if item.key() in already:
            return
        async with lock:
            if item.key() in already:
                return
            for _ in range(max(1, item.weight)):
                out_f.write(json.dumps(item.to_row(), ensure_ascii=False) + "\n")
            out_f.flush()
            already.add(item.key())
            written += 1

    # ---- 1. seeded negatives (deterministic) ----
    if args.include_seeded:
        seeded = build_seeded_negatives(seed=args.seed)
        for it in seeded:
            await emit(it)
        print(f"seeded negatives: +{len([1 for it in seeded if it.key() in already])} (total written so far: {written})")

    # ---- 2/3. LLM-assisted pairs ----
    if args.candidate_mode == "none" and args.prompts_from != "feedback":
        out_f.close()
        print(f"done (seeded only). wrote {written} unique pairs to {args.output}")
        return

    judge_client, judge_model = make_client_and_model()
    print(f"judge/rewriter model={judge_model!r} via {judge_client.base_url}")
    gen_client, gen_model = (judge_client, judge_model)
    if args.candidate_mode == "sft-vs-perturbed":
        base_url = os.environ.get("LOCAL_LLM_URL", "http://127.0.0.1:1234/v1")
        gen_model = os.environ.get("LOCAL_LLM_MODEL", "gemma-3-4b")
        gen_client = AsyncOpenAI(base_url=base_url, api_key="not-needed", timeout=180.0)
        print(f"candidate generator (local SFT model)={gen_model!r} via {gen_client.base_url}")

    sem = asyncio.Semaphore(args.concurrency)
    started = time.time()

    async def handle_sft_prompt(p: dict[str, Any]) -> None:
        if _state["aborted"]:
            return
        async with sem:
            if _state["aborted"]:
                return
            if args.candidate_mode == "degrade":
                bad = await degrade_answer(judge_client, judge_model, p["question"], p["gold"], p["context"])
                if not bad:
                    return
                good = p["gold"]
            else:  # sft-vs-perturbed
                a = await call_chat(gen_client, gen_model,
                                    f"{SYSTEM_PROMPT}\n\n{p['prompt']}", temperature=0.0)
                b = await call_chat(gen_client, gen_model,
                                    f"{SYSTEM_PROMPT}\n\n{p['prompt']}", temperature=0.9)
                if not a or not b or a.startswith("__") or b.startswith("__") or a.strip() == b.strip():
                    return
                good, bad = a.strip(), b.strip()
            if args.judge:
                verdict = await judge_pref(judge_client, judge_model, p["question"], p["context"], good, bad)
                if verdict is None:
                    return
                if verdict == "B":
                    good, bad = bad, good
        await emit(DPOItem(prompt=p["prompt"], chosen=good, rejected=bad,
                           meta={"source": args.candidate_mode, "category": p.get("category", ""),
                                 "triage": p.get("triage", ""), "drug": p.get("drug", "")}))

    async def handle_feedback_prompt(p: dict[str, Any]) -> None:
        if _state["aborted"]:
            return
        async with sem:
            if _state["aborted"]:
                return
            if p.get("correction"):
                await emit(DPOItem(prompt=p["prompt"], chosen=p["correction"].strip(), rejected=p["rejected"],
                                   weight=max(1, args.feedback_weight),
                                   meta={"source": "feedback:user_correction", "category": ""}))
                return
            ideal = await ideal_answer(judge_client, judge_model, p["question"])
            if not ideal:
                return
            verdict = await judge_pref(judge_client, judge_model, p["question"], "", ideal, p["rejected"])
            if verdict != "A":   # only keep if the fresh answer clearly beats the downvoted one
                return
        await emit(DPOItem(prompt=p["prompt"], chosen=ideal, rejected=p["rejected"],
                           weight=max(1, args.feedback_weight),
                           meta={"source": "feedback:regenerated", "category": ""}))

    tasks = []
    if args.prompts_from == "sft" and args.candidate_mode != "none":
        prompts = load_prompts_from_sft(args.sft_path, args.n_prompts, args.seed)
        print(f"loaded {len(prompts)} prompts from {args.sft_path}")
        tasks += [handle_sft_prompt(p) for p in prompts]
    if args.prompts_from == "feedback":
        fb = load_prompts_from_feedback(args.feedback_path)
        print(f"loaded {len(fb)} thumbs-down prompts from {args.feedback_path}")
        tasks += [handle_feedback_prompt(p) for p in fb]

    # Process in batches so progress prints regularly.
    for i in range(0, len(tasks), 50):
        if _state["aborted"]:
            break
        await asyncio.gather(*tasks[i : i + 50])
        elapsed = time.time() - started
        print(f"  ...{min(i + 50, len(tasks))}/{len(tasks)} prompts processed, {written} unique pairs, {elapsed:.0f}s")

    out_f.close()
    if _state["aborted"]:
        print(f"\n!! aborted: {_state['abort_reason']}\n!! output is intact and resumable.")
    print(f"done. wrote {written} unique pairs to {args.output}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--prompts-from", choices=["sft", "feedback"], default="sft")
    parser.add_argument("--candidate-mode", choices=["degrade", "sft-vs-perturbed", "none"], default="degrade",
                        help="how to produce the rejected (or both) candidates; 'none' = seeded negatives only")
    parser.add_argument("--n-prompts", type=int, default=300, help="how many SFT prompts to build pairs for")
    parser.add_argument("--judge", dest="judge", action="store_true", default=True,
                        help="confirm chosen>rejected with the preference judge (default on)")
    parser.add_argument("--no-judge", dest="judge", action="store_false")
    parser.add_argument("--include-seeded", dest="include_seeded", action="store_true", default=True)
    parser.add_argument("--no-seeded", dest="include_seeded", action="store_false")
    parser.add_argument("--feedback-weight", type=int, default=3, help="duplicate each human-feedback pair this many times")
    parser.add_argument("--concurrency", type=int, default=15)
    parser.add_argument("--sft-path", default=SFT_PATH)
    parser.add_argument("--feedback-path", default=FEEDBACK_PATH)
    parser.add_argument("--output", default=OUTPUT_PATH)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    asyncio.run(execute(args))
    return 0


if __name__ == "__main__":
    sys.exit(main())
