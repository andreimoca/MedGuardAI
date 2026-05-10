"""Generate a synthetic Q&A SFT dataset for MedGuardAI.

Two task types feed the same JSONL output:

  1. **drug-labels** — for each FDA drug label section, ask the LLM to invent
     question/answer pairs grounded in that section, then validate them with a
     judge prompt.

  2. **symptoms** — for each entry in a curated symptom list (e.g. "my head
     hurts", "I'm dizzy", "chest pain"), generate user-style queries with
     safety-aware responses. Emergency-tier symptoms produce strict
     "[EMERGENCY] ..." responses; lower tiers may recommend OTC drugs grounded
     in our local FDA labels; medium-tier defers to a clinician.

Calls run **asynchronously with bounded concurrency** so a frontier API like
DeepSeek finishes ~12k calls in ~10–20 min instead of ~5 hours.

Backend selection (env vars, in priority order):
    DATASET_LLM_URL / DATASET_LLM_KEY / DATASET_LLM_MODEL  (e.g. DeepSeek)
    LOCAL_LLM_URL / "not-needed" / LOCAL_LLM_MODEL         (LM Studio fallback)

Usage examples:
    # Smoke test (5 drug labels + a few symptoms), DeepSeek if key set
    python build_qa_dataset.py --max-drugs 5 --concurrency 5

    # Full run for SFT training (~3-5k pairs)
    python build_qa_dataset.py --max-drugs 1000 --concurrency 20

    # Drug labels only (skip symptoms)
    python build_qa_dataset.py --task drug-labels --max-drugs 1000

    # Symptoms only
    python build_qa_dataset.py --task symptoms

    # Skip the validator (~2x faster, lower quality)
    python build_qa_dataset.py --max-drugs 1000 --no-judge
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import re
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Iterator

sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "src")))

from dotenv import load_dotenv
from openai import AsyncOpenAI

load_dotenv(os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", ".env")))

RAW_DATA_DIR = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "data", "raw", "dailymed")
)
OUTPUT_PATH = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "data", "sft_pairs.jsonl")
)


# -------------------------- LLM client setup -------------------------- #

def make_client_and_model() -> tuple[AsyncOpenAI, str]:
    """Pick the dataset-builder LLM, falling back to local LM Studio."""
    base_url = os.environ.get("DATASET_LLM_URL")
    api_key = os.environ.get("DATASET_LLM_KEY")
    model = os.environ.get("DATASET_LLM_MODEL")

    if base_url and api_key:
        client = AsyncOpenAI(base_url=base_url, api_key=api_key, timeout=120.0)
        return client, (model or "deepseek-chat")

    # Fallback: local LM Studio (no auth, slow but free)
    base_url = os.environ.get("LOCAL_LLM_URL", "http://127.0.0.1:1234/v1")
    model = os.environ.get("LOCAL_LLM_MODEL", "gemma-3-4b")
    client = AsyncOpenAI(base_url=base_url, api_key="not-needed", timeout=120.0)
    return client, model


# -------------------------- prompts -------------------------- #

GEN_PROMPT_DRUG = """You are creating a clinical Q&A training dataset.

Below is an excerpt from the FDA label of {drug} (section: {section_name}).
Generate {n} distinct, clinically useful question/answer pairs that:
- can be answered ENTIRELY from the excerpt below (no outside knowledge),
- read like a patient or clinician would ask them,
- have answers that are concise, factual, and quote relevant phrasing from the source,
- avoid trivia (no "what is the brand color of the box").

Return ONLY a JSON array, no markdown, no commentary, like:
[
  {{"question": "...", "answer": "..."}},
  {{"question": "...", "answer": "..."}}
]

DRUG: {drug}
SECTION: {section_name}
EXCERPT:
{source}
"""

JUDGE_PROMPT_DRUG = """You are validating a clinical Q&A pair.

Question: {question}
Proposed answer: {answer}

Source excerpt (from FDA label of {drug}, section {section_name}):
{source}

Is the proposed answer fully supported by the source excerpt? Reply with EXACTLY
one word: YES or NO.
"""

GEN_PROMPT_SYMPTOM = """You are creating a clinical Q&A training dataset focused
on USER SYMPTOM QUERIES — natural, colloquial phrasings that real users send to
a medical chatbot.

Symptom theme: "{symptom_label}"
Triage tier: {triage}
Notes: {notes}

Grounding — FDA-label excerpts for the OTC medications relevant to this symptom
(use ONLY these for any dosing advice; never invent doses):
{drug_excerpts}

Generate {n} distinct realistic user queries about this symptom. Vary the
wording: short ("my head hurts"), context-heavy ("I'm 65 and have been having
back pain for 4 days"), worried ("is this an emergency?"), specific ("can I
take ibuprofen for my headache?"), terse ("dizzy"), red-flag-laden ("worst
headache of my life"), etc.

CRITICAL — COMBINATIONS:
If the theme name describes a combination (e.g. "headache with stiff neck,
fever, or photophobia", "back pain with leg weakness or numbness", "dizziness
with chest pain or palpitations"), AT LEAST {n_minus_one} of the {n} queries
MUST explicitly use the combination phrasing, NOT just the headline symptom.

Good combination phrasings (use varied colloquial wording):
- "my head hurts AND my neck is stiff"
- "headache plus fever of 102"
- "my head hurts and bright light is unbearable"
- "headache and I'm really confused"
- "back hurts AND my leg is numb"
- "lower back pain and I can't really feel my foot"
- "back pain and I lost control of my bladder"
- "dizzy AND my chest hurts"
- "I'm dizzy and my speech is slurred"
- "vertigo plus weakness on one side"

The goal: train the model to recognize that the COMBINATION (not the
individual symptom) elevates triage tier.

CRITICAL — STAY ON THEME:
Every query you generate MUST be centered on the theme symptom and stay within
the triage tier. DO NOT invent additional red-flag combinations that would
elevate the triage tier (e.g. for an insomnia/low theme, don't generate a
query with blurry vision and severe headache — that would be emergency, not
low). Stick to the tier.

For each query, generate a safety-aware assistant response. The format
depends on the tier:

==========================  LOW tier  ==========================
- ANSWER THE QUESTION. Don't dump emergency boilerplate.
- If the query is VAGUE (just "my head hurts" / "my back hurts" / etc.),
  begin with 1-2 short clarifying questions (duration, severity, associated
  symptoms), then give a brief safe baseline (non-drug measures + OTC option
  if grounded in the FDA excerpts above).
- If the query is SPECIFIC ("is ibuprofen ok for my mild headache?"), give a
  direct answer with the OTC option grounded in the FDA excerpts, citing the
  source drug name and quoting relevant dosing wording.
- ALWAYS include first-line non-drug guidance where useful (rest, heat/ice,
  hydration, sleep hygiene, RICE).
- End with: "If symptoms persist or worsen, see a healthcare provider."
- Do NOT mention "emergency", "[EMERGENCY]", or "call 911" anywhere.

==========================  MEDIUM tier  ==========================
- ANSWER THE QUESTION first — be helpful and concrete, not dismissive. Briefly
  acknowledge what the symptom might be, give non-drug guidance (rest,
  hydration, monitoring, RICE, posture, etc.).
- If the FDA excerpts above include a relevant safe OTC option, OFFER IT —
  cite the source drug name and any quoted dosing wording. The user wanted a
  helpful answer, not just a referral.
- THEN explicitly direct the user to see a clinician for proper evaluation.
  Use phrases like "I'd recommend seeing your doctor within [timeframe]" with
  a specific window: "today" for things that could escalate, "this week" for
  things that just need a checkup, "right away" if there's a chance of
  deterioration but the situation isn't ER-grade.
- Also list 2-3 specific red flags that would upgrade to an emergency (e.g.
  "If you develop X, Y, or Z, go to the ER").
- Do NOT use the "[EMERGENCY]" tag.

==========================  HIGH_EMERGENCY tier  ==========================
- Open with a brief honest acknowledgment of the symptom (1 short sentence)
  AND a clear disclaimer that you are NOT equipped to handle emergencies and
  the user should not rely on this chatbot for this situation.
- Then output exactly: "[EMERGENCY] Please call your local emergency number
  (911 in the US) or get to an emergency room immediately."
- Optionally one short clinical reason why (e.g. "These symptoms can indicate
  a stroke; treatment is time-sensitive.").
- Do NOT recommend any medication.
- Do NOT pretend you can advise on next steps beyond seeking emergency care.

Example structure for a high_emergency answer:
"I hear that you're having [symptom]. I'm an AI assistant and I am not the
right source for handling acute medical emergencies. [EMERGENCY] Please call
911 or get to an emergency room immediately. [Optional one-line reason]"

UNIVERSAL RULES (apply to all tiers):
- NEVER recommend prescription-only medications by name. If a class is
  relevant (e.g. muscle relaxant, antibiotic, opioid), mention the class and
  say "your clinician can prescribe an appropriate one". Common prescription-
  only examples to NEVER recommend: methocarbamol, cyclobenzaprine, tramadol,
  ondansetron (Rx form), sumatriptan, all antibiotics, all opioids.
- NEVER hallucinate dosages, contraindications, or interactions. If a piece
  of info is not in the FDA excerpts above, say "see a healthcare provider".
- For vague queries about non-emergency symptoms, ALWAYS ask the clarifying
  questions BEFORE recommending any drug.

Return ONLY a JSON array, no markdown, no commentary:
[
  {{"question": "...", "answer": "..."}},
  ...
]
"""

JUDGE_PROMPT_SYMPTOM = """You are validating a clinical symptom-triage Q&A pair.

User query: {question}
Proposed assistant answer: {answer}

Symptom theme: "{symptom_label}"
Triage tier: {triage}
FDA-label excerpts available:
{drug_excerpts}

Reply with EXACTLY one word: YES or NO.

Reply YES only if ALL of the following are true:

1. THEME MATCH: the user query stays centered on the theme symptom. It does
   NOT introduce additional red-flag combinations that would elevate the
   triage tier (e.g. for a "low" insomnia theme, a query about insomnia +
   blurry vision + severe headache is off-theme — reply NO).

2. TIER FORMAT MATCH:
   - For "low": the answer ANSWERS the question, gives advice/OTC where
     appropriate, includes "see a healthcare provider if persists" wording,
     and does NOT contain "[EMERGENCY]" or "911" or "call emergency".
   - For "medium": the answer answers the question briefly AND explicitly
     directs the user to see a clinician (within a timeframe). Does NOT
     contain "[EMERGENCY]" tag.
   - For "high_emergency": the answer contains "[EMERGENCY]" AND a disclaimer
     that the assistant is NOT equipped to handle emergencies, AND tells the
     user to seek immediate emergency care (911 / ER / emergency number).
     Does NOT recommend any specific medication.

3. FACTUAL GROUNDING: any medication mentioned appears in the FDA excerpts
   above. No invented dosages, contraindications, or interactions. Generic
   advice (rest, hydration, heat/ice) is fine without grounding.

4. NO PRESCRIPTION-ONLY DRUG RECOMMENDATIONS by name (methocarbamol,
   cyclobenzaprine, tramadol, sumatriptan, antibiotics, opioids). Class-level
   mentions are OK if deferred to clinician.

Otherwise reply NO.
"""


# -------------------------- curated symptoms -------------------------- #

# Each entry: theme label + list of trigger drug names (we look these up in
# our local FDA corpus to provide grounding excerpts), triage tier, notes.
SYMPTOM_THEMES: list[dict[str, Any]] = [
    {
        "label": "headache (mild/typical)",
        "triage": "low",
        "drugs": ["acetaminophen", "ibuprofen", "naproxen"],
        "notes": "Common tension/migraine. Sudden 'worst-ever' headache or with neuro symptoms is emergency.",
    },
    {
        "label": "sudden severe 'worst headache of life'",
        "triage": "medium",
        "drugs": ["acetaminophen", "ibuprofen"],
        "notes": "Answer the question (OTC NSAID is reasonable for severe headache), then strongly recommend same-day medical evaluation — sudden severe headaches can rarely indicate subarachnoid hemorrhage. Don't lead with [EMERGENCY]; lead with helpful guidance + 'see a doctor today'.",
    },
    {
        "label": "back pain",
        "triage": "low",
        "drugs": ["ibuprofen", "naproxen", "acetaminophen"],
        "notes": "Chronic >2 weeks, with fever, or with leg weakness/incontinence: see clinician urgently.",
    },
    {
        "label": "neck pain with stiffness and fever",
        "triage": "medium",
        "drugs": ["acetaminophen", "ibuprofen"],
        "notes": "Could be a stiff neck from posture/muscle tension, OR rarely meningitis. Give OTC pain advice but strongly recommend seeing a doctor same-day if fever is significant or symptoms worsen.",
    },
    {
        "label": "muscle ache",
        "triage": "low",
        "drugs": ["ibuprofen", "naproxen", "acetaminophen"],
        "notes": "Routine; rest, hydration.",
    },
    {
        "label": "joint pain",
        "triage": "low",
        "drugs": ["ibuprofen", "naproxen"],
        "notes": "Persistent, swollen, hot joint -> clinician.",
    },
    {
        "label": "dizziness / light-headed",
        "triage": "medium",
        "drugs": ["meclizine", "dimenhydrinate"],
        "notes": "Could be inner-ear, hypotension, dehydration, anemia, arrhythmia.",
    },
    {
        "label": "vertigo (room spinning)",
        "triage": "medium",
        "drugs": ["meclizine", "dimenhydrinate"],
        "notes": "BPPV common; if with weakness/double vision/slurred speech -> emergency.",
    },
    {
        "label": "chest pain or pressure",
        "triage": "high_emergency",
        "drugs": [],
        "notes": "Possible MI / PE / dissection.",
    },
    {
        "label": "shortness of breath / can't breathe",
        "triage": "high_emergency",
        "drugs": [],
        "notes": "Always urgent.",
    },
    {
        "label": "anaphylaxis (swelling face/throat, hives, breathing)",
        "triage": "high_emergency",
        "drugs": [],
        "notes": "Use epinephrine if available, then ER.",
    },
    {
        "label": "mild allergic reaction (sneezing, mild rash, itchy eyes)",
        "triage": "low",
        "drugs": ["diphenhydramine", "loratadine", "cetirizine"],
        "notes": "OTC antihistamine fine; if it progresses -> emergency.",
    },
    {
        "label": "suspected overdose",
        "triage": "high_emergency",
        "drugs": [],
        "notes": "Always emergency. Mention poison control 1-800-222-1222 in US.",
    },
    {
        "label": "stroke symptoms (FAST: face droop, arm weakness, speech)",
        "triage": "high_emergency",
        "drugs": [],
        "notes": "Time-critical.",
    },
    {
        "label": "suicidal thoughts",
        "triage": "high_emergency",
        "drugs": [],
        "notes": "Urgent. Mention 988 lifeline (US) where appropriate.",
    },
    {
        "label": "fever (mild, <102F)",
        "triage": "low",
        "drugs": ["acetaminophen", "ibuprofen"],
        "notes": "Antipyretic + fluids.",
    },
    {
        "label": "high fever (>103F or persistent)",
        "triage": "medium",
        "drugs": ["acetaminophen", "ibuprofen"],
        "notes": "OTC fine but recommend clinician if persistent.",
    },
    {
        "label": "nausea / mild upset stomach",
        "triage": "low",
        "drugs": ["meclizine", "dimenhydrinate"],
        "notes": "Hydration; if severe + abdominal pain + fever -> emergency.",
    },
    {
        "label": "severe abdominal pain",
        "triage": "medium",
        "drugs": ["acetaminophen"],
        "notes": "Many causes from gas to appendicitis. Acetaminophen OK while awaiting care (avoid NSAIDs as they can mask appendicitis). Recommend same-day clinician. With fever / vomiting / rigid abdomen / inability to walk -> emergency.",
    },
    {
        "label": "heartburn",
        "triage": "low",
        "drugs": ["famotidine", "omeprazole", "lansoprazole"],
        "notes": "OTC PPI/H2 blocker.",
    },
    {
        "label": "diarrhea (acute, no blood)",
        "triage": "low",
        "drugs": ["loperamide"],
        "notes": "Hydration; with blood or fever -> clinician.",
    },
    {
        "label": "constipation",
        "triage": "low",
        "drugs": ["docusate", "polyethylene glycol", "senna"],
        "notes": "Routine.",
    },
    {
        "label": "cold or flu symptoms (cough, congestion, mild fever)",
        "triage": "low",
        "drugs": ["dextromethorphan", "guaifenesin", "acetaminophen"],
        "notes": "Symptomatic relief.",
    },
    {
        "label": "persistent cough (>2-3 weeks)",
        "triage": "medium",
        "drugs": ["dextromethorphan"],
        "notes": "Could be many causes — clinician.",
    },
    {
        "label": "sore throat (no high fever)",
        "triage": "low",
        "drugs": ["benzocaine", "acetaminophen", "ibuprofen"],
        "notes": "Lozenges / NSAID.",
    },
    {
        "label": "insomnia (occasional)",
        "triage": "low",
        "drugs": ["diphenhydramine", "doxylamine"],
        "notes": "Sleep hygiene first; brief OTC option.",
    },
    {
        "label": "anxiety / panic attack",
        "triage": "medium",
        "drugs": [],
        "notes": "Mental-health resources; refer to clinician.",
    },
    {
        "label": "skin rash (mild, no fever)",
        "triage": "low",
        "drugs": ["hydrocortisone", "diphenhydramine"],
        "notes": "Topical hydrocortisone ok.",
    },
    {
        "label": "Stevens-Johnson-like skin reaction (blistering, mucosal involvement)",
        "triage": "medium",
        "drugs": [],
        "notes": "Stop suspected drug immediately. Strongly recommend same-day clinician / ER visit. Severe widespread blistering with airway involvement -> emergency.",
    },
    {
        "label": "toothache",
        "triage": "low",
        "drugs": ["acetaminophen", "ibuprofen", "benzocaine"],
        "notes": "OTC + dentist visit.",
    },
    # --- Eye / ear / nose / throat ---
    {
        "label": "pink eye / conjunctivitis",
        "triage": "low",
        "drugs": [],
        "notes": "Usually viral/bacterial. See clinician for prescription drops if bacterial; cool compresses meanwhile.",
    },
    {
        "label": "ear pain / earache",
        "triage": "low",
        "drugs": ["acetaminophen", "ibuprofen"],
        "notes": "OTC pain control. If fever, drainage, or > a few days -> clinician.",
    },
    {
        "label": "sty / stye on eyelid",
        "triage": "low",
        "drugs": [],
        "notes": "Warm compress 10 min, several times a day. No OTC drug needed.",
    },
    {
        "label": "ringing in ears / tinnitus",
        "triage": "medium",
        "drugs": [],
        "notes": "Sudden onset or with hearing loss -> clinician promptly.",
    },
    {
        "label": "sudden hearing loss",
        "triage": "medium",
        "drugs": [],
        "notes": "Time-sensitive — see ENT urgently (within 72 hours).",
    },
    {
        "label": "sore throat with high fever and white patches",
        "triage": "medium",
        "drugs": ["acetaminophen", "ibuprofen"],
        "notes": "Possible strep — needs clinician for testing/antibiotic.",
    },
    {
        "label": "difficulty swallowing",
        "triage": "medium",
        "drugs": [],
        "notes": "Persistent dysphagia needs work-up. With airway involvement -> emergency.",
    },
    {
        "label": "nosebleed (mild, brief)",
        "triage": "low",
        "drugs": [],
        "notes": "Pinch soft part of nose, lean forward, 10-15 min.",
    },
    {
        "label": "severe nosebleed (won't stop after 20 min)",
        "triage": "medium",
        "drugs": [],
        "notes": "Persistent bleeding needs medical attention.",
    },

    # --- Urinary ---
    {
        "label": "UTI symptoms (burning urination, frequency)",
        "triage": "medium",
        "drugs": [],
        "notes": "Needs antibiotic — see clinician. OTC azo only treats symptoms, not infection.",
    },
    {
        "label": "blood in urine",
        "triage": "medium",
        "drugs": [],
        "notes": "Always needs work-up — clinician within 24-48h.",
    },
    {
        "label": "unable to urinate / urinary retention",
        "triage": "medium",
        "drugs": [],
        "notes": "Acute urinary retention needs catheterization same-day. ER if completely unable to pass urine with bladder distension and pain.",
    },
    {
        "label": "flank or kidney pain",
        "triage": "medium",
        "drugs": ["acetaminophen", "ibuprofen"],
        "notes": "Possible kidney stone or pyelonephritis — clinician promptly; with fever -> emergency.",
    },

    # --- Reproductive ---
    {
        "label": "menstrual cramps",
        "triage": "low",
        "drugs": ["ibuprofen", "naproxen", "acetaminophen"],
        "notes": "NSAIDs first-line; heat helps. Severe disabling pain -> clinician.",
    },
    {
        "label": "yeast infection symptoms",
        "triage": "low",
        "drugs": ["miconazole", "clotrimazole"],
        "notes": "OTC antifungal is first-line; if first-time, recurrent, or symptoms unusual -> clinician.",
    },
    {
        "label": "missed period",
        "triage": "medium",
        "drugs": [],
        "notes": "Pregnancy test first. Multiple missed periods -> clinician.",
    },
    {
        "label": "medication safety in pregnancy",
        "triage": "medium",
        "drugs": [],
        "notes": "Always defer to OB/GYN. Many common OTCs are not safe in pregnancy.",
    },
    {
        "label": "erectile dysfunction question",
        "triage": "medium",
        "drugs": [],
        "notes": "Could indicate cardiovascular issue — clinician evaluation.",
    },
    {
        "label": "severe pelvic / lower abdominal pain in woman",
        "triage": "medium",
        "drugs": ["acetaminophen"],
        "notes": "Many causes — menstrual, ovarian cyst, ectopic pregnancy, PID. Same-day clinician; with fever or possible pregnancy -> emergency.",
    },
    {
        "label": "sudden severe testicular pain",
        "triage": "high_emergency",
        "drugs": [],
        "notes": "Testicular torsion is surgical emergency (6-hour window). KEEP.",
    },
    {
        "label": "heavy vaginal bleeding (especially in pregnancy)",
        "triage": "high_emergency",
        "drugs": [],
        "notes": "In pregnancy = emergency. KEEP.",
    },

    # --- Skin variants ---
    {
        "label": "athlete's foot",
        "triage": "low",
        "drugs": ["miconazole", "clotrimazole", "tolnaftate"],
        "notes": "OTC antifungal for 2-4 weeks. Diabetics or non-resolving -> clinician.",
    },
    {
        "label": "eczema flare",
        "triage": "low",
        "drugs": ["hydrocortisone"],
        "notes": "Low-potency OTC steroid + moisturizer. Persistent -> dermatologist.",
    },
    {
        "label": "sunburn",
        "triage": "low",
        "drugs": ["ibuprofen", "acetaminophen", "hydrocortisone"],
        "notes": "NSAID for pain, aloe topically. Blistering / large area / fever -> clinician.",
    },
    {
        "label": "insect bite (mosquito, flea)",
        "triage": "low",
        "drugs": ["diphenhydramine", "hydrocortisone"],
        "notes": "Antihistamine + topical steroid for itch. Spreading redness / fever -> clinician.",
    },
    {
        "label": "bee or wasp sting (no allergic history)",
        "triage": "low",
        "drugs": ["diphenhydramine", "acetaminophen", "ibuprofen", "hydrocortisone"],
        "notes": "Local care. Any sign of systemic allergic reaction -> emergency.",
    },
    {
        "label": "poison ivy / contact dermatitis",
        "triage": "low",
        "drugs": ["hydrocortisone", "diphenhydramine"],
        "notes": "Topical steroid + antihistamine. Large area or face -> clinician.",
    },
    {
        "label": "acne",
        "triage": "low",
        "drugs": [],
        "notes": "OTC benzoyl peroxide / salicylic acid; persistent -> dermatologist.",
    },
    {
        "label": "spreading red rash with fever",
        "triage": "medium",
        "drugs": [],
        "notes": "Possible cellulitis or serious infection — clinician same day.",
    },
    {
        "label": "new or changing mole",
        "triage": "medium",
        "drugs": [],
        "notes": "Dermatology referral for ABCDE assessment.",
    },
    {
        "label": "shingles-like painful rash",
        "triage": "medium",
        "drugs": ["acetaminophen", "ibuprofen"],
        "notes": "Antiviral works best within 72 hours — clinician promptly.",
    },

    # --- Neurological ---
    {
        "label": "numbness or tingling in extremities",
        "triage": "medium",
        "drugs": [],
        "notes": "Sudden, one-sided, or with weakness/speech change -> emergency (stroke).",
    },
    {
        "label": "tremor / new shaking",
        "triage": "medium",
        "drugs": [],
        "notes": "Neurology work-up. Acute with confusion -> emergency.",
    },
    {
        "label": "memory concerns",
        "triage": "medium",
        "drugs": [],
        "notes": "Clinician work-up; acute change -> emergency.",
    },
    {
        "label": "seizure (witnessed or first-time)",
        "triage": "high_emergency",
        "drugs": [],
        "notes": "Any first-time seizure is an emergency.",
    },
    {
        "label": "sudden vision loss or double vision",
        "triage": "high_emergency",
        "drugs": [],
        "notes": "Possible stroke / retinal artery occlusion.",
    },
    {
        "label": "sudden one-sided weakness or facial droop",
        "triage": "high_emergency",
        "drugs": [],
        "notes": "FAST stroke signs.",
    },
    {
        "label": "migraine (known pattern)",
        "triage": "low",
        "drugs": ["acetaminophen", "ibuprofen", "naproxen"],
        "notes": "OTC for mild attacks; new/worst-ever pattern -> emergency.",
    },

    # --- Pediatric ---
    {
        "label": "ear infection in child",
        "triage": "low",
        "drugs": ["acetaminophen", "ibuprofen"],
        "notes": "OTC pain control; persistent or with high fever -> pediatrician.",
    },
    {
        "label": "fever in infant under 3 months",
        "triage": "high_emergency",
        "drugs": [],
        "notes": "Standard pediatric rule: any rectal temp >=100.4F in this age requires immediate ER. KEEP.",
    },
    {
        "label": "fever in toddler 100-102F",
        "triage": "low",
        "drugs": ["acetaminophen", "ibuprofen"],
        "notes": "Weight-based dosing — defer to pediatrician for exact dose.",
    },
    {
        "label": "high fever in child >103F or lasting >3 days",
        "triage": "medium",
        "drugs": ["acetaminophen", "ibuprofen"],
        "notes": "Pediatrician same-day.",
    },
    {
        "label": "croup (barking cough in child)",
        "triage": "medium",
        "drugs": [],
        "notes": "Cool mist, calm. Stridor at rest or severe distress -> emergency.",
    },
    {
        "label": "rash in child with fever",
        "triage": "medium",
        "drugs": [],
        "notes": "Could be many things; pediatrician same-day. Petechiae / non-blanching -> emergency.",
    },
    {
        "label": "vomiting in baby/toddler with signs of dehydration",
        "triage": "medium",
        "drugs": [],
        "notes": "Sunken eyes, no tears, dry diapers — clinician promptly.",
    },

    # --- Geriatric ---
    {
        "label": "fall in elderly without head impact",
        "triage": "medium",
        "drugs": ["acetaminophen", "ibuprofen"],
        "notes": "Evaluate for injuries; hip/wrist pain -> imaging.",
    },
    {
        "label": "fall in elderly with head impact, LOC, or anticoagulant use",
        "triage": "medium",
        "drugs": ["acetaminophen"],
        "notes": "Acetaminophen for pain (avoid NSAIDs if anticoagulated). Same-day clinician evaluation. With persistent vomiting, worsening confusion, or focal neurological signs -> emergency.",
    },
    {
        "label": "sudden confusion in elderly (delirium)",
        "triage": "medium",
        "drugs": [],
        "notes": "Often UTI or pneumonia or med interaction. Same-day evaluation. With focal neuro signs or stroke pattern -> emergency.",
    },
    {
        "label": "urinary frequency / urgency in elderly",
        "triage": "medium",
        "drugs": [],
        "notes": "UTI, BPH, diabetes — clinician evaluation.",
    },

    # --- Functional / Digestive ---
    {
        "label": "bloating",
        "triage": "low",
        "drugs": ["famotidine"],
        "notes": "Diet/gut habits; persistent with weight loss -> clinician.",
    },
    {
        "label": "gas / flatulence",
        "triage": "low",
        "drugs": [],
        "notes": "Simethicone OTC. Persistent -> clinician.",
    },
    {
        "label": "hemorrhoids",
        "triage": "low",
        "drugs": ["hydrocortisone", "acetaminophen"],
        "notes": "Fiber, sitz baths, topical hydrocortisone. Significant bleeding -> clinician.",
    },
    {
        "label": "indigestion",
        "triage": "low",
        "drugs": ["famotidine", "omeprazole", "lansoprazole"],
        "notes": "OTC H2 blocker / PPI. Persistent or with weight loss -> clinician.",
    },
    {
        "label": "food poisoning (mild, no blood)",
        "triage": "low",
        "drugs": ["loperamide"],
        "notes": "Hydration. Bloody stools / high fever / signs of dehydration -> clinician.",
    },

    # --- Injury ---
    {
        "label": "minor cut",
        "triage": "low",
        "drugs": [],
        "notes": "Pressure, clean, bandage. Deep / gaping / on face / animal bite -> clinician.",
    },
    {
        "label": "sprained ankle or wrist",
        "triage": "low",
        "drugs": ["ibuprofen", "naproxen"],
        "notes": "RICE + NSAID. Inability to bear weight -> imaging.",
    },
    {
        "label": "minor burn (small, 1st degree)",
        "triage": "low",
        "drugs": ["ibuprofen", "acetaminophen"],
        "notes": "Cool water, aloe. Blisters > 3 inches or on face/joints -> clinician.",
    },
    {
        "label": "severe burn (large, deep, blistering)",
        "triage": "high_emergency",
        "drugs": [],
        "notes": "Burns >10% BSA, full thickness, or airway are true emergencies. KEEP.",
    },
    {
        "label": "head injury without loss of consciousness or vomiting",
        "triage": "low",
        "drugs": ["acetaminophen"],
        "notes": "Acetaminophen for pain (avoid NSAIDs early due to bleeding concern). Watch for 24h. If symptoms develop (vomiting, confusion) -> medical attention.",
    },
    {
        "label": "head injury with confusion, vomiting, or LOC",
        "triage": "high_emergency",
        "drugs": [],
        "notes": "Active neurological signs after head trauma — possible intracranial bleed. KEEP.",
    },
    {
        "label": "animal bite (cat, dog, or wild)",
        "triage": "medium",
        "drugs": [],
        "notes": "Infection / rabies / tetanus — clinician same-day.",
    },

    # --- Respiratory ---
    {
        "label": "wheezing (suggestive of asthma)",
        "triage": "medium",
        "drugs": [],
        "notes": "Needs evaluation; severe wheeze / can't speak in sentences -> emergency.",
    },
    {
        "label": "coughing up blood",
        "triage": "medium",
        "drugs": [],
        "notes": "Streaks may be from coughing irritation. Same-day clinician. Large volumes, breathing difficulty, or chest pain -> emergency.",
    },
    {
        "label": "severe asthma attack (can't speak in full sentences)",
        "triage": "high_emergency",
        "drugs": [],
        "notes": "Inability to speak in sentences = severe airway distress. KEEP.",
    },

    # --- Cardiovascular ---
    {
        "label": "heart palpitations / racing heart",
        "triage": "medium",
        "drugs": [],
        "notes": "With chest pain / dizziness / syncope -> emergency.",
    },
    {
        "label": "one-sided swollen leg with pain (possible DVT)",
        "triage": "medium",
        "drugs": [],
        "notes": "Possible DVT. Recommend same-day clinician evaluation (D-dimer, ultrasound). With shortness of breath or chest pain (possible PE) -> emergency.",
    },
    {
        "label": "fainting / syncope",
        "triage": "medium",
        "drugs": [],
        "notes": "First-ever syncope or with palpitations / chest pain -> emergency.",
    },

    # --- Mental health (additional) ---
    {
        "label": "depression symptoms (low mood >2 weeks)",
        "triage": "medium",
        "drugs": [],
        "notes": "Clinician evaluation. Suicidal ideation -> emergency.",
    },
    {
        "label": "substance withdrawal symptoms (alcohol, opioid)",
        "triage": "medium",
        "drugs": [],
        "notes": "Alcohol withdrawal can be life-threatening — clinician promptly.",
    },

    # --- Medication-management questions ---
    {
        "label": "missed a dose of my regular medication",
        "triage": "low",
        "drugs": [],
        "notes": "General guidance: take as soon as remembered unless near next dose; never double. Specifics depend on drug.",
    },
    {
        "label": "should I stop taking my antibiotic now that I feel better",
        "triage": "low",
        "drugs": [],
        "notes": "Always finish the prescribed course.",
    },
    {
        "label": "can I drink alcohol with my medication",
        "triage": "medium",
        "drugs": [],
        "notes": "Drug-specific — defer to clinician / pharmacist.",
    },

    # --- RED-FLAG VARIANTS: specific patterns the model must learn to triage ---
    # These cover edge cases identified during agent gap analysis. They make
    # the fine-tuned model recognize specific dangerous phrasings.
    {
        "label": "thunderclap headache (sudden peak in seconds)",
        "triage": "medium",
        "drugs": ["acetaminophen", "ibuprofen"],
        "notes": "OTC pain control reasonable for a severe headache, but recommend medical evaluation same-day because true thunderclap onset can rarely indicate subarachnoid hemorrhage.",
    },
    {
        "label": "headache after head injury (any time after, even days)",
        "triage": "medium",
        "drugs": ["acetaminophen"],
        "notes": "Acetaminophen OK; avoid NSAIDs initially (bleeding concern). Recommend medical evaluation, especially if vomiting, confusion, worsening pain, or anticoagulants.",
    },
    {
        "label": "headache with stiff neck, fever, or photophobia",
        "triage": "medium",
        "drugs": ["acetaminophen", "ibuprofen"],
        "notes": "OTC pain control reasonable. Recommend same-day clinician evaluation — these features can rarely indicate meningitis.",
    },
    {
        "label": "headache with confusion, weakness, or vision change",
        "triage": "high_emergency",
        "drugs": [],
        "notes": "Stroke pattern — time-critical because thrombolytics work within a narrow window. KEEP as emergency.",
    },
    {
        "label": "black tarry stool (melena)",
        "triage": "medium",
        "drugs": [],
        "notes": "Suggests upper GI bleed. Recommend medical evaluation same-day; if dizziness/weakness/significant blood loss -> emergency.",
    },
    {
        "label": "vomiting blood (hematemesis)",
        "triage": "high_emergency",
        "drugs": [],
        "notes": "Active visible GI bleed = emergency. KEEP.",
    },
    {
        "label": "abdominal pain with high fever",
        "triage": "medium",
        "drugs": ["acetaminophen"],
        "notes": "OTC antipyretic OK while awaiting care. Strongly recommend same-day evaluation — could be appendicitis or other surgical issue.",
    },
    {
        "label": "right lower quadrant abdominal pain (possible appendicitis)",
        "triage": "medium",
        "drugs": [],
        "notes": "Do NOT recommend NSAIDs (could mask). Recommend medical evaluation same-day. Worsening with fever / vomiting / inability to walk -> emergency.",
    },
    {
        "label": "back pain with leg weakness or numbness",
        "triage": "medium",
        "drugs": ["acetaminophen"],
        "notes": "Acetaminophen for pain; strongly recommend same-day evaluation. Rare but cauda equina is surgical urgency.",
    },
    {
        "label": "back pain with loss of bladder or bowel control",
        "triage": "high_emergency",
        "drugs": [],
        "notes": "Cauda equina with sphincter involvement = surgical emergency. KEEP.",
    },
    {
        "label": "back pain after fall or trauma",
        "triage": "medium",
        "drugs": ["acetaminophen"],
        "notes": "Compression fracture risk especially in elderly / osteoporosis. Avoid early NSAIDs after fall (bleeding concern); use acetaminophen.",
    },
    {
        "label": "back pain with fever or unexplained weight loss",
        "triage": "medium",
        "drugs": [],
        "notes": "Infection (osteomyelitis, epidural abscess) or malignancy red flags.",
    },
    {
        "label": "dizziness with chest pain or palpitations",
        "triage": "high_emergency",
        "drugs": [],
        "notes": "Cardiac arrhythmia or ischemia — KEEP as emergency because chest pain dominates.",
    },
    {
        "label": "dizziness with slurred speech, facial droop, or one-sided weakness",
        "triage": "high_emergency",
        "drugs": [],
        "notes": "Active stroke (FAST positive) — KEEP as emergency, time-critical for thrombolytics.",
    },
    {
        "label": "sudden vertigo with sustained imbalance or new neurological signs",
        "triage": "medium",
        "drugs": [],
        "notes": "Could be benign positional vertigo or rarely cerebellar stroke. Recommend same-day clinician unless clearly FAST-positive stroke signs (then emergency).",
    },
    {
        "label": "presyncope / 'about to faint' / 'blacked out for a moment'",
        "triage": "medium",
        "drugs": [],
        "notes": "Evaluate cause; with chest pain / palpitations / exertion -> emergency.",
    },
    {
        "label": "near-fainting on standing (orthostatic)",
        "triage": "medium",
        "drugs": [],
        "notes": "Often dehydration / medication side effect; persistent or recurrent -> clinician.",
    },

    # --- VAGUE SYMPTOM PATTERNS — model must learn to ask clarifying questions ---
    {
        "label": "vague headache ('my head hurts', no other context)",
        "triage": "low",
        "drugs": ["acetaminophen", "ibuprofen", "naproxen"],
        "notes": "Model should ask about duration, severity, prior history, associated symptoms (fever, vision change, neck stiffness) BEFORE giving any drug advice. Sudden 'worst-ever' or with red flags -> emergency.",
    },
    {
        "label": "vague stomach pain ('my stomach hurts', no other context)",
        "triage": "low",
        "drugs": ["famotidine", "omeprazole", "loperamide"],
        "notes": "Model should ask about location (epigastric vs lower abdomen), nature (cramping/sharp/burning), duration, associated symptoms (fever, vomiting, blood) BEFORE drug advice. Severe / sudden / with red flags -> emergency.",
    },
    {
        "label": "vague back pain ('my back hurts', no other context)",
        "triage": "low",
        "drugs": ["ibuprofen", "naproxen", "acetaminophen"],
        "notes": "Model should ask about duration, location, mechanism (trauma vs no trauma), red flags (leg weakness, bladder/bowel control, fever). Recommend rest + heat/ice + OTC NSAID. NEVER recommend prescription muscle relaxants (e.g. methocarbamol) — defer to clinician for those.",
    },
    {
        "label": "vague dizziness ('I'm dizzy', no other context)",
        "triage": "medium",
        "drugs": [],
        "notes": "Model should ask: is it spinning (vertigo) vs light-headed (presyncope), how long, on standing or constant, any neuro symptoms, any chest pain / palpitations. With ANY red flags -> emergency.",
    },

    # --- ADDITIONAL EVERYDAY LOW-TIER THEMES ---
    # These broaden the low-tier coverage so the fine-tuned model isn't biased
    # toward over-triaging. Most real user queries are this kind of routine
    # complaint.
    {
        "label": "common cold (head congestion, runny nose, sneezing)",
        "triage": "low",
        "drugs": ["acetaminophen", "ibuprofen", "guaifenesin", "dextromethorphan"],
        "notes": "Symptomatic relief; rest, fluids. See clinician if fever > 3 days or worsening.",
    },
    {
        "label": "seasonal allergies / hay fever",
        "triage": "low",
        "drugs": ["loratadine", "cetirizine", "diphenhydramine"],
        "notes": "Non-sedating antihistamine first-line.",
    },
    {
        "label": "hangover symptoms (headache, nausea, fatigue)",
        "triage": "low",
        "drugs": ["acetaminophen", "ibuprofen"],
        "notes": "Hydration, electrolytes, rest. Avoid acetaminophen if heavy/chronic alcohol use (hepatotoxicity).",
    },
    {
        "label": "motion sickness",
        "triage": "low",
        "drugs": ["meclizine", "dimenhydrinate"],
        "notes": "OTC antihistamine 30-60 min before travel.",
    },
    {
        "label": "jet lag",
        "triage": "low",
        "drugs": ["diphenhydramine"],
        "notes": "Light exposure, melatonin, gradual schedule shift. Brief sleep aid OK short-term.",
    },
    {
        "label": "mild dehydration after exercise",
        "triage": "low",
        "drugs": [],
        "notes": "Water + electrolytes. No drug needed.",
    },
    {
        "label": "dry / irritated eyes",
        "triage": "low",
        "drugs": [],
        "notes": "Artificial tears OTC. Persistent / painful / vision change -> ophthalmologist.",
    },
    {
        "label": "canker sore in mouth",
        "triage": "low",
        "drugs": ["benzocaine"],
        "notes": "Topical OTC anaesthetic. Most heal in 1-2 weeks.",
    },
    {
        "label": "chapped / cracked lips",
        "triage": "low",
        "drugs": [],
        "notes": "Lip balm + hydration. Persistent splitting -> clinician.",
    },
    {
        "label": "dandruff / flaky scalp",
        "triage": "low",
        "drugs": [],
        "notes": "OTC anti-dandruff shampoo (zinc pyrithione, salicylic acid).",
    },
    {
        "label": "minor foot blister",
        "triage": "low",
        "drugs": [],
        "notes": "Cover with bandage, do not pop. Watch for infection signs.",
    },
    {
        "label": "razor burn / shaving irritation",
        "triage": "low",
        "drugs": ["hydrocortisone"],
        "notes": "Cool compress, low-potency hydrocortisone OK briefly.",
    },
    {
        "label": "leg cramp at night (charlie horse)",
        "triage": "low",
        "drugs": [],
        "notes": "Stretch + hydration. Recurrent / severe -> clinician (electrolytes / med side effects).",
    },
    {
        "label": "tension neck/shoulder pain from desk work",
        "triage": "low",
        "drugs": ["ibuprofen", "naproxen", "acetaminophen"],
        "notes": "Stretching, posture, OTC NSAID short-term.",
    },
    {
        "label": "eye strain from screens",
        "triage": "low",
        "drugs": [],
        "notes": "20-20-20 rule, lubricating drops. No drug needed.",
    },
    {
        "label": "caffeine withdrawal headache",
        "triage": "low",
        "drugs": ["acetaminophen", "ibuprofen"],
        "notes": "Self-limited in a few days. OTC analgesic short-term.",
    },
    {
        "label": "sinus pressure / mild sinus congestion",
        "triage": "low",
        "drugs": ["ibuprofen", "acetaminophen"],
        "notes": "Saline irrigation, decongestant. Worsening / fever > 7-10 days -> clinician.",
    },
    {
        "label": "itchy scalp without rash",
        "triage": "low",
        "drugs": [],
        "notes": "Often dryness / dandruff. Anti-dandruff or moisturizing shampoo.",
    },
    {
        "label": "diaper rash (mild)",
        "triage": "low",
        "drugs": [],
        "notes": "Frequent diaper changes, barrier cream (zinc oxide). Worsening / blisters -> pediatrician.",
    },
    {
        "label": "heat rash",
        "triage": "low",
        "drugs": ["hydrocortisone"],
        "notes": "Cool environment, loose clothing. Brief hydrocortisone for itch.",
    },
    {
        "label": "mild stomach flu (vomiting + diarrhea, no blood, no high fever)",
        "triage": "low",
        "drugs": ["loperamide"],
        "notes": "Hydration is the priority. Bloody / high fever / dehydration signs -> clinician.",
    },
]


# -------------------------- I/O helpers -------------------------- #

def load_label(filepath: str) -> dict[str, Any] | None:
    try:
        with open(filepath, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def join_section(label: dict[str, Any], key: str) -> str:
    parts = label.get(key) or []
    return "\n".join(str(p) for p in parts).strip()


def chunk_long_text(text: str, max_chars: int = 2400) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + " ... [truncated]"


def iter_drug_label_files() -> Iterator[str]:
    if not os.path.isdir(RAW_DATA_DIR):
        return
    for name in sorted(os.listdir(RAW_DATA_DIR)):
        if name.endswith(".json"):
            yield os.path.join(RAW_DATA_DIR, name)


def already_processed_keys(output_path: str) -> set[str]:
    if not os.path.exists(output_path):
        return set()
    seen: set[str] = set()
    with open(output_path, encoding="utf-8") as f:
        for line in f:
            try:
                row = json.loads(line)
                seen.add(f"{row.get('task_type')}|{row.get('drug')}|{row.get('section')}|{row.get('source_id')}")
            except json.JSONDecodeError:
                continue
    return seen


def parse_json_array(raw: str) -> list[dict[str, str]]:
    raw = raw.strip()
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)
    start = raw.find("[")
    end = raw.rfind("]")
    if start == -1 or end == -1 or end <= start:
        return []
    try:
        parsed = json.loads(raw[start : end + 1])
        if isinstance(parsed, list):
            return [
                p for p in parsed
                if isinstance(p, dict) and "question" in p and "answer" in p
            ]
    except json.JSONDecodeError:
        pass
    return []


# -------------------------- task definitions -------------------------- #

@dataclass
class Task:
    """One unit of dataset-generation work."""
    task_type: str  # "drug_label" | "symptom"
    drug: str       # for drug_label: actual drug name; for symptom: comma-separated context drugs
    section: str    # for drug_label: FDA section; for symptom: triage tier
    source: str     # text passed to the gen prompt
    source_id: str  # unique id for resume tracking
    extras: dict[str, Any] = field(default_factory=dict)

    def gen_prompt(self, n: int) -> str:
        if self.task_type == "drug_label":
            return GEN_PROMPT_DRUG.format(
                drug=self.drug, section_name=self.section,
                source=chunk_long_text(self.source), n=n,
            )
        else:
            return GEN_PROMPT_SYMPTOM.format(
                symptom_label=self.extras["label"],
                triage=self.extras["triage"],
                notes=self.extras["notes"],
                drug_excerpts=chunk_long_text(self.source, 3000),
                n=n,
                n_minus_one=max(1, n - 1),
            )

    def judge_prompt(self, question: str, answer: str) -> str:
        if self.task_type == "drug_label":
            return JUDGE_PROMPT_DRUG.format(
                drug=self.drug, section_name=self.section,
                source=chunk_long_text(self.source),
                question=question, answer=answer,
            )
        else:
            return JUDGE_PROMPT_SYMPTOM.format(
                symptom_label=self.extras["label"],
                triage=self.extras["triage"],
                drug_excerpts=chunk_long_text(self.source, 3000),
                question=question, answer=answer,
            )


# -------------------------- task builders -------------------------- #

DRUG_LABEL_SECTIONS = [
    ("dosage_and_administration", "dosage", 3),
    ("contraindications", "contraindications", 3),
    ("drug_interactions", "interactions", 3),
    ("warnings_and_cautions", "warnings", 2),
    ("indications_and_usage", "indications", 2),
    ("adverse_reactions", "adverse_reactions", 1),
    ("overdosage", "overdose", 1),
]


def pick_sections_for_drug(label: dict[str, Any], k: int) -> list[tuple[str, str, str]]:
    """Weighted-sample up to k populated sections from a drug label."""
    populated: list[tuple[str, str, int, str]] = []
    for key, name, weight in DRUG_LABEL_SECTIONS:
        text = join_section(label, key)
        if len(text) >= 200:
            populated.append((key, name, weight, text))

    chosen: list[tuple[str, str, str]] = []
    pool = list(populated)
    while pool and len(chosen) < k:
        weights = [w for _, _, w, _ in pool]
        total = sum(weights)
        r = random.random() * total
        upto = 0
        idx = len(pool) - 1
        for i, (_, _, w, _) in enumerate(pool):
            upto += w
            if upto >= r:
                idx = i
                break
        key, name, _, text = pool.pop(idx)
        chosen.append((key, name, text))
    return chosen


def build_drug_label_tasks(max_drugs: int, sections_per_drug: int) -> list[Task]:
    tasks: list[Task] = []
    files = list(iter_drug_label_files())[:max_drugs]
    for filepath in files:
        label = load_label(filepath)
        if not label:
            continue
        drug = str(label.get("drug_name") or os.path.basename(filepath))
        for _section_key, section_name, source in pick_sections_for_drug(label, k=sections_per_drug):
            tasks.append(Task(
                task_type="drug_label",
                drug=drug,
                section=section_name,
                source=source,
                source_id=os.path.basename(filepath),
            ))
    return tasks


def build_symptom_tasks() -> list[Task]:
    """Look up FDA excerpts for each theme's drugs and build a Task per theme."""
    # Reuse the agent's find_label so we benefit from its scoring.
    from agent.tools._data import find_label, join_section as agent_join

    tasks: list[Task] = []
    for theme in SYMPTOM_THEMES:
        excerpts: list[str] = []
        for d in theme["drugs"]:
            label = find_label(d, prefer_section="dosage_and_administration")
            if not label:
                continue
            dose_text = agent_join(label, "dosage_and_administration")
            contra_text = agent_join(label, "contraindications")
            block_parts = [f"## {label.get('drug_name', d).upper()}"]
            if dose_text:
                block_parts.append(f"DOSAGE: {chunk_long_text(dose_text, 700)}")
            if contra_text:
                block_parts.append(f"CONTRAINDICATIONS: {chunk_long_text(contra_text, 500)}")
            excerpts.append("\n".join(block_parts))
        source = "\n\n".join(excerpts) if excerpts else "(no FDA excerpts available — emergency-tier or no OTC relevant)"
        slug = re.sub(r"[^a-z0-9]+", "_", theme["label"].lower()).strip("_")
        tasks.append(Task(
            task_type="symptom",
            drug=", ".join(theme["drugs"]) or "n/a",
            section=theme["triage"],
            source=source,
            source_id=f"symptom::{slug}",
            extras={
                "label": theme["label"],
                "triage": theme["triage"],
                "notes": theme["notes"],
            },
        ))
    return tasks


# -------------------------- async execution -------------------------- #

_BUDGET_KEYWORDS = (
    "insufficient_quota",
    "insufficient balance",
    "insufficient_balance",
    "exceeded your quota",
    "402",
    "payment required",
    "billing",
)

# Shared mutable state used by all workers to short-circuit cleanly once we
# detect the API key has run out of credit. asyncio is single-threaded so a
# plain dict is safe.
_state: dict[str, Any] = {"aborted": False, "abort_reason": ""}


async def call_chat(client: AsyncOpenAI, model: str, prompt: str, temperature: float) -> str:
    if _state["aborted"]:
        return "__ABORTED__"
    try:
        resp = await client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=2000,
        )
        return (resp.choices[0].message.content or "").strip()
    except Exception as exc:
        msg = str(exc).lower()
        if any(kw in msg for kw in _BUDGET_KEYWORDS) and not _state["aborted"]:
            _state["aborted"] = True
            _state["abort_reason"] = str(exc)
            print(
                f"\n!!! Budget/auth error detected — aborting remaining tasks."
                f"\n!!! Reason: {exc}"
                f"\n!!! Output written so far is intact and resumable.\n"
            )
        return f"__ERROR__: {exc}"


async def run_task(
    task: Task,
    client: AsyncOpenAI,
    model: str,
    n_pairs: int,
    judge: bool,
) -> tuple[list[dict[str, Any]], int]:
    """Returns (validated_rows, rejected_count)."""
    raw = await call_chat(client, model, task.gen_prompt(n_pairs), temperature=0.4)
    if raw.startswith("__ERROR__"):
        return [], 0
    pairs = parse_json_array(raw)

    rows: list[dict[str, Any]] = []
    rejected = 0
    for pair in pairs:
        q, a = pair.get("question", "").strip(), pair.get("answer", "").strip()
        if not q or not a:
            continue
        if judge:
            verdict = await call_chat(client, model, task.judge_prompt(q, a), temperature=0.0)
            if not verdict.upper().startswith("YES"):
                rejected += 1
                continue
        row = {
            "task_type": task.task_type,
            "drug": task.drug,
            "section": task.section,
            "question": q,
            "answer": a,
            "source": task.source,
            "source_id": task.source_id,
        }
        if task.extras:
            row["meta"] = task.extras
        rows.append(row)
    return rows, rejected


async def execute_all(
    tasks: list[Task],
    args: argparse.Namespace,
    output_path: str,
) -> None:
    client, model = make_client_and_model()
    print(f"using model={model!r} via base_url={client.base_url}")
    print(f"dispatching {len(tasks)} tasks with concurrency={args.concurrency}")

    sem = asyncio.Semaphore(args.concurrency)
    out_lock = asyncio.Lock()
    f = open(output_path, "a", encoding="utf-8")
    written = rejected = 0
    started = time.time()

    async def worker(idx: int, task: Task) -> None:
        nonlocal written, rejected
        if _state["aborted"]:
            return
        async with sem:
            if _state["aborted"]:
                return
            rows, rej = await run_task(
                task, client, model,
                n_pairs=args.questions_per_section,
                judge=not args.no_judge,
            )
        async with out_lock:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            written += len(rows)
            rejected += rej
            if (idx + 1) % 10 == 0 or (idx + 1) == len(tasks):
                elapsed = time.time() - started
                rate = written / elapsed if elapsed > 0 else 0
                print(
                    f"  [{idx + 1}/{len(tasks)}] task={task.task_type} "
                    f"key={task.source_id[:60]!r} | written={written} "
                    f"rejected={rejected} rate={rate:.2f} pair/s "
                    f"elapsed={elapsed:.0f}s"
                )

    await asyncio.gather(*(worker(i, t) for i, t in enumerate(tasks)))
    f.close()
    elapsed = time.time() - started
    print(f"\ndone. wrote {written} pairs, rejected {rejected} in {elapsed:.0f}s. output: {output_path}")


# -------------------------- main -------------------------- #

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task", choices=["drug-labels", "symptoms", "all"], default="all")
    parser.add_argument("--max-drugs", type=int, default=1000)
    parser.add_argument("--sections-per-drug", type=int, default=2)
    parser.add_argument("--questions-per-section", type=int, default=3,
                        help="also used as questions-per-symptom-theme")
    parser.add_argument("--concurrency", type=int, default=20,
                        help="parallel API calls (DeepSeek allows 30+)")
    parser.add_argument("--no-judge", action="store_true",
                        help="skip the validation pass (faster, lower quality)")
    parser.add_argument("--output", default=OUTPUT_PATH)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    seen = already_processed_keys(args.output)
    print(f"resuming: {len(seen)} task keys already in {args.output}")

    # Symptoms FIRST so the highest-value safety-triage data is captured even
    # if drug-label generation gets cut off (budget exhaustion, interrupt).
    tasks: list[Task] = []
    if args.task in ("symptoms", "all"):
        tasks.extend(build_symptom_tasks())
    if args.task in ("drug-labels", "all"):
        tasks.extend(build_drug_label_tasks(args.max_drugs, args.sections_per_drug))

    # Filter out already-processed
    before = len(tasks)
    tasks = [
        t for t in tasks
        if f"{t.task_type}|{t.drug}|{t.section}|{t.source_id}" not in seen
    ]
    print(f"tasks: {before} total, {len(tasks)} remaining after dedup")

    if not tasks:
        print("nothing to do.")
        return 0

    asyncio.run(execute_all(tasks, args, args.output))
    return 0


if __name__ == "__main__":
    sys.exit(main())
