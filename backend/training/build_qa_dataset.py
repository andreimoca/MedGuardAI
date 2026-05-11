"""Build the SFT dataset for MedGuardAI from MedQuAD + curated safety-triage themes.

The supervised fine-tuning data has two parts, both written to the same JSONL:

  1. **medquad** — MedQuAD (the Medical Question Answering Dataset, a published
     collection of consumer-health Q&A curated from U.S. National Institutes of
     Health websites). We pull it from Kaggle with `kagglehub`
     (`pythonafroz/medquad-medical-question-answer-for-ai-research`) and, for
     each Q&A pair, ask the builder LLM to *rewrite the existing answer* into
     MedGuardAI's safety-first voice (concise, "consult a healthcare provider",
     `[EMERGENCY]` only for true emergencies, no invented dosing or drug names).
     A judge pass checks the rewrite is faithful to the original MedQuAD answer
     — the LLM reformats, it does not introduce new clinical facts.

  2. **symptoms** — for each entry in a hand-curated symptom-triage list
     (`SYMPTOM_THEMES`), we render a safety-aware answer from a *deterministic
     template* keyed on the triage tier (low / medium / high_emergency). No LLM
     is involved for this slice.

Only the `medquad` part touches the network. `--task symptoms` runs fully offline.

Backend selection for the rewriter/judge (env vars, in priority order):
    DATASET_LLM_URL / DATASET_LLM_KEY / DATASET_LLM_MODEL  (e.g. DeepSeek)
    LOCAL_LLM_URL / "not-needed" / LOCAL_LLM_MODEL         (LM Studio fallback)
`kagglehub` downloads the public dataset; if Kaggle asks for credentials, set
KAGGLE_USERNAME / KAGGLE_KEY (or drop ~/.kaggle/kaggle.json in place).

Usage examples:
    # Smoke test: ~30 MedQuAD rows + all the deterministic symptom themes
    python build_qa_dataset.py --task all --max-rows 30 --concurrency 10

    # Full run for SFT training (resumable; ~a few $ on DeepSeek)
    python build_qa_dataset.py --task all --max-rows 4000 --concurrency 20

    # Just the deterministic symptom-triage rows (offline, no API key needed)
    python build_qa_dataset.py --task symptoms

    # Skip the faithfulness judge on the MedQuAD rewrites (faster, lower quality)
    python build_qa_dataset.py --task medquad --max-rows 4000 --no-judge

Output: training/data/sft_pairs.jsonl. Resumable — re-run after a crash and it
skips `(task_type, drug, section, source_id)` tuples already in the file.
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import glob
import hashlib
import json
import os
import random
import re
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Iterator

from dotenv import load_dotenv
from openai import AsyncOpenAI

load_dotenv(os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", ".env")))

MEDQUAD_DATASET_ID = "pythonafroz/medquad-medical-question-answer-for-ai-research"
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

GEN_PROMPT_MEDQUAD = """You are preparing a training item for MedGuardAI, a
safety-first medical assistant, from an entry in the MedQuAD dataset (consumer
health Q&A curated from U.S. National Institutes of Health websites).

Rewrite the ANSWER below into MedGuardAI's voice, following these rules:
- Use ONLY the information in the provided answer. Do NOT add facts, dosages,
  drug names, contraindications, or interactions that are not in it. If the
  answer is thin, keep your rewrite short rather than padding it.
- Be concise and well-structured (a few short paragraphs or a short list, not an
  essay). Plain language a patient can follow.
- If the topic concerns a true medical emergency (e.g. stroke signs, heart
  attack, anaphylaxis, suspected overdose, suicidal intent), begin by briefly
  acknowledging it, say you are not equipped to handle emergencies, then output
  exactly: "[EMERGENCY] Please call your local emergency number (911 in the US)
  or get to an emergency room immediately." and do NOT give other medical advice.
- Otherwise, end with a brief line encouraging the reader to consult a
  healthcare professional for personal medical advice or diagnosis.
- Never recommend a prescription-only medication by name; defer to a clinician.

Output ONLY the rewritten answer text — no preamble, no labels, no markdown
headers, no surrounding quotes.

QUESTION:
{question}

ANSWER (MedQuAD; source: {source}; focus: {focus_area}):
{answer}
"""

JUDGE_PROMPT_MEDQUAD = """You are checking a rewritten medical answer for
faithfulness to its source.

Question: {question}

Rewritten answer:
{rewritten}

Original source answer (MedQuAD):
{answer}

Does the rewritten answer stay faithful to the original — i.e. it does NOT add
any clinical fact, dosage, drug name, contraindication, or interaction that is
not present in (or directly implied by) the original? Generic safety framing
("see a healthcare provider", an [EMERGENCY] directive for a true emergency
topic) is allowed and does not count as an added fact.

Reply with EXACTLY one word: YES or NO.
"""


# -------------------------- curated symptoms -------------------------- #

# Each entry: theme label + list of relevant OTC drug names (used only to name a
# generic OTC option in the deterministic answer template), triage tier, notes.
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


# -------------------------- shared text helpers -------------------------- #
# (also imported by build_dpo_dataset.py so the seeded preference pairs use the
# exact same "good answer" templates as the SFT symptom rows)

EMERGENCY_TAG = (
    "[EMERGENCY] Please call your local emergency number (911 in the US) or get "
    "to an emergency room immediately."
)


def chunk_long_text(text: str, max_chars: int = 2400) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + " ... [truncated]"


def theme_slug(label: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", label.lower()).strip("_")


def theme_otc(theme: dict[str, Any]) -> str | None:
    """First OTC option named for a theme, or None for emergency / no-drug themes."""
    drugs = theme.get("drugs") or []
    return drugs[0] if drugs else None


def theme_queries(label: str, limit: int = 3) -> list[str]:
    """Natural-ish user phrasings derived from a theme label (deterministic)."""
    core = re.sub(r"\s*\(.*?\)\s*", " ", label).strip()
    core = re.sub(r"\s+", " ", core) or label
    paren = None
    m = re.search(r"\(([^)]+)\)", label)
    if m:
        paren = re.sub(r"\s+", " ", m.group(1)).strip()
    qs = [
        f"I have {core}.",
        f"{core[:1].upper()}{core[1:]} — what should I do?",
        f"What can I do about {core}?",
    ]
    if paren and len(paren) < 90 and not paren.lower().startswith("fast"):
        qs.insert(1, f"I'm experiencing {paren}")
    seen: set[str] = set()
    uniq: list[str] = []
    for q in qs:
        k = q.lower()
        if k not in seen:
            seen.add(k)
            uniq.append(q)
    return uniq[: max(1, limit)]


def chosen_answer_for_theme(theme: dict[str, Any]) -> str:
    """Deterministic safety-aware answer for a symptom theme, keyed on triage tier.

    Used as the SFT target for symptom rows and as the 'chosen' side of the
    seeded DPO preference pairs (so both stay perfectly consistent)."""
    label = str(theme.get("label", ""))
    triage = str(theme.get("triage", "low"))
    core = re.sub(r"\s*\(.*?\)\s*", " ", label).strip() or label
    otc = theme_otc(theme)

    if triage == "high_emergency":
        return (
            f"I hear that you're dealing with {core.lower()}, and I'm sorry — that sounds frightening. "
            f"I'm an AI assistant and I am not the right tool for an acute medical emergency. "
            f"{EMERGENCY_TAG} These situations can be time-critical, so please don't wait or rely on a chatbot for this."
        )
    if triage == "medium":
        relief = (
            f"For symptom relief in the meantime, a standard over-the-counter option like {otc} can help — follow the dose on the package. "
            if otc else
            "In the meantime, basic self-care (rest, fluids, heat or ice as appropriate) can help. "
        )
        return (
            "This could be a few different things, and the safe move is to get it checked. "
            f"{relief}"
            "I'd recommend that you see a doctor within the next day or so for a proper evaluation. "
            "Go to the ER right away if you develop trouble breathing, confusion, fainting, sudden severe pain, or it suddenly gets much worse."
        )
    # low tier
    relief = (
        f"In the meantime, rest, fluids, and a standard over-the-counter option like {otc} (follow the package dose) usually help. "
        if otc else
        "In the meantime, rest, fluids, and basic self-care (heat or ice as appropriate) usually help. "
    )
    return (
        "A couple of quick questions to point you in the right direction: how long has this been going on, "
        f"how bad is it, and is anything else happening alongside it? {relief}"
        "If it persists beyond a few days or gets worse, please see a healthcare provider."
    )


# -------------------------- MedQuAD source -------------------------- #

def _resolve_medquad_csv() -> str:
    """Download the MedQuAD Kaggle dataset (cached by kagglehub) and return its CSV path."""
    try:
        import kagglehub
    except ImportError as exc:  # pragma: no cover - dependency hint
        raise SystemExit(
            "kagglehub is required for the MedQuAD source. Install it: pip install kagglehub"
        ) from exc
    path = kagglehub.dataset_download(MEDQUAD_DATASET_ID)
    csvs = sorted(
        glob.glob(os.path.join(path, "**", "*.csv"), recursive=True),
        key=os.path.getsize, reverse=True,
    )
    if not csvs:
        raise SystemExit(f"no CSV found under the downloaded MedQuAD dataset at {path}")
    return csvs[0]


def iter_medquad_rows(csv_path: str | None = None) -> Iterator[dict[str, str]]:
    """Yield normalised {'question','answer','source','focus_area'} from the MedQuAD CSV."""
    csv_path = csv_path or _resolve_medquad_csv()
    print(f"MedQuAD CSV: {csv_path}")
    with open(csv_path, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for raw in reader:
            row = {(k or "").strip().lower(): (v or "").strip() for k, v in raw.items()}
            q = row.get("question", "")
            a = row.get("answer", "")
            if not q or not a:
                continue
            yield {
                "question": q,
                "answer": a,
                "source": row.get("source", "") or "MedQuAD",
                "focus_area": row.get("focus_area", "") or row.get("focus", ""),
            }


# -------------------------- I/O helpers -------------------------- #

def row_key(row: dict[str, Any]) -> str:
    return f"{row.get('task_type')}|{row.get('drug')}|{row.get('section')}|{row.get('source_id')}"


def already_processed_keys(output_path: str) -> set[str]:
    if not os.path.exists(output_path):
        return set()
    seen: set[str] = set()
    with open(output_path, encoding="utf-8") as f:
        for line in f:
            try:
                seen.add(row_key(json.loads(line)))
            except json.JSONDecodeError:
                continue
    return seen


# -------------------------- task definitions -------------------------- #

@dataclass
class Task:
    """One MedQuAD row whose answer needs the LLM rewrite + faithfulness judge."""
    question: str
    source_answer: str   # the original MedQuAD answer (ground truth for the judge)
    drug: str            # the MedQuAD focus_area, or "n/a"
    section: str         # MedQuAD category bucket / "informational"
    source_id: str       # "medquad::<sha1(question)[:16]>"
    extras: dict[str, Any] = field(default_factory=dict)
    task_type: str = "medquad_qa"

    def gen_prompt(self) -> str:
        return GEN_PROMPT_MEDQUAD.format(
            question=self.question,
            answer=chunk_long_text(self.source_answer, 5000),
            source=self.extras.get("origin_source") or "MedQuAD",
            focus_area=self.drug,
        )

    def judge_prompt(self, rewritten: str) -> str:
        return JUDGE_PROMPT_MEDQUAD.format(
            question=self.question,
            rewritten=rewritten,
            answer=chunk_long_text(self.source_answer, 5000),
        )


# -------------------------- task / row builders -------------------------- #

def _medquad_source_id(question: str) -> str:
    return "medquad::" + hashlib.sha1(question.strip().encode("utf-8")).hexdigest()[:16]


def build_medquad_tasks(max_rows: int, seed: int, csv_path: str | None = None) -> list[Task]:
    seen_pairs: set[tuple[str, str]] = set()
    uniq: list[dict[str, str]] = []
    for r in iter_medquad_rows(csv_path):
        k = (r["question"], r["answer"])
        if k in seen_pairs:
            continue
        seen_pairs.add(k)
        uniq.append(r)
    random.Random(seed).shuffle(uniq)
    if max_rows and max_rows > 0:
        uniq = uniq[:max_rows]
    tasks: list[Task] = []
    for r in uniq:
        tasks.append(Task(
            question=r["question"],
            source_answer=r["answer"],
            drug=r["focus_area"] or "n/a",
            section="informational",
            source_id=_medquad_source_id(r["question"]),
            extras={
                "source_dataset": f"MedQuAD (Kaggle: {MEDQUAD_DATASET_ID})",
                "origin_source": r["source"],
                "focus_area": r["focus_area"],
            },
        ))
    return tasks


def build_symptom_rows(questions_per_theme: int = 3) -> list[dict[str, Any]]:
    """Deterministic SFT rows for the curated safety-triage themes — no LLM, offline."""
    rows: list[dict[str, Any]] = []
    for theme in SYMPTOM_THEMES:
        label = str(theme.get("label", ""))
        triage = str(theme.get("triage", "low"))
        answer = chosen_answer_for_theme(theme)
        slug = theme_slug(label)
        drugs = theme.get("drugs") or []
        for i, q in enumerate(theme_queries(label, limit=questions_per_theme)):
            rows.append({
                "task_type": "symptom",
                "drug": ", ".join(drugs) or "n/a",
                "section": triage,
                "question": q,
                "answer": answer,
                "source": "(deterministic safety-triage template)",
                "source_id": f"symptom::{slug}" if i == 0 else f"symptom::{slug}::{i}",
                "meta": {
                    "source_dataset": "MedGuardAI curated safety-triage themes (deterministic)",
                    "label": label,
                    "triage": triage,
                    "notes": str(theme.get("notes", "")),
                },
            })
    return rows


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


def _clean_rewrite(raw: str) -> str:
    text = raw.strip()
    text = re.sub(r"^```[a-zA-Z]*\s*", "", text)
    text = re.sub(r"\s*```$", "", text).strip()
    # strip a single layer of surrounding quotes if the model added them
    if len(text) >= 2 and text[0] == text[-1] == '"':
        text = text[1:-1].strip()
    return text


async def run_task(task: Task, client: AsyncOpenAI, model: str, judge: bool) -> dict[str, Any] | None:
    """Rewrite one MedQuAD answer; return the output row or None if dropped."""
    raw = await call_chat(client, model, task.gen_prompt(), temperature=0.3)
    if not raw or raw.startswith("__ERROR__") or raw == "__ABORTED__":
        return None
    rewritten = _clean_rewrite(raw)
    if len(rewritten) < 15:
        return None
    if judge:
        verdict = await call_chat(client, model, task.judge_prompt(rewritten), temperature=0.0)
        if verdict.startswith("__") or not verdict.strip().upper().startswith("YES"):
            return None
    return {
        "task_type": task.task_type,
        "drug": task.drug,
        "section": task.section,
        "question": task.question,
        "answer": rewritten,
        "source": task.source_answer,
        "source_id": task.source_id,
        "meta": dict(task.extras),
    }


async def execute_all(tasks: list[Task], args: argparse.Namespace, output_path: str) -> None:
    client, model = make_client_and_model()
    print(f"rewriter/judge model={model!r} via base_url={client.base_url}")
    print(f"dispatching {len(tasks)} MedQuAD rows with concurrency={args.concurrency}")

    sem = asyncio.Semaphore(args.concurrency)
    out_lock = asyncio.Lock()
    f = open(output_path, "a", encoding="utf-8")
    written = dropped = 0
    started = time.time()

    async def worker(idx: int, task: Task) -> None:
        nonlocal written, dropped
        if _state["aborted"]:
            return
        async with sem:
            if _state["aborted"]:
                return
            row = await run_task(task, client, model, judge=not args.no_judge)
        async with out_lock:
            if row is not None:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                f.flush()
                written += 1
            else:
                dropped += 1
            done = idx + 1
            if done % 25 == 0 or done == len(tasks):
                elapsed = time.time() - started
                rate = written / elapsed if elapsed > 0 else 0
                print(f"  [{done}/{len(tasks)}] written={written} dropped={dropped} "
                      f"rate={rate:.2f} row/s elapsed={elapsed:.0f}s")

    await asyncio.gather(*(worker(i, t) for i, t in enumerate(tasks)))
    f.close()
    elapsed = time.time() - started
    if _state["aborted"]:
        print(f"\n!! aborted: {_state['abort_reason']}\n!! output is intact and resumable.")
    print(f"\ndone. wrote {written} MedQuAD rows, dropped {dropped} in {elapsed:.0f}s. output: {output_path}")


# -------------------------- main -------------------------- #

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task", choices=["medquad", "symptoms", "all"], default="all")
    parser.add_argument("--max-rows", type=int, default=4000,
                        help="cap on MedQuAD rows to process (0 = all ~16k)")
    parser.add_argument("--max-drugs", type=int, default=None, help=argparse.SUPPRESS)  # deprecated alias
    parser.add_argument("--questions-per-section", type=int, default=3,
                        help="phrasings generated per curated symptom theme")
    parser.add_argument("--concurrency", type=int, default=20,
                        help="parallel rewrite/judge calls")
    parser.add_argument("--no-judge", action="store_true",
                        help="skip the faithfulness judge on MedQuAD rewrites (faster, lower quality)")
    parser.add_argument("--output", default=OUTPUT_PATH)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.max_drugs is not None:
        args.max_rows = args.max_drugs

    random.seed(args.seed)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    seen = already_processed_keys(args.output)
    print(f"resuming: {len(seen)} rows already in {args.output}")

    # 1. Deterministic symptom rows first — offline, highest-value safety data, so
    #    they land even if the MedQuAD rewrite pass gets cut off.
    if args.task in ("symptoms", "all"):
        sym_rows = build_symptom_rows(args.questions_per_section)
        new_rows = [r for r in sym_rows if row_key(r) not in seen]
        if new_rows:
            with open(args.output, "a", encoding="utf-8") as f:
                for r in new_rows:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
            print(f"symptom themes: wrote {len(new_rows)} deterministic rows "
                  f"({len(sym_rows) - len(new_rows)} already present)")
        else:
            print(f"symptom themes: all {len(sym_rows)} rows already present")
        for r in sym_rows:
            seen.add(row_key(r))

    # 2. MedQuAD rows — LLM rewrite into our voice + faithfulness judge.
    if args.task in ("medquad", "all"):
        tasks = build_medquad_tasks(args.max_rows, args.seed)
        before = len(tasks)
        tasks = [t for t in tasks
                 if f"{t.task_type}|{t.drug}|{t.section}|{t.source_id}" not in seen]
        print(f"MedQuAD: {before} candidate rows, {len(tasks)} remaining after dedup")
        if tasks:
            asyncio.run(execute_all(tasks, args, args.output))
        else:
            print("MedQuAD: nothing to do.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
