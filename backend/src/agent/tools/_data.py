"""Shared JSON drug-label loader used by the structured tools.

The Chroma vector store is great for fuzzy semantic retrieval but the
interactions/dosage/allergy tools need section-precise lookups, so we load
the raw FDA JSON files directly and cache them in memory.
"""
import json
import os
from functools import lru_cache
from typing import Any

RAW_DATA_DIR = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "data", "raw", "dailymed")
)


@lru_cache(maxsize=1)
def load_all_labels() -> list[dict[str, Any]]:
    """Read every JSON drug label once and cache the parsed list."""
    labels: list[dict[str, Any]] = []
    if not os.path.isdir(RAW_DATA_DIR):
        return labels
    for filename in os.listdir(RAW_DATA_DIR):
        if not filename.endswith(".json"):
            continue
        try:
            with open(os.path.join(RAW_DATA_DIR, filename), encoding="utf-8") as f:
                labels.append(json.load(f))
        except (OSError, json.JSONDecodeError):
            continue
    return labels


_RICH_SECTIONS = (
    "indications_and_usage",
    "contraindications",
    "warnings_and_cautions",
    "drug_interactions",
    "dosage_and_administration",
)


def find_label(drug_name: str, prefer_section: str | None = None) -> dict[str, Any] | None:
    """Case-insensitive match against drug_name, generic_name, or active_ingredients.

    With ~10k labels in the corpus, naive "first match wins" returns the first
    OTC product alphabetically (e.g. for 'naproxen' that's an Equaline OTC
    pack with no drug_interactions). We score candidates instead:

      1. Exact match on a name field beats partial.
      2. Labels that have the section the caller wants beat ones that don't.
      3. Among ties, prefer labels with more populated sections (richer data).
    """
    if not drug_name:
        return None
    needle = drug_name.lower().strip()

    scored: list[tuple[int, int, int, dict[str, Any]]] = []
    for label in load_all_labels():
        names: list[str] = []
        if label.get("drug_name"):
            names.append(str(label["drug_name"]).lower())
        names.extend(str(g).lower() for g in label.get("generic_name", []) or [])
        names.extend(str(a).lower() for a in label.get("active_ingredients", []) or [])
        names = [n for n in names if n]

        match_score = 0
        if needle in names:
            match_score = 3
        elif any(needle == n.split(",")[0].split()[0] for n in names if n.split()):
            match_score = 2
        elif any(needle in n for n in names):
            match_score = 1
        if match_score == 0:
            continue

        section_bonus = 1 if prefer_section and label.get(prefer_section) else 0
        completeness = sum(1 for k in _RICH_SECTIONS if label.get(k))
        scored.append((match_score, section_bonus, completeness, label))

    if not scored:
        return None
    scored.sort(key=lambda x: (x[0], x[1], x[2]), reverse=True)
    return scored[0][3]


def join_section(label: dict[str, Any], key: str) -> str:
    parts = label.get(key) or []
    return "\n".join(str(p) for p in parts).strip()
