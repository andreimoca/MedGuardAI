"""Groundedness check: verify factual claims in a response are supported by
the retrieved RAG context.

Approach:
  1. Split the response into candidate sentences.
  2. Filter to sentences that are likely factual *medical* claims
     (mention dosage / drug names / contraindication / etc.).
  3. Embed each claim (using the same MiniLM the agent uses — no extra
     model load) and compute its max cosine similarity against the
     embeddings of the retrieved context chunks.
  4. Roll up: average + min similarity, plus a list of weakly-grounded claims.

The output is informational. The output_guard decides what to do with it
(append citations, warn, or rewrite).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np


# Cheap heuristic: a sentence is treated as a "factual medical claim" if it
# matches at least one of these patterns. Pure interjections, politeness, and
# disclaimers are skipped (no point grounding "I'm an AI assistant.").
_CLAIM_PATTERNS = [
    r"\b\d+\s*(mg|mcg|g|ml|iu|kg|hours?|days?|times?|weeks?)\b",  # dosage units
    r"\b(every|once|twice|thrice|three|four|q[1-9])\b.*\b(daily|hour|day)",
    r"\bcontraindicat",
    r"\binteract",
    r"\b(allerg|hypersensitiv|cross-react)",
    r"\b(side effect|adverse|reaction)",
    r"\b(boxed warning|black box)",
    r"\boverdos",
    r"\b(dose|dosage|recommended|max(imum)?)\b",
    r"\b(naproxen|ibuprofen|acetaminophen|paracetamol|aspirin|amoxicillin|"
    r"metformin|warfarin|diphenhydramine|hydrocortisone|loratadine|cetirizine)\b",
]

_CLAIM_REGEX = re.compile("|".join(_CLAIM_PATTERNS), re.IGNORECASE)

# Skip these wholly: meta-statements, disclaimers, emergency template lines,
# bullet/heading prefixes that aren't claims by themselves.
_SKIP_PATTERNS = [
    r"^\s*\[EMERGENCY\]",
    r"\bI (am|'m) an AI\b",
    r"\bdo not have sufficient information\b",
    r"\bplease (consult|seek|see|call)\b",
    r"\bif symptoms persist\b",
    r"^\s*Source[s]?:",
    r"^\s*Note:",
]
_SKIP_REGEX = re.compile("|".join(_SKIP_PATTERNS), re.IGNORECASE)


@dataclass
class GroundednessResult:
    avg_similarity: float
    min_similarity: float
    n_claims: int
    weakly_grounded: list[tuple[str, float]] = field(default_factory=list)
    threshold: float = 0.4

    @property
    def is_well_grounded(self) -> bool:
        # Empty (no claims) → trivially fine.
        if self.n_claims == 0:
            return True
        return self.avg_similarity >= self.threshold

    @property
    def is_severely_ungrounded(self) -> bool:
        if self.n_claims == 0:
            return False
        # Severe = min similarity is REALLY low AND average is below threshold.
        return self.min_similarity < 0.25 and self.avg_similarity < self.threshold


def split_sentences(text: str) -> list[str]:
    # Lightweight sentence splitter — punkt would be overkill here.
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z\[])", text.strip())
    return [p.strip() for p in parts if p.strip()]


def extract_claims(text: str) -> list[str]:
    """Return the subset of sentences that look like factual medical claims."""
    claims: list[str] = []
    for sent in split_sentences(text):
        if _SKIP_REGEX.search(sent):
            continue
        if _CLAIM_REGEX.search(sent):
            claims.append(sent)
    return claims


def split_context(context: str) -> list[str]:
    """Split formatted retrieval context into individual chunks."""
    if not context:
        return []
    chunks = re.split(r"\n*-{3,}\n*", context)
    return [c.strip() for c in chunks if c.strip()]


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def check_groundedness(
    response: str,
    retrieval_context: str,
    embed_fn,
    threshold: float = 0.4,
) -> GroundednessResult:
    """Score how well the response's factual claims are grounded in context.

    `embed_fn(list_of_strings) -> list_of_vectors` — use the agent's
    HuggingFaceEmbeddings.embed_documents (or .embed_query for one-by-one).
    """
    claims = extract_claims(response)
    if not claims:
        return GroundednessResult(
            avg_similarity=1.0, min_similarity=1.0, n_claims=0,
            threshold=threshold,
        )

    chunks = split_context(retrieval_context)
    if not chunks:
        # No context to ground against — every claim is unsupported.
        return GroundednessResult(
            avg_similarity=0.0, min_similarity=0.0, n_claims=len(claims),
            weakly_grounded=[(c, 0.0) for c in claims],
            threshold=threshold,
        )

    claim_embs = embed_fn(claims)
    chunk_embs = embed_fn(chunks)

    similarities = []
    weak: list[tuple[str, float]] = []
    for claim, c_emb in zip(claims, claim_embs):
        max_sim = max(cosine(np.asarray(c_emb), np.asarray(k_emb)) for k_emb in chunk_embs)
        similarities.append(max_sim)
        if max_sim < threshold:
            weak.append((claim, round(max_sim, 3)))

    return GroundednessResult(
        avg_similarity=round(float(np.mean(similarities)), 3),
        min_similarity=round(float(np.min(similarities)), 3),
        n_claims=len(claims),
        weakly_grounded=weak,
        threshold=threshold,
    )


def extract_source_drug_names(retrieval_context: str) -> list[str]:
    """Pull drug names out of the formatted context for citation injection.

    The format from agent/tools/retrieval.format_docs looks like:
        [Source 1: Naproxen]
        ... content ...
    """
    matches = re.findall(r"\[Source\s+\d+:\s*([^\]]+)\]", retrieval_context or "")
    seen: list[str] = []
    for m in matches:
        name = m.strip()
        if name and name not in seen and name.lower() != "unknown":
            seen.append(name)
    return seen
