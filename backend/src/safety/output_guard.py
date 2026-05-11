"""Output-side safety guard.

Three jobs, applied to the agent's final assistant message before it's sent
back to the user:

  1. **Toxicity check** — make sure the model didn't generate anything
     abusive or hateful. Detoxify if available, regex fallback otherwise.

  2. **Groundedness check** — for factual medical claims (dosages,
     contraindications, interactions), verify they have semantic support in
     the retrieved RAG context. Severely ungrounded answers get rewritten;
     mildly ungrounded ones get a "low confidence" footer; well-grounded
     answers pass through.

  3. **Citation injection** — append a `Sources:` line listing the FDA
     drug labels the answer was grounded in. This is the user-visible
     equivalent of "show your work".
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Any, Callable

from safety.groundedness import (
    check_groundedness,
    extract_source_drug_names,
)


_TOXIC_RE = re.compile(
    r"\b(fuck|shit|asshole|bitch|cunt|kill\s+yourself)\b",
    re.IGNORECASE,
)


# Hallucinated closing tags. The system prompt says no closing tag exists,
# but small models still emit them — strip them silently.
_CLOSING_TAG_RE = re.compile(
    r"\s*\[\s*(/\s*EMERGENCY|END[\s_-]*EMERGENCY|/\s*END[\s_-]*EMERGENCY)\s*\]\s*",
    re.IGNORECASE,
)

# Apostrophe variants small models emit: ASCII ', right single quote ’,
# modifier-letter apostrophe ʼ. Used so "here’s" matches as well as "here's".
_APOS = r"['’ʼ]"

# Meta-preambles the model emits before the actual answer. These leak the
# prompt machinery to the user — strip them. Anchored to the start of the
# response (after optional whitespace).
_META_PREAMBLE_RE = re.compile(
    r"^\s*(?:okay[,!.…]?\s+)?"
    r"(?:here(?:" + _APOS + r"s| is)\s+(?:a\s+)?(?:helpful\s+)?(?:response|answer)\s+)?"
    r"(?:based\s+on|according\s+to)\s+"
    r"(?:the\s+)?(?:retrieved\s+(?:FDA\s+)?(?:information|context|chunks|documentation|data|labels?)|"
    r"(?:retrieved\s+)?FDA\s+(?:labels?|documentation|data)|documentation|context)\s*[,:.\-]?\s*",
    re.IGNORECASE,
)
# Also catch the bare "Okay, here's a helpful response:" family with no
# "based on" tail — and its many phrasings ("Sure, here is my answer -",
# "Alright, here's the response:", curly-apostrophe variants, etc.).
_META_PREAMBLE_BARE_RE = re.compile(
    r"^\s*"
    r"(?:(?:okay|ok|sure|alright|all\s+right|right|got\s+it|understood|"
    r"of\s+course|no\s+problem)[,!.…\s]+)?"
    r"here(?:" + _APOS + r"s|\s+is)\s+(?:a|the|my|your)\s+"
    r"(?:helpful\s+|quick\s+|brief\s+|short\s+|simple\s+)*"
    r"(?:response|answer|reply)\b\s*[,:.—\-]?\s*",
    re.IGNORECASE,
)
# Robotic paraphrase-back openers ("My understanding is that you're..."):
# strip the lead-in, the symptom acknowledgment that follows is fine.
_META_PARAPHRASE_RE = re.compile(
    r"^\s*(?:my\s+understanding\s+is\s+that\s+|"
    r"as\s+(?:i|we)\s+understand\s+it[,]?\s+|"
    r"to\s+summarize\s+your\s+(?:question|query)[,:]?\s+)",
    re.IGNORECASE,
)

# Self-referential sentences naming the retrieval mechanism. Drop the whole
# sentence.
_META_SENTENCE_RE = re.compile(
    r"(?<=[.!?])\s*[^.!?]*\b("
    r"FDA\s+labels?\s+(?:highlight|show|indicate|state|note|say|mention|"
    r"emphasize|stress|suggest|warn)|"
    r"retrieved\s+(?:context|information|chunks|documentation)|"
    r"according\s+to\s+the\s+(?:retrieved|documentation|FDA\s+labels?)"
    r")\b[^.!?]*[.!?]",
    re.IGNORECASE,
)
# Same pattern, but anchored to the start of the response (no preceding
# sentence terminator).
_META_SENTENCE_START_RE = re.compile(
    r"^\s*[^.!?]*\b("
    r"FDA\s+labels?\s+(?:highlight|show|indicate|state|note|say|mention|"
    r"emphasize|stress|suggest|warn)|"
    r"retrieved\s+(?:context|information|chunks|documentation)|"
    r"according\s+to\s+the\s+(?:retrieved|documentation|FDA\s+labels?)"
    r")\b[^.!?]*[.!?]\s*",
    re.IGNORECASE,
)


def _sanitize_response(text: str) -> tuple[str, dict[str, bool]]:
    """Strip prompt-leak artifacts the model emits despite the system prompt:
    hallucinated closing tags, meta-preambles ("Based on the retrieved..."),
    and self-referential sentences ("The FDA labels highlight..."). Returns
    the cleaned text plus per-pattern flags for logging."""
    flags = {
        "stripped_closing_tag": False,
        "stripped_preamble": False,
        "stripped_meta_sentence": False,
    }
    cleaned = text

    if _CLOSING_TAG_RE.search(cleaned):
        cleaned = _CLOSING_TAG_RE.sub(" ", cleaned)
        flags["stripped_closing_tag"] = True

    # Models sometimes stack preambles ("Okay, here's a helpful response:
    # Based on the FDA labels, ..."), so peel them in a loop until stable.
    for _ in range(4):
        before = cleaned
        for rx in (_META_PREAMBLE_RE, _META_PREAMBLE_BARE_RE, _META_PARAPHRASE_RE):
            new = rx.sub("", cleaned, count=1)
            if new != cleaned:
                flags["stripped_preamble"] = True
                cleaned = new.lstrip()
        if cleaned == before:
            break

    new = _META_SENTENCE_START_RE.sub("", cleaned, count=1)
    if new != cleaned:
        flags["stripped_meta_sentence"] = True
        cleaned = new
    new = _META_SENTENCE_RE.sub(" ", cleaned)
    if new != cleaned:
        flags["stripped_meta_sentence"] = True
        cleaned = new

    cleaned = re.sub(r"[ \t]+", " ", cleaned)
    cleaned = re.sub(r"\s+\n", "\n", cleaned)
    cleaned = cleaned.strip()
    # Capitalize the new first letter if we chopped a preamble off and left
    # a lowercase remainder.
    if cleaned and cleaned[0].islower() and flags["stripped_preamble"]:
        cleaned = cleaned[0].upper() + cleaned[1:]
    return cleaned, flags


# User-message patterns that legitimately warrant the [EMERGENCY] short-circuit.
# Mirrors the allowlist in agent.graph.SYSTEM_PROMPT. If the model emits
# [EMERGENCY] but the user's own query does not match any of these, we do NOT
# trust the tag — we run groundedness instead, so a model that fabricates an
# emergency from drug-warning context gets caught and rewritten.
_EMERGENCY_QUERY_RE = re.compile(
    r"(crushing\s+chest|chest\s+pain|chest\s+pressure|"
    r"can('?t|not)\s+breathe|trouble\s+breathing|short(ness)?\s+of\s+breath|"
    r"overdos|"
    r"anaphylax|"
    r"(face|throat|tongue|lip)s?\s+(is\s+|are\s+)?swell|"
    r"suicid|kill\s+myself|end\s+my\s+life|"
    r"vomit(ing)?\s+blood|coughing\s+blood|"
    r"won('?t|not)\s+stop\s+bleed|uncontrol(led|lable)\s+bleed|"
    r"seiz(ure|ing)|convuls|"
    r"head\s+injury|hit\s+(my|his|her|their)\s+head|"
    r"lost\s+consciousness|unconscious|passed\s+out|"
    r"facial\s+droop|face\s+drooping|one[-\s]sided\s+weakness|"
    r"slurred\s+speech|can('?t|not)\s+speak|"
    r"stroke|"
    r"infant.*fever|newborn.*fever|baby.*fever)",
    re.IGNORECASE,
)


REWRITE_UNGROUNDED = (
    "I do not have sufficient information in the official FDA documentation "
    "to answer this safely. Please consult a healthcare professional."
)

REWRITE_TOXIC = (
    "I apologize — that response was inappropriate. "
    "Please consult a healthcare professional for medical guidance."
)


@dataclass
class OutputGuardResult:
    final_response: str
    rewrote: bool = False
    appended_warning: bool = False
    appended_citations: bool = False
    signals: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "rewrote": self.rewrote,
            "appended_warning": self.appended_warning,
            "appended_citations": self.appended_citations,
            "signals": self.signals,
        }


def _is_template_response(response: str, user_query: str = "") -> bool:
    """Skip safety modification for short, obviously-correct templates
    (emergency message, 'I don't know' fallback, refusal). They're already
    safe; modifying them would be destructive.

    The `[EMERGENCY]` short-circuit is gated on the user's own query matching
    an emergency pattern — otherwise a model that fabricates an emergency
    from drug-warning context could self-certify out of groundedness."""
    r = response.strip()
    if "[EMERGENCY]" in r and _EMERGENCY_QUERY_RE.search(user_query or ""):
        return True
    if "do not have sufficient information" in r.lower():
        return True
    if len(r) < 80:
        return True
    return False


def _detoxify_output(text: str) -> dict[str, float] | None:
    if os.environ.get("MEDGUARD_DETOXIFY", "1") == "0":
        return None
    try:
        from detoxify import Detoxify
    except ImportError:
        return None
    try:
        global _DETOX_MODEL  # type: ignore[name-defined]
        try:
            _DETOX_MODEL  # noqa: F821
        except NameError:
            _DETOX_MODEL = Detoxify("original")
        scores = _DETOX_MODEL.predict(text)
        return {k: float(v) for k, v in scores.items()}
    except Exception:
        return None


def check_output(
    response: str,
    retrieval_context: str,
    embed_fn: Callable[[list[str]], list[Any]] | None,
    *,
    groundedness_threshold: float = 0.4,
    inject_citations: bool = False,
    user_query: str = "",
) -> OutputGuardResult:
    """Run all output-side safety checks. Returns the final (possibly
    modified) response plus signals for logging/eval."""
    signals: dict[str, Any] = {}

    if not response or not response.strip():
        return OutputGuardResult(final_response="", signals=signals)

    # --- 0. Sanitize prompt-leak artifacts (closing tags, meta-preambles,
    # "the FDA labels show..." self-references). Runs first so downstream
    # checks see the cleaned text. ---
    response, sanitize_flags = _sanitize_response(response)
    if any(sanitize_flags.values()):
        signals["sanitizer"] = sanitize_flags
    if not response.strip():
        return OutputGuardResult(final_response="", signals=signals)

    # --- 1. Toxicity ---
    if _TOXIC_RE.search(response):
        signals["toxic_match"] = True
        return OutputGuardResult(
            final_response=REWRITE_TOXIC, rewrote=True, signals=signals,
        )
    detox = _detoxify_output(response)
    if detox:
        signals["detoxify"] = detox
        if detox.get("toxicity", 0.0) > 0.7:
            return OutputGuardResult(
                final_response=REWRITE_TOXIC, rewrote=True, signals=signals,
            )

    # --- 2. Skip groundedness/citations for templates ---
    if _is_template_response(response, user_query=user_query):
        signals["skipped_template"] = True
        return OutputGuardResult(final_response=response, signals=signals)

    # --- 3. Groundedness (only if embeddings are available) ---
    final = response
    appended_warning = False
    if embed_fn is not None:
        g = check_groundedness(
            response, retrieval_context, embed_fn, threshold=groundedness_threshold,
        )
        signals["groundedness"] = {
            "n_claims": g.n_claims,
            "avg_similarity": g.avg_similarity,
            "min_similarity": g.min_similarity,
            "n_weak_claims": len(g.weakly_grounded),
        }

        if g.is_severely_ungrounded:
            return OutputGuardResult(
                final_response=REWRITE_UNGROUNDED, rewrote=True, signals=signals,
            )

        if not g.is_well_grounded and g.n_claims > 0:
            final = final.rstrip() + (
                "\n\n*Note: some details here may not be fully supported by "
                "official FDA labeling — please verify with a healthcare "
                "professional.*"
            )
            appended_warning = True
    else:
        signals["groundedness"] = "skipped_embed_unavailable"

    # --- 4. Citation injection ---
    appended_citations = False
    if inject_citations:
        sources = extract_source_drug_names(retrieval_context)
        if sources:
            final = final.rstrip() + (
                f"\n\nSources: {', '.join(sources)} (FDA-approved drug labels)"
            )
            appended_citations = True
            signals["sources_cited"] = sources

    return OutputGuardResult(
        final_response=final,
        appended_warning=appended_warning,
        appended_citations=appended_citations,
        signals=signals,
    )
