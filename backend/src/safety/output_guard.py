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


def _is_template_response(response: str) -> bool:
    """Skip safety modification for short, obviously-correct templates
    (emergency message, 'I don't know' fallback, refusal). They're already
    safe; modifying them would be destructive."""
    r = response.strip()
    if "[EMERGENCY]" in r:
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
) -> OutputGuardResult:
    """Run all output-side safety checks. Returns the final (possibly
    modified) response plus signals for logging/eval."""
    signals: dict[str, Any] = {}

    if not response or not response.strip():
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
    if _is_template_response(response):
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
                "\n\n*Note: some claims in this response have low confidence "
                "against the FDA documentation retrieved for your query. "
                "Please verify with a healthcare professional.*"
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
