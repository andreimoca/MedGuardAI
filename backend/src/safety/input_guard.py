"""Input-side safety guard.

Catches three classes of bad-input before the agent runs:
  1. **Prompt injection** — user trying to override system prompt or jailbreak.
  2. **Toxicity** — abusive / hateful queries.
  3. **Disallowed content** — requests the system refuses to handle (e.g.
     instructions for self-harm, illegal drug synthesis).

The check returns a `GuardResult` with a `blocked` flag and a refusal message.
We use cheap pattern matching plus optional Detoxify if available — no extra
dependency required, and Detoxify can be enabled later by installing it.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Any

# ---- Prompt-injection detection ---- #
_PROMPT_INJECTION_PATTERNS = [
    r"\bignore\s+(all\s+)?(previous|prior|above|the\s+above)\s+(instructions|prompts?|rules?)\b",
    r"\bdisregard\s+(all\s+)?(previous|prior|above)\s+(instructions|prompts?|rules?)\b",
    r"\bforget\s+(everything|all)?\s*(you|that|the).*(instruction|rule|told|prompt)\b",
    r"\byou\s+are\s+now\s+(an?\s+)?(unrestricted|jailbroken|free|new|different)\b",
    r"\bact\s+as\s+(if\s+)?(you\s+are|though\s+you\s+were)\s+(an?\s+)?(unrestricted|jailbroken|hacker|criminal)\b",
    r"\bsystem\s+prompt\s+is\b",
    r"\bnew\s+(system|persona)\b.*\binstructions\b",
    r"\bDAN\s+mode\b",  # well-known jailbreak handle
    r"\bdeveloper\s+mode\b.*\benabled\b",
]
_PROMPT_INJECTION_REGEX = re.compile("|".join(_PROMPT_INJECTION_PATTERNS), re.IGNORECASE)

# ---- Disallowed-content detection ---- #
# Note: this is intentionally narrow. We're a medical assistant — we should not
# refuse legitimate medical questions about overdose / lethal doses (the user
# may be a clinician or worried family member). The agent's emergency
# classifier handles those. We refuse only the clearly out-of-scope stuff
# (drug synthesis, weapons, etc.) and explicit recreational-use queries.
_DISALLOWED_PATTERNS = [
    r"\b(synthesi[sz]e|cook|manufacture|produce)\s+(meth|methamphetamine|cocaine|heroin|fentanyl|lsd|mdma)\b",
    r"\bhow\s+to\s+(make|build)\s+(a\s+)?(bomb|explosive|weapon|firearm)\b",
    r"\brecreational\s+(dose|use)\s+of\b",
    r"\bget\s+(high|stoned|wasted|fucked\s+up)\s+(on|from)\b",
]
_DISALLOWED_REGEX = re.compile("|".join(_DISALLOWED_PATTERNS), re.IGNORECASE)

# ---- Toxicity (lightweight regex fallback) ---- #
# This is intentionally minimal and not the primary defense. Detoxify (when
# installed) is much better; this just catches obvious slurs so the agent
# doesn't politely engage with abusive queries.
_TOXIC_HINT = re.compile(
    r"\b(fuck\s+you|fuck\s+off|piece\s+of\s+shit|kill\s+yourself|kys)\b",
    re.IGNORECASE,
)


@dataclass
class InputGuardResult:
    blocked: bool = False
    reason: str = ""
    refusal: str = ""
    signals: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "blocked": self.blocked,
            "reason": self.reason,
            "signals": self.signals,
        }


REFUSAL_PROMPT_INJECTION = (
    "I'm MedGuardAI — I stay focused on safe, evidence-based medical guidance "
    "and can't follow instructions that ask me to drop that role. "
    "If you have a medical question, I'm happy to help."
)

REFUSAL_DISALLOWED = (
    "I can't help with that request. I'm a medical assistant focused on safe, "
    "FDA-grounded information about medications and symptoms."
)

REFUSAL_TOXIC = (
    "Let's keep this respectful. I'm here to help with medical questions — "
    "tell me what's going on and I'll do my best to help."
)


def _maybe_detoxify_score(text: str) -> dict[str, float] | None:
    """Run Detoxify on the input if installed. Returns None if not available
    or if the model fails to load (e.g. no GPU memory)."""
    if os.environ.get("MEDGUARD_DETOXIFY", "1") == "0":
        return None
    try:
        from detoxify import Detoxify  # heavy import, only when used
    except ImportError:
        return None
    try:
        # Lazy global so we don't reload the model every call.
        global _DETOXIFY_MODEL  # type: ignore[name-defined]
        try:
            _DETOXIFY_MODEL  # noqa: F821
        except NameError:
            _DETOXIFY_MODEL = Detoxify("original")
        scores = _DETOXIFY_MODEL.predict(text)
        return {k: float(v) for k, v in scores.items()}
    except Exception:
        return None


def check_input(query: str) -> InputGuardResult:
    """Run all input-side safety checks. Returns a single GuardResult."""
    if not query or not query.strip():
        return InputGuardResult()

    if _PROMPT_INJECTION_REGEX.search(query):
        return InputGuardResult(
            blocked=True,
            reason="prompt_injection",
            refusal=REFUSAL_PROMPT_INJECTION,
            signals={"matched_pattern": "prompt_injection"},
        )

    if _DISALLOWED_REGEX.search(query):
        return InputGuardResult(
            blocked=True,
            reason="disallowed_content",
            refusal=REFUSAL_DISALLOWED,
            signals={"matched_pattern": "disallowed"},
        )

    if _TOXIC_HINT.search(query):
        return InputGuardResult(
            blocked=True,
            reason="toxic_input",
            refusal=REFUSAL_TOXIC,
            signals={"matched_pattern": "toxic_hint"},
        )

    detox = _maybe_detoxify_score(query)
    if detox:
        # Detoxify thresholds: 0.5 is the conventional "definitely toxic" cut.
        toxicity = detox.get("toxicity", 0.0)
        threat = detox.get("threat", 0.0)
        if toxicity > 0.7 or threat > 0.5:
            return InputGuardResult(
                blocked=True,
                reason="detoxify_toxic",
                refusal=REFUSAL_TOXIC,
                signals={"detoxify": detox},
            )
        return InputGuardResult(blocked=False, signals={"detoxify": detox})

    return InputGuardResult(blocked=False, signals={"detoxify": "unavailable"})
