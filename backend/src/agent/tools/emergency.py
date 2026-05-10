import re

from langchain_core.tools import tool

# Tiered patterns: HIGH = unambiguous medical emergency, MEDIUM = warning sign
# that warrants the emergency response too. Patterns are word-boundary-aware so
# "chest pain" matches but "no chest pain" still matches (we are conservative).
_HIGH_PATTERNS = [
    r"\banaphylax(is|ist|y)\b",
    r"\boverdos(e|ed|ing)\b",
    r"\bheart attack\b",
    r"\bmyocardial infarction\b",
    r"\bstroke\b",
    r"\bsuicid(e|al)\b",
    r"\bchest pain\b",
    r"\bcrushing chest\b",
    r"\bcannot breathe\b",
    r"\bcan'?t breathe\b",
    r"\bdifficulty breathing\b",
    r"\bunconscious\b",
    r"\bseizure\b",
    r"\bconvulsion\b",
    r"\bsevere bleeding\b",
    r"\bcoughing blood\b",
    r"\bcoughing up blood\b",
]

_MEDIUM_PATTERNS = [
    r"\bsevere allergic\b",
    r"\bswelling.*(face|throat|tongue|lips)\b",
    r"\b(throat|tongue|lips).*swelling\b",
    r"\bpassing out\b",
    r"\bblack stool\b",
    r"\bvomiting blood\b",
]


@tool
def emergency_classifier(query: str) -> str:
    """Classify whether a user query describes a medical emergency.

    The agent should call this whenever the query mentions any acute symptom.
    On HIGH severity, the agent must abort normal RAG and respond with the
    emergency template.

    Args:
        query: the user's raw query text.

    Returns:
        One of "HIGH: <pattern>", "MEDIUM: <pattern>", or "NONE".
    """
    text = query.lower()
    for pattern in _HIGH_PATTERNS:
        if re.search(pattern, text):
            return f"HIGH: matched '{pattern}'. Respond with the [EMERGENCY] template."
    for pattern in _MEDIUM_PATTERNS:
        if re.search(pattern, text):
            return f"MEDIUM: matched '{pattern}'. Treat as emergency unless context contradicts."
    return "NONE: no emergency markers detected."


# Lightweight non-tool helper kept for backwards compatibility with the
# existing test suite and the FastAPI endpoint pre-check.
def is_emergency(query: str) -> bool:
    text = query.lower()
    return any(re.search(p, text) for p in _HIGH_PATTERNS + _MEDIUM_PATTERNS)
