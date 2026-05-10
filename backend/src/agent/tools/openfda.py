import requests
from langchain_core.tools import tool

API_BASE_URL = "https://api.fda.gov/drug/label.json"
TIMEOUT_SECONDS = 20


@tool
def fetch_openfda_live(drug_name: str) -> str:
    """Live OpenFDA lookup for drugs not present in the local vector store.

    Use this only after retrieve_drug_info returns no useful match. The result
    is the most relevant section of the live FDA label, condensed to fit the
    LLM context window.

    Args:
        drug_name: drug name to search (brand or generic).

    Returns:
        A condensed FDA label, or a not-found / error message.
    """
    if not drug_name or not drug_name.strip():
        return "fetch_openfda_live: empty drug name."

    params = {
        "search": (
            f'openfda.brand_name:"{drug_name}" '
            f'OR openfda.generic_name:"{drug_name}"'
        ),
        "limit": 1,
    }
    try:
        response = requests.get(API_BASE_URL, params=params, timeout=TIMEOUT_SECONDS)
    except requests.RequestException as exc:
        return f"fetch_openfda_live: network error: {exc}"

    if response.status_code == 404:
        return f"fetch_openfda_live: no FDA label found for '{drug_name}'."
    if not response.ok:
        return f"fetch_openfda_live: API error {response.status_code}."

    results = (response.json() or {}).get("results") or []
    if not results:
        return f"fetch_openfda_live: no FDA label found for '{drug_name}'."

    record = results[0]
    openfda = record.get("openfda", {}) or {}
    name = (openfda.get("brand_name") or openfda.get("generic_name") or [drug_name])[0]

    sections_of_interest = [
        ("Indications", "indications_and_usage"),
        ("Contraindications", "contraindications"),
        ("Boxed warning", "boxed_warning"),
        ("Warnings", "warnings_and_cautions"),
        ("Drug interactions", "drug_interactions"),
        ("Dosage", "dosage_and_administration"),
        ("Overdose", "overdosage"),
    ]

    parts = [f"Live OpenFDA label for {name}:"]
    for label, key in sections_of_interest:
        text = " ".join(record.get(key) or []).strip()
        if not text:
            continue
        # Trim each section so one tool call doesn't blow up the context.
        if len(text) > 1200:
            text = text[:1200] + " ... [truncated]"
        parts.append(f"\n## {label}\n{text}")

    return "\n".join(parts)
