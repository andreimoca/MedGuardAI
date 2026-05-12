from langchain_core.tools import tool

from agent.tools._data import find_label, join_section


@tool
def check_drug_interactions(drug_a: str, drug_b: str) -> str:
    """Check whether two drugs have a documented interaction in the FDA labels.

    Use whenever the user asks about taking two medications together, or
    mixing a medication with something already in their patient profile.

    Args:
        drug_a: name of the first drug (brand or generic).
        drug_b: name of the second drug (brand or generic).

    Returns:
        Excerpts from drug_a's "drug_interactions" section that mention drug_b,
        or a not-found message.
    """
    label = find_label(drug_a, prefer_section="drug_interactions")
    if not label:
        return f"No FDA label found locally for '{drug_a}'. Try fetch_openfda_live first."

    interactions = join_section(label, "drug_interactions")
    if not interactions:
        return f"No drug-interactions section recorded for {drug_a}."

    needle = drug_b.lower().strip()
    matched_lines = [
        line.strip()
        for line in interactions.splitlines()
        if line.strip() and needle in line.lower()
    ]

    if not matched_lines:
        return (
            f"No documented interaction between {drug_a} and {drug_b} found "
            f"in {drug_a}'s FDA label. Absence of evidence is not evidence of safety."
        )

    return f"Documented interactions between {drug_a} and {drug_b}:\n\n" + "\n".join(
        f"- {line}" for line in matched_lines[:8]
    )
