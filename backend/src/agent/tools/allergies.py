from langchain_core.tools import tool

from agent.tools._data import find_label, join_section

# Common cross-reactivity groups: an allergy on the LEFT can imply risk for any
# of the active ingredients on the RIGHT. Conservative, FDA-label-grounded.
_CROSS_REACTIVITY: dict[str, list[str]] = {
    "penicillin": ["amoxicillin", "ampicillin", "piperacillin", "cefalexin", "cephalexin"],
    "aspirin": ["naproxen", "ibuprofen", "ketoprofen", "diclofenac", "celecoxib"],
    "nsaid": ["naproxen", "ibuprofen", "ketoprofen", "diclofenac", "celecoxib", "aspirin"],
    "sulfa": ["sulfamethoxazole", "sulfasalazine", "sulfadiazine"],
}


def _expand_allergies(allergies: list[str]) -> set[str]:
    expanded: set[str] = set()
    for allergy in allergies or []:
        a = allergy.lower().strip()
        if not a:
            continue
        expanded.add(a)
        for key, group in _CROSS_REACTIVITY.items():
            if key in a:
                expanded.update(g.lower() for g in group)
    return expanded


@tool
def check_allergies(drug: str, patient_allergies: list[str]) -> str:
    """Cross-reference a drug's active ingredients against the patient's allergy list.

    Use this whenever recommending or discussing any medication for a patient
    whose profile lists allergies. It also expands well-known cross-reactivity
    classes (e.g. penicillin → amoxicillin, aspirin → other NSAIDs).

    Args:
        drug: drug name being considered.
        patient_allergies: the patient's known allergies (from their profile).

    Returns:
        A safety verdict: SAFE, ALLERGY MATCH, or CROSS-REACTIVITY RISK, with the
        matched ingredient and the relevant FDA contraindications excerpt.
    """
    if not patient_allergies:
        return f"Patient profile lists no allergies; no allergy conflict for {drug}."

    label = find_label(drug, prefer_section="contraindications")
    drug_terms: set[str] = {drug.lower().strip()}
    if label:
        drug_terms.add(str(label.get("drug_name", "")).lower())
        drug_terms.update(str(g).lower() for g in label.get("generic_name", []) or [])
        drug_terms.update(str(a).lower() for a in label.get("active_ingredients", []) or [])
    drug_terms = {t for t in drug_terms if t}

    expanded = _expand_allergies(patient_allergies)
    direct_hits = [a for a in patient_allergies if a.lower().strip() in drug_terms]
    cross_hits = [
        term for term in expanded
        if any(term in dt or dt in term for dt in drug_terms)
        and term not in {a.lower().strip() for a in patient_allergies}
    ]

    contraindications = join_section(label, "contraindications") if label else ""
    contra_excerpt = ""
    if contraindications:
        relevant_lines = [
            line.strip() for line in contraindications.splitlines()
            if line.strip() and any(term in line.lower() for term in expanded)
        ]
        if relevant_lines:
            contra_excerpt = "\nRelevant contraindications:\n" + "\n".join(
                f"- {line}" for line in relevant_lines[:5]
            )

    if direct_hits:
        return (
            f"ALLERGY MATCH: patient is allergic to {', '.join(direct_hits)}, "
            f"which matches the active ingredient(s) of {drug}. DO NOT ADMINISTER."
            f"{contra_excerpt}"
        )
    if cross_hits:
        return (
            f"CROSS-REACTIVITY RISK: patient allergies ({', '.join(patient_allergies)}) "
            f"may cross-react with {drug} via shared class membership "
            f"({', '.join(sorted(set(cross_hits)))}). Use with caution and verify with "
            f"a clinician.{contra_excerpt}"
        )
    return (
        f"SAFE on allergy axis: no overlap between patient allergies "
        f"({', '.join(patient_allergies)}) and the active ingredients of {drug}."
    )
