from langchain_core.tools import tool

from agent.tools._data import find_label, join_section


@tool
def dosage_calculator(drug: str, age: int, weight_kg: float) -> str:
    """Return the FDA-recommended dosage for a drug, with patient-specific flags.

    Use whenever the user asks "how much of X should I take" or any question
    about dose, frequency, or pediatric/geriatric dosing.

    Args:
        drug: drug name (brand or generic).
        age: patient age in years.
        weight_kg: patient weight in kilograms.

    Returns:
        The FDA "dosage_and_administration" section verbatim, plus pediatric,
        geriatric, and weight-based flags relevant to the patient.
    """
    label = find_label(drug, prefer_section="dosage_and_administration")
    if not label:
        return f"No FDA label found locally for '{drug}'. Try fetch_openfda_live first."

    dosage_text = join_section(label, "dosage_and_administration")
    if not dosage_text:
        return f"No dosage_and_administration section recorded for {drug}."

    flags: list[str] = []
    if age < 18:
        flags.append(
            f"PEDIATRIC PATIENT (age {age}): pediatric dosing is often weight-based "
            f"(mg/kg) and many FDA labels restrict use below certain ages or weights. "
            f"Verify the section below explicitly authorizes use for this age."
        )
    if age >= 65:
        flags.append(
            f"GERIATRIC PATIENT (age {age}): older adults often need reduced doses "
            f"due to renal and hepatic decline. Look for 'geriatric' or 'elderly' notes."
        )
    if weight_kg and weight_kg < 50:
        flags.append(
            f"LOW BODY WEIGHT ({weight_kg} kg): some labels (e.g. naproxen) explicitly "
            f"restrict standard tablet dosing below 50 kg."
        )

    header = f"FDA dosage section for {label.get('drug_name', drug)}:"
    flag_block = ""
    if flags:
        flag_block = "\n\nPATIENT FLAGS:\n" + "\n".join(f"- {f}" for f in flags)

    return f"{header}\n\n{dosage_text}{flag_block}"
