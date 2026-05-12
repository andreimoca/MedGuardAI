from agent.tools.allergies import check_allergies
from agent.tools.dosage import dosage_calculator
from agent.tools.emergency import emergency_classifier
from agent.tools.interactions import check_drug_interactions
from agent.tools.openfda import fetch_openfda_live
from agent.tools.retrieval import retrieve_drug_info

ALL_TOOLS = [
    retrieve_drug_info,
    check_drug_interactions,
    dosage_calculator,
    check_allergies,
    emergency_classifier,
    fetch_openfda_live,
]

__all__ = [
    "ALL_TOOLS",
    "check_allergies",
    "check_drug_interactions",
    "dosage_calculator",
    "emergency_classifier",
    "fetch_openfda_live",
    "retrieve_drug_info",
]
