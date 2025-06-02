from query_llm import query_llm
import json

def get_narrative(patient: list[dict[str, list[str]]], visit_idx: int) -> str:
    """
    Use an LLM to get a narrative of the patients data up to the given visit_idx
    """
    prompt = (
        "Here is a patient's medical visit history. Summarize their journey so far in 2–4 readable sentences. "
        "Mention key conditions, medications, or treatments if relevant.\n\n"
    )

    assert visit_idx < len(patient) + 2, "visit_idx must be less than the number of visits + 2"
    for i, visit in enumerate(patient[:visit_idx-1]):
        prompt += f"\nVisit {i}:\n{json.dumps(visit, indent=2)}\n"

    return query_llm(prompt)