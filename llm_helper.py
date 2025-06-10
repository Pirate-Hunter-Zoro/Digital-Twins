from query_llm import query_llm
import json
import re
from main import GLOBAL_CONFIG

def get_narrative(patient: list[dict[str, list[str]]]) -> str:
    """
    Use an LLM to get a narrative of the patients data up to the given visit_idx
    """
    prompt = (
        "Here is a patient's medical visit history. Summarize their journey so far in 2–4 readable sentences. "
        "Mention key conditions, medications, or treatments if relevant.\n\n"
    )

    for i, visit in enumerate(patient[:GLOBAL_CONFIG.num_visits-1]):
        prompt += f"\nVisit {i}:\n{json.dumps(visit, indent=2)}\n"

    return query_llm(prompt)

def get_relevance_score(patient_narrative: str, neighbor_narrative: str, tries_left: int=5) -> float:
    """
    Use an LLM to get a relevance score between two patient narratives.
    """
    prompt = (
        "You are an expert in medical narratives. "
        "Rate the relevance of the following two patient narratives on a scale from 0 to 9, "
        "where 0 means completely irrelevant and 9 means highly relevant.\n\n"
        f"Patient Narrative:\n{patient_narrative}\n\n"
        f"Neighbor Narrative:\n{neighbor_narrative}\n\n"
        "Relevance Score (0-9):"
    )
    
    for _ in range(tries_left):
        response = query_llm(prompt)
        try:
            match = re.search(
                r"\d(?:\.\d+)?",  # Match a number with optional decimal
                response,
                re.IGNORECASE
            )
            if not match:
                raise ValueError("No valid score found in response")
            response = match.group(0)  # Extract the matched number
            if not response.strip():
                raise ValueError("Empty response after stripping")
            # Convert to float and return
            return float(response.strip())
        except ValueError:
            continue
        except Exception:
            return 0.0  # If network error, assume no relevance
        
    return 0.0  # If all attempts fail, assume no relevance