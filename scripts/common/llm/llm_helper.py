import sys
import os
import re

# --- Dynamic sys.path adjustment for module imports ---
current_script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_script_dir, '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
# --- End of sys.path adjustment ---

from scripts.common.llm.query_llm import query_llm
from scripts.common.config import get_global_config
# --- FIX APPLIED: Import from our central utility script to break the loop! ---
from scripts.common.utils import turn_to_sentence


def get_narrative(patient: list[dict[str, list[str]]]) -> str:
    """
    Use an LLM to get a narrative of the patient's data up to the given visit_idx.
    """
    prompt = (
        "Here is a patient's medical visit history. Summarize their journey so far in 2–4 readable sentences. "
        "Mention key conditions, medications, or treatments if relevant.\n\n"
    )

    # Use the corrected turn_to_sentence function imported from our new utils file
    for i, visit in enumerate(patient[:get_global_config().num_visits-1]):
        prompt += f"\nVisit {i}:\n{turn_to_sentence(visit)}\n"

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
            response = match.group(0)
            if not response.strip():
                raise ValueError("Empty response after stripping")
            return float(response.strip())
        except ValueError:
            continue
        except Exception:
            return 0.0
        
    return 0.0
