# --- process_patient.py ---
import os
import sys

# --- Path setup ---
current_script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_script_dir, '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from scripts.llm.query_and_response import generate_prompt, force_valid_prediction

setup_done = False

def process_patient(patient_dict):
    """
    Given a patient dict, generate a prompt and return LLM predictions.
    """
    global setup_done
    if not setup_done:
        from scripts.llm.query_and_response import setup_prompt_generation
        setup_prompt_generation()
        setup_done = True
    
    prompt = generate_prompt(patient_dict)
    prediction = force_valid_prediction(prompt)
    return patient_dict['patient_id'], prediction
